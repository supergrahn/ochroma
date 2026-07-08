//! Simple solar/lunar sky model — the sun and moon positions plus a quick
//! per-pixel sky/ground backdrop and lit-geometry composite.
//!
//! A [`Sky`] is a time of day plus cloud and fog amounts. From it we derive a
//! single directional [`Light`] — the sun while it is above the horizon, else the
//! (much dimmer) moon — whose intensity is attenuated by cloud cover and fog.
//! [`backdrop`] ray-casts every pixel against the ground plane and the sky to
//! paint a solid sky/ground image (with the sun or moon disk drawn into it), and
//! [`composite_lit`] drops the GPU-rendered city geometry on top, tinted and
//! dimmed by the same light so day and night actually read differently.
//!
//! World axes: +X = east, +Y = up, +Z = north.
//!
//! # Relationship to the rest of the engine
//! For a higher-precision solar position use [`vox_core::celestial`]
//! (`compute_sun_position`, `compute_moon_position`). This module intentionally
//! keeps a lighter, self-contained footprint for quick synthetic skies, CPU
//! backdrop rendering, and unit tests.

use glam::{Mat4, Vec3, Vec4};

// ---------------------------------------------------------------------------
// Core sky state
// ---------------------------------------------------------------------------

/// The sky for a moment: the real **sun and moon directions** plus weather.
/// Build it from a geographic [`Observer`] (latitude, longitude, day of year,
/// hour) for a physically-placed sun, or from [`Sky::at_time`] for a quick
/// synthetic day.
#[derive(Debug, Clone, Copy)]
pub struct Sky {
    /// Unit direction toward the sun.
    pub sun: Vec3,
    /// Unit direction toward the moon.
    pub moon: Vec3,
    pub cloud: f32,
    pub fog: f32,
}

impl Default for Sky {
    fn default() -> Self {
        Self::at_time(0.5, 0.2, 0.12)
    }
}

/// A place and moment on Earth — what fixes where the sun and moon actually are.
#[derive(Debug, Clone, Copy)]
pub struct Observer {
    /// Latitude in degrees, +north.
    pub latitude_deg: f32,
    /// Longitude in degrees, +east.
    pub longitude_deg: f32,
    /// Day of year, 1..=366.
    pub day_of_year: u32,
    /// Local clock hour, 0..24 (the time-zone meridian is taken as the nearest
    /// 15° to the longitude, so this reads as ordinary local time).
    pub hour: f32,
}

// ---------------------------------------------------------------------------
// Internal solar mathematics
// ---------------------------------------------------------------------------

/// Solar declination (radians) for day-of-year `n` (Cooper's approximation).
fn declination_rad(n: f32) -> f32 {
    23.45_f32.to_radians() * (std::f32::consts::TAU / 365.0 * (n - 81.0)).sin()
}

/// Equation of time (hours) — the sundial-vs-clock offset across the year.
fn equation_of_time_hours(n: f32) -> f32 {
    let b = std::f32::consts::TAU / 364.0 * (n - 81.0);
    (9.87 * (2.0 * b).sin() - 7.53 * b.cos() - 1.5 * b.sin()) / 60.0
}

/// Convert an altitude/azimuth (azimuth from north, toward east) to a world
/// direction (+X east, +Y up, +Z north).
fn altaz_to_dir(alt: f32, az: f32) -> Vec3 {
    Vec3::new(alt.cos() * az.sin(), alt.sin(), alt.cos() * az.cos())
}

/// Altitude + azimuth (radians) of a body at hour-angle `h_deg` and declination
/// `dec` for an observer at `lat` (radians). Azimuth is from north toward east.
fn altaz(lat: f32, dec: f32, h_deg: f32) -> (f32, f32) {
    let h = h_deg.to_radians();
    let sin_alt = (lat.sin() * dec.sin() + lat.cos() * dec.cos() * h.cos()).clamp(-1.0, 1.0);
    let alt = sin_alt.asin();
    let denom = alt.cos() * lat.cos();
    let az = if denom.abs() < 1e-5 {
        if h < 0.0 {
            std::f32::consts::FRAC_PI_2
        } else {
            3.0 * std::f32::consts::FRAC_PI_2
        }
    } else {
        let cos_az = ((dec.sin() - sin_alt * lat.sin()) / denom).clamp(-1.0, 1.0);
        let a = cos_az.acos(); // 0..π, measured from north
        if h > 0.0 {
            std::f32::consts::TAU - a
        } else {
            a
        } // afternoon swings west
    };
    (alt, az)
}

/// Local solar hour-angle (degrees) for the observer: clock time corrected by the
/// equation of time and the longitude offset from its time-zone meridian.
fn solar_hour_angle_deg(obs: Observer) -> f32 {
    let n = obs.day_of_year as f32;
    let tz_meridian = 15.0 * (obs.longitude_deg / 15.0).round();
    let solar_time =
        obs.hour + equation_of_time_hours(n) + (obs.longitude_deg - tz_meridian) / 15.0;
    (solar_time - 12.0) * 15.0
}

// ---------------------------------------------------------------------------
// Public solar/lunar position functions
// ---------------------------------------------------------------------------

/// World direction toward the sun for this observer (real solar position).
pub fn solar_dir(obs: Observer) -> Vec3 {
    let dec = declination_rad(obs.day_of_year as f32);
    let (alt, az) = altaz(
        obs.latitude_deg.to_radians(),
        dec,
        solar_hour_angle_deg(obs),
    );
    altaz_to_dir(alt, az)
}

/// World direction toward the moon — an **approximation**: the moon lags the sun
/// by its age in the synodic month (~12.2°/day of hour angle, i.e. it rises ~50
/// min later each day) and rides the opposite declination. Good enough to put a
/// believable moon in the night sky; not an ephemeris.
pub fn lunar_dir(obs: Observer) -> Vec3 {
    let n = obs.day_of_year as f32;
    // Synodic age in days (no epoch available from day-of-year alone; this just
    // makes the moon drift through the month plausibly).
    let age = (n % 29.53) / 29.53; // 0..1 through the lunar month
    let lag_deg = age * 360.0 - 180.0; // full moon (opposite the sun) mid-month
    let h = solar_hour_angle_deg(obs) + 180.0 + lag_deg;
    let dec = -declination_rad(n) * 0.9; // roughly the opposite ecliptic swing
    let (alt, az) = altaz(obs.latitude_deg.to_radians(), dec, h);
    altaz_to_dir(alt, az)
}

// ---------------------------------------------------------------------------
// Light
// ---------------------------------------------------------------------------

/// The single directional light in the scene.
#[derive(Debug, Clone, Copy)]
pub struct Light {
    /// Unit direction **toward** the light.
    pub dir: Vec3,
    /// Linear RGB colour of the light.
    pub color: [f32; 3],
    /// 0..1 brightness after cloud + fog attenuation.
    pub intensity: f32,
    /// True if the sun is the active light; false for the moon.
    pub is_day: bool,
}

// ---------------------------------------------------------------------------
// Sky impl
// ---------------------------------------------------------------------------

impl Sky {
    /// A quick synthetic day (no geography): the sun rides a simple arc — east at
    /// `time=0.25`, overhead at `time=0.5`, west at `0.75`, below at night. The
    /// moon is the anti-sun. Use [`Sky::from_observer`] for a real solar position.
    pub fn at_time(time: f32, cloud: f32, fog: f32) -> Self {
        let p = (time - 0.25) * std::f32::consts::TAU;
        let sun = Vec3::new(p.cos(), p.sin(), -0.2).normalize();
        Self {
            sun,
            moon: -sun,
            cloud,
            fog,
        }
    }

    /// The real sky for a place and moment: the sun and moon at their actual
    /// directions for the observer's latitude/longitude/day/hour.
    pub fn from_observer(obs: Observer, cloud: f32, fog: f32) -> Self {
        Self {
            sun: solar_dir(obs),
            moon: lunar_dir(obs),
            cloud,
            fog,
        }
    }

    /// Direction toward the sun.
    pub fn sun_dir(self) -> Vec3 {
        self.sun
    }

    /// How much the sun/moon is dimmed by weather: clouds and fog both cut it.
    pub fn attenuation(self) -> f32 {
        (1.0 - 0.78 * self.cloud.clamp(0.0, 1.0)) * (1.0 - 0.6 * self.fog.clamp(0.0, 1.0))
    }

    /// The active light: the sun if it is above the horizon, otherwise the moon
    /// (far dimmer and cooler).
    pub fn light(self) -> Light {
        let atten = self.attenuation();
        if self.sun.y > 0.0 {
            // Warm near the horizon, neutral-white when high.
            let warmth = self.sun.y.clamp(0.0, 1.0);
            let color = [1.0, 0.72 + 0.26 * warmth, 0.45 + 0.50 * warmth];
            Light {
                dir: self.sun,
                color,
                intensity: (warmth * atten).clamp(0.0, 1.0),
                is_day: true,
            }
        } else {
            let up = self.moon.y.clamp(0.0, 1.0);
            Light {
                dir: self.moon,
                color: [0.55, 0.62, 0.85],
                intensity: (up * 0.18 * atten).clamp(0.0, 1.0),
                is_day: false,
            }
        }
    }

    /// Base sky colour at the zenith for this time of day (linear-ish 0..255).
    pub(crate) fn zenith(self) -> [f32; 3] {
        let l = self.light();
        if l.is_day {
            let d = l.dir.y.clamp(0.0, 1.0);
            // Pale near sunrise/sunset, deeper blue at noon; clouds grey it out.
            let blue = [
                70.0 + 40.0 * (1.0 - d),
                120.0 + 40.0 * (1.0 - d),
                200.0 + 30.0 * d,
            ];
            grey(blue, self.cloud * 0.6)
        } else {
            grey([14.0, 18.0, 38.0], self.cloud * 0.4)
        }
    }

    /// Sky colour at the horizon (haze), where ground fog accumulates.
    pub(crate) fn horizon(self) -> [f32; 3] {
        let l = self.light();
        let base = if l.is_day {
            [200.0, 212.0, 226.0]
        } else {
            [40.0, 46.0, 66.0]
        };
        grey(base, self.fog * 0.5)
    }

    /// Ground albedo colour under the current light (dimmed at night).
    pub(crate) fn ground(self) -> [f32; 3] {
        let l = self.light();
        let lit = 0.32 + 0.68 * l.intensity;
        [
            92.0 * lit * l.color[0],
            138.0 * lit * l.color[1],
            86.0 * lit * l.color[2],
        ]
    }
}

// ---------------------------------------------------------------------------
// Colour helpers (private)
// ---------------------------------------------------------------------------

/// Move a colour toward neutral grey by `t` (0 = unchanged, 1 = grey) — clouds/fog.
fn grey(c: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let m = (c[0] + c[1] + c[2]) / 3.0;
    [
        c[0] + (m - c[0]) * t,
        c[1] + (m - c[1]) * t,
        c[2] + (m - c[2]) * t,
    ]
}

fn to_u8(c: [f32; 3]) -> [u8; 4] {
    [
        c[0].clamp(0.0, 255.0) as u8,
        c[1].clamp(0.0, 255.0) as u8,
        c[2].clamp(0.0, 255.0) as u8,
        255,
    ]
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

// ---------------------------------------------------------------------------
// CPU backdrop / composite (render helpers, CPU-only)
// ---------------------------------------------------------------------------

/// Paint the sky/ground backdrop by casting a ray through every pixel. Rays that
/// point below the horizon hit the ground plane (`y=0`) and read ground colour,
/// hazed toward the horizon by fog with distance; rays above the horizon read the
/// sky gradient, with the sun or moon disk (dimmed by cloud/fog) drawn in.
pub fn backdrop(view: Mat4, proj: Mat4, sky: Sky, width: u32, height: u32) -> Vec<[u8; 4]> {
    let inv = (proj * view).inverse();
    let unproject = |ndc_x: f32, ndc_y: f32, z: f32| -> Vec3 {
        let p = inv * Vec4::new(ndc_x, ndc_y, z, 1.0);
        p.truncate() / p.w
    };

    let light = sky.light();
    let zenith = sky.zenith();
    let horizon = sky.horizon();
    let ground = sky.ground();
    let fog = sky.fog.clamp(0.0, 1.0);

    let mut out = vec![[0u8, 0, 0, 255]; (width * height) as usize];
    for y in 0..height {
        let ndc_y = 1.0 - 2.0 * (y as f32 + 0.5) / height as f32;
        for x in 0..width {
            let ndc_x = 2.0 * (x as f32 + 0.5) / width as f32 - 1.0;
            let near = unproject(ndc_x, ndc_y, 0.0);
            let far = unproject(ndc_x, ndc_y, 1.0);
            let dir = (far - near).normalize();

            let color = if dir.y < -1e-3 {
                // Hits the ground plane y=0.
                let t = near.y / -dir.y;
                let dist = (t * dir.length()).max(0.0);
                // Distance fog blends ground toward the horizon haze.
                let f = 1.0 - (-dist * (0.0009 + 0.004 * fog)).exp();
                lerp3(ground, horizon, f.clamp(0.0, 1.0))
            } else {
                // Sky: horizon -> zenith by elevation, plus the sun/moon glow.
                let up = dir.y.clamp(0.0, 1.0).powf(0.65);
                let mut c = lerp3(horizon, zenith, up);
                let d = dir.dot(light.dir).clamp(-1.0, 1.0);
                if d > 0.0 {
                    // A soft disk + glow around the light, dimmed by weather.
                    let disk = d.powf(2200.0); // tight core
                    let glow = d.powf(12.0) * 0.5; // broad halo
                    let bright = (disk + glow) * sky.attenuation();
                    let lc = if light.is_day {
                        [255.0, 244.0, 214.0]
                    } else {
                        [200.0, 210.0, 235.0]
                    };
                    c = [
                        c[0] + lc[0] * bright,
                        c[1] + lc[1] * bright,
                        c[2] + lc[2] * bright,
                    ];
                }
                c
            };
            out[(y * width + x) as usize] = to_u8(color);
        }
    }
    out
}

/// Composite GPU-rendered `geometry` over the `backdrop`: where geometry is lit
/// (non-background) it is kept but **tinted and dimmed by the light** (so the
/// city is darker and cooler at night, warmer at golden hour); elsewhere the
/// backdrop (sky/ground) shows. Returns the final RGBA and the geometry coverage.
pub fn composite_lit(geometry: &[[u8; 4]], backdrop: &[[u8; 4]], sky: Sky) -> (Vec<[u8; 4]>, f32) {
    let light = sky.light();
    // Ambient floor so night geometry is dim, not black.
    let lit = 0.28 + 0.72 * light.intensity;
    let tint = light.color;
    let mut out = Vec::with_capacity(geometry.len());
    let mut covered = 0usize;
    for (g, bg) in geometry.iter().zip(backdrop.iter()) {
        if (g[0] as u32 + g[1] as u32 + g[2] as u32) > 18 {
            covered += 1;
            out.push([
                (g[0] as f32 * lit * tint[0]).clamp(0.0, 255.0) as u8,
                (g[1] as f32 * lit * tint[1]).clamp(0.0, 255.0) as u8,
                (g[2] as f32 * lit * tint[2]).clamp(0.0, 255.0) as u8,
                255,
            ]);
        } else {
            out.push(*bg);
        }
    }
    let coverage = covered as f32 / geometry.len().max(1) as f32;
    (out, coverage)
}
