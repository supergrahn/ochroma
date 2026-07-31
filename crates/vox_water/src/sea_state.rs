//! Wind → sea state: the physics that makes a gale look like a gale.
//!
//! This is the piece `forge-water` never had. Forge's ocean took `wave_height`
//! and `wind_speed` as two *independent authored knobs*, so the sea could only
//! ever be as rough as an artist typed into a `.ron` — a calm dawn and a storm
//! rendered the identical surface. Here wave height, period, wavelength, phase
//! speed and whitecap coverage are all **derived from one number, the wind**.
//!
//! # Model
//!
//! Fully-developed sea, Pierson & Moskowitz (1964), *J. Geophys. Res.* 69(24):
//!
//! * significant wave height  `H_s = 0.21 · U² / g`
//! * peak angular frequency   `ω_p = 0.877 · g / U`,  so `T_p = 2π / ω_p`
//! * deep-water peak wavelength `λ_p = g · T_p² / (2π)`
//! * deep-water phase speed     `c   = g · T_p / (2π) = λ_p / T_p`
//!
//! Whitecap (breaking-crest) area fraction, Monahan & O'Muircheartaigh (1980),
//! *J. Phys. Oceanogr.* 10(12): `W = 3.84e-6 · U^3.41`.
//!
//! `U` is the 10 m reference wind speed `U10` in m/s throughout.
//!
//! # Determinism
//!
//! [`SeaState::from_wind`] and [`WindResponse::drive`] are **pure functions of a
//! single `f32`**: no state, no RNG, no clock, no allocation. The wind they are
//! fed IS replay-hashed sim truth (`WeatherState::wind_speed`), but their output
//! is consumed render-side only — it perturbs the cosmetic wave amplitude and
//! foam, and is never written back into sim state. So the sea reacts to the
//! deterministic weather while remaining incapable of altering a replay.

use serde::{Deserialize, Serialize};

use crate::GRAVITY;

/// The physical state of a fully-developed sea under a given wind.
///
/// Every field is a real measured-law output in SI units — **not** an art knob.
/// Art direction happens in [`WindResponse`], which consumes this.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaState {
    /// The 10 m wind speed this state was derived from (m/s).
    pub wind_speed_ms: f32,
    /// Significant wave height `H_s` (m) — the mean crest-to-trough height of the
    /// highest third of waves, which is what an observer calls "the wave height".
    pub significant_wave_height_m: f32,
    /// Peak wave period `T_p` (s).
    pub peak_period_s: f32,
    /// Deep-water peak wavelength `λ_p` (m).
    pub peak_wavelength_m: f32,
    /// Deep-water phase speed `c` (m/s) — how fast crests travel.
    pub phase_speed_ms: f32,
    /// Whitecap area fraction, 0..1.
    pub whitecap_coverage: f32,
}

impl SeaState {
    /// Derive the fully-developed sea state for a 10 m wind speed (m/s).
    ///
    /// Negative or non-finite winds are treated as dead calm. At exactly zero wind
    /// every derived quantity is zero (there is no sea), which keeps the ratios in
    /// [`WindResponse`] well defined via its own guards.
    #[must_use]
    pub fn from_wind(wind_speed_ms: f32) -> Self {
        let u = if wind_speed_ms.is_finite() {
            wind_speed_ms.max(0.0)
        } else {
            0.0
        };
        if u <= 0.0 {
            return Self {
                wind_speed_ms: 0.0,
                significant_wave_height_m: 0.0,
                peak_period_s: 0.0,
                peak_wavelength_m: 0.0,
                phase_speed_ms: 0.0,
                whitecap_coverage: 0.0,
            };
        }

        // Pierson-Moskowitz fully-developed sea.
        let hs = 0.21 * u * u / GRAVITY;
        let omega_p = 0.877 * GRAVITY / u;
        let tp = std::f32::consts::TAU / omega_p;
        let lambda_p = GRAVITY * tp * tp / std::f32::consts::TAU;
        let c = GRAVITY * tp / std::f32::consts::TAU;

        // Monahan & O'Muircheartaigh whitecap coverage, clamped to a physical
        // fraction (the power law is only fitted up to storm winds).
        let whitecap = (3.84e-6 * u.powf(3.41)).clamp(0.0, 1.0);

        Self {
            wind_speed_ms: u,
            significant_wave_height_m: hs,
            peak_period_s: tp,
            peak_wavelength_m: lambda_p,
            phase_speed_ms: c,
            whitecap_coverage: whitecap,
        }
    }

    /// Douglas sea-scale degree (0 = calm glassy … 9 = phenomenal), from the
    /// significant wave height. Useful for UI/telemetry and for asserting that a
    /// witnessed frame really is the sea state the weather claimed.
    #[must_use]
    pub fn douglas_degree(&self) -> u8 {
        let h = self.significant_wave_height_m;
        match h {
            _ if h < 0.01 => 0,
            _ if h < 0.10 => 1,
            _ if h < 0.50 => 2,
            _ if h < 1.25 => 3,
            _ if h < 2.50 => 4,
            _ if h < 4.00 => 5,
            _ if h < 6.00 => 6,
            _ if h < 9.00 => 7,
            _ if h < 14.00 => 8,
            _ => 9,
        }
    }
}

/// The render-side drive derived from a sea state: multipliers to apply to the
/// **authored** wave amplitude, crest speed and foam strength.
///
/// All three are `1.0` exactly when the wind equals
/// [`WindResponse::reference_wind_ms`], so an unconfigured / becalmed-to-reference
/// scene is **byte-identical** to the pre-feature look. That identity point is the
/// safety property: enabling this feature changes nothing until the wind moves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceDrive {
    /// Multiplier on the authored Gerstner wave amplitude (mesh displacement).
    pub amplitude_scale: f32,
    /// Multiplier on the authored crest advance speed.
    pub speed_scale: f32,
    /// Multiplier on the authored foam/whitecap strength.
    pub foam_scale: f32,
}

impl SurfaceDrive {
    /// The neutral drive — every multiplier `1.0` (authored look, untouched).
    pub const IDENTITY: Self = Self {
        amplitude_scale: 1.0,
        speed_scale: 1.0,
        foam_scale: 1.0,
    };
}

/// Config-first art response mapping a [`SeaState`] onto [`SurfaceDrive`].
///
/// The *physics* is fixed (it is a law of nature); what a game may legitimately
/// tune is how much of that physics it spends on screen. The authored ocean in
/// `urban_horizon.ron` is deliberately exaggerated relative to a real 10 m/s sea,
/// and this type preserves that exaggeration while making the **response** to
/// wind physical: it works entirely in *ratios against the reference wind*, so the
/// artist's absolute values survive untouched.
///
/// Lives in `render.ron` (`water.wind_response`) — tune it and re-run, no rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WindResponse {
    /// Master switch. `false` → [`SurfaceDrive::IDENTITY`] always, i.e. exactly the
    /// pre-feature behaviour.
    pub enabled: bool,
    /// The wind (m/s) the authored look is defined AT — the identity point, where
    /// every multiplier is exactly `1.0`. This must be the wind the game normally
    /// sits at (its weather baseline), NOT the wind an offline mesh generator was
    /// parameterised with: pick the latter and the very first frame of a normal
    /// game already deviates from the authored look.
    pub reference_wind_ms: f32,
    /// Exponent applied ON TOP of the already-physical significant-wave-height
    /// ratio `H_s(U)/H_s(U_ref)`.
    ///
    /// **`1.0` is the physically exact response** — the Pierson-Moskowitz quadratic
    /// is already inside that ratio (`H_s ∝ U²`), so raising it to 2.0 would make
    /// the sea grow as `U⁴` and double-count the physics. Values below 1.0 compress
    /// a storm into a smaller on-screen swing; above 1.0 deliberately exaggerates.
    pub amplitude_exponent: f32,
    /// Clamp on `amplitude_scale`, applied after the exponent.
    pub amplitude_min: f32,
    pub amplitude_max: f32,
    /// Exponent applied on top of the phase-speed ratio `c(U)/c(U_ref)`. Deep-water
    /// phase speed is *linear* in wind (`c = g·T_p/2π`, `T_p ∝ U`) and that is
    /// already in the ratio, so `1.0` is again the physically exact value.
    pub speed_exponent: f32,
    pub speed_min: f32,
    pub speed_max: f32,
    /// How strongly whitecap coverage drives the foam multiplier. `1.0` tracks the
    /// Monahan power law directly (very aggressive — coverage rises as `U^3.41`);
    /// the default softens it while keeping the ordering.
    pub foam_gain: f32,
    pub foam_min: f32,
    pub foam_max: f32,
    /// Exaggeration gain on the PHYSICAL significant wave height, applied by
    /// [`mesh_amplitude_scale`](Self::mesh_amplitude_scale).
    ///
    /// `1.0` means the displaced water MESH realises exactly the `H_s` this
    /// module's Pierson-Moskowitz solver reports for the live wind — so the
    /// `[sea-state]` log line and the rendered silhouette are the SAME number, and
    /// "H_s = 4.20 m" becomes a claim a witness can measure instead of a hope.
    /// Above 1.0 deliberately over-drives the swell; below 1.0 calms it.
    pub height_gain: f32,
}

impl Default for WindResponse {
    fn default() -> Self {
        Self {
            enabled: true,
            // The game's WEATHER BASELINE (`WeatherState::new` → 14 m/s, a
            // Beaufort 6 strong breeze), NOT the offline ocean-mesh authoring wind.
            // A normal game therefore renders exactly the authored sea.
            reference_wind_ms: 14.0,
            // 1.0 = physically exact (the U² is already in the H_s ratio).
            amplitude_exponent: 1.0,
            amplitude_min: 0.20,
            amplitude_max: 2.50,
            speed_exponent: 1.0,
            speed_min: 0.40,
            speed_max: 2.00,
            foam_gain: 0.35,
            foam_min: 0.30,
            foam_max: 2.00,
            // 1.0 = the mesh IS the physics. Anything else is an admitted lie.
            height_gain: 1.0,
        }
    }
}

impl WindResponse {
    /// Map a live wind speed (m/s) onto the render drive.
    ///
    /// Pure: no state, no clock, no allocation. Returns [`SurfaceDrive::IDENTITY`]
    /// when disabled, when the reference wind is not positive, or when the wind is
    /// not finite — the feature can never make the sea *undefined*, only bigger or
    /// smaller than the authored look.
    #[must_use]
    pub fn drive(&self, wind_speed_ms: f32) -> SurfaceDrive {
        if !self.enabled || !(self.reference_wind_ms > 0.0) || !wind_speed_ms.is_finite() {
            return SurfaceDrive::IDENTITY;
        }
        let reference = SeaState::from_wind(self.reference_wind_ms);
        let now = SeaState::from_wind(wind_speed_ms.max(0.0));
        self.drive_between(&now, &reference)
    }

    /// The ratio core of [`drive`](Self::drive), against an explicit reference sea
    /// state. Split out so a caller can A/B two states directly and so the identity
    /// property is testable without going through the wind twice.
    #[must_use]
    pub fn drive_between(&self, now: &SeaState, reference: &SeaState) -> SurfaceDrive {
        // Guard every ratio: a zero-wind reference has no sea to be relative to.
        let ratio = |a: f32, b: f32| -> f32 {
            if b > 1e-9 && a.is_finite() {
                a / b
            } else {
                1.0
            }
        };

        let height_ratio = ratio(
            now.significant_wave_height_m,
            reference.significant_wave_height_m,
        );
        let speed_ratio = ratio(now.phase_speed_ms, reference.phase_speed_ms);
        let whitecap_ratio = ratio(now.whitecap_coverage, reference.whitecap_coverage);

        // `x^e` with x >= 0; powf(0) is 0 for e > 0, which is the wanted dead-calm
        // behaviour (a flat sea, not a NaN).
        let amplitude_scale = height_ratio
            .max(0.0)
            .powf(self.amplitude_exponent)
            .clamp(self.amplitude_min, self.amplitude_max);
        let speed_scale = speed_ratio
            .max(0.0)
            .powf(self.speed_exponent)
            .clamp(self.speed_min, self.speed_max);
        // Soften the very steep whitecap law by interpolating from 1.0 by `gain`,
        // so `foam_gain: 0` pins foam to the authored value and `1.0` tracks
        // Monahan exactly.
        let foam_scale =
            (1.0 + self.foam_gain * (whitecap_ratio - 1.0)).clamp(self.foam_min, self.foam_max);

        SurfaceDrive {
            amplitude_scale,
            speed_scale,
            foam_scale,
        }
    }

    /// The multiplier a caller must apply to a BAKED Gerstner spectrum's
    /// amplitudes so the displaced MESH realises the live sea state's significant
    /// wave height.
    ///
    /// [`drive`](Self::drive)`.amplitude_scale` alone only makes the sea RESPOND to
    /// wind — it is a ratio against the reference wind, so it preserves whatever
    /// absolute height the offline cook happened to bake. That absolute height was
    /// never in the same units as the physics: on `forge_coastal_cove` the cooked
    /// spectrum realises `H_s ≈ 14.5 m` while the solver reports `H_s = 4.20 m`.
    /// Dividing the baked height back out (`H_s(U_ref) / H_s_baked`) CALIBRATES the
    /// authored spectrum onto the physical one, after which
    ///
    /// ```text
    ///     mesh H_s(U) = height_gain · H_s_solver(U)
    /// ```
    ///
    /// exactly (for `amplitude_exponent == 1` and inside the amplitude clamps).
    ///
    /// `baked_significant_height_m` is the spectrum's own
    /// `H_s = 4·sqrt(Σ aᵢ²/2)`; a non-positive or non-finite value skips the
    /// calibration (there is nothing to calibrate against) and returns the plain
    /// wind drive, so this can never make the sea undefined.
    #[must_use]
    pub fn mesh_amplitude_scale(&self, wind_speed_ms: f32, baked_significant_height_m: f32) -> f32 {
        let drive = self.drive(wind_speed_ms);
        if !self.enabled {
            return drive.amplitude_scale;
        }
        let gain = if self.height_gain.is_finite() && self.height_gain >= 0.0 {
            self.height_gain
        } else {
            1.0
        };
        let reference_hs = SeaState::from_wind(self.reference_wind_ms).significant_wave_height_m;
        if !baked_significant_height_m.is_finite()
            || baked_significant_height_m <= 1e-4
            || reference_hs <= 1e-6
        {
            return drive.amplitude_scale * gain;
        }
        drive.amplitude_scale * gain * (reference_hs / baked_significant_height_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pierson-Moskowitz at U10 = 10 m/s, computed by hand from the published
    /// coefficients. These are the numbers the model MUST reproduce.
    /// The CALIBRATION seam: whatever absolute height the offline cook baked, the
    /// displaced mesh must end up realising the solver's `H_s`. This is the
    /// property the `[water-swell]` witness line measures off the real vertices.
    #[test]
    fn mesh_amplitude_scale_makes_the_baked_spectrum_realise_the_solver_height() {
        let wr = WindResponse::default(); // reference 14 m/s, exponent 1, gain 1
        // The cooked forge_coastal_cove spectrum realises this; the solver says 4.20 m.
        let baked_hs = 14.55_f32;
        for wind in [6.0_f32, 10.0, 14.0, 18.0] {
            let scale = wr.mesh_amplitude_scale(wind, baked_hs);
            let realised = baked_hs * scale;
            let target = SeaState::from_wind(wind).significant_wave_height_m;
            // Inside the amplitude clamps the mesh height IS the physical height.
            let ratio =
                target / SeaState::from_wind(wr.reference_wind_ms).significant_wave_height_m;
            if ratio > wr.amplitude_min && ratio < wr.amplitude_max {
                assert!(
                    (realised - target).abs() < 1.0e-3 * target.max(1.0),
                    "wind {wind}: mesh H_s {realised} m != solver H_s {target} m"
                );
            }
        }
    }

    /// `height_gain` is the ONLY exaggeration dial, and it is exact.
    #[test]
    fn height_gain_scales_the_realised_height_exactly() {
        let baked_hs = 14.55_f32;
        let one = WindResponse::default();
        let half = WindResponse {
            height_gain: 0.5,
            ..WindResponse::default()
        };
        let a = one.mesh_amplitude_scale(14.0, baked_hs);
        let b = half.mesh_amplitude_scale(14.0, baked_hs);
        assert!(
            (b - 0.5 * a).abs() < 1.0e-6,
            "gain 0.5 gave {b}, expected {}",
            0.5 * a
        );
        // Gain 0 is a genuinely flat sea (the witness control), not a near-flat one.
        let zero = WindResponse {
            height_gain: 0.0,
            ..WindResponse::default()
        };
        assert_eq!(zero.mesh_amplitude_scale(14.0, baked_hs), 0.0);
    }

    /// A missing / nonsensical baked height must never make the sea undefined —
    /// it falls back to the plain wind drive.
    #[test]
    fn calibration_degrades_to_the_plain_drive_without_a_baked_height() {
        let wr = WindResponse::default();
        for bad in [0.0_f32, -1.0, f32::NAN] {
            let scale = wr.mesh_amplitude_scale(14.0, bad);
            assert!(
                (scale - wr.drive(14.0).amplitude_scale).abs() < 1.0e-6,
                "baked H_s {bad} gave {scale}"
            );
        }
    }

    #[test]
    fn pierson_moskowitz_matches_hand_computed_values_at_10ms() {
        let s = SeaState::from_wind(10.0);
        // H_s = 0.21 * 100 / 9.81 = 2.14067...
        assert!(
            (s.significant_wave_height_m - 2.140672).abs() < 1e-4,
            "H_s = {}",
            s.significant_wave_height_m
        );
        // omega_p = 0.877 * 9.81 / 10 = 0.860337 -> T_p = TAU/omega_p = 7.30313
        assert!(
            (s.peak_period_s - 7.303130).abs() < 1e-3,
            "T_p = {}",
            s.peak_period_s
        );
        // lambda_p = g*T^2/TAU = 9.81 * 53.3357 / 6.283185 = 83.2735
        assert!(
            (s.peak_wavelength_m - 83.2735).abs() < 2e-2,
            "lambda_p = {}",
            s.peak_wavelength_m
        );
        // c = lambda / T
        assert!(
            (s.phase_speed_ms - s.peak_wavelength_m / s.peak_period_s).abs() < 1e-3,
            "c = {} vs lambda/T = {}",
            s.phase_speed_ms,
            s.peak_wavelength_m / s.peak_period_s
        );
        // W = 3.84e-6 * 10^3.41 = 3.84e-6 * 2570.40 = 0.00987
        assert!(
            (s.whitecap_coverage - 0.0098703).abs() < 1e-5,
            "W = {}",
            s.whitecap_coverage
        );
    }

    #[test]
    fn wave_height_is_quadratic_in_wind() {
        // H_s ∝ U²: doubling the wind quadruples the significant wave height.
        let a = SeaState::from_wind(5.0);
        let b = SeaState::from_wind(10.0);
        let ratio = b.significant_wave_height_m / a.significant_wave_height_m;
        assert!((ratio - 4.0).abs() < 1e-4, "ratio = {ratio}");
    }

    #[test]
    fn phase_speed_is_linear_in_wind() {
        // c = g·T_p/2π and T_p ∝ U, so c ∝ U exactly.
        let a = SeaState::from_wind(6.0);
        let b = SeaState::from_wind(18.0);
        let ratio = b.phase_speed_ms / a.phase_speed_ms;
        assert!((ratio - 3.0).abs() < 1e-4, "ratio = {ratio}");
    }

    #[test]
    fn dead_calm_has_no_sea() {
        let s = SeaState::from_wind(0.0);
        assert_eq!(s.significant_wave_height_m, 0.0);
        assert_eq!(s.peak_period_s, 0.0);
        assert_eq!(s.whitecap_coverage, 0.0);
        assert_eq!(s.douglas_degree(), 0);
        // Negative and NaN winds are treated as calm, never as NaN sea state.
        assert_eq!(SeaState::from_wind(-4.0), s);
        assert_eq!(SeaState::from_wind(f32::NAN), s);
    }

    #[test]
    fn douglas_degrees_track_the_beaufort_ladder() {
        // A gentle breeze is a smooth/slight sea; a whole gale is a high sea.
        assert_eq!(SeaState::from_wind(4.0).douglas_degree(), 2); // H_s = 0.34 m
        assert_eq!(SeaState::from_wind(10.0).douglas_degree(), 4); // H_s = 2.14 m
        assert_eq!(SeaState::from_wind(20.0).douglas_degree(), 7); // H_s = 8.56 m
        assert_eq!(SeaState::from_wind(25.0).douglas_degree(), 8); // H_s = 13.4 m
    }

    #[test]
    fn drive_is_exactly_identity_at_the_reference_wind() {
        // THE SAFETY PROPERTY: at the authored reference wind the feature must be a
        // no-op, so turning it on cannot change an existing witnessed frame.
        let r = WindResponse::default();
        let d = r.drive(r.reference_wind_ms);
        assert_eq!(d.amplitude_scale, 1.0);
        assert_eq!(d.speed_scale, 1.0);
        assert_eq!(d.foam_scale, 1.0);
    }

    #[test]
    fn disabled_response_is_identity_at_every_wind() {
        let r = WindResponse {
            enabled: false,
            ..WindResponse::default()
        };
        for wind in [0.0, 3.0, 10.0, 25.0, 60.0] {
            assert_eq!(r.drive(wind), SurfaceDrive::IDENTITY, "wind = {wind}");
        }
    }

    #[test]
    fn rising_wind_raises_amplitude_speed_and_foam_monotonically() {
        let r = WindResponse::default();
        let mut previous = r.drive(1.0);
        for wind in [2.0f32, 4.0, 8.0, 12.0, 16.0, 20.0, 25.0] {
            let d = r.drive(wind);
            assert!(
                d.amplitude_scale >= previous.amplitude_scale,
                "amplitude fell at {wind} m/s: {} < {}",
                d.amplitude_scale,
                previous.amplitude_scale
            );
            assert!(
                d.speed_scale >= previous.speed_scale,
                "speed fell at {wind} m/s"
            );
            assert!(
                d.foam_scale >= previous.foam_scale,
                "foam fell at {wind} m/s"
            );
            previous = d;
        }
    }

    #[test]
    fn a_gale_is_visibly_rougher_than_the_reference_and_a_calm_is_visibly_flatter() {
        let r = WindResponse::default();
        // 20 m/s (Beaufort 8, fresh gale) against the 14 m/s baseline. The H_s ratio
        // is (20/14)^2 = 2.041 and the exact exponent 1.0 passes it straight
        // through — a sea twice the height of a normal day.
        let gale = r.drive(20.0);
        assert!(
            (gale.amplitude_scale - 2.0408).abs() < 1e-3,
            "gale amplitude = {}",
            gale.amplitude_scale
        );
        // Crest speed is linear in wind: 20/14 = 1.4286.
        assert!(
            (gale.speed_scale - 1.42857).abs() < 1e-4,
            "gale speed = {}",
            gale.speed_scale
        );
        assert!(gale.foam_scale > 1.5, "gale whitecaps: {}", gale.foam_scale);

        // 4 m/s (Beaufort 3, gentle breeze): H_s ratio (4/14)^2 = 0.0816, which
        // floors at 0.20. A near-glassy morning.
        let calm = r.drive(4.0);
        assert_eq!(calm.amplitude_scale, 0.20);
        assert!(calm.speed_scale < 0.5, "calm crests crawl");
        assert!(calm.foam_scale < 1.0, "calm has less foam than reference");
    }

    #[test]
    fn the_default_exponents_do_not_double_count_the_physics() {
        // GUARD against the bug this shipped with for ten minutes: `height_ratio`
        // is ALREADY quadratic in wind, so the exponent on top must be 1.0. With
        // the exact exponents, doubling the wind must quadruple the amplitude
        // (H_s ∝ U²) and exactly double the crest speed (c ∝ U) — no more.
        let r = WindResponse {
            reference_wind_ms: 10.0,
            amplitude_min: 0.0,
            amplitude_max: 1000.0,
            speed_min: 0.0,
            speed_max: 1000.0,
            ..WindResponse::default()
        };
        let doubled = r.drive(20.0);
        assert!(
            (doubled.amplitude_scale - 4.0).abs() < 1e-3,
            "amplitude should be U^2, got {}",
            doubled.amplitude_scale
        );
        assert!(
            (doubled.speed_scale - 2.0).abs() < 1e-4,
            "speed should be U^1, got {}",
            doubled.speed_scale
        );
    }

    #[test]
    fn drive_is_pure_and_reproducible() {
        // Determinism: the same wind must give bit-identical output every call and
        // in any order (no interior state).
        let r = WindResponse::default();
        let winds = [17.3f32, 0.0, 9.9, 25.0, 3.1];
        let forward: Vec<SurfaceDrive> = winds.iter().map(|w| r.drive(*w)).collect();
        let backward: Vec<SurfaceDrive> =
            winds.iter().rev().map(|w| r.drive(*w)).collect::<Vec<_>>();
        for (index, d) in forward.iter().enumerate() {
            let mirrored = backward[winds.len() - 1 - index];
            assert_eq!(*d, mirrored, "wind {} not reproducible", winds[index]);
        }
    }

    #[test]
    fn clamps_bound_even_an_absurd_wind() {
        let r = WindResponse::default();
        let hurricane = r.drive(90.0);
        assert_eq!(hurricane.amplitude_scale, r.amplitude_max);
        assert_eq!(hurricane.speed_scale, r.speed_max);
        assert_eq!(hurricane.foam_scale, r.foam_max);
        assert!(hurricane.amplitude_scale.is_finite());
    }
}
