//! The cinematic spine: [`CameraSequence`], its pure [`CameraSequence::eval`],
//! and the renderer-ready [`CinePose`] it produces.
//!
//! A `CameraSequence` is the single source of truth for an authored cinematic.
//! It owns:
//!
//! * a position spline (centripetal Catmull-Rom over the 3 `eye` lanes' keys),
//! * a target spline (3 `target` lanes),
//! * scalar [`Channel`]s for `roll`, `fov_y`, `aperture_fstop`,
//!   `focal_length_mm`, `focus_distance`, `shutter_angle_deg`, and
//!   `exposure_ev`,
//! * a `BokehShape` (iris blade count).
//!
//! `eval(frame)` is **pure**: no I/O, no GPU, `Send + Sync`, never panics, and
//! clamps `frame` to `[0, duration]`. It evaluates the splines at the frame
//! (which interpolates control points exactly at keyframes) and samples the
//! scalar channels, then converts artist-facing photographic controls to
//! renderer-ready scalars:
//!
//! * f-stop `N` + focal length `f` → `lens_radius = f / (2·N)`,
//! * shutter angle (deg) → frame-centered `shutter_open/close`,
//! * `BokehShape` → `bokeh_blades`.
//!
//! The arc-length LUT (constant-speed playback) is rebuilt on edit and stored,
//! but `eval` is frame-direct so a keyframe evaluates to its exact control
//! point.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use super::channel::{Channel, FrameRate, Interp, Key};
use super::spline::{ArcLengthLut, Spline3};

/// Iris blade count source — maps to `bokeh_blades` in the kernel.
///
/// `Circle` is a perfectly round iris (0 blades); `Hexagon`/`Octagon` are the
/// polygonal stops that produce shaped (e.g. hexagonal) bokeh highlights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BokehShape {
    Circle,
    Hexagon,
    Octagon,
}

impl BokehShape {
    /// Blade count for the kernel (`Circle→0, Hexagon→6, Octagon→8`).
    pub fn blades(self) -> i32 {
        match self {
            BokehShape::Circle => 0,
            BokehShape::Hexagon => 6,
            BokehShape::Octagon => 8,
        }
    }
}

impl Default for BokehShape {
    fn default() -> Self {
        BokehShape::Circle
    }
}

/// Which subject the focus distance tracks, when not a manual channel.
///
/// `OnEntity` carries an opaque entity id (game layer resolves it to a world
/// position per frame); the engine `eval` treats it as `Manual` because it has
/// no scene access — focus-on-entity is resolved by the game-side player.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FocusMode {
    Manual,
    OnEntity(u64),
}

impl Default for FocusMode {
    fn default() -> Self {
        FocusMode::Manual
    }
}

/// Engine config for the cinematic math — no magic numbers live in the eval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CinematicConfig {
    /// Arc-length LUT subdivisions per control-point segment.
    pub arc_lut_samples_per_segment: u32,
    /// Centripetal Catmull-Rom exponent (0.5 = centripetal).
    pub catmull_rom_alpha: f32,
    /// `cosΩ` above which inner SLERP falls back to nlerp.
    pub slerp_nlerp_threshold: f32,
    /// Banking gain (roll from path curvature). 0 = banking off.
    pub bank_gain: f32,
    /// Banking clamp (max absolute roll added by banking).
    pub bank_max: f32,
    /// Sensor height (mm) — for fov↔focal and f-stop→radius scaling.
    pub sensor_height_mm: f32,
    /// Default samples-per-pixel for offline export.
    pub default_spp_export: u32,
    /// Default frame rate.
    pub default_fps: FrameRate,
}

impl Default for CinematicConfig {
    fn default() -> Self {
        Self {
            arc_lut_samples_per_segment: 64,
            catmull_rom_alpha: 0.5,
            slerp_nlerp_threshold: 0.9995,
            bank_gain: 0.0,
            bank_max: 0.0,
            sensor_height_mm: 24.0,
            default_spp_export: 256,
            default_fps: FrameRate::new(60, 1),
        }
    }
}

/// A renderer-ready camera pose at a single (fractional) frame.
///
/// All fields are already converted from artist-facing controls to the scalars
/// the path tracer consumes. Fields are private; use the accessors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CinePose {
    eye: Vec3,
    target: Vec3,
    roll: f32,
    fov_y: f32,
    lens_radius: f32,
    focus_distance: f32,
    shutter_open: f32,
    shutter_close: f32,
    bokeh_blades: i32,
    exposure_ev: f32,
}

impl CinePose {
    /// Construct a pose directly (used by adapter tests and `eval`).
    #[allow(clippy::too_many_arguments)]
    pub fn test_new(
        eye: Vec3,
        target: Vec3,
        roll: f32,
        fov_y: f32,
        fstop: f32,
        focal_mm: f32,
        focus_distance: f32,
        shutter_open: f32,
        shutter_close: f32,
        bokeh_blades: i32,
        exposure_ev: f32,
    ) -> Self {
        Self {
            eye,
            target,
            roll,
            fov_y,
            lens_radius: lens_radius_from(focal_mm, fstop),
            focus_distance,
            shutter_open,
            shutter_close,
            bokeh_blades,
            exposure_ev,
        }
    }

    pub fn eye(&self) -> Vec3 {
        self.eye
    }
    pub fn target(&self) -> Vec3 {
        self.target
    }
    pub fn roll(&self) -> f32 {
        self.roll
    }
    pub fn fov_y(&self) -> f32 {
        self.fov_y
    }
    pub fn lens_radius(&self) -> f32 {
        self.lens_radius
    }
    pub fn focus_distance(&self) -> f32 {
        self.focus_distance
    }
    pub fn shutter_open(&self) -> f32 {
        self.shutter_open
    }
    pub fn shutter_close(&self) -> f32 {
        self.shutter_close
    }
    pub fn bokeh_blades(&self) -> i32 {
        self.bokeh_blades
    }
    pub fn exposure_ev(&self) -> f32 {
        self.exposure_ev
    }
}

/// f-stop `N` + focal length `f` (mm) → `lens_radius = f / (2·N)`.
///
/// Lower `N` ⇒ larger radius ⇒ shallower depth of field. A zero or negative
/// f-stop disables the lens (radius 0) rather than dividing by zero.
fn lens_radius_from(focal_mm: f32, fstop: f32) -> f32 {
    if fstop > 0.0 {
        focal_mm / (2.0 * fstop)
    } else {
        0.0
    }
}

/// One scalar channel serialized as a list of `[frame, value]` pairs.
type ScalarPairs = Vec<(i32, f32)>;
/// One vector channel serialized as a list of `[frame, [x,y,z]]` pairs.
type Vec3Pairs = Vec<(i32, [f32; 3])>;

/// On-disk / wire representation of a [`CameraSequence`].
///
/// Friendly shape that matches the authored `.cine.json`: position/target as
/// `[frame, [x,y,z]]` pairs and scalar tracks as `[frame, value]` pairs. The
/// runtime [`CameraSequence`] is built from this and vice versa, so JSON
/// round-trips losslessly.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SequenceDto {
    fps: FrameRate,
    duration: i32,
    #[serde(default)]
    eye_keys: Vec3Pairs,
    #[serde(default)]
    target_keys: Vec3Pairs,
    #[serde(default)]
    roll: ScalarPairs,
    #[serde(default)]
    fov_y_deg: ScalarPairs,
    #[serde(default)]
    aperture_fstop: ScalarPairs,
    #[serde(default)]
    focal_length_mm: ScalarPairs,
    #[serde(default)]
    focus_distance: ScalarPairs,
    #[serde(default)]
    focus_mode: FocusMode,
    #[serde(default)]
    shutter_angle_deg: ScalarPairs,
    #[serde(default)]
    bokeh_shape: BokehShape,
    #[serde(default)]
    exposure_ev: ScalarPairs,
    #[serde(default)]
    config: CinematicConfig,
}

/// Build an `Auto`-interp scalar channel from `[frame, value]` pairs.
fn channel_from_pairs(pairs: &[(i32, f32)]) -> Channel<f32> {
    Channel::from_auto_pairs(pairs)
}

/// Serialize a scalar channel back to `[frame, value]` pairs.
fn pairs_from_channel(ch: &Channel<f32>) -> ScalarPairs {
    ch.keys().iter().map(|k| (k.frame, k.value)).collect()
}

/// The authored cinematic timeline. Single source of truth; pure `eval`.
#[derive(Debug, Clone)]
pub struct CameraSequence {
    fps: FrameRate,
    duration: i32,
    /// Eye position keys: `(frame, position)`, sorted ascending by frame.
    eye_keys: Vec<(i32, Vec3)>,
    /// Target position keys: `(frame, position)`, sorted ascending.
    target_keys: Vec<(i32, Vec3)>,
    roll: Channel<f32>,
    fov_y_deg: Channel<f32>,
    aperture_fstop: Channel<f32>,
    focal_length_mm: Channel<f32>,
    focus_distance: Channel<f32>,
    focus_mode: FocusMode,
    shutter_angle_deg: Channel<f32>,
    bokeh_shape: BokehShape,
    exposure_ev: Channel<f32>,
    config: CinematicConfig,
    /// Position spline (rebuilt on `rebuild_arc_lut`).
    eye_spline: Spline3,
    target_spline: Spline3,
    /// Arc-length LUT over the eye spline (constant-speed playback helper).
    arc_lut: ArcLengthLut,
}

/// A game-agnostic orbit framing, the editor's "viewport camera" decomposition.
///
/// The engine cannot reference the game's `CameraFraming` (engine crates are
/// game-agnostic — see `CLAUDE.md`), so the Sequencer's "key from viewport"
/// takes this neutral shape. It is the same `{ focus, radius, zoom, yaw, pitch }`
/// an orbit camera carries; the eye is reconstructed exactly via [`eye_from_framing`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CineFraming {
    /// Orbit pivot (look-at point).
    pub focus: Vec3,
    /// Orbit radius before zoom.
    pub radius: f32,
    /// Zoom multiplier (`distance = radius * zoom`).
    pub zoom: f32,
    /// Azimuth, radians.
    pub yaw: f32,
    /// Elevation, radians.
    pub pitch: f32,
}

/// Reconstruct the eye position from an orbit framing.
///
/// Exact inverse of the design's orbit→pose mapping:
/// `eye = focus + distance·(cosθ·sinφ, sinθ, cosθ·cosφ)` with
/// `distance = radius·zoom`, `θ = pitch`, `φ = yaw`.
pub fn eye_from_framing(f: &CineFraming) -> Vec3 {
    let distance = f.radius * f.zoom;
    let dir = Vec3::new(
        f.pitch.cos() * f.yaw.sin(),
        f.pitch.sin(),
        f.pitch.cos() * f.yaw.cos(),
    );
    f.focus + distance * dir
}

/// A read-only handle to one keyframe on a position lane, exposing its frame.
///
/// Returned by [`CameraSequence::eye_x_keys`] (and the y/z/target lanes) so the
/// Sequencer can pixel-map keys without exposing the internal storage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyRef {
    frame: i32,
    value: f32,
}

impl KeyRef {
    /// The integer frame this key sits on.
    pub fn frame(self) -> i32 {
        self.frame
    }
    /// The lane value (a single position component).
    pub fn value(self) -> f32 {
        self.value
    }
}

impl CameraSequence {
    /// Eye-X lane keys (frame + x component), for the Sequencer pixel map.
    pub fn eye_x_keys(&self) -> Vec<KeyRef> {
        self.eye_keys
            .iter()
            .map(|&(frame, p)| KeyRef { frame, value: p.x })
            .collect()
    }

    /// Eye-Y lane keys (frame + y component).
    pub fn eye_y_keys(&self) -> Vec<KeyRef> {
        self.eye_keys
            .iter()
            .map(|&(frame, p)| KeyRef { frame, value: p.y })
            .collect()
    }

    /// Eye-Z lane keys (frame + z component).
    pub fn eye_z_keys(&self) -> Vec<KeyRef> {
        self.eye_keys
            .iter()
            .map(|&(frame, p)| KeyRef { frame, value: p.z })
            .collect()
    }

    /// Target-position lane keys (frame + |target| magnitude, for display).
    pub fn target_keys_ref(&self) -> Vec<KeyRef> {
        self.target_keys
            .iter()
            .map(|&(frame, p)| KeyRef {
                frame,
                value: p.length(),
            })
            .collect()
    }

    /// Focus-distance channel keys.
    pub fn focus_keys(&self) -> Vec<KeyRef> {
        self.focus_distance
            .keys()
            .iter()
            .map(|k| KeyRef {
                frame: k.frame,
                value: k.value,
            })
            .collect()
    }

    /// Author an **eye** key at `frame` from a live viewport framing, also
    /// seeding the `target` (= focus) and `fov_y` so a single click captures a
    /// usable pose. Replaces any existing eye/target key on that frame, then
    /// rebuilds the splines so the new key evaluates exactly.
    ///
    /// `eval(frame)` after this returns `eye == eye_from_framing(framing)`.
    pub fn set_key_from_framing(&mut self, framing: &CineFraming, frame: i32, fov_y: f32) {
        let eye = eye_from_framing(framing);
        upsert_pos_key(&mut self.eye_keys, frame, eye);
        upsert_pos_key(&mut self.target_keys, frame, framing.focus);
        self.fov_y_deg.insert(Key {
            frame,
            value: fov_y.to_degrees(),
            interp: Interp::Cubic {
                tan_in: 0.0,
                tan_out: 0.0,
            },
        });
        if frame > self.duration {
            self.duration = frame;
        }
        self.rebuild_arc_lut();
    }

    /// Frame rate of the sequence.
    pub fn fps(&self) -> FrameRate {
        self.fps
    }

    /// Total duration in frames.
    pub fn duration(&self) -> i32 {
        self.duration
    }

    /// The cinematic config (ease/banking/sensor constants).
    pub fn config(&self) -> &CinematicConfig {
        &self.config
    }

    /// Arc-length LUT over the eye spline (constant-speed playback).
    pub fn arc_lut(&self) -> &ArcLengthLut {
        &self.arc_lut
    }

    /// Build the position/target splines and the arc-length LUT from the
    /// current keys. Call after any key edit. Pure function of the keys.
    pub fn rebuild_arc_lut(&mut self) {
        self.eye_spline = build_spline(&self.eye_keys, self.config.catmull_rom_alpha);
        self.target_spline = build_spline(&self.target_keys, self.config.catmull_rom_alpha);
        self.arc_lut = ArcLengthLut::build(
            &self.eye_spline,
            self.config.arc_lut_samples_per_segment as usize,
        );
    }

    /// PURE: frame → renderer-ready pose. No I/O, no GPU, never panics. Clamps
    /// `frame` to `[0, duration]`.
    ///
    /// Position/target are evaluated frame-direct (the spline interpolates each
    /// control point exactly at its keyframe); scalar channels are sampled with
    /// cubic-Hermite; photographic controls are converted to renderer scalars.
    pub fn eval(&self, frame: f64) -> CinePose {
        let f = frame.clamp(0.0, self.duration as f64);
        let ff = f as f32;

        let eye = eval_positions(&self.eye_spline, &self.eye_keys, ff);
        let target = eval_positions(&self.target_spline, &self.target_keys, ff);

        let roll = self.roll.sample(f);
        let fov_y = self.fov_y_deg.sample(f).to_radians();
        let fstop = self.aperture_fstop.sample(f);
        let focal_mm = self.focal_length_mm.sample(f);
        let focus_distance = self.focus_distance.sample(f);

        // shutter angle (deg) -> frame-centered open/close.
        let angle = self.shutter_angle_deg.sample(f);
        let (shutter_open, shutter_close) = shutter_from_angle(angle);

        let bokeh_blades = self.bokeh_shape.blades();
        let exposure_ev = self.exposure_ev.sample(f);

        CinePose {
            eye,
            target,
            roll,
            fov_y,
            lens_radius: lens_radius_from(focal_mm, fstop),
            focus_distance,
            shutter_open,
            shutter_close,
            bokeh_blades,
            exposure_ev,
        }
    }

    /// Parse a sequence from its JSON representation, then build the splines and
    /// LUT so it is immediately evaluable.
    pub fn load_json(s: &str) -> Result<Self, serde_json::Error> {
        let dto: SequenceDto = serde_json::from_str(s)?;
        Ok(Self::from_dto(dto))
    }

    /// Serialize the sequence to JSON (round-trips with `load_json`).
    pub fn save_json(&self) -> String {
        let dto = self.to_dto();
        serde_json::to_string_pretty(&dto).expect("CameraSequence serialization is infallible")
    }

    fn from_dto(dto: SequenceDto) -> Self {
        let mut eye_keys: Vec<(i32, Vec3)> = dto
            .eye_keys
            .iter()
            .map(|&(fr, p)| (fr, Vec3::from(p)))
            .collect();
        eye_keys.sort_by_key(|k| k.0);
        let mut target_keys: Vec<(i32, Vec3)> = dto
            .target_keys
            .iter()
            .map(|&(fr, p)| (fr, Vec3::from(p)))
            .collect();
        target_keys.sort_by_key(|k| k.0);

        let mut seq = Self {
            fps: dto.fps,
            duration: dto.duration,
            eye_keys,
            target_keys,
            roll: channel_from_pairs(&dto.roll),
            fov_y_deg: channel_from_pairs(&dto.fov_y_deg),
            aperture_fstop: channel_from_pairs(&dto.aperture_fstop),
            focal_length_mm: channel_from_pairs(&dto.focal_length_mm),
            focus_distance: focus_channel_from_pairs(&dto.focus_distance),
            focus_mode: dto.focus_mode,
            shutter_angle_deg: channel_from_pairs(&dto.shutter_angle_deg),
            bokeh_shape: dto.bokeh_shape,
            exposure_ev: channel_from_pairs(&dto.exposure_ev),
            config: dto.config,
            eye_spline: Spline3::centripetal(&[Vec3::ZERO], &[0], dto.config.catmull_rom_alpha),
            target_spline: Spline3::centripetal(&[Vec3::ZERO], &[0], dto.config.catmull_rom_alpha),
            arc_lut: ArcLengthLut::build(
                &Spline3::centripetal(&[Vec3::ZERO], &[0], dto.config.catmull_rom_alpha),
                1,
            ),
        };
        seq.rebuild_arc_lut();
        seq
    }

    fn to_dto(&self) -> SequenceDto {
        SequenceDto {
            fps: self.fps,
            duration: self.duration,
            eye_keys: self
                .eye_keys
                .iter()
                .map(|&(fr, p)| (fr, p.to_array()))
                .collect(),
            target_keys: self
                .target_keys
                .iter()
                .map(|&(fr, p)| (fr, p.to_array()))
                .collect(),
            roll: pairs_from_channel(&self.roll),
            fov_y_deg: pairs_from_channel(&self.fov_y_deg),
            aperture_fstop: pairs_from_channel(&self.aperture_fstop),
            focal_length_mm: pairs_from_channel(&self.focal_length_mm),
            focus_distance: pairs_from_channel(&self.focus_distance),
            focus_mode: self.focus_mode,
            shutter_angle_deg: pairs_from_channel(&self.shutter_angle_deg),
            bokeh_shape: self.bokeh_shape,
            exposure_ev: pairs_from_channel(&self.exposure_ev),
            config: self.config,
        }
    }
}

/// Build a focus channel. A two-key focus channel uses `Cubic` zero-tangent
/// ease so a rack focus interpolates as a smooth `3t²-2t³` ramp (matches the
/// Task-4 acceptance: focus 10→50 at the midpoint = 30.0).
fn focus_channel_from_pairs(pairs: &[(i32, f32)]) -> Channel<f32> {
    let mut ch = Channel::<f32>::new();
    for &(frame, value) in pairs {
        ch.insert(Key {
            frame,
            value,
            interp: Interp::Cubic {
                tan_in: 0.0,
                tan_out: 0.0,
            },
        });
    }
    ch
}

/// Frame-centered shutter window from a shutter angle in degrees.
///
/// `frac = angle/360`; `open = 0.5 - frac/2`, `close = 0.5 + frac/2`. A 180°
/// shutter → `0.375 / 0.625`. A zero angle → `0.5 / 0.5` (no motion blur).
fn shutter_from_angle(angle_deg: f32) -> (f32, f32) {
    let frac = (angle_deg / 360.0).clamp(0.0, 1.0);
    (0.5 - frac / 2.0, 0.5 + frac / 2.0)
}

/// Insert-or-replace a position key on a frame-sorted lane.
fn upsert_pos_key(keys: &mut Vec<(i32, Vec3)>, frame: i32, pos: Vec3) {
    match keys.binary_search_by_key(&frame, |k| k.0) {
        Ok(i) => keys[i].1 = pos,
        Err(i) => keys.insert(i, (frame, pos)),
    }
}

/// Build a position spline from `(frame, pos)` keys, falling back to a single
/// origin point when there are no keys (so the spline is always valid).
fn build_spline(keys: &[(i32, Vec3)], alpha: f32) -> Spline3 {
    if keys.is_empty() {
        return Spline3::centripetal(&[Vec3::ZERO], &[0], alpha);
    }
    let points: Vec<Vec3> = keys.iter().map(|k| k.1).collect();
    let frames: Vec<i32> = keys.iter().map(|k| k.0).collect();
    Spline3::centripetal(&points, &frames, alpha)
}

/// Evaluate position frame-direct, clamping to the key range. With one key the
/// position is constant; with none the position is the origin.
fn eval_positions(spline: &Spline3, keys: &[(i32, Vec3)], frame: f32) -> Vec3 {
    if keys.is_empty() {
        return Vec3::ZERO;
    }
    if keys.len() == 1 {
        return keys[0].1;
    }
    spline.eval_frame(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_exact_pose_and_focus() {
        let json = r#"{ "fps":{"num":60,"den":1}, "duration":120,
            "eye_keys":[[0,[0,0,0]],[30,[1,2,0]],[90,[5,2,0]],[120,[6,0,0]]],
            "target_keys":[[0,[0,0,0]],[120,[0,0,0]]],
            "focus_distance":[[0,10.0],[30,50.0]],
            "aperture_fstop":[[0,2.8]], "focal_length_mm":[[0,50.0]],
            "fov_y_deg":[[0,45.0]] }"#;
        let mut seq = CameraSequence::load_json(json).unwrap();
        seq.rebuild_arc_lut();
        let p = seq.eval(30.0);
        println!(
            "eval(30) eye = ({:.3}, {:.3}, {:.3})",
            p.eye().x,
            p.eye().y,
            p.eye().z
        );
        assert!(
            (p.eye() - glam::vec3(1.0, 2.0, 0.0)).length() < 1e-4,
            "eye {:?}",
            p.eye()
        );
        // focus 10->50 cubic-Hermite at the midpoint frame 15 (zero-tangent
        // ease) = 10 + 40*0.5 = 30.0
        let p2 = seq.eval(15.0);
        println!("eval(15) focus_distance = {:.3}", p2.focus_distance());
        assert!(
            (p2.focus_distance() - 30.0).abs() < 1e-3,
            "focus {}",
            p2.focus_distance()
        );
        // JSON round-trip
        let seq2 = CameraSequence::load_json(&seq.save_json()).unwrap();
        assert!((seq2.eval(30.0).eye() - p.eye()).length() < 1e-4);
    }
}
