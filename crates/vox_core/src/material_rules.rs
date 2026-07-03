//! Data-driven terrain MATERIAL placement rules + a deterministic softmax evaluator.
//!
//! The architecture cure (see `urban_horizon/docs/superpowers/specs/
//! 2026-06-30-terrain-material-architecture-audit.md`): terrain material is AUTHORED at
//! map-build, not re-derived per-pixel in the renderer. A `MaterialRuleSet` is DATA — a
//! list of material layers, each with a base prior + driver terms — evaluated on the CPU
//! over the baked per-cell `DriverSample` (slope/curvature/altitude/moisture/flow/aspect)
//! to produce a CONVEX (Σ=1) per-cell material weight vector that is baked into the spray
//! field. The renderer then only SAMPLES + blends those weights.
//!
//! [`standard_terrain`] is MODELLED ON the megakernel softmax (megakernel.slang:4084-4125)
//! but is a DELIBERATE LOOK CHANGE, not a byte-for-byte reproduction. It authors from the
//! TRUE-heightmap curvature/slope drivers, whereas the live GPU path runs with
//! `slope_lowpass=1.0` (unbridged by the game) which ZEROES its curvature driver — so today's
//! image has the convex/concave/cavity terms DEAD and a facet-suppressed slope. The
//! central-difference heightmap drivers are facet-noise-immune BY CONSTRUCTION, so the GPU's
//! lowpass + flat-guard band-aids are intentionally NOT re-imported (terrain law 7: fix the
//! cause, don't carry the band-aid). Expect a look delta at the Step-4 witness; the rules are
//! authorable (config / per-biome), so tune via config and witness against INTENT (green
//! buildable core + rock borders), never against today's degraded image.
//!
//! Engine-generic + game-agnostic: this crate knows nothing about biomes or textures —
//! a material is just a `u8` palette index. DETERMINISTIC: f64 math, fixed layer order,
//! no HashMap/RNG.

/// The flat global MATERIAL palette (spray channel index = material). The renderer's
/// `ChannelTable` binds each index to a texture stack; the rules place them.
pub mod mat {
    pub const GRASS: u8 = 0;
    pub const LUSH: u8 = 1;
    pub const DRY: u8 = 2;
    pub const FOREST_FLOOR: u8 = 3;
    pub const DIRT: u8 = 4;
    pub const ROCK: u8 = 5;
    pub const SCREE: u8 = 6;
    pub const SAND: u8 = 7;
    pub const SNOW: u8 = 8;
    pub const WET_SAND: u8 = 9;
    pub const MUD: u8 = 10;
    pub const SILT: u8 = 11;
    /// Palette size (== `SPRAY_CHANNELS`).
    pub const COUNT: usize = 12;
}

/// Per-cell driver values the rules read. The CALLER computes these from the baked
/// `DriverFields` + altitude + a deterministic dirt-patch noise (the composites
/// `dirt_flat` and `snow_hold_alt` are precomputed so the rule terms stay simple
/// linear/smoothstep/band responses). Ranges are each driver's natural range.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DriverSample {
    pub steep: f64,         // [0,1] 0=flat … 1=vertical
    pub concave: f64,       // [0,1] hollow fraction
    pub convex: f64,        // [0,1] ridge fraction
    pub cavity: f64,        // [0,1] sheltered-concave fraction
    pub alt_rock: f64,      // [0,1] altitude gate for exposed rock
    pub alt_snow: f64,      // [0,1] altitude gate for snow
    pub snow_hold_alt: f64, // [0,1] snow_hold × alt_snow (precomputed product)
    pub dirt_flat: f64,     // composite low-slope dirt/dry patch term
    pub moisture: f64,      // [0,1]
    pub flow: f64,          // [0,1]
    pub aspect: f64,        // [-1,1] sun-facing
}

/// Which driver a rule term reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Driver {
    Steep,
    Concave,
    Convex,
    Cavity,
    AltRock,
    AltSnow,
    SnowHoldAlt,
    DirtFlat,
    Moisture,
    Flow,
    Aspect,
}

impl Driver {
    #[inline]
    fn value(self, s: &DriverSample) -> f64 {
        match self {
            Driver::Steep => s.steep,
            Driver::Concave => s.concave,
            Driver::Convex => s.convex,
            Driver::Cavity => s.cavity,
            Driver::AltRock => s.alt_rock,
            Driver::AltSnow => s.alt_snow,
            Driver::SnowHoldAlt => s.snow_hold_alt,
            Driver::DirtFlat => s.dirt_flat,
            Driver::Moisture => s.moisture,
            Driver::Flow => s.flow,
            Driver::Aspect => s.aspect,
        }
    }
}

#[inline]
fn smoothstep(lo: f64, hi: f64, x: f64) -> f64 {
    let t = ((x - lo) / (hi - lo).max(1e-9)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The response curve applied to a driver value before the gain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Curve {
    /// The raw driver value.
    Linear,
    /// `smoothstep(lo, hi, x)` — a soft ramp.
    Smoothstep { lo: f64, hi: f64 },
    /// A bump: `smoothstep(lo1,hi1,x) * (1 - smoothstep(lo2,hi2,x))` (opens then closes).
    Band { lo1: f64, hi1: f64, lo2: f64, hi2: f64 },
}

impl Curve {
    #[inline]
    fn apply(self, x: f64) -> f64 {
        match self {
            Curve::Linear => x,
            Curve::Smoothstep { lo, hi } => smoothstep(lo, hi, x),
            Curve::Band { lo1, hi1, lo2, hi2 } => smoothstep(lo1, hi1, x) * (1.0 - smoothstep(lo2, hi2, x)),
        }
    }
}

/// One additive term of a layer's suitability logit: `gain × curve(driver)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Term {
    pub driver: Driver,
    pub curve: Curve,
    pub gain: f64,
}

impl Term {
    #[inline]
    pub fn new(driver: Driver, curve: Curve, gain: f64) -> Self {
        Self { driver, curve, gain }
    }
}

/// One material layer: its palette index + a base prior + the driver terms that raise/
/// lower its suitability. The softmax over all layers' logits gives the convex weights.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialLayer {
    pub material: u8,
    pub prior: f64,
    pub terms: Vec<Term>,
}

impl MaterialLayer {
    fn logit(&self, s: &DriverSample) -> f64 {
        let mut v = self.prior;
        for t in &self.terms {
            v += t.gain * t.curve.apply(t.driver.value(s));
        }
        v
    }
}

/// A complete terrain material ruleset (the layers + the softmax temperature).
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialRuleSet {
    pub layers: Vec<MaterialLayer>,
    /// Softmax temperature T (small = crisp argmax pick, large = soft blend).
    pub temp: f64,
}

impl MaterialRuleSet {
    /// Evaluate the convex (Σ=1) per-material weights for a cell. Deterministic f64
    /// softmax over the layer logits; returns `(material, weight)` per layer in layer
    /// order (weights sum to 1). BOUNDED by construction — a convex blend of in-gamut
    /// materials can never blanket/whiten (the no-white law).
    pub fn evaluate(&self, s: &DriverSample) -> Vec<(u8, f32)> {
        let inv_t = 1.0 / self.temp.max(1e-3);
        // max-subtract for numerical stability (deterministic: fixed layer order).
        let mut max_l = f64::NEG_INFINITY;
        let logits: Vec<f64> = self
            .layers
            .iter()
            .map(|l| {
                let v = l.logit(s) * inv_t;
                if v > max_l {
                    max_l = v;
                }
                v
            })
            .collect();
        let mut sum = 0.0f64;
        let exps: Vec<f64> = logits
            .iter()
            .map(|&v| {
                let e = (v - max_l).exp();
                sum += e;
                e
            })
            .collect();
        let inv_sum = 1.0 / sum.max(1e-30);
        self.layers
            .iter()
            .zip(exps)
            .map(|(l, e)| (l.material, (e * inv_sum) as f32))
            .collect()
    }

    /// The single highest-weight material (the argmax pick) — for tests / a crisp tier.
    pub fn dominant(&self, s: &DriverSample) -> u8 {
        self.evaluate(s)
            .into_iter()
            .fold((0u8, -1.0f32), |acc, (m, w)| if w > acc.1 { (m, w) } else { acc })
            .0
    }
}

/// Config-sourced constants for [`standard_terrain`] (CONFIG-FIRST law: a mis-tuned look
/// is a zero-rebuild re-run, not a hardcoded edit). The defaults reproduce the megakernel
/// literals (megakernel.slang:4091-4125) so the standard ruleset is byte-for-byte today's
/// placement when wired.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainRuleConfig {
    pub temp: f64,
    // grass
    pub grass_prior: f64,
    pub grass_steep_suppress: f64,
    pub grass_concave: f64,
    pub grass_convex: f64,
    pub grass_dirt_cede: f64,
    // dirt
    pub dirt_prior: f64,
    pub dirt_lobe_gain: f64,
    /// The shoulder steepness band the dirt lobe opens/closes over (kernel
    /// `u_terrain_dirt_lobe_lo/hi/close_lo/close_hi`). Config-first so the dirt band
    /// can be retuned without editing this ruleset (and cannot drift from the kernel).
    pub dirt_lobe_lo: f64,
    pub dirt_lobe_hi: f64,
    pub dirt_lobe_close_lo: f64,
    pub dirt_lobe_close_hi: f64,
    pub dirt_cavity: f64,
    /// SEPARATE cavity gain (kernel `u_slope_cavity_amp`) added to `dirt_cavity` for the
    /// dirt Cavity term — kept its OWN knob so the two stay independent (the kernel computes
    /// `(dirt_cavity + cavity_amp) * cavity`); folding them would let a cavity_amp retune drift.
    pub dirt_cavity_amp: f64,
    pub dirt_convex_suppress: f64,
    // rock
    pub rock_prior: f64,
    pub rock_ramp_gain: f64,
    pub rock_slope_lo: f64,
    pub rock_slope_hi: f64,
    pub rock_convex: f64,
    pub rock_alt: f64,
    // snow
    pub snow_prior: f64,
    pub snow_alt_gain: f64,
    pub snow_hold_gain: f64,
}

impl Default for TerrainRuleConfig {
    fn default() -> Self {
        Self {
            temp: 0.55,
            grass_prior: 1.10,
            grass_steep_suppress: 2.40,
            grass_concave: 0.65,
            grass_convex: 0.30,
            grass_dirt_cede: 0.70,
            dirt_prior: -0.55,
            dirt_lobe_gain: 1.65,
            dirt_lobe_lo: 0.12,
            dirt_lobe_hi: 0.40,
            dirt_lobe_close_lo: 0.55,
            dirt_lobe_close_hi: 0.85,
            dirt_cavity: 0.40,
            dirt_cavity_amp: 0.12,
            dirt_convex_suppress: 0.20,
            rock_prior: -0.10,   // was -0.85 (rock too rare) → rock actually wins on the massif slopes
            rock_ramp_gain: 3.60,
            // Slope window: rock ramps in past the dirt-shoulder band and is full by the
            // massif-flank steepness (founders flanks read 0.3–0.61). Gentle buildable
            // hills (<0.18) never rock; summits above the rock line are handled by
            // AltRock, so the window no longer has to reach down to near-flat slopes.
            rock_slope_lo: 0.22,
            rock_slope_hi: 0.50,
            rock_convex: 0.90,
            // Bare-summit gain: at full AltRock (above the caller's rock altitude line)
            // the rock logit (rock_prior + rock_alt = 2.10) must decisively beat
            // grass_prior 1.10 so a smooth low-slope dome CAP authors rock, with the
            // crossover blending inside the caller's altitude band. Slope-only rock
            // leaves a green cap ringed by a rock annulus (map_spray-proven).
            rock_alt: 2.20,
            snow_prior: -3.50,
            snow_alt_gain: 5.50,
            snow_hold_gain: 1.20,
        }
    }
}

/// The default terrain ruleset — MODELS the megakernel grass/dirt/rock/snow softmax
/// (megakernel.slang:4091-4125) over the flat palette, as a DELIBERATE look change authored
/// from true-heightmap drivers (see the module doc: the live GPU curvature driver is dead under
/// `slope_lowpass=1.0`, so this is NOT a byte-for-byte reproduction and we do NOT re-import that
/// band-aid). NOTE: the megakernel's rock CLOUD-SPREAD HALO term (`H`, a NEIGHBORHOOD diffusion
/// that softens cliff borders) is OMITTED here — pointwise drivers cannot reproduce a
/// neighborhood spread; the soft rock-border halo is a DEFERRED baked dilated-steep driver (a
/// known Step-4 look risk, flagged by the design red-team).
pub fn standard_terrain(c: &TerrainRuleConfig) -> MaterialRuleSet {
    use Curve::*;
    use Driver::*;
    MaterialRuleSet {
        temp: c.temp,
        layers: vec![
            // GRASS — flat + concave + low; the default winner.
            MaterialLayer {
                material: mat::GRASS,
                prior: c.grass_prior,
                terms: vec![
                    Term::new(Steep, Linear, -c.grass_steep_suppress),
                    Term::new(Concave, Linear, c.grass_concave),
                    Term::new(Convex, Linear, -c.grass_convex),
                    Term::new(DirtFlat, Linear, -c.grass_dirt_cede),
                ],
            },
            // DIRT — the shoulder lobe + sheltered cavities + the low-slope patch.
            MaterialLayer {
                material: mat::DIRT,
                prior: c.dirt_prior,
                terms: vec![
                    Term::new(
                        Steep,
                        Band {
                            lo1: c.dirt_lobe_lo,
                            hi1: c.dirt_lobe_hi,
                            lo2: c.dirt_lobe_close_lo,
                            hi2: c.dirt_lobe_close_hi,
                        },
                        c.dirt_lobe_gain,
                    ),
                    // Cavity gain = dirt_cavity + cavity_amp (the kernel's two independent knobs).
                    Term::new(Cavity, Linear, c.dirt_cavity + c.dirt_cavity_amp),
                    Term::new(Convex, Linear, -c.dirt_convex_suppress),
                    Term::new(DirtFlat, Linear, 1.0),
                ],
            },
            // ROCK — steep ramp + convex + altitude (halo omitted, see above).
            MaterialLayer {
                material: mat::ROCK,
                prior: c.rock_prior,
                terms: vec![
                    Term::new(Steep, Smoothstep { lo: c.rock_slope_lo, hi: c.rock_slope_hi }, c.rock_ramp_gain),
                    Term::new(Convex, Linear, c.rock_convex),
                    Term::new(AltRock, Linear, c.rock_alt),
                ],
            },
            // SNOW — high altitude on flatter shelves; off below the snow line.
            MaterialLayer {
                material: mat::SNOW,
                prior: c.snow_prior,
                terms: vec![
                    Term::new(AltSnow, Linear, c.snow_alt_gain),
                    Term::new(SnowHoldAlt, Linear, c.snow_hold_gain),
                ],
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> MaterialRuleSet {
        standard_terrain(&TerrainRuleConfig::default())
    }

    #[test]
    fn flat_low_ground_is_grass() {
        // Flat, no altitude, no patches → grass must dominate (the buildable core).
        let s = DriverSample::default();
        assert_eq!(rules().dominant(&s), mat::GRASS);
    }

    #[test]
    fn steep_face_is_rock() {
        // A genuinely steep face → rock wins (the relief borders).
        let s = DriverSample { steep: 0.9, convex: 0.3, ..Default::default() };
        assert_eq!(rules().dominant(&s), mat::ROCK);
    }

    #[test]
    fn high_altitude_flat_is_snow() {
        // Above the snow line, flat shelf → snow.
        let s = DriverSample { alt_snow: 1.0, snow_hold_alt: 1.0, ..Default::default() };
        assert_eq!(rules().dominant(&s), mat::SNOW);
    }

    #[test]
    fn weights_are_convex() {
        // Σ weights == 1 (no-white law: a convex blend can never blanket).
        let s = DriverSample { steep: 0.45, convex: 0.2, dirt_flat: 0.3, ..Default::default() };
        let w: f32 = rules().evaluate(&s).iter().map(|(_, w)| *w).sum();
        assert!((w - 1.0).abs() < 1e-4, "weights must sum to 1, got {w}");
    }

    #[test]
    fn evaluate_is_deterministic() {
        let s = DriverSample { steep: 0.5, concave: 0.3, moisture: 0.4, flow: 0.2, ..Default::default() };
        assert_eq!(rules().evaluate(&s), rules().evaluate(&s));
    }

    #[test]
    fn shoulder_slope_yields_dirt_band() {
        // The shoulder steepness (past flat, before full rock) opens the dirt lobe.
        let s = DriverSample { steep: 0.28, dirt_flat: 0.2, ..Default::default() };
        let w = rules().evaluate(&s);
        let dirt = w.iter().find(|(m, _)| *m == mat::DIRT).unwrap().1;
        let rock = w.iter().find(|(m, _)| *m == mat::ROCK).unwrap().1;
        assert!(dirt > rock, "shoulder slope should favor dirt over rock: dirt={dirt} rock={rock}");
    }
}
