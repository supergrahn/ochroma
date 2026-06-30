//! `SprayField` — a world-space ground-cover weight grid (the "spray box" core).
//!
//! ONE world-space weight field, TWO consumers (per the spray-terrain design,
//! `urban_horizon/docs/superpowers/specs/2026-06-21-spray-terrain-grass-architecture.md`):
//!
//! 1. the **megakernel ground shader** samples it bilinearly per fragment and
//!    top-2 convex-blends the channel textures (kills the biome grid seams), and
//! 2. (later) the **CPU scatter** reads the same field to drive grass-patch
//!    density — which is exactly *why* the field is a WORLD GRID and not a
//!    per-vertex BLAS attribute: the scatter cannot barycentric-interpolate a GPU
//!    attribute, but it can sample a world grid.
//!
//! The field is engine-generic (this crate stays game-agnostic): it is just a
//! grid of per-channel `u8` weights that sum to ≈255 per cell. What each channel
//! *means* (which texture / grass proto) is the game's `ChannelTable`, not here.
//!
//! # Determinism
//! The `u8` weights are bit-exact — the procedural baseline + every brush stroke
//! resolve to the same bytes on every machine. Bilinear sampling (consumer side)
//! is APPEARANCE-only and never feeds the sim. Brush strokes are recorded as a
//! replay-exact delta log over the procedural baseline (same model as terraform):
//! re-running [`SprayField::apply_stroke`] in log order rebuilds the exact field.

/// Number of MATERIAL channels carried per cell. Matches the megakernel's
/// `SPRAY_CHANNELS` and the game's `ChannelTable` length. STEP 3: the channels are the
/// flat GLOBAL MATERIAL palette (grass/lush/dry/forest-floor/dirt/rock/scree/sand/snow/
/// wet-sand/mud/silt — see `material_rules::mat`), NOT biomes; biome became a map-build
/// INPUT to the material rules. KEEP IN SYNC with megakernel.slang `SPRAY_CHANNELS`.
pub const SPRAY_CHANNELS: usize = 12;

/// u32 words per cell when the field is packed for GPU upload (4 u8 channels per word,
/// rounded up). 12 channels → 3 words. Generalises the old hardcoded 2.
pub const SPRAY_WORDS_PER_CELL: usize = SPRAY_CHANNELS.div_ceil(4);

/// Per-cell weights sum to this (a `u8` budget). Σ≈255 (exact after renormalize,
/// modulo the unavoidable ±`SPRAY_CHANNELS` rounding spread across channels).
pub const SPRAY_SUM: u16 = 255;

/// One painted stroke — the replay-logged unit of authoring. Stored as a delta
/// over the procedural baseline so the field is reproducible from
/// `bake_baseline()` + the ordered stroke log (terraform's model).
///
/// All fields are plain data (no floats in the *stored* identity beyond the
/// brush params, which are applied through a fixed deterministic kernel) so the
/// log is byte-stable to serialize.
/// Stroke-log schema: which index space `channel` lives in. `0` (the serde default for
/// pre-Step-3 saved logs) = the LEGACY BIOME-indexed channel; the GAME must remap such a
/// stroke's `channel` from biome → its material channel before replay (the spray field is
/// now MATERIAL-indexed). New strokes carry [`STROKE_SCHEMA_MATERIAL`].
pub const STROKE_SCHEMA_LEGACY_BIOME: u32 = 0;
pub const STROKE_SCHEMA_MATERIAL: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SprayStroke {
    /// Index space of `channel` (see [`STROKE_SCHEMA_MATERIAL`]). `#[serde(default)]` →
    /// old saved logs deserialize as `0` (legacy biome) so they are DETECTABLE + remappable.
    #[serde(default)]
    pub schema_version: u32,
    /// Channel raised by this stroke (`0..SPRAY_CHANNELS`).
    pub channel: u8,
    /// Brush centre in world XZ metres.
    pub center: [f32; 2],
    /// Brush radius in world metres (falloff reaches 0 at the rim).
    pub radius: f32,
    /// Peak weight added at the centre, in `[0,1]` (scaled into the u8 budget).
    pub strength: f32,
    /// Coverage `[0,1]`: fraction of the falloff actually applied (a sparse vs
    /// solid spray — 1.0 = solid, <1 = lighter touch). Deterministic (no RNG):
    /// it simply scales the per-cell add, so the field stays bit-exact.
    pub density: f32,
}

/// A world-space ground-cover weight grid. Row-major, `idx = iz*res_x + ix`.
#[derive(Debug, Clone, PartialEq)]
pub struct SprayField {
    /// Grid resolution (cells) in X and Z. Independent of the heightmap.
    res_x: u32,
    res_z: u32,
    /// World-space origin (XZ) of cell (0,0)'s centre, metres.
    origin: [f32; 2],
    /// World metres per cell (square cells). ~1.5 m/cell at 1024².
    cell_size: f32,
    /// Per-cell channel weights, Σ≈255 per cell. `len == res_x*res_z`.
    weights: Vec<[u8; SPRAY_CHANNELS]>,
    /// The PROCEDURAL baseline (no strokes) — the field `weights` is reset to this
    /// before replaying the stroke log, so strokes are a delta over the baseline.
    baseline: Vec<[u8; SPRAY_CHANNELS]>,
    /// Replay-exact ordered stroke log (delta over `baseline`).
    strokes: Vec<SprayStroke>,
    /// Bumped on every mutation. The renderer uploads only when this changes.
    version: u64,
}

impl SprayField {
    /// A new field of `res_x*res_z` cells over the world rect anchored at
    /// `origin` (cell-0 centre) with `cell_size` m/cell. Every cell starts as
    /// 100% channel 0 (the default ground) — call [`SprayField::seed_from`] to
    /// install the procedural baseline.
    pub fn new(res_x: u32, res_z: u32, origin: [f32; 2], cell_size: f32) -> Self {
        let n = (res_x as usize) * (res_z as usize);
        let mut c0 = [0u8; SPRAY_CHANNELS];
        c0[0] = SPRAY_SUM as u8;
        let weights = vec![c0; n];
        Self {
            res_x,
            res_z,
            origin,
            cell_size: cell_size.max(1e-3),
            baseline: weights.clone(),
            weights,
            strokes: Vec::new(),
            version: 1,
        }
    }

    pub fn res_x(&self) -> u32 {
        self.res_x
    }
    pub fn res_z(&self) -> u32 {
        self.res_z
    }
    pub fn origin(&self) -> [f32; 2] {
        self.origin
    }
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }
    pub fn version(&self) -> u64 {
        self.version
    }
    /// Raw row-major weights (`len == res_x*res_z`), for upload / debug.
    pub fn weights(&self) -> &[[u8; SPRAY_CHANNELS]] {
        &self.weights
    }
    /// The recorded stroke log (replay-exact authoring history).
    pub fn strokes(&self) -> &[SprayStroke] {
        &self.strokes
    }

    /// World XZ → fractional cell coordinate (cell centres at integer coords).
    #[inline]
    fn world_to_cell(&self, wx: f32, wz: f32) -> (f32, f32) {
        (
            (wx - self.origin[0]) / self.cell_size,
            (wz - self.origin[1]) / self.cell_size,
        )
    }

    /// Install the procedural baseline cell-by-cell from a sampler that returns
    /// per-channel weights for the world XZ at each cell centre. The sampler's
    /// output is renormalized to Σ=255 (so the caller need not). This sets BOTH
    /// `baseline` and the live `weights`, then replays any existing stroke log on
    /// top so the field stays the baseline-plus-strokes invariant.
    ///
    /// `sample(wx, wz) -> [f32; SPRAY_CHANNELS]` — relative channel weights ≥0.
    pub fn seed_from<F>(&mut self, mut sample: F)
    where
        F: FnMut(f32, f32) -> [f32; SPRAY_CHANNELS],
    {
        for iz in 0..self.res_z {
            for ix in 0..self.res_x {
                let wx = self.origin[0] + ix as f32 * self.cell_size;
                let wz = self.origin[1] + iz as f32 * self.cell_size;
                let raw = sample(wx, wz);
                let q = quantize_to_sum(&raw);
                self.baseline[(iz * self.res_x + ix) as usize] = q;
            }
        }
        // Re-derive the live field = baseline + replayed strokes.
        self.rebuild_from_log();
        self.version += 1;
    }

    /// Reset `weights` to `baseline` and replay the stroke log in order — the
    /// determinism contract (terraform-style delta-over-baseline).
    fn rebuild_from_log(&mut self) {
        self.weights.copy_from_slice(&self.baseline);
        let strokes = std::mem::take(&mut self.strokes);
        for s in &strokes {
            self.apply_stroke_unlogged(s);
        }
        self.strokes = strokes;
    }

    /// Paint one texture stroke (the spray-box backend). Raises `channel`'s
    /// weight inside `radius` with a smooth falloff scaled by `strength*density`,
    /// then renormalizes every touched cell to Σ=255. Logged for replay; bumps
    /// `version`. Returns the number of cells changed.
    ///
    /// `radius`/`strength`/`density` are the real spray-box params:
    /// - `radius` m: extent of the brush (falloff → 0 at the rim).
    /// - `strength` [0,1]: peak weight added at the centre.
    /// - `density` [0,1]: how solidly the falloff is applied (a light vs full spray).
    pub fn paint_texture(
        &mut self,
        channel: u8,
        center: [f32; 2],
        radius: f32,
        strength: f32,
        density: f32,
    ) -> usize {
        let stroke = SprayStroke {
            schema_version: STROKE_SCHEMA_MATERIAL,
            channel,
            center,
            radius,
            strength,
            density,
        };
        let changed = self.apply_stroke_unlogged(&stroke);
        self.strokes.push(stroke);
        self.version += 1;
        changed
    }

    /// Replay one already-known stroke (used when rebuilding from a log). Does
    /// NOT touch the log or version — callers manage those.
    pub fn apply_stroke(&mut self, stroke: &SprayStroke) -> usize {
        self.apply_stroke_unlogged(stroke)
    }

    /// The deterministic brush kernel. `channel` out of range, or a degenerate
    /// radius, is a no-op (returns 0).
    fn apply_stroke_unlogged(&mut self, s: &SprayStroke) -> usize {
        let ch = s.channel as usize;
        if ch >= SPRAY_CHANNELS || !(s.radius > 0.0) {
            return 0;
        }
        let strength = s.strength.clamp(0.0, 1.0);
        let density = s.density.clamp(0.0, 1.0);
        let amp = strength * density; // peak fraction of the budget to add
        if amp <= 0.0 {
            return 0;
        }
        // Cell-space bounding box of the brush disc.
        let (cx, cz) = self.world_to_cell(s.center[0], s.center[1]);
        let r_cells = s.radius / self.cell_size;
        let ix0 = (cx - r_cells).floor().max(0.0) as i64;
        let iz0 = (cz - r_cells).floor().max(0.0) as i64;
        let ix1 = (cx + r_cells).ceil().min((self.res_x - 1) as f32) as i64;
        let iz1 = (cz + r_cells).ceil().min((self.res_z - 1) as f32) as i64;
        let r2 = r_cells * r_cells;
        let mut changed = 0usize;
        for iz in iz0..=iz1 {
            for ix in ix0..=ix1 {
                let dx = ix as f32 - cx;
                let dz = iz as f32 - cz;
                let d2 = dx * dx + dz * dz;
                if d2 > r2 {
                    continue;
                }
                // Smooth falloff: 1 at centre → 0 at the rim (smoothstep on the
                // normalized radius). Pure function of geometry → deterministic.
                let t = (d2 / r2).clamp(0.0, 1.0); // 0..1 (squared radius)
                let dist = t.sqrt();
                let fall = smoothstep(1.0, 0.0, dist); // 1 centre, 0 rim
                let add = (amp * fall * SPRAY_SUM as f32).round() as i32;
                if add <= 0 {
                    continue;
                }
                let idx = (iz as usize) * (self.res_x as usize) + ix as usize;
                if raise_channel(&mut self.weights[idx], ch, add) {
                    changed += 1;
                }
            }
        }
        changed
    }

    /// Bilinearly sample the field at world XZ → per-channel weights in `[0,1]`
    /// (Σ≈1). APPEARANCE-only (the megakernel does this on the GPU; this CPU
    /// version mirrors it for the scatter consumer + tests). Edge cells clamp.
    pub fn sample_bilinear(&self, wx: f32, wz: f32) -> [f32; SPRAY_CHANNELS] {
        let (fx, fz) = self.world_to_cell(wx, wz);
        let x0 = fx.floor();
        let z0 = fz.floor();
        let tx = (fx - x0).clamp(0.0, 1.0);
        let tz = (fz - z0).clamp(0.0, 1.0);
        let clampi = |v: f32, hi: u32| (v as i64).clamp(0, (hi - 1) as i64) as usize;
        let ix0 = clampi(x0, self.res_x);
        let ix1 = clampi(x0 + 1.0, self.res_x);
        let iz0 = clampi(z0, self.res_z);
        let iz1 = clampi(z0 + 1.0, self.res_z);
        let row = self.res_x as usize;
        let c00 = &self.weights[iz0 * row + ix0];
        let c10 = &self.weights[iz0 * row + ix1];
        let c01 = &self.weights[iz1 * row + ix0];
        let c11 = &self.weights[iz1 * row + ix1];
        let mut out = [0.0f32; SPRAY_CHANNELS];
        let mut sum = 0.0f32;
        for c in 0..SPRAY_CHANNELS {
            let top = c00[c] as f32 * (1.0 - tx) + c10[c] as f32 * tx;
            let bot = c01[c] as f32 * (1.0 - tx) + c11[c] as f32 * tx;
            let v = top * (1.0 - tz) + bot * tz;
            out[c] = v;
            sum += v;
        }
        if sum > 0.0 {
            for v in out.iter_mut() {
                *v /= sum;
            }
        }
        out
    }

    /// The dominant channel + its `[0,1]` weight at world XZ (nearest cell, no
    /// blend) — for the scatter keep-test and debug.
    pub fn dominant_at(&self, wx: f32, wz: f32) -> (u8, f32) {
        let (fx, fz) = self.world_to_cell(wx, wz);
        let ix = (fx.round() as i64).clamp(0, (self.res_x - 1) as i64) as usize;
        let iz = (fz.round() as i64).clamp(0, (self.res_z - 1) as i64) as usize;
        let cell = &self.weights[iz * self.res_x as usize + ix];
        let mut best = 0usize;
        for c in 1..SPRAY_CHANNELS {
            if cell[c] > cell[best] {
                best = c;
            }
        }
        (best as u8, cell[best] as f32 / SPRAY_SUM as f32)
    }

    /// Pack the field into u32 words for GPU upload: `SPRAY_WORDS_PER_CELL` u32 per cell
    /// (4 channels per word, little-endian byte order — channel `4w..4w+3` in word `w`).
    /// Row-major. Matches the megakernel's `spray_unpack_cell` loop. Channels past
    /// `SPRAY_CHANNELS` in the final word are zero.
    pub fn pack_u32(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.weights.len() * SPRAY_WORDS_PER_CELL);
        for cell in &self.weights {
            for w in 0..SPRAY_WORDS_PER_CELL {
                let mut word = 0u32;
                for k in 0..4 {
                    let c = w * 4 + k;
                    if c < SPRAY_CHANNELS {
                        word |= (cell[c] as u32) << (8 * k);
                    }
                }
                out.push(word);
            }
        }
        out
    }
}

/// The per-channel atlas-slot table for ONE material channel: the base PBR quad
/// (albedo/normal/rough/disp) + the detail-overlay quad (albedo/alpha/normal/disp).
/// A TYPED `#[repr(C)]` replacement for the old stride-8 raw-`i32` comment-contract
/// (megakernel `SPRAY_CH_STRIDE = 8`): the layout is the struct, so a miscount is a
/// compile error, not a silent wrong-channel read. `-1` = absent slot. The host
/// flattens a `[ChannelSlots; SPRAY_CHANNELS]` to i32 (via [`ChannelSlots::to_ints`])
/// for upload — the channel COUNT changed (8→12), the per-channel STRIDE stays 8.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelSlots {
    pub base_albedo: i32,
    pub base_normal: i32,
    pub base_rough: i32,
    pub base_disp: i32,
    pub ov_albedo: i32,
    pub ov_alpha: i32,
    pub ov_normal: i32,
    pub ov_disp: i32,
}

/// i32 ints per channel slot (== megakernel `SPRAY_CH_STRIDE`). Changing this is a
/// kernel-coupled break — do NOT bump it for a wider palette (that's `SPRAY_CHANNELS`).
pub const CHANNEL_SLOT_INTS: usize = 8;

impl Default for ChannelSlots {
    fn default() -> Self {
        // -1 everywhere (0 is a VALID atlas slot, so it can't be the "absent" sentinel).
        Self {
            base_albedo: -1, base_normal: -1, base_rough: -1, base_disp: -1,
            ov_albedo: -1, ov_alpha: -1, ov_normal: -1, ov_disp: -1,
        }
    }
}

impl ChannelSlots {
    /// Flatten to the raw `i32` stride the megakernel reads (`g_spray_channels[c*8 + k]`).
    pub const fn to_ints(&self) -> [i32; CHANNEL_SLOT_INTS] {
        [
            self.base_albedo, self.base_normal, self.base_rough, self.base_disp,
            self.ov_albedo, self.ov_alpha, self.ov_normal, self.ov_disp,
        ]
    }
}

const _: () = assert!(std::mem::size_of::<ChannelSlots>() == CHANNEL_SLOT_INTS * 4);

/// Raise `cell[ch]` by `add` (clamped into the budget) and renormalize the cell
/// to Σ=255 by proportionally scaling the OTHER channels down. Returns true if
/// the cell changed.
fn raise_channel(cell: &mut [u8; SPRAY_CHANNELS], ch: usize, add: i32) -> bool {
    let before = *cell;
    let raised = (cell[ch] as i32 + add).min(SPRAY_SUM as i32) as u32;
    // Sum of the other channels.
    let mut others: u32 = 0;
    for (c, &w) in cell.iter().enumerate() {
        if c != ch {
            others += w as u32;
        }
    }
    let remaining = (SPRAY_SUM as u32).saturating_sub(raised);
    if others == 0 {
        // Nothing to take from: pin this channel to the full budget.
        *cell = [0u8; SPRAY_CHANNELS];
        cell[ch] = SPRAY_SUM as u8;
    } else {
        // Scale the others to fill `remaining`, integer-exact (largest-remainder
        // so Σ lands on exactly SPRAY_SUM).
        let mut scaled = [0u32; SPRAY_CHANNELS];
        let mut frac = [0u32; SPRAY_CHANNELS]; // remainder numerators
        let mut acc: u32 = 0;
        for c in 0..SPRAY_CHANNELS {
            if c == ch {
                scaled[c] = raised;
                acc += raised;
            } else {
                let num = before[c] as u32 * remaining;
                scaled[c] = num / others;
                frac[c] = num % others;
                acc += scaled[c];
            }
        }
        // Distribute the leftover (SPRAY_SUM - acc) to the largest remainders,
        // skipping the raised channel — deterministic tie-break by index.
        let mut leftover = (SPRAY_SUM as u32).saturating_sub(acc);
        while leftover > 0 {
            let mut best = usize::MAX;
            for c in 0..SPRAY_CHANNELS {
                if c == ch {
                    continue;
                }
                if best == usize::MAX || frac[c] > frac[best] {
                    best = c;
                }
            }
            if best == usize::MAX || frac[best] == 0 {
                // No remainders left to assign: dump onto the raised channel.
                scaled[ch] += leftover;
                break;
            }
            scaled[best] += 1;
            frac[best] = 0;
            leftover -= 1;
        }
        for c in 0..SPRAY_CHANNELS {
            cell[c] = scaled[c].min(255) as u8;
        }
    }
    *cell != before
}

/// Quantize relative `f32` weights to a `u8[8]` cell summing to exactly 255
/// (largest-remainder rounding → deterministic, Σ exact).
fn quantize_to_sum(raw: &[f32; SPRAY_CHANNELS]) -> [u8; SPRAY_CHANNELS] {
    let mut total = 0.0f32;
    for &v in raw.iter() {
        total += v.max(0.0);
    }
    if total <= 0.0 {
        let mut c = [0u8; SPRAY_CHANNELS];
        c[0] = SPRAY_SUM as u8;
        return c;
    }
    let target = SPRAY_SUM as f32;
    let mut floored = [0u32; SPRAY_CHANNELS];
    let mut frac = [0.0f32; SPRAY_CHANNELS];
    let mut acc: u32 = 0;
    for c in 0..SPRAY_CHANNELS {
        let exact = raw[c].max(0.0) / total * target;
        let f = exact.floor();
        floored[c] = f as u32;
        frac[c] = exact - f;
        acc += floored[c];
    }
    let mut leftover = (SPRAY_SUM as u32).saturating_sub(acc);
    while leftover > 0 {
        // Largest fractional remainder wins; deterministic index tie-break.
        let mut best = 0usize;
        for c in 1..SPRAY_CHANNELS {
            if frac[c] > frac[best] {
                best = c;
            }
        }
        floored[best] += 1;
        frac[best] = -1.0; // consumed
        leftover -= 1;
    }
    let mut out = [0u8; SPRAY_CHANNELS];
    for c in 0..SPRAY_CHANNELS {
        out[c] = floored[c].min(255) as u8;
    }
    out
}

/// GLSL-style smoothstep (edge0 may be > edge1 for a descending ramp).
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(cell: &[u8; SPRAY_CHANNELS]) -> u32 {
        cell.iter().map(|&w| w as u32).sum()
    }

    #[test]
    fn new_field_is_all_channel0_sum255() {
        let f = SprayField::new(16, 16, [0.0, 0.0], 1.5);
        for cell in f.weights() {
            assert_eq!(cell[0], 255);
            assert_eq!(sum(cell), 255);
        }
        assert_eq!(f.res_x(), 16);
        assert_eq!(f.cell_size(), 1.5);
    }

    #[test]
    fn paint_raises_channel_and_keeps_sum_255() {
        let mut f = SprayField::new(64, 64, [0.0, 0.0], 1.0);
        // Seed a 2-channel baseline so renormalization has something to take from.
        f.seed_from(|_x, _z| {
            let mut w = [0.0; SPRAY_CHANNELS];
            w[0] = 0.5;
            w[1] = 0.5;
            w
        });
        let changed = f.paint_texture(3, [32.0, 32.0], 8.0, 1.0, 1.0);
        assert!(changed > 0, "a stroke at the field centre must change cells");
        // Every cell still sums to exactly 255.
        for cell in f.weights() {
            assert_eq!(sum(cell), 255, "renormalize must hold Σ=255: {cell:?}");
        }
        // The painted centre is now dominated by channel 3.
        let (dom, w) = f.dominant_at(32.0, 32.0);
        assert_eq!(dom, 3, "centre must be channel 3 after a full-strength stroke");
        assert!(w > 0.6, "centre weight must be strong, got {w}");
        // A cell far outside the radius is untouched (still the 0.5/0.5 baseline).
        let (_d, _w) = f.dominant_at(2.0, 2.0);
        let far = f.sample_bilinear(2.0, 2.0);
        assert!(far[3] < 0.05, "outside the brush channel 3 stays ~0: {far:?}");
    }

    #[test]
    fn falloff_is_strongest_at_center() {
        let mut f = SprayField::new(128, 128, [0.0, 0.0], 1.0);
        f.seed_from(|_x, _z| {
            let mut w = [0.0; SPRAY_CHANNELS];
            w[0] = 1.0;
            w
        });
        f.paint_texture(2, [64.0, 64.0], 20.0, 1.0, 1.0);
        let center = f.sample_bilinear(64.0, 64.0)[2];
        let mid = f.sample_bilinear(74.0, 64.0)[2];
        let rim = f.sample_bilinear(83.0, 64.0)[2];
        assert!(
            center > mid && mid > rim,
            "falloff must decrease center→rim: {center} {mid} {rim}"
        );
    }

    #[test]
    fn replay_log_reproduces_field_bit_exact() {
        let seed = |_x: f32, _z: f32| {
            let mut w = [0.0; SPRAY_CHANNELS];
            w[0] = 0.7;
            w[1] = 0.3;
            w
        };
        let mut a = SprayField::new(64, 64, [10.0, -5.0], 1.25);
        a.seed_from(seed);
        a.paint_texture(2, [40.0, 20.0], 6.0, 0.8, 0.9);
        a.paint_texture(4, [44.0, 24.0], 4.0, 1.0, 0.5);
        a.paint_texture(2, [20.0, 50.0], 10.0, 0.6, 1.0);

        // Rebuild a fresh field from the SAME seed + the SAME stroke log.
        let mut b = SprayField::new(64, 64, [10.0, -5.0], 1.25);
        b.seed_from(seed);
        for s in a.strokes() {
            b.apply_stroke(s);
        }
        assert_eq!(
            a.weights(),
            b.weights(),
            "replaying the stroke log must reproduce the field bit-exact"
        );
    }

    #[test]
    fn seed_reseed_keeps_strokes() {
        // Re-seeding (e.g. a baseline change) must keep the strokes applied on top.
        let mut f = SprayField::new(48, 48, [0.0, 0.0], 1.0);
        f.seed_from(|_x, _z| {
            let mut w = [0.0; SPRAY_CHANNELS];
            w[0] = 1.0;
            w
        });
        f.paint_texture(5, [24.0, 24.0], 8.0, 1.0, 1.0);
        let (dom_before, _) = f.dominant_at(24.0, 24.0);
        assert_eq!(dom_before, 5);
        // Re-seed with a different baseline; the stroke must still show.
        f.seed_from(|_x, _z| {
            let mut w = [0.0; SPRAY_CHANNELS];
            w[1] = 1.0;
            w
        });
        let (dom_after, _) = f.dominant_at(24.0, 24.0);
        assert_eq!(dom_after, 5, "stroke survives a re-seed (delta-over-baseline)");
        // And a cell with no stroke shows the new baseline (channel 1).
        let (dom_edge, _) = f.dominant_at(2.0, 2.0);
        assert_eq!(dom_edge, 1);
    }

    #[test]
    fn pack_u32_roundtrips_the_bytes() {
        let mut f = SprayField::new(4, 4, [0.0, 0.0], 1.0);
        f.seed_from(|_x, _z| {
            let mut w = [0.0; SPRAY_CHANNELS];
            w[0] = 0.25;
            w[3] = 0.25;
            w[7] = 0.5;
            w
        });
        let packed = f.pack_u32();
        assert_eq!(packed.len(), 4 * 4 * SPRAY_WORDS_PER_CELL, "SPRAY_WORDS_PER_CELL u32 per cell");
        // Unpack cell 0 and confirm it matches the stored bytes.
        let cell0 = f.weights()[0];
        let w0 = packed[0];
        let w1 = packed[1];
        assert_eq!((w0 & 0xff) as u8, cell0[0]);
        assert_eq!(((w0 >> 24) & 0xff) as u8, cell0[3]);
        assert_eq!(((w1 >> 24) & 0xff) as u8, cell0[7]);
    }

    #[test]
    fn quantize_sums_to_exactly_255() {
        // A nasty ratio that doesn't divide evenly must still land on Σ=255 (12 channels).
        let raw = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let q = quantize_to_sum(&raw);
        assert_eq!(sum(&q), 255);
        let raw2 = [0.1, 0.2, 0.3, 0.05, 0.15, 0.07, 0.08, 0.05, 0.03, 0.04, 0.02, 0.01];
        assert_eq!(sum(&quantize_to_sum(&raw2)), 255);
    }
}
