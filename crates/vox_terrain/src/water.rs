//! Phase-1 deterministic water MODEL — first slice (the L1 truth layer).
//!
//! This is the **framework + a minimal model**, NOT the full flood/flow/service
//! sim. It implements the smallest increment of the substrate-native water
//! design (`urban_horizon/docs/superpowers/specs/2026-06-27-water-system-substrate-design.md`,
//! §4.1 L1) that demonstrates *real, replay-exact* water behavior:
//!
//!   * a per-cell **water-depth field** carried as a DELTA over the immutable
//!     heightmap baseline (the bed = the replayable terraform artifact; depth =
//!     a derived overlay), `f64`, row-major, the cell index IS the id;
//!   * a deterministic **per-tick basin-fill step** toward a `sea_level` plane,
//!     connectivity-gated (a basin reachable from a sea source fills; a sealed
//!     interior pit does not), **double-buffered** so the result is independent
//!     of the `0..n` traversal order;
//!   * a fold of the field into the SAME `ReplayHasher` integer-ledger moat the
//!     cim sim uses (`vox_sim::cim_sim::report::ReplayHasher`), id/index-ordered,
//!     fixed-ε `f64` quantize → no HashMap / no RNG / no wall-clock;
//!   * a **terraform→water coupling** seam: a dirty-cell set that the next step
//!     consumes — a carve (bed lowered below the level) fills, a raise (bed
//!     lifted above the level) drains instantly at the edited cells.
//!
//! ## What is SCAFFOLD vs FULL (be honest)
//!
//! * FULL (built + tested here, locally, no CUDA): the data model, the
//!   deterministic basin-fill step, the connectivity gate, the `ReplayHasher`
//!   fold, the terraform dirty-set coupling, and the determinism + behavior
//!   tests below.
//! * SCAFFOLD / NEXT SLICE (NOT in this file): the Saint-Venant velocity/flux
//!   advection (`forge/crates/terrain/src/swe.rs:207-307`, with toroidal `roll`
//!   replaced by no-flux edges) for true flow; per-`WaterBody` sources +
//!   rainfall accumulation; the priority-flood (Barnes 2014) exact basin solve;
//!   and the game-side wiring into `CitySim::step()` / `GameState::tick()` /
//!   `sim_hash::sim_state_hash` / `apply_terraform_action` (hook points are
//!   documented in the design doc §7 and this slice's return notes).
//!
//! ## Game-agnostic rule
//!
//! This module lives in the engine crate (`vox_terrain`) and knows only a bed
//! height grid + a scalar `sea_level` plane. It does NOT know "ocean" / "lake" /
//! "WaterBody" semantics — those stay in the game layer and are passed in as a
//! plain `f32`/`f64` (mirrors `DEFAULT_SEA_LEVEL` → forge). The renderer is
//! explicitly OUT of scope; see [`WaterField::surface_height`] for the seam
//! where the field would feed the water mesh / SDF level set.

use vox_sim::cim_sim::report::ReplayHasher;

/// Fixed-ε quantization grid (1e-6 m). Every `f64` that enters the replay hash
/// is snapped to this grid first so equal inputs always produce equal bits,
/// independent of accumulation order. Mirrors `wellbeing::q` and
/// `game/economy.rs` (`(x * 1_000_000.0).round() as i64`).
const QUANT: f64 = 1_000_000.0;

/// Per-tick fraction a fed cell moves toward its equilibrium depth. A fixed
/// constant (config-first DEBT to migrate into `render.ron`/`sim.ron` later) so
/// the relaxation is bit-stable. `0 < FILL_RATE < 1` keeps every move strictly
/// between the current value and the target, so depth is naturally bounded.
const FILL_RATE: f64 = 0.34;

/// When a cell is within `SNAP_EPS` of its equilibrium it is snapped exactly, so
/// the basin reaches `sea_level` in a finite number of ticks (no asymptotic
/// tail) and the equilibrium depth is a bit-exact value a test can assert.
const SNAP_EPS: f64 = 1.0e-6;

/// Fixed-ε quantize → integer, for the replay-hash fold. `q(x) = round(x * 1e6)`.
#[inline]
fn q(x: f64) -> i64 {
    (x * QUANT).round() as i64
}

/// L1: per-cell hydrology state on the heightfield grid — the deterministic
/// TRUTH, folded into `ReplayHasher` in id (cell-index) order with fixed-ε f64.
///
/// `depth` is a DELTA over the immutable bed: `depth[i] == 0` is dry; the water
/// surface at a cell is `bed[i] + depth[i]` (see [`surface_height`]). The field
/// is a deterministic function of (bed, sea_level, tick count, terraform log),
/// so it is never the saved truth — it is re-simulated from the command log
/// (the action-log replay moat).
///
/// [`surface_height`]: WaterField::surface_height
#[derive(Debug, Clone)]
pub struct WaterField {
    width: u32,
    height: u32,
    /// Per-cell water column above the bed (m). Row-major `z * width + x`. The
    /// FRONT buffer (current tick's committed state).
    depth: Vec<f64>,
    /// Double-buffer scratch the step writes into, then swaps. Holding it on the
    /// struct avoids a per-tick allocation and keeps the step order-independent
    /// (it reads only `depth`, writes only `next`).
    next: Vec<f64>,
    /// Terraform dirty cells (row-major indices) the NEXT step consumes. A carve
    /// or raise pushes the edited AABB here; the step applies an instant cap at
    /// these cells (raise → drain) then clears the set.
    dirty: Vec<u32>,
    /// Bumped every committed step + on structural change — a cheap "did the
    /// field move" signal for the (out-of-scope) render double-buffer.
    rev: u64,
}

impl WaterField {
    /// A dry field co-sized with the heightfield grid (`width * height` cells).
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            width,
            height,
            depth: vec![0.0; n],
            next: vec![0.0; n],
            dirty: Vec::new(),
            rev: 0,
        }
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    pub fn rev(&self) -> u64 {
        self.rev
    }

    /// Number of cells.
    #[inline]
    pub fn len(&self) -> usize {
        self.depth.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.depth.is_empty()
    }

    #[inline]
    fn idx(&self, x: u32, z: u32) -> usize {
        (z * self.width + x) as usize
    }

    /// Water depth (m) above the bed at a cell. `0.0` = dry.
    #[inline]
    pub fn depth_at(&self, x: u32, z: u32) -> f64 {
        self.depth[self.idx(x, z)]
    }

    /// Read access to the per-cell depth grid (row-major, `z * width + x`). The
    /// cell index IS the id. Used by the game-side replay-hash fold + inspection
    /// (`urban_horizon::sim::sim_hash`), which folds it len-prefixed + in strict
    /// ascending index order — the same id-ordered contract as [`fold_into`].
    ///
    /// [`fold_into`]: WaterField::fold_into
    #[inline]
    pub fn depths(&self) -> &[f64] {
        &self.depth
    }

    /// RENDER SEAM (out of scope for the model): the world-Y of the water
    /// surface at a cell = bed elevation + water column. This is what a later
    /// phase samples to drive the water mesh height / the `d_water(p) =
    /// surface(x,z) - p.y` SDF level set (design §4.2). The model layer computes
    /// it; no renderer is touched here.
    #[inline]
    pub fn surface_height(&self, x: u32, z: u32, bed_height: f64) -> f64 {
        bed_height + self.depth_at(x, z)
    }

    /// Total water volume (m³) = Σ depth · cell_area, folded in index order.
    /// A real computed witness for the behavior test (carve → volume grows).
    pub fn total_volume(&self, cell_area: f64) -> f64 {
        let mut v = 0.0;
        for &d in &self.depth {
            v += d * cell_area;
        }
        v
    }

    /// TERRAFORM → WATER coupling seam. Mark the edited AABB (inclusive,
    /// clamped) dirty so the next [`step`](WaterField::step) re-couples it.
    /// Mirrors the existing terraform dirty signal `delta.aabb_cells()` /
    /// `water_flips` (`heightfield.rs:340-351`). Carve and raise both route here;
    /// the step decides fill vs drain from the new bed.
    pub fn mark_terraform(&mut self, x0: u32, z0: u32, x1: u32, z1: u32) {
        let x0 = x0.min(self.width.saturating_sub(1));
        let x1 = x1.min(self.width.saturating_sub(1));
        let z0 = z0.min(self.height.saturating_sub(1));
        let z1 = z1.min(self.height.saturating_sub(1));
        for z in z0..=z1 {
            for x in x0..=x1 {
                self.dirty.push(z * self.width + x);
            }
        }
    }

    /// Advance the water field ONE tick. Deterministic, id-ordered, fixed-ε f64,
    /// no HashMap / RNG / wall-clock.
    ///
    /// `bed` is the current heightmap (immutable baseline + terraform delta),
    /// row-major, `len == self.len()`. `sea_level` is the game-supplied source
    /// plane (the engine stays ignorant of "ocean"/"lake").
    ///
    /// Model: a connectivity-gated relaxation toward the basin equilibrium.
    ///   * A cell's equilibrium depth is `cap = max(0, sea_level - bed)`.
    ///   * A cell is *fed* iff `bed < sea_level` AND it is a sea source (a
    ///     sub-sea cell on the grid border) OR a 4-neighbor held water on the
    ///     PREVIOUS tick. Feeding spreads one ring per tick, so a basin
    ///     connected to the border fills over N ticks while a SEALED interior
    ///     pit (no border path) never wets → stays dry.
    ///   * Fed cells move `FILL_RATE` of the way toward `cap`; unfed/over-cap
    ///     cells move toward `0` (drain). Within `SNAP_EPS` of the target the
    ///     value snaps exactly, so equilibrium is bit-exact and finite-time.
    ///
    /// Order-independence: the step reads only the FRONT `depth` buffer and the
    /// `bed`, and writes only `next`, then swaps — so iterating `0..n` forward,
    /// backward, or in any order yields the identical result (no in-place
    /// neighbor coupling). This is the deterministic-grid-pass discipline.
    ///
    /// No-op (never panics) on an empty grid or a `bed` length mismatch.
    pub fn step(&mut self, bed: &[f64], sea_level: f64) {
        let n = self.depth.len();
        if n == 0 || bed.len() != n {
            return;
        }

        // (1) Consume the terraform dirty set: a raise that lifted the bed above
        // the local water surface drains INSTANTLY at the edited cells (clamp to
        // the new cap) rather than waiting for the relaxation. Carve lowers the
        // bed → cap grows → the relaxation below fills it. id-ordered, the set is
        // a Vec (no HashMap), so this is replay-stable.
        for &i in &self.dirty {
            let i = i as usize;
            let cap = (sea_level - bed[i]).max(0.0);
            if self.depth[i] > cap {
                self.depth[i] = cap;
            }
        }
        self.dirty.clear();

        let w = self.width as usize;
        let h = self.height as usize;

        // (2) Double-buffered relaxation. Read `depth` (prev tick), write `next`.
        for z in 0..h {
            for x in 0..w {
                let i = z * w + x;
                let bed_i = bed[i];
                let cap = (sea_level - bed_i).max(0.0);
                let cur = self.depth[i];

                // Connectivity (read from the PREVIOUS depth buffer only).
                let sub_sea = bed_i < sea_level;
                let border = x == 0 || z == 0 || x + 1 == w || z + 1 == h;
                let mut neighbor_wet = false;
                if x > 0 {
                    neighbor_wet |= self.depth[i - 1] > 0.0;
                }
                if x + 1 < w {
                    neighbor_wet |= self.depth[i + 1] > 0.0;
                }
                if z > 0 {
                    neighbor_wet |= self.depth[i - w] > 0.0;
                }
                if z + 1 < h {
                    neighbor_wet |= self.depth[i + w] > 0.0;
                }
                let fed = sub_sea && (border || neighbor_wet);

                let target = if fed { cap } else { 0.0 };
                let mut nv = cur + (target - cur) * FILL_RATE;
                if (nv - target).abs() < SNAP_EPS {
                    nv = target;
                }
                self.next[i] = nv;
            }
        }

        std::mem::swap(&mut self.depth, &mut self.next);
        self.rev += 1;
    }

    /// Fold the water field into the SAME integer-ledger replay moat the cim sim
    /// uses (`ReplayHasher`, `vox_sim::cim_sim::report`). The fold is:
    ///   * **id/index-ordered** — the row-major cell index IS the id; we iterate
    ///     `0..len` over a `Vec`, never a HashMap and never RNG order;
    ///   * **fixed-ε quantized** — every depth is snapped to the 1e-6 grid via
    ///     [`q`] before hashing, so order-of-accumulation float noise can never
    ///     leak into the hash;
    ///   * **identity-carrying** — the packed key is `(i << 40) ^ (q(depth) &
    ///     mask)` so distinct cells never alias (mirrors `wellbeing::mod.rs:245`,
    ///     `tick.rs:298`).
    ///
    /// A `len` guard is mixed first (so a resize is itself a hash change). After
    /// this fold runs in the live tick, the replay-hash witness changes ONCE
    /// (new state folded) then stays bit-stable across replays.
    pub fn fold_into(&self, hasher: &mut ReplayHasher) {
        hasher.mix(self.depth.len() as u64);
        for (i, &d) in self.depth.iter().enumerate() {
            let qd = (q(d) as u64) & 0xFF_FFFF_FFFF; // low 40 bits
            let packed = ((i as u64) << 40) ^ qd; // index in the high bits → no aliasing
            hasher.mix(packed);
        }
    }

    /// Convenience: the field's standalone replay hash (a fresh `ReplayHasher`
    /// folded with just this field). Used by tests + as the per-field witness.
    pub fn replay_hash(&self) -> u64 {
        let mut h = ReplayHasher::new();
        self.fold_into(&mut h);
        h.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a square "bowl" bed: a flat floor at `floor_y` ringed by a high
    /// wall at `wall_y`, with the OUTER border at `floor_y` so the basin is
    /// connected to a sea source on the edge. Row-major.
    fn bowl(size: usize, floor_y: f64, wall_y: f64) -> Vec<f64> {
        let mut bed = vec![floor_y; size * size];
        // Build an interior ring wall one cell in from the border, leaving a gap
        // on the -x edge so the interior basin is reachable from the border sea.
        for z in 1..size - 1 {
            for x in 1..size - 1 {
                let on_ring = x == 1 || z == 1 || x == size - 2 || z == size - 2;
                let gap = x == 1 && z == size / 2; // breach in the wall on -x side
                if on_ring && !gap {
                    bed[z * size + x] = wall_y;
                }
            }
        }
        bed
    }

    /// The deterministic step + fold reproduce bit-for-bit across two
    /// independent runs, AND the fold is index-ordered (not container-order
    /// dependent). Forbidden stub avoided: NOT `assert!(hash != 0)` /
    /// `assert!(depth.iter().any(|c| *c > 0.0))` — we assert bit-EQUALITY of two
    /// independent runs' hashes AND their full depth buffers.
    #[test]
    fn determinism_two_runs_bit_identical() {
        let size = 16;
        let sea_level = 4.0;
        let bed = bowl(size, 0.0, 10.0);

        let run = || {
            let mut w = WaterField::new(size as u32, size as u32);
            // A terraform carve marks a dirty AABB before the sim runs.
            w.mark_terraform(2, 2, 6, 6);
            let mut hashes = Vec::new();
            for _ in 0..40 {
                w.step(&bed, sea_level);
                hashes.push(w.replay_hash());
            }
            (w, hashes)
        };

        let (wa, ha) = run();
        let (wb, hb) = run();

        // (a) per-tick replay hashes identical across the two runs.
        assert_eq!(ha, hb, "replay-hash trajectory diverged between two runs");

        // (b) final depth buffer bit-identical (to_bits — catches one-ULP drift).
        let bits = |w: &WaterField| -> Vec<u64> { w.depth.iter().map(|d| d.to_bits()).collect() };
        assert_eq!(bits(&wa), bits(&wb), "final depth grid diverged (ULP)");

        // (c) index-ordered fold contract: re-fold by EXPLICIT ascending index
        // and confirm it equals fold_into — proving the fold never depends on a
        // hashed/RNG container order. (ReplayHasher.mix is order-sensitive, so an
        // order change WOULD change the hash.)
        let mut reference = ReplayHasher::new();
        reference.mix(wa.depth.len() as u64);
        for i in 0..wa.depth.len() {
            let qd = (q(wa.depth[i]) as u64) & 0xFF_FFFF_FFFF;
            reference.mix(((i as u64) << 40) ^ qd);
        }
        assert_eq!(
            reference.finish(),
            wa.replay_hash(),
            "fold_into is not strict ascending-index order"
        );
    }

    /// A connected, sub-sea basin FILLS to exactly `sea_level` over N ticks
    /// (assert a REAL computed depth, bit-exact — not `is_some`/`> 0`). The bed
    /// floor sits below sea_level; the equilibrium depth is `sea_level - floor`.
    #[test]
    fn connected_basin_fills_to_sea_level() {
        let size = 16;
        let sea_level = 4.0;
        let floor_y = 0.0;
        let bed = bowl(size, floor_y, 10.0); // wall above sea_level

        let mut w = WaterField::new(size as u32, size as u32);

        // Interior cell well inside the bowl (past the breach), starts dry.
        let (cx, cz) = (size / 2, size / 2);
        assert_eq!(w.depth_at(cx as u32, cz as u32), 0.0, "should start dry");

        let v0 = w.total_volume(1.0);

        // 64 ticks is more than enough for the ~12-tick fill + ring spread.
        for _ in 0..64 {
            w.step(&bed, sea_level);
        }

        let expected = sea_level - floor_y; // 4.0 m column, bit-exact after SNAP
        let got = w.depth_at(cx as u32, cz as u32);
        assert_eq!(
            got, expected,
            "connected basin cell did not fill to sea_level (got {got}, want {expected})"
        );

        // Volume strictly increased (a real computed witness, not is_some).
        let v1 = w.total_volume(1.0);
        assert!(v1 > v0 + 100.0, "water volume did not grow (v0={v0}, v1={v1})");
    }

    /// A SEALED interior pit (walled off from the border sea, no breach to it)
    /// does NOT fill — connectivity is real, not a global `bed < sea_level`
    /// fill. This is the behavior that distinguishes a model from a flat plane.
    #[test]
    fn sealed_pit_stays_dry() {
        let size = 16;
        let sea_level = 4.0;
        // Whole map below sea level EXCEPT a closed wall ring around one interior
        // cell, AND the outer border raised above sea level so there is NO sea
        // source feeding the interior. Nothing is fed → nothing fills.
        let mut bed = vec![0.0; size * size];
        // Raise the entire outer border above sea level (no border sea source).
        for z in 0..size {
            for x in 0..size {
                if x == 0 || z == 0 || x == size - 1 || z == size - 1 {
                    bed[z * size + x] = 10.0;
                }
            }
        }
        let mut w = WaterField::new(size as u32, size as u32);
        for _ in 0..64 {
            w.step(&bed, sea_level);
        }
        // No border sea source ⇒ the interior never gets fed ⇒ bit-exact dry.
        let total = w.total_volume(1.0);
        assert_eq!(total, 0.0, "sealed map with no sea source should stay dry");
    }

    /// Terraform coupling: a RAISE (bed lifted above the local water surface) at
    /// dirty cells DRAINS them; the field re-couples deterministically.
    #[test]
    fn raise_drains_filled_cells() {
        let size = 16;
        let sea_level = 4.0;
        let mut bed = bowl(size, 0.0, 10.0);
        let mut w = WaterField::new(size as u32, size as u32);

        // Fill first.
        for _ in 0..64 {
            w.step(&bed, sea_level);
        }
        let (cx, cz) = (size / 2, size / 2);
        let i = cz * size + cx;
        assert!(w.depth_at(cx as u32, cz as u32) > 0.0, "precondition: filled");

        // RAISE the cell's bed above sea_level (a dam/landfill) + mark dirty.
        bed[i] = 6.0; // above sea_level 4.0
        w.mark_terraform(cx as u32, cz as u32, cx as u32, cz as u32);

        // One step consumes the dirty set → instant cap → drains to exactly 0.
        w.step(&bed, sea_level);
        assert_eq!(
            w.depth_at(cx as u32, cz as u32),
            0.0,
            "raised cell did not drain"
        );
    }

    /// Empty / mismatched-bed steps are a no-op (never panic).
    #[test]
    fn empty_and_mismatch_are_noop() {
        let mut w = WaterField::new(0, 0);
        w.step(&[], 1.0); // empty grid
        assert_eq!(w.len(), 0);

        let mut w2 = WaterField::new(4, 4);
        w2.step(&[0.0; 3], 1.0); // wrong bed length → no-op, no panic
        assert_eq!(w2.replay_hash(), WaterField::new(4, 4).replay_hash());
    }
}
