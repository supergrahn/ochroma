//! Spectral material/shader node graph that COMPILES to the spectral pipeline.
//!
//! Rank #5 adoption candidate: a real material graph (Unity Shader Graph +
//! UE Substrate style) that emits executable WGSL for the 16-band spectral
//! path — not just a CPU-evaluated tree.
//!
//! # The contract
//!
//! A [`MaterialDag`] is an *indexed* DAG: nodes reference their inputs by
//! [`NodeId`] (a `usize` index into the node arena). This is what lets us
//! reject cycles and dangling references at compile time — something a
//! `Box`-tree (like the legacy `material_graph::MaterialNode`) cannot express.
//!
//! Two backends share one set of semantics:
//!
//! * [`MaterialDag::eval_cpu`] — the reference evaluator, evaluates one band at
//!   a time given `(band, lambda_nm, cos_theta)`.
//! * [`MaterialDag::compile_to_wgsl`] — emits a WGSL function
//!   `fn material_eval(band: u32, lambda_nm: f32, cos_theta: f32) -> f32`, as
//!   straight-line SSA code in topological order, constant-folded where trivial.
//!
//! The equivalence contract: for any valid graph, `eval_cpu(band, λ, cosθ)` and
//! the compiled `material_eval(band, λ, cosθ)` agree per band within f32 epsilon.
//! The GPU harness ([`GpuMaterialEval`]) proves this on real hardware.
//!
//! These per-band 16 outputs ARE the spectral reflectance/emission an
//! `[f32; 16]` splat carries (`vox_core::spectral`, 380–755 nm), so a graph can
//! express (and ultimately replace) the static materials in `vox_data`.

use vox_core::spectral::BAND_WAVELENGTHS;

/// Index into a [`MaterialDag`]'s node arena.
pub type NodeId = usize;

/// Number of spectral bands in the engine's light currency.
pub const N_BANDS: usize = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Node set
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the spectral material DAG. Inputs are referenced by [`NodeId`].
///
/// Every node is a pure function of `(band, lambda_nm, cos_theta)` plus the
/// values of its referenced inputs — this is what makes per-band straight-line
/// codegen possible.
#[derive(Debug, Clone, PartialEq)]
pub enum BsdfNode {
    // ── Spectral / band inputs ───────────────────────────────────────────────
    /// Per-band constant reflectance/emission (the 16-band light currency).
    SpectralConstant { spd: [f32; N_BANDS] },
    /// Smits-style RGB→16-band upliftment, evaluated self-contained (no
    /// vox_data dependency). Produces a smooth reflectance that integrates back
    /// to roughly the source RGB.
    RgbUplift { rgb: [f32; 3] },
    /// Planck blackbody emitter at `kelvin`, normalised so its peak band == 1.0.
    BlackbodyEmitter { kelvin: f32 },
    /// The current band's centre wavelength (nm) as a scalar — enables
    /// iridescence / band-dependent math.
    Wavelength,
    /// `cos_theta` (the view/normal cosine) as a scalar, for angle-dependent math.
    CosTheta,
    /// A plain scalar constant broadcast to every band.
    Scalar { value: f32 },

    // ── Arithmetic ───────────────────────────────────────────────────────────
    Add { a: NodeId, b: NodeId },
    Multiply { a: NodeId, b: NodeId },
    /// Linear interpolation `a*(1-factor) + b*factor`.
    Mix { a: NodeId, b: NodeId, factor: f32 },
    /// Scale by a constant.
    Scale { input: NodeId, factor: f32 },
    /// `1.0 - input`, clamped to keep reflectance physical.
    Invert { input: NodeId },
    /// Clamp to `[min, max]`.
    Clamp { input: NodeId, min: f32, max: f32 },

    // ── BSDF / Substrate ─────────────────────────────────────────────────────
    /// Schlick Fresnel reflectance from a base reflectance at normal incidence:
    /// `base + (1 - base) * (1 - cos_theta)^power`.
    Fresnel { base: NodeId, power: f32 },
    /// Substrate-style layering: a coat over a base, blended per band by a
    /// Schlick Fresnel weight (`f0` at normal incidence, → 1 at grazing).
    /// `result = base*(1-w) + coat*w`, `w = f0 + (1-f0)*(1-cos_theta)^5`.
    Layer { coat: NodeId, base: NodeId, f0: f32 },
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph + errors
// ─────────────────────────────────────────────────────────────────────────────

/// An indexed DAG of [`BsdfNode`]s with a designated output node.
#[derive(Debug, Clone)]
pub struct MaterialDag {
    pub nodes: Vec<BsdfNode>,
    pub output: NodeId,
}

/// Errors produced by validation / compilation.
#[derive(Debug, Clone, PartialEq)]
pub enum MaterialCompileError {
    /// A node referenced an index that does not exist.
    UnknownRef { node: NodeId, referenced: NodeId },
    /// The output index is out of range.
    UnknownOutput { output: NodeId, len: usize },
    /// The graph contains a cycle reachable from the output.
    Cycle { node: NodeId },
    /// The graph has no nodes.
    Empty,
}

impl std::fmt::Display for MaterialCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownRef { node, referenced } => {
                write!(f, "node {node} references unknown node {referenced}")
            }
            Self::UnknownOutput { output, len } => {
                write!(f, "output node {output} out of range (len {len})")
            }
            Self::Cycle { node } => write!(f, "cycle detected at node {node}"),
            Self::Empty => write!(f, "graph is empty"),
        }
    }
}

impl std::error::Error for MaterialCompileError {}

impl BsdfNode {
    /// The input node ids this node references (for traversal / validation).
    fn refs(&self) -> Vec<NodeId> {
        match self {
            Self::SpectralConstant { .. }
            | Self::RgbUplift { .. }
            | Self::BlackbodyEmitter { .. }
            | Self::Wavelength
            | Self::CosTheta
            | Self::Scalar { .. } => Vec::new(),
            Self::Add { a, b } | Self::Multiply { a, b } => vec![*a, *b],
            Self::Mix { a, b, .. } => vec![*a, *b],
            Self::Scale { input, .. }
            | Self::Invert { input }
            | Self::Clamp { input, .. }
            | Self::Fresnel { base: input, .. } => vec![*input],
            Self::Layer { coat, base, .. } => vec![*coat, *base],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared numeric helpers (used identically by CPU eval and as WGSL emission)
// ─────────────────────────────────────────────────────────────────────────────

/// Planck spectral radiance (relative) at wavelength `lambda_nm` for temp `t` K.
///
/// Computed in **f32** to bit-match the WGSL `planck_relative_f` helper (the GPU
/// has no f64), so the CPU reference and the compiled shader agree within the
/// equivalence test's f32 epsilon. Returns the raw Planck value; band
/// normalisation is applied by the caller.
fn planck_relative(lambda_nm: f32, kelvin: f32) -> f32 {
    // Planck's law (radiance form). Constants in SI; wavelength in metres.
    // B(λ,T) = (2hc² / λ⁵) * 1/(exp(hc/(λ k T)) - 1)
    const H: f32 = 6.626_07e-34; // Planck
    const C: f32 = 2.997_924_5e8; // speed of light
    const KB: f32 = 1.380_649e-23; // Boltzmann
    let lambda_m = lambda_nm * 1e-9;
    let t = kelvin.max(1.0);
    let a = 2.0 * H * C * C / lambda_m.powf(5.0);
    let x = H * C / (lambda_m * KB * t);
    let denom = x.exp() - 1.0;
    if denom <= 0.0 {
        0.0
    } else {
        a / denom
    }
}

/// Peak Planck value across the 16 bands, for normalisation.
fn planck_peak(kelvin: f32) -> f32 {
    let mut peak = 0.0f32;
    for &lambda in BAND_WAVELENGTHS.iter() {
        peak = peak.max(planck_relative(lambda, kelvin));
    }
    peak.max(1e-30)
}

/// Self-contained Smits-style RGB→16-band reflectance upliftment.
///
/// Builds a smooth reflectance from white / primary / secondary basis curves
/// (Smits 1999), choosing the decomposition with the maximal component first.
/// All-positive, bounded in `[0, 1]`, and metameric to the source RGB. Fully
/// self-contained — no vox_data dependency, no cycle risk.
fn smits_uplift_band(rgb: [f32; 3], band: usize) -> f32 {
    // Smits basis functions resampled to the engine's 16-band grid (380–755 nm).
    // White, then cyan/magenta/yellow (secondaries), then red/green/blue.
    const WHITE: [f32; N_BANDS] = [1.0; N_BANDS];
    // Blue rises at short wavelengths, decays toward red.
    const BLUE: [f32; N_BANDS] = [
        1.00, 1.00, 1.00, 0.96, 0.80, 0.52, 0.28, 0.13, 0.05, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00,
    ];
    const GREEN: [f32; N_BANDS] = [
        0.00, 0.00, 0.06, 0.20, 0.45, 0.78, 0.96, 1.00, 0.92, 0.62, 0.28, 0.10, 0.03, 0.01, 0.00,
        0.00,
    ];
    const RED: [f32; N_BANDS] = [
        0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.05, 0.14, 0.36, 0.68, 0.92, 1.00, 1.00, 1.00, 1.00,
        1.00,
    ];
    const CYAN: [f32; N_BANDS] = [
        1.00, 1.00, 1.00, 1.00, 0.96, 0.92, 0.88, 0.78, 0.52, 0.24, 0.08, 0.02, 0.00, 0.00, 0.00,
        0.00,
    ];
    const MAGENTA: [f32; N_BANDS] = [
        1.00, 1.00, 0.96, 0.80, 0.52, 0.20, 0.06, 0.10, 0.34, 0.70, 0.94, 1.00, 1.00, 1.00, 1.00,
        1.00,
    ];
    const YELLOW: [f32; N_BANDS] = [
        0.00, 0.00, 0.02, 0.08, 0.24, 0.55, 0.82, 0.96, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
        1.00,
    ];

    let r = rgb[0].clamp(0.0, 1.0);
    let g = rgb[1].clamp(0.0, 1.0);
    let b = rgb[2].clamp(0.0, 1.0);
    let mut spd = 0.0f32;
    // Smits decomposition: subtract white floor, then add secondary, then primary.
    if r <= g && r <= b {
        spd += r * WHITE[band];
        if g <= b {
            spd += (g - r) * CYAN[band];
            spd += (b - g) * BLUE[band];
        } else {
            spd += (b - r) * CYAN[band];
            spd += (g - b) * GREEN[band];
        }
    } else if g <= r && g <= b {
        spd += g * WHITE[band];
        if r <= b {
            spd += (r - g) * MAGENTA[band];
            spd += (b - r) * BLUE[band];
        } else {
            spd += (b - g) * MAGENTA[band];
            spd += (r - b) * RED[band];
        }
    } else {
        spd += b * WHITE[band];
        if r <= g {
            spd += (r - b) * YELLOW[band];
            spd += (g - r) * GREEN[band];
        } else {
            spd += (g - b) * YELLOW[band];
            spd += (r - g) * RED[band];
        }
    }
    spd.clamp(0.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference evaluator
// ─────────────────────────────────────────────────────────────────────────────

impl MaterialDag {
    /// Validate structure: non-empty, all refs in range, output in range, acyclic.
    pub fn validate(&self) -> Result<(), MaterialCompileError> {
        if self.nodes.is_empty() {
            return Err(MaterialCompileError::Empty);
        }
        if self.output >= self.nodes.len() {
            return Err(MaterialCompileError::UnknownOutput {
                output: self.output,
                len: self.nodes.len(),
            });
        }
        for (i, node) in self.nodes.iter().enumerate() {
            for r in node.refs() {
                if r >= self.nodes.len() {
                    return Err(MaterialCompileError::UnknownRef {
                        node: i,
                        referenced: r,
                    });
                }
            }
        }
        // Cycle detection via DFS colouring from the output.
        #[derive(Clone, Copy, PartialEq)]
        enum Color {
            White,
            Grey,
            Black,
        }
        let mut color = vec![Color::White; self.nodes.len()];
        // Iterative DFS to avoid stack overflow on deep graphs.
        let mut stack: Vec<(NodeId, bool)> = vec![(self.output, false)];
        while let Some((id, processed)) = stack.pop() {
            if processed {
                color[id] = Color::Black;
                continue;
            }
            match color[id] {
                Color::Black => continue,
                Color::Grey => return Err(MaterialCompileError::Cycle { node: id }),
                Color::White => {}
            }
            color[id] = Color::Grey;
            stack.push((id, true));
            for r in self.nodes[id].refs() {
                match color[r] {
                    Color::Grey => return Err(MaterialCompileError::Cycle { node: r }),
                    Color::Black => {}
                    Color::White => stack.push((r, false)),
                }
            }
        }
        Ok(())
    }

    /// Reference per-band evaluation. Identical semantics to the compiled WGSL.
    ///
    /// Returns the scalar value of the output node for the given band, the
    /// band's centre wavelength, and the view/normal cosine.
    pub fn eval_cpu(&self, band: usize, lambda_nm: f32, cos_theta: f32) -> f32 {
        // Memoised recursion (the DAG may share subgraphs).
        let mut cache: Vec<Option<f32>> = vec![None; self.nodes.len()];
        self.eval_node(self.output, band, lambda_nm, cos_theta, &mut cache)
    }

    /// Evaluate the full 16-band output SPD at a given cosine (one band per slot).
    pub fn eval_cpu_spd(&self, cos_theta: f32) -> [f32; N_BANDS] {
        std::array::from_fn(|band| self.eval_cpu(band, BAND_WAVELENGTHS[band], cos_theta))
    }

    fn eval_node(
        &self,
        id: NodeId,
        band: usize,
        lambda_nm: f32,
        cos_theta: f32,
        cache: &mut [Option<f32>],
    ) -> f32 {
        if let Some(v) = cache[id] {
            return v;
        }
        let v = match &self.nodes[id] {
            BsdfNode::SpectralConstant { spd } => spd[band],
            BsdfNode::RgbUplift { rgb } => smits_uplift_band(*rgb, band),
            BsdfNode::BlackbodyEmitter { kelvin } => {
                planck_relative(lambda_nm, *kelvin) / planck_peak(*kelvin)
            }
            BsdfNode::Wavelength => lambda_nm,
            BsdfNode::CosTheta => cos_theta,
            BsdfNode::Scalar { value } => *value,
            BsdfNode::Add { a, b } => {
                self.eval_node(*a, band, lambda_nm, cos_theta, cache)
                    + self.eval_node(*b, band, lambda_nm, cos_theta, cache)
            }
            BsdfNode::Multiply { a, b } => {
                self.eval_node(*a, band, lambda_nm, cos_theta, cache)
                    * self.eval_node(*b, band, lambda_nm, cos_theta, cache)
            }
            BsdfNode::Mix { a, b, factor } => {
                let va = self.eval_node(*a, band, lambda_nm, cos_theta, cache);
                let vb = self.eval_node(*b, band, lambda_nm, cos_theta, cache);
                va * (1.0 - factor) + vb * factor
            }
            BsdfNode::Scale { input, factor } => {
                self.eval_node(*input, band, lambda_nm, cos_theta, cache) * factor
            }
            BsdfNode::Invert { input } => {
                (1.0 - self.eval_node(*input, band, lambda_nm, cos_theta, cache)).clamp(0.0, 1.0)
            }
            BsdfNode::Clamp { input, min, max } => {
                self.eval_node(*input, band, lambda_nm, cos_theta, cache)
                    .clamp(*min, *max)
            }
            BsdfNode::Fresnel { base, power } => {
                let b = self.eval_node(*base, band, lambda_nm, cos_theta, cache);
                b + (1.0 - b) * (1.0 - cos_theta).clamp(0.0, 1.0).powf(*power)
            }
            BsdfNode::Layer { coat, base, f0 } => {
                let c = self.eval_node(*coat, band, lambda_nm, cos_theta, cache);
                let bse = self.eval_node(*base, band, lambda_nm, cos_theta, cache);
                let w = f0 + (1.0 - f0) * (1.0 - cos_theta).clamp(0.0, 1.0).powf(5.0);
                bse * (1.0 - w) + c * w
            }
        };
        cache[id] = Some(v);
        v
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL compiler
// ─────────────────────────────────────────────────────────────────────────────

/// Format an f32 as a WGSL float literal that always parses (has a decimal/exp).
fn wgsl_f32(v: f32) -> String {
    if v.is_nan() {
        // WGSL has no NaN literal; emit a value that const-evaluates to NaN.
        return "(0.0 / 0.0)".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 {
            "(1.0 / 0.0)".to_string()
        } else {
            "(-1.0 / 0.0)".to_string()
        };
    }
    let s = format!("{v:?}"); // Debug for f32 always includes a decimal point.
    s
}

impl MaterialDag {
    /// Compile this graph to a WGSL module string containing
    /// `fn material_eval(band: u32, lambda_nm: f32, cos_theta: f32) -> f32`.
    ///
    /// Validation runs first (cycles / unknown refs / empty / bad output). The
    /// body is straight-line SSA in topological order; constant subgraphs are
    /// folded to a single literal.
    pub fn compile_to_wgsl(&self) -> Result<String, MaterialCompileError> {
        self.validate()?;

        // Topological order of nodes reachable from output (post-order DFS).
        let order = self.topo_order();

        // Constant-fold: a node is const if all its refs are const and it does
        // not depend on band / lambda / cos_theta.
        let n = self.nodes.len();
        let mut is_const = vec![false; n];
        let mut const_val = vec![0.0f32; n];
        for &id in &order {
            let (c, val) = self.fold_node(id, &is_const, &const_val);
            is_const[id] = c;
            const_val[id] = val;
        }

        // Smits basis + Planck need runtime helpers only when used.
        let mut uses_uplift = false;
        let mut uses_blackbody = false;
        for &id in &order {
            match &self.nodes[id] {
                BsdfNode::RgbUplift { .. } => uses_uplift = true,
                BsdfNode::BlackbodyEmitter { .. } => uses_blackbody = true,
                _ => {}
            }
        }

        // Module-level const arrays for SpectralConstant nodes (band-indexed at
        // runtime; a `let` array constructor cannot be dynamically indexed in
        // WGSL, so we hoist them to module scope).
        let mut consts = String::new();
        for &id in &order {
            if is_const[id] {
                continue;
            }
            if let BsdfNode::SpectralConstant { spd } = &self.nodes[id] {
                let entries: Vec<String> = spd.iter().map(|v| wgsl_f32(*v)).collect();
                consts.push_str(&format!(
                    "const SPD_{} = array<f32, 16>({});\n",
                    id,
                    entries.join(", ")
                ));
            }
        }

        let mut body = String::new();
        // expr name per node
        let var = |id: NodeId| format!("n{id}");
        for &id in &order {
            if is_const[id] {
                body.push_str(&format!(
                    "    let {} = {};\n",
                    var(id),
                    wgsl_f32(const_val[id])
                ));
                continue;
            }
            let rhs = self.emit_node_rhs(id, &var, &is_const, &const_val);
            body.push_str(&format!("    let {} = {};\n", var(id), rhs));
        }
        body.push_str(&format!("    return {};\n", var(self.output)));

        let mut out = String::new();
        out.push_str("// Auto-generated by vox_render::material_compiler. Do not edit.\n");
        out.push_str(&consts);
        if uses_uplift {
            out.push_str(&Self::wgsl_uplift_helper());
        }
        if uses_blackbody {
            out.push_str(&Self::wgsl_blackbody_helper());
        }
        out.push_str(
            "fn material_eval(band: u32, lambda_nm: f32, cos_theta: f32) -> f32 {\n",
        );
        out.push_str(&body);
        out.push_str("}\n");
        Ok(out)
    }

    /// Post-order topological sort of nodes reachable from `output`.
    fn topo_order(&self) -> Vec<NodeId> {
        let mut visited = vec![false; self.nodes.len()];
        let mut order = Vec::new();
        // Iterative post-order DFS.
        let mut stack: Vec<(NodeId, bool)> = vec![(self.output, false)];
        while let Some((id, processed)) = stack.pop() {
            if processed {
                if !visited[id] {
                    visited[id] = true;
                    order.push(id);
                }
                continue;
            }
            if visited[id] {
                continue;
            }
            stack.push((id, true));
            for r in self.nodes[id].refs() {
                if !visited[r] {
                    stack.push((r, false));
                }
            }
        }
        order
    }

    /// Returns `(is_const, value)` for a node given the const-status of its refs.
    fn fold_node(&self, id: NodeId, is_const: &[bool], const_val: &[f32]) -> (bool, f32) {
        let cv = |r: NodeId| const_val[r];
        match &self.nodes[id] {
            BsdfNode::SpectralConstant { .. }
            | BsdfNode::RgbUplift { .. }
            | BsdfNode::BlackbodyEmitter { .. }
            | BsdfNode::Wavelength
            | BsdfNode::CosTheta => (false, 0.0),
            BsdfNode::Scalar { value } => (true, *value),
            BsdfNode::Add { a, b } => {
                let c = is_const[*a] && is_const[*b];
                (c, if c { cv(*a) + cv(*b) } else { 0.0 })
            }
            BsdfNode::Multiply { a, b } => {
                let c = is_const[*a] && is_const[*b];
                (c, if c { cv(*a) * cv(*b) } else { 0.0 })
            }
            BsdfNode::Mix { a, b, factor } => {
                let c = is_const[*a] && is_const[*b];
                (
                    c,
                    if c {
                        cv(*a) * (1.0 - factor) + cv(*b) * factor
                    } else {
                        0.0
                    },
                )
            }
            BsdfNode::Scale { input, factor } => {
                (is_const[*input], if is_const[*input] { cv(*input) * factor } else { 0.0 })
            }
            BsdfNode::Invert { input } => (
                is_const[*input],
                if is_const[*input] {
                    (1.0 - cv(*input)).clamp(0.0, 1.0)
                } else {
                    0.0
                },
            ),
            BsdfNode::Clamp { input, min, max } => (
                is_const[*input],
                if is_const[*input] {
                    cv(*input).clamp(*min, *max)
                } else {
                    0.0
                },
            ),
            // Fresnel & Layer depend on cos_theta → never const.
            BsdfNode::Fresnel { .. } | BsdfNode::Layer { .. } => (false, 0.0),
        }
    }

    /// Emit the WGSL right-hand-side expression for a non-const node.
    fn emit_node_rhs(
        &self,
        id: NodeId,
        var: &dyn Fn(NodeId) -> String,
        is_const: &[bool],
        const_val: &[f32],
    ) -> String {
        // Operand reference: const ref → literal, else its var.
        let op = |r: NodeId| -> String {
            if is_const[r] {
                wgsl_f32(const_val[r])
            } else {
                var(r)
            }
        };
        match &self.nodes[id] {
            BsdfNode::SpectralConstant { .. } => {
                // Indexes the module-level const array hoisted in compile_to_wgsl.
                format!("SPD_{id}[band]")
            }
            BsdfNode::RgbUplift { rgb } => format!(
                "smits_uplift_band(vec3<f32>({}, {}, {}), band)",
                wgsl_f32(rgb[0]),
                wgsl_f32(rgb[1]),
                wgsl_f32(rgb[2])
            ),
            BsdfNode::BlackbodyEmitter { kelvin } => {
                format!("blackbody_band(lambda_nm, {})", wgsl_f32(*kelvin))
            }
            BsdfNode::Wavelength => "lambda_nm".to_string(),
            BsdfNode::CosTheta => "cos_theta".to_string(),
            BsdfNode::Scalar { value } => wgsl_f32(*value),
            BsdfNode::Add { a, b } => format!("{} + {}", op(*a), op(*b)),
            BsdfNode::Multiply { a, b } => format!("{} * {}", op(*a), op(*b)),
            BsdfNode::Mix { a, b, factor } => format!(
                "{} * {} + {} * {}",
                op(*a),
                wgsl_f32(1.0 - factor),
                op(*b),
                wgsl_f32(*factor)
            ),
            BsdfNode::Scale { input, factor } => {
                format!("{} * {}", op(*input), wgsl_f32(*factor))
            }
            BsdfNode::Invert { input } => format!("clamp(1.0 - {}, 0.0, 1.0)", op(*input)),
            BsdfNode::Clamp { input, min, max } => {
                format!("clamp({}, {}, {})", op(*input), wgsl_f32(*min), wgsl_f32(*max))
            }
            BsdfNode::Fresnel { base, power } => format!(
                "{base_v} + (1.0 - {base_v}) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), {p})",
                base_v = op(*base),
                p = wgsl_f32(*power)
            ),
            BsdfNode::Layer { coat, base, f0 } => {
                // w = f0 + (1-f0)*(1-cosθ)^5, result = base*(1-w) + coat*w.
                let f0s = wgsl_f32(*f0);
                let wexpr = format!(
                    "({f0s} + (1.0 - {f0s}) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0))"
                );
                format!(
                    "{base} * (1.0 - {w}) + {coat} * {w}",
                    base = op(*base),
                    coat = op(*coat),
                    w = wexpr
                )
            }
        }
    }

    /// WGSL implementation of `smits_uplift_band` — must mirror the CPU version.
    fn wgsl_uplift_helper() -> String {
        // Basis curves identical to the CPU `smits_uplift_band` constants.
        r#"const SMITS_WHITE   = array<f32, 16>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
const SMITS_BLUE    = array<f32, 16>(1.0, 1.0, 1.0, 0.96, 0.8, 0.52, 0.28, 0.13, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
const SMITS_GREEN   = array<f32, 16>(0.0, 0.0, 0.06, 0.2, 0.45, 0.78, 0.96, 1.0, 0.92, 0.62, 0.28, 0.1, 0.03, 0.01, 0.0, 0.0);
const SMITS_RED     = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.05, 0.14, 0.36, 0.68, 0.92, 1.0, 1.0, 1.0, 1.0, 1.0);
const SMITS_CYAN    = array<f32, 16>(1.0, 1.0, 1.0, 1.0, 0.96, 0.92, 0.88, 0.78, 0.52, 0.24, 0.08, 0.02, 0.0, 0.0, 0.0, 0.0);
const SMITS_MAGENTA = array<f32, 16>(1.0, 1.0, 0.96, 0.8, 0.52, 0.2, 0.06, 0.1, 0.34, 0.7, 0.94, 1.0, 1.0, 1.0, 1.0, 1.0);
const SMITS_YELLOW  = array<f32, 16>(0.0, 0.0, 0.02, 0.08, 0.24, 0.55, 0.82, 0.96, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

fn smits_uplift_band(rgb: vec3<f32>, band: u32) -> f32 {
    let r = clamp(rgb.x, 0.0, 1.0);
    let g = clamp(rgb.y, 0.0, 1.0);
    let b = clamp(rgb.z, 0.0, 1.0);
    let white = SMITS_WHITE[band];
    let blue = SMITS_BLUE[band];
    let green = SMITS_GREEN[band];
    let red = SMITS_RED[band];
    let cyan = SMITS_CYAN[band];
    let magenta = SMITS_MAGENTA[band];
    let yellow = SMITS_YELLOW[band];
    var spd = 0.0;
    if (r <= g && r <= b) {
        spd = spd + r * white;
        if (g <= b) {
            spd = spd + (g - r) * cyan;
            spd = spd + (b - g) * blue;
        } else {
            spd = spd + (b - r) * cyan;
            spd = spd + (g - b) * green;
        }
    } else if (g <= r && g <= b) {
        spd = spd + g * white;
        if (r <= b) {
            spd = spd + (r - g) * magenta;
            spd = spd + (b - r) * blue;
        } else {
            spd = spd + (b - g) * magenta;
            spd = spd + (r - b) * red;
        }
    } else {
        spd = spd + b * white;
        if (r <= g) {
            spd = spd + (r - b) * yellow;
            spd = spd + (g - r) * green;
        } else {
            spd = spd + (g - b) * yellow;
            spd = spd + (r - g) * red;
        }
    }
    return clamp(spd, 0.0, 1.0);
}
"#
        .to_string()
    }

    /// WGSL implementation of `blackbody_band` — must mirror the CPU Planck.
    ///
    /// Normalises by the precomputed peak across the 16 band wavelengths so the
    /// emitted value matches `eval_cpu` exactly. Uses f32 here (the GPU has no
    /// f64); the CPU test mirrors with a matched-precision oracle so equivalence
    /// holds within the test epsilon (the harness compares the GPU output to the
    /// f32-WGSL Planck via the same formula).
    fn wgsl_blackbody_helper() -> String {
        r#"const PLANCK_LAMBDAS = array<f32, 16>(380.0, 405.0, 430.0, 455.0, 480.0, 505.0, 530.0, 555.0, 580.0, 605.0, 630.0, 655.0, 680.0, 705.0, 730.0, 755.0);

fn planck_relative_f(lambda_nm: f32, kelvin: f32) -> f32 {
    let h = 6.62607e-34;
    let c = 2.9979245e8;
    let kb = 1.380649e-23;
    let lambda_m = lambda_nm * 1e-9;
    let t = max(kelvin, 1.0);
    let a = 2.0 * h * c * c / pow(lambda_m, 5.0);
    let x = h * c / (lambda_m * kb * t);
    let denom = exp(x) - 1.0;
    if (denom <= 0.0) {
        return 0.0;
    }
    return a / denom;
}

fn blackbody_band(lambda_nm: f32, kelvin: f32) -> f32 {
    var peak = 1e-30;
    for (var i = 0u; i < 16u; i = i + 1u) {
        peak = max(peak, planck_relative_f(PLANCK_LAMBDAS[i], kelvin));
    }
    return planck_relative_f(lambda_nm, kelvin) / peak;
}
"#
        .to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn argmax(spd: &[f32; N_BANDS]) -> usize {
        let mut best = 0;
        for i in 1..N_BANDS {
            if spd[i] > spd[best] {
                best = i;
            }
        }
        best
    }

    /// Substrate-style layer: red coat over a dim grey base, blended by Fresnel.
    /// At normal incidence the base dominates but the red coat tints it
    /// red-dominant; at grazing incidence the coat takes over.
    #[test]
    fn layer_red_coat_red_dominant_and_grazing_shifts_to_coat() {
        // nodes: 0 = RgbUplift(red), 1 = SpectralConstant(dim grey), 2 = Layer
        let dag = MaterialDag {
            nodes: vec![
                BsdfNode::RgbUplift { rgb: [1.0, 0.0, 0.0] },
                BsdfNode::SpectralConstant { spd: [0.18; N_BANDS] },
                BsdfNode::Layer { coat: 0, base: 1, f0: 0.04 },
            ],
            output: 2,
        };
        dag.validate().unwrap();

        // Normal incidence (cos_theta = 1): red bands (indices 10-15) strictly
        // exceed blue bands (indices 2-4).
        let normal = dag.eval_cpu_spd(1.0);
        let red_avg = (normal[11] + normal[13] + normal[15]) / 3.0;
        let blue_avg = (normal[2] + normal[3] + normal[4]) / 3.0;
        assert!(
            red_avg > blue_avg,
            "red bands ({red_avg}) must exceed blue bands ({blue_avg}) at normal incidence: {normal:?}"
        );

        // Grazing incidence (cos_theta ~ 0): the coat weight → 1, so the output
        // moves toward the pure red coat. The red band rises vs normal incidence.
        let grazing = dag.eval_cpu_spd(0.0);
        assert!(
            grazing[13] > normal[13],
            "grazing should shift band 13 toward the red coat: normal={}, grazing={}",
            normal[13],
            grazing[13]
        );
        // And blue should drop toward the coat's ~0 at grazing.
        assert!(
            grazing[3] < normal[3],
            "grazing should drop blue band 3 toward the coat: normal={}, grazing={}",
            normal[3],
            grazing[3]
        );
    }

    /// Blackbody argmax ordering: 6500K peaks at a shorter wavelength (lower band
    /// index) than 2000K within the 380-755 nm window.
    #[test]
    fn blackbody_temperature_shifts_peak() {
        let hot = MaterialDag {
            nodes: vec![BsdfNode::BlackbodyEmitter { kelvin: 6500.0 }],
            output: 0,
        };
        let warm = MaterialDag {
            nodes: vec![BsdfNode::BlackbodyEmitter { kelvin: 2000.0 }],
            output: 0,
        };
        let hot_spd = hot.eval_cpu_spd(1.0);
        let warm_spd = warm.eval_cpu_spd(1.0);
        let hot_peak = argmax(&hot_spd);
        let warm_peak = argmax(&warm_spd);
        assert!(
            hot_peak < warm_peak,
            "6500K peak band ({hot_peak}, λ={}) must be bluer than 2000K ({warm_peak}, λ={}): hot={hot_spd:?} warm={warm_spd:?}",
            BAND_WAVELENGTHS[hot_peak],
            BAND_WAVELENGTHS[warm_peak],
        );
        // 2000K is so warm its peak is in the red end of the visible window.
        assert_eq!(warm_peak, 15, "2000K should peak at the reddest band");
    }

    /// A 6+ node graph compiles to WGSL that contains no placeholder and parses
    /// + validates via naga.
    #[test]
    fn compile_six_node_graph_validates_via_naga() {
        // 0:RgbUplift  1:SpectralConst  2:Layer(0 over 1)  3:Scalar
        // 4:Scale(2 by 3-ish)  5:Wavelength  6:Fresnel(2)  7:Mix(4,6)
        let dag = MaterialDag {
            nodes: vec![
                BsdfNode::RgbUplift { rgb: [0.7, 0.2, 0.1] },
                BsdfNode::SpectralConstant {
                    spd: [
                        0.1, 0.12, 0.14, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4, 0.42, 0.44, 0.45, 0.46,
                        0.46, 0.47, 0.47,
                    ],
                },
                BsdfNode::Layer { coat: 0, base: 1, f0: 0.04 },
                BsdfNode::Scalar { value: 0.8 },
                BsdfNode::Scale { input: 2, factor: 0.9 },
                BsdfNode::Wavelength,
                BsdfNode::Fresnel { base: 2, power: 5.0 },
                BsdfNode::Mix { a: 4, b: 6, factor: 0.3 },
            ],
            output: 7,
        };
        let wgsl = dag.compile_to_wgsl().expect("compile");
        assert!(
            !wgsl.contains("todo")
                && !wgsl.contains("unimplemented")
                && !wgsl.contains("PLACEHOLDER")
                && !wgsl.contains("()[band]"),
            "emitted WGSL contains a placeholder:\n{wgsl}"
        );
        assert!(wgsl.contains("fn material_eval(band: u32, lambda_nm: f32, cos_theta: f32) -> f32"));

        let parsed = naga::front::wgsl::parse_str(&wgsl);
        let module = parsed.unwrap_or_else(|e| panic!("naga parse failed: {e:?}\n{wgsl}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator.validate(&module);
        assert!(
            info.is_ok(),
            "naga validation failed: {:?}\n{wgsl}",
            info.err()
        );
    }

    /// Cycle is rejected with a typed error.
    #[test]
    fn cycle_is_rejected() {
        // node 0 references node 1, node 1 references node 0 -> cycle.
        let dag = MaterialDag {
            nodes: vec![
                BsdfNode::Scale { input: 1, factor: 0.5 },
                BsdfNode::Scale { input: 0, factor: 0.5 },
            ],
            output: 0,
        };
        match dag.validate() {
            Err(MaterialCompileError::Cycle { .. }) => {}
            other => panic!("expected Cycle error, got {other:?}"),
        }
        assert!(matches!(
            dag.compile_to_wgsl(),
            Err(MaterialCompileError::Cycle { .. })
        ));
    }

    /// Unknown reference is rejected with a typed error.
    #[test]
    fn unknown_ref_is_rejected() {
        let dag = MaterialDag {
            nodes: vec![BsdfNode::Scale { input: 99, factor: 0.5 }],
            output: 0,
        };
        match dag.validate() {
            Err(MaterialCompileError::UnknownRef { node: 0, referenced: 99 }) => {}
            other => panic!("expected UnknownRef error, got {other:?}"),
        }
    }

    /// Out-of-range output is rejected.
    #[test]
    fn unknown_output_is_rejected() {
        let dag = MaterialDag {
            nodes: vec![BsdfNode::Scalar { value: 0.5 }],
            output: 5,
        };
        assert!(matches!(
            dag.validate(),
            Err(MaterialCompileError::UnknownOutput { output: 5, len: 1 })
        ));
    }

    /// Empty graph is rejected.
    #[test]
    fn empty_graph_is_rejected() {
        let dag = MaterialDag {
            nodes: vec![],
            output: 0,
        };
        assert!(matches!(dag.validate(), Err(MaterialCompileError::Empty)));
    }

    /// Constant subgraphs are folded to a single literal (no arithmetic emitted).
    #[test]
    fn constant_folding_collapses_scalar_arithmetic() {
        // (0.2 + 0.3) * 0.4 = 0.2 — all const, should fold.
        let dag = MaterialDag {
            nodes: vec![
                BsdfNode::Scalar { value: 0.2 },
                BsdfNode::Scalar { value: 0.3 },
                BsdfNode::Add { a: 0, b: 1 },
                BsdfNode::Scalar { value: 0.4 },
                BsdfNode::Multiply { a: 2, b: 3 },
            ],
            output: 4,
        };
        let folded = dag.eval_cpu(0, BAND_WAVELENGTHS[0], 1.0);
        assert!((folded - 0.2).abs() < 1e-6, "expected 0.2, got {folded}");
        let wgsl = dag.compile_to_wgsl().unwrap();
        // The output var should be a folded literal, not a `*` expression.
        assert!(
            wgsl.contains("let n4 = 0.2"),
            "constant arithmetic should be folded to a literal:\n{wgsl}"
        );
    }

    /// Smits uplift of a pure colour produces a hue-consistent reflectance.
    #[test]
    fn rgb_uplift_blue_is_blue_dominant() {
        let dag = MaterialDag {
            nodes: vec![BsdfNode::RgbUplift { rgb: [0.0, 0.0, 1.0] }],
            output: 0,
        };
        let spd = dag.eval_cpu_spd(1.0);
        let blue_avg = (spd[2] + spd[3] + spd[4]) / 3.0;
        let red_avg = (spd[11] + spd[13] + spd[15]) / 3.0;
        assert!(
            blue_avg > red_avg,
            "blue uplift must be blue-dominant: blue={blue_avg} red={red_avg} spd={spd:?}"
        );
    }
}
