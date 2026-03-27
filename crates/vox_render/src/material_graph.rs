use vox_core::spectral::SpectralBands;
use serde::{Serialize, Deserialize};

/// A node in the spectral material graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialNode {
    /// Constant spectral reflectance.
    Constant { spd: [f32; 8] },
    /// Reference a named material from the library.
    MaterialRef { tag: String },
    /// Multiply two spectral inputs element-wise.
    Multiply { a: Box<MaterialNode>, b: Box<MaterialNode> },
    /// Add two spectral inputs.
    Add { a: Box<MaterialNode>, b: Box<MaterialNode> },
    /// Lerp between two inputs based on a factor.
    Mix { a: Box<MaterialNode>, b: Box<MaterialNode>, factor: f32 },
    /// Scale all bands by a constant.
    Scale { input: Box<MaterialNode>, factor: f32 },
    /// Fresnel effect: increase reflectance at grazing angles.
    Fresnel { base: Box<MaterialNode>, power: f32 },
    /// Clamp all bands to [min, max].
    Clamp { input: Box<MaterialNode>, min: f32, max: f32 },
    /// Invert: 1.0 - input per band.
    Invert { input: Box<MaterialNode> },
}

impl MaterialNode {
    /// Evaluate the node tree to produce an 8-band SPD.
    pub fn evaluate(&self) -> SpectralBands {
        match self {
            Self::Constant { spd } => SpectralBands(*spd),
            Self::MaterialRef { tag } => {
                // Look up from default library
                let lib = vox_data::materials::MaterialLibrary::default();
                lib.get(tag).map(|m| m.spd).unwrap_or(SpectralBands([0.5; 8]))
            }
            Self::Multiply { a, b } => {
                let va = a.evaluate();
                let vb = b.evaluate();
                SpectralBands(std::array::from_fn(|i| va.0[i] * vb.0[i]))
            }
            Self::Add { a, b } => {
                let va = a.evaluate();
                let vb = b.evaluate();
                SpectralBands(std::array::from_fn(|i| (va.0[i] + vb.0[i]).min(1.0)))
            }
            Self::Mix { a, b, factor } => {
                let va = a.evaluate();
                let vb = b.evaluate();
                let f = *factor;
                SpectralBands(std::array::from_fn(|i| va.0[i] * (1.0 - f) + vb.0[i] * f))
            }
            Self::Scale { input, factor } => {
                let v = input.evaluate();
                SpectralBands(std::array::from_fn(|i| (v.0[i] * factor).clamp(0.0, 1.0)))
            }
            Self::Fresnel { base, power } => {
                let v = base.evaluate();
                // Simplified: boost higher wavelengths (simulates grazing angle)
                SpectralBands(std::array::from_fn(|i| {
                    let t = i as f32 / 7.0;
                    (v.0[i] + (1.0 - v.0[i]) * t.powf(*power)).clamp(0.0, 1.0)
                }))
            }
            Self::Clamp { input, min, max } => {
                let v = input.evaluate();
                SpectralBands(std::array::from_fn(|i| v.0[i].clamp(*min, *max)))
            }
            Self::Invert { input } => {
                let v = input.evaluate();
                SpectralBands(std::array::from_fn(|i| 1.0 - v.0[i]))
            }
        }
    }
}

/// A complete material definition using the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralMaterialGraph {
    pub name: String,
    pub albedo: MaterialNode,
    pub roughness: f32,
    pub metallic: f32,
    pub emission: Option<MaterialNode>,
}

impl SpectralMaterialGraph {
    /// Evaluate the material's albedo SPD.
    pub fn evaluate_albedo(&self) -> SpectralBands {
        self.albedo.evaluate()
    }

    /// Evaluate emission SPD (returns zeros if no emission).
    pub fn evaluate_emission(&self) -> SpectralBands {
        self.emission.as_ref().map(|n| n.evaluate()).unwrap_or(SpectralBands([0.0; 8]))
    }
}

/// Serialize a material graph to TOML for storage.
pub fn serialize_material(mat: &SpectralMaterialGraph) -> Result<String, String> {
    toml::to_string_pretty(mat).map_err(|e| e.to_string())
}

/// Deserialize a material graph from TOML.
pub fn deserialize_material(toml_str: &str) -> Result<SpectralMaterialGraph, String> {
    toml::from_str(toml_str).map_err(|e| e.to_string())
}
