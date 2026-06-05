use vox_core::spectral::SpectralBands;
use serde::{Serialize, Deserialize};

/// A node in the spectral material graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialNode {
    /// Constant spectral reflectance.
    Constant { spd: [f32; 16] },
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
    /// Evaluate the node tree to produce a 16-band SPD.
    pub fn evaluate(&self) -> SpectralBands {
        match self {
            Self::Constant { spd } => SpectralBands(*spd),
            Self::MaterialRef { tag } => {
                // Look up from default library
                let lib = vox_data::materials::MaterialLibrary::default();
                lib.get(tag).map(|m| m.spd).unwrap_or(SpectralBands([0.5; 16]))
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
                    let t = i as f32 / 15.0;
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
        self.emission.as_ref().map(|n| n.evaluate()).unwrap_or(SpectralBands([0.0; 16]))
    }
}

/// Wrapper holding a root node that can be compiled to a naga::Module.
pub struct MaterialGraph {
    pub root: MaterialNode,
}

impl MaterialGraph {
    pub fn new(root: MaterialNode) -> Self {
        Self { root }
    }

    /// Compile the graph to a validated naga::Module.
    ///
    /// The module contains a single function `evaluate_material() -> array<f32, 16>`.
    /// Callers pass the module to `wgpu::Device::create_shader_module_from_naga()`.
    pub fn compile(
        &self,
    ) -> Result<naga::Module, Box<naga::WithSpan<naga::valid::ValidationError>>> {
        use crate::naga_builder::NagaBuilder;
        let module = self.build_module();
        NagaBuilder::validate(&module)?;
        Ok(module)
    }

    fn build_module(&self) -> naga::Module {
        use naga::{Function, FunctionResult, Module, Span, Statement};
        use crate::naga_builder::NagaBuilder;

        let mut module = Module::default();
        let arr_ty = NagaBuilder::array_f32_8(&mut module.types);

        let mut func = Function {
            name: Some("evaluate_material".into()),
            result: Some(FunctionResult { ty: arr_ty, binding: None }),
            ..Default::default()
        };

        // CPU-evaluate the node tree to produce the constant SPD for the shader.
        let spd = self.evaluate_cpu();
        let (expr, compose_start) = NagaBuilder::emit_constant_spd(
            &spd.0,
            &mut func.expressions,
            &mut module.types,
        );

        let emit_range = func.expressions.range_from(compose_start);
        func.body.push(Statement::Emit(emit_range), Span::UNDEFINED);
        func.body.push(Statement::Return { value: Some(expr) }, Span::UNDEFINED);

        module.functions.append(func, Span::UNDEFINED);
        module
    }

    /// CPU evaluation used to seed the naga constant SPD.
    fn evaluate_cpu(&self) -> vox_core::spectral::SpectralBands {
        self.root.evaluate()
    }
}

#[cfg(test)]
mod material_graph_tests {
    use super::*;

    #[test]
    fn compile_constant_graph_produces_valid_module() {
        let graph = MaterialGraph::new(MaterialNode::Constant {
            spd: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        });
        let result = graph.compile();
        assert!(result.is_ok(), "compile() returned error: {:?}", result.err());
    }

    #[test]
    fn compile_multiply_graph_validates() {
        let graph = MaterialGraph::new(MaterialNode::Multiply {
            a: Box::new(MaterialNode::Constant { spd: [0.5; 16] }),
            b: Box::new(MaterialNode::Constant { spd: [0.8; 16] }),
        });
        assert!(graph.compile().is_ok());
    }

    #[test]
    fn compile_mix_graph_validates() {
        let graph = MaterialGraph::new(MaterialNode::Mix {
            a: Box::new(MaterialNode::Constant { spd: [0.2; 16] }),
            b: Box::new(MaterialNode::Constant { spd: [0.9; 16] }),
            factor: 0.3,
        });
        assert!(graph.compile().is_ok());
    }

    #[test]
    fn compiled_module_has_correct_function_name() {
        let graph = MaterialGraph::new(MaterialNode::Scale {
            input: Box::new(MaterialNode::Constant { spd: [1.0; 16] }),
            factor: 0.5,
        });
        let module = graph.compile().unwrap();
        let names: Vec<_> = module.functions.iter()
            .filter_map(|(_, f)| f.name.as_deref())
            .collect();
        assert!(names.contains(&"evaluate_material"));
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
