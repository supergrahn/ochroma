//! Builds a `naga::Module` from a `MaterialNode` tree.
//!
//! Each `MaterialNode` variant maps to one or more naga IR expressions.
//! The output module exposes a single function:
//!
//!   fn evaluate_material() -> array<f32, 16>
//!
//! Consumers pass the module directly to `wgpu::Device::create_shader_module_from_naga()`
//! — no WGSL string serialisation required.

use naga::{
    Arena, Expression, Function, FunctionResult, Handle, Literal, Module, ScalarKind,
    Span, Statement, Type, TypeInner, UniqueArena,
};

/// Provides static helpers for building naga IR for material evaluation functions.
pub struct NagaBuilder;

impl NagaBuilder {
    /// Add an `array<f32, 16>` type to the module's type arena, returning its handle.
    /// UniqueArena deduplicates identical types.
    pub fn array_f32_8(types: &mut UniqueArena<Type>) -> Handle<Type> {
        let f32_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(naga::Scalar {
                    kind: ScalarKind::Float,
                    width: 4,
                }),
            },
            Span::UNDEFINED,
        );
        types.insert(
            Type {
                name: Some("SpectralArray".into()),
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: naga::ArraySize::Constant(
                        std::num::NonZeroU32::new(16).unwrap(),
                    ),
                    stride: 4,
                },
            },
            Span::UNDEFINED,
        )
    }

    /// Emit a constant `array<f32, 16>` from a fixed SPD value.
    /// Returns:
    /// - `compose_handle`: the `Expression::Compose` handle (needs `Statement::Emit`)
    /// - `compose_start`: the arena length just before the Compose was appended
    ///   (pass to `expressions.range_from(compose_start)` to build the emit range)
    ///
    /// Literal sub-expressions are always in scope and do not need emitting.
    pub fn emit_constant_spd(
        spd: &[f32; 16],
        expressions: &mut Arena<Expression>,
        types: &mut UniqueArena<Type>,
    ) -> (Handle<Expression>, usize) {
        let components: Vec<Handle<Expression>> = spd
            .iter()
            .map(|&v| {
                expressions.append(Expression::Literal(Literal::F32(v)), Span::UNDEFINED)
            })
            .collect();

        let arr_ty = Self::array_f32_8(types);
        // Snapshot length right before appending the Compose — only Compose needs Emit.
        let compose_start = expressions.len();
        let handle = expressions.append(
            Expression::Compose { ty: arr_ty, components },
            Span::UNDEFINED,
        );
        (handle, compose_start)
    }

    /// Build a complete naga `Module` that evaluates the given constant SPD.
    /// The module contains one function named `evaluate_material`.
    pub fn build_constant(spd: [f32; 16]) -> Module {
        let mut module = Module::default();
        let arr_ty = Self::array_f32_8(&mut module.types);

        let mut func = Function::default();
        func.name = Some("evaluate_material".into());
        func.result = Some(FunctionResult { ty: arr_ty, binding: None });

        let (expr, compose_start) = Self::emit_constant_spd(
            &spd,
            &mut func.expressions,
            &mut module.types,
        );

        // Only the Compose expression needs to be emitted into scope; Literals are always in scope.
        let emit_range = func.expressions.range_from(compose_start);
        func.body.push(Statement::Emit(emit_range), Span::UNDEFINED);
        func.body.push(
            Statement::Return { value: Some(expr) },
            Span::UNDEFINED,
        );

        module.functions.append(func, Span::UNDEFINED);
        module
    }

    /// Validate the built module using naga's validator.
    /// Returns `Ok(ModuleInfo)` if the module is valid naga IR.
    pub fn validate(module: &Module) -> Result<naga::valid::ModuleInfo, naga::WithSpan<naga::valid::ValidationError>> {
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_module_validates() {
        let spd = [0.8, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8];
        let module = NagaBuilder::build_constant(spd);
        let result = NagaBuilder::validate(&module);
        assert!(result.is_ok(), "naga validation failed: {:?}", result.err());
    }

    #[test]
    fn module_has_evaluate_material_function() {
        let module = NagaBuilder::build_constant([0.5; 16]);
        let names: Vec<_> = module.functions.iter()
            .filter_map(|(_, f)| f.name.as_deref())
            .collect();
        println!("functions: {:?}", names);
        assert!(
            names.contains(&"evaluate_material"),
            "expected 'evaluate_material' function, got: {:?}", names
        );
    }

    #[test]
    fn array_f32_8_type_has_sixteen_elements() {
        let mut types = UniqueArena::default();
        let handle = NagaBuilder::array_f32_8(&mut types);
        let ty = &types[handle];
        match &ty.inner {
            naga::TypeInner::Array { size: naga::ArraySize::Constant(n), .. } => {
                assert_eq!(n.get(), 16, "expected array size 16, got {}", n.get());
            }
            other => panic!("expected Array type, got {:?}", other),
        }
    }
}
