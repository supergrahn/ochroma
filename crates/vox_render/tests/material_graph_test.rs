use vox_render::material_graph::*;

#[test]
fn constant_node_returns_value() {
    let node = MaterialNode::Constant { spd: [0.5; 8] };
    let result = node.evaluate();
    assert_eq!(result.0, [0.5; 8]);
}

#[test]
fn mix_blends_two_materials() {
    let a = MaterialNode::Constant { spd: [0.0; 8] };
    let b = MaterialNode::Constant { spd: [1.0; 8] };
    let mix = MaterialNode::Mix { a: Box::new(a), b: Box::new(b), factor: 0.5 };
    let result = mix.evaluate();
    for v in &result.0 { assert!((v - 0.5).abs() < 0.01); }
}

#[test]
fn multiply_modulates_spectra() {
    let a = MaterialNode::Constant { spd: [0.8; 8] };
    let b = MaterialNode::Constant { spd: [0.5; 8] };
    let mul = MaterialNode::Multiply { a: Box::new(a), b: Box::new(b) };
    let result = mul.evaluate();
    for v in &result.0 { assert!((v - 0.4).abs() < 0.01); }
}

#[test]
fn invert_flips_spectrum() {
    let node = MaterialNode::Invert {
        input: Box::new(MaterialNode::Constant { spd: [0.3; 8] }),
    };
    let result = node.evaluate();
    for v in &result.0 { assert!((v - 0.7).abs() < 0.01); }
}

#[test]
fn material_graph_serializes_to_toml() {
    let mat = SpectralMaterialGraph {
        name: "weathered_brick".to_string(),
        albedo: MaterialNode::Mix {
            a: Box::new(MaterialNode::MaterialRef { tag: "brick_red".to_string() }),
            b: Box::new(MaterialNode::Constant { spd: [0.1; 8] }),
            factor: 0.3,
        },
        roughness: 0.8,
        metallic: 0.0,
        emission: None,
    };

    let toml = serialize_material(&mat).unwrap();
    assert!(toml.contains("weathered_brick"));

    let loaded = deserialize_material(&toml).unwrap();
    assert_eq!(loaded.name, "weathered_brick");
}

#[test]
fn complex_graph_evaluates() {
    let mat = SpectralMaterialGraph {
        name: "wet_metal".to_string(),
        albedo: MaterialNode::Fresnel {
            base: Box::new(MaterialNode::Scale {
                input: Box::new(MaterialNode::MaterialRef { tag: "metal_steel".to_string() }),
                factor: 0.9,
            }),
            power: 3.0,
        },
        roughness: 0.2,
        metallic: 1.0,
        emission: Some(MaterialNode::Scale {
            input: Box::new(MaterialNode::Constant { spd: [1.0; 8] }),
            factor: 0.01,
        }),
    };

    let albedo = mat.evaluate_albedo();
    let emission = mat.evaluate_emission();
    for v in &albedo.0 { assert!(*v >= 0.0 && *v <= 1.0); }
    for v in &emission.0 { assert!(*v >= 0.0 && *v <= 0.02); }
}
