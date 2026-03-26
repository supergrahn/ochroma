use vox_data::proc_gs::{emit_splats, GeometryStrategy, SplatRule};

const VICTORIAN_RULE_TOML: &str = r#"
[header]
asset_type = "building"
style = "victorian"

[geometry]
strategy = "StructuredPlacement"
floor_count_min = 2
floor_count_max = 4
height_min = 6.0
height_max = 14.0
width_min = 8.0
width_max = 14.0
depth_min = 8.0
depth_max = 14.0
splats_per_sqm = 2.0

[[material_zones]]
name = "facade"
material_tag = "brick_red"
zone_type = "wall"
coverage = 0.8

[[material_zones]]
name = "roof"
material_tag = "slate_grey"
zone_type = "roof"
coverage = 1.0

[variation]
scale_min = 0.05
scale_max = 0.15
opacity_min = 0.7
opacity_max = 1.0
"#;

#[test]
fn parse_victorian_house_rule() {
    let rule: SplatRule = toml::from_str(VICTORIAN_RULE_TOML).expect("failed to parse TOML");
    assert_eq!(rule.header.asset_type, "building");
    assert_eq!(rule.header.style, "victorian");
    assert_eq!(rule.geometry.strategy, GeometryStrategy::StructuredPlacement);
    assert_eq!(rule.geometry.floor_count_min, 2);
    assert_eq!(rule.geometry.floor_count_max, 4);
    assert_eq!(rule.material_zones.len(), 2);
}

#[test]
fn emit_splats_nonzero_count() {
    let rule: SplatRule = toml::from_str(VICTORIAN_RULE_TOML).unwrap();
    let splats = emit_splats(&rule, 42);
    assert!(!splats.is_empty(), "expected nonzero splat count, got 0");
}

#[test]
fn emit_splats_deterministic() {
    let rule: SplatRule = toml::from_str(VICTORIAN_RULE_TOML).unwrap();
    let a = emit_splats(&rule, 12345);
    let b = emit_splats(&rule, 12345);
    assert_eq!(a.len(), b.len());
    for (sa, sb) in a.iter().zip(b.iter()) {
        assert_eq!(sa.position, sb.position);
        assert_eq!(sa.opacity, sb.opacity);
    }
}

#[test]
fn emit_splats_different_seeds_produce_different_output() {
    let rule: SplatRule = toml::from_str(VICTORIAN_RULE_TOML).unwrap();
    let a = emit_splats(&rule, 1);
    let b = emit_splats(&rule, 2);
    // At minimum the counts or positions should differ
    let same = a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(sa, sb)| sa.position == sb.position);
    assert!(!same, "expected different output for different seeds");
}
