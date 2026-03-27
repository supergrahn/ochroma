use half::f16;
use vox_data::material_system::*;

const BRICK_TOML: &str = r#"
[material]
name = "Weathered Brick"
type = "spectral"

[spectral]
bands = [0.08, 0.08, 0.10, 0.15, 0.25, 0.55, 0.65, 0.60]

[properties]
roughness = 0.8
metallic = 0.0
opacity = 1.0
emission = 0.0

[worn]
bands = [0.06, 0.06, 0.08, 0.12, 0.18, 0.38, 0.45, 0.40]
wear_factor = 0.0
"#;

#[test]
fn material_system_load_from_toml_string() {
    let mat = MaterialDefinition::from_toml_str(BRICK_TOML, "test".into()).unwrap();
    assert_eq!(mat.material.name, "Weathered Brick");
    assert_eq!(mat.material.mat_type, "spectral");
    assert_eq!(mat.spectral.bands[0], 0.08);
    assert_eq!(mat.properties.roughness, 0.8);
    assert_eq!(mat.properties.metallic, 0.0);
    assert!(mat.worn.is_some());
}

#[test]
fn material_system_save_and_reload_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brick.toml");

    let mat = MaterialDefinition::from_toml_str(BRICK_TOML, "test".into()).unwrap();
    mat.save(&path).unwrap();

    let reloaded = MaterialDefinition::load(&path).unwrap();
    assert_eq!(reloaded.material.name, mat.material.name);
    assert_eq!(reloaded.spectral.bands, mat.spectral.bands);
    assert_eq!(reloaded.properties.roughness, mat.properties.roughness);
}

#[test]
fn material_system_effective_bands_wear_zero_equals_fresh() {
    let mat = MaterialDefinition::from_toml_str(BRICK_TOML, "test".into()).unwrap();
    // wear_factor = 0.0, so effective should equal fresh bands
    let effective = mat.effective_bands();
    assert_eq!(effective, mat.spectral.bands);
}

#[test]
fn material_system_effective_bands_wear_one_equals_worn() {
    let mut mat = MaterialDefinition::from_toml_str(BRICK_TOML, "test".into()).unwrap();
    mat.worn.as_mut().unwrap().wear_factor = 1.0;
    let effective = mat.effective_bands();
    let worn_bands = mat.worn.as_ref().unwrap().bands;
    assert_eq!(effective, worn_bands);
}

#[test]
fn material_system_effective_bands_half_is_midpoint() {
    let mut mat = MaterialDefinition::from_toml_str(BRICK_TOML, "test".into()).unwrap();
    mat.worn.as_mut().unwrap().wear_factor = 0.5;
    let effective = mat.effective_bands();
    let fresh = mat.spectral.bands;
    let worn = mat.worn.as_ref().unwrap().bands;
    for i in 0..8 {
        let expected = (fresh[i] + worn[i]) / 2.0;
        assert!((effective[i] - expected).abs() < 1e-6, "band {} mismatch", i);
    }
}

#[test]
fn material_system_to_splat_spectral_valid_f16() {
    let mat = MaterialDefinition::from_toml_str(BRICK_TOML, "test".into()).unwrap();
    let splat = mat.to_splat_spectral();
    for (i, &bits) in splat.iter().enumerate() {
        let val = f16::from_bits(bits).to_f32();
        assert!(val.is_finite(), "band {} is not finite: {}", i, val);
        assert!(val >= 0.0, "band {} is negative: {}", i, val);
        assert!(val <= 1.0, "band {} exceeds 1.0: {}", i, val);
    }
}

#[test]
fn material_system_loads_directory() {
    let dir = tempfile::tempdir().unwrap();
    let count = create_default_materials(dir.path()).unwrap();
    assert_eq!(count, 3);

    let mut sys = MaterialSystem::new();
    let loaded = sys.load_directory(dir.path()).unwrap();
    assert_eq!(loaded, 3);
    assert_eq!(sys.count(), 3);

    assert!(sys.get("Brick Red").is_some());
    assert!(sys.get("Glass Clear").is_some());
    assert!(sys.get("Steel").is_some());
}

#[test]
fn material_system_reload_updates_material() {
    let dir = tempfile::tempdir().unwrap();
    create_default_materials(dir.path()).unwrap();

    let mut sys = MaterialSystem::new();
    sys.load_directory(dir.path()).unwrap();

    let original_roughness = sys.get("Brick Red").unwrap().properties.roughness;
    assert!((original_roughness - 0.8).abs() < 1e-6);

    // Modify the file on disk
    let mut mat = sys.get("Brick Red").unwrap().clone();
    mat.properties.roughness = 0.3;
    let path = dir.path().join("brick_red.toml");
    mat.save(&path).unwrap();

    // Reload and check
    sys.reload("Brick Red").unwrap();
    let updated = sys.get("Brick Red").unwrap();
    assert!((updated.properties.roughness - 0.3).abs() < 1e-6);
}

#[test]
fn material_system_create_default_materials_creates_files() {
    let dir = tempfile::tempdir().unwrap();
    let count = create_default_materials(dir.path()).unwrap();
    assert_eq!(count, 3);

    let entries: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "toml")
                .unwrap_or(false)
        })
        .collect();
    assert_eq!(entries.len(), 3);

    // Verify each file is valid TOML that parses
    for entry in entries {
        let mat = MaterialDefinition::load(&entry.path()).unwrap();
        assert!(!mat.material.name.is_empty());
        assert_eq!(mat.material.mat_type, "spectral");
    }
}
