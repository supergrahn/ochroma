use vox_data::asset_catalog::*;

#[test]
fn default_catalog_has_entries() {
    let catalog = default_catalog();
    assert!(
        catalog.len() >= 15,
        "Should have many asset types, got {}",
        catalog.len()
    );
}

#[test]
fn catalog_has_all_categories() {
    let catalog = default_catalog();
    assert!(catalog
        .iter()
        .any(|e| e.category == AssetCategory::ResidentialBuilding));
    assert!(catalog
        .iter()
        .any(|e| e.category == AssetCategory::CommercialBuilding));
    assert!(catalog
        .iter()
        .any(|e| e.category == AssetCategory::IndustrialBuilding));
    assert!(catalog.iter().any(|e| e.category == AssetCategory::Tree));
    assert!(catalog.iter().any(|e| e.category == AssetCategory::Prop));
}

#[test]
fn generate_victorian_house() {
    let catalog = default_catalog();
    let victorian = catalog
        .iter()
        .find(|e| e.name.contains("Victorian"))
        .unwrap();
    let splats = victorian.generate(42);
    assert!(!splats.is_empty());
    assert!(
        splats.len() > 500,
        "Victorian house should have many splats, got {}",
        splats.len()
    );
}

#[test]
fn generate_different_seeds_different_results() {
    let catalog = default_catalog();
    let entry = &catalog[0];
    let a = entry.generate(1);
    let b = entry.generate(2);
    // Same config but different seeds may produce different counts due to rng in width/floors
    let differs = a
        .iter()
        .zip(b.iter())
        .any(|(sa, sb)| sa.position != sb.position);
    assert!(
        differs,
        "Different seeds should produce different positions"
    );
}

#[test]
fn generate_all_catalog_entries() {
    let catalog = default_catalog();
    for (i, entry) in catalog.iter().enumerate() {
        let splats = entry.generate(i as u64);
        assert!(
            !splats.is_empty(),
            "Entry '{}' should produce splats",
            entry.name
        );
    }
}
