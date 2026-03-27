use vox_terrain::foliage::*;
use vox_terrain::heightmap::*;

#[test]
fn scatter_on_flat_terrain() {
    let hm = Heightmap::flat(100, 100, 1.0, 2.0);
    let rules = vec![FoliageRule {
        name: "Tree".into(),
        asset_path: "tree.ply".into(),
        density: 10.0,
        min_height: 0.0,
        max_height: 100.0,
        max_slope: 90.0,
        min_scale: 1.0,
        max_scale: 1.0,
        random_rotation: false,
        cluster_radius: 0.0,
    }];
    let instances = scatter_foliage(&hm, &rules, 42);
    assert!(!instances.is_empty(), "Should place some foliage");
    // 100x100 = 10000m², density 10/100m² = 1000 expected
    assert!(
        instances.len() > 500 && instances.len() < 1500,
        "Expected ~1000, got {}",
        instances.len()
    );
}

#[test]
fn foliage_respects_height_limits() {
    let mut data = vec![0.0f32; 100];
    // Half the terrain at 0m, half at 20m
    for i in 50..100 {
        data[i] = 20.0;
    }
    let hm = Heightmap::from_data(10, 10, data, 1.0);
    let rules = vec![FoliageRule {
        name: "Low".into(),
        asset_path: "tree.ply".into(),
        density: 50.0,
        min_height: -1.0,
        max_height: 1.0,
        max_slope: 90.0,
        min_scale: 1.0,
        max_scale: 1.0,
        random_rotation: false,
        cluster_radius: 0.0,
    }];
    let instances = scatter_foliage(&hm, &rules, 42);
    for inst in &instances {
        assert!(inst.position[1] <= 1.0, "Should only place on low ground");
    }
}

#[test]
fn foliage_is_deterministic() {
    let hm = Heightmap::flat(50, 50, 1.0, 1.0);
    let rules = default_foliage_rules();
    let a = scatter_foliage(&hm, &rules, 42);
    let b = scatter_foliage(&hm, &rules, 42);
    assert_eq!(a.len(), b.len());
}

#[test]
fn scale_varies_within_range() {
    let hm = Heightmap::flat(50, 50, 1.0, 1.0);
    let rules = vec![FoliageRule {
        name: "Tree".into(),
        asset_path: "t.ply".into(),
        density: 20.0,
        min_height: -100.0,
        max_height: 100.0,
        max_slope: 90.0,
        min_scale: 0.5,
        max_scale: 2.0,
        random_rotation: false,
        cluster_radius: 0.0,
    }];
    let instances = scatter_foliage(&hm, &rules, 42);
    let min_s = instances.iter().map(|i| i.scale).fold(f32::MAX, f32::min);
    let max_s = instances.iter().map(|i| i.scale).fold(f32::MIN, f32::max);
    assert!(min_s >= 0.5 && max_s <= 2.0);
    assert!(max_s > min_s, "Should have scale variation");
}

#[test]
fn default_rules_have_entries() {
    assert!(default_foliage_rules().len() >= 3);
}
