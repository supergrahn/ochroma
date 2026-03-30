use glam::Vec3;
use vox_app::building_placement::{BuildingPlacer, BuildingTemplate};

fn flat_terrain(_x: f32, _z: f32) -> f32 { 0.0 }

fn sloped_terrain(_x: f32, z: f32) -> f32 { z * 0.5 }

fn house_template() -> BuildingTemplate {
    BuildingTemplate {
        name: "House".to_string(),
        footprint: [4.0, 4.0],
        max_slope_deg: 15.0,
        asset_path: "assets/buildings/house.vxm".to_string(),
    }
}

#[test]
fn building_placement_snaps_to_grid_and_validates() {
    let mut placer = BuildingPlacer::new(house_template());
    let ray_origin = Vec3::new(4.3, 10.0, 4.7);
    let ray_dir    = Vec3::new(0.0, -1.0, 0.0);
    placer.update(ray_origin, ray_dir, &flat_terrain);
    println!("ghost at [{:.1}, {:.1}, {:.1}] valid={}", placer.ghost_pos.x, placer.ghost_pos.y, placer.ghost_pos.z, placer.valid);
    assert_eq!(placer.ghost_pos.x, 4.0, "x must snap to 4.0");
    assert_eq!(placer.ghost_pos.z, 4.0, "z must snap to 4.0");
    assert!(placer.valid, "flat terrain must be valid");
}

#[test]
fn building_placement_rejects_steep_slope() {
    let mut placer = BuildingPlacer::new(house_template());
    let ray_origin = Vec3::new(4.0, 10.0, 4.0);
    let ray_dir    = Vec3::new(0.0, -1.0, 0.0);
    placer.update(ray_origin, ray_dir, &sloped_terrain);
    println!("valid={} slope_deg={:.1}", placer.valid, placer.slope_deg);
    assert!(!placer.valid, "slope > 15 deg must be invalid");
    assert!(placer.slope_deg > 15.0, "slope_deg must be computed, got {}", placer.slope_deg);
}

#[test]
fn building_placement_confirm_returns_building_when_valid() {
    let mut placer = BuildingPlacer::new(house_template());
    placer.update(Vec3::new(2.0, 10.0, 2.0), Vec3::new(0.0, -1.0, 0.0), &flat_terrain);
    let result = placer.confirm();
    assert!(result.is_some(), "confirm must return Some on valid placement");
    let building = result.unwrap();
    assert_eq!(building.position.x, 2.0);
    assert_eq!(building.template.name, "House");
}

#[test]
fn building_placement_confirm_returns_none_when_invalid() {
    let mut placer = BuildingPlacer::new(house_template());
    placer.update(Vec3::new(4.0, 10.0, 4.0), Vec3::new(0.0, -1.0, 0.0), &sloped_terrain);
    assert!(!placer.valid);
    let result = placer.confirm();
    assert!(result.is_none(), "confirm must return None on invalid placement");
}
