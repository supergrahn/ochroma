use glam::Vec3;
use vox_render::spatial_ui::{PanelContent, SpatialPanel, SpatialUIManager};

#[test]
fn ray_hits_panel_directly() {
    let panel = SpatialPanel::new(1, Vec3::new(0.0, 2.0, -5.0), [2.0, 1.0], PanelContent::Text);
    // Ray from origin looking straight at the panel.
    let t = panel.ray_intersect(Vec3::ZERO, Vec3::new(0.0, 2.0, -5.0).normalize());
    assert!(t.is_some());
}

#[test]
fn ray_misses_panel() {
    let panel = SpatialPanel::new(1, Vec3::new(0.0, 2.0, -5.0), [1.0, 1.0], PanelContent::Chart);
    // Ray going away from the panel.
    let t = panel.ray_intersect(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
    assert!(t.is_none());
}

#[test]
fn ray_misses_outside_bounds() {
    let panel = SpatialPanel::new(1, Vec3::new(0.0, 2.0, -5.0), [0.1, 0.1], PanelContent::Text);
    // Ray that hits the plane but outside the small panel bounds.
    let t = panel.ray_intersect(Vec3::ZERO, Vec3::new(1.0, 2.0, -5.0).normalize());
    assert!(t.is_none());
}

#[test]
fn manager_add_remove() {
    let mut mgr = SpatialUIManager::new();
    let id1 = mgr.add_panel(Vec3::ZERO, [1.0, 1.0], PanelContent::Text);
    let id2 = mgr.add_panel(Vec3::ONE, [1.0, 1.0], PanelContent::Buttons);
    assert_eq!(mgr.count(), 2);

    assert!(mgr.remove_panel(id1));
    assert_eq!(mgr.count(), 1);
    assert!(mgr.get(id2).is_some());
    assert!(mgr.get(id1).is_none());
}

#[test]
fn manager_ray_intersect_nearest() {
    let mut mgr = SpatialUIManager::new();
    // Near panel at z=-2, far panel at z=-5.
    let near = mgr.add_panel(Vec3::new(0.0, 0.0, -2.0), [2.0, 2.0], PanelContent::Text);
    let _far = mgr.add_panel(Vec3::new(0.0, 0.0, -5.0), [2.0, 2.0], PanelContent::Chart);

    let hit = mgr.ray_intersect(Vec3::ZERO, Vec3::NEG_Z);
    assert!(hit.is_some());
    let (panel, _t) = hit.unwrap();
    assert_eq!(panel.id, near);
}

#[test]
fn manager_nearest_panel() {
    let mut mgr = SpatialUIManager::new();
    let _a = mgr.add_panel(Vec3::new(10.0, 0.0, 0.0), [1.0, 1.0], PanelContent::Text);
    let b = mgr.add_panel(Vec3::new(1.0, 0.0, 0.0), [1.0, 1.0], PanelContent::Texture);

    let closest = mgr.nearest_panel(Vec3::ZERO);
    assert!(closest.is_some());
    assert_eq!(closest.unwrap().id, b);
}

#[test]
fn hidden_panels_ignored() {
    let mut mgr = SpatialUIManager::new();
    let id = mgr.add_panel(Vec3::new(0.0, 0.0, -2.0), [4.0, 4.0], PanelContent::Text);
    mgr.get_mut(id).unwrap().visible = false;

    assert!(mgr.ray_intersect(Vec3::ZERO, Vec3::NEG_Z).is_none());
    assert!(mgr.nearest_panel(Vec3::ZERO).is_none());
}
