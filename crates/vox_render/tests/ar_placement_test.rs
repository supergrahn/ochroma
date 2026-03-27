use glam::Vec3;
use vox_render::ar_placement::*;

#[test]
fn detect_surfaces() {
    let mut session = ARSession::new();
    session.start();
    let id1 = session.detect_horizontal(Vec3::new(0.0, 0.75, 0.0), [0.6, 0.4]);
    let id2 = session.detect_vertical(Vec3::new(0.0, 1.0, -2.0), Vec3::NEG_Z, [1.0, 1.0]);

    assert_eq!(session.surface_count(), 2);
    assert!(session.get_surface(id1).is_some());
    assert!(session.get_surface(id2).is_some());
}

#[test]
fn horizontal_filter() {
    let mut session = ARSession::new();
    session.start();
    session.detect_horizontal(Vec3::ZERO, [1.0, 1.0]);
    session.detect_vertical(Vec3::ZERO, Vec3::NEG_Z, [1.0, 1.0]);
    session.detect_horizontal(Vec3::Y, [0.5, 0.5]);

    assert_eq!(session.horizontal_surfaces().len(), 2);
}

#[test]
fn largest_horizontal() {
    let mut session = ARSession::new();
    session.start();
    let small = session.detect_horizontal(Vec3::ZERO, [0.3, 0.3]);
    let big = session.detect_horizontal(Vec3::Y, [1.0, 0.8]);

    let largest = session.largest_horizontal().unwrap();
    assert_eq!(largest.id, big);
    assert_ne!(largest.id, small);
}

#[test]
fn place_on_horizontal_surface() {
    let surface = ARSurface::new_horizontal(1, Vec3::new(0.0, 0.75, 0.0), [1.0, 1.0]);
    // Object floating above the table.
    let placed = place_on_surface(&surface, Vec3::new(0.2, 5.0, -0.1));
    // Should be snapped to the table's Y height.
    assert!((placed.y - 0.75).abs() < 1e-5);
    // X and Z preserved.
    assert!((placed.x - 0.2).abs() < 1e-5);
    assert!((placed.z - (-0.1)).abs() < 1e-5);
}

#[test]
fn place_on_vertical_surface() {
    let surface = ARSurface::new_vertical(1, Vec3::new(0.0, 1.0, -2.0), Vec3::NEG_Z, [1.0, 1.0]);
    let placed = place_on_surface(&surface, Vec3::new(0.5, 1.5, 0.0));
    // Should snap Z to the wall's Z.
    assert!((placed.z - (-2.0)).abs() < 1e-5);
}

#[test]
fn city_projection_scaling() {
    let surface = ARSurface::new_horizontal(1, Vec3::new(0.0, 0.75, 0.0), [0.5, 0.5]);
    // City is 1000m x 1000m.
    let city_min = Vec3::new(-500.0, 0.0, -500.0);
    let city_max = Vec3::new(500.0, 100.0, 500.0);

    let proj = project_city_to_table(&surface, city_min, city_max);

    // Table is 1m x 1m (extent 0.5 each side).
    // City is 1000m wide => scale = 1.0 / 1000.0 = 0.001.
    assert!((proj.scale - 0.001).abs() < 1e-6);
    assert_eq!(proj.centre, surface.position);
}

#[test]
fn city_projection_non_square() {
    let surface = ARSurface::new_horizontal(1, Vec3::ZERO, [0.3, 0.6]);
    // City 600m x 300m.
    let city_min = Vec3::new(0.0, 0.0, 0.0);
    let city_max = Vec3::new(600.0, 50.0, 300.0);

    let proj = project_city_to_table(&surface, city_min, city_max);

    // Table: 0.6m x 1.2m. City: 600m x 300m.
    // scale_x = 0.6/600 = 0.001, scale_z = 1.2/300 = 0.004.
    // Uniform scale = min(0.001, 0.004) = 0.001.
    assert!((proj.scale - 0.001).abs() < 1e-6);
}

#[test]
fn session_stop_clears() {
    let mut session = ARSession::new();
    session.start();
    session.detect_horizontal(Vec3::ZERO, [1.0, 1.0]);
    assert_eq!(session.surface_count(), 1);

    session.stop();
    assert_eq!(session.surface_count(), 0);
    assert!(!session.active);
}

#[test]
fn surface_area() {
    let surface = ARSurface::new_horizontal(1, Vec3::ZERO, [0.5, 0.3]);
    // Area = 1.0 * 0.6 = 0.6.
    assert!((surface.area() - 0.6).abs() < 1e-6);
}
