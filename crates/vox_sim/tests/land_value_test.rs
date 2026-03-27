use vox_sim::land_value::LandValueGrid;

#[test]
fn service_increases_nearby_value() {
    let mut grid = LandValueGrid::new(100, 100, 10.0);
    grid.recalculate(&[(0.0, 0.0, 200.0)], &[]);
    let center = grid.sample(0.0, 0.0);
    let far = grid.sample(400.0, 400.0);
    assert!(center > far, "Near service should have higher value");
}

#[test]
fn park_increases_value() {
    let mut grid = LandValueGrid::new(100, 100, 10.0);
    grid.recalculate(&[], &[(0.0, 0.0)]);
    assert!(grid.sample(50.0, 50.0) > grid.sample(400.0, 400.0));
}
