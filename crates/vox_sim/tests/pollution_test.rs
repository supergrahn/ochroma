use vox_sim::pollution::PollutionGrid;

#[test]
fn source_creates_pollution() {
    let mut grid = PollutionGrid::new(50, 50, 10.0);
    grid.add_source(0.0, 0.0, 100.0, 0.5, true);
    assert!(grid.air_at(0.0, 0.0) > 0.0);
    assert!(grid.air_at(0.0, 0.0) > grid.air_at(200.0, 200.0));
}

#[test]
fn decay_reduces_pollution() {
    let mut grid = PollutionGrid::new(50, 50, 10.0);
    grid.add_source(0.0, 0.0, 50.0, 0.8, true);
    let before = grid.average_air_pollution();
    grid.decay(0.1);
    let after = grid.average_air_pollution();
    assert!(after < before);
}

#[test]
fn diffusion_spreads_pollution() {
    let mut grid = PollutionGrid::new(20, 20, 10.0);
    grid.add_source(0.0, 0.0, 10.0, 1.0, true);
    let center_before = grid.air_at(0.0, 0.0);
    let edge_before = grid.air_at(50.0, 0.0);
    grid.diffuse(0.3);
    let center_after = grid.air_at(0.0, 0.0);
    let edge_after = grid.air_at(50.0, 0.0);
    // Center should decrease, edges should increase
    assert!(center_after <= center_before);
    assert!(edge_after >= edge_before);
}
