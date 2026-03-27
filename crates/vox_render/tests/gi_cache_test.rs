use vox_render::gi_cache::*;
use vox_core::spectral::SpectralBands;
use glam::Vec3;

#[test]
fn first_bounce_modulates_by_surface() {
    let mut cache = GICache::new(1.0, 2);
    let surface = SpectralBands([0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]); // red surface
    let light = SpectralBands([1.0; 8]); // white light
    cache.compute_first_bounce(Vec3::ZERO, &surface, &light, 0);

    let cell = cache.query(Vec3::ZERO).unwrap();
    // Reflected should be reddish (high first band, low others)
    assert!(cell.incoming.0[0] > cell.incoming.0[1], "Red surface should reflect more red");
}

#[test]
fn second_bounce_colour_bleeds() {
    let mut cache = GICache::new(1.0, 2);
    // Red wall at origin emits red light
    cache.add_bounce(Vec3::ZERO, SpectralBands([0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), 0);

    // White surface nearby receives colour bleed
    let white = SpectralBands([1.0; 8]);
    cache.compute_second_bounce(Vec3::new(1.5, 0.0, 0.0), &white, 3.0, 0);

    let cell = cache.query(Vec3::new(1.5, 0.0, 0.0)).unwrap();
    assert!(cell.incoming.0[0] > 0.0, "Nearby white surface should receive red bleed");
}

#[test]
fn evict_stale_cells() {
    let mut cache = GICache::new(1.0, 2);
    cache.add_bounce(Vec3::ZERO, SpectralBands([0.5; 8]), 0);
    cache.add_bounce(Vec3::new(10.0, 0.0, 0.0), SpectralBands([0.5; 8]), 100);
    assert_eq!(cache.cell_count(), 2);
    cache.evict_stale(100, 50);
    assert_eq!(cache.cell_count(), 1, "Stale cell should be evicted");
}
