// crates/vox_ui/tests/integration_ui.rs
use vox_ui::{
    SpectralHUD,
    SpectralRadianceCache,
    vello_ctx::{DrawCmd, VelloCtxCpu},
};

#[test]
fn hud_renders_16_bands_into_cpu_context() {
    let cache = SpectralRadianceCache {
        band_energy: [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    };
    let mut ctx = VelloCtxCpu::new(1920, 1080);
    ctx.begin_frame();

    SpectralHUD::render_cpu(&mut ctx, &cache, [16.0, 900.0]);

    // 16 background tracks + 16 energy bars = 32 draw commands
    assert!(
        ctx.commands().len() >= 32,
        "expected ≥32 draw commands (16 bg + 16 bars), got {}",
        ctx.commands().len(),
    );
    // Verify the first command is a FillRect (background track)
    match &ctx.commands()[0] {
        DrawCmd::FillRect { rect, color } => {
            println!("first_cmd rect[0]={} color[3]={}", rect[0], color[3]);
            assert!(color[3] > 0.0, "background track should have non-zero alpha");
        }
    }
}

#[test]
fn high_energy_band_renders_taller_bar_than_low_energy() {
    let cache = SpectralRadianceCache {
        band_energy: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    };
    let bars = SpectralHUD::bar_rects(cache.band_energy, [0.0, 0.0], 160.0, 100.0);
    assert!(
        bars[15][3] > bars[0][3],
        "band15 h={} band0 h={}",
        bars[15][3],
        bars[0][3],
    );
}

#[test]
fn hud_bars_within_declared_region() {
    let origin = [20.0f32, 800.0f32];
    let max_h  = 60.0;
    let total_w = 160.0;
    let cache  = SpectralRadianceCache { band_energy: [0.5; 16] };
    let bars   = SpectralHUD::bar_rects(cache.band_energy, origin, total_w, max_h);

    for bar in &bars {
        assert!(bar[0] >= origin[0],        "bar x left of origin: {}", bar[0]);
        assert!(bar[0] <= origin[0] + total_w, "bar x right of region: {}", bar[0]);
        assert!(bar[1] >= origin[1],        "bar y above origin: {}", bar[1]);
        assert!(bar[3] <= max_h + 1.0,      "bar height exceeds max: {}", bar[3]);
    }
}

#[test]
fn bar_colors_form_visible_spectrum_gradient() {
    let colors = SpectralHUD::band_colors();
    assert!(
        colors[0][2] > colors[0][0],
        "band 0 should be blue-dominant (violet), r={} b={}",
        colors[0][0], colors[0][2],
    );
    assert!(
        colors[15][0] > colors[15][2],
        "band 15 should be red-dominant, r={} b={}",
        colors[15][0], colors[15][2],
    );
}

#[test]
fn scene_spectral_shift_moves_hud_bars() {
    let fire_cache = SpectralRadianceCache {
        band_energy: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
    };
    let cold_cache = SpectralRadianceCache {
        band_energy: [1.0, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };

    let fire_bars = SpectralHUD::bar_rects(fire_cache.band_energy, [0.0, 0.0], 160.0, 100.0);
    let cold_bars = SpectralHUD::bar_rects(cold_cache.band_energy, [0.0, 0.0], 160.0, 100.0);

    println!("fire_band15_h={} fire_band0_h={}", fire_bars[15][3], fire_bars[0][3]);
    println!("cold_band0_h={} cold_band15_h={}", cold_bars[0][3], cold_bars[15][3]);

    assert!(fire_bars[15][3] > fire_bars[0][3], "fire: band15 should dominate");
    assert!(cold_bars[0][3] > cold_bars[15][3], "cold: band0 should dominate");
}
