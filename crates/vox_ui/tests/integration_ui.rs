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
fn live_u16_band_energy_drives_taller_bar_through_render() {
    // Simulate a live quantized band-energy readback (e.g. GPU/wire): band 9
    // is full-scale, band 2 is near-zero. Feed it through the u16 path and
    // confirm the rendered geometry reflects the input energy ordering.
    let mut bands = [0u16; 16];
    bands[2] = 512;          // low energy
    bands[9] = u16::MAX;     // full energy

    let bars = SpectralHUD::bar_rects_u16(bands, [16.0, 900.0], 160.0, 64.0);
    assert_eq!(bars.len(), 16, "expected 16 bars, got {}", bars.len());
    assert!(
        bars[9][3] > bars[2][3],
        "live u16: band9 h={} should be taller than band2 h={}",
        bars[9][3], bars[2][3],
    );

    // The same live input must reach pixels: build a cache from u16 and render.
    let cache = SpectralRadianceCache::from_u16(bands);
    let mut ctx = VelloCtxCpu::new(1920, 1080);
    ctx.begin_frame();
    SpectralHUD::render_cpu(&mut ctx, &cache, [16.0, 900.0]);
    assert!(
        ctx.commands().len() >= 32,
        "expected ≥32 draw commands (16 bg + 16 bars), got {}",
        ctx.commands().len(),
    );

    // render_cpu emits, per band, a background track followed by the energy
    // bar (foreground). Foreground bars are therefore at odd command indices.
    // Extract band 9 vs band 2 foreground heights from the real command stream.
    let fg_h = |band: usize| -> f32 {
        match &ctx.commands()[band * 2 + 1] {
            DrawCmd::FillRect { rect, .. } => rect[3],
        }
    };
    let h_full = fg_h(9);
    let h_low  = fg_h(2);
    println!("rendered band9_fg_h={} band2_fg_h={}", h_full, h_low);
    assert!(h_full > h_low, "rendered full-scale band9 h={} must exceed low band2 h={}", h_full, h_low);
    // render_cpu uses a fixed 60px max height; full-scale energy fills it.
    assert!((h_full - 60.0).abs() < 1e-3, "full-scale band should fill render_cpu max_height (60px), got {}", h_full);
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
