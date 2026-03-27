use vox_render::web_renderer::*;

#[test]
fn default_web_config_fits_budget() {
    let config = WebRenderConfig::default();
    assert!(
        config.fits_vram_budget(2048.0),
        "Default web config should fit in 2GB VRAM"
    );
}

#[test]
fn physical_resolution_with_pixel_ratio() {
    let config = WebRenderConfig {
        canvas_width: 1080,
        canvas_height: 1920,
        pixel_ratio: 2.0,
        ..Default::default()
    };
    assert_eq!(config.physical_width(), 2160);
    assert_eq!(config.physical_height(), 3840);
}

#[test]
fn platform_configs_fit_their_budgets() {
    for platform in &[
        Platform::NativeDesktop,
        Platform::WebBrowser,
        Platform::Mobile,
        Platform::CloudStreaming,
    ] {
        let config = platform.recommended_config();
        assert!(
            config.fits_vram_budget(platform.vram_budget_mb()),
            "{:?} config should fit its own VRAM budget",
            platform
        );
    }
}

#[test]
fn cloud_has_highest_splat_budget() {
    let cloud = Platform::CloudStreaming.recommended_config();
    let web = Platform::WebBrowser.recommended_config();
    assert!(cloud.max_splats > web.max_splats);
}
