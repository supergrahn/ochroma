/// WebGPU-compatible render configuration for browser deployment.
#[derive(Debug, Clone)]
pub struct WebRenderConfig {
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub pixel_ratio: f32,
    pub max_splats: usize,
    pub enable_post_processing: bool,
    pub target_fps: u32,
}

impl Default for WebRenderConfig {
    fn default() -> Self {
        Self {
            canvas_width: 1280,
            canvas_height: 720,
            pixel_ratio: 1.0,
            max_splats: 500_000, // conservative for web
            enable_post_processing: false,
            target_fps: 30,
        }
    }
}

impl WebRenderConfig {
    /// Internal render resolution accounting for pixel ratio.
    pub fn physical_width(&self) -> u32 {
        (self.canvas_width as f32 * self.pixel_ratio) as u32
    }

    pub fn physical_height(&self) -> u32 {
        (self.canvas_height as f32 * self.pixel_ratio) as u32
    }

    /// Estimate VRAM usage for the configured splat count.
    pub fn estimated_vram_mb(&self) -> f32 {
        // 64 bytes per GpuSplatData + frame buffer + depth buffer
        let splat_vram = self.max_splats as f32 * 64.0 / (1024.0 * 1024.0);
        let fb_vram = self.physical_width() as f32 * self.physical_height() as f32 * 8.0
            / (1024.0 * 1024.0);
        splat_vram + fb_vram
    }

    /// Budget check: can this config run on the given VRAM budget?
    pub fn fits_vram_budget(&self, budget_mb: f32) -> bool {
        self.estimated_vram_mb() < budget_mb
    }
}

/// Platform detection for rendering backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    NativeDesktop,
    NativeConsole,
    WebBrowser,
    Mobile,
    CloudStreaming,
}

impl Platform {
    /// Recommended render config for this platform.
    pub fn recommended_config(&self) -> WebRenderConfig {
        match self {
            Self::NativeDesktop => WebRenderConfig {
                canvas_width: 1920,
                canvas_height: 1080,
                pixel_ratio: 1.0,
                max_splats: 5_000_000,
                enable_post_processing: true,
                target_fps: 60,
            },
            Self::NativeConsole => WebRenderConfig {
                canvas_width: 3840,
                canvas_height: 2160,
                pixel_ratio: 1.0,
                max_splats: 10_000_000,
                enable_post_processing: true,
                target_fps: 60,
            },
            Self::WebBrowser => WebRenderConfig::default(),
            Self::Mobile => WebRenderConfig {
                canvas_width: 1080,
                canvas_height: 1920, // portrait
                pixel_ratio: 2.0,
                max_splats: 200_000,
                enable_post_processing: false,
                target_fps: 30,
            },
            Self::CloudStreaming => WebRenderConfig {
                canvas_width: 1920,
                canvas_height: 1080,
                pixel_ratio: 1.0,
                max_splats: 20_000_000,
                enable_post_processing: true,
                target_fps: 60,
            },
        }
    }

    /// Maximum VRAM budget for this platform (MB).
    pub fn vram_budget_mb(&self) -> f32 {
        match self {
            Self::NativeDesktop => 8192.0,     // 8 GB
            Self::NativeConsole => 12288.0,    // 12 GB (PS5)
            Self::WebBrowser => 2048.0,        // 2 GB (conservative)
            Self::Mobile => 1024.0,            // 1 GB
            Self::CloudStreaming => 16384.0,    // 16 GB
        }
    }
}
