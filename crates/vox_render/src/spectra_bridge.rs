use std::path::PathBuf;

/// Rendering backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderBackend {
    /// Built-in wgpu Gaussian splat rasteriser.
    BuiltIn,
    /// Spectra path tracer (high quality, requires Spectra installation).
    SpectraPathTracer,
    /// Spectra real-time mode (denoised, interactive).
    SpectraRealtime,
    /// CPU software rasteriser (fallback).
    Software,
}

/// Render quality presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityPreset {
    Low,       // software rasteriser, no post-processing
    Medium,    // built-in wgpu, basic post-processing
    High,      // built-in wgpu, full post-processing + denoiser
    Ultra,     // Spectra real-time with AI denoiser
    Cinematic, // Spectra path tracer (offline quality)
}

impl QualityPreset {
    pub fn backend(&self) -> RenderBackend {
        match self {
            Self::Low => RenderBackend::Software,
            Self::Medium | Self::High => RenderBackend::BuiltIn,
            Self::Ultra => RenderBackend::SpectraRealtime,
            Self::Cinematic => RenderBackend::SpectraPathTracer,
        }
    }

    pub fn enable_denoiser(&self) -> bool {
        matches!(self, Self::High | Self::Ultra | Self::Cinematic)
    }

    pub fn enable_shadows(&self) -> bool {
        matches!(self, Self::Medium | Self::High | Self::Ultra | Self::Cinematic)
    }

    pub fn enable_post_processing(&self) -> bool {
        !matches!(self, Self::Low)
    }

    pub fn enable_particles(&self) -> bool {
        !matches!(self, Self::Low)
    }

    pub fn max_visible_splats(&self) -> usize {
        match self {
            Self::Low => 100_000,
            Self::Medium => 1_000_000,
            Self::High => 5_000_000,
            Self::Ultra => 20_000_000,
            Self::Cinematic => 50_000_000,
        }
    }
}

/// Configuration for the render pipeline.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub quality: QualityPreset,
    pub resolution: (u32, u32),
    pub vsync: bool,
    pub fov: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub gamma: f32,
    pub exposure: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            quality: QualityPreset::High,
            resolution: (1920, 1080),
            vsync: true,
            fov: std::f32::consts::FRAC_PI_4,
            near_plane: 0.1,
            far_plane: 10000.0,
            gamma: 2.2,
            exposure: 1.0,
        }
    }
}

/// Spectra process handle (for Option B: subprocess rendering).
pub struct SpectraProcess {
    pub available: bool,
    pub spectra_path: Option<PathBuf>,
}

impl SpectraProcess {
    pub fn detect() -> Self {
        // Check if Spectra is available at the expected path
        let spectra_path = PathBuf::from(
            std::env::var("SPECTRA_PATH").unwrap_or_else(|_| {
                dirs_next::home_dir()
                    .map(|h| {
                        h.join("git/aetherspectra/spectra")
                            .to_string_lossy()
                            .into_owned()
                    })
                    .unwrap_or_default()
            }),
        );

        let available = spectra_path.join("pyproject.toml").exists();
        if available {
            println!(
                "[ochroma] Spectra renderer detected at {}",
                spectra_path.display()
            );
        } else {
            println!("[ochroma] Spectra not found, using built-in renderer");
        }

        Self {
            available,
            spectra_path: if available { Some(spectra_path) } else { None },
        }
    }

    pub fn is_available(&self) -> bool {
        self.available
    }
}
