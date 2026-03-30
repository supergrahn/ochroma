//! Per-platform quality profiles and capability detection.
//!
//! Defines memory budgets, splat counts, shadow atlas sizes, and feature flags
//! for Desktop/Mobile/Web/Console targets. Lets the engine auto-configure to
//! the best quality tier at startup.

use crate::web_renderer::Platform;

/// Quality tier used to select a canonical `RenderBudget`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformTier {
    Ultra,
    High,
    Medium,
    Low,
}

/// Memory and feature budget for a given quality tier.
#[derive(Debug, Clone)]
pub struct RenderBudget {
    pub max_splats: u32,
    pub vram_mb: u32,
    pub shadow_atlas_size: u32,
    pub gi_probe_count: u32,
    pub enable_volumetrics: bool,
    pub enable_oit: bool,
    /// Number of spectral bands: 8 for Desktop, 4 for Mobile/Web.
    pub spectral_bands: u8,
}

impl RenderBudget {
    /// Returns the canonical budget for the given tier.
    pub fn for_tier(tier: PlatformTier) -> Self {
        match tier {
            PlatformTier::Ultra => Self {
                max_splats: 2_000_000,
                vram_mb: 8192,
                shadow_atlas_size: 4096,
                gi_probe_count: 256,
                enable_volumetrics: true,
                enable_oit: true,
                spectral_bands: 8,
            },
            PlatformTier::High => Self {
                max_splats: 800_000,
                vram_mb: 4096,
                shadow_atlas_size: 2048,
                gi_probe_count: 64,
                enable_volumetrics: true,
                enable_oit: true,
                spectral_bands: 8,
            },
            PlatformTier::Medium => Self {
                max_splats: 300_000,
                vram_mb: 2048,
                shadow_atlas_size: 1024,
                gi_probe_count: 16,
                enable_volumetrics: false,
                enable_oit: false,
                spectral_bands: 8,
            },
            PlatformTier::Low => Self {
                max_splats: 100_000,
                vram_mb: 512,
                shadow_atlas_size: 512,
                gi_probe_count: 0,
                enable_volumetrics: false,
                enable_oit: false,
                spectral_bands: 4,
            },
        }
    }
}

/// Hardware and input capability flags for a given platform.
#[derive(Debug, Clone)]
pub struct PlatformCapabilities {
    pub compute_shaders: bool,
    pub raytracing: bool,
    pub hdr_output: bool,
    pub haptic_feedback: bool,
    pub touch_input: bool,
    pub gamepad_input: bool,
    pub max_texture_size: u32,
}

impl PlatformCapabilities {
    /// Returns sensible default capabilities for the given platform.
    pub fn for_platform(platform: &Platform) -> Self {
        match platform {
            Platform::NativeDesktop => Self {
                compute_shaders: true,
                raytracing: false, // unknown until detected at runtime
                hdr_output: true,
                haptic_feedback: false,
                touch_input: false,
                gamepad_input: true,
                max_texture_size: 16384,
            },
            Platform::WebBrowser => Self {
                compute_shaders: true, // WebGPU
                raytracing: false,
                hdr_output: false,
                haptic_feedback: false,
                touch_input: true,
                gamepad_input: true,
                max_texture_size: 4096,
            },
            Platform::Mobile => Self {
                compute_shaders: false,
                raytracing: false,
                hdr_output: false,
                haptic_feedback: true,
                touch_input: true,
                gamepad_input: false,
                max_texture_size: 4096,
            },
            Platform::NativeConsole => Self {
                compute_shaders: true,
                raytracing: false,
                hdr_output: true,
                haptic_feedback: true,
                touch_input: false,
                gamepad_input: true,
                max_texture_size: 16384,
            },
            Platform::CloudStreaming => Self {
                compute_shaders: true,
                raytracing: false,
                hdr_output: false,
                haptic_feedback: false,
                touch_input: false,
                gamepad_input: true,
                max_texture_size: 4096,
            },
        }
    }
}

/// Specific console hardware target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleTarget {
    PS5,
    XboxSeriesX,
    Switch,
}

/// Returns the canonical `RenderBudget` for a specific console target.
pub fn console_budget(target: ConsoleTarget) -> RenderBudget {
    match target {
        ConsoleTarget::PS5 => RenderBudget::for_tier(PlatformTier::Ultra),
        ConsoleTarget::XboxSeriesX => RenderBudget::for_tier(PlatformTier::Ultra),
        ConsoleTarget::Switch => RenderBudget {
            max_splats: 150_000,
            vram_mb: 3000,
            ..RenderBudget::for_tier(PlatformTier::Low)
        },
    }
}

/// HDR display configuration for a specific output target.
pub struct PlatformHdrConfig {
    pub peak_nits: f32,
    pub paper_white_nits: f32,
    pub min_nits: f32,
}

impl PlatformHdrConfig {
    /// HDR config for PlayStation 5.
    pub fn ps5() -> Self {
        Self {
            peak_nits: 2000.0,
            paper_white_nits: 200.0,
            min_nits: 0.001,
        }
    }

    /// HDR config for Xbox Series X.
    pub fn xbox_series_x() -> Self {
        Self {
            peak_nits: 1000.0,
            paper_white_nits: 203.0,
            min_nits: 0.001,
        }
    }

    /// SDR fallback when HDR is unavailable.
    pub fn sdr_fallback() -> Self {
        Self {
            peak_nits: 100.0,
            paper_white_nits: 80.0,
            min_nits: 0.0,
        }
    }
}

/// Recommends the best `PlatformTier` given available VRAM and whether the
/// device is a mobile target.
///
/// Mobile devices are capped at `Medium` regardless of VRAM.
pub fn recommend_tier(available_vram_mb: u32, is_mobile: bool) -> PlatformTier {
    let tier = if available_vram_mb >= 6000 {
        PlatformTier::Ultra
    } else if available_vram_mb >= 3000 {
        PlatformTier::High
    } else if available_vram_mb >= 1500 {
        PlatformTier::Medium
    } else {
        PlatformTier::Low
    };

    if is_mobile {
        // Cap mobile at Medium
        match tier {
            PlatformTier::Ultra | PlatformTier::High => PlatformTier::Medium,
            other => other,
        }
    } else {
        tier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ultra_budget_has_most_splats() {
        let ultra = RenderBudget::for_tier(PlatformTier::Ultra);
        let high = RenderBudget::for_tier(PlatformTier::High);
        let medium = RenderBudget::for_tier(PlatformTier::Medium);
        let low = RenderBudget::for_tier(PlatformTier::Low);
        assert!(ultra.max_splats > high.max_splats);
        assert!(high.max_splats > medium.max_splats);
        assert!(medium.max_splats > low.max_splats);
    }

    #[test]
    fn low_budget_has_fewer_spectral_bands() {
        let low = RenderBudget::for_tier(PlatformTier::Low);
        assert_eq!(low.spectral_bands, 4);
    }

    #[test]
    fn recommend_mobile_is_capped() {
        let tier = recommend_tier(8000, true);
        assert!(tier == PlatformTier::Medium || tier == PlatformTier::Low);
    }

    #[test]
    fn recommend_high_vram_is_ultra() {
        let tier = recommend_tier(8000, false);
        assert_eq!(tier, PlatformTier::Ultra);
    }

    #[test]
    fn ps5_budget_matches_ultra() {
        let ps5 = console_budget(ConsoleTarget::PS5);
        let ultra = RenderBudget::for_tier(PlatformTier::Ultra);
        assert_eq!(ps5.max_splats, ultra.max_splats);
    }

    #[test]
    fn hdr_ps5_higher_nits_than_sdr() {
        let ps5 = PlatformHdrConfig::ps5();
        let sdr = PlatformHdrConfig::sdr_fallback();
        assert!(ps5.peak_nits > sdr.peak_nits);
    }
}
