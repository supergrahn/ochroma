//! Viewport mode for per-band spectral visualization.

/// Names for the 8 spectral bands as shown in the HUD.
/// Bands are ordered high-frequency (blue, electric) to low-frequency (red, bass).
pub const BAND_NAMES: [&str; 8] = [
    "Band 0 — 8 kHz (blue/electric)",
    "Band 1 — 4 kHz",
    "Band 2 — 2 kHz",
    "Band 3 — 1 kHz",
    "Band 4 — 500 Hz",
    "Band 5 — 250 Hz",
    "Band 6 — 125 Hz",
    "Band 7 — 80 Hz (red/bass)",
];

/// Controls which render path the engine viewport uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpectralViewportMode {
    /// Normal full-color EWA rendering (default).
    #[default]
    Full,
    /// False-color heatmap of a single spectral band (0–7).
    Band(usize),
}

impl SpectralViewportMode {
    /// Cycle to the next mode: Full → Band(0) → Band(1) → … → Band(7) → Full.
    pub fn cycle_next(self) -> Self {
        match self {
            Self::Full => Self::Band(0),
            Self::Band(b) if b < 7 => Self::Band(b + 1),
            Self::Band(_) => Self::Full,
        }
    }

    /// Short label for the HUD.
    pub fn label(self) -> &'static str {
        match self {
            Self::Full => "Viewport: Full Color",
            Self::Band(b) => BAND_NAMES[b.min(7)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_full_to_band0() {
        assert_eq!(SpectralViewportMode::Full.cycle_next(), SpectralViewportMode::Band(0));
    }

    #[test]
    fn cycle_band7_wraps_to_full() {
        assert_eq!(SpectralViewportMode::Band(7).cycle_next(), SpectralViewportMode::Full);
    }

    #[test]
    fn cycle_band3_to_band4() {
        assert_eq!(SpectralViewportMode::Band(3).cycle_next(), SpectralViewportMode::Band(4));
    }

    #[test]
    fn label_full() {
        assert_eq!(SpectralViewportMode::Full.label(), "Viewport: Full Color");
    }

    #[test]
    fn label_band0_contains_8khz() {
        assert!(SpectralViewportMode::Band(0).label().contains("8 kHz"));
    }

    #[test]
    fn label_band7_contains_bass() {
        assert!(SpectralViewportMode::Band(7).label().contains("bass"));
    }

    #[test]
    fn default_is_full() {
        assert_eq!(SpectralViewportMode::default(), SpectralViewportMode::Full);
    }
}
