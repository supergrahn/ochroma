//! SceneQualityReport — iterative director feedback loop.
//! Evaluates rendered scene quality across spectral metrics and outputs
//! actionable feedback for the AssetDirector to iterate on.

use serde::{Deserialize, Serialize};

/// Per-band spectral coverage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandCoverage {
    /// Which spectral band index (0..15).
    pub band: usize,
    /// Mean energy across all evaluated splats.
    pub mean_energy: f32,
    /// Fraction of splats with energy above 0.1 in this band.
    pub coverage_fraction: f32,
}

/// Overall quality grade for a rendered scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityGrade {
    /// All checks pass — asset is ready.
    Excellent,
    /// Minor issues — renderable but could improve.
    Acceptable,
    /// Significant defects — needs director feedback loop.
    NeedsWork,
    /// Severe problems — discard and regenerate.
    Failed,
}

impl QualityGrade {
    /// Returns true when the asset passes the acceptance threshold.
    pub fn is_acceptable(self) -> bool {
        matches!(self, QualityGrade::Excellent | QualityGrade::Acceptable)
    }
}

/// Complete quality evaluation for a scene or asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneQualityReport {
    /// Number of splats evaluated.
    pub splat_count: usize,
    /// Mean energy across all bands and splats.
    pub mean_total_energy: f32,
    /// Per-band coverage breakdown.
    pub band_coverage: Vec<BandCoverage>,
    /// Overall quality grade.
    pub grade: QualityGrade,
    /// Human-readable feedback for the director.
    pub feedback: Vec<String>,
}

impl SceneQualityReport {
    /// Evaluate a set of per-splat spectral arrays (each is 16 bands).
    /// Returns a report with coverage analysis and actionable feedback.
    pub fn evaluate(splat_spectra: &[[f32; 16]]) -> Self {
        let splat_count = splat_spectra.len();
        if splat_count == 0 {
            return Self {
                splat_count: 0,
                mean_total_energy: 0.0,
                band_coverage: Vec::new(),
                grade: QualityGrade::Failed,
                feedback: vec!["No splats found — scene is empty.".into()],
            };
        }

        // Compute per-band stats
        let mut band_coverage = Vec::with_capacity(16);
        let mut total_energy_sum = 0.0f32;

        for band in 0..16 {
            let energies: Vec<f32> = splat_spectra.iter().map(|s| s[band]).collect();
            let mean = energies.iter().sum::<f32>() / splat_count as f32;
            let coverage =
                energies.iter().filter(|&&e| e > 0.1).count() as f32 / splat_count as f32;
            total_energy_sum += mean;
            band_coverage.push(BandCoverage { band, mean_energy: mean, coverage_fraction: coverage });
        }

        let mean_total_energy = total_energy_sum / 16.0;

        // Build feedback
        let mut feedback = Vec::new();
        let visible_bands = band_coverage.iter().filter(|b| b.coverage_fraction > 0.05).count();

        if visible_bands < 3 {
            feedback.push(format!(
                "Only {} spectral bands have >5% coverage — scene may appear monochromatic.",
                visible_bands
            ));
        }
        if mean_total_energy < 0.01 {
            feedback.push("Very low overall energy — scene may be too dark or splats unlit.".into());
        }
        if splat_count < 10 {
            feedback.push(format!(
                "Only {} splats — scene is sparse; consider increasing splat density.",
                splat_count
            ));
        }

        // Grade
        let grade = if feedback.is_empty() && visible_bands >= 6 {
            QualityGrade::Excellent
        } else if feedback.len() <= 1 && visible_bands >= 3 {
            QualityGrade::Acceptable
        } else if mean_total_energy > 0.0 {
            QualityGrade::NeedsWork
        } else {
            QualityGrade::Failed
        };

        Self { splat_count, mean_total_energy, band_coverage, grade, feedback }
    }

    /// Returns the index of the most energetic spectral band.
    pub fn dominant_band(&self) -> usize {
        self.band_coverage
            .iter()
            .max_by(|a, b| a.mean_energy.partial_cmp(&b.mean_energy).unwrap())
            .map(|b| b.band)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_broad_spectrum() -> [f32; 16] {
        let mut s = [0.0f32; 16];
        for (i, v) in s.iter_mut().enumerate() {
            *v = 0.3 + (i as f32) * 0.02;
        }
        s
    }

    #[test]
    fn empty_scene_is_failed() {
        let report = SceneQualityReport::evaluate(&[]);
        assert_eq!(report.grade, QualityGrade::Failed);
        assert!(!report.feedback.is_empty(), "failed report must explain why");
    }

    #[test]
    fn broad_spectrum_scene_passes() {
        let splats: Vec<[f32; 16]> = (0..100).map(|_| make_broad_spectrum()).collect();
        let report = SceneQualityReport::evaluate(&splats);
        assert!(
            report.grade.is_acceptable(),
            "broad spectrum scene must pass: {:?} feedback: {:?}",
            report.grade,
            report.feedback
        );
    }

    #[test]
    fn dominant_band_is_highest_energy() {
        let mut splats = vec![[0.1f32; 16]];
        splats[0][7] = 0.9;
        let report = SceneQualityReport::evaluate(&splats);
        assert_eq!(report.dominant_band(), 7, "band 7 must be dominant");
    }

    #[test]
    fn sparse_scene_gets_feedback() {
        let splats: Vec<[f32; 16]> = (0..5).map(|_| make_broad_spectrum()).collect();
        let report = SceneQualityReport::evaluate(&splats);
        let feedback_text = report.feedback.join(" ");
        assert!(
            feedback_text.contains("splat") || !report.grade.is_acceptable(),
            "sparse scene must produce splat-count feedback or non-acceptable grade"
        );
    }
}
