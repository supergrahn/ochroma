//! AAA Spec 10 — asset validation / import gate (core validator).
//!
//! A pure, allocation-bounded splat integrity + budget + spectral-validity lint
//! every importer can gate on. Catches the failure modes that silently corrupt a
//! scene: a NaN/Inf position (geometry vanishes or explodes), a non-positive
//! scale (a degenerate/inside-out Gaussian), an over-budget splat count, and the
//! WEDGE-SPECIFIC lint an RGB engine literally cannot express — non-finite or
//! all-zero 16-band spectral radiance (a splat that renders fine under one
//! illuminant but is undefined/black under another).
//!
//! Errors fail the gate; warnings (e.g. all-zero radiance) pass it but are
//! surfaced as receipts. No I/O, no mutation — a pure fn over an in-memory slice.

use vox_core::types::GaussianSplat;

/// Number of spectral bands per splat (16-band spectral model).
const BANDS: usize = 16;

/// Splat-count budget for an import. `UNLIMITED` imposes no cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidationBudget {
    pub max_splats: Option<usize>,
}

impl ValidationBudget {
    /// No splat-count cap.
    pub const UNLIMITED: ValidationBudget = ValidationBudget { max_splats: None };

    /// Cap the import at `max_splats`.
    pub fn with_max(max_splats: usize) -> Self {
        ValidationBudget { max_splats: Some(max_splats) }
    }
}

/// A single validation finding, carrying the EXACT offending value so the
/// receipt names what's wrong (not just "invalid").
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationIssue {
    /// A position component is NaN/Inf.
    NonFinitePosition { index: usize, position: [f32; 3] },
    /// A scale that must be positive is ≤ 0 (for a volume splat; 2DGS surface
    /// disks legitimately set `scale_w == 0`, which is NOT flagged).
    NonPositiveScale { index: usize, scales: [f32; 3] },
    /// A spectral band decoded to a non-finite value (f16 Inf/NaN bit pattern).
    NonFiniteSpectral { index: usize, band: usize, value: f32 },
    /// Every spectral band is exactly 0 — the splat has no radiance under any
    /// illuminant (a warning: legal but almost always a mistake).
    ZeroSpectral { index: usize },
    /// The splat count exceeds the import budget.
    OverBudget { count: usize, budget: usize },
}

/// Whether a finding fails the gate (`Error`) or merely warns (`Warning`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// The result of [`validate_splats`].
#[derive(Debug, Clone)]
pub struct ValidationReport {
    issues: Vec<(Severity, ValidationIssue)>,
}

impl ValidationReport {
    /// Issues that fail the gate.
    pub fn errors(&self) -> impl Iterator<Item = &ValidationIssue> {
        self.issues
            .iter()
            .filter(|(s, _)| *s == Severity::Error)
            .map(|(_, i)| i)
    }

    /// Issues that warn but pass the gate.
    pub fn warnings(&self) -> impl Iterator<Item = &ValidationIssue> {
        self.issues
            .iter()
            .filter(|(s, _)| *s == Severity::Warning)
            .map(|(_, i)| i)
    }

    pub fn error_count(&self) -> usize {
        self.errors().count()
    }

    pub fn warning_count(&self) -> usize {
        self.warnings().count()
    }

    /// True iff there are zero errors (warnings are allowed through).
    pub fn is_ok(&self) -> bool {
        self.error_count() == 0
    }

    /// One human-readable receipt line per issue, naming the exact value.
    pub fn receipts(&self) -> Vec<String> {
        self.issues
            .iter()
            .map(|(sev, issue)| {
                let tag = match sev {
                    Severity::Error => "error",
                    Severity::Warning => "warning",
                };
                let body = match issue {
                    ValidationIssue::NonFinitePosition { index, position } => {
                        format!("splat {index}: non-finite position {position:?}")
                    }
                    ValidationIssue::NonPositiveScale { index, scales } => {
                        format!("splat {index}: non-positive scale {scales:?}")
                    }
                    ValidationIssue::NonFiniteSpectral { index, band, value } => {
                        format!("splat {index}: non-finite spectral band {band} = {value}")
                    }
                    ValidationIssue::ZeroSpectral { index } => {
                        format!("splat {index}: all-zero spectral radiance (invisible under every light)")
                    }
                    ValidationIssue::OverBudget { count, budget } => {
                        format!("{count} splats exceeds the import budget of {budget}")
                    }
                };
                format!("[{tag}] {body}")
            })
            .collect()
    }
}

/// Validate a splat buffer for integrity, budget, and spectral validity.
///
/// Pure and allocation-bounded (one `Vec` of issues). Reads only verified
/// `GaussianSplat` accessors. The `scale_w` check is gated on `is_volume()` so
/// the entire legitimate 2DGS-disk population (which sets `scale_w == 0`) is
/// never flagged.
pub fn validate_splats(splats: &[GaussianSplat], budget: ValidationBudget) -> ValidationReport {
    let mut issues: Vec<(Severity, ValidationIssue)> = Vec::new();

    for (index, s) in splats.iter().enumerate() {
        // Integrity: position must be finite.
        let position = s.position();
        if position.iter().any(|c| !c.is_finite()) {
            issues.push((Severity::Error, ValidationIssue::NonFinitePosition { index, position }));
        }

        // Integrity: scales must be positive. scale_w (the 3rd axis) is only a
        // volume concept — a 2DGS surface disk legitimately has scale_w == 0.
        let scales = s.scales();
        let bad_uv = scales[0] <= 0.0 || scales[1] <= 0.0;
        let bad_w = s.is_volume() && scales[2] <= 0.0;
        if bad_uv || bad_w {
            issues.push((Severity::Error, ValidationIssue::NonPositiveScale { index, scales }));
        }

        // Spectral validity: non-finite is an error; all-zero is a warning.
        let mut all_zero = true;
        for band in 0..BANDS {
            let v = s.spectral_f32(band);
            if !v.is_finite() {
                issues.push((
                    Severity::Error,
                    ValidationIssue::NonFiniteSpectral { index, band, value: v },
                ));
                all_zero = false;
                break; // one spectral error per splat is enough signal
            }
            if v != 0.0 {
                all_zero = false;
            }
        }
        if all_zero {
            issues.push((Severity::Warning, ValidationIssue::ZeroSpectral { index }));
        }
    }

    // Budget: the whole-buffer count cap.
    if let Some(m) = budget.max_splats {
        if splats.len() > m {
            issues.push((
                Severity::Error,
                ValidationIssue::OverBudget { count: splats.len(), budget: m },
            ));
        }
    }

    ValidationReport { issues }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    /// f16 1.0 bits — a finite, nonzero, valid spectral fill.
    const ONE_F16: u16 = 0x3c00;
    fn valid_spectral() -> [u16; 16] {
        [ONE_F16; 16]
    }

    fn vol(pos: [f32; 3], scale: [f32; 3], spectral: [u16; 16]) -> GaussianSplat {
        GaussianSplat::volume(pos, scale, Quat::IDENTITY, 255, spectral)
    }

    #[test]
    fn nan_position_one_error() {
        let v = vec![
            vol([f32::NAN, 0.0, 0.0], [1.0; 3], valid_spectral()),
            vol([0.0, 0.0, 0.0], [1.0; 3], valid_spectral()),
            vol([1.0, 2.0, 3.0], [1.0; 3], valid_spectral()),
        ];
        let r = validate_splats(&v, ValidationBudget::UNLIMITED);
        assert_eq!(r.error_count(), 1, "exactly one error");
        // NaN != NaN, so destructure rather than assert_eq the value.
        let e = r.errors().next().unwrap();
        match e {
            ValidationIssue::NonFinitePosition { index, position } => {
                assert_eq!(*index, 0);
                assert!(position[0].is_nan(), "the NaN component survives into the report");
            }
            other => panic!("expected NonFinitePosition, got {other:?}"),
        }
    }

    #[test]
    fn negative_scale_flags_index() {
        let mut v: Vec<GaussianSplat> = (0..3).map(|_| vol([0.0; 3], [1.0; 3], valid_spectral())).collect();
        v[2] = vol([0.0; 3], [1.0, -0.5, 1.0], valid_spectral());
        let r = validate_splats(&v, ValidationBudget::UNLIMITED);
        assert_eq!(r.error_count(), 1);
        assert_eq!(
            r.errors().next().unwrap(),
            &ValidationIssue::NonPositiveScale { index: 2, scales: [1.0, -0.5, 1.0] }
        );
    }

    #[test]
    fn surface_zero_scale_w_not_flagged() {
        // A 2DGS surface disk: scale_w == 0 by construction, kind == 0.
        let disk = GaussianSplat::surface(
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            255,
            valid_spectral(),
        );
        let r = validate_splats(std::slice::from_ref(&disk), ValidationBudget::UNLIMITED);
        assert_eq!(r.error_count(), 0, "surface scale_w==0 must not be flagged");
    }

    #[test]
    fn over_budget_exact_count() {
        let v: Vec<GaussianSplat> = (0..5).map(|_| vol([0.0; 3], [1.0; 3], valid_spectral())).collect();
        let r = validate_splats(&v, ValidationBudget::with_max(3));
        assert_eq!(r.error_count(), 1);
        assert_eq!(
            r.errors().next().unwrap(),
            &ValidationIssue::OverBudget { count: 5, budget: 3 }
        );
    }

    #[test]
    fn zero_spectral_is_warning() {
        let v = vec![vol([0.0; 3], [1.0; 3], [0u16; 16])];
        let r = validate_splats(&v, ValidationBudget::UNLIMITED);
        assert_eq!(r.error_count(), 0);
        assert_eq!(r.warning_count(), 1);
        assert_eq!(r.warnings().next().unwrap(), &ValidationIssue::ZeroSpectral { index: 0 });
        assert!(r.is_ok(), "a warning passes the gate");
    }

    #[test]
    fn clean_buffer_ok() {
        let v: Vec<GaussianSplat> = (0..100).map(|i| vol([i as f32, 0.0, 0.0], [1.0; 3], valid_spectral())).collect();
        let r = validate_splats(&v, ValidationBudget::UNLIMITED);
        assert!(r.is_ok());
        assert_eq!(r.receipts().len(), 0, "a clean buffer produces no receipts");
    }
}
