use vox_render::comparison::{
    compare_engines, godot_profile, ochroma_profile, unreal5_profile, unity_profile,
    EngineCapability, EngineProfile,
};

#[test]
fn ochroma_beats_unreal_on_spectral() {
    let ochroma = ochroma_profile();
    let unreal = unreal5_profile();
    let report = compare_engines(&ochroma, &unreal);

    let has_spectral_advantage = report
        .advantages
        .iter()
        .any(|a| a.contains("Spectral") || a.contains("spectral"));
    assert!(
        has_spectral_advantage,
        "Ochroma should have spectral rendering advantage over Unreal. Advantages: {:?}",
        report.advantages
    );
}

#[test]
fn ochroma_beats_unreal_on_ai_generation() {
    let ochroma = ochroma_profile();
    let unreal = unreal5_profile();
    let report = compare_engines(&ochroma, &unreal);

    let has_ai_advantage = report
        .advantages
        .iter()
        .any(|a| a.contains("AIAssetGeneration"));
    assert!(
        has_ai_advantage,
        "Ochroma should have AI asset generation advantage. Advantages: {:?}",
        report.advantages
    );
}

#[test]
fn comparison_report_advantage_count() {
    let ochroma = ochroma_profile();
    let unreal = unreal5_profile();
    let report = compare_engines(&ochroma, &unreal);

    // Ochroma has more unique capabilities than Unreal in our profiles.
    assert!(
        report.advantages.len() > report.disadvantages.len(),
        "Ochroma should have more advantages than disadvantages vs Unreal. adv={}, disadv={}",
        report.advantages.len(),
        report.disadvantages.len()
    );
}

#[test]
fn score_calculation_positive_for_ochroma() {
    let ochroma = ochroma_profile();
    let unreal = unreal5_profile();
    let report = compare_engines(&ochroma, &unreal);

    // Score should be above 50 (neutral) since Ochroma has more advantages.
    assert!(
        report.overall_score > 50.0,
        "Score should be >50 for Ochroma vs Unreal, got {}",
        report.overall_score
    );
}

#[test]
fn profile_creation() {
    let ochroma = ochroma_profile();
    assert_eq!(ochroma.name, "Ochroma");
    assert!(ochroma.has_spectral);
    assert!(ochroma.max_triangles_or_splats >= 100_000_000);
    assert!(ochroma
        .capabilities
        .contains(&EngineCapability::GaussianSplatting));

    let unreal = unreal5_profile();
    assert_eq!(unreal.name, "Unreal Engine 5");
    assert!(!unreal.has_spectral);
    assert!(unreal.capabilities.contains(&EngineCapability::Nanite));
}

#[test]
fn ochroma_vs_unity() {
    let ochroma = ochroma_profile();
    let unity = unity_profile();
    let report = compare_engines(&ochroma, &unity);

    // Ochroma should dominate Unity even more.
    assert!(report.advantages.len() > report.disadvantages.len());
    assert!(report.overall_score > 50.0);
}

#[test]
fn ochroma_vs_godot() {
    let ochroma = ochroma_profile();
    let godot = godot_profile();
    let report = compare_engines(&ochroma, &godot);

    assert!(report.advantages.len() > report.disadvantages.len());
    assert!(report.overall_score > 60.0);
}

#[test]
fn identical_engines_neutral_score() {
    let a = EngineProfile {
        name: "A".to_string(),
        capabilities: vec![EngineCapability::CrossPlatform],
        max_triangles_or_splats: 1_000_000,
        has_spectral: false,
    };
    let b = EngineProfile {
        name: "B".to_string(),
        capabilities: vec![EngineCapability::CrossPlatform],
        max_triangles_or_splats: 1_000_000,
        has_spectral: false,
    };
    let report = compare_engines(&a, &b);
    assert_eq!(report.advantages.len(), 0);
    assert_eq!(report.disadvantages.len(), 0);
    assert!(!report.parity.is_empty());
    // Score should be exactly 50 (neutral).
    assert!((report.overall_score - 50.0).abs() < 1.0);
}
