/// Capabilities that a game engine may support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EngineCapability {
    SpectralRendering,
    ProceduralGeneration,
    GaussianSplatting,
    PhysicalAudio,
    AIAssetGeneration,
    LargeWorldCoordinates,
    DestructionMasking,
    NeuralDenoising,
    CrossPlatform,
    VisualScripting,
    RayTracing,
    Nanite,
    Lumen,
    VolumetricClouds,
    MetaHuman,
    BlueprintScripting,
    HotReload,
    NaturalLanguageCommands,
    SpectralAudio,
    PluginEcosystem,
}

/// Profile describing an engine's capabilities and limits.
#[derive(Debug, Clone)]
pub struct EngineProfile {
    pub name: String,
    pub capabilities: Vec<EngineCapability>,
    pub max_triangles_or_splats: u64,
    pub has_spectral: bool,
}

/// Result of comparing two engines.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    pub advantages: Vec<String>,
    pub disadvantages: Vec<String>,
    pub parity: Vec<String>,
    pub overall_score: f32,
}

/// Compare Ochroma against another engine.
pub fn compare_engines(ochroma: &EngineProfile, other: &EngineProfile) -> ComparisonReport {
    let mut advantages = Vec::new();
    let mut disadvantages = Vec::new();
    let mut parity = Vec::new();

    // Compare each capability.
    let all_caps = collect_all_capabilities(ochroma, other);

    for cap in &all_caps {
        let ochroma_has = ochroma.capabilities.contains(cap);
        let other_has = other.capabilities.contains(cap);

        let cap_name = format!("{:?}", cap);

        match (ochroma_has, other_has) {
            (true, false) => advantages.push(format!(
                "Ochroma supports {} which {} lacks",
                cap_name, other.name
            )),
            (false, true) => disadvantages.push(format!(
                "{} supports {} which Ochroma lacks",
                other.name, cap_name
            )),
            (true, true) => parity.push(format!("Both support {}", cap_name)),
            (false, false) => {} // neither has it
        }
    }

    // Compare scale.
    if ochroma.max_triangles_or_splats > other.max_triangles_or_splats {
        advantages.push(format!(
            "Ochroma handles {}x more primitives",
            ochroma.max_triangles_or_splats / other.max_triangles_or_splats.max(1)
        ));
    } else if other.max_triangles_or_splats > ochroma.max_triangles_or_splats {
        disadvantages.push(format!(
            "{} handles {}x more primitives",
            other.name,
            other.max_triangles_or_splats / ochroma.max_triangles_or_splats.max(1)
        ));
    } else {
        parity.push("Both handle similar primitive counts".to_string());
    }

    // Spectral rendering is a big differentiator.
    if ochroma.has_spectral && !other.has_spectral {
        advantages.push("Ochroma has spectral rendering for physically accurate colour".to_string());
    }

    // Calculate score: +1 for each advantage, -1 for each disadvantage, normalised.
    let total = (advantages.len() + disadvantages.len() + parity.len()).max(1) as f32;
    let score = (advantages.len() as f32 - disadvantages.len() as f32) / total;
    // Clamp to [-1, 1] and remap to [0, 100].
    let overall_score = ((score + 1.0) / 2.0 * 100.0).clamp(0.0, 100.0);

    ComparisonReport {
        advantages,
        disadvantages,
        parity,
        overall_score,
    }
}

fn collect_all_capabilities(a: &EngineProfile, b: &EngineProfile) -> Vec<EngineCapability> {
    let mut all: Vec<EngineCapability> = a.capabilities.clone();
    for cap in &b.capabilities {
        if !all.contains(cap) {
            all.push(*cap);
        }
    }
    all
}

/// Pre-built Ochroma profile.
pub fn ochroma_profile() -> EngineProfile {
    EngineProfile {
        name: "Ochroma".to_string(),
        capabilities: vec![
            EngineCapability::SpectralRendering,
            EngineCapability::GaussianSplatting,
            EngineCapability::ProceduralGeneration,
            EngineCapability::AIAssetGeneration,
            EngineCapability::NeuralDenoising,
            EngineCapability::LargeWorldCoordinates,
            EngineCapability::DestructionMasking,
            EngineCapability::PhysicalAudio,
            EngineCapability::CrossPlatform,
            EngineCapability::VisualScripting,
            EngineCapability::HotReload,
            EngineCapability::NaturalLanguageCommands,
            EngineCapability::SpectralAudio,
            EngineCapability::PluginEcosystem,
            EngineCapability::VolumetricClouds,
        ],
        max_triangles_or_splats: 100_000_000, // 100M splats
        has_spectral: true,
    }
}

/// Pre-built Unreal Engine 5 profile.
pub fn unreal5_profile() -> EngineProfile {
    EngineProfile {
        name: "Unreal Engine 5".to_string(),
        capabilities: vec![
            EngineCapability::RayTracing,
            EngineCapability::Nanite,
            EngineCapability::Lumen,
            EngineCapability::VolumetricClouds,
            EngineCapability::MetaHuman,
            EngineCapability::BlueprintScripting,
            EngineCapability::CrossPlatform,
            EngineCapability::ProceduralGeneration,
            EngineCapability::LargeWorldCoordinates,
            EngineCapability::VisualScripting,
            EngineCapability::PluginEcosystem,
        ],
        max_triangles_or_splats: 50_000_000, // 50M triangles (Nanite)
        has_spectral: false,
    }
}

/// Pre-built Unity profile.
pub fn unity_profile() -> EngineProfile {
    EngineProfile {
        name: "Unity".to_string(),
        capabilities: vec![
            EngineCapability::RayTracing,
            EngineCapability::CrossPlatform,
            EngineCapability::ProceduralGeneration,
            EngineCapability::VisualScripting,
            EngineCapability::PluginEcosystem,
            EngineCapability::VolumetricClouds,
        ],
        max_triangles_or_splats: 10_000_000, // 10M triangles
        has_spectral: false,
    }
}

/// Pre-built Godot profile.
pub fn godot_profile() -> EngineProfile {
    EngineProfile {
        name: "Godot".to_string(),
        capabilities: vec![
            EngineCapability::CrossPlatform,
            EngineCapability::VisualScripting,
            EngineCapability::ProceduralGeneration,
            EngineCapability::PluginEcosystem,
        ],
        max_triangles_or_splats: 5_000_000, // 5M triangles
        has_spectral: false,
    }
}
