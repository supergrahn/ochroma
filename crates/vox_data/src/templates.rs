use serde::{Deserialize, Serialize};

/// A project template for creating new games on the Ochroma engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectTemplate {
    pub name: String,
    pub description: String,
    pub genre: GameGenre,
    pub features: Vec<String>,
    pub default_scene: String,
    pub recommended_quality: String,
    pub estimated_complexity: Complexity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameGenre {
    CityBuilder,
    RPG,
    Horror,
    Exploration,
    Puzzle,
    Sandbox,
    Strategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Complexity {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Get all available project templates.
pub fn available_templates() -> Vec<ProjectTemplate> {
    vec![
        ProjectTemplate {
            name: "City Builder".into(),
            description: "Build and manage a thriving city. Roads, zoning, citizens, economy."
                .into(),
            genre: GameGenre::CityBuilder,
            features: vec![
                "Road drawing tools".into(),
                "Zoning system".into(),
                "Citizen simulation".into(),
                "Economy and budget".into(),
                "Service buildings".into(),
                "Traffic simulation".into(),
            ],
            default_scene: "scenes/city_starter.ochroma_scene".into(),
            recommended_quality: "High".into(),
            estimated_complexity: Complexity::Intermediate,
        },
        ProjectTemplate {
            name: "Exploration World".into(),
            description:
                "Open world exploration with procedurally generated terrain and buildings.".into(),
            genre: GameGenre::Exploration,
            features: vec![
                "Procedural terrain".into(),
                "First-person camera".into(),
                "Day/night cycle".into(),
                "Weather system".into(),
            ],
            default_scene: "scenes/exploration_starter.ochroma_scene".into(),
            recommended_quality: "Ultra".into(),
            estimated_complexity: Complexity::Beginner,
        },
        ProjectTemplate {
            name: "Horror Atmosphere".into(),
            description:
                "Atmospheric horror game. Dark environments, spectral lighting effects.".into(),
            genre: GameGenre::Horror,
            features: vec![
                "Spectral lighting".into(),
                "Volumetric fog".into(),
                "Particle effects".into(),
                "Spatial audio".into(),
            ],
            default_scene: "scenes/horror_starter.ochroma_scene".into(),
            recommended_quality: "Ultra".into(),
            estimated_complexity: Complexity::Intermediate,
        },
        ProjectTemplate {
            name: "Puzzle Garden".into(),
            description: "Relaxing puzzle game in a procedural garden environment.".into(),
            genre: GameGenre::Puzzle,
            features: vec![
                "Vegetation generation".into(),
                "Simple physics".into(),
                "Ambient audio".into(),
                "Calm lighting".into(),
            ],
            default_scene: "scenes/puzzle_starter.ochroma_scene".into(),
            recommended_quality: "Medium".into(),
            estimated_complexity: Complexity::Beginner,
        },
        ProjectTemplate {
            name: "Strategy Map".into(),
            description: "Top-down strategy game with terrain, units, and fog of war.".into(),
            genre: GameGenre::Strategy,
            features: vec![
                "Tile-based map".into(),
                "Unit management".into(),
                "Fog of war".into(),
                "Resource gathering".into(),
            ],
            default_scene: "scenes/strategy_starter.ochroma_scene".into(),
            recommended_quality: "High".into(),
            estimated_complexity: Complexity::Advanced,
        },
    ]
}

impl ProjectTemplate {
    pub fn uses_feature(&self, feature: &str) -> bool {
        self.features
            .iter()
            .any(|f| f.to_lowercase().contains(&feature.to_lowercase()))
    }
}
