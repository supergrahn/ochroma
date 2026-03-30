//! BuildingDirector — LLM-driven BuildingDescription authoring.
//! LLM generates BuildingDescription JSON; WFC generates geometry. Never mixed.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "PascalCase")]
pub enum Program {
    #[default]
    Residential,
    Agricultural,
    Civic,
    Religious,
    Commercial,
    Industrial,
    Utility,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "PascalCase")]
pub enum Setting {
    Urban,
    #[default]
    Suburban,
    Rural,
    Industrial,
    Waterfront,
    HistoricalOldTown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "PascalCase")]
pub enum BuildingCondition {
    New,
    #[default]
    Aged,
    Weathered,
    Derelict,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum GradingStrategy {
    #[default]
    LevelPad,
    Stepped,
    Pier,
    CutIntoSlope,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BuildingStyle {
    Victorian,
    Modern,
    Colonial,
    Industrial,
    Gothic,
    Brutalist,
    Medieval,
    Tudor,
    Mediterranean,
    Craftsman,
}

pub struct BuildingParams {
    pub floors: u8,
    pub floor_height: f32,
    pub style: BuildingStyle,
    pub grading: GradingStrategy,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BuildingDescription {
    pub program: Program,
    pub setting: Setting,
    pub style_key: String,
    pub era: String,
    pub condition: BuildingCondition,
    pub floors: u8,
    pub floor_height: f32,
    pub seed: u64,
    pub detail_atoms: Option<Vec<String>>,
    pub organic_atoms: Option<Vec<String>>,
}

impl BuildingDescription {
    pub fn to_building_params(&self) -> BuildingParams {
        let style = match self.style_key.to_lowercase().as_str() {
            s if s.starts_with("victorian") => BuildingStyle::Victorian,
            s if s.starts_with("modern") => BuildingStyle::Modern,
            s if s.starts_with("gothic") => BuildingStyle::Gothic,
            s if s.starts_with("brutalist") => BuildingStyle::Brutalist,
            s if s.starts_with("medieval") => BuildingStyle::Medieval,
            s if s.starts_with("tudor") => BuildingStyle::Tudor,
            s if s.starts_with("mediterranean") => BuildingStyle::Mediterranean,
            s if s.starts_with("craftsman") => BuildingStyle::Craftsman,
            s if s.starts_with("industrial") => BuildingStyle::Industrial,
            _ => BuildingStyle::Colonial,
        };
        let grading = match self.setting {
            Setting::Waterfront => GradingStrategy::Pier,
            Setting::Rural => GradingStrategy::CutIntoSlope,
            Setting::HistoricalOldTown => GradingStrategy::Stepped,
            _ => GradingStrategy::LevelPad,
        };
        BuildingParams {
            floors: self.floors.max(1),
            floor_height: if self.floor_height > 0.0 { self.floor_height } else { 3.0 },
            style,
            grading,
            seed: self.seed,
        }
    }
}

pub struct BuildingDirector;

impl BuildingDirector {
    pub fn system_prompt() -> String {
        r#"You are a building architect for a game engine. Given a description of a building, output ONLY valid JSON matching the BuildingDescription schema. No prose, no markdown fences.

Schema: { "program": "Residential"|"Civic"|"Commercial"|..., "setting": "Urban"|..., "style_key": string, "era": string, "condition": "New"|"Aged"|"Weathered"|"Derelict", "floors": int, "floor_height": float, "seed": int, "detail_atoms": [string]|null, "organic_atoms": [string]|null }

Output only the JSON object. No other text."#
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_building_description_json() {
        let json = r#"{"program":"Residential","setting":"Suburban","style_key":"craftsman","era":"1920s","condition":"Aged","floors":2,"floor_height":3.0,"seed":42,"detail_atoms":["exposed_rafter_tails","tapered_porch_columns"],"organic_atoms":["weathered_cedar"]}"#;
        let desc: BuildingDescription = serde_json::from_str(json).unwrap();
        assert_eq!(desc.style_key, "craftsman");
        assert_eq!(desc.floors, 2);
        assert_eq!(desc.detail_atoms.as_ref().unwrap().len(), 2);
        assert!(
            desc.detail_atoms.as_ref().unwrap().contains(&"exposed_rafter_tails".to_string())
        );
    }

    #[test]
    fn test_building_description_compiles_to_params() {
        let desc = BuildingDescription {
            program: Program::Residential,
            setting: Setting::Suburban,
            style_key: "craftsman".into(),
            era: "1920s".into(),
            condition: BuildingCondition::Aged,
            floors: 2,
            floor_height: 3.0,
            seed: 42,
            detail_atoms: Some(vec!["exposed_rafter_tails".into()]),
            organic_atoms: None,
        };
        let params = desc.to_building_params();
        assert_eq!(params.floors, 2);
        assert_eq!(params.floor_height, 3.0);
        assert!(matches!(params.grading, GradingStrategy::LevelPad));
    }

    #[test]
    fn test_building_director_system_prompt_contains_json_schema() {
        let prompt = BuildingDirector::system_prompt();
        assert!(prompt.contains("BuildingDescription"), "prompt must reference schema type");
        assert!(prompt.contains("detail_atoms"), "prompt must explain detail_atoms");
        assert!(prompt.contains("JSON"), "prompt must require JSON output");
    }
}
