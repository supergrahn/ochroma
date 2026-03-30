//! BiomeNode — classifies terrain cells into biome kinds.

use crate::node_graph::{
    HeightfieldSpatial, NodeDescriptor, NodeError, NodeInputs, NodeOutputs,
    OchromaNode, ParamValue, PortData, PortSpec, PortType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BiomeKind {
    Alpine             = 0,
    Tundra             = 1,
    Forest             = 2,
    Grassland          = 3,
    Desert             = 4,
    Wetland            = 5,
    Coastal            = 6,
    SubalpineShrub     = 7,
    Savanna            = 8,
    Taiga              = 9,
    TropicalRainforest = 10,
}

impl BiomeKind {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0  => Self::Alpine,
            1  => Self::Tundra,
            2  => Self::Forest,
            3  => Self::Grassland,
            4  => Self::Desert,
            5  => Self::Wetland,
            6  => Self::Coastal,
            7  => Self::SubalpineShrub,
            8  => Self::Savanna,
            9  => Self::Taiga,
            10 => Self::TropicalRainforest,
            _  => Self::Grassland,
        }
    }
}

fn classify(height: f32, world_height: f32, moisture: f32) -> BiomeKind {
    let norm_h = (height / world_height).clamp(0.0, 1.0);
    if norm_h >= 0.90 { return BiomeKind::Alpine; }
    if norm_h >= 0.70 { return BiomeKind::Tundra; }
    if norm_h >= 0.55 { return if moisture > 0.5 { BiomeKind::Taiga } else { BiomeKind::SubalpineShrub }; }
    if norm_h >= 0.30 {
        if moisture > 0.6 { return BiomeKind::TropicalRainforest; }
        if moisture > 0.4 { return BiomeKind::Forest; }
        if moisture > 0.2 { return BiomeKind::Grassland; }
        return BiomeKind::Savanna;
    }
    // Low altitude
    if moisture > 0.6 { BiomeKind::Wetland }
    else if moisture > 0.3 { BiomeKind::Coastal }
    else { BiomeKind::Desert }
}

pub fn biome_to_splat_weights(kind: BiomeKind, _altitude: f32, _moisture: f32) -> [f32; 4] {
    // [water, ground, vegetation, rock]
    match kind {
        BiomeKind::Alpine             => [0.00, 0.50, 0.05, 0.45],
        BiomeKind::Tundra             => [0.00, 0.45, 0.15, 0.40],
        BiomeKind::Forest             => [0.00, 0.05, 0.70, 0.25],
        BiomeKind::Grassland          => [0.00, 0.20, 0.65, 0.15],
        BiomeKind::Desert             => [0.00, 0.80, 0.05, 0.15],
        BiomeKind::Wetland            => [0.40, 0.05, 0.40, 0.15],
        BiomeKind::Coastal            => [0.30, 0.40, 0.20, 0.10],
        BiomeKind::SubalpineShrub     => [0.00, 0.30, 0.40, 0.30],
        BiomeKind::Savanna            => [0.00, 0.30, 0.50, 0.20],
        BiomeKind::Taiga              => [0.05, 0.10, 0.65, 0.20],
        BiomeKind::TropicalRainforest => [0.10, 0.05, 0.80, 0.05],
    }
}

/// 7-slot USGS-style spectral palette per terrain material.
/// Slots: 0=Water, 1=Sand, 2=Grass, 3=Dirt, 4=Rock, 5=Snow, 6=Bark
pub struct SpectralTerrainMaterials {
    pub slots: [[f32; 16]; 7],
}

impl Default for SpectralTerrainMaterials {
    fn default() -> Self {
        Self {
            slots: [
                // Water (dark, low reflectance)
                [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.07, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.10, 0.12],
                // Sand (warm yellow-beige)
                [0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.58, 0.56, 0.54, 0.52, 0.50, 0.48],
                // Grass (green, high NIR)
                [0.03, 0.04, 0.05, 0.05, 0.06, 0.08, 0.28, 0.42, 0.40, 0.28, 0.10, 0.07, 0.05, 0.04, 0.38, 0.48],
                // Dirt (warm brown)
                [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.28, 0.34, 0.36, 0.34, 0.32, 0.30, 0.28, 0.26],
                // Rock (gray)
                [0.22, 0.24, 0.25, 0.26, 0.27, 0.27, 0.28, 0.28, 0.28, 0.27, 0.27, 0.26, 0.26, 0.25, 0.25, 0.24],
                // Snow (near-white)
                [0.90, 0.91, 0.92, 0.92, 0.93, 0.93, 0.93, 0.93, 0.93, 0.92, 0.92, 0.91, 0.91, 0.90, 0.89, 0.88],
                // Bark (dark brown)
                [0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.20, 0.26, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18],
            ],
        }
    }
}

pub struct BiomeNode {
    pub world_height: f32,
    pub moisture:     f32,
}

impl Default for BiomeNode {
    fn default() -> Self {
        Self { world_height: 400.0, moisture: 0.5 }
    }
}

impl OchromaNode for BiomeNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "BiomeNode",
            inputs:  vec![PortSpec { name: "terrain",  port_type: PortType::Terrain,  optional: false }],
            outputs: vec![PortSpec { name: "biome_map", port_type: PortType::BiomeMap, optional: false }],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        match (key, value) {
            ("world_height", ParamValue::Float(v)) => { self.world_height = v as f32; Ok(()) }
            ("moisture",     ParamValue::Float(v)) => { self.moisture     = v as f32; Ok(()) }
            (k, _) => Err(NodeError::UnknownParam(k.into())),
        }
    }

    fn cook(&self, inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let terrain = inputs.get("terrain")
            .ok_or_else(|| NodeError::MissingInput("terrain".into()))?
            .as_terrain()
            .ok_or_else(|| NodeError::TypeMismatch("terrain".into()))?;

        let biome_bytes: Vec<u8> = terrain.heights.iter()
            .map(|&h| classify(h, self.world_height, self.moisture) as u8)
            .collect();

        let mut out = NodeOutputs::new();
        out.insert("biome_map".into(), PortData::BiomeMap(biome_bytes));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biome_node_classifies_by_height() {
        let node = BiomeNode::default(); // world_height=400
        let terrain = HeightfieldSpatial { resolution: 2, world_size: 100.0, heights: vec![360.0; 4] };
        let mut inputs = NodeInputs::new();
        inputs.insert("terrain".into(), PortData::Terrain(terrain));
        let out = node.cook(inputs).unwrap();
        let biome_bytes = out["biome_map"].as_biome_map().unwrap();
        assert!(biome_bytes.iter().all(|&b| b == BiomeKind::Alpine as u8), "cells at 90% world height should be Alpine");
    }

    #[test]
    fn test_splat_weights_sum_to_one() {
        let weights = biome_to_splat_weights(BiomeKind::Forest, 40.0, 80.0);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "weights must sum to 1, got {sum}");
        assert!(weights[2] > 0.5, "Forest: vegetation channel (2) should dominate");
    }

    #[test]
    fn test_spectral_terrain_materials_water_dark() {
        let mats = SpectralTerrainMaterials::default();
        assert!(mats.slots[0][0] < 0.1,  "Water slot UV reflectance should be dark");
        assert!(mats.slots[5][0] > 0.85, "Snow slot should be near-white");
    }
}
