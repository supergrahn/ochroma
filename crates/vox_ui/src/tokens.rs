//! The Ochroma token design system — the single source of truth for the
//! editor's look. JSON-backed (`assets/ui/ochroma.theme.json`), consumed by
//! BOTH UI stacks: egui via [`crate::egui_theme::apply`], and the Vello/UiTree
//! overlay via [`crate::ui_tree::StyleSheet::from_tokens`]. One JSON edit
//! reskins the whole ecosystem.
//!
//! NOTE (deviation from the design's literal signature): the design wrote
//! `wire_color(&self, t: vox_editor::node_graph::PortType)`. `vox_ui` is an
//! ENGINE-agnostic crate (CLAUDE.md) and must not pull `vox_editor`
//! (+vox_render/vox_data/vox_usd) into every UI build. We therefore mirror the
//! port-type set as a local [`PortType`] enum with identical variants; the
//! NodeCanvas wave maps `vox_editor::node_graph::PortType -> tokens::PortType`
//! at the single call site in `vox_app`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Node-graph port types, mirroring `vox_editor::node_graph::PortType`.
/// Drives the type-colored socket/wire palette.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortType {
    Splats,
    SpectralField,
    Terrain,
    Mesh,
    LodMesh,
    Instances,
    Scalar,
    BiomeMap,
    SplatWeights,
    /// Exec / control-flow wire (drawn white, L->R arrowhead).
    Flow,
}

/// Node category — drives the per-category colored node header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeCategory {
    Generator,
    Spatial,
    Field,
    Sink,
    Math,
}

/// The type ramp (point sizes for the egui glyph atlas — real AA vector text).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TypeRamp {
    pub caption: f32,
    pub body: f32,
    pub body_strong: f32,
    pub heading: f32,
    pub title: f32,
    pub mono: f32,
}

/// The full token set, loaded from a `*.theme.json` file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tokens {
    #[serde(default)]
    pub name: String,
    /// Dotted color keys -> RGBA bytes, e.g. `"surface.bg.0" -> [18,18,24,255]`.
    pub color: HashMap<String, [u8; 4]>,
    /// 4px spacing grid: `[s0, s1, s2, s3, s4, s5, s6]`.
    pub space: [f32; 7],
    /// Corner radii: `[sm, md, lg]`.
    pub radius: [f32; 3],
    pub type_ramp: TypeRamp,
    /// Motion durations in seconds (egui `animation_time`), keyed e.g. `"fast"`.
    pub motion_ms: HashMap<String, f32>,
    /// Root em size (the single UI-scale knob; default 14).
    pub rem: f32,
}

impl Default for Tokens {
    /// The default tokens are byte-identical to `assets/ui/ochroma.theme.json`
    /// (kept in sync by `tests::default_matches_json`).
    fn default() -> Self {
        let color = [
            ("surface.bg.0", [18, 18, 24, 255]),
            ("surface.bg.1", [24, 24, 28, 255]),
            ("surface.bg.2", [30, 30, 40, 255]),
            ("surface.bg.3", [38, 38, 47, 255]),
            ("surface.hover", [48, 48, 65, 255]),
            ("surface.active", [55, 55, 75, 255]),
            ("surface.border", [45, 45, 60, 255]),
            ("border.strong", [58, 58, 76, 255]),
            ("accent.base", [60, 130, 255, 255]),
            ("accent.hover", [80, 150, 255, 255]),
            ("accent.dim", [40, 90, 180, 255]),
            ("text.primary", [220, 222, 230, 255]),
            ("text.secondary", [140, 145, 160, 255]),
            ("text.disabled", [80, 85, 95, 255]),
            ("status.success", [50, 200, 120, 255]),
            ("status.warning", [255, 180, 50, 255]),
            ("status.error", [255, 70, 70, 255]),
            ("axis.x", [230, 60, 60, 255]),
            ("axis.y", [60, 200, 60, 255]),
            ("axis.z", [60, 100, 230, 255]),
            ("port.splats", [76, 194, 255, 255]),
            ("port.spectral_field", [180, 120, 255, 255]),
            ("port.terrain", [140, 200, 90, 255]),
            ("port.mesh", [255, 180, 90, 255]),
            ("port.lod_mesh", [255, 140, 90, 255]),
            ("port.instances", [255, 225, 100, 255]),
            ("port.scalar", [176, 182, 200, 255]),
            ("port.biome_map", [90, 200, 160, 255]),
            ("port.split_weights", [200, 150, 255, 255]),
            ("port.flow", [255, 255, 255, 255]),
            ("category.generator", [108, 90, 60, 255]),
            ("category.spatial", [60, 130, 80, 255]),
            ("category.field", [110, 90, 150, 255]),
            ("category.sink", [60, 90, 130, 255]),
            ("category.math", [90, 100, 120, 255]),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
        let motion_ms = [("fast", 0.12), ("foldout", 0.15)]
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();
        Tokens {
            name: "Ochroma Dark".into(),
            color,
            space: [0.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0],
            radius: [4.0, 6.0, 8.0],
            type_ramp: TypeRamp {
                caption: 11.0,
                body: 13.0,
                body_strong: 13.0,
                heading: 16.0,
                title: 20.0,
                mono: 12.0,
            },
            motion_ms,
            rem: 14.0,
        }
    }
}

impl Tokens {
    /// Load a theme from a `*.theme.json` file.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Tokens> {
        let bytes = std::fs::read(path.as_ref())?;
        serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Resolve a dotted color key to RGBA bytes. Unknown keys resolve to a loud
    /// magenta so a typo is pixel-visible rather than silently transparent.
    pub fn color(&self, dotted: &str) -> [u8; 4] {
        self.color
            .get(dotted)
            .copied()
            .unwrap_or([255, 0, 255, 255])
    }

    /// The socket/wire color for a port type.
    pub fn wire_color(&self, t: PortType) -> [u8; 4] {
        let key = match t {
            PortType::Splats => "port.splats",
            PortType::SpectralField => "port.spectral_field",
            PortType::Terrain => "port.terrain",
            PortType::Mesh => "port.mesh",
            PortType::LodMesh => "port.lod_mesh",
            PortType::Instances => "port.instances",
            PortType::Scalar => "port.scalar",
            PortType::BiomeMap => "port.biome_map",
            PortType::SplatWeights => "port.split_weights",
            PortType::Flow => "port.flow",
        };
        self.color(key)
    }

    /// The header color for a node category.
    pub fn category_header(&self, cat: NodeCategory) -> [u8; 4] {
        let key = match cat {
            NodeCategory::Generator => "category.generator",
            NodeCategory::Spatial => "category.spatial",
            NodeCategory::Field => "category.field",
            NodeCategory::Sink => "category.sink",
            NodeCategory::Math => "category.math",
        };
        self.color(key)
    }

    /// The root em size (UI-scale knob).
    pub fn rem(&self) -> f32 {
        self.rem
    }

    /// A motion duration in seconds (defaults to 0.12 if the key is absent).
    pub fn motion(&self, key: &str) -> f32 {
        self.motion_ms.get(key).copied().unwrap_or(0.12)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme_path(name: &str) -> std::path::PathBuf {
        // tests run with CWD = crate dir; assets live at the workspace root.
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/ui")
            .join(name)
    }

    #[test]
    fn default_matches_json() {
        // The hand-written Default MUST equal the shipped JSON, byte-for-byte
        // after deserialization (the design's round-trip invariant).
        let loaded = Tokens::load(theme_path("ochroma.theme.json")).expect("load dark theme");
        let def = Tokens::default();
        assert_eq!(loaded, def, "Tokens::default() drifted from ochroma.theme.json");
    }

    #[test]
    fn accent_base_resolves_from_json() {
        let t = Tokens::load(theme_path("ochroma.theme.json")).unwrap();
        assert_eq!(
            t.color("accent.base"),
            [60, 130, 255, 255],
            "accent.base must be the Ochroma brand blue"
        );
    }

    #[test]
    fn light_theme_swaps_window_bg() {
        // The exact test the design specifies (test 2): loading the light theme
        // changes window bg from [18,18,24] to a light value by >40 in a channel.
        let dark = Tokens::load(theme_path("ochroma.theme.json")).unwrap();
        let light = Tokens::load(theme_path("ochroma_light.theme.json")).unwrap();
        let d = dark.color("surface.bg.0");
        let l = light.color("surface.bg.0");
        assert_eq!(d, [18, 18, 24, 255], "dark window bg baseline");
        let max_delta = (0..3)
            .map(|i| (l[i] as i32 - d[i] as i32).unsigned_abs())
            .max()
            .unwrap();
        assert!(
            max_delta > 40,
            "light swap changed window bg by only {max_delta} (dark={d:?} light={l:?})"
        );
    }

    #[test]
    fn wire_and_category_colors_resolve() {
        let t = Tokens::default();
        assert_eq!(t.wire_color(PortType::Splats), [76, 194, 255, 255]);
        assert_eq!(t.wire_color(PortType::Flow), [255, 255, 255, 255]);
        assert_eq!(t.category_header(NodeCategory::Spatial), [60, 130, 80, 255]);
        // Unknown key is loud magenta, not silent transparent.
        assert_eq!(t.color("nope.nope"), [255, 0, 255, 255]);
    }
}
