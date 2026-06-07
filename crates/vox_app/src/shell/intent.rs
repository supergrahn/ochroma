//! The deterministic, offline intent parser — "Ask Ochroma" v1 (design UX
//! Principle 2: the AI DRIVES the user's surface, it does not just navigate it).
//!
//! The Ctrl+K palette is a DUAL-mode surface: command mode (fuzzy over the
//! registry) AND intent mode (typed natural language). This module is intent
//! mode's brain: a pattern/keyword grammar over the REAL command registry + the
//! REAL live cook graph that parses a sentence into an [`IntentAction`]. Every
//! action is executed through the SAME `CommandRegistry` + `GraphBridge` the
//! manual surface uses — nothing the assistant does is reachable only by the AI,
//! and vice versa (the one-command-surface invariant).
//!
//! It is fully deterministic and network-free: idioms are reused from
//! `vox_nn::nl_commands` (prefix/keyword pattern matching) without taking the
//! dependency. Unknown intents answer honestly and surface the three nearest
//! real commands by fuzzy match.
//!
//! The real-LLM seam: [`parse_intent`] is `&str -> IntentAction`. Swapping the
//! deterministic body for an LLM tool-call planner that emits the SAME
//! `IntentAction` enum changes nothing downstream — the executor, receipts, undo
//! stack, and tests are all phrased against `IntentAction`, not against the
//! parser internals.

use super::command_palette::CommandRegistry;

/// A structured action parsed from a natural-language sentence. The executor on
/// `EditorShell` turns each variant into a real registry/`GraphBridge` call. This
/// is the stable contract an LLM planner would emit instead of the deterministic
/// parser — the seam to a real model.
#[derive(Debug, Clone, PartialEq)]
pub enum IntentAction {
    /// Set a node parameter to an absolute value via `GraphBridge::apply_param`.
    /// `node_kind` is a real `type_name` (e.g. `"TerrainNode"`), `key` a real
    /// `set_param` key (e.g. `"resolution"`).
    SetParam {
        node_kind: &'static str,
        key: &'static str,
        /// Friendly label of the param (for the receipt, e.g. "terrain.resolution").
        target: String,
        value: f32,
    },
    /// Nudge a param by a relative delta ("make the terrain bigger"). Resolved to
    /// an absolute value at execution time (current + delta), so it still flows
    /// through the same `apply_param` + undo machinery.
    AdjustParam {
        node_kind: &'static str,
        key: &'static str,
        target: String,
        /// Signed delta added to the current cooked value.
        delta: f32,
    },
    /// Instantiate a new node of the given real registry kind and connect it if
    /// the connection is unambiguous.
    AddNode {
        /// Real registry `type_name` (e.g. `"VegetationNode"`).
        kind: &'static str,
        /// Friendly name used in the receipt (e.g. "vegetation").
        friendly: String,
    },
    /// Run a registered command by id (theme swap, focus a tab, recook, …).
    RunCommand {
        id: &'static str,
        /// Friendly receipt phrase (e.g. "Switched to light theme").
        receipt: String,
    },
    /// The parser could not map the sentence to any action. Carries the three
    /// nearest real command titles (fuzzy) so the assistant can suggest honestly.
    Unknown { suggestions: Vec<String> },
}

/// Parse a natural-language sentence into an [`IntentAction`].
///
/// Deterministic keyword grammar. The order of the rules below IS the grammar's
/// precedence. `registry` is used only for the unknown-intent suggestion list
/// (the nearest real commands) — so the honest fallback names commands that
/// truly exist.
pub fn parse_intent(text: &str, registry: &CommandRegistry) -> IntentAction {
    let lower = text.trim().to_lowercase();
    if lower.is_empty() {
        return IntentAction::Unknown {
            suggestions: nearest_commands(registry, ""),
        };
    }

    // --- 1. Relative size: "make the terrain bigger/smaller" -------------------
    // Mapped to terrain amplitude (height) with a sensible delta — the design's
    // "make the terrain bigger -> amplitude/world_size delta".
    if (lower.contains("terrain") || lower.contains("world")) && lower.contains("bigger")
        || (lower.contains("terrain") && (lower.contains("taller") || lower.contains("larger")))
    {
        return IntentAction::AdjustParam {
            node_kind: "TerrainNode",
            key: "amplitude",
            target: "terrain.amplitude".into(),
            delta: 80.0,
        };
    }
    if (lower.contains("terrain") || lower.contains("world"))
        && (lower.contains("smaller") || lower.contains("flatter") || lower.contains("shorter"))
    {
        return IntentAction::AdjustParam {
            node_kind: "TerrainNode",
            key: "amplitude",
            target: "terrain.amplitude".into(),
            delta: -80.0,
        };
    }

    // --- 2. Absolute param edit: "set <param> to <value>" ----------------------
    // e.g. "set terrain resolution to 128".
    if let Some(action) = try_set_param(&lower) {
        return action;
    }

    // --- 3. Add a node: "add a building node" / "add vegetation" ---------------
    if let Some(action) = try_add_node(&lower) {
        return action;
    }

    // --- 4. Theme: "switch to light theme" / "dark mode" -----------------------
    if lower.contains("light") && (lower.contains("theme") || lower.contains("mode")) {
        return IntentAction::RunCommand {
            id: "view.theme_light",
            receipt: "Switched to light theme".into(),
        };
    }
    if lower.contains("dark") && (lower.contains("theme") || lower.contains("mode")) {
        return IntentAction::RunCommand {
            id: "view.theme_dark",
            receipt: "Switched to dark theme".into(),
        };
    }

    // --- 5. Focus a tab: "show the crucible graph" / "open node graph" ---------
    if lower.contains("crucible") {
        return IntentAction::RunCommand {
            id: "view.focus_crucible",
            receipt: "Focused the Crucible graph".into(),
        };
    }
    if lower.contains("node graph") || (lower.contains("show") && lower.contains("graph")) {
        return IntentAction::RunCommand {
            id: "view.focus_node_graph",
            receipt: "Focused the Node Graph".into(),
        };
    }
    if lower.contains("viewport") || lower.contains("the scene") {
        return IntentAction::RunCommand {
            id: "view.focus_viewport",
            receipt: "Focused the Viewport".into(),
        };
    }

    IntentAction::Unknown {
        suggestions: nearest_commands(registry, &lower),
    }
}

/// "set <noun> <param> to <value>" / "set <param> to <value>". Resolves the noun
/// and param words to a real `(node_kind, key)` pair via [`resolve_param`].
fn try_set_param(lower: &str) -> Option<IntentAction> {
    // Accept "set ... to N" and "change ... to N".
    let rest = lower
        .strip_prefix("set ")
        .or_else(|| lower.strip_prefix("change "))
        .or_else(|| lower.strip_prefix("make "))?;
    let to_idx = rest.find(" to ")?;
    let subject = rest[..to_idx].trim();
    let value_str = rest[to_idx + 4..].trim();
    // Take the leading numeric token of the value tail (e.g. "128 cells" -> 128).
    let num: f32 = value_str
        .split_whitespace()
        .next()
        .and_then(|w| w.trim_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').parse().ok())?;
    let (node_kind, key, target) = resolve_param(subject)?;
    Some(IntentAction::SetParam {
        node_kind,
        key,
        target,
        value: num,
    })
}

/// Map the words of a subject phrase to a real `(node_kind, set_param key,
/// friendly target)`. Only the params the GraphBridge actually exposes are
/// reachable, so an executed intent can never name a phantom parameter.
fn resolve_param(subject: &str) -> Option<(&'static str, &'static str, String)> {
    let s = subject;
    // Terrain params.
    if s.contains("terrain") || s.contains("ground") || s.contains("landscape") {
        if s.contains("resolution") || s.contains("detail") {
            return Some(("TerrainNode", "resolution", "terrain.resolution".into()));
        }
        if s.contains("amplitude") || s.contains("height") || s.contains("tall") {
            return Some(("TerrainNode", "amplitude", "terrain.amplitude".into()));
        }
        if s.contains("seed") {
            return Some(("TerrainNode", "seed", "terrain.seed".into()));
        }
        // Bare "set terrain resolution to N" without the word terrain in the param
        // half is handled above; a bare "terrain" subject defaults to resolution.
        return Some(("TerrainNode", "resolution", "terrain.resolution".into()));
    }
    // Vegetation params.
    if s.contains("vegetation") || s.contains("tree") || s.contains("plant") {
        if s.contains("branch") {
            return Some(("VegetationNode", "branch_levels", "vegetation.branch_levels".into()));
        }
        if s.contains("trunk") {
            return Some(("VegetationNode", "trunk_radius", "vegetation.trunk_radius".into()));
        }
        if s.contains("height") || s.contains("tall") {
            return Some(("VegetationNode", "height", "vegetation.height".into()));
        }
        return Some(("VegetationNode", "branch_levels", "vegetation.branch_levels".into()));
    }
    // Biome params.
    if s.contains("biome") || s.contains("moisture") {
        if s.contains("moisture") {
            return Some(("BiomeNode", "moisture", "biome.moisture".into()));
        }
        return Some(("BiomeNode", "world_height", "biome.world_height".into()));
    }
    // Bare param words (no noun) — default to terrain (the primary subject).
    if s.contains("resolution") || s.contains("detail") {
        return Some(("TerrainNode", "resolution", "terrain.resolution".into()));
    }
    None
}

/// "add a building node" / "add vegetation" / "create a tree". Maps the friendly
/// noun to a real registry `type_name`.
fn try_add_node(lower: &str) -> Option<IntentAction> {
    let rest = lower
        .strip_prefix("add ")
        .or_else(|| lower.strip_prefix("create ")) ?;
    let rest = rest
        .trim_start_matches("a ")
        .trim_start_matches("an ")
        .trim_start_matches("some ")
        .trim();
    // Strip a trailing " node" so "building node" -> "building".
    let noun = rest.trim_end_matches(" node").trim();
    let kind = match noun {
        n if n.contains("building") || n.contains("house") => "BuildingNode",
        n if n.contains("vegetation") || n.contains("tree") || n.contains("plant") || n.contains("foliage") => "VegetationNode",
        n if n.contains("terrain") || n.contains("ground") => "TerrainNode",
        n if n.contains("biome") => "BiomeNode",
        n if n.contains("moisture") => "MoistureNode",
        n if n.contains("plot") => "PlotNode",
        _ => return None,
    };
    Some(IntentAction::AddNode {
        kind,
        friendly: noun.to_string(),
    })
}

/// The three nearest real command titles to `query` (fuzzy over the registry) —
/// the honest "I don't know that yet, try: …" suggestion list.
pub fn nearest_commands(registry: &CommandRegistry, query: &str) -> Vec<String> {
    // For an empty/garbage query the fuzzy ranker may return nothing; fall back
    // to the first registered commands so we always offer three real titles.
    let mut hits: Vec<String> = registry
        .search(query)
        .into_iter()
        .map(|c| c.title.clone())
        .collect();
    if hits.len() < 3 {
        for c in &registry.commands {
            if !hits.contains(&c.title) {
                hits.push(c.title.clone());
            }
            if hits.len() >= 3 {
                break;
            }
        }
    }
    hits.truncate(3);
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shell::command_palette::Command;

    fn registry() -> CommandRegistry {
        let mut r = CommandRegistry::new();
        r.add(Command::new("world.add", "Add to world", "Create", "Ctrl+A", || {}));
        r.add(Command::new("file.save", "Save world", "File", "Ctrl+S", || {}));
        r.add(Command::new("build.cook", "Recook graph", "Build", "F5", || {}));
        r.add(Command::new("view.wireframe", "Toggle wireframe", "Window", "", || {}));
        r
    }

    #[test]
    fn set_terrain_resolution_parses_to_real_param() {
        let action = parse_intent("set terrain resolution to 128", &registry());
        assert_eq!(
            action,
            IntentAction::SetParam {
                node_kind: "TerrainNode",
                key: "resolution",
                target: "terrain.resolution".into(),
                value: 128.0,
            }
        );
    }

    #[test]
    fn add_vegetation_parses_to_real_kind() {
        assert_eq!(
            parse_intent("add vegetation", &registry()),
            IntentAction::AddNode { kind: "VegetationNode", friendly: "vegetation".into() }
        );
        // "add a building node" -> BuildingNode.
        assert_eq!(
            parse_intent("add a building node", &registry()),
            IntentAction::AddNode { kind: "BuildingNode", friendly: "building".into() }
        );
    }

    #[test]
    fn make_terrain_bigger_maps_to_amplitude_delta() {
        match parse_intent("make the terrain bigger", &registry()) {
            IntentAction::AdjustParam { node_kind, key, delta, .. } => {
                assert_eq!(node_kind, "TerrainNode");
                assert_eq!(key, "amplitude");
                assert!(delta > 0.0, "bigger must be a positive delta");
            }
            other => panic!("expected AdjustParam, got {other:?}"),
        }
    }

    #[test]
    fn theme_and_focus_map_to_commands() {
        assert!(matches!(
            parse_intent("switch to light theme", &registry()),
            IntentAction::RunCommand { id: "view.theme_light", .. }
        ));
        assert!(matches!(
            parse_intent("show the crucible graph", &registry()),
            IntentAction::RunCommand { id: "view.focus_crucible", .. }
        ));
    }

    #[test]
    fn gibberish_is_unknown_with_real_suggestions() {
        let r = registry();
        match parse_intent("flibbertigibbet xyzzy", &r) {
            IntentAction::Unknown { suggestions } => {
                assert_eq!(suggestions.len(), 3, "must offer exactly 3 suggestions");
                let real: Vec<String> = r.commands.iter().map(|c| c.title.clone()).collect();
                for s in &suggestions {
                    assert!(real.contains(s), "suggestion {s:?} must be a real command title");
                }
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
    }
}
