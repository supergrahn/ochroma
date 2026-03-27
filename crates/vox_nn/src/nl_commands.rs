/// A parsed command from natural language input.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedCommand {
    /// "build/place X near Y"
    BuildStructure {
        what: String,
        where_near: Option<String>,
    },
    /// "zone X as Y" or "zone residential"
    ModifyZone {
        zone_type: String,
        location: Option<String>,
    },
    /// "show X overlay/map"
    ShowOverlay {
        overlay_type: String,
    },
    /// "go to X" / "show me X"
    CameraMove {
        target_description: String,
    },
    /// "why/what/how X"
    QueryInfo {
        question: String,
    },
}

/// Parse a natural language command string into a structured command.
///
/// Uses keyword-based pattern matching (not LLM) for reliability.
pub fn parse_command(text: &str) -> Option<ParsedCommand> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }

    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    if words.is_empty() {
        return None;
    }

    // --- Build/Place patterns ---
    if let Some(cmd) = try_parse_build(&lower) {
        return Some(cmd);
    }

    // --- Zone patterns ---
    if let Some(cmd) = try_parse_zone(&lower) {
        return Some(cmd);
    }

    // --- Show overlay patterns ---
    if let Some(cmd) = try_parse_overlay(&lower) {
        return Some(cmd);
    }

    // --- Camera move patterns ---
    if let Some(cmd) = try_parse_camera(&lower) {
        return Some(cmd);
    }

    // --- Query patterns ---
    if let Some(cmd) = try_parse_query(&lower) {
        return Some(cmd);
    }

    None
}

fn try_parse_build(lower: &str) -> Option<ParsedCommand> {
    // Match "build X near Y", "place X near Y", "build X", "place X"
    for prefix in &["build ", "place ", "construct ", "create "] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let rest = rest.trim_start_matches("a ").trim_start_matches("an ");
            if let Some(near_idx) = rest.find(" near ") {
                let what = rest[..near_idx].trim().to_string();
                let where_near = rest[near_idx + 6..].trim();
                let where_near = where_near
                    .trim_start_matches("the ")
                    .trim()
                    .to_string();
                return Some(ParsedCommand::BuildStructure {
                    what,
                    where_near: Some(where_near),
                });
            }
            if let Some(at_idx) = rest.find(" at ") {
                let what = rest[..at_idx].trim().to_string();
                let where_near = rest[at_idx + 4..].trim();
                let where_near = where_near
                    .trim_start_matches("the ")
                    .trim()
                    .to_string();
                return Some(ParsedCommand::BuildStructure {
                    what,
                    where_near: Some(where_near),
                });
            }
            let what = rest.trim().to_string();
            if !what.is_empty() {
                return Some(ParsedCommand::BuildStructure {
                    what,
                    where_near: None,
                });
            }
        }
    }
    None
}

fn try_parse_zone(lower: &str) -> Option<ParsedCommand> {
    // "zone X as Y", "zone X", "rezone X"
    for prefix in &["zone ", "rezone "] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let rest = rest.trim();
            if let Some(as_idx) = rest.find(" as ") {
                let location = rest[..as_idx].trim().to_string();
                let zone_type = rest[as_idx + 4..].trim().to_string();
                return Some(ParsedCommand::ModifyZone {
                    zone_type,
                    location: Some(location),
                });
            }
            // "zone residential near downtown"
            if let Some(near_idx) = rest.find(" near ") {
                let zone_type = rest[..near_idx].trim().to_string();
                let location = rest[near_idx + 6..].trim().to_string();
                return Some(ParsedCommand::ModifyZone {
                    zone_type,
                    location: Some(location),
                });
            }
            let zone_type = rest.to_string();
            if !zone_type.is_empty() {
                return Some(ParsedCommand::ModifyZone {
                    zone_type,
                    location: None,
                });
            }
        }
    }
    None
}

fn try_parse_overlay(lower: &str) -> Option<ParsedCommand> {
    // "show X overlay", "show X map", "display X overlay", "show X"
    for prefix in &["show ", "display ", "toggle ", "enable "] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let rest = rest.trim();
            // Must contain "overlay" or "map" or "heatmap" to be an overlay command
            if rest.contains("overlay") || rest.contains(" map") || rest.contains("heatmap") {
                let overlay = rest
                    .replace("overlay", "")
                    .replace("heatmap", "heat")
                    .replace(" map", "")
                    .trim()
                    .to_string();
                if !overlay.is_empty() {
                    return Some(ParsedCommand::ShowOverlay {
                        overlay_type: overlay,
                    });
                }
            }
            // "show traffic" alone is also an overlay
            let known_overlays = [
                "traffic",
                "pollution",
                "crime",
                "happiness",
                "density",
                "power",
                "water",
                "land value",
                "zoning",
            ];
            for ov in &known_overlays {
                if rest == *ov || rest.starts_with(&format!("{} ", ov)) {
                    return Some(ParsedCommand::ShowOverlay {
                        overlay_type: ov.to_string(),
                    });
                }
            }
        }
    }
    None
}

fn try_parse_camera(lower: &str) -> Option<ParsedCommand> {
    // "go to X", "show me X", "fly to X", "look at X", "navigate to X"
    for prefix in &["go to ", "fly to ", "navigate to ", "move to ", "jump to "] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let target = rest.trim_start_matches("the ").trim().to_string();
            if !target.is_empty() {
                return Some(ParsedCommand::CameraMove {
                    target_description: target,
                });
            }
        }
    }
    if let Some(rest) = lower.strip_prefix("show me ") {
        let target = rest.trim_start_matches("the ").trim().to_string();
        if !target.is_empty() {
            return Some(ParsedCommand::CameraMove {
                target_description: target,
            });
        }
    }
    if let Some(rest) = lower.strip_prefix("look at ") {
        let target = rest.trim_start_matches("the ").trim().to_string();
        if !target.is_empty() {
            return Some(ParsedCommand::CameraMove {
                target_description: target,
            });
        }
    }
    None
}

fn try_parse_query(lower: &str) -> Option<ParsedCommand> {
    // "why X", "what X", "how X", "tell me about X", "explain X"
    for prefix in &["why ", "what ", "how ", "where ", "when ", "who "] {
        if lower.starts_with(prefix) {
            return Some(ParsedCommand::QueryInfo {
                question: lower.to_string(),
            });
        }
    }
    for prefix in &["tell me about ", "explain "] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            return Some(ParsedCommand::QueryInfo {
                question: rest.trim().to_string(),
            });
        }
    }
    None
}
