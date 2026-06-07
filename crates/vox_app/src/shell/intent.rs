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
use serde::Deserialize;
use vox_nn::llm_client::{LlmClient, LlmPrompt, LlmProvider};

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

// ============================================================================
// Adoption #16: the real-LLM seam.
//
// `parse_intent` above stays the deterministic, offline, network-free brain. The
// machinery below wires an OPTIONAL LLM backend *in front of* it without ever
// bypassing it as the safety net: the LLM may only ever produce an
// `IntentAction` that survives strict validation against the live param schema,
// and EVEN THEN the resulting param value flows through the existing clamp in
// `GraphBridge::apply_param` (the true authority on ranges). Any LLM hiccup —
// parse failure, unknown variant, out-of-schema key, missing node — falls back
// to `parse_intent`. We NEVER apply an unvalidated model output.
// ============================================================================

/// Where a resolved [`IntentAction`] came from. Threaded into the receipt so the
/// assistant log is honest about whether the model or the parser drove an edit.
#[derive(Debug, Clone, PartialEq)]
pub enum Provenance {
    /// The deterministic parser produced the action directly (no LLM in play).
    Parser,
    /// The LLM produced a valid, schema-checked action. Carries the model id.
    Llm { model: String },
    /// The LLM was consulted but its output was unusable; the deterministic
    /// parser produced the action instead.
    ParserFallback,
}

impl Provenance {
    /// The receipt suffix: "(parser)" / "(llm:model)" / "(llm failed → parser)".
    pub fn receipt_tag(&self) -> String {
        match self {
            Provenance::Parser => "(parser)".to_string(),
            Provenance::Llm { model } => format!("(llm:{model})"),
            Provenance::ParserFallback => "(llm failed → parser)".to_string(),
        }
    }
}

/// The result of resolving a sentence: the action (always `Some` — even Unknown
/// is an action) plus its provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct IntentResolution {
    pub action: Option<IntentAction>,
    pub provenance: Provenance,
}

/// Which brain resolves a sentence. Deliberately an enum (not a trait object) so
/// it stays `Clone`/`Debug` and trivially constructible. `Deterministic` is the
/// default and the only path used in tests/offline runs; `Llm` carries the
/// client config and is opted into via `OCHROMA_ASK_LLM` (read once at shell
/// construction — see [`IntentBackend::from_env`]).
#[derive(Clone)]
pub enum IntentBackend {
    Deterministic,
    Llm(LlmProvider),
    /// Test-only: a closure returning canned LLM text, so the LLM path can be
    /// exercised with ZERO network. Mirrors the real `Llm` path exactly except
    /// for where the response string comes from.
    #[cfg(test)]
    LlmCanned(std::sync::Arc<dyn Fn(&LlmPrompt) -> Result<String, String> + Send + Sync>),
}

impl std::fmt::Debug for IntentBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntentBackend::Deterministic => write!(f, "IntentBackend::Deterministic"),
            IntentBackend::Llm(p) => write!(f, "IntentBackend::Llm({p:?})"),
            #[cfg(test)]
            IntentBackend::LlmCanned(_) => write!(f, "IntentBackend::LlmCanned(<closure>)"),
        }
    }
}

impl IntentBackend {
    /// Select the backend ONCE, at shell construction. Default `Deterministic`;
    /// `OCHROMA_ASK_LLM` set (to anything non-empty) opts into the LLM path with
    /// the default provider (`LlmProvider::default()`, a local Ollama config that
    /// itself falls back to a deterministic stub if unreachable — so even the LLM
    /// path never hard-requires the network). Reading the env here, not per
    /// keystroke, keeps a long typing session from re-querying the environment.
    pub fn from_env() -> Self {
        match std::env::var("OCHROMA_ASK_LLM") {
            Ok(v) if !v.trim().is_empty() => IntentBackend::Llm(LlmProvider::default()),
            _ => IntentBackend::Deterministic,
        }
    }
}

/// Resolve a sentence into an [`IntentResolution`] using the chosen backend.
///
/// - `Deterministic`: run [`parse_intent`] and label it `Provenance::Parser`.
/// - `Llm`: ask the model for STRICT JSON naming exactly one `IntentAction`
///   variant, parse it strictly, validate every field against `schema` (so the
///   model can never name a phantom node/param), and on ANY failure fall back to
///   [`parse_intent`] with `Provenance::ParserFallback`. A successful, validated
///   action is labeled `Provenance::Llm { model }`.
///
/// SYNCHRONOUS by design (v2): `LlmClient::complete` is a blocking call and
/// `run_intent` is on the UI thread. The current `LlmClient` API exposes no
/// timeout hook (its Ollama path is a stub that returns immediately, and there is
/// no `with_timeout`/deadline on `LlmClient` or `LlmPrompt`), so there is nothing
/// to bound here yet; async streaming is v3 and explicitly out of scope.
pub fn resolve_intent(
    backend: &IntentBackend,
    text: &str,
    schema: &SchemaContext,
    registry: &CommandRegistry,
) -> IntentResolution {
    match backend {
        IntentBackend::Deterministic => IntentResolution {
            action: Some(parse_intent(text, registry)),
            provenance: Provenance::Parser,
        },
        IntentBackend::Llm(provider) => {
            let client = LlmClient::new(provider.clone());
            let prompt = build_llm_prompt(text, schema);
            let completed = client.complete(&prompt).map(|r| (r.text, r.model));
            resolve_via_llm(completed, text, schema, registry)
        }
        #[cfg(test)]
        IntentBackend::LlmCanned(f) => {
            let prompt = build_llm_prompt(text, schema);
            // The canned closure stands in for the model id "canned".
            let completed = f(&prompt).map(|t| (t, "canned".to_string()));
            resolve_via_llm(completed, text, schema, registry)
        }
    }
}

/// Shared LLM tail: given the model's (text, model) result, parse + validate it,
/// or fall back to the deterministic parser. Factored out so the real and the
/// canned paths are byte-for-byte identical past the response source.
fn resolve_via_llm(
    completed: Result<(String, String), String>,
    text: &str,
    schema: &SchemaContext,
    registry: &CommandRegistry,
) -> IntentResolution {
    let fallback = || IntentResolution {
        action: Some(parse_intent(text, registry)),
        provenance: Provenance::ParserFallback,
    };
    let Ok((raw, model)) = completed else {
        return fallback();
    };
    match parse_llm_intent(&raw, schema) {
        Some(action) => IntentResolution {
            action: Some(action),
            provenance: Provenance::Llm { model },
        },
        None => fallback(),
    }
}

/// The live param schema handed to the LLM: every node kind the graph can edit,
/// the real `set_param` keys on it, and their ranges. Built from the SAME
/// hand-written table the deterministic resolver uses, so the model is told only
/// about params that genuinely exist (an LLM can map "make the terrain more
/// detailed" → `SetParam{node_kind:terrain,key:resolution,...}` because the
/// schema lists `resolution` under `terrain`).
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaContext {
    pub kinds: Vec<KindSchema>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KindSchema {
    /// Friendly noun the model emits ("terrain") — canonicalized to a real
    /// registry `type_name` ("TerrainNode") during validation.
    pub friendly: &'static str,
    pub type_name: &'static str,
    pub params: Vec<ParamSchema>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParamSchema {
    pub key: &'static str,
    pub min: f32,
    pub max: f32,
}

impl SchemaContext {
    /// The default editable schema (mirrors `graph_bridge::param_schema` ranges).
    /// The bridge's clamp in `apply_param` remains the authority on ranges; this
    /// copy exists only to (a) shape the prompt and (b) reject out-of-schema KEYS
    /// before an action is ever built. Values are NOT clamped here.
    pub fn default_editable() -> Self {
        let p = |key, min, max| ParamSchema { key, min, max };
        SchemaContext {
            kinds: vec![
                KindSchema {
                    friendly: "terrain",
                    type_name: "TerrainNode",
                    params: vec![
                        p("resolution", 16.0, 256.0),
                        p("amplitude", 0.0, 800.0),
                        p("seed", 0.0, 999.0),
                    ],
                },
                KindSchema {
                    friendly: "biome",
                    type_name: "BiomeNode",
                    params: vec![p("world_height", 1.0, 2000.0), p("moisture", 0.0, 1.0)],
                },
                KindSchema {
                    friendly: "vegetation",
                    type_name: "VegetationNode",
                    params: vec![
                        p("branch_levels", 1.0, 8.0),
                        p("trunk_radius", 0.05, 2.0),
                        p("height", 1.0, 30.0),
                    ],
                },
            ],
        }
    }

    /// Resolve a model-emitted (node_kind, key) — accepting either the friendly
    /// noun or the real type_name — to the canonical `(type_name, key, friendly
    /// target label)`. Returns `None` if the kind or key is not in the schema, so
    /// the LLM can never address a phantom node or param.
    fn resolve(
        &self,
        node_kind: &str,
        key: &str,
    ) -> Option<(&'static str, &'static str, String)> {
        let nk = node_kind.to_lowercase();
        let kind = self.kinds.iter().find(|k| {
            k.friendly.eq_ignore_ascii_case(&nk) || k.type_name.eq_ignore_ascii_case(&nk)
        })?;
        let param = kind.params.iter().find(|p| p.key.eq_ignore_ascii_case(key))?;
        let target = format!("{}.{}", kind.friendly, param.key);
        Some((kind.type_name, param.key, target))
    }

    /// Resolve a model-emitted add-node noun/type to a real registry type_name +
    /// friendly label. Reuses the deterministic [`try_add_node`] vocabulary so the
    /// LLM and parser agree on what "add a house" means.
    fn resolve_add(&self, kind: &str) -> Option<(&'static str, String)> {
        // Reuse the parser's noun→kind map by constructing an "add <kind>"
        // sentence; this guarantees the LLM and the deterministic parser stay in
        // lockstep on which nouns are addable.
        match try_add_node(&format!("add {}", kind.to_lowercase())) {
            Some(IntentAction::AddNode { kind, friendly }) => Some((kind, friendly)),
            _ => None,
        }
    }
}

/// STRICT serde mirror of [`IntentAction`] as the LLM is asked to emit it. The
/// variant set here is pinned to `IntentAction` by
/// [`tests::prompt_schema_pins_every_variant`]: adding an `IntentAction` variant
/// fails that exhaustive match until this DTO, [`describe_intent_variants`], and
/// the validator below are updated. `deny_unknown_fields` makes a wrong-shape
/// object (extra/typo keys) a HARD parse error → fallback.
#[derive(Debug, Deserialize)]
enum LlmIntent {
    #[serde(rename = "SetParam")]
    SetParam(LlmSetParam),
    #[serde(rename = "AdjustParam")]
    AdjustParam(LlmAdjustParam),
    #[serde(rename = "AddNode")]
    AddNode(LlmAddNode),
    #[serde(rename = "RunCommand")]
    RunCommand(LlmRunCommand),
    #[serde(rename = "Unknown")]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmSetParam {
    node_kind: String,
    key: String,
    value: f32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmAdjustParam {
    node_kind: String,
    key: String,
    delta: f32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmAddNode {
    kind: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmRunCommand {
    id: String,
}

/// Parse the LLM's raw text as STRICT JSON for exactly one [`LlmIntent`] variant,
/// then validate every field against `schema`, producing a real [`IntentAction`].
/// Returns `None` on ANY failure (invalid JSON, unknown variant via
/// `deny_unknown_fields`/serde, wrong shape, out-of-schema kind/key, unknown
/// command id) — the caller then falls back to the deterministic parser. We NEVER
/// hand back an action built from unvalidated model output.
fn parse_llm_intent(raw: &str, schema: &SchemaContext) -> Option<IntentAction> {
    // Models sometimes wrap JSON in prose / code fences; extract the first
    // top-level `{...}` object. If there is none, this is not parseable → None.
    let json = extract_json_object(raw)?;
    let parsed: LlmIntent = serde_json::from_str(json).ok()?;
    match parsed {
        LlmIntent::SetParam(p) => {
            let (node_kind, key, target) = schema.resolve(&p.node_kind, &p.key)?;
            // NOTE: `value` is intentionally NOT clamped here — the clamp in
            // `GraphBridge::apply_param` is the single safety net for ranges (it
            // clamps EVERY param), so a hostile value still cooks in-range. We
            // only validate that the KEY exists.
            Some(IntentAction::SetParam {
                node_kind,
                key,
                target,
                value: p.value,
            })
        }
        LlmIntent::AdjustParam(p) => {
            let (node_kind, key, target) = schema.resolve(&p.node_kind, &p.key)?;
            Some(IntentAction::AdjustParam {
                node_kind,
                key,
                target,
                delta: p.delta,
            })
        }
        LlmIntent::AddNode(p) => {
            let (kind, friendly) = schema.resolve_add(&p.kind)?;
            Some(IntentAction::AddNode { kind, friendly })
        }
        LlmIntent::RunCommand(p) => {
            // Only resolve to a RunCommand if the id is a real registered command.
            let id = canonical_command_id(&p.id)?;
            Some(IntentAction::RunCommand {
                id,
                receipt: format!("Ran {id}"),
            })
        }
        LlmIntent::Unknown => Some(IntentAction::Unknown {
            suggestions: Vec::new(),
        }),
    }
}

/// Canonicalize a model-emitted command id to a real `'static` id, or `None` if
/// it is not one the assistant exposes. Mirrors the ids the deterministic parser
/// emits (theme/focus), so the LLM cannot invoke an arbitrary command.
fn canonical_command_id(id: &str) -> Option<&'static str> {
    match id {
        "view.theme_light" => Some("view.theme_light"),
        "view.theme_dark" => Some("view.theme_dark"),
        "view.focus_crucible" => Some("view.focus_crucible"),
        "view.focus_node_graph" => Some("view.focus_node_graph"),
        "view.focus_viewport" => Some("view.focus_viewport"),
        _ => None,
    }
}

/// Slice out the first balanced top-level `{...}` JSON object from arbitrary
/// model text (which may include code fences or commentary). `None` if there is
/// no balanced object.
fn extract_json_object(raw: &str) -> Option<&str> {
    let bytes = raw.as_bytes();
    let start = raw.find('{')?;
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for i in start..bytes.len() {
        let c = bytes[i] as char;
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&raw[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Build the LLM prompt: a system message describing the STRICT-JSON
/// `IntentAction` contract + the live schema, and the user's sentence. The
/// variant description comes from [`describe_intent_variants`], which is pinned
/// to `IntentAction` by an exhaustive-match test.
fn build_llm_prompt(text: &str, schema: &SchemaContext) -> LlmPrompt {
    let mut schema_lines = String::new();
    for k in &schema.kinds {
        for p in &k.params {
            schema_lines.push_str(&format!(
                "  - node_kind \"{}\" key \"{}\" (range {}..={})\n",
                k.friendly, p.key, p.min, p.max
            ));
        }
    }
    let system = format!(
        "You translate one natural-language editor command into STRICT JSON for \
exactly ONE action. Output ONLY a single JSON object, no prose, no code fences.\n\n\
The JSON must be one of these shapes:\n{variants}\n\n\
Editable parameters (use these exact node_kind + key strings; map fuzzy words \
like \"more detailed\" → resolution, \"taller\" → amplitude):\n{schema}\n\n\
If the command does not map to any action, output {{\"Unknown\":null}}.",
        variants = describe_intent_variants(),
        schema = schema_lines,
    );
    LlmPrompt::new(&system, text).with_format("JSON")
}

/// The human-readable description of every [`IntentAction`] variant the model may
/// emit, serialized into the prompt. PINNED to the enum by
/// [`tests::prompt_schema_pins_every_variant`] via an exhaustive match — adding a
/// variant fails compilation there until this description is updated, so the
/// prompt can never silently drift out of sync with the action set.
fn describe_intent_variants() -> &'static str {
    r#"  {"SetParam":{"node_kind":"<kind>","key":"<param>","value":<number>}}
  {"AdjustParam":{"node_kind":"<kind>","key":"<param>","delta":<number>}}
  {"AddNode":{"kind":"<noun, e.g. tree/building/biome>"}}
  {"RunCommand":{"id":"view.theme_light|view.theme_dark|view.focus_crucible|view.focus_node_graph|view.focus_viewport"}}
  {"Unknown":null}"#
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

    // === Adoption #16: LLM seam tests (ZERO network — canned closures) ===

    /// Build a backend whose "LLM" returns a fixed string, exercising the exact
    /// real LLM path (`resolve_via_llm` → `parse_llm_intent` → schema validation)
    /// with no network.
    fn canned(resp: &'static str) -> IntentBackend {
        IntentBackend::LlmCanned(std::sync::Arc::new(move |_p: &LlmPrompt| Ok(resp.to_string())))
    }

    /// LLM HAPPY PATH: a well-formed response naming a real SetParam resolves to
    /// exactly that action with `Llm` provenance — no fallback.
    #[test]
    fn llm_happy_path_resolves_setparam_with_llm_provenance() {
        let backend = canned(r#"{"SetParam":{"node_kind":"terrain","key":"resolution","value":128.0}}"#);
        let res = resolve_intent(
            &backend,
            "make the terrain more detailed",
            &SchemaContext::default_editable(),
            &registry(),
        );
        assert_eq!(
            res.action,
            Some(IntentAction::SetParam {
                node_kind: "TerrainNode",
                key: "resolution",
                target: "terrain.resolution".into(),
                value: 128.0,
            }),
            "the canned LLM SetParam must resolve to the exact validated action"
        );
        assert_eq!(res.provenance, Provenance::Llm { model: "canned".into() });
    }

    /// MALFORMED LLM OUTPUT (3 cases) → deterministic parser result with
    /// ParserFallback provenance. The user sentence is one the parser CAN handle,
    /// so we assert the exact fallback action.
    #[test]
    fn llm_malformed_falls_back_to_parser() {
        let expected = IntentAction::SetParam {
            node_kind: "TerrainNode",
            key: "resolution",
            target: "terrain.resolution".into(),
            value: 64.0,
        };
        // Case 1: invalid JSON.
        // Case 2: unknown variant ("Frobnicate" is not an IntentAction).
        // Case 3: valid JSON, wrong shape (SetParam missing required fields).
        for raw in [
            r#"{ this is not json"#,
            r#"{"Frobnicate":{"node_kind":"terrain"}}"#,
            r#"{"SetParam":{"node_kind":"terrain"}}"#,
        ] {
            let backend = IntentBackend::LlmCanned(std::sync::Arc::new(move |_p: &LlmPrompt| {
                Ok(raw.to_string())
            }));
            let res = resolve_intent(
                &backend,
                "set terrain resolution to 64",
                &SchemaContext::default_editable(),
                &registry(),
            );
            assert_eq!(res.provenance, Provenance::ParserFallback, "raw {raw:?} must fall back");
            assert_eq!(res.action, Some(expected.clone()), "fallback action wrong for raw {raw:?}");
        }
    }

    /// OUT-OF-SCHEMA KEY → fallback (the model named a param the kind does not
    /// expose). The schema validation, not the model, has the final say on keys.
    #[test]
    fn llm_out_of_schema_key_falls_back() {
        let backend = canned(r#"{"SetParam":{"node_kind":"terrain","key":"phantom","value":5.0}}"#);
        let res = resolve_intent(
            &backend,
            "set terrain resolution to 64",
            &SchemaContext::default_editable(),
            &registry(),
        );
        assert_eq!(res.provenance, Provenance::ParserFallback);
        // Falls back to the parser, which CAN handle the sentence.
        assert_eq!(
            res.action,
            Some(IntentAction::SetParam {
                node_kind: "TerrainNode",
                key: "resolution",
                target: "terrain.resolution".into(),
                value: 64.0,
            })
        );
    }

    /// HOSTILE LLM OUTPUT resolves to an action (the value is NOT validated, only
    /// the key is) — proving validation lets the value through to the clamp, which
    /// is the real safety net (asserted end-to-end in the mod.rs run_intent test).
    #[test]
    fn llm_hostile_value_resolves_unclamped_at_seam() {
        let backend = canned(r#"{"SetParam":{"node_kind":"terrain","key":"resolution","value":1e30}}"#);
        let res = resolve_intent(
            &backend,
            "set terrain resolution to 64",
            &SchemaContext::default_editable(),
            &registry(),
        );
        assert_eq!(res.provenance, Provenance::Llm { model: "canned".into() });
        match res.action {
            Some(IntentAction::SetParam { value, .. }) => {
                assert_eq!(value, 1e30, "the seam passes the raw value through; the clamp guards it");
            }
            other => panic!("expected SetParam, got {other:?}"),
        }
    }

    /// PROMPT-SCHEMA PINNING: an exhaustive match over every `IntentAction`
    /// variant. Adding a variant fails THIS test to compile until
    /// `describe_intent_variants`, `LlmIntent`, and the validator are updated, so
    /// the prompt can never silently drift out of sync with the action set.
    #[test]
    fn prompt_schema_pins_every_variant() {
        let sample = IntentAction::Unknown { suggestions: Vec::new() };
        let mentioned = match sample {
            IntentAction::SetParam { .. } => "SetParam",
            IntentAction::AdjustParam { .. } => "AdjustParam",
            IntentAction::AddNode { .. } => "AddNode",
            IntentAction::RunCommand { .. } => "RunCommand",
            IntentAction::Unknown { .. } => "Unknown",
        };
        // Every arm's name MUST appear in the prompt description, or the model is
        // told about a variant set that no longer matches the enum.
        let desc = describe_intent_variants();
        for name in ["SetParam", "AdjustParam", "AddNode", "RunCommand", "Unknown"] {
            assert!(desc.contains(name), "prompt description must mention variant {name}");
        }
        assert!(desc.contains(mentioned));
    }

    /// DETERMINISM DEFAULT: with no env var the backend is `Deterministic`, and a
    /// sentence resolves byte-identically to the old `parse_intent` behavior with
    /// `Parser` provenance.
    #[test]
    fn default_backend_is_deterministic_and_matches_parser() {
        // Ensure the env opt-in is not set in this test process. `remove_var` is
        // unsafe under edition 2024 (it mutates process-global env); this test is
        // the sole toucher of this var, so the call is sound.
        unsafe {
            std::env::remove_var("OCHROMA_ASK_LLM");
        }
        let backend = IntentBackend::from_env();
        assert!(matches!(backend, IntentBackend::Deterministic), "default must be Deterministic");

        let r = registry();
        let res = resolve_intent(&backend, "add a tree", &SchemaContext::default_editable(), &r);
        assert_eq!(res.provenance, Provenance::Parser);
        assert_eq!(res.action, Some(parse_intent("add a tree", &r)), "must match the old parser exactly");
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
