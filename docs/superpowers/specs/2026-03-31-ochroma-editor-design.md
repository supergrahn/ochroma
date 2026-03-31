# Design: Ochroma Editor UI/UX (2026-03-31)

**Status:** Approved
**Scope:** Complete editor UI/UX design for the Ochroma engine editor — mode system, AI integration, scene navigation, asset library, and all advanced features across three delivery phases.
**Related:** Domain 09 Editor Plan (forthcoming)

---

## 1. Problem Statement

- No Ochroma editor exists yet; without a clear design contract, implementation will produce an incoherent set of panels that don't reinforce each other.
- Traditional game editors (Unreal, Unity) separate tool panels from workflow — the UI doesn't change based on what you're doing, so the user is always managing panel state manually.
- AI-generation workflows have no natural home in existing editor paradigms; they're bolted on as chat windows that don't know what the user is looking at.
- Procedural node graphs are invisible to beginners and navigated via menus by experts — neither audience is well served.
- Asset discovery requires knowing filenames; entirely blocks iteration when an asset doesn't exist yet.

---

## 2. Done When

**Phase 1:** Running `cargo run --bin ochroma` opens an editor window where a developer can:
1. Switch between all 6 modes via the mode strip and observe the context panel change accordingly.
2. Select an object, type a prompt in the AI bar, and see the scene update.
3. See a pulsing "N nodes changed" badge appear in the context panel after the AI action.
4. Click the badge and see the node graph slide up with AI-modified nodes highlighted in blue.
5. Search for an asset using a description ("a rock that feels ancient and menacing") and get results.
6. Hit Play/Simulate and see NPC path ghost overlays drawn in the viewport.

A human at the keyboard can verify each of the above visually without reading code.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Mode strip switches workspace | Switching to Lighting mode makes the lighting context panel appear and Sculpt tools disappear | Panel container exists but contents unchanged |
| AI bar scope toggle | Scope set to "selection" sends only selected object's data to LLM; scope "scene" sends full scene graph | AI bar renders a toggle UI that does nothing |
| Node graph badge | AI action on a 5-node graph increments badge count to 5; clicking reveals graph with 5 highlighted nodes | Badge always shows "3 nodes changed" |
| Semantic asset search | Search "ancient menacing rock" returns results ranked by LVM embedding cosine similarity > 0.6 | Search returns the same asset regardless of query |
| Ghost overlays | Simulate mode renders semi-transparent NPC paths drawn from actual NavMesh pathfinding data | Ghost overlay is a static sprite |
| Intent narrative | Scene tree node for an AI-generated building shows its creation prompt as subtitle text | Subtitle shows filename |

---

## 4. Architecture

### 4.1 Mode Strip + Context-Adaptive Workspace

The editor is organized around **6 workflow modes**: Sculpt, Objects, Lighting, Animate, Logic, Simulate. The mode strip is a vertical icon bar on the left edge of the viewport. Switching modes triggers a `WorkspaceMode` event that the context panel and floating tool strip both subscribe to.

The **context panel** is a right-side drawer that changes its entire contents based on the active mode and the current selection. It is never a fixed set of tabs — it is a reactive surface. The three-tab **right sidebar** (Context / Scene / Assets) lives within this drawer: Context shows selection-specific tools for the active mode, Scene is the scene hierarchy, Assets is the browsable/searchable asset library.

The **floating tool strip** appears inside the viewport near the selection and contains the 5–8 most common actions for the current mode. It does not persist after deselection.

There is no panel layout configuration. The workspace is not user-customizable at Phase 1 — the design adapts itself.

### 4.2 AI Bar

The AI bar occupies a fixed strip at the bottom of the editor window. It is **always visible** — it cannot be closed, only minimized to a single line (input field + scope badge + context badges).

**Scope toggle** (left of input): cycles through Selection → Mode → Scene. Controls what context is sent with each prompt. Badge shows current scope clearly.

**Context badges** (right of input): small read-only pills showing what the AI currently knows about — selected object name, active mode, any active sensory palette. Always visible even when collapsed.

**Expand button**: slides the bar upward to reveal conversation history sidebar (left) and a large input area (right). From the expanded state the user can also access the **Sensory Palettes** canvas (Phase 2).

If the user somehow hides the bar (future: user preference), a `✦` tab on the bottom edge restores it instantly.

### 4.3 Intent Narrative Hierarchy

The scene tree (Scene tab in right sidebar) renders each object with two lines: the object's name (large, white) and the AI prompt that created it (small, muted). This prompt is stored on the `SceneNode` as an `Option<String>` field named `intent`. Manually created objects have no intent subtitle.

The hierarchy is a readable history of authorship, not just a file list.

### 4.4 Node Graph On-Demand Reveal

After every AI action that modifies a procedural asset's node graph, a **badge** appears in the context panel showing "N nodes changed" with a pulsing animation. The badge is the only indication the graph was touched — beginners can ignore it entirely.

Clicking the badge triggers a **slide-up panel** from the bottom of the context panel that reveals the full node graph for the selected asset. Nodes added or modified by the AI are highlighted in blue with `+` badges. The user can edit the graph directly from this state. Closing the panel returns to normal context view without losing edits.

The graph is never shown unless the user clicks the badge. The workflow is:

```
AI action → badge appears → user clicks → graph slides up with highlights → user edits or closes
```

### 4.5 Semantic Asset Library (Phase 1 — Gemini idea)

The Assets tab replaces filename-based browsing with a search-first interface. The search bar accepts natural language. Queries are embedded using an LVM and matched against pre-computed embeddings for all assets in the library.

If no result exceeds a similarity threshold, the library shows an "AI Generate" option that sends the query to the generation backend and returns a new asset. The user is never blocked.

### 4.6 Ghost Overlays (Phase 1 — Gemini idea)

In Simulate mode, the viewport renders semi-transparent overlay geometry derived from live simulation data:

- **NPC path ghosts**: each NavMesh agent draws its last 2 seconds of path history (fading trail) and its projected next 2 seconds (dotted line with weight annotations).
- **Physics stress**: objects under stress receive a colour grade overlay (blue→red by stress magnitude).
- **Logic flow**: connected logical objects (triggers, volumes, etc.) pulse light along their connection lines when the logic fires.

These overlays are viewport-layer composites — they do not affect the scene or simulation state.

### 4.7 Phase 2: Intelligence Layer

**Branching Timeline** — scene history rendered as a river delta. Each AI action or manual edit creates a node. "What if?" branches are first-class objects. Elements can be spliced between branches. Consequence editing: define an outcome ("bombed building"), AI simulates the history and aftermath.

**Proceduralism as Instrument** — any procedural asset's node graph can be surfaced as a 2D performance pad. The AI analyses the graph, identifies the most expressive 2 parameters, and maps them to X/Y axes. Dragging the pad changes the asset in real time.

**Sensory Palettes** — a mood board canvas accessible from the AI bar's expanded state. Drag in photos, audio clips, or text. The LVM watches it as a continuous live prompt. New objects created while a palette is active inherit its visual/textural character.

**Director's Gaze** — a toggleable cinematography overlay in Lighting/Cinematic mode. Adapts rule-of-thirds and golden spiral guides to current camera framing. LVM analyses the viewport continuously and provides real-time framing feedback in the AI bar.

### 4.8 Phase 3: Mastery Layer

**Resonance Testing** (Gemini) — runs 1000+ NavMesh + CrowdSim agents through the level with varied playstyle parameters. Produces a behavioural heatmap: engagement density, stuck points, unexplored areas. Rendered as a viewport overlay in Simulate mode. Uses actual Ochroma simulation — not a prediction model.

**Spectral Intent Diffusion** (Ochroma-native) — extends Sensory Palettes to target specific spectral wavelength bands. "Make this area feel colder" shifts material spectral emission curves toward the blue end across all selected or in-view materials. The edit is physically measurable, not just a colour grade.

**Consequence Simulation Preview** — before committing a Branching Timeline consequence edit, the AI runs a preview branch and renders the aftermath as a semi-transparent ghost overlay on the current scene. The user approves or rejects before the branch is committed.

---

## 5. Data Models

```rust
/// Workflow mode; controls which context panel and tool strip are active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceMode {
    Sculpt,
    Objects,
    Lighting,
    Animate,
    Logic,
    Simulate,
}

/// AI bar scope — controls what context is sent with each prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiScope {
    Selection,
    Mode,
    Scene,
}

/// Persisted on every scene node; records the AI prompt that created it, if any.
pub struct SceneNode {
    id: NodeId,
    name: String,
    intent: Option<String>,   // AI creation prompt, shown as subtitle in hierarchy
    children: Vec<NodeId>,
    // ... transform, components, etc.
}

impl SceneNode {
    pub fn intent(&self) -> Option<&str> { self.intent.as_deref() }
    pub fn set_intent(&mut self, prompt: impl Into<String>) {
        self.intent = Some(prompt.into());
    }
}

/// Diff produced after an AI action that modifies a node graph asset.
pub struct NodeGraphDiff {
    asset_id: AssetId,
    added_nodes: Vec<NodeId>,
    modified_nodes: Vec<NodeId>,
}

impl NodeGraphDiff {
    pub fn changed_count(&self) -> usize {
        self.added_nodes.len() + self.modified_nodes.len()
    }
}

/// An entry in the asset library with a pre-computed embedding for semantic search.
pub struct AssetEntry {
    id: AssetId,
    name: String,
    embedding: Vec<f32>,      // LVM embedding, length fixed by model (e.g. 1536)
}
```

---

## 6. API

The editor UI layer communicates with the engine through an event bus. Key editor-side APIs:

```rust
// Switch active workspace mode. Broadcasts WorkspaceModeChanged event.
pub fn set_mode(mode: WorkspaceMode);

// Submit a prompt to the AI bar. Returns a job handle; result arrives via AiActionComplete event.
pub fn submit_ai_prompt(
    prompt: &str,
    scope: AiScope,
    selection: Option<&[NodeId]>,
) -> JobHandle;

// Query the asset library by natural language. Returns ranked results.
// If results.is_empty() after this call, callers should offer AI generation.
pub fn search_assets(query: &str) -> Vec<(AssetEntry, f32 /* similarity score */)>;

// Open the node graph reveal panel for the given asset, highlighting the provided diff.
pub fn reveal_node_graph(asset_id: AssetId, diff: &NodeGraphDiff);

// Toggle ghost overlays on/off in the viewport (Simulate mode only).
pub fn set_ghost_overlays_enabled(enabled: bool);
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `set_mode()` | Mode strip button click handler | `crates/vox_app/src/editor/mode_strip.rs` | Called on every mode button press |
| `WorkspaceModeChanged` event | Context panel, floating tool strip | `crates/vox_app/src/editor/context_panel.rs` | Subscribed at editor startup |
| `submit_ai_prompt()` | AI bar submit handler | `crates/vox_app/src/editor/ai_bar.rs` | Fires on Enter or button click |
| `AiActionComplete` event | Node graph badge | `crates/vox_app/src/editor/context_panel.rs` | Badge appears if `diff.changed_count() > 0` |
| `reveal_node_graph()` | Badge click handler | `crates/vox_app/src/editor/node_graph_panel.rs` | Triggers slide-up animation |
| `search_assets()` | Assets tab search bar | `crates/vox_app/src/editor/asset_library.rs` | Called on every keystroke (debounced 150ms) |
| `set_ghost_overlays_enabled()` | Simulate mode activation | `crates/vox_app/src/editor/mode_strip.rs` | Enabled when entering Simulate, disabled on exit |
| `SceneNode::intent` | Scene tree renderer | `crates/vox_app/src/editor/scene_tree.rs` | Subtitle row rendered if `intent.is_some()` |

---

## 8. Open Questions

All questions resolved before this spec was written. No open items.

---

## 9. Out of Scope

- Custom panel layout / docking system — the workspace is not user-configurable at Phase 1.
- Multiplayer / collaborative editing — single-user editor only across all three phases.
- Plugin/scripting API for editor extensions — not addressed here.
- Specific LLM/LVM vendor selection — the design assumes an AI backend exists; integration is a separate concern.
- Mobile or web editor — `vox_app` targets desktop only.
- Keyboard shortcut system — important but a separate design.

---

## 10. Related Plans / Designs

- Depends on: NavMesh + CrowdSim (already built — `crates/vox_sim`)
- Depends on: Physics ECS (already built — `crates/vox_sim`)
- Depends on: Spectral material system (`crates/vox_render`)
- Required before: Domain 09 Editor implementation plan
- Related: Domain 12 Spectral Frontier (Spectral Intent Diffusion dependency)
