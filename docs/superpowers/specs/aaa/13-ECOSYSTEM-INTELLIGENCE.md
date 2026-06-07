# AAA Program — Ecosystem Intelligence Addendum (2026-06-07)

An Opus medium-effort scout surveyed three sibling repos against the AAA roadmap.
The finding: **several gaps the roadmap rated XL / far-off already have working,
tested solutions one directory over.** Every claim below is grounded in a real
file:symbol (cited). This addendum re-scopes the roadmap accordingly.

---

## rheos — the AI-native reasoning BRAIN (the ecosystem's taproot)

`~/src/rheos` is not on this workstation — the real project lives on the production
box at **`tomespensin:/home/tomespen/git/rheos`** (scouted read-only over ssh,
2026-06-07). The local "dashboard" was only its observability shadow; rheos itself
is far more.

**What it is:** a **modular-composition reasoning system** — "grow capability by
composing many small reasoners (experts) instead of scaling one monolith"
(`IDEA.md`). A learned ROUTER/orchestrator reasons about a task and dispatches to
domain expert submodels — file-finder, code-repair, math, sequence, reasoning —
then composes/validates their results (`ARCHITECTURE.md`). 1207 Mojo + 581 Python
files; a **Brain V4 cognition trainer** (oracle-surprise curriculum, concept-SFT,
hierarchical prediction); IID/OOD/structural-OOD/adversarial/composition-OOD eval
splits. It runs the **same gate-culture as Ochroma** — a promotion ladder
(`smoke20 → short gate → bounded 1337 → promote only if the rung survives`,
`INTELLIGENCE-ROADMAP.md`). **It serves inference**: `apps/agent/api_server_v1.py`
+ `inference_service_v1.py` + `parallel_inference_v1.py`, and a real
`swebench_agent_v1.py` (autonomous code repair).

**rheos is the architectural taproot:** lumengen *borrowed rheos's HPT / verifier /
substrate machinery* and stress-tested it in 3D (`rheos/NOTES_lumengen_cross_
pollination.md`). The two already cross-pollinate.

**Why it matters for Ochroma — it IS the real AI-native backend (#13) and the
"AI creates code" engine, and it's better than "plug in an LLM":** Ochroma's Ask
Ochroma already has the seam — `IntentBackend::Llm` → `resolve_intent` →
schema-validated `IntentAction` with deny-unknown-fields + clamp + deterministic
parser fallback (an unvalidated backend output can NEVER touch the graph). rheos
serves inference over an HTTP API. So **#13 "real LLM backend" = point the
LlmBackend at rheos's `api_server_v1`** — but instead of a monolithic LLM, the
backend is a *reasoning orchestrator composing domain experts* (a placement expert,
a script-gen/repair expert, a scene-design expert). The safety story is already
paid for by the seam we built. **Seam (M):** an HTTP `IntentBackend` variant +
rheos exposing an intent-resolution expert. rheos's code-repair expert is also the
natural upgrade for "AI creates code" (the rhai/script generation backed by a real
repair model, not just templates).

> Together the three siblings are the complete **AI-native creation stack**:
> **rheos = the brain** (reasoning/orchestration), **lumengen = the content**
> (photo/text→3D), **aetherspectra = the story** (scene brief + narrative sim).
> Ochroma is the spectral-splat **engine + editor** that renders it, relights it,
> and makes it playable + provable. A domain person describes a world → aetherspectra
> structures it → rheos orchestrates the build → lumengen generates the assets →
> Ochroma renders it spectrally in a windowed editor. Every piece exists in-house.

---

## lumengen — single-photo/text → dense 3D, with native Gaussian splats

**What:** a live, JAX/Equinox image-to-3D and text-to-3D **trainer + inference
library** (`lumengen/pyproject.toml`; README:6-7). Maturity: well past prototype —
734–1044 passing tests, real checkpoints on disk (`outputs/**/model.eqx`), a 372 GB
curated dataset, a dedicated training box. Public API
(`lumengen/src/lumengen/generation/__init__.py`):
- `infer_single_photo_to_atoms(...)` → `SinglePhotoToAtomsResult`
- `infer_text_to_atoms(...)`, `parse_text_prompt(...)`
- `write_single_photo_outputs(...)` → **GLB + OBJ + JSON** (`single_photo_to_atoms.py:198`)

It carries an **"atom" representation** that quantizes opacity + covariance-diagonal
(`atoms/format.py:279-300`) — *conceptually identical to Ochroma's `GaussianSplat`* —
and a native JAX splat rasterizer (`geometry/gaussian_splatting.py`).

**Why it matters:** it **de-risks the roadmap's most expensive content gap (#26/#29,
"3DGS training backend for capture", rated XL/far-off)** down to **M**. Unlike COLMAP
(sparse cloud), lumengen produces *dense* geometry from a *single* image or prompt.

**Seam (M):** a native-optional `lumengen_native.rs` (Crucible-pattern twin):
subprocess → GLB on disk → Ochroma's **existing live glTF importer** → splats →
plant. No Python-in-Rust bridge needed. Caveat: lumengen atoms are RGB/PBR → use
the existing Smits RGB→16-band upsample on import (approximate spectral until a
spectral-supervised lumengen variant trains).

**Deeper seam (L):** a direct `atom → GaussianSplat` codec (skip the mesh round-trip)
makes lumengen a **first-class spectral-splat content generator** — both sides are
in-house, so it's defensible.

---

## aetherspectra "story" — two subsystems, both load-bearing

**(A) Director / CreativeBrief (story → scene).** `schemas/creative_brief.py`:
`CreativeBrief { idea_summary, core_themes, world_description, emotional_journey,
set_list: [SetLocation{name, description, importance: hero|supporting|background,
mood, lighting_hints}], shot_language, season/time_of_day/weather }`. A **Gemini-backed
pipeline** turns a one-line idea into a full structured scene brief — proven
end-to-end by a real artifact: `farmhouse_idea.txt` → `farmhouse_idea_output.json`
(narrative_summary + terrain + lighting rig). This is **AI-native scene authoring
from a story prompt**, output as validated Pydantic/JSON.

→ Maps to the roadmap's biggest gameplay hole ("no quest/dialogue/narrative
frameworks") **and** the AI-native pillar (#6 multi-step Ask-Ochroma, #13 real LLM
backend). `CreativeBrief.set_list` is *the same shape* as the
`IntentAction::Plan(Vec<IntentAction>)` #6 wants. **Seam (M):** a deterministic
`CreativeBrief → Vec<IntentAction>` translator in `intent.rs`, reusing the existing
schema-validated, undoable IntentBackend. The Gemini director **is** the real-LLM
backend for #13.

**(B) `mantle` narrative-event engine (temporal story sim).**
`engines/mantle/.../schemas/narrative.py`: `NarrativeEventSpec` (war/fire/flood,
year, intensity, center, radius), `TimelineSpec`, `AffectiveMoodSpec`
(tension/melancholy/hope/danger), numba `EventBuffer`/`PrimHistoryBuffer` that
**ages and weathers world prims over simulated years** (`compute_damage_batch`),
already bound into rendering (`narrative_zones.py` → hero locations get denser GI).
The closest thing to a quest/event/world-state framework in the ecosystem.
**Seam (L):** service/offline — produces an event timeline Ochroma replays.

---

## FloraPrime native bridge — re-scoped: the blocker is DATA, not code

The roadmap (and our memory) said floraprime-native is blocked because "no trained
`.pt` checkpoints exist." Truer: **the trainer, model, and sampler contract all
exist and already match Ochroma's stub** — `floraprime_gen/train.py`
(`GraphDiffusion(node_dim=14)`, `torch.save(state_dict)`), `sample.py:19`
`sample_tree(checkpoint, species_id, crown_radius, n_nodes) -> (N,14) graph` ==
the signature behind `plugins.rs:565`'s `grow_tree_skeleton` stub. What's missing is
a **QSM training corpus** (`floraprime_gen/data/` has only the dataset *loader*, no
data, no committed `.pt`). Unblock = collect/point at a QSM dataset → run the
existing `train.py` → bridge replaces the stub. **S (train) + M (bridge)**, not XL.

---

## Re-scoped gap-closure map

| Sibling | Closes / de-risks | Pattern | Effort (was → now) |
|---|---|---|---|
| **lumengen** | #26/#29 content-supply (single-photo/text → dense 3D); partial #13 (text→3D) | native-optional asset source (Crucible twin → GLB → glTF importer) | **XL → M** |
| **aetherspectra Director** | gameplay breadth + #6/#13 (story→scene plan); Gemini = the #13 LLM backend | in-Rust schema map `CreativeBrief→Vec<IntentAction>` | **L → M** |
| **aetherspectra mantle** | gameplay breadth (quest/event/world-state) + wedge synergy | service/offline event-timeline replay | **L** |
| **floraprime_gen** | unblocks floraprime-native | trainer-offline (data + 1 run) + bridge | **XL → S+M** |
| **rheos** (reasoning brain, remote on tomespensin) | **#13 real AI backend** for Ask Ochroma (orchestrator composing experts, not a monolithic LLM) + **AI-creates-code** (code-repair expert) + optional #18 metrics | **HTTP IntentBackend** → rheos `api_server_v1`, behind the existing schema-validated/undoable seam | **M** |

---

## The killer synergy (surface this)

**mantle ages a scene over story-time; relight re-illuminates it over light-time.**
A captured place that **weathers across a narrative timeline AND reads differently
under each illuminant** — same world, two orthogonal axes of change no RGB engine
can represent and no single sibling can ship alone. It fuses the gameplay-breadth
gap and the spectral wedge into one demo. This is the AAA-for-Ochroma vertical
slice, and every piece of it now exists in-house.

---

## What this changes about the plan

The roadmap's Phase 4 ("content supply at scale", assumed far-off) **moves forward**:
content generation (lumengen), the AI-native collaborator (aetherspectra Director +
Gemini), and a narrative/world-state system (mantle) are integration-bridge work in
the *proven Crucible/forge-native pattern*, not from-scratch builds. The critical-path
spine (CI → GpuContext → resident frame → relight mechanic) is unchanged, but the
**content and gameplay tracks that run parallel to it are now M-effort bridges, not
XL unknowns.** Next integration candidates, in leverage order: lumengen asset bridge
(content supply) · CreativeBrief→IntentAction (the AI collaborator #6/#13) ·
floraprime data+train (unblock the stub) · mantle timeline (the wedge synergy demo).
