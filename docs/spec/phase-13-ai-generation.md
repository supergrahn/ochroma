# Phase 13 — AI-Native Generation

**Goal:** Make Ochroma the first engine where AI-generated content is a first-class workflow, not a plugin. This is the feature Unreal cannot match — generative AI built into the engine core.

## 13.1 LLM Layout Interpreter (Real)

- Integrate a local LLM (llama.cpp via llm crate, or Ollama API)
- Prompt → structured SceneGraph output
- Spatial constraint enforcement: buildings face roads, no overlaps, proper setbacks
- Style consistency: LLM maintains architectural coherence across a district

## 13.2 Text-to-Asset Generation

- "Victorian terraced house with bay windows" → Proc-GS rule → .vxm asset
- LLM generates SplatRule TOML from natural language
- Validation pass: check rule produces valid geometry
- Reverse mode: given a .vxm asset, LLM extracts proportions and generates a matching rule

## 13.3 Neural Infill (Compositional Detail)

- Proc-GS skeleton provides rigid structure
- Diffusion model adds unique surface details (cracks, moss, graffiti)
- Latent Refinement Pass blends procedural and neural Gaussians
- Each building instance gets unique weathering without unique assets

## 13.4 Prompt-to-City

- "Build a Victorian London street, slightly run-down, evening, light rain"
- LLM generates street layout with building slots
- Proc-GS fills each slot with deterministic buildings
- Neural infill adds unique details per building
- Weather system applies rain SPD shifts
- Time-of-day sets illuminant to evening warm tones
- Result: a complete, coherent, unique city district from one sentence

## 13.5 Lyra Video Capture

- User records phone video of a real building
- Video → 3DGS reconstruction → spectral albedo extraction → .vxm
- The real world becomes an asset library

## Exit Criteria

- [ ] "A Victorian street" generates a coherent populated street in <60 seconds
- [ ] Text description generates a valid SplatRule
- [ ] Neural infill adds visible unique detail to procedural buildings
- [ ] Phone video produces a usable .vxm asset
