# Technology Foundations

Research backing for the five core technologies that power the engine's asset generation
and rendering architecture. Each entry links to the paper that validates the approach.

---

## 1. Neural Layout Interpreters

**What it does:** When a user types a prompt like *"A narrow Victorian street,"* the engine
does not generate pixels. An LLM generates a **Latent Scene Graph** — a structured output
of coordinates, scale factors, style tags, and spatial constraints. This "Architect Brain"
enforces a skeleton of logic before a single Gaussian is placed.

**Why it matters:** Solves the drift problem. Pure generative AI hallucinates incoherent
layouts (three front doors, buildings facing away from roads). The Layout Interpreter
constrains generation to architecturally valid configurations before any rendering happens.

**Research backing:**

- **GALA3D** — arXiv:2402.07207, ICML 2024 (Peking University)
  Compositional text-to-3D generation via layout-guided Gaussian Splatting. Uses an LLM
  to extract initial object positions and scales from text, then runs layout-guided
  optimization to correct LLM spatial errors. Demonstrates that the LLM + correction loop
  produces coherent multi-object 3DGS scenes.

- **DreamScape** — arXiv:2404.09227, 2024
  Uses LLMs to extract semantic primitives, spatial transformations, and object
  relationships from text prompts. Models physical collision constraints to correct LLM
  layout biases. Introduces local-to-global optimization for scene coherence.

**Implementation:** The Layout Interpreter lives in `vox_nn` (Phase 2+). It outputs a
`SceneGraph` struct consumed by the Proc-GS assembler.

---

## 2. Proc-GS: Procedural Gaussian Splatting

**What it does:** Instead of generating every brick and window from scratch, the engine
maintains a library of high-quality canonical Gaussian components (a window tile, a door
frame, a brick panel). A grammar system assembles these components into full buildings and
cities. One canonical window asset is instanced millions of times — 4–5x VRAM savings
over storing each window as unique Gaussians.

**Why it matters:** The only sustainable path to a 100km city without scaling artist hours.
Rules + seed = infinite variation. Same seed = same asset, always, deterministically.

**Research backing:**

- **Proc-GS: Procedural Building Generation for City Assembly with 3D Gaussians**
  arXiv:2412.07660, CVPR 2025 Workshop on Urban Scene Modeling 3D (IEEE proceedings
  pages 2031–2040). Authors: Yixuan Li et al., City-Super group.
  Project page: city-super.github.io/procgs/

  Two-stage system: (1) procedural code drives 3DGS training to extract canonical base
  Gaussian assets; (2) procedural code is manipulated to assemble diverse buildings into
  a full city. Demonstrates 4–5x model compression via instancing and high-fidelity
  rendering at city scale.

**Implementation:** `vox_data` crate, `SplatRule` trait system. See
`asset-generation-pipeline.md` for full specification.

---

## 3. Compositional Generation

**What it does:** A city is never built as one large blob. It is built compositionally.
The Proc-GS skeleton provides the rigid structural frame (walls, roof, floor plan). An
Image-to-3D diffusion model generates the *unique* parts — the specific weathered door,
the cracked chimney, the graffiti on the back wall. A **Latent Refinement Pass** blends
the procedural and neural Gaussians at their boundaries so no seam is visible.

**Why it matters:** Combines the rigid correctness of a city builder (buildings face the
road, windows are aligned, rooflines are consistent) with the organic messiness of neural
generation (no two doors look the same, wear patterns are unique). Neither approach alone
achieves both.

**Research backing:**

- **CompGS: Unleashing 2D Compositionality for Compositional Text-to-3D via Dynamically
  Optimizing 3D Gaussians** — arXiv:2410.20723, October 2024
  Compositional text-to-3D generation using 3DGS as the representation.

- **CG3D: Compositional Generation for Text-to-3D via Gaussian Splatting**
  arXiv:2311.17907, November 2023
  Establishes the foundational approach of compositional multi-object 3DGS scene
  generation using Score Distillation Sampling.

**Implementation:** Phase 2+. The Latent Refinement Pass runs as a post-assembly CUDA
kernel that blends Gaussian opacity at procedural/neural boundaries.

---

## 4. Spectral Neural Rendering

**What it does:** Each Gaussian splat stores a **Spectral Reflectance Curve** — 8
coefficients spanning 380–720nm — instead of RGB values. The CUDA rasterizer integrates
these curves against the current illuminant (D65 daylight, sodium streetlamp, LED, etc.)
to produce the correct RGB output at render time. Changing streetlights from sodium to LED
does not "tint" the image — it recalculates how each material physically reflects the new
light spectrum.

**Why it matters:** Re-lighting is physically correct. Time-of-day, weather, IR/UV modes,
and seasonal spectral shifts all work without baking new textures. No lighting is ever
baked into an asset.

**Research backing:**

- **SpectralGaussians** — arXiv:2408.06975, August 2024
  Multi-spectral scene representation framework extending 3DGS to work with registered
  multi-view images from different spectral bands. Encodes reflectance and spectral
  properties per Gaussian. Demonstrates spectral rendering, semantic segmentation, and
  reflectance estimation within the 3DGS framework.

**Implementation:** Core to the engine from Phase 0. Spectral coefficients are stored in
the `.vxm` format and processed by the `spectral_rasterize` CUDA kernel.

---

## 5. Self-Distilling Video World Models

**What it does:** Instead of requiring hundreds of photos of an object, the engine can
ingest a short video and use a video diffusion model to "hallucinate" the sides of the
object that were never filmed. A user points their phone at a real Victorian building,
records a short video, and the engine distills it into a ploppable `.vxm` asset.

**Why it matters:** Removes the multi-view capture requirement. Real-world assets can be
created from footage that any user can capture, democratising asset creation for the city
builder.

**Research backing:**

- **Lyra** — arXiv:2509.19296, NVIDIA Toronto AI Lab, accepted ICLR 2026
  GitHub: github.com/nv-tlabs/lyra

  Self-distillation framework using video diffusion models to generate 3DGS scenes from
  monocular video input. Two parallel decoders: a standard RGB decoder (teacher) supervises
  a 3DGS decoder (student). Infers novel views and full 3D structure from single-viewpoint
  video. Removes the need for calibrated multi-view capture setups.

**Implementation:** Phase 3+. Runs as an offline processing pipeline outside the engine
core. Outputs `.vxm` assets consumed by the standard asset pipeline.

---

## Supporting Technologies

### Semantic Splat Segmentation (S3)

Every Gaussian carries a 16-bit `entity_id`. When the Neural Layout Interpreter or Proc-GS
assembler places a component cluster (a window, a door), it tags all Gaussians in that
cluster with a shared `entity_id`. The CUDA rasterizer writes these IDs to a separate
entity buffer. Click detection, agent interaction, and semantic queries all operate on
entity IDs, not raw Gaussian positions.

This means the engine "knows" a window is a window at the Gaussian level — without any
runtime mesh or collision geometry.

### Shadow Catchers (MeshSplats)

Gaussian splats are volumetric and do not cast hard shadows natively. When a Proc-GS or
compositional asset is plopped, the engine generates an invisible low-poly convex hull
(the Shadow Catcher mesh) that follows the asset's Gaussian bounds. The spectral sun
casts shadows from the Shadow Catcher mesh onto the terrain and neighbouring assets,
giving physically accurate shadow edges without requiring the Gaussians themselves to act
as shadow casters.

### Alpha-Gaussian Blending (Latent Refinement Pass)

At the boundary between a procedural Proc-GS wall and a neural-generated door component,
a CUDA blending pass calculates a soft transition zone. Procedural Gaussians taper their
opacity toward the boundary while neural Gaussians fade in. The result looks like a single
coherent surface. No visible seam between rule-generated and AI-generated geometry.

---

## Research Reference Index

| Technology | Paper | arXiv | Venue |
|---|---|---|---|
| Neural Layout Interpreter | GALA3D | 2402.07207 | ICML 2024 |
| Neural Layout Interpreter | DreamScape | 2404.09227 | 2024 |
| Proc-GS | Proc-GS: Procedural Building Generation | 2412.07660 | CVPR 2025 Workshop |
| Compositional Generation | CompGS | 2410.20723 | 2024 |
| Compositional Generation | CG3D | 2311.17907 | 2023 |
| Spectral Rendering | SpectralGaussians | 2408.06975 | 2024 |
| Video World Models | Lyra | 2509.19296 | ICLR 2026 |
| Feed-forward reconstruction | Turbo3D | 2412.04470 | CVPR 2025 |
| Feed-forward reconstruction | GRM | 2403.14621 | ECCV 2024 |
| SDS generation | GSGen | 2309.16585 | 2024 |
