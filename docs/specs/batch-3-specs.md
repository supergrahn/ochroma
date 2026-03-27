# Batch 3 — Polish & Ship

Final batch. After this, the engine is ready for other developers to use.

---

## Feature 11: Lighting & Visual Polish

Ambient occlusion approximation, improved bloom, fog integration, and sky rendering wired into the software rasteriser output.

**Create:** `crates/vox_render/src/visual_effects.rs`
Apply post-processing to rendered framebuffers: SSAO approximation (darken pixels where normals change rapidly), distance fog, and vignette. All CPU-side pixel operations.

Tests: fog darkens distant pixels, vignette darkens corners, SSAO darkens crevices.

---

## Feature 12: Release Build & Packaging

**Create:** `scripts/build_release.sh`
Script that builds optimized binaries and packages with README + example assets.

**Create:** `scripts/package.sh`
Creates a .tar.gz / .zip with: ochroma binary, example assets, README, getting started guide.

Tests: script produces a valid archive with expected files.

---

## Feature 13: Documentation Audit

**Modify:** `README.md`, `docs/getting_started.md`
Audit every claim against actual code. Remove anything that doesn't work. Add what does work.

---

## Feature 14: Performance Benchmarks

**Create:** `crates/vox_render/tests/benchmark_suite.rs`
Automated benchmarks: render 10k, 50k, 100k, 500k splats. Record frame times. Verify no regressions.

---

## Feature 15: Polished Example Game

**Modify:** `crates/vox_app/src/bin/walking_sim.rs`
Use ALL new systems: CharacterController for movement, GameUI for HUD, SpatialAudio for sound effects, RigidAnimation for a windmill, shadows on terrain.
