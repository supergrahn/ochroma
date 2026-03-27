# Phase 6 — Spectra Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate the Spectra renderer for production-quality rendering, add real audio, polish the UI, and validate performance.

**Architecture:** Spectra's Rust crates become workspace dependencies. A bridge module routes rendering through Spectra when available.

---

### Task 1: Spectra Bridge Module

Create the bridge between Ochroma's rendering pipeline and Spectra's Gaussian splatting renderer.

**Files:**
- Create: `crates/vox_render/src/spectra_bridge.rs`

### Task 2: Production Sky and Lighting Model

Sun position from time-of-day, spectral sky dome, point lights from buildings.

**Files:**
- Create: `crates/vox_render/src/lighting.rs`
- Test: `crates/vox_render/tests/lighting_test.rs`

### Task 3: Shadow Pipeline Integration

Wire shadow catchers into the render pass.

**Files:**
- Modify: `crates/vox_render/src/gpu/gpu_rasteriser.rs`

### Task 4: Audio Activation + Soundscape

Enable rodio, create city soundscape, wire UI sounds.

**Files:**
- Create: `crates/vox_app/src/soundscape.rs`

### Task 5: Production UI

Notifications, mini-map, settings, tutorials.

**Files:**
- Create: `crates/vox_app/src/notifications.rs`
- Create: `crates/vox_app/src/minimap.rs`
- Create: `crates/vox_app/src/settings.rs`

### Task 6: Performance Profiling + Optimisation

Profile bottlenecks, wire GPU sort, validate frame budget.

### Task 7: Final Integration Test

30-minute stability run with all systems active.
