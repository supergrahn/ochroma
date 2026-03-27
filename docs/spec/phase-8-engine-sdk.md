# Phase 8 — Engine SDK & Generalisation

**Goal:** Package Ochroma as a general-purpose game engine SDK. The city builder becomes one game built on the engine. A developer should be able to create any genre of game using the Ochroma SDK.

## 8.1 Public Engine API (`ochroma_engine` crate)

- Single crate that re-exports all engine-layer crates
- Documented public API with examples
- Versioned with semver
- Feature flags for optional subsystems (audio, networking, scripting, VR)

## 8.2 Scene Editor (generalised from city builder UI)

- General-purpose entity placement, selection, transformation
- Property inspector for any ECS component
- Asset import dialog
- Scene serialisation to .ochroma_scene format
- Multi-viewport (perspective + orthographic)

## 8.3 Build & Deploy Pipeline

- Package games as standalone executables
- Asset bundling with compression
- Platform targets: Windows, Linux, macOS, SteamOS
- Debug/Release/Shipping build configurations

## 8.4 Second Game Proof

- Build a small game in a different genre (exploration, puzzle, or horror)
- Proves the engine works beyond city building
- Identifies engine limitations to fix

## Exit Criteria

- [ ] `ochroma_engine` crate compiles and provides full engine API
- [ ] A non-city-builder game runs on the engine
- [ ] Documentation covers all public types
- [ ] Engine can package a standalone game binary
