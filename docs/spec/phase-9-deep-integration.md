# Phase 9 — Deep Integration & Content Pipeline

**Goal:** Wire every system together into a cohesive game loop. Generate a starter content library of procedural assets. Harden the engine for continuous play sessions.

## 9.1 Full Game Loop Integration

Every simulation system must tick and produce visible results:
- Citizens commute (visible agent movement on roads)
- Buildings consume utilities (brownouts if insufficient power)
- Traffic density affects road segment rendering colour
- Pollution from industry visible as particle smoke
- Disasters trigger fire particles and service response
- Seasons change terrain appearance and affect economy
- Milestones trigger notifications and unlock new buildings

## 9.2 Procedural Asset Library

Generate a starter library of .vxm assets using the Proc-GS system:
- 10 residential building variants (Victorian, Modern, Suburban)
- 5 commercial building variants
- 3 industrial building variants
- Service buildings (school, hospital, fire, police)
- Props: trees (3 species), benches, lamp posts, bins, fountains
- Terrain tiles: grass, cobblestone, asphalt, water, sand
- Vehicles: cars, buses, trucks

## 9.3 Simulation Depth

- Citizen employment matching (find nearest job matching education)
- Housing market (citizens move to better housing when available)
- Crime simulation (affected by police coverage and unemployment)
- Health simulation (affected by hospital coverage and pollution)
- Education pipeline (children → school → university → skilled jobs)

## 9.4 UI/UX Completion

- Full overlay rendering (zone colours visible on terrain)
- Budget graph over time (not just current values)
- Population graph
- Advisor system (contextual tips based on city state)
- Tool tooltips with cost/requirements
- Construction progress bars on growing buildings

## 9.5 Error Recovery & Robustness

- Corrupted save files show error dialog, don't crash
- Missing assets replaced with placeholder splats
- GPU timeout recovery (recreate device if lost)
- Auto-save recovery on crash
- Graceful shutdown (save state before exit)

## Exit Criteria

- [ ] Start game → place roads → zone → buildings grow → citizens commute → 30 min stable
- [ ] 10+ distinct building types visible in a city
- [ ] All overlays render correctly
- [ ] Save/load preserves all simulation state
- [ ] Budget and population graphs show historical data
- [ ] Disasters trigger, spread, and resolve with visible effects
