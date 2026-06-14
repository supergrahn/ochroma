# Handoff prompt — Ochroma / Urban Horizon content pipeline

You are picking up an in-progress game-engine + city-builder project. Work fast, fix in
batches (audit a whole class before fixing one symptom), prove changes with cheap measured
gates, and never fake or relax a gate. The owner wants RESULTS with visible proof (a render
or a measured number), not process narration.

## The project (4 repos, all local path-deps)
- `~/src/ochroma` — the engine (Rust). `crates/vox_render/src/splat_backend.rs` holds the
  path-trace entry points + render tests. Branch: `master`.
- `~/src/spectra` — the GPU path tracer (Rust + Slang kernels in `slang/`). Vulkan/RADV on
  an AMD 780M (no CUDA — every OptiX/DLSS/NRC path silently stubs; ignore those warnings).
  Current checkout = branch `feat/khr-rt` (its tip IS the mainline; KHR hardware ray
  tracing just landed: GPU-built BLAS/TLAS + ray-query, parity-proven, `SPECTRA_HW_RT=0`
  kill switch). Has an UNCOMMITTED `slang/megakernel.slang` diff from an abandoned SDF
  wave — leave it or stash it, don't commit it.
- `~/src/forge` — procedural building generator (Rust). Branch `fix/remove-siding-slats`
  (tip, ahead of master: slat removal + test re-pin — MERGE this to master first thing).
  108/108 tests green on the tip.
- `~/Ochroma/projects/urban_horizon` — the game. The COOK lives at
  `src/bin/game_asset_cook.rs` (directive JSON → Forge geometry → cooked
  `ReadyAssetPayload`: mesh + atoms + SDF + materials). Branch: `master` (green).

## Architecture decisions (settled by the owner — do not relitigate)
- Render primitive = MESH (real triangle geometry, "no faking"). SDF = engine substrate
  only (collision/destruction/terraforming). One Spectra path tracer renders everything.
- Buildings come from a DIRECTIVE grammar (axes: footprint × massing × height × facade ×
  material × crown). Forge = "one engine in back, template collection in front" — the
  directive is authoritative; generators must MEASURE geometry they made, never predict
  from directive numbers (Forge applies per-style floor multipliers, e.g. Industrial 1.3).
- Target: 60 fps @ 4K on an RTX 4070 Ti, 100K building instances / ~200-400 types, 1M
  moving objects (mesh, instanced). The 780M dev box is a floor, never the bar.

## Build / test / cook commands (exact)
- Render tests (ochroma): `SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json scripts/build-spectra-native.sh test -p vox_render --features spectra-native --profile release-fast --lib <test_name> -- --nocapture --test-threads=1`
  ALWAYS `--profile release-fast` (a no-LTO profile; `--release` full-LTO is for shipping
  only — it costs minutes per rebuild). Keep gate renders ≤32 spp ≤256–512²; denoise PNGs
  for eyeballing via `crate::denoiser::SpectralDenoiser` (see `forge_facade_zoning_closeup`).
- Spectra direct builds need: `SLANG_DIR=$HOME/slang-sdk LD_LIBRARY_PATH=$HOME/slang-sdk/lib BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/lib/gcc/x86_64-linux-gnu/13/include"`; workspace is `~/src/spectra/rust`. SPIR-V disk cache at `~/.cache/spectra-spirv` (content-hashed, safe).
- The cook (civitas): `GAME_FORGE_BIN=$HOME/src/forge/target/release/forge ./target/release/game_asset_cook` (full catalog, ~113 assets, currently GREEN exit=0).
  No `--only` flag: subset-cook via `--no-starters --source <tmp dir with one directive> --output <tmp>`. Rebuild the forge binary after forge changes
  (`cargo build --release -p forge-cli` in ~/src/forge). NEVER pipe the cook through
  `head`/`grep -m` — SIGPIPE kills it mid-run; redirect to a log file.
- Forge tests: `cargo test --release -p forge-building` (108 green; golden mesh hashes pin
  box-mode geometry — re-pin ONLY for intentional geometry changes, documented).

## Landmines (each cost hours today — do not rediscover)
1. MEASURE, don't predict: the cook's surface descriptors/gates must derive heights from
   the mesh (`roof_height_at` probes, `snap_main_roof_to_mesh`), never floors×floor_height.
2. Material indexing: cooked `materials` is slot-ordered but `mesh.material_ids` carry
   FORGE channel ids (0 wall,1 roof,2 glass,3 reveal,4 trim,5 cornice,6 door). Always map
   by forge id (`load_building_mesh_pbr_by_forge_id`). Never positional.
3. Coverage holes: CookedKind arms (House/Rowhouse/School/Police) × roof × massing combos
   are unevenly wired (House only just got the flat-roof kit). After ANY forge/cook change
   run the FULL cook as the integration test.
4. Seal oracle: forge geometry flush to walls breaks `orient_outward` (probe epsilon
   0.02 m) — detached boxes need >0.025 m clearance or embed fully; 1 mm gaps between
   stacked boxes (see porch stairs) keep solids provably closed.
5. Real transmissive glass works on the mesh path (`PbrMaterial.transmission/ior/
   thin_walled` → packer emits MAT_GLASS: type a[0]=3, ior a[10], absorption a[24..28],
   thin_walled a[73]). Known quirk: shading normals flip → glass slabs slightly magnify
   (engine fix pending; fine for windows).
6. Git: NEVER delete a branch until `git merge-base --is-ancestor` proves it merged. Use
   `git -C <repo>` on every command (cd-slips corrupted state twice). Don't run git
   surgery in a repo while another agent/process edits it.
7. Era gap: features emitted as ATOMS (rooftop HVAC kit, plot fences/driveways,
   vegetation) are INVISIBLE on the mesh render path — porting them to mesh is open work.

## State: shipped + proven today (don't redo)
Full cook green (113 assets); KHR hardware RT (ray kernels 3.4×, CPU BVH build gone);
real glass (checkerboard refraction r=1.000); material zoning fix (no brick on window
frames); fake siding slats removed (stripe artifact gone — clean closeup proof);
footprint-aware roofs/floors (L/U/T genuinely real, gate green, `void_cov=0.000`);
massing axis (podium+tower 30-floor glass skyscraper, setback ziggurat, box byte-identical);
curtain-wall facade family (960-cell mullion grid, transmissive vision glass); Mesh M0
(textured lit craftsman); 22 modern directives; 90 GB stale builds cleaned.

## Open work, in priority order
1. A read-only audit of `game_asset_cook.rs` may still be running/reporting (classes:
   predicted-vs-measured, gate assumptions, coverage matrix, two-sources-of-truth, dead
   directive fields). Take its findings table and BATCH-FIX by class. If absent, do that
   audit yourself first — it's the highest-leverage file.
2. Materials part 2: brick library (red/brown/beige/white) + wood via `palette.wall_material`
   (#[serde(default)]) — design at `civitas docs/specs/facade-materials-design.md`; partial
   WIP exists on `feat/materials` branches (forge+civitas) + civitas stash
   'materials-part2-wip'. Finish + render proof (brick wall stays off the windows).
   Then: metal (needs metallic plumbed through forge wall_material), per-volume materials
   (podium ≠ tower — one glass channel currently leaks between volumes).
3. Entrances (design: `civitas docs/specs/building-entrances-design.md`): glazed_lobby on
   the tower podium first (GradeBandMask suppresses the default door bay; reuse
   curtain_wall.rs), then vestibule/storefront/canopy. Two prior attempts stalled by
   over-exploring — the design has exact hooks; just build them.
4. Port the rooftop HVAC kit (atoms → real Forge mesh or kit-socket pieces; placement
   logic already in cook `add_flat_roof_kit`) + audit all other atom-era features for the
   same port (fences, driveways, vegetation).
5. Render quality: bridge the vendor-neutral Slang denoiser (`slang/realtime_denoise.slang`)
   into the Vulkan path (config DenoiserMode defaults to OptiX = stub on AMD); fix the EWA
   footprint cap that mushes textures; check tonemap on the readback path.
6. Weathering: `forge.condition` (New/Aged/Weathered/Derelict) currently just tints a
   tiling texture → looks systematic. Make it geometry-anchored masks (under-sill streaks,
   grade splash, drip lines, orientation bias; deterministic via seed).
7. Performance: resident renderer (entry points rebuild renderer+scene per call — the
   host seam), then TLAS-over-instances for city scale, agents/movers later.
8. The grammar designs to keep extending: `civitas docs/specs/
   building-composition-grammar-design.md` (axes/coverage),
   `forge-directive-factory-design.md` (universal directive envelope + type dispatch).

## Working style the owner demands
- Results with proof, fast. Inline fixes over delegation. No long exploration before
  producing. Audit a whole class, then batch-fix. Cheap renders for gates; denoised PNGs
  for humans. Never claim a visual proof you haven't actually looked at.
