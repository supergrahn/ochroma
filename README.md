# Ochroma Engine

**Spectral Gaussian Splatting Game Engine**

Ochroma is a game engine built on 3D Gaussian Splatting with spectral rendering.
Scenes are volumetric Gaussian splats whose materials carry 8-band spectral
reflectance, so they respond physically to any illuminant — the same geometry
shifts colour from warm sunrise through noon to cool moonlight as the day/night
clock advances.

## Binaries

The engine ships three runnable binaries (all in the `vox_app` crate):

| Binary | What it is |
|--------|-----------|
| `ochroma` | The editor / engine runner — windowed scene with the editor overlay |
| `walking_sim` | The game — a first-person walking simulator built on the engine |
| `net_walk` | The networking demo — exercises the multiplayer replication path |

```bash
cargo run --release --bin ochroma       # editor
cargo run --release --bin walking_sim   # game
cargo run --release --bin net_walk      # net demo
```

## Features

- **Spectral rendering** — 8-band spectral reflectance (380–660nm); materials
  re-illuminate under the active light.
- **Day / night cycle** — time-of-day clock drives the illuminant; scenes recolour
  across the cycle.
- **Global illumination** — GI via the `crucible` renderer (default feature),
  with band-energy output the smoke tests assert on.
- **Gaussian splat native** — load `.ply` files from any 3DGS training tool and
  render them directly.
- **Physics** — built-in AABB collision plus optional Rapier rigid-body physics;
  fracture/drop interactions in the game.
- **Spatial audio** — distance-attenuated audio synthesis with reachable-room
  reverb and a biome soundscape mixer (CPAL backend, needs ALSA on Linux).
- **Netplay** — multiplayer replication path, exercised by `net_walk`.
- **Procedural generation** — terrain (SDF volumetric with overhangs/caves),
  foliage scatter, texture paint.
- **Rhai scripting** — embedded scripting for game config and the debug console.

GPU hardware rasterisation and hardware DLSS require an NVIDIA GPU with driver
support; the engine falls back to a software rasteriser automatically, which is
what the headless smoke path uses.

## Verifying a build: `--smoke`

Both shipped game binaries accept `--smoke`, a headless gate that runs the REAL
per-frame simulation plus a software-rendered frame and asserts on the output
(non-black coverage, sim advancement, GI band energy, fracture/audio counters,
HUD pixels). It exits non-zero on any regression — this catches "compiles but
the game is broken".

```bash
target/release/walking_sim --smoke   # exits 0 on success
target/release/ochroma --smoke       # exits 0 on success
```

Release-profile smoke runs in a few seconds (far faster than the debug build's
multi-minute run).

## Building from source

Ochroma's workspace depends on two **sibling repositories** via relative path
deps:

- `spectra` (`supergrahn/spectra`) — Gaussian render / GPU backends, referenced
  by `vox_render`.
- `crucible` (`supergrahn/crucible`) — the GI renderer, a **default** feature of
  `vox_nodes`.

Cargo must load those manifests for *any* build, so the three repos must be
checked out **side-by-side**:

```
src/
├── ochroma/    <- this repo
├── spectra/
└── crucible/
```

Then, from `ochroma/`:

```bash
# Linux needs ALSA dev headers for the CPAL audio backend.
sudo apt-get install -y libasound2-dev pkg-config

cargo build --release --workspace
cargo test --workspace
```

CI (`.github/workflows/ci.yml`) reproduces this layout by checking out all three
repos under `$GITHUB_WORKSPACE`, authenticated with the `SIBLING_REPOS_PAT`
secret (the sibling repos are private). A lone checkout cannot build.

## Release artifacts

Pushing a version tag (e.g. `v0.1.0`) triggers
`.github/workflows/release.yml`, which builds `--release --workspace` with the
same three-repo checkout, smoke-tests the binaries, and publishes a GitHub
Release with a Linux x86_64 tarball + SHA-256 checksum.

The release pipeline is **hand-rolled** rather than cargo-dist: cargo-dist's
standard runner flow checks out only the tagged repo and offers no clean way to
lay the sibling repos out side-by-side, so it cannot build this workspace. The
hand-rolled workflow mirrors CI's checkout exactly.

**Supported platform: Linux x86_64 only.** Windows and macOS have never
compiled (the CI portability job is experimental); the workflow has a comment
marking the expansion path.

### Installing a release

```bash
tar -xzf ochroma-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz
cd ochroma-vX.Y.Z-x86_64-unknown-linux-gnu
./walking_sim --smoke   # verify it runs (exits 0)
./walking_sim           # play (needs a display)
./ochroma               # editor
```

## Architecture

```
ochroma_engine (SDK crate)
├── vox_core     -- Types, ECS (Bevy ECS), math, spectral, input, scripting API
├── vox_render   -- Spectral pipeline, DLSS (software), CLAS, particles, lighting
├── vox_data     -- Asset formats (.vxm, .ply loader), procedural generation, maps
├── vox_terrain  -- Volumetric SDF terrain, heightmaps, foliage scatter
├── vox_sim      -- Game simulation systems
├── vox_audio    -- Spatial audio, room reverb, biome soundscape mixer
├── vox_physics  -- AABB physics (built-in) + Rapier rigid body (optional feature)
├── vox_net      -- Multiplayer networking with CRDT replication
├── vox_script   -- Rhai scripting runtime, visual scripting, plugin system
├── vox_nodes    -- Crucible GI renderer bindings
├── vox_nn       -- LLM integration, procedural city generation (in development)
├── vox_ui       -- UI framework (SpectralHUD)
└── vox_tools    -- CLI asset pipeline tool (turnaround capture, GLTF import)
```

## License

MIT OR Apache-2.0
