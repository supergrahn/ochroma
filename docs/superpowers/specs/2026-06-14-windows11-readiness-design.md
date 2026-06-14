# Design: Windows 11 Readiness — Editor + Render + Load/Save (2026-06-14)

**Status:** Draft
**Scope:** Make the Ochroma editor (`ochroma_editor`) build, open a window, render a frame, and load/save a game on Windows 11 with AMD + NVIDIA GPUs; identify exactly what is portable now vs what must be proven on a real Win11 box; scope the optional spectra-native / fsr path as a separate, deferred phase.
**Related:** `[Spectra/Crucible deps]`, `[GPU-first directive]`, `[hybrid-atom-sdf-lod-direction]`

---

## 1. Problem Statement

The dev box is Linux. We want editor-based game dev on Windows 11. Concrete questions answered here, with static evidence from the four-repo tree (`ochroma`, `spectra`, `crucible`, `openusd-rs`):

- The default editor (`cargo run -p vox_app --bin ochroma_editor`) graph is **pure-Rust and portable** — no `build.rs`, no `std::os::unix`, no `$HOME`, no `VK_ICD_FILENAMES` in default runtime code. Verified across `vox_render/vox_data/vox_app` and the non-optional native edges (`spectra-gaussian-render`, `crucible-core`, `crucible-types`, `openusd-rs`) — all have no `build.rs` and no unix coupling.
- `slang-sys/build.rs:67` emits `cargo:rustc-link-arg=-Wl,-rpath,...` **unconditionally** (GNU-ld syntax) — a hard MSVC `link.exe` failure for the spectra-native graph until gated behind `cfg(unix)`.
- `ffx-sys/build.rs` (the `fsr` feature) hardcodes `-G "Unix Makefiles"`, `-fshort-wchar`/`-fpermissive`/`-fno-tree-vectorize`, a `spectra_gcc_compat.h` `-include` shim, `.a` archive discovery, `dylib=vulkan`, `dylib=stdc++` — no Windows arm. `fsr` is OFF by default.
- Enabling `--features spectra-native` pulls `spectra-renderer` **with its own defaults** (`cuda,ocio,dlss,optix-denoiser`), and `spectra-gpu` defaults to `cuda` → `cudarc` becomes a build dependency (pinned without `dynamic-loading`). The "CUDA stays optional" framing is true at *runtime* (`splat_backend.rs` eprintln, not panic) but **false at build time** for the spectra-native graph.
- Three concrete Windows string bugs in the spectra-native path: `slangc` resolved with no `.exe` and `PATH` split on `:` (`spectra/rust/spectra-gpu/src/vulkan_backend.rs:1492-1510`); SPIR-V cache returns `None` on Windows → silent disable (`vulkan_backend.rs:1546-1558`); kernel cache falls back to a literal `/tmp` → lands in `C:\tmp` (`kernel_cache.rs:120-141`).
- Two CWD-relative paths break a shortcut launch: `quick_save_path()` = bare `"saves/quicksave.json"` (`vox_data/src/world_save.rs:202-205`) and the UI theme `"assets/ui/ochroma.theme.json"` (`ochroma_editor.rs:128-138`, non-fatal).

---

## 2. Done When

**Verifiable now on Linux** (no behavior change to Linux):

- Running `cargo build -p vox_app --bin ochroma_editor` on Linux still succeeds after the cross-platform edits (DX12 tier, `split_paths`/`.exe` slangc, `dirs::cache_dir()` cache roots, anchored save/asset paths). Output: `Finished` with exit code 0, and `cargo run -p vox_app --bin ochroma_editor` still opens the editor on the Linux box exactly as before.

**Done When (run on Win 11)** — provable only on a real Win11 GPU box:

- Running `cargo run -p vox_app --bin ochroma_editor` on `x86_64-pc-windows-msvc` opens a 1600x900 window with the egui dock visible; `cargo run -p vox_app --bin ochroma_editor -- --frames 120` exits with code 0; `cargo run -p vox_app --bin ochroma_editor -- --shot out.png` writes `out.png` as a >0-byte PNG containing non-background pixels (color other than the clear color). A human at the keyboard sees the editor window and a non-blank screenshot.
- In the running editor, save a world (F5 / Save) then load it (F9 / Load): a file appears under `%APPDATA%\ochroma\saves\` and reloading restores the identical splat scene (same splat count shown in the editor's scene panel).

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| DX12 backend tier (cross-platform, additive) | After edit, `cargo build -p vox_app --bin ochroma_editor` on Linux exits 0 AND `backend_attempts` contains `("DX12", wgpu::Backends::DX12)` between Vulkan and GL (asserted by reading the slice in a unit test) | `assert!(backend_attempts.len() > 0)` — passes without DX12 |
| slangc resolver finds `.exe` and splits PATH per-OS | Unit test: `find_slangc()` given a temp dir with `slangc.exe` returns that path; given PATH with `;`-joined dirs on Windows / `:` on Linux, resolves the right entry via `std::env::split_paths` | `assert!(result.is_some())` against a dir that always contains `slangc` |
| Cache roots resolve to `%LOCALAPPDATA%` on Win, `~/.cache` on Linux | `spirv_cache_dir()` returns `Some(path)` where `path` ends with `Ochroma/cache/spirv` and is under `dirs::cache_dir()`; on Linux the byte path is unchanged when `XDG_CACHE_HOME` set | function returns `Some(_)` regardless of platform |
| Save round-trip path is OS-correct | `quick_save_path()` returns a path under `dirs_next::data_dir()/ochroma/saves/quicksave.json` and `create_dir_all` of its parent succeeds | `assert!(path.to_str().is_some())` |
| Editor opens + renders + saves on Win11 | (run on Win 11) `--frames 120` exits 0; `--shot out.png` PNG has non-bg pixels; save file appears under `%APPDATA%\ochroma\saves` | screenshot exists but is all clear-color |

---

## 4. Architecture

### 4.1 Verdict

**Portable-with-work.** The DEFAULT editor build is genuinely portable — there is no Linux-only build dependency and no unix-only runtime code in its graph. The work splits cleanly into two phases: (A) the portable editor — buildable from Linux today via safe cross-platform edits, render/save provable only on a Win11 box; (B) the optional spectra-native path-traced renderer + fsr upscaler — real porting work (slang.dll/.lib wiring, ffx-sys MSVC arm, the spectra-* `default-features = false` fix, the `-fshort-wchar` ABI proof) that is OFF by default and deferred.

Ranked blockers:

1. **(Phase A, validate-on-Win)** Build links under MSVC and wgpu acquires a hardware adapter on AMD + NVIDIA. No Linux-only dep found — but unprovable from Linux.
2. **(Phase B, static)** `slang-sys/build.rs:67` `-Wl,-rpath` breaks MSVC link → gate behind `cfg(unix)`.
3. **(Phase B, static)** spectra-* default features pull `cuda/optix/dlss` at build time → add `default-features = false` on the spectra path deps in `vox_render/Cargo.toml` + a Windows feature subset.
4. **(Phase B, static)** `ffx-sys/build.rs` is entirely GCC/Unix-Makefiles → needs a full `cfg(windows)` arm; recommend leaving `fsr` OFF indefinitely for the editor.
5. **(Phase B, validate-on-Win, UNPROVEN)** the `-fshort-wchar` keystone: MSVC's native 2-byte `wchar_t` *should* map, but the `FFX_*_CONTEXT_SIZE` static_asserts can only be proven by an MSVC struct-layout/link test.

### 4.2 wgpu backend selection (default editor)

`crates/vox_render/src/gpu/wgpu_backend.rs:29-33` tries `Vulkan → GL → all()`. On Win11 the vendor driver ships the Vulkan ICD and the loader auto-discovers it from the registry — **no `VK_ICD_FILENAMES`** (it appears only in doc-comments/tests today). DX12 is reachable only via `all()`, *after* GL. Insert an explicit `("DX12", wgpu::Backends::DX12)` tier **between Vulkan and GL** so a named, deterministic DX12 path exists before GL can grab a weak adapter. On Linux `Backends::DX12` yields no adapter and the existing per-attempt `continue` (lines 60-66) skips it — Linux order is byte-unchanged. `adapter.rs` `is_software()` keys authoritatively on `device_type == DeviceType::Cpu` (name-substring is belt-and-suspenders), so legitimate Win AMD/NVIDIA `DiscreteGpu`/`IntegratedGpu` adapters are not falsely rejected; `OCHROMA_ALLOW_SOFTWARE_GPU=1` is the headless/RDP escape.

### 4.3 Save/load + paths (default editor)

`world_save.rs` is plain `std::fs` + `serde_json` over `&Path`; `auto_save_path()` already uses `dirs_next::data_dir()` → `%APPDATA%`. Only two CWD-relative paths need anchoring: `quick_save_path()` and the UI theme path. Path construction uses `Path::join` throughout (backslash-correct on Win); no `fs::canonicalize`, no `PermissionsExt`/`set_mode` in the default save/asset IO. `urban_horizon` `instanced.rs:413` `PathBuf::from("civitas_data")` is CWD-relative — anchor to exe-dir or `CIVITAS_DATA_DIR` (civitas is not in the portable editor set, but the fix is Linux-safe).

### 4.4 spectra-native build path (deferred, Phase B)

Headless compute: the spectra Vulkan backend creates the instance with no surface/swapchain and selects a compute queue — **zero Win32 surface porting**. KHR ray-query is capability-probed (extension presence + feature bits) with software-BVH fallback and `SPECTRA_HW_RT=0` — vendor differences are handled at runtime. The build coupling is the real work: `slang-sys` rpath gate, slang.dll/.lib link arms, `ffx-sys` MSVC arm, the spectra-* `default-features=false` fix, and the three Windows string bugs (slangc `.exe`/`split_paths`, cache roots via `dirs::cache_dir()`). `dunce::canonicalize` (or `\\?\`-prefix stripping) for any path handed to slangc/cmake.

---

## 5. Data Models

No new persisted types. One internal helper introduced:

```rust
/// Resolve a regenerable cache subtree under the OS cache dir,
/// honoring OCHROMA_CACHE_DIR then dirs::cache_dir(), then temp_dir().
/// Windows: %LOCALAPPDATA%\Ochroma\cache\<sub>. Linux: $XDG_CACHE_HOME|~/.cache/Ochroma/cache/<sub>.
pub fn cache_subdir(sub: &str) -> Option<std::path::PathBuf>;
```

Cache layout (all regenerable; never the saves dir):

```
%LOCALAPPDATA%\Ochroma\cache\spirv     (was ~/.cache/spectra-spirv)
%LOCALAPPDATA%\Ochroma\cache\kernels   (was ~/.cache/spectra/kernels, /tmp fallback)
%LOCALAPPDATA%\Ochroma\cache\assets    (polyhaven/blenderkit)
%LOCALAPPDATA%\Ochroma\cache\blas      (already via dirs::cache_dir())
%APPDATA%\ochroma\saves                (unchanged; data_dir())
```

---

## 6. API

Real signatures the plan must match (verified against the tree):

```rust
// crates/vox_render/src/gpu/wgpu_backend.rs (current)
let backend_attempts: &[(&str, wgpu::Backends)] = &[
    ("Vulkan", wgpu::Backends::VULKAN),
    ("GL", wgpu::Backends::GL),
    ("all", wgpu::Backends::all()),
];
// CHANGE: insert ("DX12", wgpu::Backends::DX12) BETWEEN Vulkan and GL.

// crates/vox_data/src/world_save.rs
pub fn auto_save_path() -> std::path::PathBuf; // already dirs_next::data_dir()-based (line ~230)
pub fn quick_save_path() -> std::path::PathBuf; // CHANGE from "saves/quicksave.json" to data_dir()/ochroma/saves/quicksave.json

// spectra/rust/spectra-gpu/src/vulkan_backend.rs:1492-1510 (slangc resolve)
// CHANGE: try "slangc" AND "slangc.exe"; use std::env::split_paths(&path_var) instead of .split(':')
// spectra/rust/spectra-gpu/src/vulkan_backend.rs:1546-1558 spirv_cache_dir() -> Option<PathBuf>
// spectra/rust/spectra-gpu/src/kernel_cache.rs:120-141 KernelCache::cache_dir() -> PathBuf
//   CHANGE both: env override → dirs::cache_dir().join("Ochroma/cache/<sub>") → std::env::temp_dir()

// spectra/rust/vendor/slang-sys/build.rs:67 (unconditional today)
//   cargo:rustc-link-arg=-Wl,-rpath,...  → wrap in #[cfg(unix)]
```

dirs-crate platform mapping (the evidence for the Windows folders): `dirs::cache_dir()` → `%LOCALAPPDATA%` (SHGetKnownFolderPath, FOLDERID_LocalAppData); `dirs_next::data_dir()` → `%APPDATA%` (Roaming). `$HOME` does not exist on Windows (`USERPROFILE`/`HOMEDRIVE`+`HOMEPATH` do), so every `env::var("HOME")` branch is dead on Win and must be replaced.

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| DX12 backend tier | `WgpuBackend::new_async` | `crates/vox_render/src/gpu/wgpu_backend.rs:29` | additive; Linux skips DX12 (no adapter) |
| DX12 in present-adapter resolve | `resolve_present_adapter_info` | `crates/vox_app/src/bin/ochroma_editor.rs:~721` | only under `OCHROMA_GI=gpu` |
| anchored `quick_save_path()` | F5/F9 quicksave | `crates/vox_app/src/bin/engine_runner.rs:2056-2067` | now `data_dir()`-based + `create_dir_all` |
| anchored theme path | `load_tokens` | `crates/vox_app/src/bin/ochroma_editor.rs:128-138` | `OCHROMA_ASSET_DIR` else exe-dir; keeps `unwrap_or_default` |
| `cache_subdir()` (spirv) | `spirv_cache_dir()` | `spectra/rust/spectra-gpu/src/vulkan_backend.rs:1546` | spectra-native graph only |
| `cache_subdir()` (kernels) | `KernelCache::cache_dir()` | `spectra/rust/spectra-gpu/src/kernel_cache.rs:120` | spectra-native graph only |
| `cfg(unix)` gate on rpath | build script | `spectra/rust/vendor/slang-sys/build.rs:67` | spectra-native link fix |
| MSVC arm | build script | `spectra/rust/vendor/ffx-sys/build.rs` | fsr only; recommend leave OFF |
| `default-features = false` | dep decl | `crates/vox_render/Cargo.toml` (spectra-renderer/spectra-gpu) | drops cuda/optix/dlss for Win spectra-native |
| `build-spectra-native.ps1` | manual / xtask | `scripts/build-spectra-native.ps1` (new) | sets `SLANG_DIR`, prepends slang DLL dir to `PATH`, `SPECTRA_BACKEND=vulkan`, NO `VK_ICD_FILENAMES` |

---

## 8. Coupling → Fix Table

| # | Linux assumption (file:line) | Windows fix | Status |
|---|---|---|---|
| 1 | wgpu order Vulkan→GL→all() (`wgpu_backend.rs:29-33`) | insert explicit DX12 tier between Vulkan and GL | **verifiable-now** (compiles, Linux unchanged); DX12 viewport = validate-on-Win |
| 2 | `quick_save_path()` = `"saves/quicksave.json"` (`world_save.rs:202-205`) | rebase on `dirs_next::data_dir()/ochroma/saves` + `create_dir_all` | **verifiable-now** (Linux save location moves — confirm) |
| 3 | theme `"assets/ui/..."` rel CWD (`ochroma_editor.rs:128-138`) | anchor to `OCHROMA_ASSET_DIR`/exe-dir; keep fallback | **verifiable-now** |
| 4 | `PathBuf::from("civitas_data")` rel CWD (`instanced.rs:413`) | exe-dir or `CIVITAS_DATA_DIR` | **verifiable-now** (civitas, not editor) |
| 5 | `slangc` no `.exe`, PATH split `:` (`vulkan_backend.rs:1492-1510`) | try `slangc.exe`; `std::env::split_paths` | **verifiable-now** (compiles); resolve = validate-on-Win |
| 6 | SPIR-V cache `None` on Win (`vulkan_backend.rs:1546-1558`) | `dirs::cache_dir()` root | **verifiable-now**; cache hit = validate-on-Win |
| 7 | kernel cache `/tmp` literal (`kernel_cache.rs:120-141`) | `dirs::cache_dir()` then `temp_dir()` | **verifiable-now**; lands in `C:\tmp` today, not disabled |
| 8 | `slang-sys` `-Wl,-rpath` unconditional (`build.rs:67`) | gate behind `cfg(unix)`; add Win `.lib` link-search | **verifiable-now** (static); link = validate-on-Win |
| 9 | spectra-* defaults pull `cuda/optix/dlss` (Cargo) | `default-features = false` + Win feature subset | **verifiable-now** (static); build = validate-on-Win |
| 10 | `ffx-sys` `-G "Unix Makefiles"` + GCC flags (`build.rs`) | `cfg(windows)` cmake gen + drop GCC flags + `vulkan-1`/no `stdc++` | **validate-on-Win**; recommend leave `fsr` OFF |
| 11 | `.cargo/config.toml:21,24` Linux SLANG_DIR + GCC `-isystem` | `[target.'cfg(windows)'.env]` override / scope to `cfg(unix)` | **verifiable-now** (inert for default build) |
| 12 | bash `build-spectra-native.sh` (LD_LIBRARY_PATH) | add `build-spectra-native.ps1` (PATH); keep `.sh` | **verifiable-now** (new file) |
| 13 | `-fshort-wchar` keystone (`ffx-sys build.rs:176-183,394`) | omit on MSVC (native 2-byte wchar_t) | **UNPROVEN — validate-on-Win** via static_assert link test |
| 14 | `Command::new("colmap")` (`colmap_pipeline.rs:178`) | n/a — optional photogrammetry, not editor | note only |
| 15 | four sibling repos via `../../../` rel path deps | mirror `C:\src\{ochroma,spectra,crucible,openusd-rs}` | **verifiable-now** (mandatory even for default build) |

---

## 9. Cross-Platform Changes That Do NOT Break Linux (safe work from here)

All of these compile and run on Linux with byte-identical behavior (or a flagged save-location move), and are doable now from the Linux box:

1. DX12 tier in `wgpu_backend.rs` + `resolve_present_adapter_info` (additive; Linux skips DX12).
2. `quick_save_path()` → `data_dir()`-based + `create_dir_all` (flag: Linux save location moves from `CWD/saves` to XDG data dir — confirm with user).
3. Theme path + `civitas_data` anchoring (exe-dir / env override; falls back to current behavior).
4. slangc `.exe` + `std::env::split_paths` (Linux: `slangc` still resolves, split on `:`).
5. Cache roots via `dirs::cache_dir()` in `spirv_cache_dir`/`kernel_cache`/`spectra-bin` (Linux: relocates a regenerable cache, one cold rebuild).
6. `slang-sys/build.rs:67` rpath behind `cfg(unix)` (Linux behavior identical).
7. `ffx-sys` GCC path behind `cfg(unix)` + a `cfg(windows)` arm (Linux compiles the same `cfg(unix)` body).
8. `default-features = false` on spectra-* with an explicit Linux feature list reproducing today's set (cuda/ocio/dlss/optix on Linux).
9. New files: `build-spectra-native.ps1`, `build-editor.ps1`, `[target.'cfg(windows)'.env]` block in `.cargo/config.toml`.
10. Test helpers: replace `env::var("HOME").unwrap()` with `dirs_next::home_dir()`/`dirs::cache_dir()` in `splat_backend_tests/*` (test-only; Linux identical).

Validation from Linux: `cargo build -p vox_app --bin ochroma_editor` and `cargo build -p vox_render --no-default-features --features crucible` must both stay green; a unit test asserts the DX12 tier ordering and the slangc resolver.

---

## 10. Win 11 Dev-Session Recipe

### 10.1 Install (Phase A — portable editor only)

- Rust MSVC toolchain: `rustup toolchain install stable-x86_64-pc-windows-msvc`
- Visual Studio 2022 Build Tools (provides `link.exe`, Windows SDK)
- Up-to-date AMD Adrenalin / NVIDIA Game Ready driver (ships the Vulkan ICD + DX12 — **no LunarG SDK needed for the editor**)
- Check out all four repos side by side:
  `C:\src\ochroma`, `C:\src\spectra`, `C:\src\crucible`, `C:\src\openusd-rs`

### 10.2 Build, open, render, save (Phase A)

```powershell
cd C:\src\ochroma
cargo run -p vox_app --bin ochroma_editor                 # opens 1600x900 editor window
cargo run -p vox_app --bin ochroma_editor -- --frames 120 # headed smoke; exits 0
cargo run -p vox_app --bin ochroma_editor -- --shot out.png  # writes non-blank PNG
# In the editor: build/cook a scene, press Save (F5), then Load (F9).
# File appears under %APPDATA%\ochroma\saves\
```

If the editor refuses to open on an RDP/headless box (software GPU): `$env:OCHROMA_ALLOW_SOFTWARE_GPU=1`. If Vulkan adapter selection is flaky, the DX12 tier should pick up automatically (validate).

### 10.3 Install + build (Phase B — optional spectra-native, deferred)

- LunarG Vulkan SDK (for `VULKAN_SDK\include` bindgen headers + `vulkan-1.lib`)
- slang Windows SDK (`slang.dll`, `slang.lib`, `slangc.exe`)
- LLVM/clang (libclang for bindgen)
- (only if `fsr`) CMake + Ninja or VS2022 generator; CUDA toolkit only if not using `default-features=false`

```powershell
# after the Phase-B build.rs fixes land:
.\scripts\build-spectra-native.ps1 -- run -p vox_app --bin ochroma_editor --features spectra-native
# .ps1 sets SLANG_DIR, prepends the slang DLL dir to PATH, SPECTRA_BACKEND=vulkan, does NOT set VK_ICD_FILENAMES
```

---

## 11. Cache-Size Control + Build Bloat

- **Bounded layout:** all regenerable caches under one root `%LOCALAPPDATA%\Ochroma\cache\{spirv,kernels,assets,blas}`, overridable via a single `OCHROMA_CACHE_DIR`. Saves stay under `%APPDATA%\ochroma\saves` and are never pruned.
- **LRU/size cap:** `OCHROMA_CACHE_MAX_MB` + a prune helper invoked **only at cache-dir resolution / startup** (never while the renderer holds the cache), capping by total bytes, applied to spirv/kernels/assets only.
- **Shared target dir:** the Linux box's 27 GB target dirs prove per-repo `target/` balloons, multiplied by four side-by-side repos. Set `CARGO_TARGET_DIR=C:\cargo-target` (per-machine env — `.cargo/config.toml` cannot conditionally set it by OS without breaking Linux).
- **sccache:** `RUSTC_WRAPPER=sccache`, `SCCACHE_DIR=%LOCALAPPDATA%\Mozilla\sccache`, `SCCACHE_CACHE_SIZE` cap. Note: sccache C++ object caching for the cmake-driven `ffx-sys` is unproven and only relevant if `fsr` is later enabled.

---

## 12. Blockers / Unknowns — Resolvable ONLY on a Real Windows Build

(run on Win 11):

1. `cargo build -p vox_app --bin ochroma_editor` links under `link.exe` with all four repos present; `x11-dl 2.21.0` (in `Cargo.lock` via winit's Linux transitive group) does NOT compile on Windows (believed target-gated by winit `cfg(target_os)` — unproven).
2. wgpu acquires a **hardware** Vulkan adapter on AMD AND NVIDIA via the auto-discovered ICD (no ICD file); DX12 tier works when Vulkan is forced off.
3. `--frames 120` exits 0; `--shot` PNG has non-bg pixels — the BGRA/RGBA swizzle keyed off `surface_format` must be correct (risk: an unexpected 10-bit/HDR surface format corrupts the PNG).
4. `PresentMode::Mailbox` is accepted by the Win surface — **the code has no Mailbox→Fifo fallback**; recommend ADDING one as a portability fix rather than deferring (Fifo is the wgpu-guaranteed mode).
5. Save→load round-trips byte-identical to `%APPDATA%\ochroma\saves`; the launch-from-shortcut (arbitrary CWD) asset/save anchoring works.
6. **Phase B only:** `slang.dll` loads at build time, MSVC links against `slang.lib`, bindgen resolves `<stddef.h>` from LLVM + `$VULKAN_SDK/include`, the `-fshort-wchar` ABI / `FFX_*_CONTEXT_SIZE` static_asserts hold under MSVC, and `cudarc` either builds with a CUDA toolkit or is dropped via `default-features=false`.
7. KHR ray-query / mega-geometry feature bits per AMD/NVIDIA Win driver; software-BVH fallback path (less-tested on this HW class). Discrete-GPU memory-type selection — the headless backend assumes HOST_VISIBLE|HOST_COHERENT (tuned for unified-memory iGPUs); discrete desktop GPUs may select different memory types.

---

## 13. Phased Plan

**Phase 0 — from Linux now (safe, no Linux regression):**
1. DX12 tier + present-adapter resolve.
2. Anchor `quick_save_path()`, theme path, `civitas_data`.
3. slangc `.exe`/`split_paths`; cache roots via `dirs::cache_dir()`; add `dirs` to `spectra-gpu/Cargo.toml`.
4. `cfg(unix)` gate on `slang-sys` rpath + `ffx-sys` GCC body; add `cfg(windows)` arms (compile-gated).
5. `default-features = false` on spectra-* + explicit Linux feature list.
6. New scripts: `build-editor.ps1`, `build-spectra-native.ps1`; `[target.'cfg(windows)'.env]` config block.
7. Add Mailbox→Fifo fallback (portability fix, Linux-safe).
8. Add a `windows-latest` CI job building the **actual** editor (`cargo build -p vox_app --bin ochroma_editor`) with four repos side by side — distinct from the existing `--no-default-features` job which does NOT prove the editor builds.

**Phase 1 — on the Win11 box (editor deliverable):** install MSVC + drivers, lay out `C:\src\{...}`, run the §10.2 recipe; prove open/render/--shot/save-load on AMD and NVIDIA.

**Phase 2 — on the Win11 box (spectra-native, optional):** install Vulkan SDK + slang + LLVM; resolve the CUDA build coupling; build `--features spectra-native`; prove the slang→SPIR-V path and runtime RT probing. `fsr` stays OFF (separate substantial MSVC ffx-sys effort).

---

## 14. Out of Scope

- `fsr`/FFX on Windows (full MSVC ffx-sys rewrite + `-fshort-wchar`/`spectra_gcc_compat.h`; recommend OFF indefinitely for the editor).
- The GPU-native viewport replacing the CPU `SoftwareRasteriser` (the editor opens/renders/saves with the CPU rasteriser today; GPU viewport is an enhancement).
- macOS.
- `urban_horizon` as a full Windows target (it builds vox_render with spectra-native ON; only `ochroma_editor` is in the portable set).

---

## 15. Related Plans / Designs

- Depends on: the four-repo relative layout being mirrored on Windows.
- Required before: any Windows CI job that claims to prove the editor builds.
- Related: `[Spectra path tracer runs]`, `[GPU-first directive]`, `[Config-first / no-hardcode directive]`.
