# Domain 8 — Platform Support

**Status:** Draft — 2026-03-29
**Scope:** Platform abstraction layer, PC/Mac/Linux, console (PS5/Xbox), mobile (iOS/Android), WebGPU/browser, VR/AR (OpenXR), cross-platform input
**Engine:** Ochroma spectral Gaussian Splatting — Rust workspace, wgpu 24, WGSL shaders, winit, openxr, gilrs, wasm-bindgen

---

## Goals

Ochroma must run on PC (Windows/Mac/Linux), console (PS5/Xbox Series X), mobile (iOS/Android), browser (WebGPU), and XR headsets (Meta Quest 3, HoloLens, SteamVR), with a single Rust codebase and WGSL shader source. The platform abstraction layer must be thin and zero-cost: each platform backend compiles only the code it needs. Spectral fidelity degrades gracefully — mobile low-tier uses 4-band spectral; the full 8-band path requires a desktop-class GPU. The CPU EWA renderer (rayon, 70 FPS / 4K) serves as the ultimate fallback for platforms without adequate compute shader support. All platform-specific code lives in `crates/vox_platform`; the engine crates (`vox_render`, `vox_core`, etc.) never `#[cfg(target_os = "...")]`.

---

## 8.1 Platform Abstraction Layer

### PlatformBackend Trait

```rust
// crates/vox_platform/src/backend.rs

pub trait PlatformBackend: Send + Sync + 'static {
    fn init_window(&mut self, config: &WindowConfig) -> WindowHandle;
    fn poll_events(&mut self) -> Vec<PlatformEvent>;
    fn swap_buffers(&mut self);
    fn get_gpu_device(&self, adapter: &wgpu::Adapter) -> wgpu::Device;
    fn storage_path(&self) -> std::path::PathBuf;
    fn get_gamepad_state(&self) -> Vec<GamepadState>;
}

pub struct WindowConfig {
    pub title:       String,
    pub width:       u32,
    pub height:      u32,
    pub fullscreen:  bool,
    pub vsync:       bool,
    pub hdr:         bool,
}

pub enum PlatformEvent {
    WindowResized(u32, u32),
    WindowFocusChanged(bool),
    CloseRequested,
    Input(InputEvent),
    ThermalStateChanged(ThermalState),  // mobile only
    SuspendRequested,                   // mobile + console
    ResumeRequested,
}
```

`WindowHandle` is a `raw-window-handle` `RawWindowHandle` wrapped in a newtype. `wgpu` accepts this directly for surface creation. The trait is object-safe; `EngineApp` holds a `Box<dyn PlatformBackend>` and calls through the vtable.

### Implementations

| Backend | Crate feature flag | Notes |
|---|---|---|
| `WinitBackend` | `platform-winit` (default) | PC/Mac/Linux; wraps `winit 0.30` |
| `ConsoleBackend` | `platform-console` | PS5 (VulkanSC) / Xbox (DX12); stubs in open build |
| `MobileBackend` | `platform-mobile` | iOS (Metal) / Android (Vulkan); winit + mobile extras |
| `WebBackend` | `platform-web` | WebGPU; winit wasm32 + web-sys canvas integration |
| `OxrBackend` | `platform-oxr` | OpenXR; wraps WinitBackend for flat displays, OXR for XR |

### PlatformCapabilities

`PlatformCapabilities` is queried once during startup and drives all quality scaling decisions. It is constructed by each backend's `capabilities()` method.

```rust
pub struct PlatformCapabilities {
    pub max_splat_count:          u32,
    pub max_texture_memory_mb:    u32,
    pub supports_compute:         bool,
    pub supports_timestamp_queries: bool,
    pub target_fps:               u32,
    pub hdr_display:              bool,
    pub spatial_audio:            bool,
    pub spectral_band_count:      u8,    // 4 or 8
    pub quality_tier:             QualityTier,
}

pub enum QualityTier { Low, Medium, High, Ultra }
```

Representative values by platform:

| Platform | max_splat_count | spectral_band_count | target_fps |
|---|---|---|---|
| PC RTX 3070+ | 10_000_000 | 8 | 60 or 120 |
| Console (PS5/Xbox) | 5_000_000 | 8 | 60 |
| Mobile High (iPhone 15 Pro) | 500_000 | 8 | 60 |
| Mobile Medium (iPhone 14) | 200_000 | 8 | 60 |
| Mobile Low (mid-range Android) | 50_000 | 4 | 30 |
| WebGPU (desktop) | 1_000_000 | 8 | 60 |
| WebGL2 fallback | 100_000 | 4 | 30 |
| XR (Quest 3) | 2_000_000 | 8 | 72 or 90 |

`QualityScaler` in `vox_render` reads `PlatformCapabilities` at startup and selects the `SplatBudget` and `ShaderVariant` accordingly. No manual quality setting is needed on console (target is fixed); on PC the user can override via settings.

---

## 8.2 Console Support (PS5 / Xbox Series X)

### Render Backend

wgpu targets the native graphics API on each console:
- PS5: Vulkan via a VulkanSC-compatible Vulkan adapter. Ochroma uses wgpu's Vulkan backend with no PS5-specific shader changes; WGSL compiles to SPIR-V via `naga` as on PC.
- Xbox Series X: Direct3D 12 via wgpu's DX12 backend. The same WGSL source compiles to DXIL via `naga`.

No render backend changes are required; the abstraction is complete.

### PS5-Specific Features

**DualSense Haptics**

The spectral audio pipeline maps directly to DualSense haptic output. `HapticsEvent` is emitted by `SpectralAudioManager` when synthesis produces significant content in specific bands:

```rust
// crates/vox_platform/src/ps5/haptics.rs

pub enum HapticsEvent {
    Impact { force: f32, frequency_band: u8 },
    Rumble { left: f32, right: f32 },
    TriggerResistance { trigger: Trigger, mode: TriggerMode },
}

pub enum Trigger { Left, Right }
pub enum TriggerMode {
    Off,
    Feedback(f32),      // constant resistance 0.0–1.0
    Weapon(f32, f32),   // start_position, end_position for click effect
    Vibration(f32, f32),// amplitude, frequency
}
```

Mapping from spectral bands to haptics:
- Band 7 (NIR-range, 720 nm+) magnitude → left trigger resistance via `TriggerMode::Feedback`. High-energy impacts and fire feel heavy on the left trigger.
- Band 0 (violet, 380 nm) magnitude → right trigger rumble. Electrical/magical effects produce right-side sensation.
- `synthesize_impact` in `vox_audio` returns a `SpectralImpactResult` with per-band peak magnitudes; `HapticsMapper::map(&impact)` converts this to a `Vec<HapticsEvent>` dispatched to the PS5 `SCE_PAD_PORT_TYPE_STANDARD` haptics API.

**PS5 SSD Streaming**

The PS5 raw SSD bandwidth is 5.5 GB/s. The cell loader (`vox_world/src/cell_loader.rs`) must use a PS5-tuned I/O path to exploit this:
- Replace the generic `tokio::fs::File` async read with `libSceNpCustomMenuUI` / DirectStorage-equivalent PS5 API calls — these DMA directly into GPU-mapped memory without CPU decompression.
- `.vxm` cell files are pre-compressed with Zstd at level 3 (fast decompression). On PC, the CPU decompresses; on PS5, the hardware decompressor runs inline on the DMA path.
- Target: a 512 MB spectral terrain cell loads in < 100 ms on PS5 (5.5 GB/s raw / compression ratio ~3× = ~560 MB/s effective → 512 MB / 560 MB/s = ~0.9 s without hardware decompression; with PS5 hardware decompressor, effective rate approaches raw bandwidth → < 100 ms).

**Activity Feed and Game Help**

`PsActivityManager` reports game context to the PlayStation Activity system using the SCE Activity API. `PsGameHelp` provides online hints; hint records are stored in `assets/ps5/game_help/` as JSON and uploaded to PlayStation's CDN at submission time. Both are stubs in the open codebase with a `platform-ps5` feature gate.

### Xbox-Specific Features

**DirectStorage GDK Integration**

`XboxStorageBackend` wraps the GDK `XStorageOpenFile` / `XStorageReadFile` APIs via a thin FFI layer. The cell loader's `IoBackend` trait has an `XboxIoBackend` implementation that requests GPU-visible heap allocations and issues `IDStorageQueue::EnqueueRequest` calls. This eliminates the CPU copy step on Xbox just as the PS5 DMA path does.

**Xbox Live Services**

`XboxLiveClient` issues REST API calls (via `reqwest` with `rustls`) to the Xbox Live services endpoints for achievements and leaderboards. Achievement definitions are in `assets/xbox/achievements.json`; `AchievementManager::unlock(id)` is the single call site. Leaderboard reads and writes go through `LeaderboardClient::submit_score` and `::fetch_top_n`.

**Smart Delivery**

The Xbox submission includes an `XboxOneCompat` variant compiled with `--features xbox-one-compat`: reduced max_splat_count (1_000_000), no ray-traced shadows (only SDF shadows), 4K capped to 30 Hz. The Xbox Series X build has full capabilities. Smart Delivery routes each console to the correct package variant automatically; no client-side detection is required.

### Certification Checklist Items

**Memory budget:** Total process memory must stay under 16 GB on both consoles. VRAM allocation plan:
- Splat geometry buffer: up to 8 GB (5_000_000 splats × 80 bytes GpuSplatFull ≈ 400 MB; streaming budget for all loaded cells ≈ 8 GB).
- Texture atlas: up to 4 GB.
- GI probe atlas: 512 MB.
- Audio buffers: 64 MB.
- Network + physics: 128 MB.
- Remaining: OS + engine overhead ≈ 3.3 GB.

**Loading time:** Cold start to first gameplay frame must complete in < 30 seconds. The async cell loader streams the player's starting cell during the loading screen; the spectral GI probes for the starting cell are pre-baked in `.vxgi` files and loaded synchronously during the load screen without blocking rendering.

**Frame pacing:** No frame time hitches > 33 ms (2× target at 60 Hz). The GPU profiler (`wgpu::QuerySet` timestamp queries, gated by `supports_timestamp_queries`) reports frame time breakdown each frame; a `FramePacingMonitor` logs any hitch > 20 ms to `frame_pacing.log` for cert submission evidence.

**TCR/XR Compliance:**
- Subtitles: all dialogue has subtitle tracks in `assets/localization/`; subtitle rendering uses `vox_ui`'s `SubtitleOverlay` component.
- Colorblind modes: spectral-to-RGB tone mapper has three LUT variants (normal, deuteranopia, protanopia, tritanopia) selectable in accessibility settings.
- Crash handling: `std::panic::set_hook` captures panics, writes a crash report to the console's crash reporting path, and calls the platform-specific crash upload API before unwinding.
- Save system: `SaveManager` uses the platform's save data API (PS5 `SceSaveData`, Xbox `XGameSaveFiles`); saves are validated with a CRC32 checksum on load.
- Online requirements: the game runs fully offline if no network is available; online features (multiplayer, leaderboards) are disabled with a UI message.

---

## 8.3 Mobile Support (iOS / Android)

### wgpu Targets

- iOS: wgpu's Metal backend. `MobileBackend` requests a `CAMetalLayer` surface via `raw-window-handle` from winit's iOS backend.
- Android: wgpu's Vulkan backend. `MobileBackend` requests a `NativeWindow` surface. Android API level minimum: 29 (Android 10) for Vulkan 1.1 support.

### Quality Tiers

`MobileQualityScaler` detects the device GPU at startup (via `wgpu::AdapterInfo`) and assigns a tier:

```rust
pub fn detect_mobile_tier(info: &wgpu::AdapterInfo) -> QualityTier {
    // Known high-tier GPUs: Apple A17 Pro, A16, Apple M2; Adreno 750, Mali-G715
    // Known medium-tier: Apple A15, Adreno 730, Mali-G610
    // All others: Low
    match gpu_tier_db::lookup(&info.name) {
        Some(t) => t,
        None => QualityTier::Low,  // conservative default for unknown hardware
    }
}
```

Tier parameters:

| Tier | Splat Count | Spectral Bands | Resolution | GI Probes | Shadows |
|---|---|---|---|---|---|
| Low | 50_000 | 4 (bands 0,2,4,6) | 720p | None | None |
| Medium | 200_000 | 8 | 1080p | Static (baked .vxgi) | SDF, simplified |
| High | 500_000 | 8 | 1440p dynamic | Real-time GI probes | Full SDF soft shadows |

**4-band spectral on Low tier:** bands 1, 3, 5, 7 are zeroed; the spectral-to-RGB tone mapper uses a 4-band LUT baked from the full 8-band mapping. The visual difference is small for natural materials and imperceptible at 720p screen resolution.

### Shader Variants

WGSL shader compilation is gated by `wgpu::Features` at runtime: if `SHADER_F16` is not supported, the spectral accumulation uses f32 throughout. The main tile rasterizer shader (`tile_raster.wgsl`) has compile-time `override` constants:

```wgsl
override SPECTRAL_BANDS: u32 = 8u;  // set to 4 on Low tier via wgpu PipelineCompilation overrides
override ENABLE_GI:      bool = true;
override ENABLE_SHADOWS: bool = true;
```

`wgpu::PipelineDescriptor.compilation_options.constants` is populated from `PlatformCapabilities` at pipeline creation time. This produces exactly two compiled shader variants per pipeline (Low and Med/High) rather than a combinatorial explosion.

### Thermal Management

```rust
// crates/vox_platform/src/mobile/thermal.rs

pub enum ThermalState { Nominal, Fair, Serious, Critical }

pub struct ThermalMonitor {
    current_state: ThermalState,
}

impl ThermalMonitor {
    /// iOS: reads from NSProcessInfo.thermalState
    /// Android: reads from android.os.PowerManager.getCurrentThermalStatus()
    pub fn poll(&mut self) -> Option<ThermalState>;
}
```

`EngineApp` polls `ThermalMonitor` each second. On `Serious`: reduce `SplatBudget` by 25%, enable frame-skipping (render every 2nd frame at 30 Hz effective). On `Critical`: reduce splat budget by 50%, drop to 30 Hz unconditionally, suspend spectral GI probes. On return to `Nominal`: restore settings over a 5-second ramp to avoid oscillation.

### Touch Input

```rust
// crates/vox_platform/src/mobile/touch.rs

pub struct TouchInputAdapter {
    virtual_joystick: Option<VirtualJoystickState>,
    pinch_start_dist: Option<f32>,
}

impl TouchInputAdapter {
    pub fn process(&mut self, events: &[PlatformEvent]) -> Vec<InputEvent> {
        // Two-finger pan → camera orbit (look_xy delta)
        // Pinch zoom → zoom axis
        // Single tap → interact
        // Double tap → fire
        // Virtual joystick area → move_xy
    }
}
```

The virtual joystick is rendered by `vox_ui`'s `VirtualJoystickWidget` when `PlatformCapabilities::input_type == InputType::Touch`. Its position and dead-zone radius are configurable in `settings.toml`.

### Battery Optimization

When `PlatformEvent::SuspendRequested` fires (app backgrounded):
1. `vox_render` drops to 10 Hz (skip 5 of every 6 frames, render one to keep the OS compositor happy).
2. `vox_audio` calls `SpatialAudioManager::suspend()` which drains the audio queue and pauses the rodio output stream.
3. Active QUIC connections call `ClientNetworkManager::suspend()` which sends a `WorldEvent::PlayerIdle` to the server and suppresses input sends.

On `ResumeRequested`: restore all systems over one frame; re-establish the network keep-alive immediately.

### App Thinning & Asset Variants

Mobile asset pipeline produces `SplatCompressed` at 4-band for Low tier (20 bytes/splat vs 40 bytes for full 8-band) and full 40-byte `SplatCompressed` for Med/High. These are stored in separate asset packs:

- iOS: `OnDemandResources` tags `splats-low` and `splats-high`; App Slicing delivers only the appropriate pack.
- Android: App Bundle with `<dist:module>` for each splat pack; `SplitInstallManager` requests the appropriate module on first launch.

Base APK size target is < 150 MB by storing only the initial play area's splat data in the base module. All other cells are streamed from CDN or delivered as dynamic feature modules.

---

## 8.4 WebGPU / Browser

### WASM Target

```toml
# crates/vox_app/Cargo.toml (excerpt)
[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen        = "0.2"
web-sys             = { version = "0.3", features = ["Window", "Document", "HtmlCanvasElement", "AudioContext"] }
wasm-bindgen-rayon  = "1.2"
js-sys              = "0.3"
```

Build pipeline:

```
cargo build --target wasm32-unknown-unknown --features platform-web
wasm-bindgen --target web --out-dir dist/ target/wasm32-unknown-unknown/release/vox_app.wasm
wasm-opt -O3 -o dist/vox_app_opt.wasm dist/vox_app_bg.wasm
```

`wasm-opt -O3` typically reduces binary size by 15–20% and improves runtime performance by ~10% for compute-heavy WASM. The final bundle is assembled by `Vite` from the `web/` directory, which also handles asset hashing and CDN upload configuration.

### Asset Streaming

Splat cell files are served from a CDN. The browser loader uses `fetch()` with `Range` headers to load individual cell tiles on demand, mirroring the native async cell loader:

```rust
// crates/vox_platform/src/web/asset_loader.rs

pub async fn load_cell_range(url: &str, byte_range: (u64, u64)) -> Result<Vec<u8>> {
    let window = web_sys::window().unwrap();
    let resp = JsFuture::from(window.fetch_with_str_and_init(
        url,
        web_sys::RequestInit::new().headers(
            js_sys::Object::from(js_sys::Array::of1(
                &js_sys::Array::of2(&"Range".into(), &format!("bytes={}-{}", byte_range.0, byte_range.1).into())
            ))
        ),
    )).await?;
    // ...
}
```

A `ServiceWorker` caches loaded cells in `IndexedDB` (keyed by `cell_id + version_hash`). On second visit, cells are served from the local cache without network requests, enabling offline play for previously-visited areas.

### Audio

`vox_audio`'s rodio backend is replaced on `wasm32` by a `WebAudioBackend` that wraps the browser's `AudioContext` API via `web-sys`:

```rust
#[cfg(target_arch = "wasm32")]
pub struct WebAudioBackend {
    ctx:   web_sys::AudioContext,
    nodes: HashMap<SoundId, web_sys::AudioBufferSourceNode>,
}
```

Spectral synthesis output (a `Vec<f32>` PCM buffer per frame) is written into a `web_sys::AudioBuffer` and played via `AudioBufferSourceNode`. The spatial audio panning uses `PannerNode` for azimuth/elevation instead of the native `SpatialAudioManager`'s HRTF convolution (browser HRTF is less accurate but avoids shipping a 2 MB HRTF dataset in the WASM bundle).

### Parallel EWA via WebWorkers

The CPU EWA fallback uses `wasm-bindgen-rayon` to distribute tile work across `WebWorker` threads. Rayon's thread pool is initialised with:

```rust
rayon::ThreadPoolBuilder::new()
    .num_threads(wasm_bindgen_rayon::current_thread_worker_count())
    .spawn_handler(wasm_bindgen_rayon::spawn_handler)
    .build_global()
    .unwrap();
```

The standard rayon EWA tile loop in `spectra_render.rs` runs unmodified; only the thread pool initialisation differs. Typical browser thread count is 4–8 workers, giving 4–8× speedup over single-threaded WASM.

### WebGL2 Fallback

For browsers without WebGPU support (estimated ~15% of users as of 2026):

- Compile with `--features webgl2-fallback`.
- WGSL shaders are cross-compiled to GLSL ES 3.0 via `naga`'s GLSL output backend.
- Compute shaders (tile assignment, depth sort, GI baking) are replaced with equivalent vertex/fragment shader passes on a fullscreen quad — less efficient but functionally equivalent.
- `spectral_band_count` reduced to 4; EWA rasterizer uses the CPU rayon path instead of GPU compute.
- Target: 30 FPS at 720p for 100K splats on a 2021 mid-range laptop with integrated graphics.

### Persistent Storage

- Save data: `IndexedDB` via `rexie` crate (async IndexedDB wrapper). `SaveManager` on web uses `WebSaveBackend` which wraps `rexie::Database`.
- Settings: `localStorage` via `web-sys`; serialised as JSON.

---

## 8.5 VR / AR Support

### OpenXR Backend

```rust
// crates/vox_platform/src/oxr/backend.rs

pub struct OxrBackend {
    instance:       openxr::Instance,
    session:        openxr::Session<openxr::Vulkan>,  // or openxr::D3D12 on Windows
    frame_waiter:   openxr::FrameWaiter,
    frame_stream:   openxr::FrameStream<openxr::Vulkan>,
    swapchain_left:  openxr::Swapchain<openxr::Vulkan>,
    swapchain_right: openxr::Swapchain<openxr::Vulkan>,
    stage_space:    openxr::Space,
    hand_trackers:  Option<[openxr::HandTracker; 2]>,
}
```

`OxrBackend` implements `PlatformBackend`. `init_window` creates the OpenXR session instead of a desktop window; `poll_events` drains the OpenXR event queue and translates to `PlatformEvent`. `swap_buffers` calls `frame_stream.end(...)` to submit both eye views to the compositor.

### Stereo Rendering

The EWA render path is invoked twice per frame — once per eye — with eye-specific view matrices derived from `openxr::View` structs returned by `locate_views`. The expensive tile-assignment compute pass runs once with a frustum that is the union of both eye frustums, then both rasterization passes reuse the same tile→splat assignment, saving ~40% of tile assignment cost.

```
per-frame:
  1. tile_assign.wgsl (union frustum) — run once
  2. depth_sort.wgsl              — run once
  3. tile_raster_left.wgsl        — left eye view matrix
  4. tile_raster_right.wgsl       — right eye view matrix
  5. tonemap_left.wgsl / tonemap_right.wgsl
```

### Foveated Rendering

Requires OpenXR extension `XR_EXT_eye_gaze_interaction` (supported on Quest Pro, PSVR2, HoloLens 2).

```rust
pub struct GazeSampler {
    eye_gaze_action: openxr::Action<openxr::Posef>,
}

impl GazeSampler {
    pub fn sample(&self, session: &openxr::Session<openxr::Vulkan>, frame: u64) -> GazeDirection;
}
```

`GazeSampler::sample()` polls at 120 Hz (matching the display refresh). `FoveatedEwaRenderer` uses the gaze direction to partition the screen into three rings and adjusts `SplatBudget` per ring:

| Zone | Angular Radius from Gaze | Splat Density | Spectral Bands |
|---|---|---|---|
| Fovea | 0–15° | 100% | 8 |
| Para-fovea | 15–30° | 50% | 8 |
| Periphery | 30°+ | 25% | 4 |

The tile assignment shader takes the gaze position as a uniform and outputs a `tile_density` override per tile. The rasterizer reads `tile_density` and caps the number of splats drawn per tile accordingly. This achieves ~2.5× reduction in total splat draws, enabling 90 Hz on Quest 3.

### Asynchronous TimeWarp (Reprojection)

`ReprojectionEngine` kicks in when a frame misses its deadline (detected by the OpenXR frame timing API):

1. The previous frame's colour and depth textures are kept in `ReprojectionBuffer { color: Texture, depth: Texture, submitted_pose: Posef }`.
2. Current head pose is acquired from OpenXR.
3. `reproject.wgsl` warps the previous colour texture using the depth buffer for occlusion-correct reprojection: each output pixel's 3D position is reconstructed from depth, then projected with the current pose, and the previous colour is sampled at that location.
4. The reprojected frame is submitted to the OpenXR compositor as a `XR_COMPOSITION_LAYER_REPROJECTION_MODE_ORIENTATION_ONLY` layer when depth data is unavailable, or as a full `DEPTH_REPROJECTION` layer when the splat depth buffer is provided.

Reprojection is a fallback only; the primary target is native 90 Hz with no reprojection needed.

### Hand Tracking

```rust
pub struct OxrHandTracker {
    tracker: openxr::HandTracker,
    joints:  [openxr::HandJointLocation; 26],
}

impl OxrHandTracker {
    pub fn update(&mut self, session: &openxr::Session<openxr::Vulkan>, space: &openxr::Space, time: openxr::Time);
    pub fn joint_transform(&self, joint: HandJoint) -> Option<glam::Mat4>;
}
```

26 joint positions per hand are mapped to the `HandSkeleton` used by the character animation system. `HandIkSolver` drives the wrist, knuckle, and fingertip splats of a hand splat assembly (a `ReplicatedSplatSet` with `SplatCategory::Animated`). The IK solver runs on the CPU using `glam`; hand splats update at 60 Hz from the OpenXR hand tracking API's native rate.

### Passthrough AR

`OxrPassthrough` wraps `XR_FB_passthrough` (Meta) or `XR_MSFT_composition_layer_reprojection` (HoloLens):

```rust
pub struct OxrPassthrough {
    layer: openxr::ext::PassthroughLayer,
}
```

In passthrough mode, the camera feed is rendered as the background layer; the Ochroma splat scene is composited on top with per-splat opacity honoured. `SpectralAR` extends passthrough: the spectral renderer annotates real-world objects detected by the passthrough camera's depth map with overlay splat fields. For example, a detected flat surface can receive a spectral material overlay showing the engine's `spectral` values at that location.

### XR Interaction System

```rust
pub trait XrInteractable: Send + Sync {
    fn on_hover(&mut self, hand: HandSide, distance: f32);
    fn on_grab(&mut self, hand: HandSide, grab_point: glam::Vec3);
    fn on_release(&mut self, hand: HandSide);
}

pub struct XrGrabComponent {
    pub physics_body:   rapier3d::prelude::RigidBodyHandle,
    pub grab_offset:    glam::Vec3,
    pub currently_held: Option<HandSide>,
}
```

`XrInteractionSystem` queries `OxrHandTracker` joint positions each frame, finds `XrInteractable` entities whose splat bounding spheres intersect within 5 cm of a fingertip joint, and fires the appropriate callbacks. `XrGrabComponent` drives the Rapier body's velocity to follow the hand joint transform when held.

---

## 8.6 Input System (Cross-Platform)

### InputAction & InputDevice

```rust
// crates/vox_platform/src/input/mod.rs

pub trait InputDevice: Send + Sync {
    fn poll(&mut self) -> Vec<InputEvent>;
    fn device_type(&self) -> DeviceType;
}

pub enum DeviceType { KeyboardMouse, Gamepad, Touch, XrController, XrHand }

pub enum InputEvent {
    ButtonPressed(InputAction),
    ButtonReleased(InputAction),
    AxisChanged(InputAxis, f32),
    TextInput(char),
}

pub enum InputAction {
    MoveForward, MoveBack, MoveLeft, MoveRight,
    LookUp, LookDown, LookLeft, LookRight,
    Jump, Fire, Interact, Pause, Sprint,
    // extensible by game layer via InputAction::Custom(u32)
    Custom(u32),
}
```

`InputAction` values are defined in `keybindings.toml` (already in-tree); the `KeyBindings` loader maps physical keys/buttons to `InputAction` values. `InputSystem` collects all `InputDevice` implementations, polls each each frame, routes events through the binding table, and outputs a `FrameInput { actions: HashSet<InputAction>, axes: HashMap<InputAxis, f32> }` consumed by the game layer.

### Implementations

**KeyboardMouseInput** (winit):
- Wraps `winit::event::KeyEvent` and `MouseButton`.
- Already implemented in `vox_app`; migrates to `vox_platform/src/input/keyboard_mouse.rs`.

**GamepadInput** (gilrs):
```rust
pub struct GamepadInput {
    gilrs: gilrs::Gilrs,
}

impl InputDevice for GamepadInput {
    fn poll(&mut self) -> Vec<InputEvent> {
        let mut events = Vec::new();
        while let Some(gilrs::Event { event, .. }) = self.gilrs.next_event() {
            match event {
                gilrs::EventType::ButtonPressed(btn, _) => { /* map via binding table */ }
                gilrs::EventType::AxisChanged(axis, val, _) => { /* map axis with deadzone */ }
                _ => {}
            }
        }
        events
    }
}
```

`gilrs` supports Xbox, PlayStation, Switch Pro, and generic HID gamepads. Deadzone and axis curve configuration is in `keybindings.toml` under `[gamepad]`.

**TouchInput**: implemented in `vox_platform/src/mobile/touch.rs` as described in 8.3.

**OxrControllerInput**: polls OpenXR action state for standard controller bindings (grip, trigger, thumbstick) via `openxr::Action<f32>` and `openxr::Action<bool>`. Binding profiles for Oculus Touch, Index Controller, and WMR controller are in `assets/openxr/interaction_profiles/`.

### Axis Deadzone & Remapping

```rust
pub struct AxisConfig {
    pub deadzone:     f32,  // raw values below this are clamped to 0
    pub sensitivity:  f32,  // multiplier after deadzone
    pub invert:       bool,
    pub curve:        AxisCurve,  // Linear, Quadratic, Cubic
}
```

`AxisConfig` per device per axis is loaded from `keybindings.toml`. The `InputSystem::remap()` function applies deadzone, curve, and sensitivity before producing the final `f32` axis value.

### Accessibility

- One-handed keyboard mode: `AccessibilityMode::OneHandedLeft` / `OneHandedRight` maps half the keyboard to cover all actions (configured in `keybindings.toml` under `[accessibility]`).
- Console text entry: `TextInputOverlay` in `vox_ui` renders a virtual on-screen keyboard; triggered by `InputSystem::request_text_input()`.
- Input recording: `InputRecorder` stores `(frame, InputEvent)` pairs to a `Vec`; `InputReplayer` can play them back deterministically for debugging and cinematics.

```rust
pub struct InputRecorder {
    recording: Vec<(u64, InputEvent)>,
    is_active: bool,
}

impl InputRecorder {
    pub fn start(&mut self);
    pub fn stop(&mut self) -> RecordedInput;
    pub fn record(&mut self, frame: u64, event: InputEvent);
}
```

---

## File Map

```
crates/vox_platform/
  Cargo.toml              # winit, gilrs, openxr, wasm-bindgen, web-sys, wasm-bindgen-rayon (feature-gated)
  src/
    lib.rs                # pub mod, PlatformBackend trait, PlatformCapabilities, WindowConfig
    backend.rs            # PlatformBackend trait definition
    capabilities.rs       # PlatformCapabilities, QualityTier, detect_tier()
    quality_scaler.rs     # QualityScaler: reads capabilities, sets SplatBudget + ShaderVariant

    winit/
      backend.rs          # WinitBackend impl
    mobile/
      backend.rs          # MobileBackend impl
      thermal.rs          # ThermalMonitor (iOS + Android)
      touch.rs            # TouchInputAdapter
      quality.rs          # MobileQualityScaler
    web/
      backend.rs          # WebBackend impl
      asset_loader.rs     # HTTP range fetch, ServiceWorker integration
      audio.rs            # WebAudioBackend
      storage.rs          # IndexedDB + localStorage SaveBackend
    console/
      ps5/
        backend.rs        # ConsoleBackend PS5 stub + feature-gated real impl
        haptics.rs        # HapticsEvent, HapticsMapper, DualSense API
        storage.rs        # PS5 SSD streaming I/O backend
        activity.rs       # PS Activity Feed + Game Help stubs
      xbox/
        backend.rs        # ConsoleBackend Xbox stub + feature-gated real impl
        storage.rs        # DirectStorage GDK backend
        live.rs           # XboxLiveClient (achievements, leaderboards)
    oxr/
      backend.rs          # OxrBackend impl PlatformBackend
      session.rs          # OxrSession init, frame loop
      stereo_renderer.rs  # per-eye view matrices, shared tile assignment
      foveation.rs        # GazeSampler, FoveatedEwaRenderer
      reprojection.rs     # ReprojectionEngine, reproject.wgsl dispatch
      hand_tracking.rs    # OxrHandTracker, HandIkSolver
      passthrough.rs      # OxrPassthrough, SpectralAR
      interaction.rs      # XrInteractable, XrGrabComponent, XrInteractionSystem

    input/
      mod.rs              # InputDevice trait, InputEvent, InputAction, InputAxis
      keyboard_mouse.rs   # KeyboardMouseInput (winit)
      gamepad.rs          # GamepadInput (gilrs)
      recording.rs        # InputRecorder, InputReplayer
      remapping.rs        # AxisConfig, deadzone, curves
      accessibility.rs    # one-handed modes, text input overlay trigger

crates/vox_render/
  src/
    shader_variants.rs    # PipelineCompilationOptions from PlatformCapabilities

assets/
  openxr/
    interaction_profiles/ # Oculus Touch, Index, WMR controller binding JSONs
  xbox/
    achievements.json
  ps5/
    game_help/
  localization/           # subtitle tracks

shaders/
  reproject.wgsl          # ATW reprojection shader
  tile_raster.wgsl        # SPECTRAL_BANDS + ENABLE_GI + ENABLE_SHADOWS overrides
```

---

## Milestones

### M1 — PC Polish
Migrate existing `vox_app` input and window code into `WinitBackend` implementing `PlatformBackend`. `PlatformCapabilities` detection on PC. `QualityScaler` selects tier. `GamepadInput` via gilrs tested with Xbox and PlayStation controllers. `InputRecorder` implemented and used in one existing debug test.

**Duration:** 3 days
**Done when:** `cargo test -p vox_platform` passes; gamepad input correctly drives character movement in-engine.

### M2 — Web (WebGPU + WASM)
`WebBackend` builds and runs in Chrome Canary (WebGPU). `wasm-bindgen-rayon` parallel EWA confirmed working in browser with 4 WebWorker threads. ServiceWorker cell caching tested in Chromium DevTools offline mode. WebGL2 fallback confirmed on Firefox 122 (no WebGPU). WASM bundle < 15 MB after `wasm-opt -O3`.

**Duration:** 5 days
**Done when:** `cargo build --target wasm32-unknown-unknown --features platform-web` succeeds; 100K splat scene runs at 30+ FPS in Chrome on a 2021 laptop; ServiceWorker caches cells for offline use.

### M3 — Mobile
iOS Metal and Android Vulkan builds produce working binaries. `ThermalMonitor` triggers quality reduction in a device lab thermal test (hairdryer on device). `TouchInputAdapter` drives camera orbit on physical devices. App Thinning variants validated with `xcrun` (iOS) and `bundletool` (Android). APK base < 150 MB.

**Duration:** 6 days
**Done when:** Game runs at target FPS per tier on iPhone 14, iPhone 15 Pro, Pixel 7 (Medium), and Galaxy A54 (Low); thermal reduction fires correctly under load.

### M4 — Console
PS5 and Xbox Series X builds compile (with licensed SDKs in private branch). `HapticsMapper` converts spectral impact data to DualSense trigger resistance — validated with DualSense SDK emulator. DirectStorage cell loading tested on Xbox dev kit. Certification checklist items implemented (subtitles, crash handler, save CRC). Frame pacing monitor integrated.

**Duration:** 8 days
**Done when:** Both console builds pass internal cert pre-check tools; loading time < 30 s on PS5 hardware; no frame pacing hitches > 33 ms in a 5-minute play session.

### M5 — XR
`OxrBackend` runs on Meta Quest 3 (via Air Link and natively via Android APK). Stereo rendering confirmed correct (no eye-swap, no depth artefacts). Foveated rendering reduces GPU load by > 40% when eye tracking is active. Hand tracking drives hand splat assembly. `ReprojectionEngine` fires on deliberately-missed frames and produces artefact-free output. Passthrough AR composites correctly on Quest 3.

**Duration:** 8 days
**Done when:** 90 Hz sustained on Quest 3 with 2M splats using foveated rendering; hand tracking IK visually correct on both hands; passthrough AR overlay renders without z-fighting.

---

## Acceptance Criteria

1. A single WGSL shader source file compiles correctly via `naga` to SPIR-V (Vulkan/PS5), DXIL (DX12/Xbox), MSL (Metal/iOS), and GLSL ES 3.0 (WebGL2 fallback).
2. `PlatformBackend` implementors pass a conformance test suite: `platform_conformance_tests.rs` verifies event delivery, swap_buffers timing, and storage_path writability.
3. PC: 60 FPS at 4K with 10M splats on RTX 3070 (verified by existing render benchmark).
4. Mobile Low tier: 30 FPS at 720p with 50K splats on mid-range Android (Adreno 610 class) without thermal throttling within first 5 minutes.
5. Mobile High tier: 60 FPS at 1440p with 500K splats on iPhone 15 Pro.
6. WebGPU: 30+ FPS at 1080p with 100K splats in Chrome on a 2021 Intel integrated graphics laptop.
7. WebGL2 fallback: 30+ FPS at 720p with 100K splats on Firefox 122 on the same laptop.
8. Console: < 30 s cold start to gameplay on PS5; no frame hitches > 33 ms in 5-minute session.
9. XR: 90 Hz sustained on Quest 3 with 2M splats and foveated rendering active; < 72 Hz triggers reprojection only (< 5% of frames in a 5-minute session).
10. `vox_platform` crate contains no `#[cfg(target_os = "...")]` in `vox_render`, `vox_core`, `vox_data`, or `vox_audio` — all platform gating is in `vox_platform` (verified by `grep` in CI).
11. GamepadInput works with Xbox One, Xbox Series, DualShock 4, DualSense, and Nintendo Switch Pro controllers on PC (tested via gilrs' HID backend).
12. `InputRecorder` roundtrip: recorded session played back produces identical `FrameInput` sequence (determinism test in `cargo test`).

---

## Effort Estimate

| Component | Engineer-Days |
|---|---|
| PlatformBackend trait + WinitBackend migration | 3 |
| PlatformCapabilities + QualityScaler | 2 |
| GamepadInput (gilrs) + InputRecorder | 2 |
| WebGPU/WASM backend + wasm-bindgen-rayon | 5 |
| WebGL2 fallback + GLSL ES naga output | 3 |
| Mobile (iOS Metal + Android Vulkan) | 4 |
| Mobile quality tiers + thermal management | 2 |
| Mobile app thinning (iOS App Slicing + Android App Bundle) | 2 |
| Console stubs + PS5 haptics | 3 |
| Console cert checklist (saves, subtitles, crash, frame pacing) | 3 |
| Console DirectStorage I/O paths | 3 |
| OpenXR stereo rendering + session management | 4 |
| Foveated rendering (GazeSampler + FoveatedEwaRenderer) | 3 |
| Reprojection (ATW) | 3 |
| Hand tracking + XR interaction + grab | 3 |
| Passthrough AR + SpectralAR | 2 |
| Conformance test suite | 2 |
| **Total** | **49** |
