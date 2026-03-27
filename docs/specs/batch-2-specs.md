# Batch 2 — Detailed Specifications

Pre-requisite: Batch 1 complete (shadows, audio, animation, gizmos, GLTF import). Engine runtime uses Bevy ECS with fixed timestep.

---

## Feature 6: Character Controller

### What
A physics-driven first/third person character that walks on terrain, slides along walls, handles gravity, and can jump. This replaces the ad-hoc WASD camera movement.

### Architecture fit
- New Bevy component: `CharacterController { speed, jump_force, gravity, grounded, velocity }`
- New Bevy system in fixed schedule: `character_controller_system`
- Reads `InputResource`, writes `TransformComponent`
- Uses physics raycasting for ground detection

### Implementation
```rust
// In vox_core/src/character_controller.rs

#[derive(Component, Debug, Clone)]
pub struct CharacterController {
    pub speed: f32,
    pub sprint_multiplier: f32,
    pub jump_force: f32,
    pub gravity: f32,
    pub velocity: Vec3,
    pub grounded: bool,
    pub height: f32,      // character capsule height
    pub radius: f32,      // character capsule radius
}

impl Default for CharacterController {
    fn default() -> Self {
        Self {
            speed: 8.0, sprint_multiplier: 1.6, jump_force: 6.0,
            gravity: 20.0, velocity: Vec3::ZERO, grounded: false,
            height: 1.8, radius: 0.3,
        }
    }
}

// System: runs at fixed timestep
pub fn character_controller_system(
    fixed_time: Res<FixedTime>,
    input: Res<InputResource>,
    camera: Res<CameraState>,
    mut query: Query<(&mut CharacterController, &mut TransformComponent)>,
) {
    let dt = fixed_time.dt;
    for (mut cc, mut transform) in query.iter_mut() {
        // Ground detection: raycast downward
        let feet_y = transform.position.y - cc.height * 0.5;
        cc.grounded = feet_y <= 0.1; // simplified: ground at y=0

        // Gravity
        if !cc.grounded {
            cc.velocity.y -= cc.gravity * dt;
        } else {
            cc.velocity.y = cc.velocity.y.max(0.0);
            transform.position.y = transform.position.y.max(cc.height * 0.5);
        }

        // Movement from input
        let forward = Vec3::new(camera.forward.x, 0.0, camera.forward.z).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let mut move_dir = Vec3::ZERO;

        if input.state.is_pressed(InputSource::Key(17)) { move_dir += forward; }  // W
        if input.state.is_pressed(InputSource::Key(31)) { move_dir -= forward; }  // S
        if input.state.is_pressed(InputSource::Key(30)) { move_dir -= right; }    // A
        if input.state.is_pressed(InputSource::Key(32)) { move_dir += right; }    // D

        if move_dir.length() > 0.01 {
            move_dir = move_dir.normalize();
        }

        let speed = cc.speed;
        cc.velocity.x = move_dir.x * speed;
        cc.velocity.z = move_dir.z * speed;

        // Jump
        if cc.grounded && input.state.was_just_pressed(InputSource::Key(57)) { // Space
            cc.velocity.y = cc.jump_force;
            cc.grounded = false;
        }

        // Apply velocity
        transform.position += cc.velocity * dt;
    }
}
```

### Acceptance test
- Character walks with WASD
- Character falls when walking off a ledge
- Character jumps with Space
- Character can't fall below ground (y=0)
- Movement direction relative to camera facing

---

## Feature 7: Game UI Framework

### What
A system for rendering game HUD elements (health bars, text, menus) that scripts can control.

### Architecture fit
- New Bevy resource: `GameUI` holding UI state
- Scripts modify UI via `ctx.set_ui_text("score", "Score: 42")`
- Rendered as bitmap text overlay (same technique as walking sim HUD)
- For menus: simple state machine (MainMenu → Playing → Paused → GameOver)

### Implementation
```rust
// In vox_core/src/game_ui.rs

#[derive(Resource, Default)]
pub struct GameUI {
    pub elements: Vec<UIElement>,
    pub game_state: GameState,
    pub menu_selection: usize,
}

pub struct UIElement {
    pub id: String,
    pub text: String,
    pub position: UIPosition,
    pub color: [u8; 3],
    pub visible: bool,
    pub size: UISize,
}

pub enum UIPosition {
    TopLeft, TopCenter, TopRight,
    CenterLeft, Center, CenterRight,
    BottomLeft, BottomCenter, BottomRight,
    Custom { x: u32, y: u32 },
}

pub enum UISize { Small, Normal, Large }

pub enum GameState {
    MainMenu,
    Playing,
    Paused,
    GameOver { message: String },
}

impl GameUI {
    pub fn set_text(&mut self, id: &str, text: &str);
    pub fn set_visible(&mut self, id: &str, visible: bool);
    pub fn add_element(&mut self, element: UIElement);
    pub fn remove_element(&mut self, id: &str);
    pub fn render_to_pixels(&self, pixels: &mut [[u8; 4]], width: u32, height: u32);
}
```

### Acceptance test
- Walking sim shows "Orbs: 3/10" using GameUI system
- Pause menu shows when Escape pressed (in Playing state)
- Main menu has "Play" and "Quit" options
- Game over screen shows "YOU WIN!"

---

## Feature 8: Content Browser

### What
An editor panel that shows files in the assets directory, with type icons and search.

### Architecture fit
- Part of the editor module (SceneEditor)
- Scans a directory on disk
- Shows files grouped by type (.ply, .vxm, .wav, .rhai, .glb)
- Double-click a .ply → loads it and places in scene

### Implementation
```rust
// In vox_app/src/content_browser.rs

pub struct ContentBrowser {
    pub root_path: PathBuf,
    pub entries: Vec<ContentEntry>,
    pub selected: Option<usize>,
    pub search_query: String,
    pub current_dir: PathBuf,
}

pub struct ContentEntry {
    pub name: String,
    pub path: PathBuf,
    pub entry_type: ContentType,
    pub size_bytes: u64,
}

pub enum ContentType {
    GaussianSplat,   // .ply
    OchromaAsset,    // .vxm
    AudioClip,       // .wav, .ogg
    Script,          // .rhai
    Mesh,            // .glb, .gltf
    Map,             // .ochroma_map
    Unknown,
}

impl ContentBrowser {
    pub fn new(root: &Path) -> Self;
    pub fn scan(&mut self);  // re-read directory
    pub fn filtered_entries(&self) -> Vec<&ContentEntry>;  // apply search filter
    pub fn navigate_to(&mut self, dir: &Path);
    pub fn parent_dir(&mut self);
    pub fn show(&mut self, ctx: &egui::Context) -> Option<ContentAction>;
}

pub enum ContentAction {
    LoadAsset(PathBuf),
    OpenMap(PathBuf),
    PlayAudio(PathBuf),
}
```

### Acceptance test
- Browser shows files from assets/ directory
- Search filters entries by name
- Files grouped by type with appropriate labels
- Selecting a .ply file returns a LoadAsset action

---

## Feature 9: Hot-Reload

### What
When a .rhai script file or .ply asset changes on disk, automatically reload it without restarting.

### Architecture fit
- New Bevy system in frame schedule: `hot_reload_system`
- Uses `AssetWatcher` (already exists in vox_data) to poll for changes
- On script change: recompile Rhai script
- On asset change: reload .ply file, update AssetRef components

### Implementation
```rust
// In vox_core/src/hot_reload.rs

#[derive(Resource)]
pub struct HotReloadState {
    watcher: vox_data::hot_reload::AssetWatcher,
    watch_dirs: Vec<PathBuf>,
    last_check: f32,
    check_interval: f32,
}

impl HotReloadState {
    pub fn new(check_interval: f32) -> Self;
    pub fn watch_directory(&mut self, path: PathBuf);
    pub fn check(&mut self, dt: f32) -> Vec<HotReloadEvent>;
}

pub enum HotReloadEvent {
    ScriptChanged(PathBuf),
    AssetChanged(PathBuf),
    MapChanged(PathBuf),
}
```

### Acceptance test
- Change a .rhai file → engine recompiles it next frame
- Change a .ply file → entities using it re-render with new data
- Notification appears: "Reloaded: player.rhai"

---

## Feature 10: Save/Load ECS World

### What
Serialize the entire game state (all entities + components) to a file. Load restores exact state.

### Architecture fit
- Serialize: iterate all entities, serialize each component to JSON
- Deserialize: create entities with components from JSON
- Quick-save (F5), Quick-load (F9), Auto-save every 5 minutes

### Implementation
```rust
// In vox_data/src/world_save.rs

pub struct WorldSave {
    pub version: u32,
    pub timestamp: String,
    pub entities: Vec<SavedEntity>,
    pub resources: SavedResources,
}

pub struct SavedEntity {
    pub name: String,
    pub transform: [f32; 10],  // pos(3) + rot(4) + scale(3)
    pub asset_path: Option<String>,
    pub scripts: Vec<String>,
    pub tags: Vec<String>,
    pub custom_data: HashMap<String, serde_json::Value>,
    pub collider: Option<ColliderSave>,
}

pub struct SavedResources {
    pub time_of_day: f32,
    pub camera_position: [f32; 3],
    pub camera_rotation: [f32; 4],
}

impl WorldSave {
    pub fn from_world(world: &World) -> Self;  // serialize
    pub fn apply_to_world(&self, world: &mut World);  // deserialize
    pub fn save_to_file(&self, path: &Path) -> Result<(), String>;
    pub fn load_from_file(path: &Path) -> Result<Self, String>;
}
```

### Acceptance test
- Place 5 entities, save, restart engine, load → same 5 entities at same positions
- Quick-save (F5) creates file instantly
- Quick-load (F9) restores state
- Save file is human-readable JSON (can be edited)

---

## Integration Contract

After all 5 agents finish, the engine_runner.rs must:
1. Use `CharacterController` for player movement (not ad-hoc WASD)
2. Use `GameUI` for HUD rendering
3. Show `ContentBrowser` in editor mode
4. `HotReloadState` checks for file changes each frame
5. F5/F9 trigger save/load via `WorldSave`
