# Asset Import Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three stub importers in `import_pipeline.rs` to use the production-ready loaders already in vox_data (ply_loader, gltf_import, vxm), and wire the ContentBrowser to trigger import so users can drag assets into the scene.

**Architecture:** `import_pipeline.rs` is the single entry point (`import_asset(path, settings)`). Each format delegates to its loader: `import_ply` → `ply_loader::load_ply`, `import_gltf_full` → `gltf_import::import_gltf`, `import_vxm` → `VxmFile::read`. After fixing the importers, `ContentBrowser` gains an `ImportAsset` action that calls `import_helpers::import_and_cache` and registers the result in the `AssetLibrary`.

**Tech Stack:** Rust, vox_data (ply_loader, gltf_import, vxm, vxm_v2, library, import_helpers), vox_app (content_browser, simulation).

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/vox_data/src/import_pipeline.rs` | Modify | Replace stubs with real loader calls |
| `crates/vox_app/src/content_browser.rs` | Modify | Add `ImportAsset` action + UI button |
| `crates/vox_app/src/simulation.rs` | Modify | Handle import action, register in AssetLibrary |
| `crates/vox_data/tests/import_test.rs` | Create | Integration tests with real PLY/VXM files |

---

### Task 1: Fix import_ply to use ply_loader

**Files:**
- Modify: `crates/vox_data/src/import_pipeline.rs`
- Test: `crates/vox_data/tests/import_test.rs`

The current `import_ply` reads ASCII PLY headers and generates dummy splats. Replace it with a call to `ply_loader::load_ply()`.

- [ ] **Step 1: Write the failing test**

Create `crates/vox_data/tests/import_test.rs`:
```rust
use std::io::Write;
use vox_data::import_pipeline::{import_asset, ImportSettings};

fn write_binary_ply(path: &std::path::Path, count: usize) {
    // Minimal binary PLY with x,y,z f32 + scale_0,scale_1,scale_2 f32
    // + rot_0..rot_3 f32 + opacity f32 + f_dc_0..f_dc_2 f32
    // That's 14 f32 = 56 bytes per vertex
    use std::io::BufWriter;
    let mut f = BufWriter::new(std::fs::File::create(path).unwrap());
    write!(f, "ply\nformat binary_little_endian 1.0\nelement vertex {}\n", count).unwrap();
    write!(f, "property float x\nproperty float y\nproperty float z\n").unwrap();
    write!(f, "property float scale_0\nproperty float scale_1\nproperty float scale_2\n").unwrap();
    write!(f, "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n").unwrap();
    write!(f, "property float opacity\n").unwrap();
    write!(f, "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n").unwrap();
    write!(f, "end_header\n").unwrap();
    // Write binary splat data
    for i in 0..count {
        let x = i as f32 * 0.1f32;
        let scale = (-2.3f32).to_le_bytes(); // exp(-2.3) ≈ 0.1
        let rot_w = 1.0f32.to_le_bytes();
        let rot_zero = 0.0f32.to_le_bytes();
        let opacity = 0.0f32.to_le_bytes(); // logit(0.5) = 0.0 → sigmoid = 0.5
        let color = 0.5f32.to_le_bytes();
        f.write_all(&x.to_le_bytes()).unwrap();
        f.write_all(&0.0f32.to_le_bytes()).unwrap(); // y
        f.write_all(&0.0f32.to_le_bytes()).unwrap(); // z
        for _ in 0..3 { f.write_all(&scale).unwrap(); }
        f.write_all(&rot_w).unwrap();
        for _ in 0..3 { f.write_all(&rot_zero).unwrap(); }
        f.write_all(&opacity).unwrap();
        for _ in 0..3 { f.write_all(&color).unwrap(); }
    }
}

#[test]
fn import_ply_produces_real_splats() {
    let dir = std::env::temp_dir().join("ochroma_import_test_real");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("real.ply");
    write_binary_ply(&path, 10);

    let settings = ImportSettings::default();
    let result = import_asset(&path, &settings).unwrap();
    // Real PLY import should produce exactly 10 splats (one per vertex, no density scaling for PLY)
    assert_eq!(result.splats.len(), 10, "PLY import should produce one splat per vertex");
    // Position should not be a dummy line (first splat at x=0.0)
    assert!((result.splats[0].position[0]).abs() < 0.01, "first splat should be near x=0");
    // Scale should be exp(-2.3) ≈ 0.1, not the dummy 0.01
    let scale = result.splats[0].scale[0];
    assert!(scale > 0.05 && scale < 0.2, "scale should be ~0.1 not dummy 0.01, got {}", scale);

    std::fs::remove_dir_all(&dir).ok();
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p vox_data import_ply_produces_real_splats
```
Expected: FAIL — splat count or scale is wrong (stub generates dummy data).

- [ ] **Step 3: Fix import_ply in import_pipeline.rs**

In `crates/vox_data/src/import_pipeline.rs`, replace the entire `import_ply` function:
```rust
fn import_ply(path: &Path, settings: &ImportSettings) -> Result<ImportResult, String> {
    use crate::ply_loader;

    let mut splats = ply_loader::load_ply(path)
        .map_err(|e| format!("PLY load error: {:?}", e))?;

    // Apply scale factor
    if (settings.scale_factor - 1.0).abs() > f32::EPSILON {
        for s in splats.iter_mut() {
            s.position[0] *= settings.scale_factor;
            s.position[1] *= settings.scale_factor;
            s.position[2] *= settings.scale_factor;
        }
    }

    // Compute bounding box for collision
    let collision_box = if settings.generate_collision
        && settings.collision_type != CollisionGenType::None
        && !splats.is_empty()
    {
        let mut min = splats[0].position;
        let mut max = splats[0].position;
        for s in &splats {
            for i in 0..3 {
                if s.position[i] < min[i] { min[i] = s.position[i]; }
                if s.position[i] > max[i] { max[i] = s.position[i]; }
            }
        }
        Some((min, max))
    } else {
        None
    };

    Ok(ImportResult {
        splats,
        collision_box,
        material_names: vec!["default".to_string()],
        skeleton_joint_count: 0,
        animation_count: 0,
        warnings: vec![],
    })
}
```

- [ ] **Step 4: Run test**

```bash
cargo test -p vox_data import_ply_produces_real_splats
```
Expected: PASS.

- [ ] **Step 5: Ensure existing tests still pass**

```bash
cargo test -p vox_data
```
Expected: all pass (existing tests use ASCII PLY which ply_loader supports if it handles it — if not, the ASCII test will fail and you need to note a warning).

Note: `ply_loader::load_ply` supports binary PLY. The existing `test_import_ply` test writes an ASCII PLY. If it fails, add a warning to the ImportResult: "ASCII PLY fallback: vertex count only". The ASCII test just checks `!result.splats.is_empty()` which passes even with the old path — but the new code tries to parse it as binary and may return an error. In that case, add an ASCII fallback in `import_ply`:
```rust
let mut splats = ply_loader::load_ply(path).unwrap_or_else(|_| {
    // ASCII PLY or unsupported variant — fall back to empty (caller warned)
    vec![]
});
if splats.is_empty() {
    warnings.push("PLY could not be decoded as binary; splat cloud empty".to_string());
}
```
Collect warnings before the return.

- [ ] **Step 6: Commit**

```bash
git add crates/vox_data/src/import_pipeline.rs crates/vox_data/tests/import_test.rs
git commit -m "fix(import-pipeline): import_ply uses ply_loader for real splat extraction"
```

---

### Task 2: Fix import_gltf_full to use gltf_import

**Files:**
- Modify: `crates/vox_data/src/import_pipeline.rs`
- Test: `crates/vox_data/tests/import_test.rs` (add test)

The current `import_gltf_full` counts primitives and generates a sine-wave dummy. Replace with `gltf_import::import_gltf()`.

- [ ] **Step 1: Write the failing test**

Append to `crates/vox_data/tests/import_test.rs`:
```rust
fn write_minimal_glb(path: &std::path::Path) {
    // Minimal valid GLB: one triangle mesh
    // GLB = 12-byte header + JSON chunk + BIN chunk
    // We use the simplest possible valid GLB
    let json = r#"{"asset":{"version":"2.0"},"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],"bufferViews":[{"buffer":0,"byteLength":36,"byteOffset":0},{"buffer":0,"byteLength":6,"byteOffset":36}],"buffers":[{"byteLength":44}]}"#;
    let json_bytes = json.as_bytes();
    let json_len = json_bytes.len();
    let padded_json_len = (json_len + 3) & !3;
    // Three vertices: (0,0,0), (1,0,0), (0,1,0) — 3 * 12 = 36 bytes
    let mut bin_data = vec![0u8; 36];
    // v0: (0,0,0)
    // v1: (1,0,0)
    bin_data[12..16].copy_from_slice(&1.0f32.to_le_bytes());
    // v2: (0,1,0)
    bin_data[28..32].copy_from_slice(&1.0f32.to_le_bytes());
    // Indices: 0,1,2 as u16 = 6 bytes
    let indices = [0u16, 1u16, 2u16];
    let mut idx_bytes = [0u8; 6];
    for (i, &idx) in indices.iter().enumerate() {
        let b = idx.to_le_bytes();
        idx_bytes[i*2] = b[0];
        idx_bytes[i*2+1] = b[1];
    }
    bin_data.extend_from_slice(&idx_bytes);
    let bin_len = bin_data.len();
    let padded_bin_len = (bin_len + 3) & !3;

    let total_len = 12 + 8 + padded_json_len + 8 + padded_bin_len;
    let mut out = Vec::with_capacity(total_len);
    // Header
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total_len as u32).to_le_bytes());
    // JSON chunk
    out.extend_from_slice(&(padded_json_len as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
    out.extend_from_slice(json_bytes);
    out.resize(12 + 8 + padded_json_len, 0x20);
    // BIN chunk
    out.extend_from_slice(&(padded_bin_len as u32).to_le_bytes());
    out.extend_from_slice(&0x004E4942u32.to_le_bytes()); // "BIN\0"
    out.extend_from_slice(&bin_data);
    out.resize(total_len, 0);

    std::fs::write(path, &out).unwrap();
}

#[test]
fn import_gltf_produces_real_splats() {
    let dir = std::env::temp_dir().join("ochroma_gltf_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.glb");
    write_minimal_glb(&path);

    let settings = ImportSettings::default();
    let result = import_asset(&path, &settings).unwrap();
    // Real import should produce splats placed on triangle geometry (not dummy sine wave)
    assert!(!result.splats.is_empty(), "GLTF import should produce splats");
    // Splats should be in the triangle's bounding box [0..1, 0..1, 0..0]
    for s in &result.splats {
        assert!(s.position[0] >= -0.01 && s.position[0] <= 1.01,
            "splat x out of triangle range: {}", s.position[0]);
        assert!(s.position[1] >= -0.01 && s.position[1] <= 1.01,
            "splat y out of triangle range: {}", s.position[1]);
    }

    std::fs::remove_dir_all(&dir).ok();
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p vox_data import_gltf_produces_real_splats
```
Expected: FAIL — splat positions are on a sine wave, not in triangle bounds.

- [ ] **Step 3: Fix import_gltf_full**

In `crates/vox_data/src/import_pipeline.rs`, replace the entire `import_gltf_full` function:
```rust
fn import_gltf_full(path: &Path, settings: &ImportSettings) -> Result<ImportResult, String> {
    use crate::gltf_import;
    use gltf::Gltf;

    // Use gltf_import for the actual splat conversion
    let gr = gltf_import::import_gltf(path)
        .map_err(|e| format!("GLTF import error: {}", e))?;

    let mut splats = gr.splats;

    // Apply scale factor
    if (settings.scale_factor - 1.0).abs() > f32::EPSILON {
        for s in splats.iter_mut() {
            s.position[0] *= settings.scale_factor;
            s.position[1] *= settings.scale_factor;
            s.position[2] *= settings.scale_factor;
        }
    }

    // Extract metadata separately (materials, skeleton, animations)
    let gltf_doc = Gltf::open(path).map_err(|e| format!("GLTF metadata error: {}", e))?;

    let mut material_names = Vec::new();
    if settings.extract_materials {
        for mat in gltf_doc.materials() {
            material_names.push(mat.name().unwrap_or("unnamed_material").to_string());
        }
    }

    let mut skeleton_joint_count = 0;
    if settings.extract_skeleton {
        for skin in gltf_doc.skins() {
            skeleton_joint_count += skin.joints().count();
        }
    }

    let animation_count = if settings.extract_animations {
        gltf_doc.animations().count()
    } else {
        0
    };

    let mut warnings = Vec::new();
    if material_names.is_empty() {
        warnings.push("No materials found in GLTF file".to_string());
    }
    if splats.is_empty() {
        warnings.push("No geometry found — splat cloud is empty".to_string());
    }

    // Compute bounding box
    let collision_box = if settings.generate_collision
        && settings.collision_type != CollisionGenType::None
        && !splats.is_empty()
    {
        let mut min = splats[0].position;
        let mut max = splats[0].position;
        for s in &splats {
            for i in 0..3 {
                if s.position[i] < min[i] { min[i] = s.position[i]; }
                if s.position[i] > max[i] { max[i] = s.position[i]; }
            }
        }
        Some((min, max))
    } else {
        None
    };

    Ok(ImportResult {
        splats,
        collision_box,
        material_names,
        skeleton_joint_count,
        animation_count,
        warnings,
    })
}
```

- [ ] **Step 4: Run test**

```bash
cargo test -p vox_data import_gltf_produces_real_splats
```
Expected: PASS.

- [ ] **Step 5: Full vox_data suite**

```bash
cargo test -p vox_data
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add crates/vox_data/src/import_pipeline.rs crates/vox_data/tests/import_test.rs
git commit -m "fix(import-pipeline): import_gltf_full uses gltf_import for real triangle-sampled splats"
```

---

### Task 3: Fix import_vxm to use VxmFile::read

**Files:**
- Modify: `crates/vox_data/src/import_pipeline.rs`
- Test: `crates/vox_data/tests/import_test.rs` (add test)

The current `import_vxm` estimates splat count from byte count. Replace with `VxmFile::read()`.

- [ ] **Step 1: Write the failing test**

Append to `crates/vox_data/tests/import_test.rs`:
```rust
use vox_data::vxm::{VxmFile, MaterialType};
use vox_core::types::GaussianSplat;
use uuid::Uuid;

#[test]
fn import_vxm_produces_exact_splats() {
    let dir = std::env::temp_dir().join("ochroma_vxm_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.vxm");

    // Write a real VXM with 5 splats
    let mut file = VxmFile::new(Uuid::new_v4(), 5, MaterialType::Generic);
    file.splats = (0..5).map(|i| GaussianSplat {
        position: [i as f32, 0.0, 0.0],
        scale: [0.1, 0.1, 0.1],
        rotation: [0, 0, 0, 16384],
        opacity: 200,
        _pad: [0; 3],
        spectral: [100, 200, 150, 100, 80, 60, 40, 20],
    }).collect();
    let mut buf = Vec::new();
    file.write(&mut buf).unwrap();
    std::fs::write(&path, &buf).unwrap();

    let settings = ImportSettings::default();
    let result = import_asset(&path, &settings).unwrap();
    assert_eq!(result.splats.len(), 5, "VXM import should produce exactly 5 splats");
    assert!((result.splats[0].position[0]).abs() < 0.01, "first splat at x=0");
    assert!((result.splats[4].position[0] - 4.0).abs() < 0.01, "fifth splat at x=4");

    std::fs::remove_dir_all(&dir).ok();
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p vox_data import_vxm_produces_exact_splats
```
Expected: FAIL — byte-estimate produces wrong count.

- [ ] **Step 3: Fix import_vxm**

In `crates/vox_data/src/import_pipeline.rs`, replace the entire `import_vxm` function:
```rust
fn import_vxm(path: &Path, settings: &ImportSettings) -> Result<ImportResult, String> {
    use crate::vxm::VxmFile;

    let data = std::fs::read(path).map_err(|e| format!("Failed to read VXM: {}", e))?;
    let file = VxmFile::read(std::io::Cursor::new(&data))
        .map_err(|e| format!("VXM parse error: {}", e))?;

    let mut splats = file.splats;

    // Apply scale factor
    if (settings.scale_factor - 1.0).abs() > f32::EPSILON {
        for s in splats.iter_mut() {
            s.position[0] *= settings.scale_factor;
            s.position[1] *= settings.scale_factor;
            s.position[2] *= settings.scale_factor;
        }
    }

    let collision_box = if settings.generate_collision
        && settings.collision_type != CollisionGenType::None
        && !splats.is_empty()
    {
        let mut min = splats[0].position;
        let mut max = splats[0].position;
        for s in &splats {
            for i in 0..3 {
                if s.position[i] < min[i] { min[i] = s.position[i]; }
                if s.position[i] > max[i] { max[i] = s.position[i]; }
            }
        }
        Some((min, max))
    } else {
        None
    };

    Ok(ImportResult {
        splats,
        collision_box,
        material_names: vec!["vxm_default".to_string()],
        skeleton_joint_count: 0,
        animation_count: 0,
        warnings: vec![],
    })
}
```

- [ ] **Step 4: Run test**

```bash
cargo test -p vox_data import_vxm_produces_exact_splats
```
Expected: PASS.

- [ ] **Step 5: Full suite**

```bash
cargo test -p vox_data
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add crates/vox_data/src/import_pipeline.rs crates/vox_data/tests/import_test.rs
git commit -m "fix(import-pipeline): import_vxm uses VxmFile::read for exact splat extraction"
```

---

### Task 4: ContentBrowser import action + simulation wiring

**Files:**
- Modify: `crates/vox_app/src/content_browser.rs`
- Modify: `crates/vox_app/src/simulation.rs`
- Test: `crates/vox_app/tests/content_browser_test.rs`

Wire the ContentBrowser so clicking a PLY/GLTF/VXM file triggers an import into the AssetLibrary and spawns a SplatAssetComponent.

- [ ] **Step 1: Write the failing test**

Create `crates/vox_app/tests/content_browser_test.rs`:
```rust
use vox_app::content_browser::{ContentBrowser, ContentAction, classify, ContentType};
use std::path::Path;

#[test]
fn classify_ply_is_gaussian_splat() {
    assert_eq!(classify(Path::new("foo.ply")), ContentType::GaussianSplat);
}

#[test]
fn classify_glb_is_mesh() {
    assert_eq!(classify(Path::new("model.glb")), ContentType::Mesh);
}

#[test]
fn classify_vxm_is_ochroma_asset() {
    assert_eq!(classify(Path::new("asset.vxm")), ContentType::OchromaAsset);
}

#[test]
fn content_action_import_asset_variant() {
    let action = ContentAction::ImportAsset(std::path::PathBuf::from("model.glb"));
    if let ContentAction::ImportAsset(p) = action {
        assert_eq!(p.extension().unwrap(), "glb");
    } else {
        panic!("wrong variant");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p vox_app content_browser_test
```
Expected: `classify_*` tests pass, `content_action_import_asset_variant` FAIL — `ImportAsset` variant missing.

- [ ] **Step 3: Add ImportAsset to ContentAction**

In `crates/vox_app/src/content_browser.rs`, add to `ContentAction` enum:
```rust
pub enum ContentAction {
    LoadAsset(PathBuf),
    OpenMap(PathBuf),
    PlayAudio(PathBuf),
    ImportAsset(PathBuf),  // ← add this
}
```

- [ ] **Step 4: Add import button to ContentBrowser show()**

In `crates/vox_app/src/content_browser.rs`, find the `show` method (where entries are rendered). After the existing asset list item click handling, add an import button for importable types:

Find where entries are rendered (look for `ui.selectable_label` or similar). After the label is shown, add:
```rust
// After rendering the label for a selected entry:
if self.selected == Some(idx) {
    let entry = &self.entries[idx];
    let is_importable = matches!(
        entry.entry_type,
        ContentType::GaussianSplat | ContentType::Mesh | ContentType::OchromaAsset
    );
    if is_importable && ui.small_button("Import").clicked() {
        // Return action via pending_action or push to a Vec<ContentAction>
        self.pending_action = Some(ContentAction::ImportAsset(entry.path.clone()));
    }
}
```

You will need to add `pub pending_action: Option<ContentAction>` to the `ContentBrowser` struct and initialize it to `None`.

Full struct addition in `ContentBrowser`:
```rust
pub pending_action: Option<ContentAction>,
```

In `ContentBrowser::new()`:
```rust
pending_action: None,
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p vox_app content_browser_test
```
Expected: all 4 tests pass.

- [ ] **Step 6: Handle ImportAsset in simulation.rs**

In `crates/vox_app/src/simulation.rs` (or wherever `ContentAction` is handled), add handling for `ImportAsset`. Find the match on `ContentAction` or where `content_browser.pending_action` is consumed, and add:

```rust
if let Some(action) = content_browser.pending_action.take() {
    match action {
        ContentAction::ImportAsset(path) => {
            use vox_data::import_helpers::import_and_cache;
            let cache_dir = std::path::Path::new("assets/cache");
            match import_and_cache(&path, cache_dir) {
                Ok(imported) => {
                    // Register in asset library
                    let uuid = uuid::Uuid::new_v4();
                    let splat_count = imported.splat_count as u32;
                    // Load the cached VXM to get actual splats
                    if let Ok(data) = std::fs::read(&imported.cached_path) {
                        if let Ok(vxm) = vox_data::vxm::VxmFile::read(std::io::Cursor::new(&data)) {
                            world.spawn(vox_core::ecs::SplatAssetComponent {
                                uuid,
                                splat_count: vxm.splats.len() as u32,
                                splats: vxm.splats,
                            });
                            println!("[ochroma] Imported: {} ({} splats)", path.display(), splat_count);
                        }
                    }
                }
                Err(e) => eprintln!("[ochroma] Import failed: {}", e),
            }
        }
        ContentAction::LoadAsset(_) | ContentAction::OpenMap(_) | ContentAction::PlayAudio(_) => {
            // existing handling unchanged
        }
    }
}
```

- [ ] **Step 7: Full suite**

```bash
cargo test
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add crates/vox_app/src/content_browser.rs crates/vox_app/src/simulation.rs crates/vox_app/tests/content_browser_test.rs
git commit -m "feat(asset-pipeline): ContentBrowser ImportAsset action wired to import_and_cache"
```

---

## Summary

| Task | Deliverable |
|------|-------------|
| 1 | `import_ply` uses `ply_loader::load_ply` — real splat positions and scales |
| 2 | `import_gltf_full` uses `gltf_import::import_gltf` — triangle-sampled splats |
| 3 | `import_vxm` uses `VxmFile::read` — exact splat round-trip |
| 4 | ContentBrowser Import button → `import_and_cache` → ECS SplatAssetComponent |
