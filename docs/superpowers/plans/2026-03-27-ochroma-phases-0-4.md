# Ochroma Phases 0–4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Ochroma spectral Gaussian splatting engine from zero to a scaffolded city builder in one session.

**Architecture:** Rust workspace with engine-layer crates (`vox_core`, `vox_render`, `vox_data`) and game-layer crates (`vox_app`, `vox_sim`). CUDA kernels via `cudarc` for GPU rasterisation. Spectral rendering with 8-band SPD coefficients per Gaussian. Bevy ECS for entity management. Engine crates are game-agnostic — no city-builder concepts leak into them.

**Tech Stack:** Rust 2024 edition, CUDA (cudarc + PTX), wgpu (Vulkan backend), Bevy ECS, egui, tokio, zstd, glam, half, bytemuck, serde.

**Prioritisation:**
- **Phase 0–1:** Full implementation, tested, working
- **Phase 2–3:** Working scaffold, core logic compiles, key paths functional
- **Phase 4:** Interfaces + types + stubs, architecture locked

---

## Phase 0 — MVP

### Task 1: Workspace and Crate Scaffolding

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/vox_core/Cargo.toml`
- Create: `crates/vox_core/src/lib.rs`
- Create: `crates/vox_data/Cargo.toml`
- Create: `crates/vox_data/src/lib.rs`
- Create: `crates/vox_render/Cargo.toml`
- Create: `crates/vox_render/src/lib.rs`
- Create: `crates/vox_app/Cargo.toml`
- Create: `crates/vox_app/src/main.rs`
- Create: `.gitignore`
- Create: `CLAUDE.md`

- [ ] **Step 1: Initialise git repo**

```bash
cd /home/tomespen/git/ochroma
git init
```

- [ ] **Step 2: Create workspace root Cargo.toml**

```toml
[workspace]
resolver = "2"
members = [
    "crates/vox_core",
    "crates/vox_data",
    "crates/vox_render",
    "crates/vox_app",
]

[workspace.package]
edition = "2024"
version = "0.1.0"
license = "MIT"

[workspace.dependencies]
glam = "0.29"
uuid = { version = "1", features = ["v4", "serde"] }
half = { version = "2", features = ["bytemuck"] }
bytemuck = { version = "1", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
thiserror = "2"
zstd = "0.13"
```

- [ ] **Step 3: Create vox_core crate**

`crates/vox_core/Cargo.toml`:
```toml
[package]
name = "vox_core"
edition.workspace = true
version.workspace = true

[dependencies]
glam = { workspace = true }
uuid = { workspace = true }
half = { workspace = true }
bytemuck = { workspace = true }
serde = { workspace = true }
```

`crates/vox_core/src/lib.rs`:
```rust
pub mod types;
pub mod spectral;
```

- [ ] **Step 4: Create vox_data crate**

`crates/vox_data/Cargo.toml`:
```toml
[package]
name = "vox_data"
edition.workspace = true
version.workspace = true

[dependencies]
vox_core = { path = "../vox_core" }
zstd = { workspace = true }
thiserror = { workspace = true }
bytemuck = { workspace = true }
serde = { workspace = true }
uuid = { workspace = true }
half = { workspace = true }
```

`crates/vox_data/src/lib.rs`:
```rust
pub mod vxm;
```

- [ ] **Step 5: Create vox_render crate**

`crates/vox_render/Cargo.toml`:
```toml
[package]
name = "vox_render"
edition.workspace = true
version.workspace = true

[dependencies]
vox_core = { path = "../vox_core" }
wgpu = "24"
raw-window-handle = "0.6"
pollster = "0.4"
bytemuck = { workspace = true }
half = { workspace = true }
glam = { workspace = true }
```

`crates/vox_render/src/lib.rs`:
```rust
pub mod gpu;
pub mod spectral;
```

- [ ] **Step 6: Create vox_app crate**

`crates/vox_app/Cargo.toml`:
```toml
[package]
name = "vox_app"
edition.workspace = true
version.workspace = true

[dependencies]
vox_core = { path = "../vox_core" }
vox_data = { path = "../vox_data" }
vox_render = { path = "../vox_render" }
winit = "0.30"
pollster = "0.4"
glam = { workspace = true }
uuid = { workspace = true }
```

`crates/vox_app/src/main.rs`:
```rust
fn main() {
    println!("Ochroma engine starting...");
}
```

- [ ] **Step 7: Create .gitignore and CLAUDE.md**

`.gitignore`:
```
/target
*.ptx
*.cubin
.env
```

`CLAUDE.md`:
```markdown
# Ochroma Engine

Spectral Gaussian Splatting game engine.

## Build

```bash
cargo build
cargo test
```

## Architecture

- `vox_core` — shared types, math, spectral definitions (ENGINE — game-agnostic)
- `vox_data` — .vxm file format, asset I/O (ENGINE — game-agnostic)
- `vox_render` — GPU rendering, spectral pipeline (ENGINE — game-agnostic)
- `vox_app` — application binary, UI (GAME layer)

**Rule: Engine crates must NEVER contain game-specific concepts (buildings, zoning, traffic). Game logic belongs in vox_app or vox_sim.**

## Specs

See `docs/spec/` for phase specifications.
```

- [ ] **Step 8: Verify workspace compiles**

Run: `cargo build`
Expected: compiles with no errors

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: initialise workspace with vox_core, vox_data, vox_render, vox_app crates"
```

---

### Task 2: Core Types (`vox_core`)

**Files:**
- Create: `crates/vox_core/src/types.rs`
- Create: `crates/vox_core/src/spectral.rs`
- Test: `crates/vox_core/tests/types_test.rs`
- Test: `crates/vox_core/tests/spectral_test.rs`

- [ ] **Step 1: Write failing test for GaussianSplat**

`crates/vox_core/tests/types_test.rs`:
```rust
use vox_core::types::{GaussianSplat, SplatInstance};
use glam::{Vec3, Quat};
use uuid::Uuid;

#[test]
fn gaussian_splat_size_is_52_bytes() {
    assert_eq!(std::mem::size_of::<GaussianSplat>(), 52);
}

#[test]
fn splat_instance_has_required_fields() {
    let inst = SplatInstance {
        asset_uuid: Uuid::new_v4(),
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        instance_id: 1,
    };
    assert_eq!(inst.instance_id, 1);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vox_core`
Expected: FAIL — types not defined

- [ ] **Step 3: Implement core types**

`crates/vox_core/src/types.rs`:
```rust
use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use half::f16;
use uuid::Uuid;

/// A single Gaussian splat as stored in .vxm v0.1 (52 bytes).
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GaussianSplat {
    /// Position relative to asset origin.
    pub position: [f32; 3],
    /// Half-axes of Gaussian ellipsoid.
    pub scale: [f32; 3],
    /// Quantised unit quaternion (each component / 32767.0).
    pub rotation: [i16; 4],
    /// Linear opacity 0–255.
    pub opacity: u8,
    /// Reserved padding.
    pub _pad: u8,
    /// 8 spectral band coefficients, 380–720nm at 40nm intervals.
    pub spectral: [u16; 8], // f16 stored as u16 for Pod
}

/// A placed instance of an asset in the world.
#[derive(Debug, Clone)]
pub struct SplatInstance {
    pub asset_uuid: Uuid,
    pub position: Vec3,
    pub rotation: Quat,
    pub instance_id: u32,
}

impl GaussianSplat {
    /// Decode quantised quaternion to glam Quat.
    pub fn decoded_rotation(&self) -> Quat {
        Quat::from_xyzw(
            self.rotation[0] as f32 / 32767.0,
            self.rotation[1] as f32 / 32767.0,
            self.rotation[2] as f32 / 32767.0,
            self.rotation[3] as f32 / 32767.0,
        )
    }

    /// Get spectral coefficient at band index as f32.
    pub fn spectral_f32(&self, band: usize) -> f32 {
        f16::from_bits(self.spectral[band]).to_f32()
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vox_core`
Expected: PASS

- [ ] **Step 5: Write failing test for spectral types**

`crates/vox_core/tests/spectral_test.rs`:
```rust
use vox_core::spectral::{SpectralBands, Illuminant, spectral_to_xyz, xyz_to_srgb};

#[test]
fn d65_illuminant_has_8_bands() {
    let d65 = Illuminant::d65();
    assert_eq!(d65.bands.len(), 8);
}

#[test]
fn flat_white_spd_under_d65_is_near_white() {
    let spd = SpectralBands([1.0; 8]);
    let d65 = Illuminant::d65();
    let xyz = spectral_to_xyz(&spd, &d65);
    let rgb = xyz_to_srgb(xyz);
    // White should have all channels > 0.9
    assert!(rgb[0] > 0.8, "R={}", rgb[0]);
    assert!(rgb[1] > 0.8, "G={}", rgb[1]);
    assert!(rgb[2] > 0.8, "B={}", rgb[2]);
}

#[test]
fn zero_spd_produces_black() {
    let spd = SpectralBands([0.0; 8]);
    let d65 = Illuminant::d65();
    let xyz = spectral_to_xyz(&spd, &d65);
    let rgb = xyz_to_srgb(xyz);
    assert_eq!(rgb, [0.0, 0.0, 0.0]);
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cargo test -p vox_core`
Expected: FAIL — spectral module not implemented

- [ ] **Step 7: Implement spectral pipeline**

`crates/vox_core/src/spectral.rs`:
```rust
/// 8 spectral bands: 380nm, 420nm, 460nm, 500nm, 540nm, 580nm, 620nm, 660nm
pub const BAND_WAVELENGTHS: [f32; 8] = [380.0, 420.0, 460.0, 500.0, 540.0, 580.0, 620.0, 660.0];
pub const BAND_SPACING: f32 = 40.0;

/// 8 spectral reflectance coefficients.
#[derive(Debug, Clone, Copy)]
pub struct SpectralBands(pub [f32; 8]);

/// Illuminant SPD (power per band).
#[derive(Debug, Clone)]
pub struct Illuminant {
    pub bands: [f32; 8],
}

/// CIE 1931 2° observer colour matching functions, sampled at our 8 bands.
/// Values from CIE tables at 380, 420, 460, 500, 540, 580, 620, 660nm.
const CIE_X: [f32; 8] = [0.0014, 0.0434, 0.3362, 0.0049, 0.2904, 0.9163, 0.5419, 0.0874];
const CIE_Y: [f32; 8] = [0.0000, 0.0116, 0.0600, 0.3230, 0.9540, 0.8700, 0.3810, 0.0468];
const CIE_Z: [f32; 8] = [0.0065, 0.2074, 1.7721, 0.2720, 0.0633, 0.0017, 0.0017, 0.0000];

impl Illuminant {
    /// CIE D65 daylight (6500K), normalised.
    pub fn d65() -> Self {
        Self {
            bands: [49.98, 68.70, 100.15, 109.35, 104.05, 97.74, 86.56, 74.35],
        }
    }

    /// CIE D50 warm daylight (5000K).
    pub fn d50() -> Self {
        Self {
            bands: [25.83, 52.93, 86.68, 100.00, 100.76, 97.74, 84.34, 70.06],
        }
    }

    /// CIE Illuminant A (incandescent, 2856K).
    pub fn a() -> Self {
        Self {
            bands: [9.80, 17.68, 29.49, 45.78, 66.29, 90.01, 115.92, 142.08],
        }
    }

    /// F11 fluorescent.
    pub fn f11() -> Self {
        Self {
            bands: [3.00, 15.00, 60.00, 40.00, 80.00, 120.00, 55.00, 15.00],
        }
    }
}

/// Integrate SPD × illuminant × CIE observer → XYZ.
pub fn spectral_to_xyz(spd: &SpectralBands, illuminant: &Illuminant) -> [f32; 3] {
    let mut x = 0.0_f32;
    let mut y = 0.0_f32;
    let mut z = 0.0_f32;
    let mut norm = 0.0_f32;

    for i in 0..8 {
        let power = spd.0[i] * illuminant.bands[i];
        x += power * CIE_X[i] * BAND_SPACING;
        y += power * CIE_Y[i] * BAND_SPACING;
        z += power * CIE_Z[i] * BAND_SPACING;
        norm += illuminant.bands[i] * CIE_Y[i] * BAND_SPACING;
    }

    if norm > 0.0 {
        [x / norm, y / norm, z / norm]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// XYZ to linear sRGB.
pub fn xyz_to_srgb(xyz: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = xyz;
    let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let b = 0.0557 * x - 0.2040 * y + 1.0570 * z;
    [r.max(0.0), g.max(0.0), b.max(0.0)]
}

/// Linear sRGB to gamma-corrected sRGB.
pub fn linear_to_srgb_gamma(c: f32) -> f32 {
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}
```

- [ ] **Step 8: Run tests**

Run: `cargo test -p vox_core`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git add crates/vox_core/
git commit -m "feat(vox_core): add GaussianSplat, SplatInstance, spectral pipeline types"
```

---

### Task 3: VXM File Format (`vox_data`)

**Files:**
- Create: `crates/vox_data/src/vxm.rs`
- Test: `crates/vox_data/tests/vxm_test.rs`

- [ ] **Step 1: Write failing test for .vxm round-trip**

`crates/vox_data/tests/vxm_test.rs`:
```rust
use vox_core::types::GaussianSplat;
use vox_data::vxm::{VxmHeader, VxmFile, MaterialType};
use uuid::Uuid;

#[test]
fn header_is_64_bytes() {
    assert_eq!(std::mem::size_of::<VxmHeader>(), 64);
}

#[test]
fn round_trip_write_read() {
    let uuid = Uuid::new_v4();
    let splats = vec![
        GaussianSplat {
            position: [1.0, 2.0, 3.0],
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767], // identity quaternion (w=1)
            opacity: 255,
            _pad: 0,
            spectral: [15360; 8], // f16 for 1.0 = 0x3C00 = 15360
        },
    ];

    let file = VxmFile {
        header: VxmHeader::new(uuid, splats.len() as u32, MaterialType::Generic),
        splats: splats.clone(),
    };

    let mut buf = Vec::new();
    file.write(&mut buf).unwrap();

    let loaded = VxmFile::read(&buf[..]).unwrap();
    assert_eq!(loaded.header.magic, *b"VXMF");
    assert_eq!(loaded.header.version, 1);
    assert_eq!(loaded.header.splat_count, 1);
    assert_eq!(loaded.splats.len(), 1);
    assert_eq!(loaded.splats[0].position, [1.0, 2.0, 3.0]);
    assert_eq!(loaded.splats[0].opacity, 255);
}

#[test]
fn round_trip_many_splats() {
    let uuid = Uuid::new_v4();
    let splats: Vec<GaussianSplat> = (0..1000)
        .map(|i| GaussianSplat {
            position: [i as f32, 0.0, 0.0],
            scale: [0.05, 0.05, 0.05],
            rotation: [0, 0, 0, 32767],
            opacity: 200,
            _pad: 0,
            spectral: [15360; 8],
        })
        .collect();

    let file = VxmFile {
        header: VxmHeader::new(uuid, splats.len() as u32, MaterialType::Concrete),
        splats,
    };

    let mut buf = Vec::new();
    file.write(&mut buf).unwrap();

    let loaded = VxmFile::read(&buf[..]).unwrap();
    assert_eq!(loaded.splats.len(), 1000);
    assert_eq!(loaded.splats[999].position[0], 999.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vox_data`
Expected: FAIL — vxm module not implemented

- [ ] **Step 3: Implement VXM format**

`crates/vox_data/src/vxm.rs`:
```rust
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice, try_from_bytes};
use std::io::{Read, Write};
use thiserror::Error;
use uuid::Uuid;
use vox_core::types::GaussianSplat;

const MAGIC: [u8; 4] = *b"VXMF";
const VERSION: u16 = 1;

#[derive(Debug, Error)]
pub enum VxmError {
    #[error("invalid magic bytes")]
    InvalidMagic,
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u16),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("decompression error: {0}")]
    Decompress(std::io::Error),
    #[error("compression error: {0}")]
    Compress(std::io::Error),
    #[error("invalid data alignment")]
    InvalidAlignment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MaterialType {
    Generic = 0,
    Concrete = 1,
    Glass = 2,
    Vegetation = 3,
    Metal = 4,
    Water = 5,
}

impl MaterialType {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Concrete,
            2 => Self::Glass,
            3 => Self::Vegetation,
            4 => Self::Metal,
            5 => Self::Water,
            _ => Self::Generic,
        }
    }
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct VxmHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub flags: u16,
    pub asset_uuid: [u8; 16],
    pub splat_count: u32,
    pub material_type: u8,
    pub _pad: [u8; 23],
}

impl VxmHeader {
    pub fn new(uuid: Uuid, splat_count: u32, material_type: MaterialType) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            flags: 0,
            asset_uuid: *uuid.as_bytes(),
            splat_count,
            material_type: material_type as u8,
            _pad: [0; 23],
        }
    }

    pub fn uuid(&self) -> Uuid {
        Uuid::from_bytes(self.asset_uuid)
    }

    pub fn material(&self) -> MaterialType {
        MaterialType::from_u8(self.material_type)
    }
}

#[derive(Debug, Clone)]
pub struct VxmFile {
    pub header: VxmHeader,
    pub splats: Vec<GaussianSplat>,
}

impl VxmFile {
    pub fn write<W: Write>(&self, mut writer: W) -> Result<(), VxmError> {
        // Write header (64 bytes, uncompressed)
        writer.write_all(bytes_of(&self.header))?;

        // Compress splat data with zstd
        let splat_bytes = cast_slice::<GaussianSplat, u8>(&self.splats);
        let compressed = zstd::bulk::compress(splat_bytes, 3)
            .map_err(VxmError::Compress)?;

        // Write compressed size then compressed data
        writer.write_all(&(compressed.len() as u64).to_le_bytes())?;
        writer.write_all(&compressed)?;

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> Result<Self, VxmError> {
        // Read header
        let mut header_bytes = [0u8; 64];
        reader.read_exact(&mut header_bytes)?;
        let header: VxmHeader = *try_from_bytes(&header_bytes)
            .map_err(|_| VxmError::InvalidAlignment)?;

        if header.magic != MAGIC {
            return Err(VxmError::InvalidMagic);
        }
        if header.version != VERSION {
            return Err(VxmError::UnsupportedVersion(header.version));
        }

        // Read compressed size
        let mut size_bytes = [0u8; 8];
        reader.read_exact(&mut size_bytes)?;
        let compressed_size = u64::from_le_bytes(size_bytes) as usize;

        // Read compressed data
        let mut compressed = vec![0u8; compressed_size];
        reader.read_exact(&mut compressed)?;

        // Decompress
        let expected_size = header.splat_count as usize * std::mem::size_of::<GaussianSplat>();
        let decompressed = zstd::bulk::decompress(&compressed, expected_size)
            .map_err(VxmError::Decompress)?;

        // Cast to splats
        let splats: Vec<GaussianSplat> = cast_slice::<u8, GaussianSplat>(&decompressed).to_vec();

        Ok(VxmFile { header, splats })
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p vox_data`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add crates/vox_data/
git commit -m "feat(vox_data): implement .vxm v0.1 format with zstd compression"
```

---

### Task 4: Software Spectral Rasteriser (`vox_render`)

**Files:**
- Create: `crates/vox_render/src/spectral.rs`
- Create: `crates/vox_render/src/gpu/mod.rs`
- Create: `crates/vox_render/src/gpu/software_rasteriser.rs`
- Test: `crates/vox_render/tests/rasteriser_test.rs`

Phase 0 starts with a CPU software rasteriser to validate the spectral pipeline before CUDA integration. This de-risks the pipeline — CUDA kernels come in Task 5.

- [ ] **Step 1: Write failing test for software rasteriser**

`crates/vox_render/tests/rasteriser_test.rs`:
```rust
use vox_core::types::GaussianSplat;
use vox_core::spectral::{Illuminant, SpectralBands};
use vox_render::spectral::RenderCamera;
use vox_render::gpu::software_rasteriser::{SoftwareRasteriser, Framebuffer};
use glam::{Vec3, Mat4};

fn make_test_splat(pos: [f32; 3]) -> GaussianSplat {
    GaussianSplat {
        position: pos,
        scale: [0.5, 0.5, 0.5],
        rotation: [0, 0, 0, 32767],
        opacity: 255,
        spectral: [15360; 8], // f16 1.0 on all bands = white
        _pad: 0,
    }
}

#[test]
fn framebuffer_starts_black() {
    let fb = Framebuffer::new(64, 64);
    assert_eq!(fb.width, 64);
    assert_eq!(fb.height, 64);
    assert!(fb.pixels.iter().all(|p| *p == [0u8; 4]));
}

#[test]
fn single_splat_renders_nonblack_pixel() {
    let mut rasteriser = SoftwareRasteriser::new(64, 64);
    let splats = vec![make_test_splat([0.0, 0.0, -5.0])];

    let camera = RenderCamera {
        view: Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y),
        proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0),
    };

    let fb = rasteriser.render(&splats, &camera, &Illuminant::d65());

    // At least one pixel should be non-black
    let has_colour = fb.pixels.iter().any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0);
    assert!(has_colour, "Expected at least one non-black pixel");
}

#[test]
fn two_splats_at_different_positions_both_render() {
    let mut rasteriser = SoftwareRasteriser::new(128, 128);
    let splats = vec![
        make_test_splat([-2.0, 0.0, -5.0]),
        make_test_splat([2.0, 0.0, -5.0]),
    ];

    let camera = RenderCamera {
        view: Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y),
        proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0),
    };

    let fb = rasteriser.render(&splats, &camera, &Illuminant::d65());

    // Check left and right halves both have colour
    let left_has_colour = fb.pixels.iter().enumerate()
        .any(|(i, p)| (i % 128) < 64 && (p[0] > 0 || p[1] > 0 || p[2] > 0));
    let right_has_colour = fb.pixels.iter().enumerate()
        .any(|(i, p)| (i % 128) >= 64 && (p[0] > 0 || p[1] > 0 || p[2] > 0));

    assert!(left_has_colour, "Left splat should produce pixels");
    assert!(right_has_colour, "Right splat should produce pixels");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vox_render`
Expected: FAIL — modules not defined

- [ ] **Step 3: Implement RenderCamera**

`crates/vox_render/src/spectral.rs`:
```rust
use glam::Mat4;

/// Camera matrices for rendering.
#[derive(Debug, Clone)]
pub struct RenderCamera {
    pub view: Mat4,
    pub proj: Mat4,
}

impl RenderCamera {
    pub fn view_proj(&self) -> Mat4 {
        self.proj * self.view
    }
}
```

- [ ] **Step 4: Implement Framebuffer and SoftwareRasteriser**

`crates/vox_render/src/gpu/mod.rs`:
```rust
pub mod software_rasteriser;
```

`crates/vox_render/src/gpu/software_rasteriser.rs`:
```rust
use glam::{Vec3, Vec4};
use vox_core::spectral::{
    linear_to_srgb_gamma, spectral_to_xyz, xyz_to_srgb, Illuminant, SpectralBands,
};
use vox_core::types::GaussianSplat;

use crate::spectral::RenderCamera;

/// RGBA8 framebuffer.
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA pixels.
    pub pixels: Vec<[u8; 4]>,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![[0u8; 4]; (width * height) as usize],
        }
    }

    fn set_pixel(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        if x < self.width && y < self.height {
            self.pixels[(y * self.width + x) as usize] = rgba;
        }
    }

    fn blend_pixel(&mut self, x: u32, y: u32, r: f32, g: f32, b: f32, a: f32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = (y * self.width + x) as usize;
        let existing = self.pixels[idx];
        let er = existing[0] as f32 / 255.0;
        let eg = existing[1] as f32 / 255.0;
        let eb = existing[2] as f32 / 255.0;
        let ea = existing[3] as f32 / 255.0;

        // Front-to-back alpha compositing
        let out_a = a + ea * (1.0 - a);
        if out_a < 0.001 {
            return;
        }
        let out_r = (r * a + er * ea * (1.0 - a)) / out_a;
        let out_g = (g * a + eg * ea * (1.0 - a)) / out_a;
        let out_b = (b * a + eb * ea * (1.0 - a)) / out_a;

        self.pixels[idx] = [
            (out_r.clamp(0.0, 1.0) * 255.0) as u8,
            (out_g.clamp(0.0, 1.0) * 255.0) as u8,
            (out_b.clamp(0.0, 1.0) * 255.0) as u8,
            (out_a.clamp(0.0, 1.0) * 255.0) as u8,
        ];
    }
}

/// Projected 2D Gaussian for sorting and rasterisation.
struct ProjectedSplat {
    screen_x: f32,
    screen_y: f32,
    depth: f32,
    radius_px: f32,
    opacity: f32,
    spectral: SpectralBands,
}

pub struct SoftwareRasteriser {
    pub width: u32,
    pub height: u32,
}

impl SoftwareRasteriser {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn render(
        &mut self,
        splats: &[GaussianSplat],
        camera: &RenderCamera,
        illuminant: &Illuminant,
    ) -> Framebuffer {
        let mut fb = Framebuffer::new(self.width, self.height);
        let view_proj = camera.view_proj();
        let hw = self.width as f32 * 0.5;
        let hh = self.height as f32 * 0.5;

        // Project all splats
        let mut projected: Vec<ProjectedSplat> = splats
            .iter()
            .filter_map(|s| {
                let pos = Vec4::new(s.position[0], s.position[1], s.position[2], 1.0);
                let clip = view_proj * pos;

                // Clip behind camera
                if clip.w <= 0.0 {
                    return None;
                }

                let ndc_x = clip.x / clip.w;
                let ndc_y = clip.y / clip.w;
                let depth = clip.z / clip.w;

                // Clip outside frustum (with margin)
                if ndc_x < -1.5 || ndc_x > 1.5 || ndc_y < -1.5 || ndc_y > 1.5 {
                    return None;
                }

                let screen_x = (ndc_x * 0.5 + 0.5) * self.width as f32;
                let screen_y = ((1.0 - (ndc_y * 0.5 + 0.5))) * self.height as f32;

                // Approximate screen-space radius from scale and depth
                let avg_scale = (s.scale[0] + s.scale[1] + s.scale[2]) / 3.0;
                let radius_px = (avg_scale * hw / clip.w).max(1.0);

                let spectral = SpectralBands(std::array::from_fn(|i| {
                    half::f16::from_bits(s.spectral[i]).to_f32()
                }));

                Some(ProjectedSplat {
                    screen_x,
                    screen_y,
                    depth,
                    radius_px,
                    opacity: s.opacity as f32 / 255.0,
                    spectral,
                })
            })
            .collect();

        // Sort back-to-front
        projected.sort_by(|a, b| b.depth.partial_cmp(&a.depth).unwrap_or(std::cmp::Ordering::Equal));

        // Rasterise each splat as a 2D Gaussian
        for splat in &projected {
            let xyz = spectral_to_xyz(&splat.spectral, illuminant);
            let rgb = xyz_to_srgb(xyz);
            let r = linear_to_srgb_gamma(rgb[0]);
            let g = linear_to_srgb_gamma(rgb[1]);
            let b = linear_to_srgb_gamma(rgb[2]);

            let radius = splat.radius_px.ceil() as i32;
            let cx = splat.screen_x as i32;
            let cy = splat.screen_y as i32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let px = cx + dx;
                    let py = cy + dy;

                    if px < 0 || py < 0 || px >= self.width as i32 || py >= self.height as i32 {
                        continue;
                    }

                    let dist_sq = (dx * dx + dy * dy) as f32;
                    let sigma = splat.radius_px * 0.5;
                    let gauss = (-dist_sq / (2.0 * sigma * sigma)).exp();
                    let alpha = splat.opacity * gauss;

                    if alpha > 0.003 {
                        fb.blend_pixel(px as u32, py as u32, r, g, b, alpha);
                    }
                }
            }
        }

        fb
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p vox_render`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add crates/vox_render/
git commit -m "feat(vox_render): add software spectral rasteriser with depth sort and Gaussian splatting"
```

---

### Task 5: wgpu Window and Render Loop (`vox_app`)

**Files:**
- Create: `crates/vox_render/src/gpu/wgpu_backend.rs`
- Modify: `crates/vox_render/src/gpu/mod.rs`
- Modify: `crates/vox_app/src/main.rs`
- Create: `crates/vox_app/src/demo_asset.rs`

- [ ] **Step 1: Add wgpu backend that blits a framebuffer to a window**

`crates/vox_render/src/gpu/wgpu_backend.rs`:
```rust
use wgpu::{
    Device, Queue, Surface, SurfaceConfiguration, TextureFormat, TextureUsages,
    Instance, RequestAdapterOptions, DeviceDescriptor, Features, Limits,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct WgpuBackend {
    pub device: Device,
    pub queue: Queue,
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
}

impl WgpuBackend {
    pub fn new(
        window: std::sync::Arc<winit::window::Window>,
        width: u32,
        height: u32,
    ) -> Self {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("Failed to find GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("ochroma"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                ..Default::default()
            },
            None,
        ))
        .expect("Failed to create device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Self {
            device,
            queue,
            surface,
            config,
        }
    }

    /// Blit RGBA8 pixel data to the surface.
    pub fn present_framebuffer(&self, pixels: &[[u8; 4]], width: u32, height: u32) {
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output.texture.create_view(&Default::default());

        // Create a texture from our pixel data
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("framebuffer"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        // For Phase 0 we do a simple copy. A proper blit shader comes later.
        // Use a compute/render pass to copy texture to surface.
        // For now, we just write directly to the surface texture.
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &output.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width: width.min(self.config.width),
                height: height.min(self.config.height),
                depth_or_array_layers: 1,
            },
        );

        output.present();
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}
```

Update `crates/vox_render/src/gpu/mod.rs`:
```rust
pub mod software_rasteriser;
pub mod wgpu_backend;
```

- [ ] **Step 2: Create demo asset generator**

`crates/vox_app/src/demo_asset.rs`:
```rust
use half::f16;
use uuid::Uuid;
use vox_core::types::GaussianSplat;
use vox_data::vxm::{MaterialType, VxmFile, VxmHeader};

/// Generate a synthetic test building — a box of Gaussians with spectral properties.
pub fn generate_demo_asset() -> VxmFile {
    let uuid = Uuid::new_v4();
    let mut splats = Vec::new();

    // Brick wall (red SPD)
    let brick_spd = [
        f16::from_f32(0.1).to_bits(),  // 380nm
        f16::from_f32(0.1).to_bits(),  // 420nm
        f16::from_f32(0.15).to_bits(), // 460nm
        f16::from_f32(0.2).to_bits(),  // 500nm
        f16::from_f32(0.3).to_bits(),  // 540nm
        f16::from_f32(0.6).to_bits(),  // 580nm
        f16::from_f32(0.7).to_bits(),  // 620nm — red peak
        f16::from_f32(0.65).to_bits(), // 660nm
    ];

    // Front wall
    for ix in 0..20 {
        for iy in 0..15 {
            splats.push(GaussianSplat {
                position: [ix as f32 * 0.5, iy as f32 * 0.5, 0.0],
                scale: [0.25, 0.25, 0.1],
                rotation: [0, 0, 0, 32767],
                opacity: 240,
                _pad: 0,
                spectral: brick_spd,
            });
        }
    }

    // Side wall
    for iz in 0..12 {
        for iy in 0..15 {
            splats.push(GaussianSplat {
                position: [0.0, iy as f32 * 0.5, -(iz as f32 * 0.5)],
                scale: [0.1, 0.25, 0.25],
                rotation: [0, 0, 0, 32767],
                opacity: 240,
                _pad: 0,
                spectral: brick_spd,
            });
        }
    }

    // Roof (slate grey SPD)
    let slate_spd = [
        f16::from_f32(0.15).to_bits(),
        f16::from_f32(0.15).to_bits(),
        f16::from_f32(0.18).to_bits(),
        f16::from_f32(0.2).to_bits(),
        f16::from_f32(0.2).to_bits(),
        f16::from_f32(0.2).to_bits(),
        f16::from_f32(0.2).to_bits(),
        f16::from_f32(0.18).to_bits(),
    ];

    for ix in 0..20 {
        for iz in 0..12 {
            splats.push(GaussianSplat {
                position: [ix as f32 * 0.5, 7.5, -(iz as f32 * 0.5)],
                scale: [0.25, 0.1, 0.25],
                rotation: [0, 0, 0, 32767],
                opacity: 250,
                _pad: 0,
                spectral: slate_spd,
            });
        }
    }

    VxmFile {
        header: VxmHeader::new(uuid, splats.len() as u32, MaterialType::Concrete),
        splats,
    }
}
```

- [ ] **Step 3: Implement main render loop with two instances**

`crates/vox_app/src/main.rs`:
```rust
mod demo_asset;

use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Vec3};
use vox_core::spectral::Illuminant;
use vox_core::types::{GaussianSplat, SplatInstance};
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::gpu::wgpu_backend::WgpuBackend;
use vox_render::spectral::RenderCamera;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

struct App {
    window: Option<Arc<Window>>,
    backend: Option<WgpuBackend>,
    rasteriser: SoftwareRasteriser,
    world_splats: Vec<GaussianSplat>,
    camera_angle: f32,
    last_frame: Instant,
    frame_count: u64,
    fps_timer: Instant,
}

impl App {
    fn new() -> Self {
        // Generate demo asset
        let asset = demo_asset::generate_demo_asset();
        println!("Demo asset: {} splats", asset.splats.len());

        // Two instances at different positions (the Phase 0 plop demo)
        let instances = [
            SplatInstance {
                asset_uuid: asset.header.uuid(),
                position: Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                instance_id: 0,
            },
            SplatInstance {
                asset_uuid: asset.header.uuid(),
                position: Vec3::new(20.0, 0.0, 0.0),
                rotation: glam::Quat::IDENTITY,
                instance_id: 1,
            },
        ];

        // Transform splats by instance positions into world space
        let mut world_splats = Vec::new();
        for inst in &instances {
            for splat in &asset.splats {
                let mut ws = *splat;
                ws.position[0] += inst.position.x;
                ws.position[1] += inst.position.y;
                ws.position[2] += inst.position.z;
                world_splats.push(ws);
            }
        }
        println!("Total world splats: {}", world_splats.len());

        Self {
            window: None,
            backend: None,
            rasteriser: SoftwareRasteriser::new(WIDTH, HEIGHT),
            world_splats,
            camera_angle: 0.0,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Ochroma — Phase 0")
            .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let backend = WgpuBackend::new(window.clone(), WIDTH, HEIGHT);

        self.window = Some(window);
        self.backend = Some(backend);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(backend) = &mut self.backend {
                    backend.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                // Orbit camera
                self.camera_angle += dt * 0.3;
                let radius = 30.0;
                let eye = Vec3::new(
                    self.camera_angle.cos() * radius + 10.0,
                    8.0,
                    self.camera_angle.sin() * radius - 3.0,
                );
                let target = Vec3::new(10.0, 3.0, -3.0);

                let camera = RenderCamera {
                    view: Mat4::look_at_rh(eye, target, Vec3::Y),
                    proj: Mat4::perspective_rh(
                        std::f32::consts::FRAC_PI_4,
                        WIDTH as f32 / HEIGHT as f32,
                        0.1,
                        200.0,
                    ),
                };

                let fb = self.rasteriser.render(
                    &self.world_splats,
                    &camera,
                    &Illuminant::d65(),
                );

                if let Some(backend) = &self.backend {
                    backend.present_framebuffer(&fb.pixels, fb.width, fb.height);
                }

                // FPS counter
                self.frame_count += 1;
                if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
                    let fps = self.frame_count as f32 / self.fps_timer.elapsed().as_secs_f32();
                    println!("FPS: {:.1} ({} splats)", fps, self.world_splats.len());
                    self.frame_count = 0;
                    self.fps_timer = Instant::now();
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo build`
Expected: compiles with no errors

- [ ] **Step 5: Run the demo**

Run: `cargo run -p vox_app`
Expected: window opens, two brick buildings visible, camera orbits, FPS printed to console

- [ ] **Step 6: Commit**

```bash
git add crates/
git commit -m "feat: Phase 0 MVP — wgpu window, software spectral rasteriser, two-instance plop demo"
```

---

### Task 6: .vxm File I/O Integration Test

**Files:**
- Create: `crates/vox_data/tests/file_io_test.rs`

- [ ] **Step 1: Write integration test — write to disk, read back, render**

`crates/vox_data/tests/file_io_test.rs`:
```rust
use std::io::Cursor;
use uuid::Uuid;
use half::f16;
use vox_core::types::GaussianSplat;
use vox_data::vxm::{VxmFile, VxmHeader, MaterialType};

#[test]
fn write_to_file_read_back_identical() {
    let uuid = Uuid::new_v4();
    let splats: Vec<GaussianSplat> = (0..500)
        .map(|i| GaussianSplat {
            position: [i as f32 * 0.1, (i as f32 * 0.7).sin(), 0.0],
            scale: [0.05, 0.05, 0.05],
            rotation: [0, 0, 0, 32767],
            opacity: 200,
            _pad: 0,
            spectral: [f16::from_f32(0.5).to_bits(); 8],
        })
        .collect();

    let original = VxmFile {
        header: VxmHeader::new(uuid, splats.len() as u32, MaterialType::Concrete),
        splats: splats.clone(),
    };

    // Write to buffer
    let mut buf = Vec::new();
    original.write(&mut buf).unwrap();

    // Verify compression actually reduced size
    let uncompressed_size = 64 + 52 * 500;
    assert!(
        buf.len() < uncompressed_size,
        "Expected compression: {} < {}",
        buf.len(),
        uncompressed_size
    );

    // Read back
    let loaded = VxmFile::read(Cursor::new(&buf)).unwrap();

    // Verify header
    assert_eq!(loaded.header.uuid(), uuid);
    assert_eq!(loaded.header.splat_count, 500);
    assert_eq!(loaded.header.material(), MaterialType::Concrete);

    // Verify every splat matches
    for (i, (orig, load)) in splats.iter().zip(loaded.splats.iter()).enumerate() {
        assert_eq!(orig.position, load.position, "splat {} position mismatch", i);
        assert_eq!(orig.scale, load.scale, "splat {} scale mismatch", i);
        assert_eq!(orig.rotation, load.rotation, "splat {} rotation mismatch", i);
        assert_eq!(orig.opacity, load.opacity, "splat {} opacity mismatch", i);
        assert_eq!(orig.spectral, load.spectral, "splat {} spectral mismatch", i);
    }
}
```

- [ ] **Step 2: Run test**

Run: `cargo test -p vox_data file_io`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/vox_data/tests/
git commit -m "test(vox_data): add .vxm file I/O integration test with compression verification"
```

---

## Phase 1 — Foundation

### Task 7: Bevy ECS Integration

**Files:**
- Modify: `Cargo.toml` (workspace deps)
- Modify: `crates/vox_app/Cargo.toml`
- Create: `crates/vox_core/src/ecs.rs`
- Modify: `crates/vox_core/src/lib.rs`
- Create: `crates/vox_app/src/ecs_app.rs`
- Test: `crates/vox_core/tests/ecs_test.rs`

- [ ] **Step 1: Add bevy dependency to workspace**

Add to workspace `Cargo.toml` `[workspace.dependencies]`:
```toml
bevy_ecs = "0.16"
bevy_app = "0.16"
```

Add to `crates/vox_core/Cargo.toml`:
```toml
bevy_ecs = { workspace = true }
```

Add to `crates/vox_app/Cargo.toml`:
```toml
bevy_ecs = { workspace = true }
bevy_app = { workspace = true }
```

- [ ] **Step 2: Write failing test for ECS components**

`crates/vox_core/tests/ecs_test.rs`:
```rust
use bevy_ecs::prelude::*;
use glam::{Vec3, Quat};
use uuid::Uuid;
use vox_core::ecs::{SplatInstanceComponent, SplatAssetComponent, LodLevel};

#[test]
fn can_spawn_splat_instance() {
    let mut world = World::new();
    let entity = world.spawn(SplatInstanceComponent {
        asset_uuid: Uuid::new_v4(),
        position: Vec3::new(1.0, 2.0, 3.0),
        rotation: Quat::IDENTITY,
        scale: 1.0,
        instance_id: 42,
        lod: LodLevel::Full,
    }).id();

    let inst = world.get::<SplatInstanceComponent>(entity).unwrap();
    assert_eq!(inst.instance_id, 42);
    assert_eq!(inst.position, Vec3::new(1.0, 2.0, 3.0));
}

#[test]
fn can_query_instances() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn(SplatInstanceComponent {
            asset_uuid: Uuid::new_v4(),
            position: Vec3::new(i as f32, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            instance_id: i,
            lod: LodLevel::Full,
        });
    }

    let mut query = world.query::<&SplatInstanceComponent>();
    let count = query.iter(&world).count();
    assert_eq!(count, 100);
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p vox_core ecs`
Expected: FAIL

- [ ] **Step 4: Implement ECS components**

`crates/vox_core/src/ecs.rs`:
```rust
use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use uuid::Uuid;

/// LOD level for rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodLevel {
    /// Full detail (< 200m)
    Full,
    /// Reduced density (> 200m)
    Reduced,
}

/// A placed instance of a Gaussian splat asset.
#[derive(Component, Debug, Clone)]
pub struct SplatInstanceComponent {
    pub asset_uuid: Uuid,
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: f32,
    pub instance_id: u32,
    pub lod: LodLevel,
}

/// A loaded Gaussian splat asset in VRAM/CPU memory.
#[derive(Component, Debug)]
pub struct SplatAssetComponent {
    pub uuid: Uuid,
    pub splat_count: u32,
    /// CPU-side splat data (for software rasteriser; GPU buffer added later).
    pub splats: Vec<crate::types::GaussianSplat>,
}
```

Update `crates/vox_core/src/lib.rs`:
```rust
pub mod types;
pub mod spectral;
pub mod ecs;
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p vox_core`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add crates/vox_core/ Cargo.toml
git commit -m "feat(vox_core): add Bevy ECS components for splat instances and assets"
```

---

### Task 8: SVO and Spatial Hash

**Files:**
- Create: `crates/vox_core/src/svo.rs`
- Modify: `crates/vox_core/src/lib.rs`
- Modify: `Cargo.toml` (workspace deps)
- Modify: `crates/vox_core/Cargo.toml`
- Test: `crates/vox_core/tests/svo_test.rs`

- [ ] **Step 1: Add dashmap dependency**

Add to workspace `Cargo.toml`:
```toml
dashmap = "6"
```

Add to `crates/vox_core/Cargo.toml`:
```toml
dashmap = { workspace = true }
```

- [ ] **Step 2: Write failing test**

`crates/vox_core/tests/svo_test.rs`:
```rust
use vox_core::svo::SpatialHash;
use glam::Vec3;

#[test]
fn insert_and_query_by_voxel() {
    let mut sh = SpatialHash::new(8.0); // 8m voxels
    sh.insert(1, Vec3::new(1.0, 2.0, 3.0));
    sh.insert(2, Vec3::new(1.5, 2.5, 3.5)); // same voxel
    sh.insert(3, Vec3::new(100.0, 0.0, 0.0)); // different voxel

    let nearby = sh.query_voxel(Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(nearby.len(), 2);
    assert!(nearby.contains(&1));
    assert!(nearby.contains(&2));
}

#[test]
fn query_radius() {
    let mut sh = SpatialHash::new(8.0);
    sh.insert(1, Vec3::new(0.0, 0.0, 0.0));
    sh.insert(2, Vec3::new(5.0, 0.0, 0.0));
    sh.insert(3, Vec3::new(50.0, 0.0, 0.0));

    let nearby = sh.query_radius(Vec3::ZERO, 10.0);
    assert!(nearby.contains(&1));
    assert!(nearby.contains(&2));
    assert!(!nearby.contains(&3));
}

#[test]
fn remove_instance() {
    let mut sh = SpatialHash::new(8.0);
    sh.insert(1, Vec3::new(0.0, 0.0, 0.0));
    sh.remove(1, Vec3::new(0.0, 0.0, 0.0));

    let nearby = sh.query_voxel(Vec3::ZERO);
    assert!(nearby.is_empty());
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p vox_core svo`
Expected: FAIL

- [ ] **Step 4: Implement spatial hash**

`crates/vox_core/src/svo.rs`:
```rust
use dashmap::DashMap;
use glam::Vec3;

/// Voxel key: (x, y, z) grid cell indices.
type VoxelKey = (i32, i32, i32);

/// Spatial hash for fast instance lookup within a 1km tile.
/// Uses a DashMap for concurrent read/write access.
pub struct SpatialHash {
    cell_size: f32,
    map: DashMap<VoxelKey, Vec<u32>>,
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            map: DashMap::new(),
        }
    }

    fn key(&self, pos: Vec3) -> VoxelKey {
        (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
            (pos.z / self.cell_size).floor() as i32,
        )
    }

    pub fn insert(&mut self, instance_id: u32, position: Vec3) {
        let key = self.key(position);
        self.map.entry(key).or_default().push(instance_id);
    }

    pub fn remove(&mut self, instance_id: u32, position: Vec3) {
        let key = self.key(position);
        if let Some(mut ids) = self.map.get_mut(&key) {
            ids.retain(|id| *id != instance_id);
        }
    }

    /// Get all instance IDs in the same voxel cell as `position`.
    pub fn query_voxel(&self, position: Vec3) -> Vec<u32> {
        let key = self.key(position);
        self.map
            .get(&key)
            .map(|ids| ids.clone())
            .unwrap_or_default()
    }

    /// Get all instance IDs within `radius` of `position`.
    /// Checks all voxel cells that overlap the query sphere.
    pub fn query_radius(&self, position: Vec3, radius: f32) -> Vec<u32> {
        let cells = (radius / self.cell_size).ceil() as i32 + 1;
        let centre_key = self.key(position);
        let mut result = Vec::new();

        for dx in -cells..=cells {
            for dy in -cells..=cells {
                for dz in -cells..=cells {
                    let key = (
                        centre_key.0 + dx,
                        centre_key.1 + dy,
                        centre_key.2 + dz,
                    );
                    if let Some(ids) = self.map.get(&key) {
                        result.extend(ids.iter());
                    }
                }
            }
        }

        result
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }
}
```

Update `crates/vox_core/src/lib.rs`:
```rust
pub mod types;
pub mod spectral;
pub mod ecs;
pub mod svo;
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p vox_core svo`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add crates/vox_core/ Cargo.toml
git commit -m "feat(vox_core): add DashMap spatial hash for voxel-based instance lookup"
```

---

### Task 9: Spectral Materials Library

**Files:**
- Create: `crates/vox_data/src/materials.rs`
- Modify: `crates/vox_data/src/lib.rs`
- Test: `crates/vox_data/tests/materials_test.rs`

- [ ] **Step 1: Write failing test**

`crates/vox_data/tests/materials_test.rs`:
```rust
use vox_data::materials::{MaterialLibrary, SpectralMaterial};
use vox_core::spectral::{Illuminant, spectral_to_xyz, xyz_to_srgb};

#[test]
fn library_has_base_materials() {
    let lib = MaterialLibrary::default();
    assert!(lib.get("brick_red").is_some());
    assert!(lib.get("concrete_raw").is_some());
    assert!(lib.get("glass_clear").is_some());
    assert!(lib.get("vegetation_leaf").is_some());
    assert!(lib.get("metal_steel").is_some());
    assert!(lib.get("asphalt_dry").is_some());
    assert!(lib.get("slate_grey").is_some());
    assert!(lib.get("water_still").is_some());
    assert!(lib.get("soil_dry").is_some());
    assert!(lib.get("wood_painted_green").is_some());
}

#[test]
fn brick_red_looks_reddish_under_d65() {
    let lib = MaterialLibrary::default();
    let brick = lib.get("brick_red").unwrap();
    let xyz = spectral_to_xyz(&brick.spd, &Illuminant::d65());
    let rgb = xyz_to_srgb(xyz);
    // Red channel should dominate
    assert!(rgb[0] > rgb[1], "brick R={:.3} should be > G={:.3}", rgb[0], rgb[1]);
    assert!(rgb[0] > rgb[2], "brick R={:.3} should be > B={:.3}", rgb[0], rgb[2]);
}

#[test]
fn vegetation_looks_greenish_under_d65() {
    let lib = MaterialLibrary::default();
    let veg = lib.get("vegetation_leaf").unwrap();
    let xyz = spectral_to_xyz(&veg.spd, &Illuminant::d65());
    let rgb = xyz_to_srgb(xyz);
    assert!(rgb[1] > rgb[0], "vegetation G={:.3} should be > R={:.3}", rgb[1], rgb[0]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vox_data materials`
Expected: FAIL

- [ ] **Step 3: Implement materials library**

`crates/vox_data/src/materials.rs`:
```rust
use std::collections::HashMap;
use vox_core::spectral::SpectralBands;

/// A spectral material with reflectance curve.
#[derive(Debug, Clone)]
pub struct SpectralMaterial {
    pub tag: String,
    pub description: String,
    /// Reflectance SPD (8 bands, 380–660nm).
    pub spd: SpectralBands,
    /// Worn/aged variant SPD.
    pub spd_worn: SpectralBands,
}

/// Library of physically derived spectral materials.
pub struct MaterialLibrary {
    materials: HashMap<String, SpectralMaterial>,
}

impl MaterialLibrary {
    pub fn get(&self, tag: &str) -> Option<&SpectralMaterial> {
        self.materials.get(tag)
    }

    pub fn all(&self) -> impl Iterator<Item = &SpectralMaterial> {
        self.materials.values()
    }
}

impl Default for MaterialLibrary {
    fn default() -> Self {
        let mut materials = HashMap::new();

        let mut add = |tag: &str, desc: &str, spd: [f32; 8], worn: [f32; 8]| {
            materials.insert(
                tag.to_string(),
                SpectralMaterial {
                    tag: tag.to_string(),
                    description: desc.to_string(),
                    spd: SpectralBands(spd),
                    spd_worn: SpectralBands(worn),
                },
            );
        };

        //                         380   420   460   500   540   580   620   660
        add("concrete_raw",    "Unpainted concrete",
            [0.28, 0.30, 0.31, 0.32, 0.32, 0.31, 0.30, 0.29],
            [0.18, 0.19, 0.20, 0.20, 0.20, 0.19, 0.18, 0.17]);

        add("brick_red",       "Red clay brick",
            [0.08, 0.08, 0.10, 0.15, 0.25, 0.55, 0.65, 0.60],
            [0.06, 0.06, 0.08, 0.12, 0.18, 0.38, 0.45, 0.40]);

        add("glass_clear",     "Clear float glass",
            [0.85, 0.88, 0.90, 0.91, 0.91, 0.90, 0.89, 0.87],
            [0.75, 0.78, 0.80, 0.81, 0.81, 0.80, 0.79, 0.77]);

        add("vegetation_leaf", "Broadleaf foliage",
            [0.03, 0.04, 0.06, 0.10, 0.45, 0.30, 0.08, 0.04],
            [0.05, 0.06, 0.08, 0.12, 0.25, 0.18, 0.08, 0.05]);

        add("metal_steel",     "Bare steel",
            [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.63, 0.63],
            [0.25, 0.22, 0.20, 0.22, 0.30, 0.45, 0.50, 0.48]);

        add("metal_oxidized",  "Rusted steel",
            [0.06, 0.06, 0.08, 0.12, 0.20, 0.40, 0.50, 0.45],
            [0.04, 0.04, 0.06, 0.10, 0.15, 0.30, 0.38, 0.34]);

        add("asphalt_dry",     "Dry road surface",
            [0.04, 0.04, 0.05, 0.05, 0.05, 0.05, 0.06, 0.06],
            [0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]);

        add("slate_grey",      "Slate roof tile",
            [0.12, 0.13, 0.15, 0.17, 0.18, 0.18, 0.17, 0.16],
            [0.08, 0.09, 0.10, 0.11, 0.12, 0.12, 0.11, 0.10]);

        add("water_still",     "Still water surface",
            [0.02, 0.03, 0.04, 0.05, 0.04, 0.03, 0.02, 0.02],
            [0.02, 0.03, 0.04, 0.05, 0.04, 0.03, 0.02, 0.02]);

        add("soil_dry",        "Bare earth",
            [0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.32, 0.30],
            [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.24, 0.22]);

        add("wood_painted_green", "Green painted wood",
            [0.04, 0.05, 0.08, 0.12, 0.30, 0.18, 0.06, 0.04],
            [0.04, 0.05, 0.07, 0.10, 0.18, 0.12, 0.05, 0.04]);

        Self { materials }
    }
}
```

Update `crates/vox_data/src/lib.rs`:
```rust
pub mod vxm;
pub mod materials;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p vox_data materials`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add crates/vox_data/
git commit -m "feat(vox_data): add spectral materials library with 11 physically derived materials"
```

---

### Task 10: Proc-GS Rule System

**Files:**
- Create: `crates/vox_data/src/proc_gs.rs`
- Create: `crates/vox_data/src/rule_parser.rs`
- Modify: `crates/vox_data/src/lib.rs`
- Modify: `crates/vox_data/Cargo.toml`
- Test: `crates/vox_data/tests/proc_gs_test.rs`

- [ ] **Step 1: Add toml dependency**

Add to workspace `Cargo.toml`:
```toml
toml = "0.8"
```

Add to `crates/vox_data/Cargo.toml`:
```toml
toml = { workspace = true }
glam = { workspace = true }
rand = "0.9"
```

- [ ] **Step 2: Write failing test**

`crates/vox_data/tests/proc_gs_test.rs`:
```rust
use vox_data::proc_gs::{SplatRule, GeometryStrategy, emit_splats};

const VICTORIAN_RULE: &str = r#"
[rule]
asset_type = "House"
style = "victorian_terraced"

[geometry]
strategy = "structured_placement"
floor_count_min = 2
floor_count_max = 4
floor_height_min = 3.2
floor_height_max = 3.8
base_width_min = 4.5
base_width_max = 6.0
depth = 12.0

[[materials]]
tag = "brick_facade"
spd = "brick_red"
density = 800.0
scale_min = 0.04
scale_max = 0.08
entity_id_zone = 1

[[materials]]
tag = "slate_roof"
spd = "slate_grey"
density = 600.0
scale_min = 0.05
scale_max = 0.10
entity_id_zone = 3

[variation]
facade_color_shift = 0.15
wear_level_min = 0.0
wear_level_max = 0.4
"#;

#[test]
fn parse_victorian_rule() {
    let rule: SplatRule = toml::from_str(VICTORIAN_RULE).unwrap();
    assert_eq!(rule.rule.asset_type, "House");
    assert_eq!(rule.rule.style, "victorian_terraced");
    assert_eq!(rule.geometry.strategy, GeometryStrategy::StructuredPlacement);
    assert_eq!(rule.materials.len(), 2);
    assert_eq!(rule.materials[0].tag, "brick_facade");
}

#[test]
fn emit_produces_deterministic_output() {
    let rule: SplatRule = toml::from_str(VICTORIAN_RULE).unwrap();
    let splats_a = emit_splats(&rule, 42);
    let splats_b = emit_splats(&rule, 42);

    assert_eq!(splats_a.len(), splats_b.len());
    for (a, b) in splats_a.iter().zip(splats_b.iter()) {
        assert_eq!(a.position, b.position);
    }
}

#[test]
fn different_seeds_produce_different_output() {
    let rule: SplatRule = toml::from_str(VICTORIAN_RULE).unwrap();
    let splats_a = emit_splats(&rule, 42);
    let splats_b = emit_splats(&rule, 99);

    // Same count (same rule) but different positions
    assert_eq!(splats_a.len(), splats_b.len());
    let different = splats_a.iter().zip(splats_b.iter())
        .any(|(a, b)| a.position != b.position);
    assert!(different, "Different seeds should produce different splats");
}

#[test]
fn emit_produces_nonzero_splats() {
    let rule: SplatRule = toml::from_str(VICTORIAN_RULE).unwrap();
    let splats = emit_splats(&rule, 1);
    assert!(!splats.is_empty(), "Should produce splats");
    assert!(splats.len() > 100, "A building should have many splats, got {}", splats.len());
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p vox_data proc_gs`
Expected: FAIL

- [ ] **Step 4: Implement Proc-GS rule types and emitter**

`crates/vox_data/src/proc_gs.rs`:
```rust
use half::f16;
use rand::prelude::*;
use rand::SeedableRng;
use serde::Deserialize;
use vox_core::types::GaussianSplat;

use crate::materials::MaterialLibrary;

#[derive(Debug, Deserialize)]
pub struct SplatRule {
    pub rule: RuleHeader,
    pub geometry: GeometryConfig,
    pub materials: Vec<MaterialZoneConfig>,
    pub variation: VariationConfig,
}

#[derive(Debug, Deserialize)]
pub struct RuleHeader {
    pub asset_type: String,
    pub style: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GeometryStrategy {
    StructuredPlacement,
    GrowthAlgorithm,
    ComponentAssembly,
    SurfaceScattering,
}

#[derive(Debug, Deserialize)]
pub struct GeometryConfig {
    pub strategy: GeometryStrategy,
    #[serde(default = "default_floor_count_min")]
    pub floor_count_min: u32,
    #[serde(default = "default_floor_count_max")]
    pub floor_count_max: u32,
    #[serde(default = "default_floor_height_min")]
    pub floor_height_min: f32,
    #[serde(default = "default_floor_height_max")]
    pub floor_height_max: f32,
    #[serde(default = "default_base_width_min")]
    pub base_width_min: f32,
    #[serde(default = "default_base_width_max")]
    pub base_width_max: f32,
    #[serde(default = "default_depth")]
    pub depth: f32,
}

fn default_floor_count_min() -> u32 { 1 }
fn default_floor_count_max() -> u32 { 1 }
fn default_floor_height_min() -> f32 { 3.0 }
fn default_floor_height_max() -> f32 { 3.0 }
fn default_base_width_min() -> f32 { 5.0 }
fn default_base_width_max() -> f32 { 5.0 }
fn default_depth() -> f32 { 10.0 }

#[derive(Debug, Deserialize)]
pub struct MaterialZoneConfig {
    pub tag: String,
    pub spd: String,
    pub density: f32,
    pub scale_min: f32,
    pub scale_max: f32,
    pub entity_id_zone: u16,
}

#[derive(Debug, Deserialize)]
pub struct VariationConfig {
    #[serde(default)]
    pub facade_color_shift: f32,
    #[serde(default)]
    pub wear_level_min: f32,
    #[serde(default)]
    pub wear_level_max: f32,
}

/// Emit Gaussians from a rule and seed. Deterministic: same seed = same output.
pub fn emit_splats(rule: &SplatRule, seed: u64) -> Vec<GaussianSplat> {
    let mut rng = StdRng::seed_from_u64(seed);
    let lib = MaterialLibrary::default();

    let floor_count = rng.random_range(rule.geometry.floor_count_min..=rule.geometry.floor_count_max);
    let floor_height = rng.random_range(rule.geometry.floor_height_min..=rule.geometry.floor_height_max);
    let base_width = rng.random_range(rule.geometry.base_width_min..=rule.geometry.base_width_max);
    let depth = rule.geometry.depth;
    let wear = rng.random_range(rule.variation.wear_level_min..=rule.variation.wear_level_max);

    let mut splats = Vec::new();

    for mat_config in &rule.materials {
        let material = match lib.get(&mat_config.spd) {
            Some(m) => m,
            None => continue,
        };

        // Interpolate SPD based on wear
        let spd: [u16; 8] = std::array::from_fn(|i| {
            let fresh = material.spd.0[i];
            let worn = material.spd_worn.0[i];
            let v = fresh + (worn - fresh) * wear;
            // Apply facade colour shift
            let shift = (rng.random::<f32>() - 0.5) * rule.variation.facade_color_shift;
            f16::from_f32((v + shift).max(0.0)).to_bits()
        });

        let scale_range = mat_config.scale_min..=mat_config.scale_max;

        match mat_config.tag.as_str() {
            tag if tag.contains("roof") || tag.contains("slate") => {
                // Roof: scatter on top
                emit_surface(
                    &mut splats, &mut rng,
                    0.0, base_width, floor_count as f32 * floor_height, floor_count as f32 * floor_height + 0.5,
                    0.0, depth,
                    mat_config.density, scale_range, spd, mat_config.entity_id_zone,
                );
            }
            _ => {
                // Walls: front, back, sides
                // Front wall (z = 0)
                emit_wall_xz(
                    &mut splats, &mut rng,
                    0.0, base_width, 0.0, floor_count as f32 * floor_height,
                    0.0,
                    mat_config.density, scale_range.clone(), spd, mat_config.entity_id_zone,
                );
                // Back wall (z = -depth)
                emit_wall_xz(
                    &mut splats, &mut rng,
                    0.0, base_width, 0.0, floor_count as f32 * floor_height,
                    -depth,
                    mat_config.density, scale_range.clone(), spd, mat_config.entity_id_zone,
                );
                // Left wall (x = 0)
                emit_wall_yz(
                    &mut splats, &mut rng,
                    0.0, -depth, 0.0, floor_count as f32 * floor_height,
                    0.0,
                    mat_config.density, scale_range.clone(), spd, mat_config.entity_id_zone,
                );
                // Right wall (x = base_width)
                emit_wall_yz(
                    &mut splats, &mut rng,
                    0.0, -depth, 0.0, floor_count as f32 * floor_height,
                    base_width,
                    mat_config.density, scale_range, spd, mat_config.entity_id_zone,
                );
            }
        }
    }

    splats
}

fn emit_surface(
    splats: &mut Vec<GaussianSplat>,
    rng: &mut StdRng,
    x_min: f32, x_max: f32, y_min: f32, y_max: f32, z_min: f32, z_max: f32,
    density: f32, scale_range: std::ops::RangeInclusive<f32>,
    spd: [u16; 8], entity_id_zone: u16,
) {
    let area = (x_max - x_min) * (z_max - z_min);
    let count = (area * density).round() as usize;
    for _ in 0..count {
        let x = rng.random_range(x_min..=x_max);
        let y = rng.random_range(y_min..=y_max);
        let z = rng.random_range(z_min..=z_max);
        let s = rng.random_range(scale_range.clone());
        splats.push(GaussianSplat {
            position: [x, y, z],
            scale: [s, s * 0.3, s],
            rotation: [0, 0, 0, 32767],
            opacity: 240,
            _pad: 0,
            spectral: spd,
        });
    }
}

fn emit_wall_xz(
    splats: &mut Vec<GaussianSplat>,
    rng: &mut StdRng,
    x_min: f32, x_max: f32, y_min: f32, y_max: f32,
    z: f32,
    density: f32, scale_range: std::ops::RangeInclusive<f32>,
    spd: [u16; 8], entity_id_zone: u16,
) {
    let area = (x_max - x_min) * (y_max - y_min);
    let count = (area * density).round() as usize;
    for _ in 0..count {
        let x = rng.random_range(x_min..=x_max);
        let y = rng.random_range(y_min..=y_max);
        let s = rng.random_range(scale_range.clone());
        splats.push(GaussianSplat {
            position: [x, y, z],
            scale: [s, s, s * 0.3],
            rotation: [0, 0, 0, 32767],
            opacity: 240,
            _pad: 0,
            spectral: spd,
        });
    }
}

fn emit_wall_yz(
    splats: &mut Vec<GaussianSplat>,
    rng: &mut StdRng,
    z_min: f32, z_max: f32, y_min: f32, y_max: f32,
    x: f32,
    density: f32, scale_range: std::ops::RangeInclusive<f32>,
    spd: [u16; 8], entity_id_zone: u16,
) {
    let area = (z_max - z_min).abs() * (y_max - y_min);
    let count = (area * density).round() as usize;
    for _ in 0..count {
        let z = rng.random_range(z_min..=z_max);
        let y = rng.random_range(y_min..=y_max);
        let s = rng.random_range(scale_range.clone());
        splats.push(GaussianSplat {
            position: [x, y, z],
            scale: [s * 0.3, s, s],
            rotation: [0, 0, 0, 32767],
            opacity: 240,
            _pad: 0,
            spectral: spd,
        });
    }
}
```

Update `crates/vox_data/src/lib.rs`:
```rust
pub mod vxm;
pub mod materials;
pub mod proc_gs;
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p vox_data proc_gs`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add crates/vox_data/
git commit -m "feat(vox_data): add Proc-GS rule parser and deterministic Gaussian emitter"
```

---

### Task 11: Asset Library Index

**Files:**
- Create: `crates/vox_data/src/library.rs`
- Modify: `crates/vox_data/src/lib.rs`
- Test: `crates/vox_data/tests/library_test.rs`

- [ ] **Step 1: Write failing test**

`crates/vox_data/tests/library_test.rs`:
```rust
use uuid::Uuid;
use vox_data::library::{AssetLibrary, AssetEntry, AssetType, AssetPipeline};

#[test]
fn register_and_lookup_by_uuid() {
    let mut lib = AssetLibrary::new();
    let uuid = Uuid::new_v4();
    lib.register(AssetEntry {
        uuid,
        path: "buildings/house_01.vxm".into(),
        asset_type: AssetType::Building,
        style: "victorian_terraced".into(),
        tags: vec!["victorian".into(), "residential".into()],
        pipeline: AssetPipeline::ProcGS,
    });

    let entry = lib.get(uuid).unwrap();
    assert_eq!(entry.style, "victorian_terraced");
}

#[test]
fn search_by_tag() {
    let mut lib = AssetLibrary::new();
    let uuid1 = Uuid::new_v4();
    let uuid2 = Uuid::new_v4();
    lib.register(AssetEntry {
        uuid: uuid1,
        path: "buildings/house_01.vxm".into(),
        asset_type: AssetType::Building,
        style: "victorian".into(),
        tags: vec!["victorian".into(), "residential".into()],
        pipeline: AssetPipeline::ProcGS,
    });
    lib.register(AssetEntry {
        uuid: uuid2,
        path: "props/bench_01.vxm".into(),
        asset_type: AssetType::Prop,
        style: "victorian".into(),
        tags: vec!["victorian".into(), "park".into()],
        pipeline: AssetPipeline::Turnaround,
    });

    let victorian = lib.search_by_tag("victorian");
    assert_eq!(victorian.len(), 2);

    let park = lib.search_by_tag("park");
    assert_eq!(park.len(), 1);
    assert_eq!(park[0].uuid, uuid2);
}

#[test]
fn search_by_type() {
    let mut lib = AssetLibrary::new();
    lib.register(AssetEntry {
        uuid: Uuid::new_v4(),
        path: "buildings/house_01.vxm".into(),
        asset_type: AssetType::Building,
        style: "victorian".into(),
        tags: vec![],
        pipeline: AssetPipeline::ProcGS,
    });
    lib.register(AssetEntry {
        uuid: Uuid::new_v4(),
        path: "props/bench_01.vxm".into(),
        asset_type: AssetType::Prop,
        style: "victorian".into(),
        tags: vec![],
        pipeline: AssetPipeline::Turnaround,
    });

    let buildings = lib.search_by_type(AssetType::Building);
    assert_eq!(buildings.len(), 1);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vox_data library`
Expected: FAIL

- [ ] **Step 3: Implement asset library**

`crates/vox_data/src/library.rs`:
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetType {
    Building,
    Prop,
    Vegetation,
    Terrain,
    Character,
    Component,
    Vehicle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetPipeline {
    ProcGS,
    Turnaround,
    NeuralInfill,
    LyraCapture,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetEntry {
    pub uuid: Uuid,
    pub path: String,
    pub asset_type: AssetType,
    pub style: String,
    pub tags: Vec<String>,
    pub pipeline: AssetPipeline,
}

pub struct AssetLibrary {
    entries: HashMap<Uuid, AssetEntry>,
}

impl AssetLibrary {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn register(&mut self, entry: AssetEntry) {
        self.entries.insert(entry.uuid, entry);
    }

    pub fn get(&self, uuid: Uuid) -> Option<&AssetEntry> {
        self.entries.get(&uuid)
    }

    pub fn search_by_tag(&self, tag: &str) -> Vec<&AssetEntry> {
        self.entries
            .values()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .collect()
    }

    pub fn search_by_type(&self, asset_type: AssetType) -> Vec<&AssetEntry> {
        self.entries
            .values()
            .filter(|e| e.asset_type == asset_type)
            .collect()
    }

    pub fn all(&self) -> impl Iterator<Item = &AssetEntry> {
        self.entries.values()
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }
}
```

Update `crates/vox_data/src/lib.rs`:
```rust
pub mod vxm;
pub mod materials;
pub mod proc_gs;
pub mod library;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p vox_data library`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add crates/vox_data/
git commit -m "feat(vox_data): add asset library with UUID indexing and tag/type search"
```

---

### Task 12: VXM v0.2 with Entity ID

**Files:**
- Create: `crates/vox_data/src/vxm_v2.rs`
- Modify: `crates/vox_data/src/lib.rs`
- Test: `crates/vox_data/tests/vxm_v2_test.rs`

- [ ] **Step 1: Write failing test**

`crates/vox_data/tests/vxm_v2_test.rs`:
```rust
use vox_data::vxm_v2::{GaussianSplatV2, VxmFileV2, VxmHeaderV2};
use uuid::Uuid;
use half::f16;

#[test]
fn splat_v2_is_54_bytes() {
    assert_eq!(std::mem::size_of::<GaussianSplatV2>(), 54);
}

#[test]
fn round_trip_v2() {
    let uuid = Uuid::new_v4();
    let splats = vec![GaussianSplatV2 {
        position: [1.0, 2.0, 3.0],
        scale: [0.1, 0.1, 0.1],
        rotation: [0, 0, 0, 32767],
        opacity: 255,
        semantic_zone: 2,
        entity_id: 42,
        spectral: [f16::from_f32(0.5).to_bits(); 8],
    }];

    let file = VxmFileV2::new(uuid, splats);
    let mut buf = Vec::new();
    file.write(&mut buf).unwrap();

    let loaded = VxmFileV2::read(&buf[..]).unwrap();
    assert_eq!(loaded.splats[0].entity_id, 42);
    assert_eq!(loaded.splats[0].semantic_zone, 2);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vox_data vxm_v2`
Expected: FAIL

- [ ] **Step 3: Implement VXM v0.2**

`crates/vox_data/src/vxm_v2.rs`:
```rust
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice, try_from_bytes};
use std::io::{Read, Write};
use uuid::Uuid;

use crate::vxm::VxmError;

const MAGIC: [u8; 4] = *b"VXMF";
const VERSION: u16 = 2;

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GaussianSplatV2 {
    pub position: [f32; 3],
    pub scale: [f32; 3],
    pub rotation: [i16; 4],
    pub opacity: u8,
    pub semantic_zone: u8,
    pub entity_id: u16,
    pub spectral: [u16; 8],
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct VxmHeaderV2 {
    pub magic: [u8; 4],
    pub version: u16,
    pub flags: u16,
    pub asset_uuid: [u8; 16],
    pub splat_count: u32,
    pub material_type: u8,
    pub _pad: [u8; 23],
}

#[derive(Debug, Clone)]
pub struct VxmFileV2 {
    pub header: VxmHeaderV2,
    pub splats: Vec<GaussianSplatV2>,
}

impl VxmFileV2 {
    pub fn new(uuid: Uuid, splats: Vec<GaussianSplatV2>) -> Self {
        Self {
            header: VxmHeaderV2 {
                magic: MAGIC,
                version: VERSION,
                flags: 0,
                asset_uuid: *uuid.as_bytes(),
                splat_count: splats.len() as u32,
                material_type: 0,
                _pad: [0; 23],
            },
            splats,
        }
    }

    pub fn write<W: Write>(&self, mut writer: W) -> Result<(), VxmError> {
        writer.write_all(bytes_of(&self.header))?;
        let splat_bytes = cast_slice::<GaussianSplatV2, u8>(&self.splats);
        let compressed = zstd::bulk::compress(splat_bytes, 3).map_err(VxmError::Compress)?;
        writer.write_all(&(compressed.len() as u64).to_le_bytes())?;
        writer.write_all(&compressed)?;
        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> Result<Self, VxmError> {
        let mut header_bytes = [0u8; 64];
        reader.read_exact(&mut header_bytes)?;
        let header: VxmHeaderV2 = *try_from_bytes(&header_bytes).map_err(|_| VxmError::InvalidAlignment)?;

        if header.magic != MAGIC { return Err(VxmError::InvalidMagic); }
        if header.version != VERSION { return Err(VxmError::UnsupportedVersion(header.version)); }

        let mut size_bytes = [0u8; 8];
        reader.read_exact(&mut size_bytes)?;
        let compressed_size = u64::from_le_bytes(size_bytes) as usize;

        let mut compressed = vec![0u8; compressed_size];
        reader.read_exact(&mut compressed)?;

        let expected_size = header.splat_count as usize * std::mem::size_of::<GaussianSplatV2>();
        let decompressed = zstd::bulk::decompress(&compressed, expected_size).map_err(VxmError::Decompress)?;
        let splats: Vec<GaussianSplatV2> = cast_slice::<u8, GaussianSplatV2>(&decompressed).to_vec();

        Ok(VxmFileV2 { header, splats })
    }
}
```

Update `crates/vox_data/src/lib.rs`:
```rust
pub mod vxm;
pub mod vxm_v2;
pub mod materials;
pub mod proc_gs;
pub mod library;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p vox_data vxm_v2`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add crates/vox_data/
git commit -m "feat(vox_data): add .vxm v0.2 format with entity_id and semantic_zone per splat"
```

---

## Phase 2 — Intelligence (Scaffold)

### Task 13: Neural Layout Interpreter Interface (`vox_nn`)

**Files:**
- Create: `crates/vox_nn/Cargo.toml`
- Create: `crates/vox_nn/src/lib.rs`
- Create: `crates/vox_nn/src/layout.rs`
- Create: `crates/vox_nn/src/scene_graph.rs`
- Modify: `Cargo.toml` (workspace members)
- Test: `crates/vox_nn/tests/layout_test.rs`

- [ ] **Step 1: Create vox_nn crate and add to workspace**

Add `"crates/vox_nn"` to workspace members in root `Cargo.toml`.

`crates/vox_nn/Cargo.toml`:
```toml
[package]
name = "vox_nn"
edition.workspace = true
version.workspace = true

[dependencies]
vox_core = { path = "../vox_core" }
vox_data = { path = "../vox_data" }
serde = { workspace = true }
serde_json = "1"
glam = { workspace = true }
uuid = { workspace = true }
thiserror = { workspace = true }
tokio = { version = "1", features = ["full"] }
```

- [ ] **Step 2: Implement SceneGraph types**

`crates/vox_nn/src/scene_graph.rs`:
```rust
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneGraph {
    pub street: Option<StreetLayout>,
    pub slots: Vec<BuildingSlot>,
    pub props: Vec<PropSlot>,
    pub vegetation: Vec<VegetationSlot>,
    pub atmosphere: AtmosphereState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreetLayout {
    pub width: f32,
    pub length: f32,
    pub orientation_degrees: f32,
    pub surface: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingSlot {
    pub position: [f32; 3],
    pub rule: String,
    pub seed: u64,
    pub wear: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropSlot {
    pub position: [f32; 3],
    pub asset: String,
    pub rotation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VegetationSlot {
    pub position: [f32; 3],
    pub rule: String,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtmosphereState {
    pub weather: Weather,
    pub time_of_day: f32,
    pub season: Season,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Weather {
    Clear,
    Overcast,
    LightRain,
    HeavyRain,
    Fog,
    Snow,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Season {
    Spring,
    Summer,
    Autumn,
    Winter,
}
```

- [ ] **Step 3: Implement Layout Interpreter interface**

`crates/vox_nn/src/layout.rs`:
```rust
use thiserror::Error;
use crate::scene_graph::SceneGraph;

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("LLM inference failed: {0}")]
    InferenceFailed(String),
    #[error("invalid scene graph from LLM: {0}")]
    InvalidSceneGraph(String),
    #[error("constraint enforcement failed: {0}")]
    ConstraintFailed(String),
}

/// Trait for Layout Interpreters — the "Architect Brain".
/// Phase 2 will provide a real LLM-backed implementation.
pub trait LayoutInterpreter: Send + Sync {
    /// Generate a SceneGraph from a text prompt.
    fn interpret(&self, prompt: &str) -> Result<SceneGraph, LayoutError>;
}

/// Stub implementation that returns a hardcoded scene for testing.
pub struct StubLayoutInterpreter;

impl LayoutInterpreter for StubLayoutInterpreter {
    fn interpret(&self, prompt: &str) -> Result<SceneGraph, LayoutError> {
        use crate::scene_graph::*;

        Ok(SceneGraph {
            street: Some(StreetLayout {
                width: 8.0,
                length: 100.0,
                orientation_degrees: 0.0,
                surface: "cobblestone_panel".into(),
            }),
            slots: vec![
                BuildingSlot { position: [0.0, 0.0, 5.0], rule: "house_victorian_terraced".into(), seed: 1, wear: 0.3 },
                BuildingSlot { position: [7.0, 0.0, 5.0], rule: "house_victorian_terraced".into(), seed: 2, wear: 0.5 },
                BuildingSlot { position: [14.0, 0.0, 5.0], rule: "house_victorian_terraced".into(), seed: 3, wear: 0.2 },
            ],
            props: vec![
                PropSlot { position: [3.5, 0.0, 2.0], asset: "lamp_post_gas_era".into(), rotation: 0.0 },
            ],
            vegetation: vec![
                VegetationSlot { position: [10.0, 0.0, 1.0], rule: "oak_summer".into(), seed: 99 },
            ],
            atmosphere: AtmosphereState {
                weather: Weather::Clear,
                time_of_day: 14.0,
                season: Season::Autumn,
            },
        })
    }
}
```

`crates/vox_nn/src/lib.rs`:
```rust
pub mod scene_graph;
pub mod layout;
```

- [ ] **Step 4: Write test**

`crates/vox_nn/tests/layout_test.rs`:
```rust
use vox_nn::layout::{LayoutInterpreter, StubLayoutInterpreter};
use vox_nn::scene_graph::Weather;

#[test]
fn stub_interpreter_returns_valid_scene_graph() {
    let interpreter = StubLayoutInterpreter;
    let sg = interpreter.interpret("A Victorian street").unwrap();

    assert!(sg.street.is_some());
    assert!(!sg.slots.is_empty());
    assert!(!sg.props.is_empty());

    let street = sg.street.unwrap();
    assert!(street.width > 0.0);
    assert!(street.length > 0.0);
}

#[test]
fn scene_graph_serialises_to_json() {
    let interpreter = StubLayoutInterpreter;
    let sg = interpreter.interpret("test").unwrap();
    let json = serde_json::to_string_pretty(&sg).unwrap();
    assert!(json.contains("victorian_terraced"));

    // Round-trip
    let parsed: vox_nn::scene_graph::SceneGraph = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.slots.len(), sg.slots.len());
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p vox_nn`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add crates/vox_nn/ Cargo.toml
git commit -m "feat(vox_nn): add SceneGraph types and stub LayoutInterpreter for Phase 2"
```

---

### Task 14: Weather and Wear SPD Shifts

**Files:**
- Create: `crates/vox_render/src/spectral_shift.rs`
- Modify: `crates/vox_render/src/lib.rs`
- Test: `crates/vox_render/tests/spectral_shift_test.rs`

- [ ] **Step 1: Write failing test**

`crates/vox_render/tests/spectral_shift_test.rs`:
```rust
use vox_core::spectral::SpectralBands;
use vox_render::spectral_shift::{WeatherState, apply_weather_shift, apply_wear_shift};

#[test]
fn rain_adds_specular_spike() {
    let base = SpectralBands([0.5; 8]);
    let shifted = apply_weather_shift(&base, WeatherState::LightRain);
    // Rain should increase some bands (specular)
    let total_base: f32 = base.0.iter().sum();
    let total_shifted: f32 = shifted.0.iter().sum();
    assert!(total_shifted != total_base, "Rain should modify SPD");
}

#[test]
fn wear_darkens_material() {
    let base = SpectralBands([0.5; 8]);
    let worn = apply_wear_shift(&base, &base, 0.8);
    // At high wear, should be closer to worn SPD (darker)
    // With equal base and worn, should be unchanged
    for i in 0..8 {
        assert!((worn.0[i] - 0.5).abs() < 0.001);
    }

    // With different worn SPD
    let worn_spd = SpectralBands([0.2; 8]);
    let result = apply_wear_shift(&base, &worn_spd, 1.0);
    for i in 0..8 {
        assert!((result.0[i] - 0.2).abs() < 0.001, "At wear=1.0 should equal worn SPD");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vox_render spectral_shift`
Expected: FAIL

- [ ] **Step 3: Implement spectral shifts**

`crates/vox_render/src/spectral_shift.rs`:
```rust
use vox_core::spectral::SpectralBands;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeatherState {
    Clear,
    Overcast,
    LightRain,
    HeavyRain,
    Fog,
    Snow,
}

/// Apply weather-driven SPD modification.
pub fn apply_weather_shift(base: &SpectralBands, weather: WeatherState) -> SpectralBands {
    let mut result = base.0;
    match weather {
        WeatherState::Clear => {} // No change
        WeatherState::Overcast => {
            // Slight blue-shift, reduced overall
            for i in 0..8 {
                result[i] *= 0.85;
            }
            result[0] *= 1.05; // Boost short wavelengths slightly
            result[1] *= 1.05;
        }
        WeatherState::LightRain => {
            // Wet surface specular spike
            for i in 0..8 {
                result[i] *= 0.9;
                result[i] += 0.05; // Specular reflection added uniformly
            }
        }
        WeatherState::HeavyRain => {
            for i in 0..8 {
                result[i] *= 0.7;
                result[i] += 0.1;
            }
        }
        WeatherState::Fog => {
            // Desaturate toward grey
            let mean: f32 = result.iter().sum::<f32>() / 8.0;
            for i in 0..8 {
                result[i] = result[i] * 0.5 + mean * 0.5;
            }
        }
        WeatherState::Snow => {
            // Shift toward white on horizontal surfaces
            for i in 0..8 {
                result[i] = result[i] * 0.3 + 0.7; // Strong shift toward 1.0
            }
        }
    }
    SpectralBands(result)
}

/// Interpolate between fresh and worn SPD based on wear level (0.0–1.0).
pub fn apply_wear_shift(fresh: &SpectralBands, worn: &SpectralBands, wear: f32) -> SpectralBands {
    let wear = wear.clamp(0.0, 1.0);
    SpectralBands(std::array::from_fn(|i| {
        fresh.0[i] + (worn.0[i] - fresh.0[i]) * wear
    }))
}

/// Apply time-of-day illuminant blend factor.
pub fn time_of_day_illuminant_blend(hour: f32) -> (f32, f32, f32) {
    // Returns (d65_weight, d50_weight, a_weight)
    match hour {
        h if h < 6.0 => (0.0, 0.0, 1.0),           // Night: artificial only
        h if h < 8.0 => {
            let t = (h - 6.0) / 2.0;
            (t * 0.5, t * 0.5, 1.0 - t)              // Dawn: warm mix
        }
        h if h < 17.0 => (1.0, 0.0, 0.0),           // Day: D65
        h if h < 19.0 => {
            let t = (h - 17.0) / 2.0;
            (1.0 - t, t, 0.0)                         // Sunset: D65 → D50
        }
        h if h < 21.0 => {
            let t = (h - 19.0) / 2.0;
            (0.0, 1.0 - t, t)                         // Dusk: D50 → A
        }
        _ => (0.0, 0.0, 1.0),                         // Night
    }
}
```

Update `crates/vox_render/src/lib.rs`:
```rust
pub mod gpu;
pub mod spectral;
pub mod spectral_shift;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p vox_render spectral_shift`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/
git commit -m "feat(vox_render): add weather, wear, and time-of-day spectral shift functions"
```

---

## Phase 3 — Scale (Scaffold)

### Task 15: Large World Coordinates and Tile Types

**Files:**
- Create: `crates/vox_core/src/lwc.rs`
- Modify: `crates/vox_core/src/lib.rs`
- Test: `crates/vox_core/tests/lwc_test.rs`

- [ ] **Step 1: Write failing test**

`crates/vox_core/tests/lwc_test.rs`:
```rust
use vox_core::lwc::{WorldCoord, TileCoord};

#[test]
fn world_coord_from_absolute() {
    let wc = WorldCoord::from_absolute(47500.0, 5.0, 83200.0);
    assert_eq!(wc.tile, TileCoord { x: 47, z: 83 });
    assert!((wc.local.x - 500.0).abs() < 0.01);
    assert!((wc.local.z - 200.0).abs() < 0.01);
}

#[test]
fn world_coord_to_absolute_round_trips() {
    let original = WorldCoord::from_absolute(12345.678, 50.0, 67890.123);
    let (ax, ay, az) = original.to_absolute();
    assert!((ax - 12345.678).abs() < 0.01);
    assert!((ay - 50.0).abs() < 0.01);
    assert!((az - 67890.123).abs() < 0.01);
}

#[test]
fn tile_coord_anchor() {
    let tile = TileCoord { x: 47, z: 83 };
    let (ax, az) = tile.anchor();
    assert_eq!(ax, 47000.0);
    assert_eq!(az, 83000.0);
}

#[test]
fn no_jitter_at_50km() {
    // At 50km, f32 loses precision. LWC local offset should be sub-mm.
    let wc = WorldCoord::from_absolute(50000.5, 0.0, 50000.5);
    assert!(wc.local.x.abs() < 1000.0, "Local offset should be within tile");
    assert!((wc.local.x - 0.5).abs() < 0.001, "Sub-mm precision expected");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vox_core lwc`
Expected: FAIL

- [ ] **Step 3: Implement LWC**

`crates/vox_core/src/lwc.rs`:
```rust
use glam::Vec3;
use serde::{Deserialize, Serialize};

pub const TILE_SIZE: f64 = 1000.0; // metres

/// Tile grid coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileCoord {
    pub x: i32,
    pub z: i32,
}

impl TileCoord {
    /// World-space anchor position (f64 precision).
    pub fn anchor(&self) -> (f64, f64) {
        (self.x as f64 * TILE_SIZE, self.z as f64 * TILE_SIZE)
    }
}

/// World coordinate: tile anchor (f64) + local offset (f32).
/// Local offset is always within [-500, +500] for sub-mm f32 precision.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WorldCoord {
    pub tile: TileCoord,
    pub local: Vec3, // f32, within [-500, +500] on x/z
}

impl WorldCoord {
    /// Create from absolute world position.
    pub fn from_absolute(x: f64, y: f64, z: f64) -> Self {
        let tile_x = (x / TILE_SIZE).floor() as i32;
        let tile_z = (z / TILE_SIZE).floor() as i32;
        let local_x = (x - tile_x as f64 * TILE_SIZE) as f32;
        let local_z = (z - tile_z as f64 * TILE_SIZE) as f32;

        Self {
            tile: TileCoord { x: tile_x, z: tile_z },
            local: Vec3::new(local_x, y as f32, local_z),
        }
    }

    /// Convert back to absolute coordinates (f64).
    pub fn to_absolute(&self) -> (f64, f64, f64) {
        let (ax, az) = self.tile.anchor();
        (
            ax + self.local.x as f64,
            self.local.y as f64,
            az + self.local.z as f64,
        )
    }

    /// Get the local offset relative to a camera's tile.
    /// This is what the GPU receives — no jitter at any world scale.
    pub fn local_relative_to(&self, camera_tile: TileCoord) -> Vec3 {
        let dx = (self.tile.x - camera_tile.x) as f32 * TILE_SIZE as f32;
        let dz = (self.tile.z - camera_tile.z) as f32 * TILE_SIZE as f32;
        Vec3::new(
            self.local.x + dx,
            self.local.y,
            self.local.z + dz,
        )
    }
}

/// State of a tile in the streaming system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileState {
    /// On NVMe, not loaded.
    Cold,
    /// Decompressing, loading to system RAM.
    Warming,
    /// In system RAM, ready for GPU upload.
    Warm,
    /// In VRAM, renderable.
    Active,
    /// Being evicted from VRAM.
    Evicting,
}
```

Update `crates/vox_core/src/lib.rs`:
```rust
pub mod types;
pub mod spectral;
pub mod ecs;
pub mod svo;
pub mod lwc;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p vox_core lwc`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add crates/vox_core/
git commit -m "feat(vox_core): add Large World Coordinates — tile-anchored f64/f32 split for 100km scale"
```

---

### Task 16: Tile Streaming Skeleton

**Files:**
- Create: `crates/vox_render/src/streaming.rs`
- Modify: `crates/vox_render/src/lib.rs`
- Test: `crates/vox_render/tests/streaming_test.rs`

- [ ] **Step 1: Write failing test**

`crates/vox_render/tests/streaming_test.rs`:
```rust
use vox_core::lwc::{TileCoord, TileState};
use vox_render::streaming::TileManager;

#[test]
fn initial_state_all_cold() {
    let tm = TileManager::new();
    assert_eq!(tm.tile_state(TileCoord { x: 0, z: 0 }), TileState::Cold);
}

#[test]
fn update_camera_activates_nearby_tiles() {
    let mut tm = TileManager::new();
    tm.update_camera(TileCoord { x: 5, z: 5 });

    // Camera tile should be active
    assert_eq!(tm.tile_state(TileCoord { x: 5, z: 5 }), TileState::Active);
    // Adjacent tiles should be active (1-tile buffer)
    assert_eq!(tm.tile_state(TileCoord { x: 4, z: 5 }), TileState::Active);
    assert_eq!(tm.tile_state(TileCoord { x: 6, z: 5 }), TileState::Active);
    // Far tiles should be cold
    assert_eq!(tm.tile_state(TileCoord { x: 50, z: 50 }), TileState::Cold);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vox_render streaming`
Expected: FAIL

- [ ] **Step 3: Implement tile manager skeleton**

`crates/vox_render/src/streaming.rs`:
```rust
use std::collections::HashMap;
use vox_core::lwc::{TileCoord, TileState};

/// Manages tile streaming states based on camera position.
pub struct TileManager {
    states: HashMap<TileCoord, TileState>,
    active_radius: i32,
}

impl TileManager {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            active_radius: 1, // 1-tile buffer around camera
        }
    }

    pub fn tile_state(&self, tile: TileCoord) -> TileState {
        self.states.get(&tile).copied().unwrap_or(TileState::Cold)
    }

    /// Update which tiles are active based on camera tile position.
    pub fn update_camera(&mut self, camera_tile: TileCoord) {
        // Mark tiles outside active radius as cold
        let to_evict: Vec<TileCoord> = self.states.keys()
            .filter(|t| {
                (t.x - camera_tile.x).abs() > self.active_radius + 1
                    || (t.z - camera_tile.z).abs() > self.active_radius + 1
            })
            .copied()
            .collect();

        for tile in to_evict {
            self.states.remove(&tile);
        }

        // Activate tiles within radius
        for dx in -self.active_radius..=self.active_radius {
            for dz in -self.active_radius..=self.active_radius {
                let tile = TileCoord {
                    x: camera_tile.x + dx,
                    z: camera_tile.z + dz,
                };
                self.states.entry(tile).or_insert(TileState::Active);
            }
        }
    }

    /// Get all currently active tiles.
    pub fn active_tiles(&self) -> Vec<TileCoord> {
        self.states
            .iter()
            .filter(|(_, s)| **s == TileState::Active)
            .map(|(t, _)| *t)
            .collect()
    }
}
```

Update `crates/vox_render/src/lib.rs`:
```rust
pub mod gpu;
pub mod spectral;
pub mod spectral_shift;
pub mod streaming;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p vox_render streaming`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/
git commit -m "feat(vox_render): add tile streaming manager skeleton for VRAM virtualization"
```

---

### Task 17: Agent Simulation Skeleton

**Files:**
- Create: `crates/vox_sim/Cargo.toml`
- Create: `crates/vox_sim/src/lib.rs`
- Create: `crates/vox_sim/src/agent.rs`
- Modify: `Cargo.toml` (workspace members)
- Test: `crates/vox_sim/tests/agent_test.rs`

- [ ] **Step 1: Create vox_sim crate**

Add `"crates/vox_sim"` to workspace members.

`crates/vox_sim/Cargo.toml`:
```toml
[package]
name = "vox_sim"
edition.workspace = true
version.workspace = true

[dependencies]
vox_core = { path = "../vox_core" }
glam = { workspace = true }
serde = { workspace = true }
```

- [ ] **Step 2: Write failing test**

`crates/vox_sim/tests/agent_test.rs`:
```rust
use glam::Vec3;
use vox_core::lwc::WorldCoord;
use vox_sim::agent::{Agent, AgentManager};

#[test]
fn agent_moves_toward_destination() {
    let mut mgr = AgentManager::new();
    let id = mgr.spawn(Agent {
        position: WorldCoord::from_absolute(0.0, 0.0, 0.0),
        velocity: Vec3::ZERO,
        destination: Some(WorldCoord::from_absolute(10.0, 0.0, 0.0)),
        speed: 1.4, // walking speed m/s
    });

    mgr.tick(1.0); // 1 second tick

    let agent = mgr.get(id).unwrap();
    let (x, _, _) = agent.position.to_absolute();
    assert!(x > 0.0, "Agent should have moved toward destination");
    assert!(x < 10.0, "Agent should not have reached destination in 1s");
}

#[test]
fn agent_stops_at_destination() {
    let mut mgr = AgentManager::new();
    let id = mgr.spawn(Agent {
        position: WorldCoord::from_absolute(0.0, 0.0, 0.0),
        velocity: Vec3::ZERO,
        destination: Some(WorldCoord::from_absolute(1.0, 0.0, 0.0)),
        speed: 10.0, // fast
    });

    mgr.tick(1.0);

    let agent = mgr.get(id).unwrap();
    let (x, _, _) = agent.position.to_absolute();
    assert!((x - 1.0).abs() < 0.1, "Agent should be at destination, got {}", x);
    assert!(agent.destination.is_none(), "Destination should be cleared on arrival");
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p vox_sim agent`
Expected: FAIL

- [ ] **Step 4: Implement agent manager**

`crates/vox_sim/src/agent.rs`:
```rust
use glam::Vec3;
use std::collections::HashMap;
use vox_core::lwc::WorldCoord;

#[derive(Debug, Clone)]
pub struct Agent {
    pub position: WorldCoord,
    pub velocity: Vec3,
    pub destination: Option<WorldCoord>,
    pub speed: f32,
}

pub struct AgentManager {
    agents: HashMap<u32, Agent>,
    next_id: u32,
}

impl AgentManager {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn spawn(&mut self, agent: Agent) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.agents.insert(id, agent);
        id
    }

    pub fn get(&self, id: u32) -> Option<&Agent> {
        self.agents.get(&id)
    }

    pub fn count(&self) -> usize {
        self.agents.len()
    }

    /// Advance all agents by `dt` seconds.
    pub fn tick(&mut self, dt: f32) {
        let ids: Vec<u32> = self.agents.keys().copied().collect();
        for id in ids {
            let agent = self.agents.get_mut(&id).unwrap();
            if let Some(dest) = &agent.destination {
                let (cx, cy, cz) = agent.position.to_absolute();
                let (dx, dy, dz) = dest.to_absolute();

                let dir = Vec3::new(
                    (dx - cx) as f32,
                    (dy - cy) as f32,
                    (dz - cz) as f32,
                );

                let dist = dir.length();
                if dist < 0.5 {
                    // Arrived
                    agent.position = *dest;
                    agent.destination = None;
                    agent.velocity = Vec3::ZERO;
                } else {
                    let move_dir = dir / dist;
                    let move_dist = (agent.speed * dt).min(dist);
                    agent.velocity = move_dir * agent.speed;

                    let new_x = cx + (move_dir.x * move_dist) as f64;
                    let new_y = cy + (move_dir.y * move_dist) as f64;
                    let new_z = cz + (move_dir.z * move_dist) as f64;
                    agent.position = WorldCoord::from_absolute(new_x, new_y, new_z);
                }
            }
        }
    }
}
```

`crates/vox_sim/src/lib.rs`:
```rust
pub mod agent;
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p vox_sim agent`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add crates/vox_sim/ Cargo.toml
git commit -m "feat(vox_sim): add agent simulation skeleton with position-based movement"
```

---

## Phase 4 — Ship the City Builder (Interfaces + Stubs)

### Task 18: Phase 4 Crate Scaffolding

**Files:**
- Create: `crates/vox_net/Cargo.toml`, `crates/vox_net/src/lib.rs`
- Create: `crates/vox_script/Cargo.toml`, `crates/vox_script/src/lib.rs`
- Create: `crates/vox_audio/Cargo.toml`, `crates/vox_audio/src/lib.rs`
- Create: `crates/vox_physics/Cargo.toml`, `crates/vox_physics/src/lib.rs`
- Create: `crates/vox_terrain/Cargo.toml`, `crates/vox_terrain/src/lib.rs`
- Create: `crates/vox_ui/Cargo.toml`, `crates/vox_ui/src/lib.rs`
- Modify: `Cargo.toml` (workspace members)

- [ ] **Step 1: Create all Phase 4 crates**

Add to workspace members: `"crates/vox_net"`, `"crates/vox_script"`, `"crates/vox_audio"`, `"crates/vox_physics"`, `"crates/vox_terrain"`, `"crates/vox_ui"`

`crates/vox_net/Cargo.toml`:
```toml
[package]
name = "vox_net"
edition.workspace = true
version.workspace = true

[dependencies]
vox_core = { path = "../vox_core" }
serde = { workspace = true }
thiserror = { workspace = true }
```

`crates/vox_net/src/lib.rs`:
```rust
//! CRDT networking crate — engine-layer, game-agnostic.
//! Provides entity replication over QUIC transport.

pub mod replication;

pub use replication::{ReplicationServer, ReplicationClient, EntityDelta};
```

Create `crates/vox_net/src/replication.rs`:
```rust
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NetError {
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("replication error: {0}")]
    ReplicationError(String),
}

/// A delta update for a single entity component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDelta {
    pub entity_id: u32,
    pub component: String,
    pub data: Vec<u8>,
    pub timestamp: u64,
}

/// Server-side replication manager.
pub struct ReplicationServer {
    tick: u64,
}

impl ReplicationServer {
    pub fn new() -> Self {
        Self { tick: 0 }
    }

    pub fn tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Apply a client input and return deltas to broadcast.
    pub fn apply_input(&mut self, _input: &[u8]) -> Vec<EntityDelta> {
        // Stub: real implementation validates + applies + returns deltas
        Vec::new()
    }
}

/// Client-side replication receiver.
pub struct ReplicationClient {
    server_tick: u64,
}

impl ReplicationClient {
    pub fn new() -> Self {
        Self { server_tick: 0 }
    }

    pub fn apply_deltas(&mut self, deltas: &[EntityDelta]) {
        for delta in deltas {
            self.server_tick = self.server_tick.max(delta.timestamp);
            // Stub: apply delta to local ECS
        }
    }
}
```

`crates/vox_script/Cargo.toml`:
```toml
[package]
name = "vox_script"
edition.workspace = true
version.workspace = true

[dependencies]
vox_core = { path = "../vox_core" }
thiserror = { workspace = true }
```

`crates/vox_script/src/lib.rs`:
```rust
//! Wasm scripting runtime — engine-layer, game-agnostic.
//! Provides sandboxed mod execution via wasmtime.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScriptError {
    #[error("module load failed: {0}")]
    LoadFailed(String),
    #[error("execution exceeded budget")]
    BudgetExceeded,
    #[error("sandbox violation: {0}")]
    SandboxViolation(String),
}

/// A loaded Wasm mod module.
pub struct ScriptModule {
    pub name: String,
    pub memory_budget_bytes: usize,
    pub cpu_budget_ms: f32,
}

/// The scripting runtime that manages loaded mods.
pub struct ScriptRuntime {
    modules: Vec<ScriptModule>,
}

impl ScriptRuntime {
    pub fn new() -> Self {
        Self { modules: Vec::new() }
    }

    pub fn load_module(&mut self, name: &str, _wasm_bytes: &[u8]) -> Result<(), ScriptError> {
        self.modules.push(ScriptModule {
            name: name.to_string(),
            memory_budget_bytes: 64 * 1024 * 1024, // 64MB
            cpu_budget_ms: 2.0,
        });
        Ok(())
    }

    pub fn tick(&mut self, _dt: f32) {
        // Stub: execute all module tick functions within budget
    }

    pub fn module_count(&self) -> usize {
        self.modules.len()
    }
}
```

`crates/vox_audio/Cargo.toml`:
```toml
[package]
name = "vox_audio"
edition.workspace = true
version.workspace = true

[dependencies]
vox_core = { path = "../vox_core" }
glam = { workspace = true }
```

`crates/vox_audio/src/lib.rs`:
```rust
//! Spatial audio engine — engine-layer, game-agnostic.

use glam::Vec3;

/// An audio source in the world.
pub struct AudioSource {
    pub id: u32,
    pub position: Vec3,
    pub volume: f32,
    pub looping: bool,
    pub clip: String,
}

/// The audio engine manages sources and the listener.
pub struct AudioEngine {
    sources: Vec<AudioSource>,
    listener_position: Vec3,
    max_sources: usize,
}

impl AudioEngine {
    pub fn new(max_sources: usize) -> Self {
        Self {
            sources: Vec::new(),
            listener_position: Vec3::ZERO,
            max_sources,
        }
    }

    pub fn set_listener(&mut self, position: Vec3) {
        self.listener_position = position;
    }

    pub fn play(&mut self, source: AudioSource) -> u32 {
        let id = source.id;
        if self.sources.len() < self.max_sources {
            self.sources.push(source);
        }
        id
    }

    pub fn stop(&mut self, id: u32) {
        self.sources.retain(|s| s.id != id);
    }

    pub fn active_count(&self) -> usize {
        self.sources.len()
    }
}
```

`crates/vox_physics/Cargo.toml`:
```toml
[package]
name = "vox_physics"
edition.workspace = true
version.workspace = true

[dependencies]
vox_core = { path = "../vox_core" }
glam = { workspace = true }
```

`crates/vox_physics/src/lib.rs`:
```rust
//! Physics engine — engine-layer, game-agnostic.
//! Rigid body simulation with Gaussian-native collision shapes.

use glam::Vec3;

#[derive(Debug, Clone)]
pub struct RigidBody {
    pub id: u32,
    pub position: Vec3,
    pub velocity: Vec3,
    pub mass: f32,
    pub is_static: bool,
}

/// XPBD-based physics world.
pub struct PhysicsWorld {
    bodies: Vec<RigidBody>,
    gravity: Vec3,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            gravity: Vec3::new(0.0, -9.81, 0.0),
        }
    }

    pub fn add_body(&mut self, body: RigidBody) -> u32 {
        let id = body.id;
        self.bodies.push(body);
        id
    }

    pub fn step(&mut self, dt: f32) {
        for body in &mut self.bodies {
            if !body.is_static {
                body.velocity += self.gravity * dt;
                body.position += body.velocity * dt;

                // Simple ground plane collision
                if body.position.y < 0.0 {
                    body.position.y = 0.0;
                    body.velocity.y = -body.velocity.y * 0.3; // Bounce
                }
            }
        }
    }

    pub fn get_body(&self, id: u32) -> Option<&RigidBody> {
        self.bodies.iter().find(|b| b.id == id)
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }
}
```

`crates/vox_terrain/Cargo.toml`:
```toml
[package]
name = "vox_terrain"
edition.workspace = true
version.workspace = true

[dependencies]
vox_core = { path = "../vox_core" }
glam = { workspace = true }
serde = { workspace = true }
```

`crates/vox_terrain/src/lib.rs`:
```rust
//! Terrain engine — engine-layer, game-agnostic.
//! Heightmap-based terrain with elevation, rivers, and sculpting.

use serde::{Deserialize, Serialize};

pub const HEIGHTMAP_SIZE: usize = 4096;
pub const HEIGHTMAP_RESOLUTION: f32 = 0.25; // metres per sample

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurfaceType {
    Grass,
    Dirt,
    Rock,
    Sand,
    Snow,
    WaterBed,
}

/// A terrain tile heightmap.
pub struct TerrainTile {
    /// Height values, row-major.
    pub heights: Vec<f32>,
    /// Surface type per sample.
    pub surfaces: Vec<SurfaceType>,
    pub size: usize,
}

impl TerrainTile {
    pub fn flat(height: f32) -> Self {
        let count = HEIGHTMAP_SIZE * HEIGHTMAP_SIZE;
        Self {
            heights: vec![height; count],
            surfaces: vec![SurfaceType::Grass; count],
            size: HEIGHTMAP_SIZE,
        }
    }

    pub fn height_at(&self, x: usize, z: usize) -> f32 {
        if x < self.size && z < self.size {
            self.heights[z * self.size + x]
        } else {
            0.0
        }
    }

    pub fn set_height(&mut self, x: usize, z: usize, height: f32) {
        if x < self.size && z < self.size {
            self.heights[z * self.size + x] = height;
        }
    }

    /// Sample height at world-local position (bilinear interpolation).
    pub fn sample(&self, local_x: f32, local_z: f32) -> f32 {
        let fx = local_x / HEIGHTMAP_RESOLUTION;
        let fz = local_z / HEIGHTMAP_RESOLUTION;
        let ix = fx.floor() as usize;
        let iz = fz.floor() as usize;
        let tx = fx.fract();
        let tz = fz.fract();

        let h00 = self.height_at(ix, iz);
        let h10 = self.height_at(ix + 1, iz);
        let h01 = self.height_at(ix, iz + 1);
        let h11 = self.height_at(ix + 1, iz + 1);

        let h0 = h00 + (h10 - h00) * tx;
        let h1 = h01 + (h11 - h01) * tx;
        h0 + (h1 - h0) * tz
    }
}
```

`crates/vox_ui/Cargo.toml`:
```toml
[package]
name = "vox_ui"
edition.workspace = true
version.workspace = true

[dependencies]
vox_core = { path = "../vox_core" }
glam = { workspace = true }
serde = { workspace = true }
```

`crates/vox_ui/src/lib.rs`:
```rust
//! Game UI framework — engine-layer, game-agnostic.
//! Retained-mode UI with flexbox layout and theming.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theme {
    pub background: [f32; 4],
    pub text: [f32; 4],
    pub accent: [f32; 4],
    pub font_size: f32,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            background: [0.1, 0.1, 0.12, 0.95],
            text: [0.9, 0.9, 0.9, 1.0],
            accent: [0.3, 0.6, 0.9, 1.0],
            font_size: 16.0,
        }
    }
}

/// A UI node in the retained-mode tree.
#[derive(Debug)]
pub enum UiNode {
    Panel { children: Vec<UiNode>, padding: f32 },
    Text { content: String },
    Button { label: String, on_click: Option<String> },
    Slider { label: String, value: f32, min: f32, max: f32 },
}

/// The UI root manages the node tree and theme.
pub struct UiRoot {
    pub theme: Theme,
    pub nodes: Vec<UiNode>,
}

impl UiRoot {
    pub fn new() -> Self {
        Self {
            theme: Theme::default(),
            nodes: Vec::new(),
        }
    }

    pub fn add(&mut self, node: UiNode) {
        self.nodes.push(node);
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }
}
```

- [ ] **Step 2: Verify all crates compile**

Run: `cargo build`
Expected: compiles with no errors

- [ ] **Step 3: Commit**

```bash
git add crates/vox_net/ crates/vox_script/ crates/vox_audio/ crates/vox_physics/ crates/vox_terrain/ crates/vox_ui/ Cargo.toml
git commit -m "feat: scaffold Phase 4 crates — vox_net, vox_script, vox_audio, vox_physics, vox_terrain, vox_ui"
```

---

### Task 19: City Simulation Types (`vox_sim`)

**Files:**
- Create: `crates/vox_sim/src/citizen.rs`
- Create: `crates/vox_sim/src/economy.rs`
- Create: `crates/vox_sim/src/zoning.rs`
- Create: `crates/vox_sim/src/services.rs`
- Create: `crates/vox_sim/src/transport.rs`
- Modify: `crates/vox_sim/src/lib.rs`
- Test: `crates/vox_sim/tests/citizen_test.rs`

- [ ] **Step 1: Implement citizen types**

`crates/vox_sim/src/citizen.rs`:
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleStage {
    Child,
    Student,
    Worker,
    Retired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EducationLevel {
    None,
    Primary,
    Secondary,
    University,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Needs {
    pub housing: f32,
    pub food: f32,
    pub health: f32,
    pub safety: f32,
    pub education: f32,
    pub employment: f32,
    pub leisure: f32,
}

impl Needs {
    pub fn satisfaction(&self) -> f32 {
        (self.housing + self.food + self.health + self.safety
            + self.education + self.employment + self.leisure)
            / 7.0
    }
}

impl Default for Needs {
    fn default() -> Self {
        Self {
            housing: 0.5,
            food: 0.5,
            health: 0.5,
            safety: 0.5,
            education: 0.5,
            employment: 0.5,
            leisure: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citizen {
    pub id: u32,
    pub agent_id: u32,
    pub age: f32,
    pub lifecycle: LifecycleStage,
    pub education: EducationLevel,
    pub employment: Option<u32>,
    pub residence: Option<u32>,
    pub satisfaction: f32,
    pub needs: Needs,
}

impl Citizen {
    pub fn lifecycle_for_age(age: f32) -> LifecycleStage {
        match age {
            a if a < 6.0 => LifecycleStage::Child,
            a if a < 18.0 => LifecycleStage::Student,
            a if a < 65.0 => LifecycleStage::Worker,
            _ => LifecycleStage::Retired,
        }
    }
}
```

- [ ] **Step 2: Implement economy types**

`crates/vox_sim/src/economy.rs`:
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CityBudget {
    pub funds: f64,
    pub income_residential_tax: f64,
    pub income_commercial_tax: f64,
    pub income_industrial_tax: f64,
    pub income_transport_fares: f64,
    pub expense_services: f64,
    pub expense_infrastructure: f64,
    pub expense_loans: f64,
    pub tax_rate_residential: f32,
    pub tax_rate_commercial: f32,
    pub tax_rate_industrial: f32,
}

impl Default for CityBudget {
    fn default() -> Self {
        Self {
            funds: 50_000.0,
            income_residential_tax: 0.0,
            income_commercial_tax: 0.0,
            income_industrial_tax: 0.0,
            income_transport_fares: 0.0,
            expense_services: 0.0,
            expense_infrastructure: 0.0,
            expense_loans: 0.0,
            tax_rate_residential: 0.09,
            tax_rate_commercial: 0.10,
            tax_rate_industrial: 0.12,
        }
    }
}

impl CityBudget {
    pub fn total_income(&self) -> f64 {
        self.income_residential_tax + self.income_commercial_tax
            + self.income_industrial_tax + self.income_transport_fares
    }

    pub fn total_expenses(&self) -> f64 {
        self.expense_services + self.expense_infrastructure + self.expense_loans
    }

    pub fn net(&self) -> f64 {
        self.total_income() - self.total_expenses()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    Timber,
    Stone,
    Iron,
    Clay,
    Wheat,
    Vegetables,
    Livestock,
    Planks,
    Tools,
    Bread,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStock {
    pub resource: ResourceType,
    pub quantity: f32,
    pub capacity: f32,
}
```

- [ ] **Step 3: Implement zoning types**

`crates/vox_sim/src/zoning.rs`:
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZoneType {
    ResidentialLow,
    ResidentialMedium,
    ResidentialHigh,
    CommercialLocal,
    CommercialRegional,
    IndustrialLight,
    IndustrialHeavy,
    Office,
    MixedUse,
    Agricultural,
    Park,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZoneDensity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZonePlot {
    pub id: u32,
    pub zone_type: ZoneType,
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub building_id: Option<u32>,
    pub land_value: f32,
    pub district_id: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandMeter {
    pub residential: f32,
    pub commercial: f32,
    pub industrial: f32,
}

impl Default for DemandMeter {
    fn default() -> Self {
        Self {
            residential: 0.5,
            commercial: 0.3,
            industrial: 0.2,
        }
    }
}
```

- [ ] **Step 4: Implement services types**

`crates/vox_sim/src/services.rs`:
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceType {
    PrimarySchool,
    SecondarySchool,
    University,
    Clinic,
    Hospital,
    FireStation,
    PoliceStation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceBuilding {
    pub id: u32,
    pub service_type: ServiceType,
    pub position: [f32; 2],
    pub coverage_radius: f32,
    pub capacity: u32,
    pub current_load: u32,
    pub operational_cost: f64,
    pub staff_required: u32,
}

impl ServiceBuilding {
    pub fn utilisation(&self) -> f32 {
        if self.capacity == 0 { 0.0 } else { self.current_load as f32 / self.capacity as f32 }
    }
}
```

- [ ] **Step 5: Implement transport types**

`crates/vox_sim/src/transport.rs`:
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransportType {
    Bus,
    Tram,
    Metro,
    Rail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportRoute {
    pub id: u32,
    pub transport_type: TransportType,
    pub stops: Vec<TransportStop>,
    pub vehicle_count: u32,
    pub frequency_minutes: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportStop {
    pub id: u32,
    pub position: [f32; 2],
    pub name: String,
    pub waiting_count: u32,
}
```

- [ ] **Step 6: Update lib.rs**

`crates/vox_sim/src/lib.rs`:
```rust
pub mod agent;
pub mod citizen;
pub mod economy;
pub mod zoning;
pub mod services;
pub mod transport;
```

- [ ] **Step 7: Write citizen test**

`crates/vox_sim/tests/citizen_test.rs`:
```rust
use vox_sim::citizen::{Citizen, LifecycleStage, EducationLevel, Needs};

#[test]
fn lifecycle_stage_from_age() {
    assert_eq!(Citizen::lifecycle_for_age(3.0), LifecycleStage::Child);
    assert_eq!(Citizen::lifecycle_for_age(12.0), LifecycleStage::Student);
    assert_eq!(Citizen::lifecycle_for_age(30.0), LifecycleStage::Worker);
    assert_eq!(Citizen::lifecycle_for_age(70.0), LifecycleStage::Retired);
}

#[test]
fn needs_satisfaction_average() {
    let needs = Needs {
        housing: 1.0,
        food: 1.0,
        health: 1.0,
        safety: 1.0,
        education: 1.0,
        employment: 1.0,
        leisure: 1.0,
    };
    assert!((needs.satisfaction() - 1.0).abs() < 0.001);

    let needs = Needs::default();
    assert!((needs.satisfaction() - 0.5).abs() < 0.001);
}
```

- [ ] **Step 8: Run all tests**

Run: `cargo test`
Expected: all PASS across all crates

- [ ] **Step 9: Commit**

```bash
git add crates/vox_sim/
git commit -m "feat(vox_sim): add citizen lifecycle, economy, zoning, services, and transport types"
```

---

### Task 20: Final Integration — Verify Full Workspace

**Files:**
- Modify: `crates/vox_app/Cargo.toml` (add all crate deps)
- Modify: `CLAUDE.md`

- [ ] **Step 1: Wire all crates into vox_app**

Update `crates/vox_app/Cargo.toml` dependencies:
```toml
[dependencies]
vox_core = { path = "../vox_core" }
vox_data = { path = "../vox_data" }
vox_render = { path = "../vox_render" }
vox_nn = { path = "../vox_nn" }
vox_sim = { path = "../vox_sim" }
vox_net = { path = "../vox_net" }
vox_script = { path = "../vox_script" }
vox_audio = { path = "../vox_audio" }
vox_physics = { path = "../vox_physics" }
vox_terrain = { path = "../vox_terrain" }
vox_ui = { path = "../vox_ui" }
winit = "0.30"
pollster = "0.4"
glam = { workspace = true }
uuid = { workspace = true }
bevy_ecs = { workspace = true }
bevy_app = { workspace = true }
```

- [ ] **Step 2: Update CLAUDE.md with full crate map**

Update the Architecture section in `CLAUDE.md`:
```markdown
## Architecture

Engine crates (game-agnostic — NEVER add game-specific concepts here):
- `vox_core` — types, math, spectral, ECS components, SVO, LWC
- `vox_data` — .vxm format, Proc-GS rules, materials library, asset library
- `vox_render` — GPU rendering, spectral pipeline, spectral shifts, tile streaming
- `vox_nn` — neural model interfaces (layout interpreter, infill)
- `vox_net` — CRDT networking, entity replication
- `vox_script` — Wasm scripting runtime for mods
- `vox_audio` — spatial audio engine
- `vox_physics` — rigid body physics, collision
- `vox_terrain` — heightmap terrain, sculpting
- `vox_ui` — retained-mode game UI framework

Game crates (city-builder-specific):
- `vox_sim` — city simulation: agents, citizens, economy, zoning, services, transport
- `vox_app` — binary entry point, game UI, editor

Tools:
- `vox_tools` — offline CLI tools (turnaround pipeline, map editor)
```

- [ ] **Step 3: Full build and test**

Run: `cargo build && cargo test`
Expected: all crates compile, all tests pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: complete Phase 0-4 scaffolding — all crates wired, full test suite passing"
```

---

## Summary

| Phase | Tasks | Status |
|---|---|---|
| Phase 0 (MVP) | Tasks 1–6 | Full implementation |
| Phase 1 (Foundation) | Tasks 7–12 | Full implementation |
| Phase 2 (Intelligence) | Tasks 13–14 | Working scaffold |
| Phase 3 (Scale) | Tasks 15–17 | Working scaffold |
| Phase 4 (Ship) | Tasks 18–20 | Interfaces + types + stubs |

**Total tasks: 20**
**Estimated parallel execution: ~7 hours with subagent-driven development**

After execution, the workspace has:
- 12 crates, all compiling
- Working software spectral rasteriser with 2-instance demo
- .vxm v0.1 and v0.2 file formats with zstd compression
- Proc-GS rule system with deterministic emitter
- 11 physically derived spectral materials
- Bevy ECS components, spatial hash, asset library
- SceneGraph types and stub LayoutInterpreter
- Weather/wear/time-of-day spectral shifts
- Large World Coordinates for 100km scale
- Tile streaming manager
- Agent simulation with movement
- Full city simulation type system (citizens, economy, zoning, services, transport)
- Networking, scripting, audio, physics, terrain, UI crate skeletons
