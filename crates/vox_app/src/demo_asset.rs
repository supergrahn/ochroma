use half::f16;
use uuid::Uuid;
use vox_core::types::GaussianSplat;
use vox_data::vxm::{MaterialType, VxmFile, VxmHeader};

/// Encode 8 spectral band values (f32) as [u16; 8] (each stored as f16 bits).
#[allow(dead_code)]
fn encode_spectral(bands: [f32; 8]) -> [u16; 8] {
    [
        f16::from_f32(bands[0]).to_bits(),
        f16::from_f32(bands[1]).to_bits(),
        f16::from_f32(bands[2]).to_bits(),
        f16::from_f32(bands[3]).to_bits(),
        f16::from_f32(bands[4]).to_bits(),
        f16::from_f32(bands[5]).to_bits(),
        f16::from_f32(bands[6]).to_bits(),
        f16::from_f32(bands[7]).to_bits(),
    ]
}

/// Brick-red SPD: low blue (380-460 nm), high red (620 nm peak).
/// Band wavelengths: [380, 420, 460, 500, 540, 580, 620, 660] nm.
#[allow(dead_code)]
fn brick_spd() -> [f32; 8] {
    // [380, 420, 460, 500, 540, 580, 620, 660]
    [0.04, 0.05, 0.06, 0.10, 0.18, 0.35, 0.65, 0.55]
}

/// Slate-grey SPD: flat reflectance ~0.15-0.20 across all bands.
#[allow(dead_code)]
fn slate_spd() -> [f32; 8] {
    [0.16, 0.16, 0.17, 0.17, 0.18, 0.18, 0.19, 0.19]
}

/// Build a single GaussianSplat at world position (x, y, z).
#[allow(dead_code)]
fn make_splat(x: f32, y: f32, z: f32, spd: [f32; 8]) -> GaussianSplat {
    GaussianSplat {
        position: [x, y, z],
        // ~0.5 m radius per splat
        scale: [0.5, 0.5, 0.5],
        // Identity quaternion encoded as i16 (w = 1.0 → 32767)
        rotation: [0, 0, 0, 32767],
        opacity: 220,
        _pad: [0u8; 3],
        spectral: encode_spectral(spd),
    }
}

/// Generate a synthetic test building as a VxmFile.
///
/// Geometry (origin at bottom-front-left corner):
/// - Front wall : 20 columns x 15 rows, XY plane at Z=0
/// - Side wall  : 12 columns x 15 rows, YZ plane at X=0
/// - Roof       : 20 columns x 12 rows, XZ plane at Y=15
#[allow(dead_code)]
pub fn generate_building() -> VxmFile {
    let brick = brick_spd();
    let slate = slate_spd();
    let mut splats: Vec<GaussianSplat> = Vec::new();

    // --- Front wall (XY plane, z = 0) ---
    for row in 0..15u32 {
        for col in 0..20u32 {
            let x = col as f32 * 1.0 + 0.5;
            let y = row as f32 * 1.0 + 0.5;
            let z = 0.0_f32;
            splats.push(make_splat(x, y, z, brick));
        }
    }

    // --- Side wall (YZ plane, x = 0) ---
    for row in 0..15u32 {
        for col in 0..12u32 {
            let x = 0.0_f32;
            let y = row as f32 * 1.0 + 0.5;
            let z = col as f32 * 1.0 + 0.5;
            splats.push(make_splat(x, y, z, brick));
        }
    }

    // --- Roof (XZ plane, y = 15) ---
    for row in 0..12u32 {
        for col in 0..20u32 {
            let x = col as f32 * 1.0 + 0.5;
            let y = 15.0_f32;
            let z = row as f32 * 1.0 + 0.5;
            splats.push(make_splat(x, y, z, slate));
        }
    }

    let uuid = Uuid::new_v4();
    let header = VxmHeader::new(uuid, splats.len() as u32, MaterialType::Concrete);
    VxmFile { header, splats }
}
