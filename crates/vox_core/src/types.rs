use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use half::f16;
use uuid::Uuid;

/// A single Gaussian splat as stored in .vxm v0.1 (52 bytes).
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GaussianSplat {
    pub position: [f32; 3],
    pub scale: [f32; 3],
    pub rotation: [i16; 4],
    pub opacity: u8,
    pub _pad: [u8; 3],
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
    pub fn decoded_rotation(&self) -> Quat {
        Quat::from_xyzw(
            self.rotation[0] as f32 / 32767.0,
            self.rotation[1] as f32 / 32767.0,
            self.rotation[2] as f32 / 32767.0,
            self.rotation[3] as f32 / 32767.0,
        )
    }

    pub fn spectral_f32(&self, band: usize) -> f32 {
        f16::from_bits(self.spectral[band]).to_f32()
    }
}
