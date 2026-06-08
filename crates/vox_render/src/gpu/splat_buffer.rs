//! Persistent GPU splat buffer with free-list allocator.
//! `GpuSplatFull` is the std430 splat type for compute passes.

use bytemuck::{Pod, Zeroable};
use half::f16;
use vox_core::types::GaussianSplat;

/// Full GPU splat representation for compute passes — 80 bytes, std430 compatible.
///
/// Layout:
///   [0..16]  position_depth: xyz = world pos, w = view-space depth (written by tile_assign)
///   [16..28] conic: 2D EWA conic coefficients (written by tile_assign)
///   [28..32] _pad0
///   [32..48] opacity_color: w = opacity (0..1), xyz = reserved
///   [48..80] spectral: 8 spectral bands as f32
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuSplatFull {
    pub position_depth: [f32; 4],
    pub conic: [f32; 3],
    pub _pad0: f32,
    pub opacity_color: [f32; 4],
    pub spectral: [f32; 8],
}

const _: () = assert!(std::mem::size_of::<GpuSplatFull>() == 80);

pub struct SplatBufferAllocator {
    pub max_splats: usize,
    pub buffer: wgpu::Buffer,
    free_list: Vec<u32>,
    occupied: Vec<bool>,
    /// Next sequential slot to hand out when free_list is empty.
    next_sequential: u32,
}

impl SplatBufferAllocator {
    pub const MAX_SPLATS: usize = 8_000_000;

    /// Allocate the GPU storage buffer and initialise the allocator state.
    pub fn new(device: &wgpu::Device) -> Self {
        let byte_size = (Self::MAX_SPLATS * std::mem::size_of::<GpuSplatFull>()) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat_full_buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            max_splats: Self::MAX_SPLATS,
            buffer,
            free_list: Vec::new(),
            occupied: vec![false; Self::MAX_SPLATS],
            next_sequential: 0,
        }
    }

    /// Allocate a slot.  Returns the index, or `None` when the buffer is full.
    pub fn alloc(&mut self) -> Option<u32> {
        if let Some(idx) = self.free_list.pop() {
            self.occupied[idx as usize] = true;
            return Some(idx);
        }
        if (self.next_sequential as usize) < self.max_splats {
            let idx = self.next_sequential;
            self.next_sequential += 1;
            self.occupied[idx as usize] = true;
            return Some(idx);
        }
        None
    }

    /// Return a slot to the free list.
    pub fn free(&mut self, idx: u32) {
        debug_assert!((idx as usize) < self.max_splats, "splat index out of range");
        self.occupied[idx as usize] = false;
        self.free_list.push(idx);
    }

    /// Upload a single splat to the GPU buffer at the given slot.
    pub fn upload(&mut self, queue: &wgpu::Queue, idx: u32, splat: &GpuSplatFull) {
        let offset = idx as u64 * std::mem::size_of::<GpuSplatFull>() as u64;
        queue.write_buffer(&self.buffer, offset, bytemuck::bytes_of(splat));
    }

    /// Expose the underlying GPU buffer for binding.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Count of currently occupied slots.
    pub fn occupied_count(&self) -> usize {
        self.occupied.iter().filter(|&&v| v).count()
    }
}

/// Convert a `GaussianSplat` (as stored in .vxm) to the full GPU representation.
///
/// Fields filled here:
///   - `position_depth.xyz` = world position; `.w` = 0.0 (tile_assign will fill view-Z)
///   - `conic` = [0.0; 3] (tile_assign will compute from scale+quat)
///   - `opacity_color.w` = opacity / 255.0
///   - `spectral` = f16 bands unpacked to f32
pub fn gaussian_splat_to_gpu_full(s: &GaussianSplat) -> GpuSplatFull {
    let spectral = [
        f16::from_bits(s.spectral()[0]).to_f32(),
        f16::from_bits(s.spectral()[1]).to_f32(),
        f16::from_bits(s.spectral()[2]).to_f32(),
        f16::from_bits(s.spectral()[3]).to_f32(),
        f16::from_bits(s.spectral()[4]).to_f32(),
        f16::from_bits(s.spectral()[5]).to_f32(),
        f16::from_bits(s.spectral()[6]).to_f32(),
        f16::from_bits(s.spectral()[7]).to_f32(),
    ];

    GpuSplatFull {
        position_depth: [s.position()[0], s.position()[1], s.position()[2], 0.0],
        conic: [0.0; 3],
        _pad0: 0.0,
        opacity_color: [0.0, 0.0, 0.0, s.opacity() as f32 / 255.0],
        spectral,
    }
}

/// Pack per-splat transforms for `tile_assign`'s binding 6 (`splat_transforms`):
/// two `vec4<f32>` per splat — `[scale_u, scale_v, scale_w, 0]` then the decoded
/// rotation quaternion `[x, y, z, w]` (`tile_assign` builds the world covariance
/// from these). [`gaussian_splat_to_gpu_full`] does NOT emit them, so the tiled
/// renderer uploads this alongside the packed splats (both written once).
pub fn gaussian_splats_to_transforms(splats: &[GaussianSplat]) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(splats.len() * 2);
    for s in splats {
        let sc = s.scales();
        let q = s.decoded_rotation();
        out.push([sc[0], sc[1], sc[2], 0.0]);
        out.push([q.x, q.y, q.z, q.w]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_pack_scale_then_quat_xyzw() {
        use glam::Quat;
        let s = GaussianSplat::volume([1.0, 2.0, 3.0], [0.3, 0.5, 0.7], Quat::IDENTITY, 255, [0u16; 16]);
        let t = gaussian_splats_to_transforms(std::slice::from_ref(&s));
        assert_eq!(t.len(), 2, "two vec4 per splat");
        assert_eq!(t[0], [0.3, 0.5, 0.7, 0.0], "first vec4 = scale.xyz, 0");
        // identity quaternion is (x,y,z,w) = (0,0,0,1)
        let q = t[1];
        assert!(
            q[0].abs() < 1e-3 && q[1].abs() < 1e-3 && q[2].abs() < 1e-3 && (q[3] - 1.0).abs() < 1e-3,
            "second vec4 = identity quat xyzw (0,0,0,1), got {q:?}"
        );
    }

    #[test]
    fn size_is_80_bytes() {
        assert_eq!(std::mem::size_of::<GpuSplatFull>(), 80);
    }

    #[test]
    fn gpu_splat_full_is_pod() {
        let z = bytemuck::Zeroable::zeroed();
        let _: GpuSplatFull = z;
    }

    #[test]
    fn gaussian_splat_to_gpu_full_converts_correctly() {
        // Build a GaussianSplat with known values.
        let mut s = GaussianSplat::zeroed();
        s.set_position([1.0, 2.0, 3.0]);
        s.set_opacity(255);
        // Encode 1.0 as f16 in the first band.
        s.spectral_mut()[0] = half::f16::from_f32(1.0).to_bits();
        // Encode 0.5 in the second band.
        s.spectral_mut()[1] = half::f16::from_f32(0.5).to_bits();

        let gpu = gaussian_splat_to_gpu_full(&s);

        assert_eq!(gpu.position_depth[0], 1.0);
        assert_eq!(gpu.position_depth[1], 2.0);
        assert_eq!(gpu.position_depth[2], 3.0);
        assert_eq!(gpu.position_depth[3], 0.0, "depth must be 0 (set by tile_assign)");

        assert_eq!(gpu.conic, [0.0; 3], "conic must be 0 (set by tile_assign)");

        assert!((gpu.opacity_color[3] - 1.0).abs() < 1e-6, "opacity 255 → 1.0");

        assert!((gpu.spectral[0] - 1.0).abs() < 1e-3);
        assert!((gpu.spectral[1] - 0.5).abs() < 1e-3);
    }

    /// Test free-list logic without a real wgpu device by exercising the
    /// allocator fields directly (replacing the buffer with a no-op stand-in
    /// is not needed — we just test alloc/free state tracking).
    #[test]
    fn alloc_and_free_logic() {
        // We cannot construct a wgpu::Device in a unit test, so we verify the
        // free-list and occupied-vec logic through a lightweight mirror struct.
        struct FakeAllocator {
            max: usize,
            free_list: Vec<u32>,
            occupied: Vec<bool>,
            next: u32,
        }
        impl FakeAllocator {
            fn new(max: usize) -> Self {
                Self { max, free_list: Vec::new(), occupied: vec![false; max], next: 0 }
            }
            fn alloc(&mut self) -> Option<u32> {
                if let Some(idx) = self.free_list.pop() {
                    self.occupied[idx as usize] = true;
                    return Some(idx);
                }
                if (self.next as usize) < self.max {
                    let idx = self.next;
                    self.next += 1;
                    self.occupied[idx as usize] = true;
                    return Some(idx);
                }
                None
            }
            fn free(&mut self, idx: u32) {
                self.occupied[idx as usize] = false;
                self.free_list.push(idx);
            }
            fn occupied_count(&self) -> usize {
                self.occupied.iter().filter(|&&v| v).count()
            }
        }

        let mut a = FakeAllocator::new(4);
        let i0 = a.alloc().unwrap();
        let i1 = a.alloc().unwrap();
        assert_eq!(a.occupied_count(), 2);

        a.free(i0);
        assert_eq!(a.occupied_count(), 1);

        // Re-alloc should reuse i0 from the free list.
        let i2 = a.alloc().unwrap();
        assert_eq!(i2, i0);
        assert_eq!(a.occupied_count(), 2);

        // Fill up remaining two slots.
        let _ = a.alloc().unwrap();
        let _ = a.alloc().unwrap();
        assert_eq!(a.occupied_count(), 4);

        // Now allocator is full — free one and verify.
        a.free(i1);
        assert_eq!(a.occupied_count(), 3);
        let _ = a.alloc().unwrap();
        assert_eq!(a.occupied_count(), 4);

        // Should return None when truly full.
        assert!(a.alloc().is_none());
    }
}
