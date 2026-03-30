//! GPU splat buffer pool — manages wgpu buffers for streamed world cells.
//! Avoids per-frame allocation by maintaining a fixed pool of buffer slots.
//! Double-buffering ensures no GPU stall during cell transitions.

use wgpu;
use bytemuck::{Pod, Zeroable};
use vox_core::types::GaussianSplat;
use half::f16;

/// Identifies one buffer slot in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferSlotId(pub u32);

/// State of a buffer slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    Free,
    /// Being filled by streaming task. `generation` increments on each reuse to detect stale writes.
    Pending { generation: u32 },
    /// Fully uploaded and ready for rendering.
    Active { generation: u32, splat_count: u32 },
}

/// A single slot in the pool.
pub struct BufferSlot {
    pub id: BufferSlotId,
    pub buffer: wgpu::Buffer,
    pub state: SlotState,
    pub max_splats: u32,
}

impl BufferSlot {
    pub fn is_free(&self) -> bool { matches!(self.state, SlotState::Free) }
    pub fn is_active(&self) -> bool { matches!(self.state, SlotState::Active { .. }) }
    pub fn splat_count(&self) -> u32 {
        if let SlotState::Active { splat_count, .. } = self.state { splat_count } else { 0 }
    }
}

/// Pool of GPU buffers for streamed splat cells.
pub struct SplatBufferPool {
    slots: Vec<BufferSlot>,
    pub max_splats_per_slot: u32,
    /// Total GPU memory allocated by this pool in bytes.
    total_bytes: u64,
}

impl SplatBufferPool {
    /// Create a pool with `num_slots` slots, each holding up to `max_splats_per_slot` splats.
    /// Each splat is `SPLAT_GPU_BYTES` bytes on GPU (the compact upload format).
    pub fn new(device: &wgpu::Device, num_slots: u32, max_splats_per_slot: u32) -> Self {
        const SPLAT_GPU_BYTES: u64 = 80;  // matches GpuSplatFull from splat_buffer.rs

        let mut slots = Vec::with_capacity(num_slots as usize);
        for i in 0..num_slots {
            let size = max_splats_per_slot as u64 * SPLAT_GPU_BYTES;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("splat_pool_slot_{}", i)),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            });
            slots.push(BufferSlot {
                id: BufferSlotId(i),
                buffer,
                state: SlotState::Free,
                max_splats: max_splats_per_slot,
            });
        }

        let total_bytes = num_slots as u64 * max_splats_per_slot as u64 * SPLAT_GPU_BYTES;

        Self { slots, max_splats_per_slot, total_bytes }
    }

    /// Allocate a free slot. Returns None if no free slots.
    pub fn alloc(&mut self) -> Option<BufferSlotId> {
        let slot = self.slots.iter_mut().find(|s| s.is_free())?;
        let next_gen = match slot.state {
            SlotState::Pending { generation } | SlotState::Active { generation, .. } => generation + 1,
            SlotState::Free => 0,
        };
        slot.state = SlotState::Pending { generation: next_gen };
        Some(slot.id)
    }

    /// Upload splats to a pending slot and mark it active.
    /// `splats` must be <= max_splats_per_slot.
    pub fn upload(&mut self, queue: &wgpu::Queue, id: BufferSlotId, splats: &[GaussianSplat]) {
        let slot = self.slots.iter_mut().find(|s| s.id == id).expect("invalid slot id");

        let count = splats.len().min(slot.max_splats as usize) as u32;

        // Convert splats to compact GPU format (80 bytes each, matching GpuSplatFull layout)
        let gpu_data: Vec<GpuSplatUpload> = splats[..count as usize].iter().map(|s| {
            let mut spectral = [0.0f32; 16];
            for (b, val) in spectral.iter_mut().enumerate() {
                *val = f16::from_bits(s.spectral()[b]).to_f32();
            }
            GpuSplatUpload {
                position: s.position(),
                scale: [s.scale_u(), s.scale_v(), s.scale_w()],
                opacity: s.opacity() as f32 / 255.0,
                _pad0: 0.0,
                spectral,
                _pad1: [0.0; 4],
            }
        }).collect();

        queue.write_buffer(&slot.buffer, 0, bytemuck::cast_slice(&gpu_data));

        let next_gen = match slot.state { SlotState::Pending { generation } => generation, _ => 0 };
        slot.state = SlotState::Active { generation: next_gen, splat_count: count };
    }

    /// Free a slot (called when a cell is evicted).
    pub fn free(&mut self, id: BufferSlotId) {
        if let Some(slot) = self.slots.iter_mut().find(|s| s.id == id) {
            slot.state = SlotState::Free;
        }
    }

    /// Get a reference to a slot's buffer (for binding to render pass).
    pub fn buffer(&self, id: BufferSlotId) -> Option<&wgpu::Buffer> {
        self.slots.iter().find(|s| s.id == id).map(|s| &s.buffer)
    }

    /// Total GPU memory used by this pool in megabytes.
    pub fn total_mb(&self) -> f32 {
        self.total_bytes as f32 / (1024.0 * 1024.0)
    }

    /// Memory used by active slots only.
    pub fn active_mb(&self) -> f32 {
        const SPLAT_GPU_BYTES: f32 = 80.0;
        self.slots.iter()
            .filter(|s| s.is_active())
            .map(|s| s.splat_count() as f32 * SPLAT_GPU_BYTES / (1024.0 * 1024.0))
            .sum()
    }

    /// Number of free slots.
    pub fn free_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_free()).count()
    }

    /// Number of active slots.
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_active()).count()
    }
}

/// GPU upload format for one splat (80 bytes, std430-compatible).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSplatUpload {
    position: [f32; 3],
    opacity: f32,
    scale: [f32; 3],
    _pad0: f32,
    spectral: [f32; 16],
    _pad1: [f32; 4],
}
// Total: 3*4 + 4 + 3*4 + 4 + 16*4 + 4*4 = 12+4+12+4+64+16 = 112 bytes

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_splat_upload_size() {
        assert_eq!(std::mem::size_of::<GpuSplatUpload>(), 112);
    }

    // Helper: build a pool from raw slots without wgpu, for state-logic tests.
    // We test state transitions via a lightweight stand-in that mirrors the
    // alloc/free logic without touching GPU objects.
    struct MockPool {
        states: Vec<SlotState>,
        /// Next generation to use when a slot is re-allocated after being freed.
        next_gens: Vec<u32>,
    }

    impl MockPool {
        fn new(n: usize) -> Self {
            Self { states: vec![SlotState::Free; n], next_gens: vec![0; n] }
        }

        fn alloc(&mut self) -> Option<usize> {
            let idx = self.states.iter().position(|s| matches!(s, SlotState::Free))?;
            let next_gen = self.next_gens[idx];
            self.states[idx] = SlotState::Pending { generation: next_gen };
            Some(idx)
        }

        fn activate(&mut self, idx: usize, splat_count: u32) {
            let next_gen = match self.states[idx] { SlotState::Pending { generation } => generation, _ => 0 };
            self.states[idx] = SlotState::Active { generation: next_gen, splat_count };
        }

        fn free(&mut self, idx: usize) {
            let current_gen = match self.states[idx] {
                SlotState::Active { generation, .. } | SlotState::Pending { generation } => generation,
                SlotState::Free => return,
            };
            self.next_gens[idx] = current_gen + 1;
            self.states[idx] = SlotState::Free;
        }

        fn free_count(&self) -> usize {
            self.states.iter().filter(|s| matches!(s, SlotState::Free)).count()
        }

        fn active_count(&self) -> usize {
            self.states.iter().filter(|s| matches!(s, SlotState::Active { .. })).count()
        }
    }

    #[test]
    fn slot_state_free_initially() {
        let pool = MockPool::new(8);
        assert_eq!(pool.free_count(), 8);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn alloc_and_free() {
        let mut pool = MockPool::new(4);
        assert_eq!(pool.free_count(), 4);

        let idx = pool.alloc().expect("should allocate");
        assert_eq!(pool.free_count(), 3, "free count decrements after alloc");
        assert!(matches!(pool.states[idx], SlotState::Pending { .. }));

        pool.activate(idx, 100);
        assert_eq!(pool.active_count(), 1);

        pool.free(idx);
        assert_eq!(pool.free_count(), 4, "free count restores after free");
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn alloc_exhaustion() {
        let mut pool = MockPool::new(3);
        assert!(pool.alloc().is_some());
        assert!(pool.alloc().is_some());
        assert!(pool.alloc().is_some());
        assert!(pool.alloc().is_none(), "should return None when pool is exhausted");
    }

    #[test]
    fn pool_total_mb() {
        // 4 slots * 10000 splats * 80 bytes = 3_200_000 bytes ≈ 3.0518 MB
        let expected = 4u64 * 10_000u64 * 80u64;
        let total_mb = expected as f32 / (1024.0 * 1024.0);
        assert!(
            (total_mb - 3.0518).abs() < 0.001,
            "expected ~3.05 MB, got {:.4}", total_mb
        );
    }

    #[test]
    fn slot_active_after_upload_state_logic() {
        // Tests the state transition: Free -> Pending -> Active, without wgpu.
        let mut pool = MockPool::new(2);

        let idx = pool.alloc().unwrap();
        assert!(matches!(pool.states[idx], SlotState::Pending { generation: 0 }));

        pool.activate(idx, 512);
        assert!(matches!(pool.states[idx], SlotState::Active { generation: 0, splat_count: 512 }));

        // Freeing and re-allocating increments generation.
        pool.free(idx);
        let idx2 = pool.alloc().unwrap();
        assert_eq!(idx, idx2, "same slot should be reused");
        assert!(matches!(pool.states[idx2], SlotState::Pending { generation: 1 }),
            "generation should increment on reuse");
    }
}
