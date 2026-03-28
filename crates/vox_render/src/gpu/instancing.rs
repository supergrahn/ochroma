use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Quat};
use std::collections::HashMap;
use uuid::Uuid;

/// Per-instance transform data sent to GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct InstanceTransform {
    pub model_matrix: [[f32; 4]; 4],
    pub instance_id: u32,
    pub lod_level: u32,
    pub _pad: [u32; 2],
}

/// A batch of instances sharing the same asset.
#[derive(Debug)]
pub struct InstanceBatch {
    pub asset_uuid: Uuid,
    pub splat_count: u32,
    pub instances: Vec<InstanceTransform>,
}

/// Manages GPU instancing — groups instances by asset for batched drawing.
pub struct InstanceManager {
    batches: HashMap<Uuid, InstanceBatch>,
}

impl Default for InstanceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl InstanceManager {
    pub fn new() -> Self {
        Self { batches: HashMap::new() }
    }

    /// Register an asset with its splat count.
    pub fn register_asset(&mut self, uuid: Uuid, splat_count: u32) {
        self.batches.entry(uuid).or_insert_with(|| InstanceBatch {
            asset_uuid: uuid,
            splat_count,
            instances: Vec::new(),
        });
    }

    /// Add an instance of an asset.
    pub fn add_instance(&mut self, uuid: Uuid, position: Vec3, rotation: Quat, scale: f32, instance_id: u32, lod_level: u32) {
        if let Some(batch) = self.batches.get_mut(&uuid) {
            let model = Mat4::from_scale_rotation_translation(
                Vec3::splat(scale),
                rotation,
                position,
            );
            batch.instances.push(InstanceTransform {
                model_matrix: model.to_cols_array_2d(),
                instance_id,
                lod_level,
                _pad: [0; 2],
            });
        }
    }

    /// Clear all instances (called each frame before re-gathering).
    pub fn clear_instances(&mut self) {
        for batch in self.batches.values_mut() {
            batch.instances.clear();
        }
    }

    /// Get all batches with at least one instance.
    pub fn active_batches(&self) -> Vec<&InstanceBatch> {
        self.batches.values().filter(|b| !b.instances.is_empty()).collect()
    }

    /// Total instance count across all batches.
    pub fn total_instances(&self) -> usize {
        self.batches.values().map(|b| b.instances.len()).sum()
    }

    /// Total splat count (instances x splats per asset).
    pub fn total_splats(&self) -> u64 {
        self.batches.values()
            .map(|b| b.instances.len() as u64 * b.splat_count as u64)
            .sum()
    }

    /// Memory savings: how many splats would be needed without instancing.
    pub fn memory_savings_ratio(&self) -> f32 {
        let with_instancing: u64 = self.batches.values()
            .map(|b| b.splat_count as u64 + b.instances.len() as u64 * 80) // 80 bytes per transform
            .sum();
        let without_instancing = self.total_splats() * 52; // 52 bytes per splat

        if without_instancing == 0 { return 1.0; }
        with_instancing as f32 / without_instancing as f32
    }
}
