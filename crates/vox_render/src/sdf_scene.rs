//! Render-side packing for resident signed-distance-field assets.
//!
//! `vox_physics` owns SDF sampling and collision. This module provides the
//! stable, flat representation a renderer can upload: deduplicated volume
//! distance grids plus per-instance transforms and world bounds.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use std::collections::HashMap;
use vox_physics::ecs::SdfColliderRuntime;
use vox_physics::sdf::{SdfAssetId, SdfColliderSet, SdfInstanceId};

#[derive(Debug, Clone, PartialEq)]
pub struct RenderSdfScene {
    pub volumes: Vec<RenderSdfVolume>,
    pub instances: Vec<RenderSdfInstance>,
    pub distances: Vec<f32>,
}

impl RenderSdfScene {
    pub fn empty() -> Self {
        Self {
            volumes: Vec::new(),
            instances: Vec::new(),
            distances: Vec::new(),
        }
    }

    pub fn from_runtime(runtime: &SdfColliderRuntime) -> Self {
        Self::from_collider_set(runtime.colliders())
    }

    pub fn from_collider_set(colliders: &SdfColliderSet) -> Self {
        let mut scene = Self::empty();
        let mut volume_indices = HashMap::<SdfAssetId, u32>::new();

        for (instance, &(world_aabb_min, world_aabb_max)) in
            colliders.instances().iter().zip(colliders.world_aabbs())
        {
            let volume_index = if let Some(&index) = volume_indices.get(&instance.asset_id) {
                index
            } else {
                let volume = &instance.volume;
                let desc = volume.desc();
                let distance_offset = scene.distances.len() as u32;
                let distance_count = volume.distances().len() as u32;
                scene.distances.extend_from_slice(volume.distances());

                let index = scene.volumes.len() as u32;
                scene.volumes.push(RenderSdfVolume {
                    asset_id: instance.asset_id,
                    resolution: desc.resolution,
                    origin: desc.origin.to_array(),
                    voxel_size: desc.voxel_size,
                    narrow_band: desc.narrow_band,
                    distance_offset,
                    distance_count,
                });
                volume_indices.insert(instance.asset_id, index);
                index
            };

            scene.instances.push(RenderSdfInstance {
                instance_id: instance.id,
                asset_id: instance.asset_id,
                volume_index,
                position: vec3_to_array(instance.position),
                rotation_xyzw: quat_to_array(instance.rotation),
                uniform_scale: instance.uniform_scale,
                world_aabb_min: vec3_to_array(world_aabb_min),
                world_aabb_max: vec3_to_array(world_aabb_max),
            });
        }

        scene
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    pub fn volume_count(&self) -> usize {
        self.volumes.len()
    }

    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    pub fn distance_count(&self) -> usize {
        self.distances.len()
    }

    pub fn gpu_volume_headers(&self) -> Vec<GpuSdfVolumeHeader> {
        self.volumes
            .iter()
            .map(RenderSdfVolume::to_gpu_header)
            .collect()
    }

    pub fn gpu_instance_headers(&self) -> Vec<GpuSdfInstanceHeader> {
        self.instances
            .iter()
            .map(RenderSdfInstance::to_gpu_header)
            .collect()
    }

    pub fn volume_for_instance(&self, instance_index: usize) -> Option<&RenderSdfVolume> {
        let volume_index = self.instances.get(instance_index)?.volume_index as usize;
        self.volumes.get(volume_index)
    }

    #[cfg(feature = "spectra-native")]
    pub fn to_spectra_sdf_layer(&self) -> spectra_scene_state::SdfLayer {
        let volumes = self
            .volumes
            .iter()
            .map(|volume| spectra_scene_state::SdfVolumeHeader {
                asset_id: volume.asset_id.0,
                resolution: volume.resolution,
                origin: volume.origin,
                voxel_size: volume.voxel_size,
                narrow_band: volume.narrow_band,
                distance_offset: volume.distance_offset,
                distance_count: volume.distance_count,
            })
            .collect();
        let instances = self
            .instances
            .iter()
            .map(|instance| spectra_scene_state::SdfInstanceHeader {
                instance_id: instance.instance_id.0,
                asset_id: instance.asset_id.0,
                volume_index: instance.volume_index,
                position: instance.position,
                rotation_xyzw: instance.rotation_xyzw,
                uniform_scale: instance.uniform_scale,
                world_aabb_min: instance.world_aabb_min,
                world_aabb_max: instance.world_aabb_max,
            })
            .collect();

        spectra_scene_state::SdfLayer::from_parts(volumes, instances, self.distances.clone())
    }

    #[cfg(feature = "spectra-native")]
    pub fn write_to_spectra_scene_state(&self, state: &mut spectra_scene_state::SceneState) {
        state.sdf = self.to_spectra_sdf_layer();
        state.mark_sdf_changed();
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderSdfVolume {
    pub asset_id: SdfAssetId,
    pub resolution: [u32; 3],
    pub origin: [f32; 3],
    pub voxel_size: f32,
    pub narrow_band: f32,
    pub distance_offset: u32,
    pub distance_count: u32,
}

impl RenderSdfVolume {
    pub fn to_gpu_header(&self) -> GpuSdfVolumeHeader {
        GpuSdfVolumeHeader {
            origin_voxel_size: [
                self.origin[0],
                self.origin[1],
                self.origin[2],
                self.voxel_size,
            ],
            narrow_band_pad: [self.narrow_band, 0.0, 0.0, 0.0],
            resolution_distance_offset: [
                self.resolution[0],
                self.resolution[1],
                self.resolution[2],
                self.distance_offset,
            ],
            distance_count_pad: [self.distance_count, 0, 0, 0],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderSdfInstance {
    pub instance_id: SdfInstanceId,
    pub asset_id: SdfAssetId,
    pub volume_index: u32,
    pub position: [f32; 3],
    pub rotation_xyzw: [f32; 4],
    pub uniform_scale: f32,
    pub world_aabb_min: [f32; 3],
    pub world_aabb_max: [f32; 3],
}

impl RenderSdfInstance {
    pub fn to_gpu_header(&self) -> GpuSdfInstanceHeader {
        GpuSdfInstanceHeader {
            position_scale: [
                self.position[0],
                self.position[1],
                self.position[2],
                self.uniform_scale,
            ],
            rotation_xyzw: self.rotation_xyzw,
            world_aabb_min: [
                self.world_aabb_min[0],
                self.world_aabb_min[1],
                self.world_aabb_min[2],
                0.0,
            ],
            world_aabb_max: [
                self.world_aabb_max[0],
                self.world_aabb_max[1],
                self.world_aabb_max[2],
                0.0,
            ],
            ids: [
                self.volume_index,
                self.instance_id.0 as u32,
                self.asset_id.0 as u32,
                0,
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuSdfVolumeHeader {
    pub origin_voxel_size: [f32; 4],
    pub narrow_band_pad: [f32; 4],
    pub resolution_distance_offset: [u32; 4],
    pub distance_count_pad: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuSdfInstanceHeader {
    pub position_scale: [f32; 4],
    pub rotation_xyzw: [f32; 4],
    pub world_aabb_min: [f32; 4],
    pub world_aabb_max: [f32; 4],
    pub ids: [u32; 4],
}

fn vec3_to_array(value: Vec3) -> [f32; 3] {
    value.to_array()
}

fn quat_to_array(value: Quat) -> [f32; 4] {
    value.to_array()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use vox_physics::sdf::{SdfColliderSet, SdfVolume, SdfVolumeDesc};

    #[test]
    fn render_sdf_scene_deduplicates_shared_asset_volume() {
        let asset_id = SdfAssetId(7);
        let volume = test_box_volume(Vec3::splat(0.5));
        let mut colliders = SdfColliderSet::new();
        colliders.add_instance(
            asset_id,
            Arc::clone(&volume),
            Vec3::new(1.0, 2.0, 3.0),
            Quat::IDENTITY,
            2.0,
        );
        colliders.add_instance(
            asset_id,
            Arc::clone(&volume),
            Vec3::new(6.0, 0.0, -1.0),
            Quat::from_rotation_y(0.4),
            1.0,
        );

        let scene = RenderSdfScene::from_collider_set(&colliders);

        assert_eq!(scene.volume_count(), 1);
        assert_eq!(scene.instance_count(), 2);
        assert_eq!(scene.distance_count(), volume.distances().len());
        assert_eq!(scene.distances, volume.distances());
        assert_eq!(scene.instances[0].volume_index, 0);
        assert_eq!(scene.instances[1].volume_index, 0);
        assert_eq!(scene.volume_for_instance(1), Some(&scene.volumes[0]));
    }

    #[test]
    fn render_sdf_scene_preserves_physics_world_aabbs() {
        let asset_id = SdfAssetId(9);
        let volume = test_box_volume(Vec3::new(0.5, 0.75, 1.0));
        let mut colliders = SdfColliderSet::new();
        colliders.add_instance(
            asset_id,
            volume,
            Vec3::new(2.0, -1.0, 3.0),
            Quat::from_rotation_z(0.25),
            1.5,
        );

        let scene = RenderSdfScene::from_collider_set(&colliders);
        let (aabb_min, aabb_max) = colliders.world_aabbs()[0];

        assert_eq!(scene.instances[0].world_aabb_min, aabb_min.to_array());
        assert_eq!(scene.instances[0].world_aabb_max, aabb_max.to_array());
    }

    #[test]
    fn render_sdf_scene_produces_pod_gpu_headers() {
        let asset_id = SdfAssetId(11);
        let volume = test_box_volume(Vec3::splat(0.5));
        let mut colliders = SdfColliderSet::new();
        colliders.add_instance(
            asset_id,
            volume,
            Vec3::new(1.0, 2.0, 3.0),
            Quat::IDENTITY,
            1.25,
        );

        let scene = RenderSdfScene::from_collider_set(&colliders);
        let volume_headers = scene.gpu_volume_headers();
        let instance_headers = scene.gpu_instance_headers();

        assert_eq!(volume_headers.len(), 1);
        assert_eq!(instance_headers.len(), 1);
        assert_eq!(volume_headers[0].resolution_distance_offset, [3, 3, 3, 0]);
        assert_eq!(volume_headers[0].distance_count_pad[0], 27);
        assert_eq!(instance_headers[0].position_scale, [1.0, 2.0, 3.0, 1.25]);
        assert_eq!(instance_headers[0].ids[0], 0);
        let _: &[u8] = bytemuck::cast_slice(&volume_headers);
        let _: &[u8] = bytemuck::cast_slice(&instance_headers);
    }

    #[test]
    fn empty_render_sdf_scene_has_no_buffers() {
        let scene = RenderSdfScene::from_collider_set(&SdfColliderSet::new());

        assert!(scene.is_empty());
        assert!(scene.gpu_volume_headers().is_empty());
        assert!(scene.gpu_instance_headers().is_empty());
        assert!(scene.distances.is_empty());
    }

    #[cfg(feature = "spectra-native")]
    #[test]
    fn render_sdf_scene_writes_spectra_scene_state_layer() {
        let asset_id = SdfAssetId(17);
        let volume = test_box_volume(Vec3::splat(0.5));
        let mut colliders = SdfColliderSet::new();
        colliders.add_instance(asset_id, volume, Vec3::X, Quat::IDENTITY, 1.0);

        let scene = RenderSdfScene::from_collider_set(&colliders);
        let mut spectra_state = spectra_scene_state::SceneState::new(64, 64);
        spectra_state.mark_uploaded();
        scene.write_to_spectra_scene_state(&mut spectra_state);

        assert_eq!(spectra_state.sdf.volume_count, 1);
        assert_eq!(spectra_state.sdf.instance_count, 1);
        assert_eq!(spectra_state.sdf.distances.len(), scene.distances.len());
        assert!(spectra_state.dirty.is_sdf_dirty());
        assert!(!spectra_state.dirty.is_geometry_dirty());
    }

    fn test_box_volume(half_extents: Vec3) -> Arc<SdfVolume> {
        let resolution = [3, 3, 3];
        let origin = Vec3::splat(-1.0);
        let voxel_size = 1.0;
        let narrow_band = 2.0;
        let mut distances = Vec::new();
        for z in 0..resolution[2] {
            for y in 0..resolution[1] {
                for x in 0..resolution[0] {
                    let point = origin + Vec3::new(x as f32, y as f32, z as f32) * voxel_size;
                    let q = point.abs() - half_extents;
                    let outside = q.max(Vec3::ZERO).length();
                    let inside = q.x.max(q.y.max(q.z)).min(0.0);
                    distances.push(outside + inside);
                }
            }
        }

        Arc::new(
            SdfVolume::from_f32_grid(
                SdfVolumeDesc {
                    resolution,
                    origin,
                    voxel_size,
                    narrow_band,
                },
                distances,
            )
            .unwrap(),
        )
    }
}
