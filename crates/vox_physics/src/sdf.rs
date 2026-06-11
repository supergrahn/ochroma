//! CPU signed-distance-field queries for physics and placement.
//!
//! SDFs are stored asset-local and instanced into world space. Rapier remains
//! the rigid-body backend; this module provides sampled collision/projection
//! queries for particles, soft bodies, placement checks, and debug raymarching.

use glam::{Quat, Vec3};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

// Canonical SDF representation now lives in vox_core::sdf (SDF-pillar design
// §5). Re-exported here additively so the query layer and the canonical types
// share one import path; `SdfVolume` et al. below are untouched this wave
// (the full `SdfVolume = SdfField` aliasing is a later wave).
pub use vox_core::sdf::{SdfDesc, SdfField, SdfPayload, SdfSign, SdfValidation};

const MIN_GRADIENT_EPS: f32 = 1.0e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SdfAssetId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SdfInstanceId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdfVolumeDesc {
    pub resolution: [u32; 3],
    pub origin: Vec3,
    pub voxel_size: f32,
    pub narrow_band: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SdfVolume {
    desc: SdfVolumeDesc,
    distances: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SdfVolumeError {
    ResolutionTooSmall([u32; 3]),
    InvalidVoxelSize,
    InvalidNarrowBand,
    DistanceCountMismatch { expected: usize, actual: usize },
}

impl fmt::Display for SdfVolumeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResolutionTooSmall(resolution) => {
                write!(
                    f,
                    "SDF resolution must be at least 2 on every axis: {resolution:?}"
                )
            }
            Self::InvalidVoxelSize => write!(f, "SDF voxel_size must be positive and finite"),
            Self::InvalidNarrowBand => write!(f, "SDF narrow_band must be positive and finite"),
            Self::DistanceCountMismatch { expected, actual } => write!(
                f,
                "SDF distance count mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for SdfVolumeError {}

impl SdfVolume {
    pub fn from_f32_grid(desc: SdfVolumeDesc, distances: Vec<f32>) -> Result<Self, SdfVolumeError> {
        validate_desc(desc, distances.len())?;
        Ok(Self { desc, distances })
    }

    pub fn from_snorm16_grid(
        desc: SdfVolumeDesc,
        distances_snorm16: &[i16],
    ) -> Result<Self, SdfVolumeError> {
        validate_desc(desc, distances_snorm16.len())?;
        let distances = distances_snorm16
            .iter()
            .map(|&value| {
                let normalized = if value == i16::MIN {
                    -1.0
                } else {
                    value as f32 / i16::MAX as f32
                };
                normalized.clamp(-1.0, 1.0) * desc.narrow_band
            })
            .collect();
        Ok(Self { desc, distances })
    }

    pub fn desc(&self) -> SdfVolumeDesc {
        self.desc
    }

    pub fn distances(&self) -> &[f32] {
        &self.distances
    }

    pub fn local_bounds(&self) -> (Vec3, Vec3) {
        let max = self.desc.origin
            + Vec3::new(
                (self.desc.resolution[0] - 1) as f32 * self.desc.voxel_size,
                (self.desc.resolution[1] - 1) as f32 * self.desc.voxel_size,
                (self.desc.resolution[2] - 1) as f32 * self.desc.voxel_size,
            );
        (self.desc.origin, max)
    }

    pub fn sample_local(&self, local_point: Vec3) -> Option<f32> {
        let grid = (local_point - self.desc.origin) / self.desc.voxel_size;
        if grid.x < 0.0
            || grid.y < 0.0
            || grid.z < 0.0
            || grid.x > (self.desc.resolution[0] - 1) as f32
            || grid.y > (self.desc.resolution[1] - 1) as f32
            || grid.z > (self.desc.resolution[2] - 1) as f32
        {
            return None;
        }

        let (x0, x1, tx) = axis_sample(grid.x, self.desc.resolution[0]);
        let (y0, y1, ty) = axis_sample(grid.y, self.desc.resolution[1]);
        let (z0, z1, tz) = axis_sample(grid.z, self.desc.resolution[2]);

        let c000 = self.distance_at(x0, y0, z0);
        let c100 = self.distance_at(x1, y0, z0);
        let c010 = self.distance_at(x0, y1, z0);
        let c110 = self.distance_at(x1, y1, z0);
        let c001 = self.distance_at(x0, y0, z1);
        let c101 = self.distance_at(x1, y0, z1);
        let c011 = self.distance_at(x0, y1, z1);
        let c111 = self.distance_at(x1, y1, z1);

        let c00 = lerp(c000, c100, tx);
        let c10 = lerp(c010, c110, tx);
        let c01 = lerp(c001, c101, tx);
        let c11 = lerp(c011, c111, tx);
        let c0 = lerp(c00, c10, ty);
        let c1 = lerp(c01, c11, ty);
        Some(lerp(c0, c1, tz))
    }

    pub fn normal_local(&self, local_point: Vec3) -> Option<Vec3> {
        self.sample_local(local_point)?;
        let eps = self.desc.voxel_size.max(MIN_GRADIENT_EPS);
        let center = self.sample_local(local_point)?;
        let dx = sample_or_center(self, local_point + Vec3::X * eps, center)
            - sample_or_center(self, local_point - Vec3::X * eps, center);
        let dy = sample_or_center(self, local_point + Vec3::Y * eps, center)
            - sample_or_center(self, local_point - Vec3::Y * eps, center);
        let dz = sample_or_center(self, local_point + Vec3::Z * eps, center)
            - sample_or_center(self, local_point - Vec3::Z * eps, center);
        let gradient = Vec3::new(dx, dy, dz);
        (gradient.length_squared() > MIN_GRADIENT_EPS * MIN_GRADIENT_EPS)
            .then(|| gradient.normalize())
    }

    fn distance_at(&self, x: u32, y: u32, z: u32) -> f32 {
        let [sx, sy, _] = self.desc.resolution;
        let index = (x + sx * (y + sy * z)) as usize;
        self.distances[index]
    }
}

#[derive(Debug, Clone, Default)]
pub struct SdfVolumeCache {
    volumes: HashMap<SdfAssetId, Arc<SdfVolume>>,
}

impl SdfVolumeCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.volumes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.volumes.is_empty()
    }

    pub fn clear(&mut self) {
        self.volumes.clear();
    }

    pub fn get(&self, asset_id: SdfAssetId) -> Option<Arc<SdfVolume>> {
        self.volumes.get(&asset_id).map(Arc::clone)
    }

    pub fn insert(
        &mut self,
        asset_id: SdfAssetId,
        volume: Arc<SdfVolume>,
    ) -> Option<Arc<SdfVolume>> {
        self.volumes.insert(asset_id, volume)
    }

    pub fn get_or_insert_f32_grid(
        &mut self,
        asset_id: SdfAssetId,
        desc: SdfVolumeDesc,
        distances: Vec<f32>,
    ) -> Result<Arc<SdfVolume>, SdfVolumeError> {
        if let Some(volume) = self.get(asset_id) {
            return Ok(volume);
        }
        let volume = Arc::new(SdfVolume::from_f32_grid(desc, distances)?);
        self.volumes.insert(asset_id, Arc::clone(&volume));
        Ok(volume)
    }

    pub fn get_or_insert_snorm16_grid(
        &mut self,
        asset_id: SdfAssetId,
        desc: SdfVolumeDesc,
        distances_snorm16: &[i16],
    ) -> Result<Arc<SdfVolume>, SdfVolumeError> {
        if let Some(volume) = self.get(asset_id) {
            return Ok(volume);
        }
        let volume = Arc::new(SdfVolume::from_snorm16_grid(desc, distances_snorm16)?);
        self.volumes.insert(asset_id, Arc::clone(&volume));
        Ok(volume)
    }
}

#[derive(Debug, Clone)]
pub struct SdfInstance {
    pub id: SdfInstanceId,
    pub asset_id: SdfAssetId,
    pub volume: Arc<SdfVolume>,
    pub position: Vec3,
    pub rotation: Quat,
    pub uniform_scale: f32,
}

impl SdfInstance {
    pub fn new(
        id: SdfInstanceId,
        asset_id: SdfAssetId,
        volume: Arc<SdfVolume>,
        position: Vec3,
        rotation: Quat,
        uniform_scale: f32,
    ) -> Self {
        Self {
            id,
            asset_id,
            volume,
            position,
            rotation,
            uniform_scale: uniform_scale.max(MIN_GRADIENT_EPS),
        }
    }

    pub fn sample_world(&self, world_point: Vec3) -> Option<f32> {
        let local_point = self.world_to_local(world_point);
        self.volume
            .sample_local(local_point)
            .map(|distance| distance * self.uniform_scale)
    }

    pub fn normal_world(&self, world_point: Vec3) -> Option<Vec3> {
        let local_point = self.world_to_local(world_point);
        self.volume
            .normal_local(local_point)
            .map(|normal| (self.rotation * normal).normalize_or_zero())
            .filter(|normal| normal.length_squared() > 0.0)
    }

    pub fn world_aabb(&self) -> (Vec3, Vec3) {
        let (local_min, local_max) = self.volume.local_bounds();
        let corners = [
            Vec3::new(local_min.x, local_min.y, local_min.z),
            Vec3::new(local_max.x, local_min.y, local_min.z),
            Vec3::new(local_min.x, local_max.y, local_min.z),
            Vec3::new(local_max.x, local_max.y, local_min.z),
            Vec3::new(local_min.x, local_min.y, local_max.z),
            Vec3::new(local_max.x, local_min.y, local_max.z),
            Vec3::new(local_min.x, local_max.y, local_max.z),
            Vec3::new(local_max.x, local_max.y, local_max.z),
        ];

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for corner in corners {
            let world = self.local_to_world(corner);
            min = min.min(world);
            max = max.max(world);
        }
        (min, max)
    }

    pub fn raycast_sdf(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        hit_epsilon: f32,
    ) -> Option<SdfRayHit> {
        self.raycast_sdf_in_aabb(
            origin,
            direction,
            max_distance,
            hit_epsilon,
            self.world_aabb(),
        )
    }

    fn raycast_sdf_in_aabb(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        hit_epsilon: f32,
        world_aabb: (Vec3, Vec3),
    ) -> Option<SdfRayHit> {
        let direction = direction.try_normalize()?;
        let (aabb_min, aabb_max) = world_aabb;
        let (mut t, t_end) = ray_aabb(origin, direction, aabb_min, aabb_max)?;
        if t > max_distance || t_end < 0.0 {
            return None;
        }
        t = t.max(0.0);
        let t_end = t_end.min(max_distance);
        let hit_epsilon = hit_epsilon.max(MIN_GRADIENT_EPS);

        while t <= t_end {
            let point = origin + direction * t;
            if let Some(distance) = self.sample_world(point) {
                if distance <= hit_epsilon {
                    let normal = self.normal_world(point).unwrap_or(-direction);
                    return Some(SdfRayHit {
                        instance_id: self.id,
                        asset_id: self.asset_id,
                        point,
                        normal,
                        distance_along_ray: t,
                        signed_distance: distance,
                    });
                }
                t += distance.abs().max(hit_epsilon * 0.5);
            } else {
                t += hit_epsilon;
            }
        }

        None
    }

    fn world_to_local(&self, world_point: Vec3) -> Vec3 {
        self.rotation.inverse() * ((world_point - self.position) / self.uniform_scale)
    }

    fn local_to_world(&self, local_point: Vec3) -> Vec3 {
        self.position + self.rotation * (local_point * self.uniform_scale)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SdfSample {
    pub instance_id: SdfInstanceId,
    pub asset_id: SdfAssetId,
    pub point: Vec3,
    pub signed_distance: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SdfContact {
    pub instance_id: SdfInstanceId,
    pub asset_id: SdfAssetId,
    pub point: Vec3,
    pub projected_point: Vec3,
    pub normal: Vec3,
    pub signed_distance: f32,
    pub penetration: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SdfRayHit {
    pub instance_id: SdfInstanceId,
    pub asset_id: SdfAssetId,
    pub point: Vec3,
    pub normal: Vec3,
    pub distance_along_ray: f32,
    pub signed_distance: f32,
}

pub trait SdfQuery {
    fn sample_world(&self, point: Vec3) -> Option<SdfSample>;
    fn normal_world(&self, point: Vec3) -> Option<Vec3>;
    fn project_point(&self, point: Vec3, clearance: f32) -> Option<SdfContact>;
    fn raycast_sdf(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        hit_epsilon: f32,
    ) -> Option<SdfRayHit>;
}

#[derive(Debug, Clone, Default)]
pub struct SdfColliderSet {
    instances: Vec<SdfInstance>,
    instance_aabbs: Vec<(Vec3, Vec3)>,
    next_instance_id: u64,
}

impl SdfColliderSet {
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            instance_aabbs: Vec::new(),
            next_instance_id: 1,
        }
    }

    pub fn from_instances(instances: Vec<SdfInstance>) -> Self {
        let next_instance_id = instances
            .iter()
            .map(|instance| instance.id.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let instance_aabbs = instances.iter().map(SdfInstance::world_aabb).collect();
        Self {
            instances,
            instance_aabbs,
            next_instance_id,
        }
    }

    pub fn add_instance(
        &mut self,
        asset_id: SdfAssetId,
        volume: Arc<SdfVolume>,
        position: Vec3,
        rotation: Quat,
        uniform_scale: f32,
    ) -> SdfInstanceId {
        let id = SdfInstanceId(self.next_instance_id);
        self.next_instance_id = self.next_instance_id.saturating_add(1);
        let instance = SdfInstance::new(id, asset_id, volume, position, rotation, uniform_scale);
        self.instance_aabbs.push(instance.world_aabb());
        self.instances.push(instance);
        id
    }

    pub fn clear(&mut self) {
        self.instances.clear();
        self.instance_aabbs.clear();
    }

    pub fn instances(&self) -> &[SdfInstance] {
        &self.instances
    }

    pub fn world_aabbs(&self) -> &[(Vec3, Vec3)] {
        &self.instance_aabbs
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    pub fn sample_world(&self, point: Vec3) -> Option<SdfSample> {
        self.instances
            .iter()
            .zip(self.instance_aabbs.iter())
            .filter_map(|(instance, aabb)| {
                aabb_contains(*aabb, point).then(|| {
                    instance
                        .sample_world(point)
                        .map(|signed_distance| SdfSample {
                            instance_id: instance.id,
                            asset_id: instance.asset_id,
                            point,
                            signed_distance,
                        })
                })?
            })
            .min_by(|a, b| a.signed_distance.total_cmp(&b.signed_distance))
    }

    pub fn normal_world(&self, point: Vec3) -> Option<Vec3> {
        let sample = self.sample_world(point)?;
        let instance = self
            .instances
            .iter()
            .find(|instance| instance.id == sample.instance_id)?;
        instance.normal_world(point)
    }

    pub fn project_point(&self, point: Vec3, clearance: f32) -> Option<SdfContact> {
        let sample = self.sample_world(point)?;
        let penetration = clearance - sample.signed_distance;
        if penetration <= 0.0 {
            return None;
        }
        let normal = self.normal_world(point)?;
        Some(SdfContact {
            instance_id: sample.instance_id,
            asset_id: sample.asset_id,
            point,
            projected_point: point + normal * penetration,
            normal,
            signed_distance: sample.signed_distance,
            penetration,
        })
    }

    pub fn raycast_sdf(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        hit_epsilon: f32,
    ) -> Option<SdfRayHit> {
        self.instances
            .iter()
            .zip(self.instance_aabbs.iter())
            .filter_map(|(instance, aabb)| {
                instance.raycast_sdf_in_aabb(origin, direction, max_distance, hit_epsilon, *aabb)
            })
            .min_by(|a, b| a.distance_along_ray.total_cmp(&b.distance_along_ray))
    }
}

impl SdfQuery for SdfColliderSet {
    fn sample_world(&self, point: Vec3) -> Option<SdfSample> {
        SdfColliderSet::sample_world(self, point)
    }

    fn normal_world(&self, point: Vec3) -> Option<Vec3> {
        SdfColliderSet::normal_world(self, point)
    }

    fn project_point(&self, point: Vec3, clearance: f32) -> Option<SdfContact> {
        SdfColliderSet::project_point(self, point, clearance)
    }

    fn raycast_sdf(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        hit_epsilon: f32,
    ) -> Option<SdfRayHit> {
        SdfColliderSet::raycast_sdf(self, origin, direction, max_distance, hit_epsilon)
    }
}

fn validate_desc(desc: SdfVolumeDesc, actual_len: usize) -> Result<(), SdfVolumeError> {
    if desc.resolution.iter().any(|&axis| axis < 2) {
        return Err(SdfVolumeError::ResolutionTooSmall(desc.resolution));
    }
    if !desc.voxel_size.is_finite() || desc.voxel_size <= 0.0 {
        return Err(SdfVolumeError::InvalidVoxelSize);
    }
    if !desc.narrow_band.is_finite() || desc.narrow_band <= 0.0 {
        return Err(SdfVolumeError::InvalidNarrowBand);
    }

    let expected = desc
        .resolution
        .iter()
        .try_fold(1usize, |acc, &axis| acc.checked_mul(axis as usize))
        .unwrap_or(usize::MAX);
    if expected != actual_len {
        return Err(SdfVolumeError::DistanceCountMismatch {
            expected,
            actual: actual_len,
        });
    }

    Ok(())
}

fn axis_sample(grid_coord: f32, resolution: u32) -> (u32, u32, f32) {
    let max_index = resolution - 1;
    if grid_coord >= max_index as f32 {
        return (max_index - 1, max_index, 1.0);
    }

    let i0 = grid_coord.floor().max(0.0) as u32;
    let i1 = (i0 + 1).min(max_index);
    let t = (grid_coord - i0 as f32).clamp(0.0, 1.0);
    (i0, i1, t)
}

fn sample_or_center(volume: &SdfVolume, local_point: Vec3, center: f32) -> f32 {
    volume.sample_local(local_point).unwrap_or(center)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn aabb_contains((min, max): (Vec3, Vec3), point: Vec3) -> bool {
    point.x >= min.x
        && point.y >= min.y
        && point.z >= min.z
        && point.x <= max.x
        && point.y <= max.y
        && point.z <= max.z
}

fn ray_aabb(origin: Vec3, direction: Vec3, min: Vec3, max: Vec3) -> Option<(f32, f32)> {
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;

    for axis in 0..3 {
        let origin_axis = origin[axis];
        let direction_axis = direction[axis];
        let min_axis = min[axis];
        let max_axis = max[axis];

        if direction_axis.abs() < 1.0e-8 {
            if origin_axis < min_axis || origin_axis > max_axis {
                return None;
            }
            continue;
        }

        let inv = 1.0 / direction_axis;
        let mut t0 = (min_axis - origin_axis) * inv;
        let mut t1 = (max_axis - origin_axis) * inv;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_min = t_min.max(t0);
        t_max = t_max.min(t1);
        if t_min > t_max {
            return None;
        }
    }

    Some((t_min, t_max))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_sdf_volume(half_extents: Vec3) -> Arc<SdfVolume> {
        let resolution = [17, 17, 17];
        let voxel_size = 0.25;
        let origin = Vec3::splat(-2.0);
        let mut distances = Vec::new();
        for z in 0..resolution[2] {
            for y in 0..resolution[1] {
                for x in 0..resolution[0] {
                    let p = origin + Vec3::new(x as f32, y as f32, z as f32) * voxel_size;
                    distances.push(sd_box(p, half_extents));
                }
            }
        }
        Arc::new(
            SdfVolume::from_f32_grid(
                SdfVolumeDesc {
                    resolution,
                    origin,
                    voxel_size,
                    narrow_band: 2.0,
                },
                distances,
            )
            .unwrap(),
        )
    }

    fn sd_box(point: Vec3, half_extents: Vec3) -> f32 {
        let q = point.abs() - half_extents;
        q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
    }

    #[test]
    fn volume_samples_signed_distance_in_local_space() {
        let volume = box_sdf_volume(Vec3::splat(1.0));
        assert!(volume.sample_local(Vec3::ZERO).unwrap() < -0.9);
        assert!(volume.sample_local(Vec3::new(1.5, 0.0, 0.0)).unwrap() > 0.4);
        assert!(volume.sample_local(Vec3::new(3.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn snorm16_constructor_decodes_narrow_band() {
        let volume = SdfVolume::from_snorm16_grid(
            SdfVolumeDesc {
                resolution: [2, 2, 2],
                origin: Vec3::ZERO,
                voxel_size: 1.0,
                narrow_band: 4.0,
            },
            &[i16::MIN, 0, i16::MAX, 0, 0, 0, 0, 0],
        )
        .unwrap();

        assert!((volume.distances()[0] + 4.0).abs() < 0.001);
        assert!((volume.distances()[2] - 4.0).abs() < 0.001);
    }

    #[test]
    fn volume_cache_reuses_decoded_asset_sdfs() {
        let mut cache = SdfVolumeCache::new();
        let desc = SdfVolumeDesc {
            resolution: [2, 2, 2],
            origin: Vec3::ZERO,
            voxel_size: 1.0,
            narrow_band: 2.0,
        };
        let first = cache
            .get_or_insert_snorm16_grid(SdfAssetId(9), desc, &[0; 8])
            .unwrap();
        let second = cache
            .get_or_insert_snorm16_grid(SdfAssetId(9), desc, &[i16::MAX; 8])
            .unwrap();

        assert_eq!(cache.len(), 1);
        assert!(
            Arc::ptr_eq(&first, &second),
            "same asset id should reuse the decoded volume"
        );
        assert_eq!(second.distances()[0], 0.0);
    }

    #[test]
    fn collider_set_applies_instance_rotation() {
        let volume = box_sdf_volume(Vec3::new(0.35, 0.5, 1.4));
        let mut set = SdfColliderSet::new();
        set.add_instance(
            SdfAssetId(7),
            volume,
            Vec3::ZERO,
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            1.0,
        );

        let sample = set.sample_world(Vec3::new(1.0, 0.0, 0.0)).unwrap();
        assert!(
            sample.signed_distance < 0.0,
            "rotated long box should include the world-x point, got {}",
            sample.signed_distance
        );
    }

    #[test]
    fn collider_set_caches_world_aabbs() {
        let volume = box_sdf_volume(Vec3::splat(1.0));
        let mut set = SdfColliderSet::new();
        set.add_instance(
            SdfAssetId(1),
            volume,
            Vec3::new(5.0, 0.0, -2.0),
            Quat::IDENTITY,
            1.0,
        );

        assert_eq!(set.instances().len(), set.world_aabbs().len());
        let (min, max) = set.world_aabbs()[0];
        assert!(min.x < 4.0 && max.x > 6.0);
        assert!(min.z < -3.0 && max.z > -1.0);
    }

    #[test]
    fn project_point_pushes_out_of_solid() {
        let volume = box_sdf_volume(Vec3::splat(1.0));
        let mut set = SdfColliderSet::new();
        set.add_instance(SdfAssetId(1), volume, Vec3::ZERO, Quat::IDENTITY, 1.0);

        let contact = set.project_point(Vec3::new(0.8, 0.0, 0.0), 0.0).unwrap();
        assert!(contact.penetration > 0.15);
        assert!(contact.projected_point.x > 0.99);
    }

    #[test]
    fn raycast_sdf_finds_surface() {
        let volume = box_sdf_volume(Vec3::splat(1.0));
        let mut set = SdfColliderSet::new();
        set.add_instance(SdfAssetId(1), volume, Vec3::ZERO, Quat::IDENTITY, 1.0);

        let hit = set
            .raycast_sdf(Vec3::new(-3.0, 0.0, 0.0), Vec3::X, 8.0, 0.01)
            .unwrap();
        assert!((hit.point.x + 1.0).abs() < 0.05, "hit at {:?}", hit.point);
        assert!(hit.normal.x < -0.9, "normal {:?}", hit.normal);
    }
}
