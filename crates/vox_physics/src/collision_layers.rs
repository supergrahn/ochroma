//! Collision layer system and physics queries.

use glam::Vec3;

/// Collision layer bits for the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CollisionLayer {
    Default    = 0b00000001,
    Player     = 0b00000010,
    Enemy      = 0b00000100,
    Terrain    = 0b00001000,
    Projectile = 0b00010000,
    Trigger    = 0b00100000,
    Fluid      = 0b01000000,
    Debris     = 0b10000000,
}

/// Bitmask filter controlling which layers collide.
#[derive(Debug, Clone, Copy)]
pub struct CollisionFilter {
    pub membership: u32, // which layers this object is in
    pub mask: u32,       // which layers this object collides with
}

impl CollisionFilter {
    pub fn default_solid() -> Self {
        Self {
            membership: CollisionLayer::Default as u32,
            mask: 0xFF,
        }
    }

    pub fn player() -> Self {
        Self {
            membership: CollisionLayer::Player as u32,
            mask: 0xFF & !(CollisionLayer::Player as u32),
        }
    }

    pub fn collides_with(&self, other: &CollisionFilter) -> bool {
        (self.mask & other.membership) != 0 && (other.mask & self.membership) != 0
    }
}

/// Result of a physics raycast query.
#[derive(Debug, Clone)]
pub struct RaycastHit {
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub layer: u32, // CollisionLayer bitmask of the hit object
    /// Spectral bands at hit surface (for spectral_raycast)
    pub spectral: Option<[f32; 8]>,
}

/// Physics query interface.
pub trait PhysicsQuery {
    fn raycast(&self, origin: Vec3, direction: Vec3, max_dist: f32, mask: u32) -> Option<RaycastHit>;
    fn sphere_overlap(&self, center: Vec3, radius: f32, mask: u32) -> Vec<u32>;
    fn spectral_raycast(&self, origin: Vec3, direction: Vec3, max_dist: f32) -> Option<RaycastHit>;
}

/// Simple AABB-based physics world that implements PhysicsQuery.
pub struct LayeredPhysicsWorld {
    /// (position, half_extents, filter, optional spectral)
    bodies: Vec<(Vec3, Vec3, CollisionFilter, Option<[f32; 8]>)>,
}

impl LayeredPhysicsWorld {
    pub fn new() -> Self {
        Self { bodies: Vec::new() }
    }

    pub fn add_body(
        &mut self,
        position: Vec3,
        half_extents: Vec3,
        filter: CollisionFilter,
        spectral: Option<[f32; 8]>,
    ) -> usize {
        let idx = self.bodies.len();
        self.bodies.push((position, half_extents, filter, spectral));
        idx
    }
}

impl PhysicsQuery for LayeredPhysicsWorld {
    /// Slab test against all AABB bodies filtered by mask.
    fn raycast(&self, origin: Vec3, direction: Vec3, max_dist: f32, mask: u32) -> Option<RaycastHit> {
        // Normalize direction; if zero-length return None
        let dir_len = direction.length();
        if dir_len < 1e-10 {
            return None;
        }
        let dir = direction / dir_len;

        let mut best: Option<RaycastHit> = None;

        for (pos, half, filter, spectral) in &self.bodies {
            // Layer mask check
            if (mask & filter.membership) == 0 {
                continue;
            }

            // AABB slab test
            let aabb_min = *pos - *half;
            let aabb_max = *pos + *half;

            // Per-axis t values (avoid div-by-zero with large sentinel)
            let inv_dir = Vec3::new(
                if dir.x.abs() > 1e-10 { 1.0 / dir.x } else { f32::INFINITY },
                if dir.y.abs() > 1e-10 { 1.0 / dir.y } else { f32::INFINITY },
                if dir.z.abs() > 1e-10 { 1.0 / dir.z } else { f32::INFINITY },
            );

            let t1 = (aabb_min - origin) * inv_dir;
            let t2 = (aabb_max - origin) * inv_dir;

            let t_near_x = t1.x.min(t2.x);
            let t_far_x  = t1.x.max(t2.x);
            let t_near_y = t1.y.min(t2.y);
            let t_far_y  = t1.y.max(t2.y);
            let t_near_z = t1.z.min(t2.z);
            let t_far_z  = t1.z.max(t2.z);

            let t_near = t_near_x.max(t_near_y).max(t_near_z);
            let t_far  = t_far_x.min(t_far_y).min(t_far_z);

            if t_near > t_far || t_far < 0.0 {
                continue; // miss
            }

            // Hit t (use 0 if origin is inside the AABB)
            let t_hit = if t_near >= 0.0 { t_near } else { 0.0 };

            if t_hit > max_dist {
                continue;
            }

            // Determine which face was hit for the normal
            let normal = if t_near < 0.0 {
                // Inside the AABB — use the exit face normal pointing inward
                if t_far_x <= t_far_y && t_far_x <= t_far_z {
                    Vec3::new(if dir.x > 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0)
                } else if t_far_y <= t_far_x && t_far_y <= t_far_z {
                    Vec3::new(0.0, if dir.y > 0.0 { 1.0 } else { -1.0 }, 0.0)
                } else {
                    Vec3::new(0.0, 0.0, if dir.z > 0.0 { 1.0 } else { -1.0 })
                }
            } else if t_near_x >= t_near_y && t_near_x >= t_near_z {
                Vec3::new(if dir.x < 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0)
            } else if t_near_y >= t_near_x && t_near_y >= t_near_z {
                Vec3::new(0.0, if dir.y < 0.0 { 1.0 } else { -1.0 }, 0.0)
            } else {
                Vec3::new(0.0, 0.0, if dir.z < 0.0 { 1.0 } else { -1.0 })
            };

            let hit = RaycastHit {
                point: origin + dir * t_hit,
                normal,
                distance: t_hit,
                layer: filter.membership,
                spectral: *spectral,
            };

            // Keep the closest hit
            match &best {
                None => best = Some(hit),
                Some(prev) if hit.distance < prev.distance => best = Some(hit),
                _ => {}
            }
        }

        best
    }

    /// Return body indices of all bodies within radius (sphere-AABB overlap).
    fn sphere_overlap(&self, center: Vec3, radius: f32, mask: u32) -> Vec<u32> {
        let mut result = Vec::new();

        for (idx, (pos, half, filter, _)) in self.bodies.iter().enumerate() {
            if (mask & filter.membership) == 0 {
                continue;
            }

            // Closest point on AABB to sphere center
            let closest = Vec3::new(
                center.x.clamp(pos.x - half.x, pos.x + half.x),
                center.y.clamp(pos.y - half.y, pos.y + half.y),
                center.z.clamp(pos.z - half.z, pos.z + half.z),
            );

            let dist_sq = (closest - center).length_squared();
            if dist_sq <= radius * radius {
                result.push(idx as u32);
            }
        }

        result
    }

    fn spectral_raycast(&self, origin: Vec3, direction: Vec3, max_dist: f32) -> Option<RaycastHit> {
        // Raycast with mask = ALL, returns spectral data if available
        self.raycast(origin, direction, max_dist, 0xFF)
    }
}

impl Default for LayeredPhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collision_filter_collides_with() {
        // Player does NOT collide with another Player (Player mask excludes Player bit)
        let p1 = CollisionFilter::player();
        let p2 = CollisionFilter::player();
        assert!(
            !p1.collides_with(&p2),
            "Player filters should not collide with each other"
        );
    }

    #[test]
    fn collision_filter_player_vs_enemy() {
        let player = CollisionFilter::player();
        let enemy = CollisionFilter {
            membership: CollisionLayer::Enemy as u32,
            mask: 0xFF,
        };
        assert!(
            player.collides_with(&enemy),
            "Player and Enemy should collide"
        );
    }

    #[test]
    fn raycast_hits_aabb() {
        let mut world = LayeredPhysicsWorld::new();
        world.add_body(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::splat(0.5),
            CollisionFilter::default_solid(),
            None,
        );

        let hit = world.raycast(
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, 1.0),
            100.0,
            0xFF,
        );

        assert!(hit.is_some(), "Ray should hit the AABB");
        let h = hit.unwrap();
        assert!(
            (h.distance - 4.5).abs() < 0.01,
            "Hit distance should be ~4.5 (from z=-5 to z=-0.5), got {}",
            h.distance
        );
        assert!(
            (h.normal - Vec3::new(0.0, 0.0, -1.0)).length() < 0.01,
            "Normal should point toward -Z, got {:?}",
            h.normal
        );
    }

    #[test]
    fn raycast_misses_masked_layer() {
        let mut world = LayeredPhysicsWorld::new();
        // Body with Trigger membership and mask=0 (collides with nothing)
        world.add_body(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::splat(0.5),
            CollisionFilter {
                membership: CollisionLayer::Trigger as u32,
                mask: 0,
            },
            None,
        );

        // Raycast with Default mask — Trigger is not in Default mask check
        let hit = world.raycast(
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, 1.0),
            100.0,
            CollisionLayer::Default as u32,
        );

        assert!(
            hit.is_none(),
            "Ray should miss body whose membership is not in the query mask"
        );
    }

    #[test]
    fn sphere_overlap_finds_nearby_body() {
        let mut world = LayeredPhysicsWorld::new();
        world.add_body(
            Vec3::ZERO,
            Vec3::splat(0.5),
            CollisionFilter::default_solid(),
            None,
        );

        let overlaps = world.sphere_overlap(Vec3::new(0.5, 0.0, 0.0), 1.0, 0xFF);
        assert!(
            !overlaps.is_empty(),
            "Sphere at (0.5,0,0) with radius 1.0 should overlap AABB at origin"
        );
    }
}
