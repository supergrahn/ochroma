use glam::Vec3;
use std::collections::HashMap;

const GRAVITY: f32 = -9.81;

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn from_center_half_extents(center: Vec3, half: Vec3) -> Self {
        Self { min: center - half, max: center + half }
    }

    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    pub fn center(&self) -> Vec3 { (self.min + self.max) * 0.5 }
    pub fn half_extents(&self) -> Vec3 { (self.max - self.min) * 0.5 }
}

#[derive(Debug, Clone)]
pub struct RigidBody {
    pub id: u32,
    pub position: Vec3,
    pub velocity: Vec3,
    pub mass: f32,
    pub is_static: bool,
}

struct BodyEntry {
    body: RigidBody,
    collider: Option<Vec3>, // half_extents, if present
}

pub struct PhysicsWorld {
    entries: HashMap<u32, BodyEntry>,
    next_id: u32,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn add_body(&mut self, mut body: RigidBody) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        body.id = id;
        self.entries.insert(id, BodyEntry { body, collider: None });
        id
    }

    pub fn add_body_with_collider(&mut self, mut body: RigidBody, half_extents: Vec3) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        body.id = id;
        self.entries.insert(id, BodyEntry { body, collider: Some(half_extents) });
        id
    }

    /// Returns pairs of body IDs whose AABBs overlap.
    pub fn check_collisions(&self) -> Vec<(u32, u32)> {
        let ids: Vec<u32> = self.entries.keys().copied().collect();
        let mut pairs = Vec::new();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let id_a = ids[i];
                let id_b = ids[j];
                let ea = &self.entries[&id_a];
                let eb = &self.entries[&id_b];
                if let (Some(ha), Some(hb)) = (&ea.collider, &eb.collider) {
                    let aabb_a = Aabb::from_center_half_extents(ea.body.position, *ha);
                    let aabb_b = Aabb::from_center_half_extents(eb.body.position, *hb);
                    if aabb_a.intersects(&aabb_b) {
                        pairs.push((id_a, id_b));
                    }
                }
            }
        }
        pairs
    }

    pub fn step(&mut self, dt: f32) {
        // Integrate dynamics
        for entry in self.entries.values_mut() {
            let body = &mut entry.body;
            if body.is_static {
                continue;
            }
            body.velocity.y += GRAVITY * dt;
            body.position += body.velocity * dt;
            // Simple ground plane at y=0
            if body.position.y < 0.0 {
                body.position.y = 0.0;
                body.velocity.y = 0.0;
            }
        }

        // Collision resolution: separate overlapping bodies along minimum penetration axis
        let ids: Vec<u32> = self.entries.keys().copied().collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let id_a = ids[i];
                let id_b = ids[j];

                let (ha, hb, pos_a, pos_b, static_a, static_b) = {
                    let ea = &self.entries[&id_a];
                    let eb = &self.entries[&id_b];
                    match (&ea.collider, &eb.collider) {
                        (Some(ha), Some(hb)) => (
                            *ha, *hb,
                            ea.body.position, eb.body.position,
                            ea.body.is_static, eb.body.is_static,
                        ),
                        _ => continue,
                    }
                };

                let aabb_a = Aabb::from_center_half_extents(pos_a, ha);
                let aabb_b = Aabb::from_center_half_extents(pos_b, hb);
                if !aabb_a.intersects(&aabb_b) {
                    continue;
                }

                // Compute per-axis penetration depths
                let overlap_x = (aabb_a.max.x.min(aabb_b.max.x) - aabb_a.min.x.max(aabb_b.min.x)).max(0.0);
                let overlap_y = (aabb_a.max.y.min(aabb_b.max.y) - aabb_a.min.y.max(aabb_b.min.y)).max(0.0);
                let overlap_z = (aabb_a.max.z.min(aabb_b.max.z) - aabb_a.min.z.max(aabb_b.min.z)).max(0.0);

                // Separation axis: smallest overlap
                let sep = if overlap_x <= overlap_y && overlap_x <= overlap_z {
                    let sign = if pos_a.x < pos_b.x { -1.0 } else { 1.0 };
                    Vec3::new(sign * overlap_x, 0.0, 0.0)
                } else if overlap_y <= overlap_x && overlap_y <= overlap_z {
                    let sign = if pos_a.y < pos_b.y { -1.0 } else { 1.0 };
                    Vec3::new(0.0, sign * overlap_y, 0.0)
                } else {
                    let sign = if pos_a.z < pos_b.z { -1.0 } else { 1.0 };
                    Vec3::new(0.0, 0.0, sign * overlap_z)
                };

                match (static_a, static_b) {
                    (false, false) => {
                        self.entries.get_mut(&id_a).unwrap().body.position += sep * 0.5;
                        self.entries.get_mut(&id_b).unwrap().body.position -= sep * 0.5;
                    }
                    (false, true) => {
                        self.entries.get_mut(&id_a).unwrap().body.position += sep;
                    }
                    (true, false) => {
                        self.entries.get_mut(&id_b).unwrap().body.position -= sep;
                    }
                    (true, true) => {}
                }
            }
        }
    }

    pub fn get_body(&self, id: u32) -> Option<&RigidBody> {
        self.entries.get(&id).map(|e| &e.body)
    }

    pub fn body_count(&self) -> usize {
        self.entries.len()
    }
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Rapier3D integration — always enabled, the real physics engine
// ---------------------------------------------------------------------------

pub mod cloth;

pub mod rapier;

pub use rapier::RapierPhysicsWorld;

// Re-export Rapier handle types so consumers don't need a direct rapier3d dependency
pub use rapier3d::prelude::{ColliderHandle, RigidBodyHandle};
