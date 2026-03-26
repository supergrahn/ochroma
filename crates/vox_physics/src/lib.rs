use glam::Vec3;
use std::collections::HashMap;

const GRAVITY: f32 = -9.81;

#[derive(Debug, Clone)]
pub struct RigidBody {
    pub id: u32,
    pub position: Vec3,
    pub velocity: Vec3,
    pub mass: f32,
    pub is_static: bool,
}

pub struct PhysicsWorld {
    bodies: HashMap<u32, RigidBody>,
    next_id: u32,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            bodies: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn add_body(&mut self, mut body: RigidBody) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        body.id = id;
        self.bodies.insert(id, body);
        id
    }

    pub fn step(&mut self, dt: f32) {
        for body in self.bodies.values_mut() {
            if body.is_static {
                continue;
            }
            // Apply gravity
            body.velocity.y += GRAVITY * dt;
            // Integrate position
            body.position += body.velocity * dt;
            // Simple ground plane collision at y=0
            if body.position.y < 0.0 {
                body.position.y = 0.0;
                body.velocity.y = 0.0;
            }
        }
    }

    pub fn get_body(&self, id: u32) -> Option<&RigidBody> {
        self.bodies.get(&id)
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}
