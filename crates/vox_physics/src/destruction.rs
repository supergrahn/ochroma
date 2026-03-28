//! Physics destruction system — fracture objects into pieces on impact.

use glam::{Quat, Vec3};

/// A fragment of a destroyed object.
#[derive(Debug, Clone)]
pub struct Fragment {
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Quat,
    pub angular_velocity: Vec3,
    pub mass: f32,
    pub size: Vec3,
    pub active: bool,
}

/// A destructible object that fractures into fragments upon taking enough damage.
pub struct DestructibleObject {
    pub fragments: Vec<Fragment>,
    pub health: f32,
    pub is_destroyed: bool,
    #[allow(dead_code)]
    center: Vec3,
    #[allow(dead_code)]
    max_health: f32,
}

const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

impl DestructibleObject {
    /// Create a destructible from a bounding box, pre-fractured into N pieces.
    /// Fragments are arranged in a rough grid within the box.
    pub fn from_box(center: Vec3, half_extents: Vec3, fragment_count: u32) -> Self {
        let n = fragment_count.max(1);
        // Approximate a cube root for 3D subdivision
        let per_axis = (n as f32).cbrt().ceil() as u32;
        let frag_size = (half_extents * 2.0) / per_axis as f32;

        let mut fragments = Vec::new();
        let start = center - half_extents + frag_size * 0.5;

        for ix in 0..per_axis {
            for iy in 0..per_axis {
                for iz in 0..per_axis {
                    if fragments.len() >= n as usize {
                        break;
                    }
                    let pos = start + Vec3::new(
                        ix as f32 * frag_size.x,
                        iy as f32 * frag_size.y,
                        iz as f32 * frag_size.z,
                    );
                    fragments.push(Fragment {
                        position: pos,
                        velocity: Vec3::ZERO,
                        rotation: Quat::IDENTITY,
                        angular_velocity: Vec3::ZERO,
                        mass: 1.0,
                        size: frag_size * 0.5, // half-extents
                        active: false,         // inactive until destroyed
                    });
                }
            }
        }

        Self {
            fragments,
            health: 100.0,
            is_destroyed: false,
            center,
            max_health: 100.0,
        }
    }

    /// Apply damage at a point. If health reaches 0, fracture.
    pub fn apply_damage(&mut self, point: Vec3, force: f32) {
        if self.is_destroyed {
            return;
        }

        self.health = (self.health - force).max(0.0);

        if self.health <= 0.0 {
            self.is_destroyed = true;
            // Activate all fragments and give them velocity radiating from impact point
            for frag in &mut self.fragments {
                frag.active = true;
                let dir = (frag.position - point).normalize_or_zero();
                let dist = frag.position.distance(point).max(0.1);
                // Closer fragments get more force
                let impulse = force / (dist * frag.mass);
                frag.velocity = dir * impulse.min(20.0); // cap velocity
                // Some angular velocity for visual effect
                frag.angular_velocity = Vec3::new(
                    dir.z * 3.0,
                    0.0,
                    -dir.x * 3.0,
                );
            }
        }
    }

    /// Step physics on active fragments (gravity + simple dynamics).
    pub fn step(&mut self, dt: f32) {
        if !self.is_destroyed {
            return;
        }

        for frag in &mut self.fragments {
            if !frag.active {
                continue;
            }

            // Gravity
            frag.velocity += GRAVITY * dt;

            // Integrate position
            frag.position += frag.velocity * dt;

            // Integrate rotation (simplified: treat angular_velocity as axis-angle rate)
            let ang_speed = frag.angular_velocity.length();
            if ang_speed > 1e-6 {
                let axis = frag.angular_velocity / ang_speed;
                let delta_rot = Quat::from_axis_angle(axis, ang_speed * dt);
                frag.rotation = (delta_rot * frag.rotation).normalize();
            }

            // Simple ground plane
            if frag.position.y < frag.size.y {
                frag.position.y = frag.size.y;
                frag.velocity.y *= -0.3;
                frag.velocity.x *= 0.9; // friction
                frag.velocity.z *= 0.9;
                frag.angular_velocity *= 0.9;
            }
        }
    }

    /// Get fragment positions/rotations/sizes for rendering.
    pub fn fragment_transforms(&self) -> Vec<(Vec3, Quat, Vec3)> {
        self.fragments
            .iter()
            .filter(|f| f.active)
            .map(|f| (f.position, f.rotation, f.size))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undamaged_object_is_intact() {
        let obj = DestructibleObject::from_box(Vec3::ZERO, Vec3::splat(1.0), 8);
        assert!(!obj.is_destroyed);
        assert!(obj.health > 0.0);
        assert!(obj.fragments.iter().all(|f| !f.active));
    }

    #[test]
    fn enough_damage_destroys() {
        let mut obj = DestructibleObject::from_box(Vec3::ZERO, Vec3::splat(1.0), 8);
        obj.apply_damage(Vec3::ZERO, 50.0);
        assert!(!obj.is_destroyed, "50 damage should not destroy 100hp object");

        obj.apply_damage(Vec3::ZERO, 60.0);
        assert!(obj.is_destroyed, "110 total damage should destroy 100hp object");
        assert!(obj.fragments.iter().all(|f| f.active));
    }

    #[test]
    fn fragments_have_velocity_after_destruction() {
        let mut obj = DestructibleObject::from_box(Vec3::ZERO, Vec3::splat(1.0), 8);
        obj.apply_damage(Vec3::new(-2.0, 0.0, 0.0), 200.0);

        assert!(obj.is_destroyed);
        let has_velocity = obj.fragments.iter().any(|f| f.velocity.length() > 0.1);
        assert!(has_velocity, "fragments should have velocity after explosion");
    }

    #[test]
    fn fragments_fall_under_gravity() {
        let mut obj = DestructibleObject::from_box(
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::splat(0.5),
            8,
        );
        obj.apply_damage(Vec3::new(0.0, 10.0, 0.0), 200.0);

        let start_positions: Vec<Vec3> = obj.fragments.iter().map(|f| f.position).collect();

        for _ in 0..100 {
            obj.step(0.016);
        }

        // At least some fragments should have moved downward
        let moved_down = obj.fragments.iter().enumerate().any(|(i, f)| {
            f.position.y < start_positions[i].y
        });
        assert!(moved_down, "fragments should fall under gravity");
    }
}
