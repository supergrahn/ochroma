//! Rope and cable simulation — position-based Verlet integration.
//! 4 substeps per physics tick. Splats sync orientation + elongation along segment.

use glam::Vec3;
use vox_core::types::GaussianSplat;

const ROPE_SUBSTEPS: u32 = 4;

#[derive(Debug, Clone)]
pub struct RopeNode {
    pub position: Vec3,
    pub prev_position: Vec3,
    pub is_pinned: bool,
    /// Radius of this node (affects splat scale)
    pub radius: f32,
}

#[derive(Debug, Clone)]
pub struct Rope {
    pub nodes: Vec<RopeNode>,
    pub segment_rest_length: f32,
    pub stiffness: f32, // distance constraint compliance (0 = rigid)
    pub damping: f32,
    pub gravity: Vec3,
    /// Splat indices assigned to each segment (between nodes i and i+1).
    /// Length = nodes.len() - 1.
    pub splat_bindings: Vec<Option<usize>>,
}

impl Rope {
    /// Create a rope hanging from `anchor` with `num_nodes` nodes spaced `segment_length` apart downward.
    pub fn new_hanging(anchor: Vec3, num_nodes: usize, segment_length: f32) -> Self {
        let mut nodes = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let pos = anchor + Vec3::new(0.0, -(i as f32 * segment_length), 0.0);
            nodes.push(RopeNode {
                position: pos,
                prev_position: pos,
                is_pinned: i == 0, // anchor is pinned
                radius: 0.05,
            });
        }
        let num_segs = num_nodes.saturating_sub(1);
        Self {
            nodes,
            segment_rest_length: segment_length,
            stiffness: 0.0001,
            damping: 0.98,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            splat_bindings: vec![None; num_segs],
        }
    }

    /// Advance physics by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        let sub_dt = dt / ROPE_SUBSTEPS as f32;
        for _ in 0..ROPE_SUBSTEPS {
            self.substep(sub_dt);
        }
    }

    fn substep(&mut self, h: f32) {
        let n = self.nodes.len();

        // a) Verlet integration for non-pinned nodes
        for node in &mut self.nodes {
            if node.is_pinned {
                continue;
            }
            let velocity = (node.position - node.prev_position) * self.damping;
            let new_pos = node.position + velocity + self.gravity * h * h;
            node.prev_position = node.position;
            node.position = new_pos;
        }

        // b) Distance constraint (XPBD-style)
        for i in 0..n.saturating_sub(1) {
            let j = i + 1;
            let pinned_i = self.nodes[i].is_pinned;
            let pinned_j = self.nodes[j].is_pinned;

            // Skip if both pinned
            if pinned_i && pinned_j {
                continue;
            }

            let pos_i = self.nodes[i].position;
            let pos_j = self.nodes[j].position;
            let delta = pos_j - pos_i;
            let current_len = delta.length();

            if current_len < 1e-8 {
                continue;
            }

            let rest_len = self.segment_rest_length;
            let constraint = current_len - rest_len;

            // XPBD: compliance = stiffness, alpha = compliance / (h*h)
            let alpha = self.stiffness / (h * h);
            // Each particle has unit mass (w = 1.0)
            let w_i = if pinned_i { 0.0 } else { 1.0 };
            let w_j = if pinned_j { 0.0 } else { 1.0 };
            let w_sum = w_i + w_j + alpha;

            if w_sum < 1e-10 {
                continue;
            }

            let lambda = -constraint / w_sum;
            let correction = (delta / current_len) * lambda;

            if !pinned_i {
                self.nodes[i].position -= correction * w_i;
            }
            if !pinned_j {
                self.nodes[j].position += correction * w_j;
            }
        }

        // c) Ground plane: clamp y >= 0
        for node in &mut self.nodes {
            if node.position.y < 0.0 {
                node.position.y = 0.0;
                node.prev_position.y = 0.0;
            }
        }
    }

    /// Set the position of the anchor node (node 0).
    pub fn set_anchor(&mut self, pos: Vec3) {
        if let Some(node) = self.nodes.first_mut() {
            node.position = pos;
            node.prev_position = pos;
        }
    }

    /// Apply current node positions to bound splats.
    /// Each splat is positioned at the midpoint of its segment,
    /// oriented along the segment direction, and scaled in Y by segment elongation.
    pub fn apply_to_splats(&self, splats: &mut [GaussianSplat]) {
        for (i, binding) in self.splat_bindings.iter().enumerate() {
            let Some(splat_idx) = *binding else {
                continue;
            };
            if splat_idx >= splats.len() {
                continue;
            }
            if i + 1 >= self.nodes.len() {
                continue;
            }

            let node_a = &self.nodes[i];
            let node_b = &self.nodes[i + 1];

            let midpoint = (node_a.position + node_b.position) * 0.5;
            let seg_dir = node_b.position - node_a.position;
            let actual_len = seg_dir.length();
            let elongation = if self.segment_rest_length > 1e-8 {
                actual_len / self.segment_rest_length
            } else {
                1.0
            };

            let splat = &mut splats[splat_idx];
            splat.set_position(midpoint.to_array());

            // Scale: Y axis along rope = radius * elongation, X and Z = radius
            let radius = node_a.radius;
            splat.set_scales(radius, radius * elongation, radius);

            // Orient the splat along the segment direction
            if actual_len > 1e-8 {
                let dir_norm = seg_dir / actual_len;
                // Build a quaternion that rotates Y-up to dir_norm
                let y_up = Vec3::Y;
                let rot = if (dir_norm - y_up).length() < 1e-6 {
                    // Already aligned
                    glam::Quat::IDENTITY
                } else if (dir_norm + y_up).length() < 1e-6 {
                    // Opposite — rotate 180 around X
                    glam::Quat::from_rotation_x(std::f32::consts::PI)
                } else {
                    glam::Quat::from_rotation_arc(y_up, dir_norm)
                };
                // Reconstruct splat with new rotation while preserving other fields
                *splat = GaussianSplat::volume(
                    splat.position(),
                    splat.scales(),
                    rot,
                    splat.opacity(),
                    *splat.spectral(),
                );
            }
        }
    }

    /// Total rope length (sum of segment distances)
    pub fn current_length(&self) -> f32 {
        self.nodes
            .windows(2)
            .map(|w| w[1].position.distance(w[0].position))
            .sum()
    }
}

pub struct RopeWorld {
    pub ropes: Vec<Rope>,
}

impl RopeWorld {
    pub fn new() -> Self {
        Self { ropes: Vec::new() }
    }

    pub fn add_rope(&mut self, rope: Rope) -> usize {
        let idx = self.ropes.len();
        self.ropes.push(rope);
        idx
    }

    pub fn step(&mut self, dt: f32) {
        for rope in &mut self.ropes {
            rope.step(dt);
        }
    }

    pub fn apply_to_splats(&self, splats: &mut [GaussianSplat]) {
        for rope in &self.ropes {
            rope.apply_to_splats(splats);
        }
    }
}

impl Default for RopeWorld {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_hangs_under_gravity() {
        let anchor = Vec3::new(0.0, 5.0, 0.0);
        let mut rope = Rope::new_hanging(anchor, 5, 0.5);
        let initial_last_y = rope.nodes.last().unwrap().position.y;

        // Step 1.0 second
        for _ in 0..60 {
            rope.step(1.0 / 60.0);
        }

        let final_last_y = rope.nodes.last().unwrap().position.y;
        assert!(
            final_last_y < initial_last_y,
            "last node should fall: initial={initial_last_y}, final={final_last_y}"
        );
    }

    #[test]
    fn rope_anchor_stays_fixed() {
        let anchor = Vec3::new(1.0, 10.0, 0.0);
        let mut rope = Rope::new_hanging(anchor, 5, 0.5);

        for _ in 0..120 {
            rope.step(1.0 / 60.0);
        }

        let node0 = &rope.nodes[0];
        assert!(node0.is_pinned, "node 0 must be pinned");
        assert!(
            (node0.position - anchor).length() < 1e-4,
            "pinned node should not move: pos={:?}",
            node0.position
        );
    }

    #[test]
    fn rope_length_is_approximate() {
        let anchor = Vec3::new(0.0, 10.0, 0.0);
        let num_nodes = 6;
        let seg_len = 1.0;
        let mut rope = Rope::new_hanging(anchor, num_nodes, seg_len);
        let rest_total = (num_nodes - 1) as f32 * seg_len;

        for _ in 0..120 {
            rope.step(1.0 / 60.0);
        }

        let length = rope.current_length();
        assert!(
            length < rest_total * 2.0,
            "rope should not explode: length={length}, rest={rest_total}"
        );
    }

    #[test]
    fn rope_world_step_advances_all() {
        let mut world = RopeWorld::new();
        let r1 = Rope::new_hanging(Vec3::new(0.0, 5.0, 0.0), 4, 0.5);
        let r2 = Rope::new_hanging(Vec3::new(5.0, 5.0, 0.0), 4, 0.5);

        let initial_y1 = r1.nodes.last().unwrap().position.y;
        let initial_y2 = r2.nodes.last().unwrap().position.y;

        world.add_rope(r1);
        world.add_rope(r2);

        for _ in 0..60 {
            world.step(1.0 / 60.0);
        }

        let final_y1 = world.ropes[0].nodes.last().unwrap().position.y;
        let final_y2 = world.ropes[1].nodes.last().unwrap().position.y;

        assert!(
            final_y1 < initial_y1,
            "rope 1 last node should fall: {initial_y1} -> {final_y1}"
        );
        assert!(
            final_y2 < initial_y2,
            "rope 2 last node should fall: {initial_y2} -> {final_y2}"
        );
    }
}
