use glam::{Vec3, Quat};

/// Content type for a spatial UI panel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PanelContent {
    /// Plain text label.
    Text,
    /// Chart or graph visualization.
    Chart,
    /// Interactive button grid.
    Buttons,
    /// Rendered texture (e.g., minimap).
    Texture,
}

/// A 3D UI panel floating in world space.
#[derive(Debug, Clone)]
pub struct SpatialPanel {
    pub id: u64,
    pub position: Vec3,
    pub rotation: Quat,
    /// Width and height in world-space metres.
    pub size: [f32; 2],
    pub content: PanelContent,
    pub visible: bool,
}

impl SpatialPanel {
    pub fn new(id: u64, position: Vec3, size: [f32; 2], content: PanelContent) -> Self {
        Self {
            id,
            position,
            rotation: Quat::IDENTITY,
            size,
            content,
            visible: true,
        }
    }

    /// Local X axis (right) in world space.
    pub fn local_right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Local Y axis (up) in world space.
    pub fn local_up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    /// Panel normal (facing direction) in world space.
    pub fn normal(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    /// Test whether a ray (origin + direction) intersects this panel.
    /// Returns `Some(t)` where `t` is the ray parameter at intersection,
    /// or `None` if the ray misses.
    pub fn ray_intersect(&self, ray_origin: Vec3, ray_dir: Vec3) -> Option<f32> {
        let n = self.normal();
        let denom = n.dot(ray_dir);
        // Ray roughly parallel to panel — no hit.
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = n.dot(self.position - ray_origin) / denom;
        if t < 0.0 {
            return None;
        }
        // Intersection point in world space.
        let hit = ray_origin + ray_dir * t;
        let local = hit - self.position;
        let u = local.dot(self.local_right());
        let v = local.dot(self.local_up());
        let hw = self.size[0] * 0.5;
        let hh = self.size[1] * 0.5;
        if u.abs() <= hw && v.abs() <= hh {
            Some(t)
        } else {
            None
        }
    }
}

/// Manages a collection of spatial UI panels.
pub struct SpatialUIManager {
    panels: Vec<SpatialPanel>,
    next_id: u64,
}

impl SpatialUIManager {
    pub fn new() -> Self {
        Self {
            panels: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a panel and return its assigned id.
    pub fn add_panel(&mut self, position: Vec3, size: [f32; 2], content: PanelContent) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.panels.push(SpatialPanel::new(id, position, size, content));
        id
    }

    /// Remove a panel by id. Returns `true` if found and removed.
    pub fn remove_panel(&mut self, id: u64) -> bool {
        let len_before = self.panels.len();
        self.panels.retain(|p| p.id != id);
        self.panels.len() < len_before
    }

    /// Get a panel by id.
    pub fn get(&self, id: u64) -> Option<&SpatialPanel> {
        self.panels.iter().find(|p| p.id == id)
    }

    /// Get a mutable reference to a panel by id.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut SpatialPanel> {
        self.panels.iter_mut().find(|p| p.id == id)
    }

    /// How many panels are managed.
    pub fn count(&self) -> usize {
        self.panels.len()
    }

    /// Cast a ray and return the first (nearest) panel hit, with the
    /// intersection distance `t`.
    pub fn ray_intersect(&self, ray_origin: Vec3, ray_dir: Vec3) -> Option<(&SpatialPanel, f32)> {
        let mut best: Option<(&SpatialPanel, f32)> = None;
        for panel in &self.panels {
            if !panel.visible {
                continue;
            }
            if let Some(t) = panel.ray_intersect(ray_origin, ray_dir) {
                if best.is_none() || t < best.unwrap().1 {
                    best = Some((panel, t));
                }
            }
        }
        best
    }

    /// Return the nearest visible panel to a world position.
    pub fn nearest_panel(&self, pos: Vec3) -> Option<&SpatialPanel> {
        self.panels
            .iter()
            .filter(|p| p.visible)
            .min_by(|a, b| {
                let da = a.position.distance_squared(pos);
                let db = b.position.distance_squared(pos);
                da.partial_cmp(&db).unwrap()
            })
    }
}

impl Default for SpatialUIManager {
    fn default() -> Self {
        Self::new()
    }
}
