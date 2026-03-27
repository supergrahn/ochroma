/// Editor tools: Grid, Snapping, Copy/Paste/Duplicate.

use glam::Vec3;

/// Grid configuration for the editor viewport.
pub struct EditorGrid {
    pub visible: bool,
    pub size: f32,
    pub extent: f32,
    pub major_every: u32,
    pub color: [u8; 3],
    pub major_color: [u8; 3],
}

impl Default for EditorGrid {
    fn default() -> Self {
        Self {
            visible: true,
            size: 1.0,
            extent: 50.0,
            major_every: 10,
            color: [60, 60, 60],
            major_color: [100, 100, 100],
        }
    }
}

impl EditorGrid {
    /// Generate grid line endpoints for rendering.
    pub fn generate_lines(&self) -> Vec<(Vec3, Vec3, [u8; 3])> {
        let mut lines = Vec::new();
        let count = (self.extent / self.size) as i32;
        for i in -count..=count {
            let pos = i as f32 * self.size;
            let color = if i % self.major_every as i32 == 0 {
                self.major_color
            } else {
                self.color
            };
            // X-axis lines
            lines.push((
                Vec3::new(pos, 0.0, -self.extent),
                Vec3::new(pos, 0.0, self.extent),
                color,
            ));
            // Z-axis lines
            lines.push((
                Vec3::new(-self.extent, 0.0, pos),
                Vec3::new(self.extent, 0.0, pos),
                color,
            ));
        }
        lines
    }
}

/// Snap settings.
pub struct SnapSettings {
    pub grid_snap: bool,
    pub grid_size: f32,
    pub rotation_snap: bool,
    pub rotation_increment: f32,
    pub scale_snap: bool,
    pub scale_increment: f32,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            grid_snap: false,
            grid_size: 1.0,
            rotation_snap: false,
            rotation_increment: 15.0,
            scale_snap: false,
            scale_increment: 0.25,
        }
    }
}

impl SnapSettings {
    pub fn snap_position(&self, pos: Vec3) -> Vec3 {
        if !self.grid_snap {
            return pos;
        }
        Vec3::new(
            (pos.x / self.grid_size).round() * self.grid_size,
            (pos.y / self.grid_size).round() * self.grid_size,
            (pos.z / self.grid_size).round() * self.grid_size,
        )
    }

    pub fn snap_rotation(&self, degrees: f32) -> f32 {
        if !self.rotation_snap {
            return degrees;
        }
        (degrees / self.rotation_increment).round() * self.rotation_increment
    }

    pub fn snap_scale(&self, scale: f32) -> f32 {
        if !self.scale_snap {
            return scale;
        }
        (scale / self.scale_increment).round() * self.scale_increment
    }
}

/// Clipboard for copy/paste operations.
pub struct EditorClipboard {
    pub entities: Vec<ClipboardEntity>,
}

#[derive(Debug, Clone)]
pub struct ClipboardEntity {
    pub name: String,
    pub position: Vec3,
    pub rotation: [f32; 4],
    pub scale: Vec3,
    pub asset_path: Option<String>,
    pub scripts: Vec<String>,
    pub tags: Vec<String>,
}

impl EditorClipboard {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
        }
    }

    pub fn copy(&mut self, entities: Vec<ClipboardEntity>) {
        self.entities = entities;
    }

    pub fn paste(&self, offset: Vec3) -> Vec<ClipboardEntity> {
        self.entities
            .iter()
            .map(|e| {
                let mut cloned = e.clone();
                cloned.position += offset;
                cloned.name = format!("{} (copy)", e.name);
                cloned
            })
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    pub fn count(&self) -> usize {
        self.entities.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_generates_correct_line_count() {
        let grid = EditorGrid::default();
        let lines = grid.generate_lines();
        // count = (50/1) = 50, range -50..=50 = 101 values
        // 101 X-lines + 101 Z-lines = 202
        assert_eq!(lines.len(), 202);
    }

    #[test]
    fn grid_major_lines_use_major_color() {
        let grid = EditorGrid::default();
        let lines = grid.generate_lines();
        // Line at i=0 should be major (0 % 10 == 0)
        let origin_line = lines.iter().find(|(a, _, _)| a.x == 0.0 && a.z < 0.0).unwrap();
        assert_eq!(origin_line.2, grid.major_color);
    }

    #[test]
    fn snap_rounds_to_grid() {
        let mut snap = SnapSettings::default();
        snap.grid_snap = true;
        snap.grid_size = 1.0;

        let pos = Vec3::new(1.3, 2.7, -0.4);
        let snapped = snap.snap_position(pos);
        assert_eq!(snapped, Vec3::new(1.0, 3.0, 0.0));
    }

    #[test]
    fn snap_disabled_passes_through() {
        let snap = SnapSettings::default(); // grid_snap = false
        let pos = Vec3::new(1.3, 2.7, -0.4);
        assert_eq!(snap.snap_position(pos), pos);
    }

    #[test]
    fn rotation_snap_works() {
        let mut snap = SnapSettings::default();
        snap.rotation_snap = true;
        snap.rotation_increment = 15.0;

        assert!((snap.snap_rotation(17.0) - 15.0).abs() < 0.001);
        assert!((snap.snap_rotation(23.0) - 30.0).abs() < 0.001);
        assert!((snap.snap_rotation(45.0) - 45.0).abs() < 0.001);
    }

    #[test]
    fn scale_snap_works() {
        let mut snap = SnapSettings::default();
        snap.scale_snap = true;
        snap.scale_increment = 0.25;

        assert!((snap.snap_scale(0.3) - 0.25).abs() < 0.001);
        assert!((snap.snap_scale(0.6) - 0.5).abs() < 0.001);
        assert!((snap.snap_scale(1.1) - 1.0).abs() < 0.001);
    }

    #[test]
    fn clipboard_copy_paste_with_offset() {
        let mut clipboard = EditorClipboard::new();
        assert!(clipboard.is_empty());

        clipboard.copy(vec![ClipboardEntity {
            name: "Cube".into(),
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: Vec3::ONE,
            asset_path: Some("cube.vxm".into()),
            scripts: vec![],
            tags: vec!["static".into()],
        }]);

        assert!(!clipboard.is_empty());
        assert_eq!(clipboard.count(), 1);

        let pasted = clipboard.paste(Vec3::new(5.0, 0.0, 0.0));
        assert_eq!(pasted.len(), 1);
        assert_eq!(pasted[0].position, Vec3::new(6.0, 2.0, 3.0));
    }

    #[test]
    fn paste_renames_with_copy() {
        let mut clipboard = EditorClipboard::new();
        clipboard.copy(vec![ClipboardEntity {
            name: "Tree".into(),
            position: Vec3::ZERO,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: Vec3::ONE,
            asset_path: None,
            scripts: vec![],
            tags: vec![],
        }]);
        let pasted = clipboard.paste(Vec3::ZERO);
        assert_eq!(pasted[0].name, "Tree (copy)");
    }

    #[test]
    fn empty_clipboard() {
        let clipboard = EditorClipboard::new();
        assert!(clipboard.is_empty());
        assert_eq!(clipboard.count(), 0);
        let pasted = clipboard.paste(Vec3::ZERO);
        assert!(pasted.is_empty());
    }
}
