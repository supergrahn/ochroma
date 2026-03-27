/// Editor panels: Outliner tree + Details panel.

use std::collections::HashSet;

/// Editor panel state for the outliner tree.
pub struct OutlinerState {
    pub expanded: HashSet<u32>,
    pub selected: Vec<u32>,
    pub search_query: String,
    pub context_menu_target: Option<u32>,
}

impl OutlinerState {
    pub fn new() -> Self {
        Self {
            expanded: HashSet::new(),
            selected: Vec::new(),
            search_query: String::new(),
            context_menu_target: None,
        }
    }

    pub fn toggle_expand(&mut self, id: u32) {
        if !self.expanded.remove(&id) {
            self.expanded.insert(id);
        }
    }

    pub fn is_expanded(&self, id: u32) -> bool {
        self.expanded.contains(&id)
    }

    pub fn select(&mut self, id: u32, multi: bool) {
        if multi {
            if !self.selected.contains(&id) {
                self.selected.push(id);
            }
        } else {
            self.selected.clear();
            self.selected.push(id);
        }
    }

    pub fn deselect_all(&mut self) {
        self.selected.clear();
    }

    pub fn is_selected(&self, id: u32) -> bool {
        self.selected.contains(&id)
    }

    pub fn selected_count(&self) -> usize {
        self.selected.len()
    }
}

/// Entity data needed to render the outliner row.
pub struct OutlinerEntry {
    pub id: u32,
    pub name: String,
    pub entity_type: EntityTypeIcon,
    pub visible: bool,
    pub locked: bool,
    pub depth: u32,
    pub has_children: bool,
    pub parent: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityTypeIcon {
    Mesh,
    Light,
    Audio,
    Camera,
    Script,
    Empty,
    Terrain,
    Particle,
}

/// Context menu actions from the outliner.
#[derive(Debug, Clone)]
pub enum OutlinerAction {
    Rename(u32, String),
    Delete(Vec<u32>),
    Duplicate(Vec<u32>),
    CreateChild(u32),
    ToggleVisibility(u32),
    ToggleLock(u32),
    SelectAll,
    Group(Vec<u32>),
}

/// Details panel auto-generation for entity components.
pub struct DetailsPanel {
    pub editing_name: Option<String>,
}

impl DetailsPanel {
    pub fn new() -> Self {
        Self {
            editing_name: None,
        }
    }
}

/// A property that can be displayed and edited.
#[derive(Debug, Clone)]
pub enum PropertyValue {
    Float(f32),
    Vec3([f32; 3]),
    Quat([f32; 4]),
    Bool(bool),
    String(String),
    Color([f32; 3]),
    FilePath(String),
    Enum {
        options: Vec<String>,
        selected: usize,
    },
    Range {
        value: f32,
        min: f32,
        max: f32,
    },
}

/// A section of properties for a component.
pub struct PropertySection {
    pub name: String,
    pub properties: Vec<(String, PropertyValue)>,
    pub collapsed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outliner_select_deselect() {
        let mut state = OutlinerState::new();
        state.select(1, false);
        assert!(state.is_selected(1));
        assert_eq!(state.selected_count(), 1);

        state.deselect_all();
        assert!(!state.is_selected(1));
        assert_eq!(state.selected_count(), 0);
    }

    #[test]
    fn outliner_multi_select() {
        let mut state = OutlinerState::new();
        state.select(1, false);
        state.select(2, true);
        state.select(3, true);
        assert_eq!(state.selected_count(), 3);
        assert!(state.is_selected(1));
        assert!(state.is_selected(2));
        assert!(state.is_selected(3));
    }

    #[test]
    fn single_select_replaces() {
        let mut state = OutlinerState::new();
        state.select(1, false);
        state.select(2, false);
        assert_eq!(state.selected_count(), 1);
        assert!(!state.is_selected(1));
        assert!(state.is_selected(2));
    }

    #[test]
    fn expand_collapse() {
        let mut state = OutlinerState::new();
        assert!(!state.is_expanded(5));
        state.toggle_expand(5);
        assert!(state.is_expanded(5));
        state.toggle_expand(5);
        assert!(!state.is_expanded(5));
    }

    #[test]
    fn search_query_filter() {
        let mut state = OutlinerState::new();
        state.search_query = "light".to_string();

        let entries = vec![
            OutlinerEntry {
                id: 1,
                name: "PointLight_01".to_string(),
                entity_type: EntityTypeIcon::Light,
                visible: true,
                locked: false,
                depth: 0,
                has_children: false,
                parent: None,
            },
            OutlinerEntry {
                id: 2,
                name: "MeshCube".to_string(),
                entity_type: EntityTypeIcon::Mesh,
                visible: true,
                locked: false,
                depth: 0,
                has_children: false,
                parent: None,
            },
        ];

        let query = state.search_query.to_lowercase();
        let filtered: Vec<_> = entries
            .iter()
            .filter(|e| e.name.to_lowercase().contains(&query))
            .collect();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "PointLight_01");
    }

    #[test]
    fn entity_type_classification() {
        assert_eq!(EntityTypeIcon::Mesh, EntityTypeIcon::Mesh);
        assert_ne!(EntityTypeIcon::Mesh, EntityTypeIcon::Light);
        assert_ne!(EntityTypeIcon::Camera, EntityTypeIcon::Script);
    }

    #[test]
    fn property_value_types() {
        let f = PropertyValue::Float(3.14);
        let b = PropertyValue::Bool(true);
        let s = PropertyValue::String("hello".into());
        let e = PropertyValue::Enum {
            options: vec!["A".into(), "B".into()],
            selected: 0,
        };
        let r = PropertyValue::Range {
            value: 0.5,
            min: 0.0,
            max: 1.0,
        };

        // Just verify they construct without panic
        match f {
            PropertyValue::Float(v) => assert!((v - 3.14).abs() < 0.001),
            _ => panic!("wrong variant"),
        }
        match b {
            PropertyValue::Bool(v) => assert!(v),
            _ => panic!("wrong variant"),
        }
        match s {
            PropertyValue::String(ref v) => assert_eq!(v, "hello"),
            _ => panic!("wrong variant"),
        }
        match e {
            PropertyValue::Enum { selected, .. } => assert_eq!(selected, 0),
            _ => panic!("wrong variant"),
        }
        match r {
            PropertyValue::Range { value, min, max } => {
                assert!((value - 0.5).abs() < 0.001);
                assert!((min - 0.0).abs() < 0.001);
                assert!((max - 1.0).abs() < 0.001);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn details_panel_new() {
        let panel = DetailsPanel::new();
        assert!(panel.editing_name.is_none());
    }
}
