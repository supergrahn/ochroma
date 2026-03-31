use crate::editor::EditorEntity;

/// Renders the scene hierarchy with AI intent subtitles.
pub struct SceneTree {
    selected: Option<u32>,
}

impl SceneTree {
    pub fn new() -> Self {
        Self { selected: None }
    }

    pub fn selected(&self) -> Option<u32> { self.selected }

    pub fn set_selected(&mut self, id: Option<u32>) { self.selected = id; }

    /// Returns only root entities (those with no parent).
    pub fn root_entities(entities: &[EditorEntity]) -> Vec<&EditorEntity> {
        entities.iter().filter(|e| e.parent.is_none()).collect()
    }

    /// Returns true if the entity has an AI intent prompt to display.
    pub fn has_visible_intent(entity: &EditorEntity) -> bool {
        entity.intent().is_some()
    }

    /// Render the scene tree inside `ui`. Calls `on_select` when an entity is clicked.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        entities: &[EditorEntity],
        on_select: &mut dyn FnMut(u32),
    ) {
        let roots = Self::root_entities(entities);
        for entity in roots {
            self.show_entity(ui, entity, entities, on_select);
        }
    }

    fn show_entity(
        &mut self,
        ui: &mut egui::Ui,
        entity: &EditorEntity,
        all_entities: &[EditorEntity],
        on_select: &mut dyn FnMut(u32),
    ) {
        let selected = self.selected == Some(entity.id);
        let has_children = !entity.children.is_empty();

        if has_children {
            egui::CollapsingHeader::new(&entity.name)
                .show(ui, |ui| {
                    ui.vertical(|ui| {
                        let resp = ui.selectable_label(selected, &entity.name);
                        if resp.clicked() {
                            on_select(entity.id);
                        }
                        if let Some(intent) = entity.intent() {
                            ui.label(
                                egui::RichText::new(intent)
                                    .small()
                                    .color(egui::Color32::from_gray(130))
                            );
                        }
                    });
                    for &child_id in &entity.children {
                        if let Some(child) = all_entities.iter().find(|e| e.id == child_id) {
                            self.show_entity(ui, child, all_entities, on_select);
                        }
                    }
                });
        } else {
            ui.vertical(|ui| {
                let resp = ui.selectable_label(selected, &entity.name);
                if resp.clicked() {
                    on_select(entity.id);
                }
                if let Some(intent) = entity.intent() {
                    ui.label(
                        egui::RichText::new(intent)
                            .small()
                            .color(egui::Color32::from_gray(130))
                    );
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    fn make_entity(id: u32, name: &str, intent: Option<&str>) -> EditorEntity {
        let mut e = EditorEntity {
            id,
            name: name.into(),
            asset_path: "".into(),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            visible: true,
            locked: false,
            scripts: vec![],
            parent: None,
            children: vec![],
            intent: None,
        };
        if let Some(p) = intent {
            e.set_intent(p);
        }
        e
    }

    #[test]
    fn scene_tree_collects_root_entities() {
        let entities = vec![
            make_entity(1, "Tower", Some("a ruined watchtower")),
            make_entity(2, "Tree", None),
        ];
        let roots = SceneTree::root_entities(&entities);
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn scene_tree_root_excludes_children() {
        let entities = vec![
            make_entity(1, "Parent", None),
            EditorEntity {
                id: 2,
                name: "Child".into(),
                parent: Some(1),
                children: vec![],
                intent: None,
                asset_path: "".into(),
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                visible: true,
                locked: false,
                scripts: vec![],
            },
        ];
        let roots = SceneTree::root_entities(&entities);
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, 1);
    }

    #[test]
    fn scene_tree_intent_visible_for_ai_entity() {
        let e = make_entity(1, "Ruin", Some("a crumbling stone ruin"));
        assert!(SceneTree::has_visible_intent(&e));
    }

    #[test]
    fn scene_tree_no_intent_for_manual_entity() {
        let e = make_entity(2, "Box", None);
        assert!(!SceneTree::has_visible_intent(&e));
    }
}
