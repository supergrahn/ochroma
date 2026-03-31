use crate::desc::AgentStateDesc;

pub struct AgentNodeEditor;

impl AgentNodeEditor {
    pub fn new() -> Self { Self }

    pub fn show(&mut self, _ui: &mut egui::Ui, _desc: &AgentStateDesc) {}
}
