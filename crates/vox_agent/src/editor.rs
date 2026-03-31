use egui::Ui;
use crate::node_graph::{AgentNodeGraph, AgentNodeKind, AgentNodeRegistry};
use crate::desc::AgentStateDesc;
use crate::codegen::AgentShaderGen;

pub struct AgentNodeEditor {
    graph: AgentNodeGraph,
    registry: AgentNodeRegistry,
    status: String,
    pub(crate) pending_wgsl: Option<String>,
    next_pos: [f32; 2],
}

impl AgentNodeEditor {
    pub fn new() -> Self {
        Self {
            graph: AgentNodeGraph::new("default"),
            registry: AgentNodeRegistry::new(),
            status: "No shader compiled".to_string(),
            pending_wgsl: None,
            next_pos: [50.0, 50.0],
        }
    }

    fn alloc_pos(&mut self) -> [f32; 2] {
        let pos = self.next_pos;
        self.next_pos[0] += 20.0;
        if self.next_pos[0] > 400.0 {
            self.next_pos[0] = 50.0;
            self.next_pos[1] += 40.0;
        }
        pos
    }

    /// Returns compiled WGSL if a compile was triggered this frame.
    pub fn take_pending_wgsl(&mut self) -> Option<String> {
        self.pending_wgsl.take()
    }

    pub fn show(&mut self, ui: &mut Ui, desc: &AgentStateDesc) {
        ui.label("Agent Node Editor");
        ui.separator();

        // Palette sidebar — compute clicks before entering closures to avoid borrow conflicts
        let clicked_get_position  = ui.small_button("GetPosition").clicked();
        let clicked_get_velocity  = ui.small_button("GetVelocity").clicked();
        let clicked_set_velocity  = ui.small_button("SetVelocity").clicked();
        let clicked_add_velocity  = ui.small_button("AddVelocity").clicked();
        let clicked_normalize     = ui.small_button("Normalize").clicked();
        let clicked_noise         = ui.small_button("Noise").clicked();
        let clicked_spectral      = desc.spectral && ui.small_button("SampleSpectral").clicked();

        if clicked_get_position  { let pos = self.alloc_pos(); self.graph.add_node(AgentNodeKind::GetPosition, pos); }
        if clicked_get_velocity  { let pos = self.alloc_pos(); self.graph.add_node(AgentNodeKind::GetVelocity, pos); }
        if clicked_set_velocity  { let pos = self.alloc_pos(); self.graph.add_node(AgentNodeKind::SetVelocity, pos); }
        if clicked_add_velocity  { let pos = self.alloc_pos(); self.graph.add_node(AgentNodeKind::AddVelocity, pos); }
        if clicked_normalize     { let pos = self.alloc_pos(); self.graph.add_node(AgentNodeKind::Normalize, pos); }
        if clicked_noise         { let pos = self.alloc_pos(); self.graph.add_node(AgentNodeKind::Noise, pos); }
        if clicked_spectral      { let pos = self.alloc_pos(); self.graph.add_node(AgentNodeKind::SampleSpectral { band: 5 }, pos); }

        ui.horizontal(|ui| {
            ui.group(|ui| {
                ui.label("Nodes");
                ui.separator();
                ui.label("Use buttons above to add nodes");
            });

            ui.group(|ui| {
                ui.label("Graph");
                ui.label(format!("{} nodes, {} connections",
                    self.graph.nodes().len(), self.graph.connections().len()));
            });
        });

        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Compile").clicked() {
                match AgentShaderGen::generate(&self.graph, &self.registry, desc) {
                    Ok(wgsl) => {
                        self.status = "Compiled OK".to_string();
                        self.pending_wgsl = Some(wgsl.source);
                    }
                    Err(e) => {
                        self.status = format!("Error: {e}");
                    }
                }
            }
            ui.label(&self.status);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_new_has_empty_graph() {
        let editor = AgentNodeEditor::new();
        assert_eq!(editor.graph.nodes().len(), 0);
    }

    #[test]
    fn take_pending_wgsl_clears_after_first_call() {
        let mut editor = AgentNodeEditor::new();
        // Inject a pending value directly to test take semantics
        editor.pending_wgsl = Some("// test wgsl".to_string());
        let first = editor.take_pending_wgsl();
        let second = editor.take_pending_wgsl();
        assert_eq!(first.as_deref(), Some("// test wgsl"), "first call must return the pending value");
        assert!(second.is_none(), "second call must return None (value was consumed)");
    }

    #[test]
    fn editor_renders_without_panic() {
        let ctx = egui::Context::default();
        let desc = crate::desc::AgentStateDesc {
            agent_count: 10, custom_floats: 0, spectral: false, spatial_hash: None,
        };
        ctx.run(egui::RawInput::default(), |ctx| {
            egui::Window::new("test").show(ctx, |ui| {
                let mut editor = AgentNodeEditor::new();
                editor.show(ui, &desc);
            });
        });
    }
}
