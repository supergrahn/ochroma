use egui::Ui;
use crate::node_graph::{AgentNodeGraph, AgentNodeKind, AgentNodeRegistry};
use crate::desc::AgentStateDesc;
use crate::codegen::AgentShaderGen;

pub struct AgentNodeEditor {
    graph: AgentNodeGraph,
    registry: AgentNodeRegistry,
    status: String,
    pending_wgsl: Option<String>,
}

impl AgentNodeEditor {
    pub fn new() -> Self {
        Self {
            graph: AgentNodeGraph::new("default"),
            registry: AgentNodeRegistry::new(),
            status: "No shader compiled".to_string(),
            pending_wgsl: None,
        }
    }

    /// Returns compiled WGSL if a compile was triggered this frame.
    pub fn take_pending_wgsl(&mut self) -> Option<String> {
        self.pending_wgsl.take()
    }

    pub fn show(&mut self, ui: &mut Ui, desc: &AgentStateDesc) {
        ui.label("Agent Node Editor");
        ui.separator();

        // Palette sidebar
        ui.horizontal(|ui| {
            ui.group(|ui| {
                ui.label("Nodes");
                ui.separator();
                if ui.small_button("GetPosition").clicked() {
                    self.graph.add_node(AgentNodeKind::GetPosition, [50.0, 50.0]);
                }
                if ui.small_button("GetVelocity").clicked() {
                    self.graph.add_node(AgentNodeKind::GetVelocity, [50.0, 100.0]);
                }
                if ui.small_button("SetVelocity").clicked() {
                    self.graph.add_node(AgentNodeKind::SetVelocity, [250.0, 50.0]);
                }
                if ui.small_button("AddVelocity").clicked() {
                    self.graph.add_node(AgentNodeKind::AddVelocity, [250.0, 100.0]);
                }
                if ui.small_button("Normalize").clicked() {
                    self.graph.add_node(AgentNodeKind::Normalize, [150.0, 50.0]);
                }
                if ui.small_button("Noise").clicked() {
                    self.graph.add_node(AgentNodeKind::Noise, [150.0, 100.0]);
                }
                if desc.spectral {
                    if ui.small_button("SampleSpectral").clicked() {
                        self.graph.add_node(AgentNodeKind::SampleSpectral { band: 5 }, [50.0, 150.0]);
                    }
                }
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
    fn take_pending_wgsl_returns_none_initially() {
        let mut editor = AgentNodeEditor::new();
        assert!(editor.take_pending_wgsl().is_none());
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
