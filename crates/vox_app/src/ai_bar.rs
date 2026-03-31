use crate::editor_app::{AiScope, JobHandle, NodeGraphDiff};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Context sent with each AI prompt.
#[derive(Clone)]
pub struct AiContext {
    pub scope: AiScope,
    pub selection_ids: Vec<u32>,
}

/// The result of an AI action.
#[derive(Clone)]
pub struct AiResult {
    /// If the AI modified a node graph, this diff describes the changes.
    pub diff: Option<NodeGraphDiff>,
    /// Human-readable summary of what the AI did.
    pub summary: String,
}

/// The AI backend trait.
pub trait AiBackend: Send + 'static {
    fn submit(&self, prompt: &str, context: AiContext) -> JobHandle;
    fn poll(&self, handle: &JobHandle) -> Option<AiResult>;
}

/// A synchronous stub backend used in Phase 1 and tests.
pub struct StubAiBackend {
    next_id: Arc<AtomicU64>,
}

impl StubAiBackend {
    pub fn new() -> Self {
        Self { next_id: Arc::new(AtomicU64::new(1)) }
    }
}

impl AiBackend for StubAiBackend {
    fn submit(&self, _prompt: &str, _context: AiContext) -> JobHandle {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        JobHandle(id)
    }

    fn poll(&self, _handle: &JobHandle) -> Option<AiResult> {
        Some(AiResult {
            diff: None,
            summary: "AI action applied (stub)".into(),
        })
    }
}

/// All mutable state for the AI bar.
pub struct AiBarState {
    scope: AiScope,
    expanded: bool,
    input_text: String,
    history: Vec<(String, AiResult)>,
    pending: Option<JobHandle>,
}

impl AiBarState {
    pub fn new() -> Self {
        Self {
            scope: AiScope::Selection,
            expanded: false,
            input_text: String::new(),
            history: Vec::new(),
            pending: None,
        }
    }

    pub fn scope(&self) -> AiScope { self.scope }

    pub fn cycle_scope(&mut self) {
        self.scope = self.scope.next();
    }

    pub fn is_expanded(&self) -> bool { self.expanded }

    pub fn toggle_expand(&mut self) { self.expanded = !self.expanded; }

    pub fn input_text(&self) -> &str { &self.input_text }

    pub fn input_text_mut(&mut self) -> &mut String { &mut self.input_text }

    pub fn history(&self) -> &[(String, AiResult)] { &self.history }

    pub fn submit(
        &mut self,
        backend: &dyn AiBackend,
        selection_ids: Vec<u32>,
    ) -> Option<JobHandle> {
        let text = self.input_text.trim().to_owned();
        if text.is_empty() {
            return None;
        }
        let handle = backend.submit(&text, AiContext {
            scope: self.scope,
            selection_ids,
        });
        self.pending = Some(handle);
        self.input_text.clear();
        Some(handle)
    }

    pub fn tick(&mut self, backend: &dyn AiBackend) -> Option<AiResult> {
        let handle = self.pending?;
        let result = backend.poll(&handle)?;
        self.pending = None;
        self.history.push(("(prompt)".into(), result.clone()));
        Some(result)
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        backend: &dyn AiBackend,
        selection_ids: Vec<u32>,
        on_result: &mut dyn FnMut(AiResult),
    ) {
        if let Some(result) = self.tick(backend) {
            on_result(result);
        }

        ui.horizontal(|ui| {
            let scope_label = match self.scope {
                AiScope::Selection => "Sel",
                AiScope::Mode      => "Mode",
                AiScope::Scene     => "Scene",
            };
            if ui.button(scope_label).clicked() {
                self.cycle_scope();
            }

            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.input_text)
                    .hint_text("Describe what you want…")
                    .desired_width(ui.available_width() - 80.0),
            );

            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.submit(backend, selection_ids);
            }

            let expand_label = if self.expanded { "▾" } else { "▸" };
            if ui.button(expand_label).clicked() {
                self.toggle_expand();
            }
        });

        if self.expanded {
            ui.separator();
            egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                for (prompt, result) in self.history.iter().rev() {
                    ui.label(format!("> {}", prompt));
                    ui.label(
                        egui::RichText::new(&result.summary)
                            .color(egui::Color32::from_gray(160))
                            .small()
                    );
                    ui.separator();
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ai_bar_default_scope_is_selection() {
        let bar = AiBarState::new();
        assert_eq!(bar.scope(), AiScope::Selection);
    }

    #[test]
    fn ai_bar_cycle_scope_selection_to_mode() {
        let mut bar = AiBarState::new();
        bar.cycle_scope();
        assert_eq!(bar.scope(), AiScope::Mode);
    }

    #[test]
    fn ai_bar_cycle_scope_wraps_scene_to_selection() {
        let mut bar = AiBarState::new();
        bar.cycle_scope();
        bar.cycle_scope();
        bar.cycle_scope();
        assert_eq!(bar.scope(), AiScope::Selection);
    }

    #[test]
    fn ai_bar_starts_collapsed() {
        let bar = AiBarState::new();
        assert!(!bar.is_expanded());
    }

    #[test]
    fn ai_bar_toggle_expand_changes_state() {
        let mut bar = AiBarState::new();
        bar.toggle_expand();
        assert!(bar.is_expanded());
        bar.toggle_expand();
        assert!(!bar.is_expanded());
    }

    #[test]
    fn ai_bar_input_text_initially_empty() {
        let bar = AiBarState::new();
        assert_eq!(bar.input_text(), "");
    }

    #[test]
    fn stub_backend_returns_handle_on_submit() {
        let backend = StubAiBackend::new();
        let handle = backend.submit("make it more mossy", AiContext {
            scope: AiScope::Selection,
            selection_ids: vec![1],
        });
        assert_eq!(handle.0, 1);
    }

    #[test]
    fn stub_backend_poll_returns_result_immediately() {
        let backend = StubAiBackend::new();
        let handle = backend.submit("test", AiContext {
            scope: AiScope::Scene,
            selection_ids: vec![],
        });
        assert!(backend.poll(&handle).is_some());
    }
}
