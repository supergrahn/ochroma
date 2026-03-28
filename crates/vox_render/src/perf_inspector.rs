use serde::{Deserialize, Serialize};

/// Breakdown of frame timing in milliseconds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrameBreakdown {
    pub total_ms: f32,
    pub sort_ms: f32,
    pub cull_ms: f32,
    pub render_ms: f32,
    pub ui_ms: f32,
    pub sim_ms: f32,
}

impl FrameBreakdown {
    /// Sum of individual components.
    pub fn component_sum(&self) -> f32 {
        self.sort_ms + self.cull_ms + self.render_ms + self.ui_ms + self.sim_ms
    }
}

/// Breakdown of VRAM usage in megabytes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VramBreakdown {
    pub total_mb: f32,
    pub splats_mb: f32,
    pub textures_mb: f32,
    pub buffers_mb: f32,
}

/// Breakdown of entity counts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityBreakdown {
    pub total: u32,
    pub buildings: u32,
    pub citizens: u32,
    pub vehicles: u32,
    pub trees: u32,
    pub props: u32,
}

/// Snapshot of all performance metrics for a single frame.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerfSnapshot {
    pub frame: FrameBreakdown,
    pub vram: VramBreakdown,
    pub entities: EntityBreakdown,
}

/// In-game performance inspector overlay.
pub struct PerfInspector {
    pub visible: bool,
    history: Vec<PerfSnapshot>,
    max_history: usize,
}

impl PerfInspector {
    /// Create a new inspector with default settings.
    pub fn new() -> Self {
        Self {
            visible: false,
            history: Vec::new(),
            max_history: 300, // ~5 seconds at 60fps
        }
    }

    /// Toggle overlay visibility.
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Record a frame's performance data.
    pub fn record_frame(&mut self, snapshot: PerfSnapshot) {
        self.history.push(snapshot);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get the number of recorded frames.
    pub fn frame_count(&self) -> usize {
        self.history.len()
    }

    /// Compute average frame breakdown over the last `n` frames.
    pub fn average_over(&self, n: usize) -> FrameBreakdown {
        let count = n.min(self.history.len());
        if count == 0 {
            return FrameBreakdown::default();
        }

        let start = self.history.len() - count;
        let frames = &self.history[start..];

        let mut avg = FrameBreakdown::default();
        for snap in frames {
            avg.total_ms += snap.frame.total_ms;
            avg.sort_ms += snap.frame.sort_ms;
            avg.cull_ms += snap.frame.cull_ms;
            avg.render_ms += snap.frame.render_ms;
            avg.ui_ms += snap.frame.ui_ms;
            avg.sim_ms += snap.frame.sim_ms;
        }

        let c = count as f32;
        avg.total_ms /= c;
        avg.sort_ms /= c;
        avg.cull_ms /= c;
        avg.render_ms /= c;
        avg.ui_ms /= c;
        avg.sim_ms /= c;

        avg
    }

    /// Get the latest snapshot.
    pub fn latest(&self) -> Option<&PerfSnapshot> {
        self.history.last()
    }

    /// Export all recorded data as JSON.
    pub fn export_json(&self) -> String {
        serde_json::to_string_pretty(&self.history).unwrap_or_else(|_| "[]".to_string())
    }

    /// Render the performance overlay using egui.
    #[allow(unexpected_cfgs)]
    #[cfg(feature = "egui")]
    pub fn show(&self, ctx: &egui::Context) {
        if !self.visible {
            return;
        }

        let avg = self.average_over(60);
        let fps = if avg.total_ms > 0.0 {
            1000.0 / avg.total_ms
        } else {
            0.0
        };

        egui::Window::new("Performance Inspector")
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("Frame Timing");
                ui.label(format!("FPS: {fps:.1}"));
                ui.label(format!("Total: {:.2}ms", avg.total_ms));
                ui.label(format!("  Sort:   {:.2}ms", avg.sort_ms));
                ui.label(format!("  Cull:   {:.2}ms", avg.cull_ms));
                ui.label(format!("  Render: {:.2}ms", avg.render_ms));
                ui.label(format!("  UI:     {:.2}ms", avg.ui_ms));
                ui.label(format!("  Sim:    {:.2}ms", avg.sim_ms));

                if let Some(snap) = self.latest() {
                    ui.separator();
                    ui.heading("VRAM");
                    ui.label(format!("Total:    {:.1} MB", snap.vram.total_mb));
                    ui.label(format!("  Splats:   {:.1} MB", snap.vram.splats_mb));
                    ui.label(format!("  Textures: {:.1} MB", snap.vram.textures_mb));
                    ui.label(format!("  Buffers:  {:.1} MB", snap.vram.buffers_mb));

                    ui.separator();
                    ui.heading("Entities");
                    ui.label(format!("Total: {}", snap.entities.total));
                    ui.label(format!("  Buildings: {}", snap.entities.buildings));
                    ui.label(format!("  Citizens:  {}", snap.entities.citizens));
                    ui.label(format!("  Vehicles:  {}", snap.entities.vehicles));
                    ui.label(format!("  Trees:     {}", snap.entities.trees));
                    ui.label(format!("  Props:     {}", snap.entities.props));
                }
            });
    }
}

impl Default for PerfInspector {
    fn default() -> Self {
        Self::new()
    }
}
