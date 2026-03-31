use vox_core::types::GaussianSplat;
use std::collections::VecDeque;

/// Manages semi-transparent NPC path ghost overlays in Simulate mode.
pub struct GhostOverlays {
    enabled: bool,
    history: Vec<VecDeque<[f32; 3]>>,
}

impl GhostOverlays {
    pub const MAX_HISTORY_FRAMES: usize = 120;

    pub fn new() -> Self {
        Self { enabled: false, history: Vec::new() }
    }

    pub fn enabled(&self) -> bool { self.enabled }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.history.clear();
        }
    }

    pub fn agent_count(&self) -> usize { self.history.len() }

    pub fn history_for_agent(&self, agent_idx: usize) -> &VecDeque<[f32; 3]> {
        &self.history[agent_idx]
    }

    pub fn update(&mut self, positions: &[[f32; 3]]) {
        if !self.enabled {
            return;
        }
        self.history.resize_with(positions.len(), VecDeque::new);
        for (i, &pos) in positions.iter().enumerate() {
            let buf = &mut self.history[i];
            buf.push_back(pos);
            if buf.len() > Self::MAX_HISTORY_FRAMES {
                buf.pop_front();
            }
        }
    }

    pub fn generate_path_splats(&self) -> Vec<GaussianSplat> {
        if !self.enabled {
            return Vec::new();
        }
        let mut splats = Vec::new();
        for buf in &self.history {
            let len = buf.len();
            for (frame_idx, &pos) in buf.iter().enumerate() {
                let age = frame_idx as f32 / len.max(1) as f32;
                let alpha = age * 0.35;
                splats.push(make_ghost_splat(pos, alpha));
            }
        }
        splats
    }
}

fn make_ghost_splat(pos: [f32; 3], alpha: f32) -> GaussianSplat {
    let opacity = (alpha * 255.0) as u8;
    // Cyan ghost: boost spectral bands ~480-530nm (indices 4-6 in 380-755nm at 25nm steps)
    let v = (alpha * 30000.0) as u16;
    let mut spectral = [0u16; 16];
    spectral[4] = v;
    spectral[5] = v;
    spectral[6] = (v as f32 * 0.7) as u16;
    GaussianSplat::surface(
        pos,
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        0.15,
        0.15,
        opacity,
        spectral,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghost_overlays_start_disabled() {
        let overlays = GhostOverlays::new();
        assert!(!overlays.enabled());
    }

    #[test]
    fn enable_disable_toggles_state() {
        let mut overlays = GhostOverlays::new();
        overlays.set_enabled(true);
        assert!(overlays.enabled());
        overlays.set_enabled(false);
        assert!(!overlays.enabled());
    }

    #[test]
    fn update_adds_position_to_history() {
        let mut overlays = GhostOverlays::new();
        overlays.set_enabled(true);
        let positions: &[[f32; 3]] = &[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        overlays.update(positions);
        assert_eq!(overlays.agent_count(), 2);
        assert_eq!(overlays.history_for_agent(0).len(), 1);
    }

    #[test]
    fn history_capped_at_max_frames() {
        let mut overlays = GhostOverlays::new();
        overlays.set_enabled(true);
        let pos: &[[f32; 3]] = &[[0.0, 0.0, 0.0]];
        for _ in 0..100 {
            overlays.update(pos);
        }
        assert!(overlays.history_for_agent(0).len() <= GhostOverlays::MAX_HISTORY_FRAMES);
    }

    #[test]
    fn generate_path_splats_produces_splats_per_history_point() {
        let mut overlays = GhostOverlays::new();
        overlays.set_enabled(true);
        overlays.update(&[[1.0, 0.0, 0.0]]);
        overlays.update(&[[2.0, 0.0, 0.0]]);
        overlays.update(&[[3.0, 0.0, 0.0]]);
        let splats = overlays.generate_path_splats();
        assert!(!splats.is_empty());
        assert!(splats.len() >= 3);
    }

    #[test]
    fn no_splats_when_disabled() {
        let mut overlays = GhostOverlays::new();
        overlays.update(&[[1.0, 0.0, 0.0]]);
        let splats = overlays.generate_path_splats();
        assert!(splats.is_empty());
    }
}
