use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct FrameProfile {
    pub frame_number: u64,
    pub total_ms: f32,
    pub phases: Vec<PhaseProfile>,
}

#[derive(Debug, Clone)]
pub struct PhaseProfile {
    pub name: String,
    pub start_ms: f32,
    pub duration_ms: f32,
    pub children: Vec<PhaseProfile>,
}

pub struct FrameDebugger {
    pub history: VecDeque<FrameProfile>,
    pub max_history: usize,
    pub recording: bool,
    current_frame: Option<FrameProfile>,
    phase_stack: Vec<(String, std::time::Instant)>,
    frame_start: Option<std::time::Instant>,
}

impl FrameDebugger {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::new(),
            max_history,
            recording: true,
            current_frame: None,
            phase_stack: Vec::new(),
            frame_start: None,
        }
    }

    pub fn begin_frame(&mut self, frame_number: u64) {
        if !self.recording {
            return;
        }
        let now = std::time::Instant::now();
        self.frame_start = Some(now);
        self.phase_stack.clear();
        self.current_frame = Some(FrameProfile {
            frame_number,
            total_ms: 0.0,
            phases: Vec::new(),
        });
    }

    pub fn begin_phase(&mut self, name: &str) {
        if !self.recording || self.current_frame.is_none() {
            return;
        }
        self.phase_stack.push((name.to_string(), std::time::Instant::now()));
    }

    pub fn end_phase(&mut self) {
        if !self.recording {
            return;
        }
        if let Some((name, start)) = self.phase_stack.pop() {
            let duration_ms = start.elapsed().as_secs_f32() * 1000.0;
            let start_ms = self
                .frame_start
                .map(|fs| (start - fs).as_secs_f32() * 1000.0)
                .unwrap_or(0.0);
            let phase = PhaseProfile {
                name,
                start_ms,
                duration_ms,
                children: Vec::new(),
            };
            if let Some(ref mut frame) = self.current_frame {
                frame.phases.push(phase);
            }
        }
    }

    pub fn end_frame(&mut self) {
        if !self.recording {
            return;
        }
        if let Some(mut frame) = self.current_frame.take() {
            if let Some(start) = self.frame_start.take() {
                frame.total_ms = start.elapsed().as_secs_f32() * 1000.0;
            }
            self.history.push_back(frame);
            while self.history.len() > self.max_history {
                self.history.pop_front();
            }
        }
    }

    pub fn avg_frame_ms(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|f| f.total_ms).sum();
        sum / self.history.len() as f32
    }

    pub fn worst_frame_ms(&self) -> f32 {
        self.history
            .iter()
            .map(|f| f.total_ms)
            .fold(0.0_f32, f32::max)
    }

    pub fn phase_average(&self, name: &str) -> f32 {
        let mut total = 0.0_f32;
        let mut count = 0u32;
        for frame in &self.history {
            for phase in &frame.phases {
                if phase.name == name {
                    total += phase.duration_ms;
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }

    pub fn latest(&self) -> Option<&FrameProfile> {
        self.history.back()
    }

    pub fn fps(&self) -> f32 {
        let avg = self.avg_frame_ms();
        if avg <= 0.0 {
            0.0
        } else {
            1000.0 / avg
        }
    }

    /// Get a summary string for display.
    pub fn summary(&self) -> String {
        let avg = self.avg_frame_ms();
        let worst = self.worst_frame_ms();
        let fps = self.fps();
        format!(
            "FPS: {:.0} | Avg: {:.1}ms | Worst: {:.1}ms | Frames: {}",
            fps,
            avg,
            worst,
            self.history.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_end_frame_records() {
        let mut dbg = FrameDebugger::new(100);
        dbg.begin_frame(1);
        dbg.end_frame();
        assert_eq!(dbg.history.len(), 1);
        assert_eq!(dbg.history[0].frame_number, 1);
    }

    #[test]
    fn phase_timing_works() {
        let mut dbg = FrameDebugger::new(100);
        dbg.begin_frame(1);
        dbg.begin_phase("render");
        // Spin briefly so duration > 0
        std::thread::sleep(std::time::Duration::from_millis(1));
        dbg.end_phase();
        dbg.end_frame();
        assert_eq!(dbg.history[0].phases.len(), 1);
        assert_eq!(dbg.history[0].phases[0].name, "render");
        assert!(dbg.history[0].phases[0].duration_ms > 0.0);
    }

    #[test]
    fn history_caps_at_max() {
        let mut dbg = FrameDebugger::new(3);
        for i in 0..10 {
            dbg.begin_frame(i);
            dbg.end_frame();
        }
        assert_eq!(dbg.history.len(), 3);
        // Oldest should have been evicted; latest frame_number is 9
        assert_eq!(dbg.history.back().unwrap().frame_number, 9);
    }

    #[test]
    fn avg_worst_computed() {
        let mut dbg = FrameDebugger::new(100);
        // Manually push frames with known total_ms
        for ms in [10.0, 20.0, 30.0] {
            dbg.history.push_back(FrameProfile {
                frame_number: 0,
                total_ms: ms,
                phases: Vec::new(),
            });
        }
        assert!((dbg.avg_frame_ms() - 20.0).abs() < 0.001);
        assert!((dbg.worst_frame_ms() - 30.0).abs() < 0.001);
    }

    #[test]
    fn summary_string_formatted() {
        let mut dbg = FrameDebugger::new(100);
        dbg.history.push_back(FrameProfile {
            frame_number: 0,
            total_ms: 16.6,
            phases: Vec::new(),
        });
        let s = dbg.summary();
        assert!(s.contains("FPS:"));
        assert!(s.contains("Avg:"));
        assert!(s.contains("Worst:"));
        assert!(s.contains("Frames: 1"));
    }
}
