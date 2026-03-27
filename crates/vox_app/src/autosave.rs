use std::time::Instant;
use crate::persistence;

/// Auto-save manager that saves periodically.
pub struct AutoSave {
    interval_secs: f32,
    last_save: Instant,
    enabled: bool,
    slot_name: String,
}

impl AutoSave {
    pub fn new(interval_secs: f32) -> Self {
        Self {
            interval_secs,
            last_save: Instant::now(),
            enabled: true,
            slot_name: "autosave".to_string(),
        }
    }

    /// Check if it's time to auto-save. Returns true if save was triggered.
    pub fn tick(&mut self, game_time_hours: f64, citizen_count: u32, funds: f64) -> bool {
        if !self.enabled { return false; }
        if self.last_save.elapsed().as_secs_f32() >= self.interval_secs {
            self.last_save = Instant::now();
            match persistence::save_current(
                "AutoSave City",
                game_time_hours,
                citizen_count,
                funds,
                &self.slot_name,
            ) {
                Ok(_) => {
                    println!("[ochroma] Auto-saved");
                    true
                }
                Err(e) => {
                    eprintln!("[ochroma] Auto-save failed: {}", e);
                    false
                }
            }
        } else {
            false
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) { self.enabled = enabled; }
    pub fn set_interval(&mut self, secs: f32) { self.interval_secs = secs; }
}
