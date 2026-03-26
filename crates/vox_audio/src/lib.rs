use glam::Vec3;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AudioSource {
    pub id: u32,
    pub position: Vec3,
    pub volume: f32,
    pub looping: bool,
    pub clip: String,
}

pub struct AudioEngine {
    max_sources: usize,
    sources: HashMap<u32, AudioSource>,
    next_id: u32,
    listener_pos: Vec3,
}

impl AudioEngine {
    pub fn new(max_sources: usize) -> Self {
        Self {
            max_sources,
            sources: HashMap::new(),
            next_id: 1,
            listener_pos: Vec3::ZERO,
        }
    }

    pub fn set_listener(&mut self, pos: Vec3) {
        self.listener_pos = pos;
    }

    pub fn play(&mut self, source: AudioSource) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        if self.sources.len() < self.max_sources {
            let mut s = source;
            s.id = id;
            self.sources.insert(id, s);
        }
        id
    }

    pub fn stop(&mut self, id: u32) {
        self.sources.remove(&id);
    }

    pub fn active_count(&self) -> usize {
        self.sources.len()
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new(64)
    }
}
