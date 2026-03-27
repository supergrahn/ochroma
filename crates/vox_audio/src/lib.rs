use glam::Vec3;

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
    pub sources: Vec<AudioSource>,
    next_id: u32,
    pub listener_position: Vec3,
}

impl AudioEngine {
    pub fn new(max_sources: usize) -> Self {
        Self {
            max_sources,
            sources: Vec::new(),
            next_id: 1,
            listener_position: Vec3::ZERO,
        }
    }

    pub fn set_listener(&mut self, pos: Vec3) {
        self.listener_position = pos;
    }

    pub fn play(&mut self, mut source: AudioSource) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        source.id = id;
        self.sources.push(source);
        id
    }

    pub fn stop(&mut self, id: u32) {
        self.sources.retain(|s| s.id != id);
    }

    pub fn active_count(&self) -> usize {
        self.sources.len()
    }

    /// Calculate effective volume using inverse-distance attenuation.
    pub fn effective_volume(&self, source: &AudioSource) -> f32 {
        Self::effective_volume_at(source, self.listener_position)
    }

    /// Stateless helper that avoids borrow conflicts during tick.
    fn effective_volume_at(source: &AudioSource, listener: Vec3) -> f32 {
        let dist = source.position.distance(listener);
        let attenuation = 1.0 / (1.0 + dist * 0.1);
        source.volume * attenuation
    }

    /// Get all active sources sorted by effective volume (loudest first).
    pub fn active_sources_by_priority(&self) -> Vec<&AudioSource> {
        let mut sources: Vec<&AudioSource> = self.sources.iter().collect();
        sources.sort_by(|a, b| {
            self.effective_volume(b)
                .partial_cmp(&self.effective_volume(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sources
    }

    /// Tick: evict lowest-priority sources if over budget.
    pub fn tick(&mut self, _dt: f32) {
        let listener = self.listener_position;
        while self.sources.len() > self.max_sources {
            if let Some(idx) = self.sources
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    Self::effective_volume_at(a, listener)
                        .partial_cmp(&Self::effective_volume_at(b, listener))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.sources.remove(idx);
            } else {
                break;
            }
        }
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new(64)
    }
}
