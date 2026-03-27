/// Manages smooth LOD transitions for an instance.
#[derive(Debug, Clone)]
pub struct LodTransition {
    pub instance_id: u32,
    pub from_lod: u32,
    pub to_lod: u32,
    pub progress: f32,    // 0.0 = fully from_lod, 1.0 = fully to_lod
    pub duration: f32,    // transition duration in seconds
    pub elapsed: f32,
}

impl LodTransition {
    pub fn new(instance_id: u32, from_lod: u32, to_lod: u32, duration: f32) -> Self {
        Self { instance_id, from_lod, to_lod, progress: 0.0, duration, elapsed: 0.0 }
    }

    pub fn tick(&mut self, dt: f32) {
        self.elapsed += dt;
        self.progress = (self.elapsed / self.duration).clamp(0.0, 1.0);
    }

    pub fn is_complete(&self) -> bool { self.progress >= 1.0 }

    /// Get opacity for the outgoing LOD (fades out).
    pub fn from_opacity(&self) -> f32 { 1.0 - self.progress }

    /// Get opacity for the incoming LOD (fades in).
    pub fn to_opacity(&self) -> f32 { self.progress }
}

/// Manages all active LOD transitions.
pub struct LodCrossfadeManager {
    pub transitions: Vec<LodTransition>,
    pub transition_duration: f32, // default duration in seconds
}

impl LodCrossfadeManager {
    pub fn new(transition_duration: f32) -> Self {
        Self { transitions: Vec::new(), transition_duration }
    }

    /// Request a LOD change for an instance. Initiates a crossfade.
    pub fn request_lod_change(&mut self, instance_id: u32, from_lod: u32, to_lod: u32) {
        // Remove any existing transition for this instance
        self.transitions.retain(|t| t.instance_id != instance_id);

        if from_lod != to_lod {
            self.transitions.push(LodTransition::new(
                instance_id, from_lod, to_lod, self.transition_duration,
            ));
        }
    }

    /// Advance all transitions.
    pub fn tick(&mut self, dt: f32) {
        for transition in &mut self.transitions {
            transition.tick(dt);
        }
        self.transitions.retain(|t| !t.is_complete());
    }

    /// Check if an instance is currently transitioning.
    pub fn get_transition(&self, instance_id: u32) -> Option<&LodTransition> {
        self.transitions.iter().find(|t| t.instance_id == instance_id)
    }

    pub fn active_count(&self) -> usize { self.transitions.len() }
}
