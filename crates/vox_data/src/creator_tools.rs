use serde::{Deserialize, Serialize};

/// A brush stroke for terrain or material painting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrushStroke {
    /// World-space position of the brush centre.
    pub position: [f32; 3],
    /// Brush radius in world units.
    pub radius: f32,
    /// Material identifier to paint.
    pub material: String,
    /// Pressure from 0.0 (no effect) to 1.0 (full effect).
    pub pressure: f32,
    /// Optional timestamp for stroke ordering.
    pub timestamp: f64,
}

impl BrushStroke {
    pub fn new(position: [f32; 3], radius: f32, material: &str, pressure: f32) -> Self {
        Self {
            position,
            radius,
            material: material.to_string(),
            pressure: pressure.clamp(0.0, 1.0),
            timestamp: 0.0,
        }
    }

    /// Returns the area of influence (circle area).
    pub fn area_of_influence(&self) -> f32 {
        std::f32::consts::PI * self.radius * self.radius
    }

    /// Check if a world-space point is within this brush stroke.
    pub fn contains_point(&self, point: [f32; 3]) -> bool {
        let dx = point[0] - self.position[0];
        let dy = point[1] - self.position[1];
        let dz = point[2] - self.position[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        dist_sq <= self.radius * self.radius
    }
}

/// Terrain sculpting operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TerrainSculptOp {
    /// Raise terrain by the given amount.
    Raise(f32),
    /// Lower terrain by the given amount.
    Lower(f32),
    /// Flatten terrain to the given height.
    Flatten(f32),
    /// Smooth terrain with the given strength (0.0-1.0).
    Smooth(f32),
    /// Paint a material onto terrain.
    Paint(String),
}

impl TerrainSculptOp {
    /// Apply the sculpt operation to a height value, returning the new height.
    pub fn apply_to_height(&self, current_height: f32) -> f32 {
        match self {
            Self::Raise(amount) => current_height + amount,
            Self::Lower(amount) => current_height - amount,
            Self::Flatten(target) => *target,
            Self::Smooth(strength) => {
                // Smooth moves height toward the average (assumed 0.0 for simplicity).
                current_height * (1.0 - strength)
            }
            Self::Paint(_) => current_height, // Paint does not change height.
        }
    }
}

/// A sculpt command combining a brush stroke with an operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SculptCommand {
    pub brush: BrushStroke,
    pub operation: TerrainSculptOp,
}

/// A camera keyframe in a cutscene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraKeyframe {
    /// Time in seconds from the start of the cutscene.
    pub time: f32,
    /// Camera position.
    pub position: [f32; 3],
    /// Camera look-at target.
    pub look_at: [f32; 3],
    /// Field of view in degrees.
    pub fov: f32,
}

/// An audio cue in a cutscene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioCue {
    /// Time in seconds from the start of the cutscene.
    pub time: f32,
    /// Audio asset identifier.
    pub asset_id: String,
    /// Volume from 0.0 to 1.0.
    pub volume: f32,
}

/// An entity animation reference in a cutscene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityAnimation {
    /// Time in seconds from the start of the cutscene.
    pub time: f32,
    /// Entity identifier.
    pub entity_id: String,
    /// Animation clip name.
    pub animation: String,
    /// Duration of the animation in seconds.
    pub duration: f32,
}

/// A cutscene timeline with camera keyframes, audio cues, and entity animations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutsceneTimeline {
    pub name: String,
    pub camera_keyframes: Vec<CameraKeyframe>,
    pub audio_cues: Vec<AudioCue>,
    pub entity_animations: Vec<EntityAnimation>,
}

impl CutsceneTimeline {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            camera_keyframes: Vec::new(),
            audio_cues: Vec::new(),
            entity_animations: Vec::new(),
        }
    }

    pub fn add_camera_keyframe(&mut self, time: f32, position: [f32; 3], look_at: [f32; 3], fov: f32) {
        self.camera_keyframes.push(CameraKeyframe {
            time,
            position,
            look_at,
            fov,
        });
    }

    pub fn add_audio_cue(&mut self, time: f32, asset_id: &str, volume: f32) {
        self.audio_cues.push(AudioCue {
            time,
            asset_id: asset_id.to_string(),
            volume: volume.clamp(0.0, 1.0),
        });
    }

    pub fn add_entity_animation(
        &mut self,
        time: f32,
        entity_id: &str,
        animation: &str,
        duration: f32,
    ) {
        self.entity_animations.push(EntityAnimation {
            time,
            entity_id: entity_id.to_string(),
            animation: animation.to_string(),
            duration,
        });
    }

    /// Total duration of the cutscene (maximum end time across all elements).
    pub fn total_duration(&self) -> f32 {
        let cam_max = self
            .camera_keyframes
            .iter()
            .map(|k| k.time)
            .fold(0.0f32, f32::max);
        let audio_max = self
            .audio_cues
            .iter()
            .map(|a| a.time)
            .fold(0.0f32, f32::max);
        let anim_max = self
            .entity_animations
            .iter()
            .map(|a| a.time + a.duration)
            .fold(0.0f32, f32::max);

        cam_max.max(audio_max).max(anim_max)
    }

    /// Get the interpolated camera position at a given time.
    /// Returns the position of the nearest keyframe before or at the given time.
    pub fn camera_at(&self, time: f32) -> Option<&CameraKeyframe> {
        self.camera_keyframes
            .iter()
            .filter(|k| k.time <= time)
            .last()
    }
}

/// Spectral material properties for the paint brush.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralMaterial {
    /// Material name.
    pub name: String,
    /// Spectral reflectance coefficients (e.g., per wavelength band).
    pub reflectance: Vec<f32>,
    /// Roughness from 0.0 (mirror) to 1.0 (fully diffuse).
    pub roughness: f32,
    /// Metallic factor from 0.0 to 1.0.
    pub metallic: f32,
}

/// A material paint brush that applies spectral properties to surfaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialPaintBrush {
    /// The spectral material to apply.
    pub material: SpectralMaterial,
    /// Brush radius.
    pub radius: f32,
    /// Brush pressure/opacity (0.0 to 1.0).
    pub pressure: f32,
    /// Falloff type: how the effect decreases from center to edge.
    pub falloff: BrushFalloff,
}

/// How the brush effect falls off from center to edge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BrushFalloff {
    /// Constant pressure across the brush.
    Constant,
    /// Linear falloff from center to edge.
    Linear,
    /// Smooth (cosine) falloff.
    Smooth,
}

impl MaterialPaintBrush {
    pub fn new(material: SpectralMaterial, radius: f32, pressure: f32) -> Self {
        Self {
            material,
            radius,
            pressure: pressure.clamp(0.0, 1.0),
            falloff: BrushFalloff::Smooth,
        }
    }

    /// Calculate the effective pressure at a given distance from the brush center.
    pub fn pressure_at_distance(&self, distance: f32) -> f32 {
        if distance >= self.radius {
            return 0.0;
        }
        let t = distance / self.radius;
        let falloff = match self.falloff {
            BrushFalloff::Constant => 1.0,
            BrushFalloff::Linear => 1.0 - t,
            BrushFalloff::Smooth => ((1.0 - t) * std::f32::consts::PI * 0.5).sin(),
        };
        self.pressure * falloff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brush_stroke_basic() {
        let brush = BrushStroke::new([10.0, 0.0, 10.0], 5.0, "grass", 0.8);
        assert_eq!(brush.material, "grass");
        assert!((brush.pressure - 0.8).abs() < f32::EPSILON);
        assert!(brush.contains_point([12.0, 0.0, 10.0]));
        assert!(!brush.contains_point([100.0, 0.0, 100.0]));
    }

    #[test]
    fn test_sculpt_ops() {
        assert!((TerrainSculptOp::Raise(5.0).apply_to_height(10.0) - 15.0).abs() < f32::EPSILON);
        assert!((TerrainSculptOp::Lower(3.0).apply_to_height(10.0) - 7.0).abs() < f32::EPSILON);
        assert!(
            (TerrainSculptOp::Flatten(20.0).apply_to_height(10.0) - 20.0).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_cutscene_timeline() {
        let mut timeline = CutsceneTimeline::new("intro");
        timeline.add_camera_keyframe(0.0, [0.0; 3], [10.0, 0.0, 0.0], 60.0);
        timeline.add_camera_keyframe(5.0, [5.0, 5.0, 0.0], [10.0, 0.0, 0.0], 45.0);
        timeline.add_audio_cue(1.0, "music_intro", 0.8);
        timeline.add_entity_animation(2.0, "hero", "wave", 3.0);

        assert_eq!(timeline.camera_keyframes.len(), 2);
        assert!((timeline.total_duration() - 5.0).abs() < f32::EPSILON);
    }
}
