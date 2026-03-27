use glam::Vec3;

/// Frequency bands: [low, mid, high] Hz ranges.
pub type FrequencyBands = [f32; 3];

/// Acoustic material with absorption coefficients per frequency band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticMaterial {
    /// Absorption per band [low, mid, high], 0.0 = perfect reflection, 1.0 = full absorption.
    pub absorption: FrequencyBands,
    /// Human-readable name for debugging.
    pub name: &'static str,
}

impl AcousticMaterial {
    /// Brick: absorbs high frequencies, reflects low.
    pub const BRICK: Self = Self {
        absorption: [0.05, 0.15, 0.60],
        name: "brick",
    };

    /// Glass: transmits high frequencies, reflects low.
    pub const GLASS: Self = Self {
        absorption: [0.30, 0.10, 0.05],
        name: "glass",
    };

    /// Concrete: absorbs mid frequencies.
    pub const CONCRETE: Self = Self {
        absorption: [0.10, 0.50, 0.20],
        name: "concrete",
    };

    /// Wood: absorbs low frequencies.
    pub const WOOD: Self = Self {
        absorption: [0.55, 0.15, 0.10],
        name: "wood",
    };
}

/// A ray used for acoustic simulation.
#[derive(Debug, Clone)]
pub struct AcousticRay {
    pub origin: Vec3,
    pub direction: Vec3,
    pub energy: FrequencyBands,
    pub bounces_remaining: u32,
    /// Total distance traveled so far (meters).
    pub distance_traveled: f32,
}

impl AcousticRay {
    pub fn new(origin: Vec3, direction: Vec3, max_bounces: u32) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
            energy: [1.0, 1.0, 1.0],
            bounces_remaining: max_bounces,
            distance_traveled: 0.0,
        }
    }

    /// Apply material absorption to the ray's energy.
    pub fn absorb(&mut self, material: &AcousticMaterial) {
        for i in 0..3 {
            self.energy[i] *= 1.0 - material.absorption[i];
        }
    }
}

/// A surface in the acoustic scene.
#[derive(Debug, Clone)]
pub struct AcousticSurface {
    pub position: Vec3,
    /// Normal of the surface (facing outward).
    pub normal: Vec3,
    pub material: AcousticMaterial,
    /// Radius of the surface for intersection tests.
    pub radius: f32,
}

/// The acoustic scene containing all reflective surfaces.
#[derive(Debug, Clone)]
pub struct AcousticScene {
    pub surfaces: Vec<AcousticSurface>,
}

/// A single reflection path from source to listener.
#[derive(Debug, Clone)]
pub struct ReflectionPath {
    /// Total path length in meters.
    pub distance: f32,
    /// Delay in seconds (distance / speed_of_sound).
    pub delay: f32,
    /// Frequency response after material absorption.
    pub frequency_response: FrequencyBands,
    /// Number of bounces in this path.
    pub bounce_count: u32,
}

/// Result of a full acoustic trace.
#[derive(Debug, Clone)]
pub struct AcousticTraceResult {
    /// Direct path attenuation per frequency band (inverse square law).
    pub direct_attenuation: FrequencyBands,
    /// Early reflections (first-order and second-order bounces).
    pub early_reflections: Vec<ReflectionPath>,
    /// Estimated RT60 reverberation time in seconds.
    pub rt60: f32,
}

/// Speed of sound in air at ~20C (m/s).
pub const SPEED_OF_SOUND: f32 = 343.0;

/// Compute inverse-square-law attenuation for a given distance.
fn inverse_square_attenuation(distance: f32) -> f32 {
    if distance < 0.01 {
        return 1.0;
    }
    1.0 / (distance * distance)
}

/// Intersect a ray with a planar disc surface. Returns the hit distance or None.
fn intersect_surface(ray_origin: Vec3, ray_dir: Vec3, surface: &AcousticSurface) -> Option<f32> {
    let denom = ray_dir.dot(surface.normal);
    if denom.abs() < 1e-6 {
        return None; // parallel
    }
    let t = (surface.position - ray_origin).dot(surface.normal) / denom;
    if t < 0.01 {
        return None; // behind ray
    }
    let hit = ray_origin + ray_dir * t;
    if hit.distance(surface.position) <= surface.radius {
        Some(t)
    } else {
        None
    }
}

/// Reflect a direction vector around a normal.
fn reflect(direction: Vec3, normal: Vec3) -> Vec3 {
    direction - 2.0 * direction.dot(normal) * normal
}

/// Trace sound from source to listener through a scene, computing direct path,
/// early reflections, and late reverb estimate.
pub fn trace_sound(
    source_pos: Vec3,
    listener_pos: Vec3,
    scene: &AcousticScene,
    max_bounces: u32,
) -> AcousticTraceResult {
    let direct_dist = source_pos.distance(listener_pos);
    let direct_atten = inverse_square_attenuation(direct_dist);
    let direct_attenuation = [direct_atten, direct_atten, direct_atten];

    let mut early_reflections = Vec::new();

    // Trace first-order reflections off each surface.
    for surface in &scene.surfaces {
        // Mirror the source across the surface plane.
        let to_surface = surface.position - source_pos;
        let dist_to_plane = to_surface.dot(surface.normal);
        if dist_to_plane < 0.0 {
            continue; // source is behind surface
        }
        let mirror_source = source_pos + 2.0 * dist_to_plane * surface.normal;

        // Check if reflection point is on the surface.
        let mirror_to_listener = listener_pos - mirror_source;
        if let Some(t) = intersect_surface(mirror_source, mirror_to_listener.normalize(), surface) {
            let reflection_point = mirror_source + mirror_to_listener.normalize() * t;
            let total_dist = source_pos.distance(reflection_point)
                + reflection_point.distance(listener_pos);
            let atten = inverse_square_attenuation(total_dist);

            let mut freq_response = [atten, atten, atten];
            for i in 0..3 {
                freq_response[i] *= 1.0 - surface.material.absorption[i];
            }

            early_reflections.push(ReflectionPath {
                distance: total_dist,
                delay: total_dist / SPEED_OF_SOUND,
                frequency_response: freq_response,
                bounce_count: 1,
            });
        }
    }

    // Trace multi-bounce reflections using stochastic ray casting.
    if max_bounces > 1 {
        let num_rays = 16;
        for i in 0..num_rays {
            let angle = (i as f32 / num_rays as f32) * std::f32::consts::TAU;
            let dir = Vec3::new(angle.cos(), 0.0, angle.sin()).normalize();
            let mut ray = AcousticRay::new(source_pos, dir, max_bounces);

            let mut current_origin = source_pos;
            let mut current_dir = dir;
            let mut bounces_done = 0u32;

            while ray.bounces_remaining > 0 {
                // Find closest surface hit.
                let mut closest: Option<(f32, usize)> = None;
                for (idx, surface) in scene.surfaces.iter().enumerate() {
                    if let Some(t) = intersect_surface(current_origin, current_dir, surface) {
                        if closest.is_none() || t < closest.unwrap().0 {
                            closest = Some((t, idx));
                        }
                    }
                }

                if let Some((t, idx)) = closest {
                    let hit_point = current_origin + current_dir * t;
                    ray.distance_traveled += t;
                    ray.absorb(&scene.surfaces[idx].material);
                    ray.bounces_remaining -= 1;
                    bounces_done += 1;

                    current_dir = reflect(current_dir, scene.surfaces[idx].normal);
                    current_origin = hit_point + current_dir * 0.001;

                    // Check if reflected ray can reach listener.
                    let to_listener = listener_pos - hit_point;
                    let total_dist = ray.distance_traveled + to_listener.length();
                    let atten = inverse_square_attenuation(total_dist);

                    if bounces_done >= 2 {
                        let mut freq_response = ray.energy;
                        for i in 0..3 {
                            freq_response[i] *= atten;
                        }

                        early_reflections.push(ReflectionPath {
                            distance: total_dist,
                            delay: total_dist / SPEED_OF_SOUND,
                            frequency_response: freq_response,
                            bounce_count: bounces_done,
                        });
                    }
                } else {
                    break;
                }
            }
        }
    }

    // Estimate RT60 from average absorption.
    let rt60 = estimate_rt60(scene);

    AcousticTraceResult {
        direct_attenuation,
        early_reflections,
        rt60,
    }
}

/// Estimate RT60 (time for sound to decay by 60 dB) using Sabine's formula.
/// RT60 = 0.161 * V / A, where V is estimated volume and A is total absorption area.
fn estimate_rt60(scene: &AcousticScene) -> f32 {
    if scene.surfaces.is_empty() {
        // Open space: very long reverb time.
        return 10.0;
    }

    let total_surface_area: f32 = scene
        .surfaces
        .iter()
        .map(|s| std::f32::consts::PI * s.radius * s.radius)
        .sum();

    let avg_absorption: f32 = if total_surface_area > 0.0 {
        scene
            .surfaces
            .iter()
            .map(|s| {
                let area = std::f32::consts::PI * s.radius * s.radius;
                let avg_abs =
                    (s.material.absorption[0] + s.material.absorption[1] + s.material.absorption[2])
                        / 3.0;
                area * avg_abs
            })
            .sum::<f32>()
            / total_surface_area
    } else {
        0.01
    };

    let total_absorption = total_surface_area * avg_absorption;

    if total_absorption < 0.001 {
        return 10.0;
    }

    // Estimate volume from bounding box of surfaces.
    let volume = estimate_volume(scene).max(1.0);

    // Sabine's equation.
    let rt60 = 0.161 * volume / total_absorption;
    rt60.clamp(0.1, 20.0)
}

/// Estimate volume from the bounding box of scene surfaces.
fn estimate_volume(scene: &AcousticScene) -> f32 {
    if scene.surfaces.len() < 2 {
        return 1000.0; // default large volume for open spaces
    }
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for s in &scene.surfaces {
        min = min.min(s.position - Vec3::splat(s.radius));
        max = max.max(s.position + Vec3::splat(s.radius));
    }
    let extent = max - min;
    (extent.x * extent.y * extent.z).abs()
}

/// Check if the line of sight between source and listener is blocked by obstacles.
/// Returns per-frequency-band attenuation (1.0 = fully blocked, 0.0 = clear).
pub fn compute_obstruction(
    source: Vec3,
    listener: Vec3,
    obstacles: &[AcousticSurface],
) -> FrequencyBands {
    let dir = (listener - source).normalize();
    let total_dist = source.distance(listener);
    let mut attenuation = [0.0f32; 3];

    for surface in obstacles {
        if let Some(t) = intersect_surface(source, dir, surface) {
            if t < total_dist {
                // Each blocking surface adds its absorption as attenuation.
                // Low frequencies diffract around obstacles better.
                attenuation[0] += surface.material.absorption[0] * 0.5; // low freqs diffract
                attenuation[1] += surface.material.absorption[1] * 0.8;
                attenuation[2] += 0.9; // high freqs blocked heavily
            }
        }
    }

    // Clamp to [0, 1].
    for a in &mut attenuation {
        *a = a.clamp(0.0, 1.0);
    }

    attenuation
}

/// Compute the Doppler frequency shift multiplier.
///
/// Returns a multiplier > 1.0 if source is approaching, < 1.0 if receding.
/// `source_velocity` and `listener_velocity` are velocities along the
/// source-to-listener axis (positive = toward listener for source).
pub fn doppler_shift(
    source_velocity: Vec3,
    listener_velocity: Vec3,
    source_pos: Vec3,
    listener_pos: Vec3,
    speed_of_sound: f32,
) -> f32 {
    let direction = (listener_pos - source_pos).normalize();

    // Project velocities onto the source-listener axis.
    let v_source = source_velocity.dot(direction);
    let v_listener = listener_velocity.dot(direction);

    // Doppler formula: f' = f * (c + v_listener) / (c + v_source)
    // Sign convention: positive velocity = moving toward listener.
    let numerator = speed_of_sound + v_listener;
    let denominator = speed_of_sound - v_source;

    if denominator.abs() < 0.01 {
        return 1.0; // avoid division by near-zero
    }

    (numerator / denominator).clamp(0.1, 10.0)
}

/// Procedural sound types for environmental audio.
#[derive(Debug, Clone)]
pub enum ProceduralSound {
    Traffic {
        vehicle_count: u32,
        avg_speed: f32,
    },
    Rain {
        intensity: f32, // 0.0 to 1.0
    },
    Wind {
        speed: f32, // m/s
    },
    Construction {
        progress: f32, // 0.0 to 1.0
    },
}

impl ProceduralSound {
    /// Generate frequency spectrum as [low, mid, high] energy values.
    pub fn generate_frequency_spectrum(&self) -> FrequencyBands {
        match self {
            ProceduralSound::Traffic {
                vehicle_count,
                avg_speed,
            } => {
                let base = (*vehicle_count as f32).sqrt() * 0.1;
                let speed_factor = (*avg_speed / 60.0).clamp(0.0, 2.0);
                [
                    base * 0.8 * speed_factor, // engine rumble (low)
                    base * 0.5 * speed_factor, // tire noise (mid)
                    base * 0.2 * speed_factor, // wind noise (high)
                ]
            }
            ProceduralSound::Rain { intensity } => {
                let i = intensity.clamp(0.0, 1.0);
                [
                    i * 0.2,  // low rumble of heavy rain
                    i * 0.5,  // mid patter
                    i * 0.85, // high frequency drops
                ]
            }
            ProceduralSound::Wind { speed } => {
                let s = (speed / 20.0).clamp(0.0, 2.0);
                [
                    s * 0.7,  // low whoosh
                    s * 0.4,  // mid
                    s * 0.15, // little high frequency content
                ]
            }
            ProceduralSound::Construction { progress } => {
                let p = progress.clamp(0.0, 1.0);
                [
                    0.6 * p, // heavy machinery (low)
                    0.8 * p, // hammering/drilling (mid)
                    0.4 * p, // metal impacts (high)
                ]
            }
        }
    }
}
