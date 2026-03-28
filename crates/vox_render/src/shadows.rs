use glam::{Mat4, Vec3, Vec4};

/// Configuration for a single cascade in a cascaded shadow map.
#[derive(Debug, Clone)]
pub struct CascadeConfig {
    pub near: f32,
    pub far: f32,
    pub resolution: usize,
}

/// A single cascade's shadow map data.
#[derive(Debug, Clone)]
pub struct CascadeShadowMap {
    pub config: CascadeConfig,
    pub light_view_proj: Mat4,
    pub depth_buffer: Vec<f32>,
    pub resolution: usize,
}

impl CascadeShadowMap {
    pub fn new(config: CascadeConfig) -> Self {
        let resolution = config.resolution;
        Self {
            config,
            light_view_proj: Mat4::IDENTITY,
            depth_buffer: vec![f32::INFINITY; resolution * resolution],
            resolution,
        }
    }

    /// Clear depth buffer to infinity (far plane).
    pub fn clear(&mut self) {
        self.depth_buffer.fill(f32::INFINITY);
    }
}

/// Cascaded shadow mapper for directional (sun) light.
///
/// Uses a CPU depth buffer for software rasteriser compatibility.
/// Each cascade covers a different distance range from the camera.
pub struct ShadowMapper {
    pub cascades: Vec<CascadeShadowMap>,
    pub sun_direction: Vec3,
    /// World-space radius used when building the orthographic projection
    /// for each cascade. Stored so tests can inspect it.
    pub cascade_radii: Vec<f32>,
}

impl ShadowMapper {
    /// Create a shadow mapper with the default 3-cascade configuration:
    /// cascade 0: 0 - 20 m
    /// cascade 1: 20 - 100 m
    /// cascade 2: 100 - 500 m
    pub fn new(resolution: usize) -> Self {
        let configs = vec![
            CascadeConfig { near: 0.0, far: 20.0, resolution },
            CascadeConfig { near: 20.0, far: 100.0, resolution },
            CascadeConfig { near: 100.0, far: 500.0, resolution },
        ];
        let cascades = configs.into_iter().map(CascadeShadowMap::new).collect();
        Self {
            cascades,
            sun_direction: Vec3::new(0.0, -1.0, 0.0).normalize(),
            cascade_radii: Vec::new(),
        }
    }

    /// Update light view-projection matrices for each cascade given the camera
    /// state and sun direction.
    ///
    /// `camera_pos`  - world position of the camera
    /// `camera_fwd`  - normalised forward vector of the camera
    /// `sun_dir`     - normalised direction the sun shines *towards* (points toward ground)
    pub fn update(&mut self, camera_pos: Vec3, camera_fwd: Vec3, sun_dir: Vec3) {
        self.sun_direction = sun_dir.normalize();
        self.cascade_radii.clear();

        for cascade in &mut self.cascades {
            // Compute the centre of the cascade frustum slice along the camera's view direction.
            let mid = (cascade.config.near + cascade.config.far) * 0.5;
            let centre = camera_pos + camera_fwd * mid;

            // Radius encloses the frustum slice (conservative sphere approximation).
            let radius = (cascade.config.far - cascade.config.near) * 0.5
                + cascade.config.far * 0.4; // padding for off-axis coverage
            self.cascade_radii.push(radius);

            // Light view: look from above the centre along the sun direction.
            let light_pos = centre - self.sun_direction * radius * 2.0;
            // Choose an up vector that isn't parallel to the light direction.
            let up = if self.sun_direction.cross(Vec3::Y).length() < 1e-3 {
                Vec3::Z
            } else {
                Vec3::Y
            };
            let light_view = Mat4::look_at_rh(light_pos, centre, up);

            // Orthographic projection sized to enclose the cascade sphere.
            let light_proj = Mat4::orthographic_rh(
                -radius, radius, -radius, radius, 0.01, radius * 4.0,
            );

            cascade.light_view_proj = light_proj * light_view;
            cascade.clear();
        }
    }

    /// Render the shadow map (CPU depth buffer) from the light's perspective.
    ///
    /// `splat_positions` - world-space positions of splats / occluders
    /// `splat_radii`     - radius of each splat (used to rasterise a small footprint)
    pub fn render_shadow_map(&mut self, splat_positions: &[Vec3], splat_radii: &[f32]) {
        assert_eq!(splat_positions.len(), splat_radii.len());

        for cascade in &mut self.cascades {
            cascade.clear();
            let res = cascade.resolution;
            let vp = cascade.light_view_proj;

            for (pos, &radius) in splat_positions.iter().zip(splat_radii.iter()) {
                // Project centre to light clip space.
                let clip = vp * Vec4::new(pos.x, pos.y, pos.z, 1.0);
                if clip.w <= 0.0 {
                    continue;
                }
                let ndc_x = clip.x / clip.w;
                let ndc_y = clip.y / clip.w;
                let ndc_z = clip.z / clip.w;

                // Discard if outside NDC cube.
                if !(-1.0..=1.0).contains(&ndc_x) || !(-1.0..=1.0).contains(&ndc_y) {
                    continue;
                }
                if !(0.0..=1.0).contains(&ndc_z) {
                    continue;
                }

                // Map to pixel coordinates.
                let px = ((ndc_x * 0.5 + 0.5) * res as f32) as i32;
                let py = ((ndc_y * 0.5 + 0.5) * res as f32) as i32;

                // Rasterise a small footprint proportional to the splat radius.
                // Project radius into screen space (approximate).
                let screen_radius = {
                    let edge = vp * Vec4::new(pos.x + radius, pos.y, pos.z, 1.0);
                    if edge.w > 0.0 {
                        let edge_ndc_x = edge.x / edge.w;
                        ((edge_ndc_x - ndc_x).abs() * 0.5 * res as f32).max(1.0) as i32
                    } else {
                        1
                    }
                };

                let half = screen_radius;
                for dy in -half..=half {
                    for dx in -half..=half {
                        let sx = px + dx;
                        let sy = py + dy;
                        if sx >= 0 && sx < res as i32 && sy >= 0 && sy < res as i32 {
                            let idx = sy as usize * res + sx as usize;
                            if ndc_z < cascade.depth_buffer[idx] {
                                cascade.depth_buffer[idx] = ndc_z;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Test whether a world-space point is in shadow.
    ///
    /// Returns `true` if the point is shadowed (something is closer to the light).
    /// Uses a small bias to avoid shadow acne.
    pub fn is_in_shadow(&self, world_pos: Vec3, bias: f32) -> bool {
        // Find the appropriate cascade for this point.
        // We test all cascades and use the first one that contains the point.
        for cascade in &self.cascades {
            let vp = cascade.light_view_proj;
            let clip = vp * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
            if clip.w <= 0.0 {
                continue;
            }
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            let ndc_z = clip.z / clip.w;

            // Check if point falls within this cascade's projection.
            if !(-1.0..=1.0).contains(&ndc_x) || !(-1.0..=1.0).contains(&ndc_y) {
                continue;
            }
            if !(0.0..=1.0).contains(&ndc_z) {
                continue;
            }

            let res = cascade.resolution;
            let px = ((ndc_x * 0.5 + 0.5) * res as f32) as usize;
            let py = ((ndc_y * 0.5 + 0.5) * res as f32) as usize;

            let px = px.min(res - 1);
            let py = py.min(res - 1);

            let idx = py * res + px;
            let stored_depth = cascade.depth_buffer[idx];

            // If the stored depth is closer (smaller) than this point's depth minus bias,
            // the point is in shadow.
            if stored_depth < ndc_z - bias {
                return true;
            }
            // Point is lit in this cascade.
            return false;
        }

        // Point not in any cascade -- assume lit.
        false
    }

    /// Convenience: number of cascades.
    pub fn cascade_count(&self) -> usize {
        self.cascades.len()
    }

    /// Get the distance range for a cascade by index.
    pub fn cascade_range(&self, index: usize) -> (f32, f32) {
        let c = &self.cascades[index].config;
        (c.near, c.far)
    }
}
