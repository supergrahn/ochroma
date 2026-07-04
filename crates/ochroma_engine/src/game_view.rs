//! Renderer-agnostic view intent owned by the engine.
//!
//! A game running on Ochroma can expose view state through engine-owned traits.
//! Ochroma invokes that state from the engine loop and turns it into renderer
//! work. This module intentionally contains no `wgpu`, Spectra handles,
//! swapchain types, or game-specific concepts.

use glam::{Mat4, Vec3, Vec4};

/// Pixel dimensions of a game view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewportSize {
    pub width: u32,
    pub height: u32,
}

impl ViewportSize {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn aspect_ratio(self) -> f32 {
        if self.height == 0 {
            1.0
        } else {
            self.width as f32 / self.height as f32
        }
    }
}

/// Why the engine is producing a view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewPurpose {
    Interactive,
    Screenshot,
    AssetPreview,
    Diagnostic,
}

/// View quality preference. Renderer backend choice remains an engine decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewQuality {
    Performance,
    Balanced,
    Quality,
    Cinematic,
}

/// A camera in engine terms, not renderer terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewCamera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl ViewCamera {
    pub fn new(eye: Vec3, target: Vec3) -> Self {
        Self {
            eye,
            target,
            up: Vec3::Y,
            fov_y: std::f32::consts::FRAC_PI_4,
            near: 0.5,
            far: 10_000.0,
        }
    }

    pub fn view_matrix(self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    pub fn projection_matrix(self, viewport: ViewportSize) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, viewport.aspect_ratio(), self.near, self.far)
    }

    pub fn view_projection(self, viewport: ViewportSize) -> Mat4 {
        self.projection_matrix(viewport) * self.view_matrix()
    }

    pub fn to_render_camera(self, viewport: ViewportSize) -> vox_render::spectral::RenderCamera {
        vox_render::spectral::RenderCamera {
            view: self.view_matrix(),
            proj: self.projection_matrix(viewport),
        }
    }
}

/// Orbit-camera intent around a target point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitView {
    pub center: Vec3,
    pub radius: f32,
    /// Radians around +Y.
    pub yaw: f32,
    /// Radians above the ground plane.
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    /// Additional far-plane padding after `radius * 14`.
    pub far_padding: f32,
}

impl OrbitView {
    pub fn new(center: Vec3, radius: f32, yaw: f32, pitch: f32) -> Self {
        Self {
            center,
            radius,
            yaw,
            pitch,
            fov_y: std::f32::consts::FRAC_PI_4,
            near: 0.5,
            far_padding: 1000.0,
        }
    }

    pub fn camera(self) -> ViewCamera {
        let radius = self.radius.max(0.001);
        let cp = self.pitch.cos();
        // NO SILENT CLAMPING (user law 2026-07-04): the old `.max(0.05)` floor on
        // sin(pitch) silently re-lofted every low/upward camera pose (witnessed:
        // a requested eye_y 430 rendered from 575.5 = center.y + r*0.05, to the
        // decimal). A caller that asks to look up gets to look up.
        let eye = self.center
            + Vec3::new(
                radius * cp * self.yaw.sin(),
                radius * self.pitch.sin(),
                radius * cp * self.yaw.cos(),
            );

        ViewCamera {
            eye,
            target: self.center,
            up: Vec3::Y,
            fov_y: self.fov_y,
            near: self.near,
            far: radius * 14.0 + self.far_padding,
        }
    }
}

/// Context supplied by Ochroma when it asks game state for view intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewContext {
    pub viewport: ViewportSize,
    pub frame_index: u64,
}

impl ViewContext {
    pub const fn new(viewport: ViewportSize, frame_index: u64) -> Self {
        Self {
            viewport,
            frame_index,
        }
    }
}

/// A game-state hook the engine can invoke to obtain view intent.
///
/// The flow is engine-driven: game state implements/provides this, but it does
/// not drive presentation or renderer work.
pub trait GameViewSource {
    fn view_intent(&self, context: ViewContext) -> ViewIntent;
}

/// Complete view intent. It is data the engine can route to whatever renderer
/// backend is active.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewIntent {
    pub viewport: ViewportSize,
    pub camera: ViewCamera,
    pub purpose: ViewPurpose,
    pub quality: ViewQuality,
    pub frame_index: u64,
}

impl ViewIntent {
    pub fn interactive(viewport: ViewportSize, camera: ViewCamera, frame_index: u64) -> Self {
        Self {
            viewport,
            camera,
            purpose: ViewPurpose::Interactive,
            quality: ViewQuality::Quality,
            frame_index,
        }
    }
}

/// Compatibility alias for the initial contract name. Prefer [`ViewIntent`].
#[deprecated(
    since = "0.1.0",
    note = "use ViewIntent; Ochroma invokes game view state instead of the game requesting rendering"
)]
pub type GameViewRequest = ViewIntent;

/// Cast a screen pixel into the ground plane `y = ground_y`.
pub fn screen_to_ground(
    view: Mat4,
    proj: Mat4,
    px: f32,
    py: f32,
    viewport: ViewportSize,
    ground_y: f32,
) -> Option<[f32; 2]> {
    if viewport.width == 0 || viewport.height == 0 {
        return None;
    }
    let inv = (proj * view).inverse();
    let ndc_x = 2.0 * (px + 0.5) / viewport.width as f32 - 1.0;
    let ndc_y = 1.0 - 2.0 * (py + 0.5) / viewport.height as f32;
    let unproject = |z: f32| -> Vec3 {
        let p = inv * Vec4::new(ndc_x, ndc_y, z, 1.0);
        p.truncate() / p.w
    };
    let near = unproject(0.0);
    let far = unproject(1.0);
    let dir = (far - near).normalize_or_zero();
    if dir.y.abs() < 1e-5 {
        return None;
    }
    let t = (ground_y - near.y) / dir.y;
    if t < 0.0 {
        return None;
    }
    let hit = near + dir * t;
    Some([hit.x, hit.z])
}

/// Project a point on the ground plane to screen pixel coordinates.
pub fn world_to_screen(
    view: Mat4,
    proj: Mat4,
    x: f32,
    z: f32,
    viewport: ViewportSize,
    ground_y: f32,
) -> Option<(f32, f32)> {
    if viewport.width == 0 || viewport.height == 0 {
        return None;
    }
    let clip = proj * view * Vec4::new(x, ground_y, z, 1.0);
    if clip.w <= 1e-5 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    let px = (ndc.x + 1.0) * 0.5 * viewport.width as f32 - 0.5;
    let py = (1.0 - ndc.y) * 0.5 * viewport.height as f32 - 0.5;
    Some((px, py))
}

pub fn screen_to_ground_from_camera(
    camera: ViewCamera,
    px: f32,
    py: f32,
    viewport: ViewportSize,
    ground_y: f32,
) -> Option<[f32; 2]> {
    screen_to_ground(
        camera.view_matrix(),
        camera.projection_matrix(viewport),
        px,
        py,
        viewport,
        ground_y,
    )
}

pub fn world_to_screen_from_camera(
    camera: ViewCamera,
    x: f32,
    z: f32,
    viewport: ViewportSize,
    ground_y: f32,
) -> Option<(f32, f32)> {
    world_to_screen(
        camera.view_matrix(),
        camera.projection_matrix(viewport),
        x,
        z,
        viewport,
        ground_y,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_view_builds_camera_without_renderer_state() {
        let camera = OrbitView::new(Vec3::ZERO, 100.0, 0.0, 0.5).camera();
        assert!(camera.eye.y > 0.0);
        assert!(camera.far > camera.near);
        assert_eq!(camera.target, Vec3::ZERO);
    }

    #[test]
    fn viewport_aspect_handles_zero_height() {
        assert_eq!(ViewportSize::new(1920, 0).aspect_ratio(), 1.0);
        assert!((ViewportSize::new(1920, 1080).aspect_ratio() - 16.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn zero_sized_viewport_does_not_project() {
        let camera = OrbitView::new(Vec3::ZERO, 100.0, 0.0, 0.7).camera();
        let viewport = ViewportSize::new(0, 800);
        assert_eq!(
            world_to_screen_from_camera(camera, 0.0, 0.0, viewport, 0.0),
            None
        );
        assert_eq!(
            screen_to_ground_from_camera(camera, 0.0, 0.0, viewport, 0.0),
            None
        );
    }

    #[test]
    fn ground_projection_round_trips_center() {
        let viewport = ViewportSize::new(1280, 800);
        let camera = OrbitView::new(Vec3::ZERO, 100.0, 0.0, 0.7).camera();
        let Some((px, py)) = world_to_screen_from_camera(camera, 0.0, 0.0, viewport, 0.0) else {
            panic!("target should project");
        };
        let Some(hit) = screen_to_ground_from_camera(camera, px, py, viewport, 0.0) else {
            panic!("projected target should hit ground");
        };
        assert!(hit[0].abs() < 0.05, "x roundtrip drift: {}", hit[0]);
        assert!(hit[1].abs() < 0.05, "z roundtrip drift: {}", hit[1]);
    }

    #[test]
    fn view_source_is_engine_invoked_intent() {
        struct StaticView(ViewCamera);

        impl GameViewSource for StaticView {
            fn view_intent(&self, context: ViewContext) -> ViewIntent {
                ViewIntent::interactive(context.viewport, self.0, context.frame_index)
            }
        }

        let viewport = ViewportSize::new(320, 200);
        let camera = ViewCamera::new(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO);
        let source = StaticView(camera);
        let intent = source.view_intent(ViewContext::new(viewport, 7));
        assert_eq!(intent.purpose, ViewPurpose::Interactive);
        assert_eq!(intent.frame_index, 7);
    }
}
