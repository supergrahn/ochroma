/// Multi-viewport support: single, quad-split, horizontal, vertical layouts.

use glam::{Vec3, Mat4};
use crate::spectral::RenderCamera;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportType {
    Perspective,
    Top,
    Front,
    Right,
}

#[derive(Debug, Clone)]
pub struct Viewport {
    pub viewport_type: ViewportType,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub active: bool,
    pub camera: ViewportCamera,
}

#[derive(Debug, Clone)]
pub struct ViewportCamera {
    pub position: Vec3,
    pub target: Vec3,
    pub zoom: f32,
}

impl Viewport {
    pub fn to_render_camera(&self) -> RenderCamera {
        match self.viewport_type {
            ViewportType::Perspective => RenderCamera {
                view: Mat4::look_at_rh(self.camera.position, self.camera.target, Vec3::Y),
                proj: Mat4::perspective_rh(
                    std::f32::consts::FRAC_PI_4,
                    self.width as f32 / self.height.max(1) as f32,
                    0.1,
                    1000.0,
                ),
            },
            ViewportType::Top => {
                let half = self.camera.zoom;
                let aspect = self.width as f32 / self.height.max(1) as f32;
                RenderCamera {
                    view: Mat4::look_at_rh(
                        self.camera.target + Vec3::new(0.0, 100.0, 0.01),
                        self.camera.target,
                        Vec3::Z,
                    ),
                    proj: Mat4::orthographic_rh(
                        -half * aspect, half * aspect, -half, half, 0.1, 500.0,
                    ),
                }
            }
            ViewportType::Front => {
                let half = self.camera.zoom;
                let aspect = self.width as f32 / self.height.max(1) as f32;
                RenderCamera {
                    view: Mat4::look_at_rh(
                        self.camera.target + Vec3::new(0.0, 0.0, 100.0),
                        self.camera.target,
                        Vec3::Y,
                    ),
                    proj: Mat4::orthographic_rh(
                        -half * aspect, half * aspect, -half, half, 0.1, 500.0,
                    ),
                }
            }
            ViewportType::Right => {
                let half = self.camera.zoom;
                let aspect = self.width as f32 / self.height.max(1) as f32;
                RenderCamera {
                    view: Mat4::look_at_rh(
                        self.camera.target + Vec3::new(100.0, 0.0, 0.0),
                        self.camera.target,
                        Vec3::Y,
                    ),
                    proj: Mat4::orthographic_rh(
                        -half * aspect, half * aspect, -half, half, 0.1, 500.0,
                    ),
                }
            }
        }
    }
}

/// Multi-viewport layout manager.
pub struct ViewportLayout {
    pub viewports: Vec<Viewport>,
    pub layout_mode: LayoutMode,
    pub total_width: u32,
    pub total_height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutMode {
    Single,
    QuadSplit,
    HorizontalSplit,
    VerticalSplit,
}

impl ViewportLayout {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            viewports: vec![Viewport {
                viewport_type: ViewportType::Perspective,
                x: 0,
                y: 0,
                width,
                height,
                active: true,
                camera: ViewportCamera {
                    position: Vec3::new(0.0, 10.0, 30.0),
                    target: Vec3::ZERO,
                    zoom: 50.0,
                },
            }],
            layout_mode: LayoutMode::Single,
            total_width: width,
            total_height: height,
        }
    }

    pub fn set_layout(&mut self, mode: LayoutMode) {
        self.layout_mode = mode;
        self.viewports.clear();

        let w = self.total_width;
        let h = self.total_height;
        let cam = ViewportCamera {
            position: Vec3::new(0.0, 10.0, 30.0),
            target: Vec3::ZERO,
            zoom: 50.0,
        };

        match mode {
            LayoutMode::Single => {
                self.viewports.push(Viewport {
                    viewport_type: ViewportType::Perspective,
                    x: 0, y: 0, width: w, height: h,
                    active: true, camera: cam,
                });
            }
            LayoutMode::QuadSplit => {
                let hw = w / 2;
                let hh = h / 2;
                self.viewports.push(Viewport {
                    viewport_type: ViewportType::Perspective,
                    x: 0, y: 0, width: hw, height: hh,
                    active: true, camera: cam.clone(),
                });
                self.viewports.push(Viewport {
                    viewport_type: ViewportType::Top,
                    x: hw, y: 0, width: w - hw, height: hh,
                    active: false, camera: cam.clone(),
                });
                self.viewports.push(Viewport {
                    viewport_type: ViewportType::Front,
                    x: 0, y: hh, width: hw, height: h - hh,
                    active: false, camera: cam.clone(),
                });
                self.viewports.push(Viewport {
                    viewport_type: ViewportType::Right,
                    x: hw, y: hh, width: w - hw, height: h - hh,
                    active: false, camera: cam,
                });
            }
            LayoutMode::HorizontalSplit => {
                let hw = w / 2;
                self.viewports.push(Viewport {
                    viewport_type: ViewportType::Perspective,
                    x: 0, y: 0, width: hw, height: h,
                    active: true, camera: cam.clone(),
                });
                self.viewports.push(Viewport {
                    viewport_type: ViewportType::Top,
                    x: hw, y: 0, width: w - hw, height: h,
                    active: false, camera: cam,
                });
            }
            LayoutMode::VerticalSplit => {
                let hh = h / 2;
                self.viewports.push(Viewport {
                    viewport_type: ViewportType::Perspective,
                    x: 0, y: 0, width: w, height: hh,
                    active: true, camera: cam.clone(),
                });
                self.viewports.push(Viewport {
                    viewport_type: ViewportType::Front,
                    x: 0, y: hh, width: w, height: h - hh,
                    active: false, camera: cam,
                });
            }
        }
    }

    /// Find which viewport contains a screen position.
    pub fn viewport_at(&self, screen_x: f32, screen_y: f32) -> Option<usize> {
        self.viewports.iter().position(|v| {
            screen_x >= v.x as f32
                && screen_x < (v.x + v.width) as f32
                && screen_y >= v.y as f32
                && screen_y < (v.y + v.height) as f32
        })
    }

    pub fn viewport_count(&self) -> usize {
        self.viewports.len()
    }

    pub fn active_viewport(&self) -> Option<&Viewport> {
        self.viewports.iter().find(|v| v.active)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_layout_has_1_viewport() {
        let layout = ViewportLayout::new(1920, 1080);
        assert_eq!(layout.viewport_count(), 1);
        assert_eq!(layout.layout_mode, LayoutMode::Single);
    }

    #[test]
    fn test_quad_split_has_4_viewports() {
        let mut layout = ViewportLayout::new(1920, 1080);
        layout.set_layout(LayoutMode::QuadSplit);
        assert_eq!(layout.viewport_count(), 4);
        assert_eq!(layout.viewports[0].viewport_type, ViewportType::Perspective);
        assert_eq!(layout.viewports[1].viewport_type, ViewportType::Top);
        assert_eq!(layout.viewports[2].viewport_type, ViewportType::Front);
        assert_eq!(layout.viewports[3].viewport_type, ViewportType::Right);
    }

    #[test]
    fn test_viewport_at_finds_correct() {
        let mut layout = ViewportLayout::new(1920, 1080);
        layout.set_layout(LayoutMode::QuadSplit);
        // Top-left quadrant (perspective)
        assert_eq!(layout.viewport_at(100.0, 100.0), Some(0));
        // Top-right quadrant (top view)
        assert_eq!(layout.viewport_at(1500.0, 100.0), Some(1));
        // Bottom-left quadrant (front view)
        assert_eq!(layout.viewport_at(100.0, 800.0), Some(2));
        // Bottom-right quadrant (right view)
        assert_eq!(layout.viewport_at(1500.0, 800.0), Some(3));
    }

    #[test]
    fn test_ortho_cameras_produce_valid_matrices() {
        let mut layout = ViewportLayout::new(1920, 1080);
        layout.set_layout(LayoutMode::QuadSplit);
        for vp in &layout.viewports {
            let cam = vp.to_render_camera();
            // Matrices should not be NaN or zero
            let view_cols = [cam.view.x_axis, cam.view.y_axis, cam.view.z_axis, cam.view.w_axis];
            for col in &view_cols {
                assert!(!col.x.is_nan() && !col.y.is_nan() && !col.z.is_nan() && !col.w.is_nan());
            }
            let proj_cols = [cam.proj.x_axis, cam.proj.y_axis, cam.proj.z_axis, cam.proj.w_axis];
            for col in &proj_cols {
                assert!(!col.x.is_nan() && !col.y.is_nan() && !col.z.is_nan() && !col.w.is_nan());
            }
        }
    }

    #[test]
    fn test_resize_updates_dimensions() {
        let mut layout = ViewportLayout::new(800, 600);
        assert_eq!(layout.total_width, 800);
        assert_eq!(layout.total_height, 600);
        layout.total_width = 1920;
        layout.total_height = 1080;
        layout.set_layout(LayoutMode::QuadSplit);
        // Verify the viewports use new dimensions
        assert_eq!(layout.viewports[0].width, 960);
        assert_eq!(layout.viewports[0].height, 540);
    }
}
