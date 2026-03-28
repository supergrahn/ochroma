//! Editor viewport gizmos — 2D overlay lines drawn AFTER the scene render.
//!
//! Supports translate, rotate, and scale modes with mouse hit-testing
//! and drag interaction.

use glam::{Mat4, Vec3, Vec4};

/// Which manipulation mode the gizmo is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

/// Which axis a gizmo operation targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// Renders gizmo overlays into a pixel buffer and handles hit-testing / drag.
pub struct GizmoRenderer {
    pub mode: GizmoMode,
    pub active_axis: Option<Axis>,
    pub dragging: bool,
    drag_start_screen: Option<(f32, f32)>,
    drag_start_world: Option<Vec3>,
}

// ── colours ──────────────────────────────────────────────────────────────────

const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 128, 255, 255];
const YELLOW: [u8; 4] = [255, 255, 0, 255];

/// Desired on-screen arrow length in pixels.
const ARROW_PIXELS: f32 = 80.0;

/// Hit-test tolerance in pixels.
const HIT_TOLERANCE: f32 = 8.0;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Project a world-space point through `view_proj` into screen-space pixels.
/// Returns `None` if the point is behind the camera.
fn project_to_screen(pos: Vec3, view_proj: Mat4, width: u32, height: u32) -> Option<(f32, f32)> {
    let clip = view_proj * Vec4::new(pos.x, pos.y, pos.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    let sx = (ndc_x * 0.5 + 0.5) * width as f32;
    let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * height as f32; // y-down
    Some((sx, sy))
}

/// Compute a world-space arrow length such that the arrow appears roughly
/// `ARROW_PIXELS` on screen at the given entity depth.
fn world_arrow_length(
    entity_pos: Vec3,
    view_proj: Mat4,
    width: u32,
    height: u32,
) -> f32 {
    let Some(center) = project_to_screen(entity_pos, view_proj, width, height) else {
        return 1.0;
    };
    // Try projecting entity_pos + 1 unit along X and measure pixel distance
    let probe = entity_pos + Vec3::X;
    let Some(probe_screen) = project_to_screen(probe, view_proj, width, height) else {
        return 1.0;
    };
    let px_per_unit = ((probe_screen.0 - center.0).powi(2)
        + (probe_screen.1 - center.1).powi(2))
        .sqrt();
    if px_per_unit < 0.001 {
        return 1.0;
    }
    ARROW_PIXELS / px_per_unit
}

/// Bresenham line drawing into an RGBA pixel buffer.
pub fn draw_line(
    pixels: &mut [[u8; 4]],
    width: u32,
    height: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 4],
) {
    let mut x0 = x0;
    let mut y0 = y0;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x0 >= 0 && y0 >= 0 && (x0 as u32) < width && (y0 as u32) < height {
            let idx = y0 as usize * width as usize + x0 as usize;
            if idx < pixels.len() {
                pixels[idx] = color;
            }
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Draw a small arrowhead (triangle) at `tip` pointing from `base` toward `tip`.
fn draw_arrowhead(
    pixels: &mut [[u8; 4]],
    width: u32,
    height: u32,
    base_x: f32,
    base_y: f32,
    tip_x: f32,
    tip_y: f32,
    color: [u8; 4],
) {
    let dx = tip_x - base_x;
    let dy = tip_y - base_y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0 {
        return;
    }
    let ux = dx / len;
    let uy = dy / len;
    // Perpendicular
    let px = -uy;
    let py = ux;
    let size = 6.0;
    let back_x = tip_x - ux * size * 2.0;
    let back_y = tip_y - uy * size * 2.0;
    let left_x = back_x + px * size;
    let left_y = back_y + py * size;
    let right_x = back_x - px * size;
    let right_y = back_y - py * size;
    draw_line(pixels, width, height, tip_x as i32, tip_y as i32, left_x as i32, left_y as i32, color);
    draw_line(pixels, width, height, tip_x as i32, tip_y as i32, right_x as i32, right_y as i32, color);
    draw_line(pixels, width, height, left_x as i32, left_y as i32, right_x as i32, right_y as i32, color);
}

/// Draw a small square at `center` for scale-mode endpoints.
fn draw_cube_endpoint(
    pixels: &mut [[u8; 4]],
    width: u32,
    height: u32,
    cx: f32,
    cy: f32,
    color: [u8; 4],
) {
    let s = 4;
    let x0 = cx as i32 - s;
    let y0 = cy as i32 - s;
    let x1 = cx as i32 + s;
    let y1 = cy as i32 + s;
    draw_line(pixels, width, height, x0, y0, x1, y0, color);
    draw_line(pixels, width, height, x1, y0, x1, y1, color);
    draw_line(pixels, width, height, x1, y1, x0, y1, color);
    draw_line(pixels, width, height, x0, y1, x0, y0, color);
}

/// Draw a screen-space circle arc for rotate-mode gizmos.
fn draw_circle(
    pixels: &mut [[u8; 4]],
    width: u32,
    height: u32,
    cx: f32,
    cy: f32,
    radius: f32,
    color: [u8; 4],
) {
    let segments = 64;
    for i in 0..segments {
        let a0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let a1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
        let x0 = cx + radius * a0.cos();
        let y0 = cy + radius * a0.sin();
        let x1 = cx + radius * a1.cos();
        let y1 = cy + radius * a1.sin();
        draw_line(pixels, width, height, x0 as i32, y0 as i32, x1 as i32, y1 as i32, color);
    }
}

/// Minimum distance from a point to a line segment.
fn point_to_segment_dist(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let abx = bx - ax;
    let aby = by - ay;
    let apx = px - ax;
    let apy = py - ay;
    let ab_sq = abx * abx + aby * aby;
    if ab_sq < 0.0001 {
        return (apx * apx + apy * apy).sqrt();
    }
    let t = ((apx * abx + apy * aby) / ab_sq).clamp(0.0, 1.0);
    let closest_x = ax + t * abx;
    let closest_y = ay + t * aby;
    ((px - closest_x).powi(2) + (py - closest_y).powi(2)).sqrt()
}

// ── GizmoRenderer ────────────────────────────────────────────────────────────

impl GizmoRenderer {
    pub fn new() -> Self {
        Self {
            mode: GizmoMode::Translate,
            active_axis: None,
            dragging: false,
            drag_start_screen: None,
            drag_start_world: None,
        }
    }

    /// Compute the three axis endpoints in screen-space.
    /// Returns `(center, x_end, y_end, z_end)` or `None` if the entity is behind the camera.
    fn axis_endpoints(
        &self,
        entity_world_pos: Vec3,
        view_proj: Mat4,
        width: u32,
        height: u32,
    ) -> Option<((f32, f32), (f32, f32), (f32, f32), (f32, f32))> {
        let center = project_to_screen(entity_world_pos, view_proj, width, height)?;
        let arrow_len = world_arrow_length(entity_world_pos, view_proj, width, height);
        let x_end = project_to_screen(
            entity_world_pos + Vec3::X * arrow_len,
            view_proj,
            width,
            height,
        )?;
        let y_end = project_to_screen(
            entity_world_pos + Vec3::Y * arrow_len,
            view_proj,
            width,
            height,
        )?;
        let z_end = project_to_screen(
            entity_world_pos + Vec3::Z * arrow_len,
            view_proj,
            width,
            height,
        )?;
        Some((center, x_end, y_end, z_end))
    }

    /// Draw gizmo overlay into a pixel buffer. Called AFTER scene render.
    pub fn draw_overlay(
        &self,
        pixels: &mut [[u8; 4]],
        width: u32,
        height: u32,
        entity_world_pos: Vec3,
        view_proj: Mat4,
    ) {
        let Some((center, x_end, y_end, z_end)) =
            self.axis_endpoints(entity_world_pos, view_proj, width, height)
        else {
            return;
        };

        // Highlight the active axis
        let x_col = if self.active_axis == Some(Axis::X) { YELLOW } else { RED };
        let y_col = if self.active_axis == Some(Axis::Y) { YELLOW } else { GREEN };
        let z_col = if self.active_axis == Some(Axis::Z) { YELLOW } else { BLUE };

        match self.mode {
            GizmoMode::Translate => {
                // Axis lines
                draw_line(pixels, width, height, center.0 as i32, center.1 as i32, x_end.0 as i32, x_end.1 as i32, x_col);
                draw_line(pixels, width, height, center.0 as i32, center.1 as i32, y_end.0 as i32, y_end.1 as i32, y_col);
                draw_line(pixels, width, height, center.0 as i32, center.1 as i32, z_end.0 as i32, z_end.1 as i32, z_col);
                // Arrowheads
                draw_arrowhead(pixels, width, height, center.0, center.1, x_end.0, x_end.1, x_col);
                draw_arrowhead(pixels, width, height, center.0, center.1, y_end.0, y_end.1, y_col);
                draw_arrowhead(pixels, width, height, center.0, center.1, z_end.0, z_end.1, z_col);
            }
            GizmoMode::Rotate => {
                // Draw circles around each axis (screen-space approximation)
                let radius = ((x_end.0 - center.0).powi(2) + (x_end.1 - center.1).powi(2)).sqrt();
                draw_circle(pixels, width, height, center.0, center.1, radius, x_col);
                let radius_y = ((y_end.0 - center.0).powi(2) + (y_end.1 - center.1).powi(2)).sqrt();
                draw_circle(pixels, width, height, center.0, center.1, radius_y * 0.8, y_col);
                let radius_z = ((z_end.0 - center.0).powi(2) + (z_end.1 - center.1).powi(2)).sqrt();
                draw_circle(pixels, width, height, center.0, center.1, radius_z * 0.6, z_col);
            }
            GizmoMode::Scale => {
                // Lines with cube endpoints
                draw_line(pixels, width, height, center.0 as i32, center.1 as i32, x_end.0 as i32, x_end.1 as i32, x_col);
                draw_line(pixels, width, height, center.0 as i32, center.1 as i32, y_end.0 as i32, y_end.1 as i32, y_col);
                draw_line(pixels, width, height, center.0 as i32, center.1 as i32, z_end.0 as i32, z_end.1 as i32, z_col);
                draw_cube_endpoint(pixels, width, height, x_end.0, x_end.1, x_col);
                draw_cube_endpoint(pixels, width, height, y_end.0, y_end.1, y_col);
                draw_cube_endpoint(pixels, width, height, z_end.0, z_end.1, z_col);
            }
        }
    }

    /// Hit test: is the mouse near a gizmo axis?
    pub fn hit_test(
        &self,
        mouse_x: f32,
        mouse_y: f32,
        entity_world_pos: Vec3,
        view_proj: Mat4,
        screen_width: u32,
        screen_height: u32,
    ) -> Option<Axis> {
        let (center, x_end, y_end, z_end) =
            self.axis_endpoints(entity_world_pos, view_proj, screen_width, screen_height)?;

        let dist_x = point_to_segment_dist(mouse_x, mouse_y, center.0, center.1, x_end.0, x_end.1);
        let dist_y = point_to_segment_dist(mouse_x, mouse_y, center.0, center.1, y_end.0, y_end.1);
        let dist_z = point_to_segment_dist(mouse_x, mouse_y, center.0, center.1, z_end.0, z_end.1);

        // Find the closest axis within tolerance
        let mut best: Option<(Axis, f32)> = None;
        for (axis, dist) in [(Axis::X, dist_x), (Axis::Y, dist_y), (Axis::Z, dist_z)] {
            if dist < HIT_TOLERANCE
                && (best.is_none() || dist < best.unwrap().1) {
                    best = Some((axis, dist));
                }
        }
        best.map(|(a, _)| a)
    }

    /// Begin dragging an axis.
    pub fn begin_drag(&mut self, axis: Axis, mouse_x: f32, mouse_y: f32) {
        self.active_axis = Some(axis);
        self.dragging = true;
        self.drag_start_screen = Some((mouse_x, mouse_y));
        self.drag_start_world = None;
    }

    /// Update drag -- returns world-space delta to apply to entity.
    pub fn update_drag(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        entity_pos: Vec3,
        view_proj: Mat4,
        screen_width: u32,
        screen_height: u32,
    ) -> Vec3 {
        if !self.dragging {
            return Vec3::ZERO;
        }
        let Some(axis) = self.active_axis else {
            return Vec3::ZERO;
        };
        let Some((start_x, start_y)) = self.drag_start_screen else {
            return Vec3::ZERO;
        };

        // Compute pixels-per-world-unit at entity depth
        let arrow_len = world_arrow_length(entity_pos, view_proj, screen_width, screen_height);
        let px_per_unit = ARROW_PIXELS / arrow_len;
        if px_per_unit < 0.001 {
            return Vec3::ZERO;
        }

        // Determine the screen-space direction of this axis
        let Some(center) = project_to_screen(entity_pos, view_proj, screen_width, screen_height) else {
            return Vec3::ZERO;
        };
        let axis_dir = match axis {
            Axis::X => Vec3::X,
            Axis::Y => Vec3::Y,
            Axis::Z => Vec3::Z,
        };
        let Some(axis_end) = project_to_screen(entity_pos + axis_dir, view_proj, screen_width, screen_height) else {
            return Vec3::ZERO;
        };

        // Screen-space axis direction (normalised)
        let screen_axis_x = axis_end.0 - center.0;
        let screen_axis_y = axis_end.1 - center.1;
        let screen_axis_len = (screen_axis_x * screen_axis_x + screen_axis_y * screen_axis_y).sqrt();
        if screen_axis_len < 0.001 {
            return Vec3::ZERO;
        }
        let sax = screen_axis_x / screen_axis_len;
        let say = screen_axis_y / screen_axis_len;

        // Project the mouse delta onto the screen-space axis direction
        let dx = mouse_x - start_x;
        let dy = mouse_y - start_y;
        let projected_px = dx * sax + dy * say;

        // Convert pixel delta to world units
        let world_delta = projected_px / px_per_unit;

        // Update drag start so subsequent calls are incremental
        self.drag_start_screen = Some((mouse_x, mouse_y));

        axis_dir * world_delta
    }

    /// End drag.
    pub fn end_drag(&mut self) {
        self.dragging = false;
        self.active_axis = None;
        self.drag_start_screen = None;
        self.drag_start_world = None;
    }
}

impl Default for GizmoRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_view_proj() -> Mat4 {
        let view = Mat4::look_at_rh(
            Vec3::new(0.0, 5.0, 10.0),
            Vec3::ZERO,
            Vec3::Y,
        );
        let proj = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            1000.0,
        );
        proj * view
    }

    #[test]
    fn project_to_screen_basic() {
        let vp = test_view_proj();
        let s = project_to_screen(Vec3::ZERO, vp, 800, 600);
        assert!(s.is_some());
        let (sx, sy) = s.unwrap();
        // Origin should project roughly to center of screen
        assert!((sx - 400.0).abs() < 50.0, "sx={sx}");
        assert!(sy > 0.0 && sy < 600.0, "sy={sy}");
    }

    #[test]
    fn draw_line_non_empty() {
        let mut pixels = vec![[0u8; 4]; 100 * 100];
        draw_line(&mut pixels, 100, 100, 10, 10, 90, 90, [255, 0, 0, 255]);
        let lit = pixels.iter().filter(|p| p[0] == 255).count();
        assert!(lit > 10, "expected many lit pixels, got {lit}");
    }
}
