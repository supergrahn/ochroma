//! Editor-side application of a [`GizmoDelta`] to an entity transform.
//!
//! This is the canonical helper that turns the *incremental* per-frame change
//! produced by [`vox_render::gizmos::GizmoRenderer::update_drag`] into a mutation
//! of an editor entity's transform, honouring the active gizmo mode
//! (translate / rotate / scale) and optional snap-to-grid.
//!
//! It mirrors the inline logic currently living in `engine_runner.rs`
//! (`translation()` extraction + axis-aware snap of the translation delta) and
//! generalises it to rotate / scale so both call sites can share one rule.
//!
//! `engine_runner.rs` can adopt [`apply_delta`] as a follow-up to delete its
//! duplicated inline snap block.

use glam::{Quat, Vec3};
use vox_render::gizmos::{Axis, GizmoDelta};

/// A minimal translate / rotate / scale transform — the subset of entity state
/// a gizmo drag mutates. Mirrors `vox_app::editor::EditorEntity`'s
/// `position` / `rotation` / `scale` fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// A transform positioned at `translation` with identity rotation and unit
    /// scale.
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::default()
        }
    }
}

/// Snap-to-grid configuration for a translation drag.
///
/// When `enabled` and `grid > 0.0`, the translation delta along the active axis
/// is quantised to the nearest multiple of `grid` *before* it is added to the
/// transform — exactly mirroring `engine_runner.rs`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapSettings {
    pub enabled: bool,
    pub grid: f32,
}

impl SnapSettings {
    /// Snapping off.
    pub const OFF: SnapSettings = SnapSettings {
        enabled: false,
        grid: 0.0,
    };

    /// Snapping on, quantising to multiples of `grid`.
    pub fn grid(grid: f32) -> Self {
        Self {
            enabled: true,
            grid,
        }
    }

    fn active(&self) -> bool {
        self.enabled && self.grid > 0.0
    }
}

/// Quantise the translation `delta` to the snap grid along `axis`, matching
/// `engine_runner.rs`: only the active-axis component is snapped, the others are
/// forced to zero (a translate drag only moves along one axis at a time).
fn snap_translation(delta: Vec3, axis: Option<Axis>, snap: SnapSettings) -> Vec3 {
    if !snap.active() {
        return delta;
    }
    let grid = snap.grid;
    match axis {
        Some(Axis::X) => Vec3::new((delta.x / grid).round() * grid, 0.0, 0.0),
        Some(Axis::Y) => Vec3::new(0.0, (delta.y / grid).round() * grid, 0.0),
        Some(Axis::Z) => Vec3::new(0.0, 0.0, (delta.z / grid).round() * grid),
        None => delta,
    }
}

/// Apply one frame's [`GizmoDelta`] to `transform`, honouring the gizmo mode and
/// (for translation) the snap grid along `active_axis`.
///
/// * [`GizmoDelta::Translate`] → snap the offset, then add it to `translation`.
/// * [`GizmoDelta::Rotate`]    → compose the incremental quaternion onto
///   `rotation` (left-multiply, so the delta is applied in world space) and
///   re-normalise.
/// * [`GizmoDelta::Scale`]     → multiply `scale` component-wise.
///
/// Returns the *effective* translation that was applied (after snapping), so
/// callers can record undo / change-tracking exactly as the gizmo moved the
/// entity. For non-translate deltas this is [`Vec3::ZERO`].
pub fn apply_delta(
    transform: &mut Transform,
    delta: GizmoDelta,
    active_axis: Option<Axis>,
    snap: SnapSettings,
) -> Vec3 {
    match delta {
        GizmoDelta::Translate(_) => {
            let applied = snap_translation(delta.translation(), active_axis, snap);
            transform.translation += applied;
            applied
        }
        GizmoDelta::Rotate(q) => {
            transform.rotation = (q * transform.rotation).normalize();
            Vec3::ZERO
        }
        GizmoDelta::Scale(s) => {
            transform.scale *= s;
            Vec3::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;
    use vox_render::gizmos::{GizmoMode, GizmoRenderer};

    /// Camera used to drive the real gizmo: looking at the origin from
    /// (0, 5, 10), 45° FOV, 16:9. Identical shape to vox_render's own gizmo
    /// tests so the projection math is exercised the same way.
    const CAM_EYE: Vec3 = Vec3::new(0.0, 5.0, 10.0);
    const SCREEN_W: u32 = 1920;
    const SCREEN_H: u32 = 1080;

    fn view_proj() -> Mat4 {
        let view = Mat4::look_at_rh(CAM_EYE, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            SCREEN_W as f32 / SCREEN_H as f32,
            0.1,
            1000.0,
        );
        proj * view
    }

    /// Re-derive what the gizmo *should* translate for a given screen-space
    /// pixel delta along the X axis, from the same camera math the gizmo uses:
    /// pixels → world units = projected_pixels / (ARROW_PIXELS / world_arrow_len).
    ///
    /// We reconstruct it geometrically rather than hard-coding a magic number:
    /// project the entity origin and entity+X, measure pixels-per-world-unit,
    /// then invert.
    fn expected_x_world_delta(entity_pos: Vec3, mouse_dx: f32, mouse_dy: f32) -> f32 {
        let vp = view_proj();
        let project = |p: Vec3| -> (f32, f32) {
            let clip = vp * p.extend(1.0);
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            (
                (ndc_x * 0.5 + 0.5) * SCREEN_W as f32,
                (1.0 - (ndc_y * 0.5 + 0.5)) * SCREEN_H as f32,
            )
        };
        let center = project(entity_pos);
        let x_end = project(entity_pos + Vec3::X);
        // Screen-space X-axis direction (normalised).
        let ax = x_end.0 - center.0;
        let ay = x_end.1 - center.1;
        let alen = (ax * ax + ay * ay).sqrt();
        let sax = ax / alen;
        let say = ay / alen;
        // px-per-world-unit at this depth == screen length of one world unit
        // along X == `alen` (since x_end is entity_pos + 1*X).
        //
        // The gizmo instead uses ARROW_PIXELS / world_arrow_length, where
        // world_arrow_length = ARROW_PIXELS / alen, so ARROW_PIXELS cancels and
        // px_per_unit == alen. So the two agree exactly.
        let px_per_unit = alen;
        let projected_px = mouse_dx * sax + mouse_dy * say;
        projected_px / px_per_unit
    }

    #[test]
    fn gizmo_drag_updates_translation() {
        let entity_pos = Vec3::ZERO;
        let vp = view_proj();

        // Drive the REAL gizmo: translate mode, drag the X axis.
        let mut gizmo = GizmoRenderer::new();
        gizmo.mode = GizmoMode::Translate;

        // Begin at a start pixel, then drag +120px in screen X.
        let (sx, sy) = (900.0_f32, 540.0_f32);
        let mouse_dx = 120.0_f32;
        let mouse_dy = 0.0_f32;
        gizmo.begin_drag(Axis::X, sx, sy);
        let delta = gizmo.update_drag(
            sx + mouse_dx,
            sy + mouse_dy,
            entity_pos,
            vp,
            SCREEN_W,
            SCREEN_H,
        );

        // It must be a Translate delta.
        match delta {
            GizmoDelta::Translate(_) => {}
            other => panic!("expected Translate delta, got {other:?}"),
        }

        // Apply through the canonical helper (no snapping).
        let mut t = Transform::from_translation(entity_pos);
        let applied = apply_delta(&mut t, delta, gizmo.active_axis, SnapSettings::OFF);

        // Expectation computed from the camera math, not a magic number.
        let expected = expected_x_world_delta(entity_pos, mouse_dx, mouse_dy);
        assert!(
            expected.abs() > 0.01,
            "sanity: drag should produce a non-trivial world delta, got {expected}"
        );
        assert!(
            (t.translation.x - expected).abs() < 1e-3,
            "translation.x moved by {} but expected {expected}",
            t.translation.x
        );
        assert!(
            (applied.x - expected).abs() < 1e-3,
            "returned applied delta {} != expected {expected}",
            applied.x
        );
        // X gizmo must not touch Y or Z.
        assert!(t.translation.y.abs() < 1e-6, "y changed: {}", t.translation.y);
        assert!(t.translation.z.abs() < 1e-6, "z changed: {}", t.translation.z);
        // Rotation / scale untouched by a translate drag.
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn gizmo_drag_snaps_to_grid() {
        // With snap=0.5, a raw translation of 0.7 units must land on 0.5.
        let mut t = Transform::from_translation(Vec3::ZERO);
        let raw = GizmoDelta::Translate(Vec3::new(0.7, 0.0, 0.0));
        let applied = apply_delta(
            &mut t,
            raw,
            Some(Axis::X),
            SnapSettings::grid(0.5),
        );
        assert_eq!(applied, Vec3::new(0.5, 0.0, 0.0), "0.7 should snap to 0.5");
        assert_eq!(t.translation, Vec3::new(0.5, 0.0, 0.0));

        // And a raw 0.7 with snap 0.5 on Y lands on 0.5 on Y only.
        let mut t2 = Transform::from_translation(Vec3::ZERO);
        let raw_y = GizmoDelta::Translate(Vec3::new(0.0, 0.7, 0.0));
        apply_delta(&mut t2, raw_y, Some(Axis::Y), SnapSettings::grid(0.5));
        assert_eq!(t2.translation, Vec3::new(0.0, 0.5, 0.0));
    }

    #[test]
    fn gizmo_rotate_drag_rotates_only() {
        let entity_pos = Vec3::ZERO;
        let vp = view_proj();

        // Drive the REAL gizmo in rotate mode about the Y axis.
        let mut gizmo = GizmoRenderer::new();
        gizmo.mode = GizmoMode::Rotate;

        let (sx, sy) = (900.0_f32, 540.0_f32);
        // The Y axis projects to a roughly-vertical screen line; dragging
        // along it produces a non-trivial projected pixel motion. Use a large
        // mixed delta so the projection onto the Y screen-axis is substantial.
        let mouse_dx = 0.0_f32;
        let mouse_dy = -150.0_f32;
        gizmo.begin_drag(Axis::Y, sx, sy);
        let delta = gizmo.update_drag(
            sx + mouse_dx,
            sy + mouse_dy,
            entity_pos,
            vp,
            SCREEN_W,
            SCREEN_H,
        );

        let q = match delta {
            GizmoDelta::Rotate(q) => q,
            other => panic!("expected Rotate delta, got {other:?}"),
        };

        // The returned quaternion spins about Y; its angle is the gizmo's
        // pixel→radian mapping. Capture it as the expectation directly from the
        // real delta, then confirm the helper applies exactly that rotation.
        let (delta_axis, delta_angle) = q.to_axis_angle();
        assert!(
            delta_angle.abs() > 1e-3,
            "rotate drag should produce a non-trivial angle, got {delta_angle}"
        );
        // Axis should be (anti)parallel to world Y.
        assert!(
            delta_axis.normalize().dot(Vec3::Y).abs() > 0.999,
            "rotation axis {delta_axis:?} not aligned with Y"
        );

        let mut t = Transform::default();
        apply_delta(&mut t, delta, gizmo.active_axis, SnapSettings::OFF);

        // The resulting rotation equals the delta applied to identity.
        let (res_axis, res_angle) = t.rotation.to_axis_angle();
        // Normalise sign: axis-angle is ambiguous up to (−axis, −angle).
        let signed_res = res_axis.dot(Vec3::Y).signum() * res_angle;
        let signed_delta = delta_axis.dot(Vec3::Y).signum() * delta_angle;
        assert!(
            (signed_res - signed_delta).abs() < 1e-4,
            "applied rotation angle {signed_res} != delta angle {signed_delta}"
        );

        // Translation and scale must be untouched by a rotate drag.
        assert_eq!(t.translation, Vec3::ZERO, "rotate must not translate");
        assert_eq!(t.scale, Vec3::ONE, "rotate must not scale");
    }

    #[test]
    fn rotate_delta_composes_onto_existing_rotation() {
        // Two successive 90° Y rotations compose to 180° about Y.
        let mut t = Transform::default();
        let ninety = GizmoDelta::Rotate(Quat::from_axis_angle(
            Vec3::Y,
            std::f32::consts::FRAC_PI_2,
        ));
        apply_delta(&mut t, ninety, Some(Axis::Y), SnapSettings::OFF);
        apply_delta(&mut t, ninety, Some(Axis::Y), SnapSettings::OFF);

        let expected = Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI);
        // Quaternion equality up to sign.
        let dot = t.rotation.dot(expected).abs();
        assert!(dot > 0.9999, "composed rotation {:?} != 180° about Y", t.rotation);
        assert_eq!(t.translation, Vec3::ZERO);
    }

    #[test]
    fn scale_delta_multiplies_only_active_axis() {
        let mut t = Transform::default();
        let s = GizmoDelta::Scale(Vec3::new(2.0, 1.0, 1.0));
        apply_delta(&mut t, s, Some(Axis::X), SnapSettings::OFF);
        assert_eq!(t.scale, Vec3::new(2.0, 1.0, 1.0));
        assert_eq!(t.translation, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
    }
}
