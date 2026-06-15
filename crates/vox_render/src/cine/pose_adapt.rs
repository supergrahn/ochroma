//! Photographic adapters from a pure [`CinePose`] to renderer/game state.
//!
//! Two adapters drive the cinematic pose into the two consumers:
//!
//! * [`CinePose::to_camera_layer`] → a Spectra [`CameraLayer`] carrying the
//!   **real** path-traced depth of field (`lens_radius`/`focus_distance`),
//!   shutter window (motion blur), and iris blade count (bokeh shape). The
//!   view/projection are built from the pose's `eye`/`target`/`roll`/`fov_y`.
//! * [`CinePose::to_orbit`] → a game-agnostic [`CineOrbit`]
//!   (`focus`/`distance`/`yaw`/`pitch`) the game layer maps to its orbit camera
//!   (`CityCamera::framing`) for live preview / in-game takeover. The engine
//!   stays game-agnostic: it never names a game type.
//!
//! Both are **pure** (no I/O, no GPU, `Send + Sync`) and never panic.

use glam::{Mat4, Vec3};
use spectra_scene_state::CameraLayer;

use super::sequence::CinePose;
use crate::splat_convert::camera_layer;

/// Near clip plane used when building the cinematic projection (metres).
const CINE_NEAR: f32 = 0.05;
/// Far clip plane used when building the cinematic projection (metres).
const CINE_FAR: f32 = 10_000.0;

/// A game-agnostic orbit decomposition of a [`CinePose`].
///
/// The engine cannot reference the game's `CameraFraming` (engine crates are
/// game-agnostic — see `CLAUDE.md`), so the pose decomposes to this neutral
/// orbit description. The game layer (`CinePlayer::to_framing`, Task 8) maps it
/// onto `CityCamera`'s `{ focus, radius, zoom, yaw, pitch }`.
///
/// The decomposition is the exact inverse of the design's orbit→pose mapping:
/// `eye = focus + distance·(cosθ·sinφ, sinθ, cosθ·cosφ)`, `target = focus`.
/// Here `focus = target`, `distance = |eye - target|`, `pitch (θ) = asin(dir.y)`,
/// `yaw (φ) = atan2(dir.x, dir.z)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CineOrbit {
    /// Look-at point (the orbit pivot).
    pub focus: Vec3,
    /// Eye distance from the focus point.
    pub distance: f32,
    /// Azimuth, radians: `atan2(dir.x, dir.z)`.
    pub yaw: f32,
    /// Elevation, radians: `asin(dir.y)`.
    pub pitch: f32,
    /// Vertical field of view, radians (forwarded for the game's `CameraConfig`).
    pub fov_y: f32,
}

impl CinePose {
    /// Build the Spectra [`CameraLayer`] this pose renders through.
    ///
    /// * View = `Mat4::look_at_rh(eye, target, roll_up)` where the up vector is
    ///   the world up rotated by `roll` about the view forward axis (so a
    ///   non-zero roll is a real Dutch-angle camera, not faked in post).
    /// * Projection = `Mat4::perspective_rh(fov_y, w/h, near, far)`.
    /// * DoF / shutter / bokeh come straight off the pose, threaded through the
    ///   shared [`camera_layer`] builder so DoF is **never zeroed** downstream.
    ///
    /// Pure; never panics (a zero height falls back to aspect 1.0).
    pub fn to_camera_layer(&self, w: u32, h: u32) -> CameraLayer {
        let eye = self.eye();
        let target = self.target();

        // Roll the world-up about the view forward axis so roll != 0 tilts the
        // horizon (a real lens roll, not a post effect).
        let forward = {
            let d = target - eye;
            if d.length_squared() < 1e-12 {
                Vec3::NEG_Z
            } else {
                d.normalize()
            }
        };
        let up = if self.roll().abs() < 1e-9 {
            Vec3::Y
        } else {
            Mat4::from_axis_angle(forward, self.roll()).transform_vector3(Vec3::Y)
        };

        let view = Mat4::look_at_rh(eye, target, up);

        let aspect = if h == 0 {
            1.0
        } else {
            w as f32 / h as f32
        };
        let proj = Mat4::perspective_rh(self.fov_y(), aspect, CINE_NEAR, CINE_FAR);

        // Shared builder threads lens_radius/focus_distance (stops zeroing DoF).
        let mut cam = camera_layer(
            view.to_cols_array(),
            self.fov_y(),
            w,
            h,
            self.lens_radius(),
            self.focus_distance(),
        );
        cam.proj_matrix = proj.to_cols_array();
        cam.near = CINE_NEAR;
        cam.far = CINE_FAR;
        // Cine sidecar fields (unlocked in CameraLayer): drive real motion blur
        // and bokeh shape in the kernel.
        cam.bokeh_blades = self.bokeh_blades();
        cam.shutter_open = self.shutter_open();
        cam.shutter_close = self.shutter_close();
        cam
    }

    /// Decompose the pose into a game-agnostic orbit description for live
    /// preview / in-game takeover.
    ///
    /// Inverse of `eye = focus + distance·(cosθ·sinφ, sinθ, cosθ·cosφ)`. With a
    /// degenerate (zero-length) eye→target offset the orbit collapses to
    /// `distance = 0`, `yaw = pitch = 0` so it never produces NaNs.
    pub fn to_orbit(&self) -> CineOrbit {
        let focus = self.target();
        let offset = self.eye() - focus;
        let distance = offset.length();
        let (yaw, pitch) = if distance < 1e-6 {
            (0.0, 0.0)
        } else {
            let dir = offset / distance;
            (dir.x.atan2(dir.z), dir.y.clamp(-1.0, 1.0).asin())
        };
        CineOrbit {
            focus,
            distance,
            yaw,
            pitch,
            fov_y: self.fov_y(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_to_layer_maps_photographic() {
        let pose = CinePose::test_new(
            /*eye*/ glam::vec3(10.0, 5.0, 10.0),
            /*target*/ glam::vec3(0.0, 0.0, 0.0),
            /*roll*/ 0.0,
            /*fov_y*/ 0.785398,
            /*fstop*/ 2.8,
            /*focal_mm*/ 50.0,
            /*focus_dist*/ 12.0,
            /*shutter_open*/ 0.375,
            /*shutter_close*/ 0.625,
            /*bokeh*/ 6,
            /*ev*/ 0.0,
        );
        let cam = pose.to_camera_layer(1920, 1080);
        println!("lens_radius = {:.4}", cam.lens_radius);
        println!("bokeh_blades = {}", cam.bokeh_blades);
        println!("shutter_close = {:.4}", cam.shutter_close);
        assert!(
            (cam.lens_radius - 50.0 / (2.0 * 2.8)).abs() < 1e-4,
            "lens {}",
            cam.lens_radius
        );
        assert!((cam.focus_distance - 12.0).abs() < 1e-6);
        assert_eq!(cam.bokeh_blades, 6);
        assert!((cam.shutter_close - 0.625).abs() < 1e-6);
        assert!((cam.shutter_open - 0.375).abs() < 1e-6);
        assert_eq!(cam.width, 1920);
        assert_eq!(cam.height, 1080);
    }

    #[test]
    fn to_orbit_inverts_orbit_to_pose() {
        // Construct an eye exactly on the orbit sphere for a known yaw/pitch,
        // then assert to_orbit recovers them.
        let focus = glam::vec3(2.0, 1.0, -3.0);
        let distance = 7.0_f32;
        let yaw = 0.6_f32;
        let pitch = 0.3_f32;
        let dir = glam::vec3(
            pitch.cos() * yaw.sin(),
            pitch.sin(),
            pitch.cos() * yaw.cos(),
        );
        let eye = focus + distance * dir;
        let pose = CinePose::test_new(
            eye, focus, 0.0, 0.785, 2.8, 50.0, distance, 0.5, 0.5, 0, 0.0,
        );
        let orbit = pose.to_orbit();
        assert!((orbit.distance - distance).abs() < 1e-4, "dist {}", orbit.distance);
        assert!((orbit.yaw - yaw).abs() < 1e-4, "yaw {}", orbit.yaw);
        assert!((orbit.pitch - pitch).abs() < 1e-4, "pitch {}", orbit.pitch);
        assert!((orbit.focus - focus).length() < 1e-5);
    }

    #[test]
    fn view_matrix_places_eye() {
        // The view matrix must map the eye to the origin (look_at_rh property).
        let pose = CinePose::test_new(
            glam::vec3(4.0, 3.0, 5.0),
            glam::vec3(0.0, 0.0, 0.0),
            0.0,
            0.785,
            5.6,
            35.0,
            8.0,
            0.5,
            0.5,
            0,
            0.0,
        );
        let cam = pose.to_camera_layer(800, 600);
        let view = Mat4::from_cols_array(&cam.view_matrix);
        let eye_in_view = view.transform_point3(glam::vec3(4.0, 3.0, 5.0));
        assert!(eye_in_view.length() < 1e-4, "eye not at origin: {:?}", eye_in_view);
    }
}
