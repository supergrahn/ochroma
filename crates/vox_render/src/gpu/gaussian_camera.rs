//! Pure CPU camera-convention bridge: ochroma `RenderCamera` → spectra
//! `GaussianCamera`.
//!
//! This is camera *math* only — no rasterization — so it lives in a module that
//! both the GPU compositor (`gpu::hybrid_compose_gpu`) and the CPU compositor
//! (`crate::hybrid_compose`) depend on. It was lifted out of the (now-excised)
//! banned rasterizer stack so the Spectra product build obtains the camera
//! bridge without any rasterizer code.

use crate::spectral::RenderCamera;
use spectra_gaussian_render::renderer::GaussianCamera;

/// Build a spectra-convention `GaussianCamera` from ochroma's glam `RenderCamera`.
///
/// ochroma uses `glam::Mat4::look_at_rh` (camera +X right, +Y up, looking down
/// -Z, visible points have negative view-space z) and `perspective_rh`. The
/// spectra projector instead assumes an OpenCV-style camera: +X right, +Y
/// **down**, looking down **+Z**, with `screen = f*cam_xy/cam_z + size/2` and a
/// `cam_z >= near` visibility test. We bridge the conventions by left-multiplying
/// the glam view matrix with `C = diag(1, -1, -1)`, which negates the y and z
/// view-space axes — turning the glam camera basis into the spectra one. Focal
/// lengths come straight from the perspective matrix diagonal:
/// `fx = proj[0][0] * width/2`, `fy = proj[1][1] * height/2`.
pub(crate) fn build_gaussian_camera(
    camera: &RenderCamera,
    width: usize,
    height: usize,
) -> GaussianCamera {
    // glam Mat4 is column-major; `to_cols_array_2d()[c][r]` indexes col c row r.
    let v = camera.view.to_cols_array_2d();
    // Element accessor: view[row][col].
    let m = |r: usize, c: usize| v[c][r];

    // spectra_view = C * glam_view, C = diag(1, -1, -1): negate rows 1 and 2.
    // Stored row-major as required by project_gaussian (view_matrix[row*4+col]).
    let mut view_matrix = [0.0f32; 16];
    for r in 0..4 {
        let s = if r == 1 || r == 2 { -1.0 } else { 1.0 };
        for c in 0..4 {
            view_matrix[r * 4 + c] = s * m(r, c);
        }
    }

    let p = camera.proj.to_cols_array_2d();
    // perspective_rh diagonal: p[0][0] = 1/(aspect*tan(fov/2)), p[1][1] = 1/tan(fov/2).
    let fx = p[0][0].abs() * (width as f32) * 0.5;
    let fy = p[1][1].abs() * (height as f32) * 0.5;

    let proj_matrix = {
        let mut pm = [0.0f32; 16];
        for r in 0..4 {
            for c in 0..4 {
                pm[r * 4 + c] = p[c][r];
            }
        }
        pm
    };

    GaussianCamera {
        view_matrix,
        proj_matrix,
        width,
        height,
        fx,
        fy,
        near: 0.05,
        far: 1.0e6,
    }
}
