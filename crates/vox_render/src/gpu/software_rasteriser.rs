use glam::Vec4;
use half::f16;
use vox_core::spectral::{
    linear_to_srgb_gamma, spectral_to_xyz, xyz_to_srgb, Illuminant, SpectralBands,
};
use vox_core::types::GaussianSplat;

use crate::spectral::RenderCamera;

/// RGBA8 framebuffer, row-major layout.
#[derive(Debug, Clone)]
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[u8; 4]>,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![[0u8; 4]; (width * height) as usize],
        }
    }

    /// Front-to-back alpha compositing (pre-multiplied over).
    pub fn blend_pixel(&mut self, x: u32, y: u32, r: u8, g: u8, b: u8, a: u8) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = (y * self.width + x) as usize;
        let dst = &mut self.pixels[idx];

        let sa = a as f32 / 255.0;
        let da = dst[3] as f32 / 255.0;

        // Standard "over" compositing: result = src * src_a + dst * dst_a * (1 - src_a)
        let out_a = sa + da * (1.0 - sa);
        if out_a <= 0.0 {
            return;
        }

        let blend = |sc: u8, dc: u8| -> u8 {
            let s = sc as f32 / 255.0;
            let d = dc as f32 / 255.0;
            let c = (s * sa + d * da * (1.0 - sa)) / out_a;
            (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
        };

        dst[0] = blend(r, dst[0]);
        dst[1] = blend(g, dst[1]);
        dst[2] = blend(b, dst[2]);
        dst[3] = (out_a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
}

/// Intermediate structure for a projected splat.
struct ProjectedSplat {
    screen_x: f32,
    screen_y: f32,
    depth: f32,
    radius_px: f32,
    r: u8,
    g: u8,
    b: u8,
    opacity: f32,
}

/// A CPU software rasteriser that projects 3D Gaussian splats to 2D,
/// sorts them back-to-front, and composites using the spectral pipeline.
pub struct SoftwareRasteriser {
    pub width: u32,
    pub height: u32,
}

impl SoftwareRasteriser {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn render(
        &mut self,
        splats: &[GaussianSplat],
        camera: &RenderCamera,
        illuminant: &Illuminant,
    ) -> Framebuffer {
        let mut fb = Framebuffer::new(self.width, self.height);
        let vp = camera.view_proj();
        let hw = self.width as f32 * 0.5;
        let hh = self.height as f32 * 0.5;

        // 1-3: Project all splats, filter, compute screen-space info
        let mut projected: Vec<ProjectedSplat> = splats
            .iter()
            .filter_map(|splat| {
                let pos = Vec4::new(splat.position[0], splat.position[1], splat.position[2], 1.0);
                let clip = vp * pos;

                // Cull behind camera
                if clip.w <= 0.0 {
                    return None;
                }

                let ndc_x = clip.x / clip.w;
                let ndc_y = clip.y / clip.w;
                let ndc_z = clip.z / clip.w;

                // Frustum cull (with generous margin for large splats)
                if !(-2.0..=2.0).contains(&ndc_x) || !(-2.0..=2.0).contains(&ndc_y) || !(-1.0..=1.0).contains(&ndc_z) {
                    return None;
                }

                // Screen coordinates (y is flipped: NDC +Y is up, screen +Y is down)
                let sx = (ndc_x + 1.0) * hw;
                let sy = (1.0 - ndc_y) * hh;

                // Average scale as world-space radius
                let avg_scale =
                    (splat.scale[0].abs() + splat.scale[1].abs() + splat.scale[2].abs()) / 3.0;

                // Approximate screen-space radius: project a world-space extent
                // Using perspective division: radius_screen = (scale * focal) / depth
                // focal ~ hw / tan(fov/2), but we can derive from proj matrix element [0][0]
                let focal_x = camera.proj.col(0).x * hw;
                let radius_px = (avg_scale * focal_x / clip.w).abs().max(1.0);

                // Convert spectral to sRGB
                let bands = SpectralBands([
                    f16::from_bits(splat.spectral[0]).to_f32(),
                    f16::from_bits(splat.spectral[1]).to_f32(),
                    f16::from_bits(splat.spectral[2]).to_f32(),
                    f16::from_bits(splat.spectral[3]).to_f32(),
                    f16::from_bits(splat.spectral[4]).to_f32(),
                    f16::from_bits(splat.spectral[5]).to_f32(),
                    f16::from_bits(splat.spectral[6]).to_f32(),
                    f16::from_bits(splat.spectral[7]).to_f32(),
                ]);
                let xyz = spectral_to_xyz(&bands, illuminant);
                let linear_rgb = xyz_to_srgb(xyz);
                let r = (linear_to_srgb_gamma(linear_rgb[0]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                let g = (linear_to_srgb_gamma(linear_rgb[1]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                let b = (linear_to_srgb_gamma(linear_rgb[2]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;

                let opacity = splat.opacity as f32 / 255.0;

                Some(ProjectedSplat {
                    screen_x: sx,
                    screen_y: sy,
                    depth: clip.w,
                    radius_px,
                    r,
                    g,
                    b,
                    opacity,
                })
            })
            .collect();

        // 4: Sort back-to-front (farthest first)
        projected.sort_by(|a, b| b.depth.partial_cmp(&a.depth).unwrap_or(std::cmp::Ordering::Equal));

        // 5: Rasterise each projected splat as 2D Gaussian
        for ps in &projected {
            let r_ceil = ps.radius_px.ceil() as i32 * 3; // 3-sigma extent
            let cx = ps.screen_x;
            let cy = ps.screen_y;
            let sigma = ps.radius_px;
            let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);

            let x_min = ((cx - r_ceil as f32).floor() as i32).max(0);
            let x_max = ((cx + r_ceil as f32).ceil() as i32).min(self.width as i32 - 1);
            let y_min = ((cy - r_ceil as f32).floor() as i32).max(0);
            let y_max = ((cy + r_ceil as f32).ceil() as i32).min(self.height as i32 - 1);

            for py in y_min..=y_max {
                for px in x_min..=x_max {
                    let dx = px as f32 + 0.5 - cx;
                    let dy = py as f32 + 0.5 - cy;
                    let dist2 = dx * dx + dy * dy;
                    let gauss = (-dist2 * inv_2sigma2).exp();
                    let alpha = (ps.opacity * gauss * 255.0 + 0.5) as u8;
                    if alpha > 0 {
                        fb.blend_pixel(px as u32, py as u32, ps.r, ps.g, ps.b, alpha);
                    }
                }
            }
        }

        fb
    }
}
