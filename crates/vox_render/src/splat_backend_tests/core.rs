use super::*;
use super::super::*;


    #[test]
    fn spectra_render_backend_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SpectraRenderBackend>();
    }


    #[test]
    fn read_last_output_before_first_frame_returns_empty() {
        // Construction requires GPU — just verify the type structure compiles correctly.
        // For GPU smoke test, run manually with: cargo test -p vox_render --features spectra-native
    }


    #[test]
    fn read_last_output_returns_arc() {
        // Type-level check: read_last_output() must return Arc<Vec<u8>>, not Vec<u8>.
        let _: fn(&SpectraRenderBackend) -> std::sync::Arc<Vec<u8>> =
            SpectraRenderBackend::read_last_output;
    }


    /// Real GPU render through the Vulkan (Slang->SPIR-V + ash) backend.
    ///
    /// Constructs a `VulkanSlangBackend` (runs on Mesa/lavapipe CPU Vulkan when no
    /// discrete GPU is present), builds a `Renderer`, loads a small quad scene
    /// directly in front of the camera, renders one frame, and asserts the beauty
    /// buffer has the expected size and real (non-zero-variance) pixel content.
    ///
    /// Run with:
    ///   LD_LIBRARY_PATH=$HOME/.local/slang/lib \
    ///     cargo test -p vox_render --features spectra-native \
    ///     vulkan_backend_renders_quad -- --nocapture
    #[test]
    fn vulkan_backend_renders_quad() {
        use crate::splat_convert::{camera_layer, splats_to_scene};
        use spectra_gpu::VulkanSlangBackend;
        use spectra_renderer::{RenderConfig, Renderer};
        use vox_core::types::GaussianSplat;

        let (w, h) = (64u32, 64u32);

        // 1. Bring up the Vulkan backend. If no Vulkan loader/ICD is reachable
        //    this is an environment limitation, not a code defect — report and skip.
        let gpu = match VulkanSlangBackend::new(0) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[vulkan_backend_renders_quad] backend init failed: {e}");
                panic!("VulkanSlangBackend::new(0) failed: {e}");
            }
        };
        eprintln!(
            "[vulkan_backend_renders_quad] backend up: {}",
            spectra_gpu::GpuBackend::device_name(&gpu)
        );

        // 2. A near-realtime config at our target resolution (low spp for speed).
        let mut config = RenderConfig::near_realtime(w, h);
        config.slang_kernel_dir = super::resolve_slang_kernel_dir();
        assert!(
            config.slang_kernel_dir.is_some(),
            "Slang kernel dir not found (set SPECTRA_SLANG_DIR)"
        );
        let mut renderer = Renderer::new(gpu, config);

        // 3. One bright surface splat at the origin, facing the camera (+Z normal).
        let splat = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            255,
            std::array::from_fn(|_| half::f16::from_f32(0.9).to_bits()),
        );
        let scene = splats_to_scene(&[splat], w, h);
        renderer
            .load_scene_state(scene)
            .expect("load_scene_state should succeed");

        // 4. Camera looking down -Z at the quad from +Z (column-major view).
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.0, 3.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        )
        .to_cols_array();
        let cam = camera_layer(view, std::f32::consts::FRAC_PI_4, w, h);
        renderer.set_camera_view_matrix(cam.view_matrix);
        renderer.set_view_proj(cam.view_matrix);

        // 5. Render one frame.
        let frame = renderer.render().expect("render() should succeed");

        // 6. Assert real pixel content: correct size + non-zero variance.
        assert_eq!(frame.width, w, "output width must match target");
        assert_eq!(frame.height, h, "output height must match target");
        assert_eq!(
            frame.beauty.len(),
            (w * h * 4) as usize,
            "beauty buffer must be width*height*4 floats"
        );

        let n = frame.beauty.len() as f64;
        let mean = frame.beauty.iter().map(|&v| v as f64).sum::<f64>() / n;
        let variance = frame
            .beauty
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        eprintln!(
            "[vulkan_backend_renders_quad] mean={mean:.6} variance={variance:.9} \
             samples_done={}",
            frame.samples_done
        );
        assert!(
            variance > 1e-9,
            "rendered image must have real per-pixel variation (variance={variance})"
        );
    }


    /// First-frame correctness (plan Task 3): the FIRST `render()` of a fresh
    /// process must produce the same image as the second render of the same
    /// scene. Today the first frame races the runtime Slang compile — late
    /// kernels are skipped and it shades through a mismatched fallback path,
    /// so the first mean luma diverges from the second.
    ///
    /// Renders the textured checker quad twice in-process (each call builds a
    /// fresh backend + renderer, so the only shared state is process-wide
    /// kernel-compile state) and asserts the mean lumas agree within 2%.
    ///
    /// NOTE: must be the first GPU work in the process — run it alone:
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --lib first_frame_matches_second -- --nocapture
    #[test]
    fn first_frame_matches_second() {
        use super::{PbrMaterial, TextureImage, pathtrace_mesh_textured_to_rgba};

        let (w, h) = (128u32, 128u32);

        // Same checker quad as vulkan_pathtrace_textured_quad_checkerboard:
        // two CCW triangles facing +Z, UVs [0,1]², 64x64 2x2 checker albedo.
        let positions = [
            [-1.0f32, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let normals = [[0.0f32, 0.0, 1.0]; 4];
        let uvs = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let indices = [[0u32, 1, 2], [0, 2, 3]];
        let material_ids = [0u8, 0];

        let (tw, th) = (64u32, 64u32);
        let mut data = Vec::with_capacity((tw * th * 3) as usize);
        for y in 0..th {
            for x in 0..tw {
                let v = if (x >= tw / 2) != (y >= th / 2) {
                    0.9f32
                } else {
                    0.05f32
                };
                data.extend_from_slice(&[v, v, v]);
            }
        }
        let checker = TextureImage {
            width: tw,
            height: th,
            channels: 3,
            data,
        };
        let materials = [PbrMaterial {
            albedo_tex: 0,
            ..Default::default()
        }];

        let render = || {
            pathtrace_mesh_textured_to_rgba(
                &positions,
                &normals,
                &uvs,
                &indices,
                &material_ids,
                &materials,
                std::slice::from_ref(&checker),
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0],
                std::f32::consts::FRAC_PI_4,
                w,
                h,
                8,
                [0.3, 0.5, 0.8],
            )
            .expect("pathtrace_mesh_textured_to_rgba should succeed")
        };

        let mean_luma = |img: &[u8]| -> f64 {
            let mut sum = 0.0f64;
            let n = (w * h) as usize;
            for i in 0..n {
                sum += 0.2126 * img[i * 4] as f64
                    + 0.7152 * img[i * 4 + 1] as f64
                    + 0.0722 * img[i * 4 + 2] as f64;
            }
            sum / n as f64
        };

        let first = render();
        let second = render();
        assert_eq!(first.len(), (w * h * 4) as usize);
        assert_eq!(second.len(), (w * h * 4) as usize);

        let l1 = mean_luma(&first);
        let l2 = mean_luma(&second);
        let rel = (l1 - l2).abs() / l2.max(1e-9);
        eprintln!(
            "[first_frame_matches_second] first mean luma={l1:.3} second mean luma={l2:.3} \
             rel delta={:.2}%",
            rel * 100.0
        );
        assert!(
            l2 > 1.0,
            "second render must show the lit quad (mean luma={l2:.3}) — \
             scene/kernels are broken if this is black"
        );
        assert!(
            rel <= 0.02,
            "first frame of a fresh process must match the second within 2% \
             (first={l1:.3}, second={l2:.3}, rel delta={:.2}%)",
            rel * 100.0
        );
    }


    /// LightRig parameterization (plan Task 4): the legacy entry point
    /// `pathtrace_mesh_textured_to_rgba` and the new
    /// `pathtrace_mesh_lit_to_rgba` with `LightRig { sun_dir, ..Default }`
    /// must produce BYTE-IDENTICAL images on the same scene (the default rig
    /// reproduces the previously hardcoded sun/sky/fill/rim exactly), and
    /// halving `sun_intensity` must measurably darken the image (>10% lower
    /// mean luma).
    ///
    /// Run with:
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --lib light_rig -- --nocapture
    #[test]
    fn light_rig_default_byte_identical_half_sun_darker() {
        use super::{
            LightRig, PbrMaterial, TextureImage, pathtrace_mesh_lit_to_rgba,
            pathtrace_mesh_textured_to_rgba,
        };

        let (w, h) = (128u32, 128u32);

        // Textured quad facing +Z (same shape as the checkerboard test) so the
        // byte-identity comparison also covers the texture-atlas path. Checker
        // albedo 0.55/0.05 keeps the sunlit half below tonemap saturation so
        // halving the sun shows up linearly in the mean luma.
        let positions = [
            [-1.0f32, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let normals = [[0.0f32, 0.0, 1.0]; 4];
        let uvs = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let indices = [[0u32, 1, 2], [0, 2, 3]];
        let material_ids = [0u8, 0];

        let (tw, th) = (64u32, 64u32);
        let mut data = Vec::with_capacity((tw * th * 3) as usize);
        for y in 0..th {
            for x in 0..tw {
                let v = if (x >= tw / 2) != (y >= th / 2) {
                    0.55f32
                } else {
                    0.05f32
                };
                data.extend_from_slice(&[v, v, v]);
            }
        }
        let checker = TextureImage {
            width: tw,
            height: th,
            channels: 3,
            data,
        };
        let materials = [PbrMaterial {
            albedo_tex: 0,
            ..Default::default()
        }];

        let eye = [0.0f32, 0.0, 3.0];
        let target = [0.0f32, 0.0, 0.0];
        let fov_y = std::f32::consts::FRAC_PI_4;
        let sun_dir = [0.3f32, 0.5, 0.8];
        let spp = 8;

        let legacy = pathtrace_mesh_textured_to_rgba(
            &positions,
            &normals,
            &uvs,
            &indices,
            &material_ids,
            &materials,
            std::slice::from_ref(&checker),
            eye,
            target,
            fov_y,
            w,
            h,
            spp,
            sun_dir,
        )
        .expect("legacy pathtrace_mesh_textured_to_rgba should succeed");

        let render_rig = |rig: &LightRig| {
            pathtrace_mesh_lit_to_rgba(
                &positions,
                &normals,
                &uvs,
                &indices,
                &material_ids,
                &materials,
                std::slice::from_ref(&checker),
                eye,
                target,
                fov_y,
                w,
                h,
                spp,
                rig,
            )
            .expect("pathtrace_mesh_lit_to_rgba should succeed")
        };

        let default_rig = render_rig(&LightRig {
            sun_dir,
            ..Default::default()
        });
        let half_sun = render_rig(&LightRig {
            sun_dir,
            sun_intensity: LightRig::default().sun_intensity * 0.5,
            ..Default::default()
        });

        assert_eq!(legacy.len(), (w * h * 4) as usize);
        assert_eq!(default_rig.len(), (w * h * 4) as usize);
        assert_eq!(half_sun.len(), (w * h * 4) as usize);

        let mean_luma = |img: &[u8]| -> f64 {
            let mut sum = 0.0f64;
            let n = (w * h) as usize;
            for i in 0..n {
                sum += 0.2126 * img[i * 4] as f64
                    + 0.7152 * img[i * 4 + 1] as f64
                    + 0.0722 * img[i * 4 + 2] as f64;
            }
            sum / n as f64
        };

        let l_legacy = mean_luma(&legacy);
        let l_default = mean_luma(&default_rig);
        let l_half = mean_luma(&half_sun);
        let diff_bytes = legacy
            .iter()
            .zip(default_rig.iter())
            .filter(|(a, b)| a != b)
            .count();
        let darker_pct = (1.0 - l_half / l_default.max(1e-9)) * 100.0;
        eprintln!(
            "[light_rig] legacy mean luma={l_legacy:.3} default-rig mean luma={l_default:.3} \
             differing bytes={diff_bytes} | half-sun mean luma={l_half:.3} \
             ({darker_pct:.1}% darker than default)"
        );

        assert!(
            l_default > 10.0,
            "default-rig render must show the lit quad (mean luma={l_default:.3}) — \
             scene/kernels are broken if this is black"
        );
        assert_eq!(
            legacy, default_rig,
            "default LightRig must reproduce the legacy hardcoded rig byte-identically \
             ({diff_bytes} bytes differ; legacy luma={l_legacy:.3}, default luma={l_default:.3})"
        );
        assert!(
            l_half < 0.9 * l_default,
            "halving sun_intensity must darken the image by >10% \
             (default={l_default:.3}, half-sun={l_half:.3}, only {darker_pct:.1}% darker)"
        );
    }


    /// Real GPU render: path-trace a quad with a 2x2 checkerboard albedo
    /// texture and assert the checker actually shows up in the pixels.
    ///
    /// The quad spans [-1,1]² at z=0 with UVs [0,1]², viewed head-on from
    /// (0,0,3). The 64x64 RGB texture is bright (0.9) where the texel's
    /// half-x and half-y quadrant parities differ, dark (0.05) where they
    /// match — so in image space the bright checker quadrants land on one
    /// image diagonal and the dark on the other, regardless of axis flips.
    /// Asserts bright-diagonal mean luminance >= 2x dark-diagonal mean, and
    /// that the SAME quad rendered without a texture shows no such asymmetry
    /// (min/max diagonal ratio > 0.8).
    ///
    /// Run with:
    ///   SPECTRA_BACKEND=vulkan scripts/build-spectra-native.sh test -p vox_render \
    ///     --features spectra-native --lib vulkan_pathtrace_textured_quad_checkerboard \
    ///     -- --nocapture
    #[test]
    fn vulkan_pathtrace_textured_quad_checkerboard() {
        use super::{PbrMaterial, TextureImage, pathtrace_mesh_textured_to_rgba};

        let (w, h) = (128u32, 128u32);

        // Quad: two CCW triangles facing +Z, UVs spanning [0,1]².
        let positions = [
            [-1.0f32, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let normals = [[0.0f32, 0.0, 1.0]; 4];
        let uvs = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let indices = [[0u32, 1, 2], [0, 2, 3]];
        let material_ids = [0u8, 0];

        // 64x64 RGB checkerboard: 2x2 quadrants, bright where the quadrant
        // parities differ (diagonal pattern), dark where they match.
        let (tw, th) = (64u32, 64u32);
        let mut data = Vec::with_capacity((tw * th * 3) as usize);
        for y in 0..th {
            for x in 0..tw {
                let v = if (x >= tw / 2) != (y >= th / 2) {
                    0.9f32
                } else {
                    0.05f32
                };
                data.extend_from_slice(&[v, v, v]);
            }
        }
        let checker = TextureImage {
            width: tw,
            height: th,
            channels: 3,
            data,
        };

        let eye = [0.0f32, 0.0, 3.0];
        let target = [0.0f32, 0.0, 0.0];
        let fov_y = std::f32::consts::FRAC_PI_4;
        let sun_dir = [0.3f32, 0.5, 0.8];
        let spp = 8;

        let render = |materials: &[PbrMaterial], textures: &[TextureImage]| {
            pathtrace_mesh_textured_to_rgba(
                &positions,
                &normals,
                &uvs,
                &indices,
                &material_ids,
                materials,
                textures,
                eye,
                target,
                fov_y,
                w,
                h,
                spp,
                sun_dir,
            )
            .expect("pathtrace_mesh_textured_to_rgba should succeed")
        };

        let textured = render(
            &[PbrMaterial {
                albedo_tex: 0,
                ..Default::default()
            }],
            std::slice::from_ref(&checker),
        );
        let untextured = render(
            &[PbrMaterial {
                base_color: [0.8, 0.8, 0.8],
                ..Default::default()
            }],
            &[],
        );
        assert_eq!(textured.len(), (w * h * 4) as usize);
        assert_eq!(untextured.len(), (w * h * 4) as usize);

        // Mean RGB luminance over a central window (half-size w/8) of each
        // image quadrant. Window centers at 1/4 and 3/4 of the frame project
        // well inside the quad (NDC 0.25..0.75 => world 0.31..0.93 at z=0)
        // and never touch the UV-0.5 checker boundary at world 0.
        let quadrant_mean = |img: &[u8], cx: u32, cy: u32| -> f64 {
            let r = w / 8;
            let mut sum = 0.0f64;
            let mut n = 0u32;
            for y in (cy - r)..(cy + r) {
                for x in (cx - r)..(cx + r) {
                    let i = ((y * w + x) * 4) as usize;
                    sum += (img[i] as f64 + img[i + 1] as f64 + img[i + 2] as f64) / 3.0;
                    n += 1;
                }
            }
            sum / n as f64
        };
        let quads = |img: &[u8]| -> [f64; 4] {
            [
                quadrant_mean(img, w / 4, h / 4),     // image TL
                quadrant_mean(img, 3 * w / 4, h / 4), // image TR
                quadrant_mean(img, w / 4, 3 * h / 4), // image BL
                quadrant_mean(img, 3 * w / 4, 3 * h / 4), // image BR
            ]
        };

        // Diagonal grouping: one diagonal holds the bright checker quadrants,
        // the other the dark — whichever way the image axes map to UV.
        let diag = |q: [f64; 4]| -> (f64, f64) {
            let a = (q[0] + q[3]) / 2.0; // TL + BR
            let b = (q[1] + q[2]) / 2.0; // TR + BL
            (a.max(b), a.min(b))
        };

        let tq = quads(&textured);
        let uq = quads(&untextured);
        let (t_bright, t_dark) = diag(tq);
        let (u_bright, u_dark) = diag(uq);
        eprintln!(
            "[vulkan_pathtrace_textured_quad_checkerboard] textured quadrants \
             TL={:.1} TR={:.1} BL={:.1} BR={:.1} -> bright={t_bright:.1} dark={t_dark:.1}",
            tq[0], tq[1], tq[2], tq[3]
        );
        eprintln!(
            "[vulkan_pathtrace_textured_quad_checkerboard] untextured quadrants \
             TL={:.1} TR={:.1} BL={:.1} BR={:.1} -> max={u_bright:.1} min={u_dark:.1}",
            uq[0], uq[1], uq[2], uq[3]
        );

        assert!(
            t_bright >= 2.0 * t_dark,
            "bright checker quadrants must be >= 2x dark ones \
             (bright={t_bright:.2}, dark={t_dark:.2})"
        );
        assert!(
            u_dark > 0.8 * u_bright,
            "untextured quad must NOT show the checker asymmetry \
             (max={u_bright:.2}, min={u_dark:.2})"
        );
    }
