//! Resident-renderer steady-state witness (Render Keystone T1).
//!
//! Constructs a [`ResidentCityRenderer`] ONCE over a city-representative scene
//! (a grid of cubes with distinct PBR materials) and renders N frames reusing all
//! GPU state, streaming only the camera. Prints `cold_ms` (frame 0: device init +
//! slangc compile + BLAS/TLAS build) and `steady_ms` (median of frames 1..N).
//!
//! Run:
//!   cargo bench -p vox_render --bench spectra_resident --features spectra-native -- --nocapture
//!
//! Pass criterion (the resident proof): `steady_ms < cold_ms * 0.1` (steady is
//! >10x faster than cold — no per-frame rebuild) and, for the city scene on the
//! RADV/780M, `steady_ms <= 33.0`.

#[cfg(not(feature = "spectra-native"))]
fn main() {
    eprintln!("spectra_resident bench requires --features spectra-native");
}

#[cfg(feature = "spectra-native")]
fn main() {
    use vox_render::resident_renderer::ResidentCityRenderer;
    use vox_render::splat_backend::LightRig;

    // ── Honest cold start: redirect the SPIR-V artifact cache to a FRESH temp
    //    dir so frame 0 pays the genuine slangc compile (the multi-second
    //    one-time cost the resident loop amortizes). Without this, a developer's
    //    warm ~/.cache/spectra-spirv makes cold≈steady and the ratio gate would
    //    be vacuous. Set RESIDENT_KEEP_CACHE=1 to use the real warm cache. ──
    let mut tmp_cache_dir: Option<std::path::PathBuf> = None;
    if std::env::var("RESIDENT_KEEP_CACHE").as_deref() != Ok("1") {
        let dir = std::env::temp_dir().join(format!(
            "spectra-spirv-resident-bench-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp spirv cache dir");
        // SAFETY: single-threaded bench setup, before any renderer construction.
        unsafe { std::env::set_var("SPECTRA_SPIRV_CACHE_DIR", &dir) };
        eprintln!("spectra_resident: isolated SPIR-V cache at {}", dir.display());
        tmp_cache_dir = Some(dir);
    }

    // ── Levers (env-overridable). Default internal res 480x270 is the measured
    //    RADV/780M real-time point (steady_ms <= 33.0); FSR upscales output. ──
    let width: u32 = env_u32("RESIDENT_W", 480);
    let height: u32 = env_u32("RESIDENT_H", 270);
    let spp: u32 = env_u32("RESIDENT_SPP", 1);
    let max_bounces: u32 = env_u32("RESIDENT_BOUNCES", 2);
    let frames: u32 = env_u32("RESIDENT_FRAMES", 16);
    let grid: u32 = env_u32("RESIDENT_GRID", 6); // grid×grid city blocks

    let scene = build_city_scene(width, height, grid);
    let blocks = grid * grid;
    eprintln!(
        "spectra_resident: building city scene — {blocks} blocks, {}x{} internal, spp={spp}, bounces={max_bounces}, frames={frames}",
        width, height
    );

    let rig = LightRig::default();
    let t_cold = std::time::Instant::now();
    let mut r = match ResidentCityRenderer::new(width, height, rig, spp, max_bounces, scene) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ResidentCityRenderer::new failed: {e}");
            // Surface as a hard failure so CI/the gate notices.
            std::process::exit(1);
        }
    };
    let construct_ms = t_cold.elapsed().as_secs_f64() * 1000.0;

    // ── Frame loop: construct once, stream camera. ──
    let target = [0.0f32, 4.0, 0.0];
    let mut cold_ms = 0.0f64;
    let mut steady: Vec<f64> = Vec::with_capacity(frames.saturating_sub(1) as usize);
    let proj = perspective(60f32.to_radians(), width as f32 / height as f32, 0.1, 1000.0);

    for f in 0..frames {
        // Orbit the camera so each frame is a genuine camera-only stream (not a
        // repeat of the identical view).
        let theta = 0.35 + (f as f32) * 0.04;
        let radius = (grid as f32) * 9.0;
        let eye = [
            target[0] + radius * theta.cos(),
            target[1] + radius * 0.45,
            target[2] + radius * theta.sin(),
        ];
        let view = look_at_rh(eye, target, [0.0, 1.0, 0.0]);

        let t0 = std::time::Instant::now();
        let frame = match r.render_camera(view, proj) {
            Ok(fo) => fo,
            Err(e) => {
                eprintln!("render_camera frame {f} failed: {e}");
                std::process::exit(1);
            }
        };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        // Touch the output so the optimizer can't elide the render.
        let lum = frame.mean_luminance();
        if f == 0 {
            cold_ms = ms;
            eprintln!("frame 0 (cold): {ms:.2}ms, mean_luminance={lum:.4}");
        } else {
            steady.push(ms);
        }
    }

    // Clean up the isolated SPIR-V cache (best-effort).
    if let Some(dir) = tmp_cache_dir.take() {
        let _ = std::fs::remove_dir_all(&dir);
    }

    let steady_ms = median(&mut steady);
    let ratio = if cold_ms > 0.0 {
        steady_ms / cold_ms
    } else {
        f64::INFINITY
    };

    // ── Report (parsed by the gate). ──
    println!("construct_ms: {construct_ms:.2}");
    println!("cold_ms: {cold_ms:.2}");
    println!("steady_ms: {steady_ms:.2}");
    println!("steady/cold ratio: {ratio:.4} (target < 0.1)");
    println!(
        "PASS_REUSE: {}",
        steady_ms < cold_ms * 0.1
    );
    println!("PASS_REALTIME_33ms: {}", steady_ms <= 33.0);

    // Hard-fail the bench if the resident proof or the real-time bar is missed,
    // so the gate cannot pass on a rebuild-every-frame regression.
    if !(steady_ms < cold_ms * 0.1) {
        eprintln!(
            "FAIL: steady_ms ({steady_ms:.2}) not < cold_ms*0.1 ({:.2}) — renderer is rebuilding per frame",
            cold_ms * 0.1
        );
        std::process::exit(2);
    }
    if steady_ms > 33.0 {
        eprintln!("FAIL: steady_ms ({steady_ms:.2}) > 33.0 — misses the 780M real-time bar");
        std::process::exit(3);
    }
}

// ─────────────────────────── helpers ───────────────────────────

#[cfg(feature = "spectra-native")]
fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "spectra-native")]
fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

/// Build a city-representative `SceneState`: a `grid`×`grid` field of axis-aligned
/// box "buildings" of varied height, each with a distinct PBR material. Geometry
/// is a single triangle soup with per-triangle `material_ids`; one TLAS instance
/// covers the whole soup (T1 exercises the resident loop; multi-instance TLAS is
/// T2/T3). Deterministic — same grid yields byte-identical geometry.
#[cfg(feature = "spectra-native")]
fn build_city_scene(width: u32, height: u32, grid: u32) -> spectra_scene_state::SceneState {
    use spectra_scene_state::{MaterialLayer, SceneState};
    use vox_render::splat_backend::{PbrMaterial, pack_vulkan_mesh_material};

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();
    let mut material_ids: Vec<u8> = Vec::new();
    let mut materials: Vec<PbrMaterial> = Vec::new();

    let spacing = 6.0f32;
    let half = (grid as f32 - 1.0) * spacing * 0.5;
    let mut block = 0u32;
    for gz in 0..grid {
        for gx in 0..grid {
            let cx = gx as f32 * spacing - half;
            let cz = gz as f32 * spacing - half;
            // Deterministic per-block height + size variation.
            let h = 3.0 + ((block * 37 + 11) % 13) as f32;
            let w = 2.0 + ((block * 17 + 5) % 5) as f32 * 0.4;
            let mat_id = (materials.len() % 256) as u8;

            // Distinct material per block (cycle brick / glass / metal / stone).
            let m = match block % 4 {
                0 => PbrMaterial {
                    base_color: [0.55, 0.18, 0.12],
                    roughness: 0.85,
                    metallic: 0.0,
                    ..Default::default()
                },
                1 => PbrMaterial {
                    base_color: [0.6, 0.7, 0.85],
                    roughness: 0.05,
                    metallic: 0.0,
                    transmission: 0.85,
                    ior: 1.5,
                    ..Default::default()
                },
                2 => PbrMaterial {
                    base_color: [0.8, 0.8, 0.82],
                    roughness: 0.15,
                    metallic: 0.9,
                    ..Default::default()
                },
                _ => PbrMaterial {
                    base_color: [0.7, 0.68, 0.6],
                    roughness: 0.7,
                    metallic: 0.0,
                    ..Default::default()
                },
            };
            materials.push(m);

            push_box(
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut indices,
                &mut material_ids,
                [cx, h * 0.5, cz],
                [w, h, w],
                mat_id,
            );
            block += 1;
        }
    }

    // Ground plane (large flat box) with a stone material.
    let ground_mat = (materials.len() % 256) as u8;
    materials.push(PbrMaterial {
        base_color: [0.3, 0.32, 0.3],
        roughness: 0.9,
        metallic: 0.0,
        ..Default::default()
    });
    let extent = half + spacing * 2.0;
    push_box(
        &mut positions,
        &mut normals,
        &mut uvs,
        &mut indices,
        &mut material_ids,
        [0.0, -0.25, 0.0],
        [extent * 2.0, 0.5, extent * 2.0],
        ground_mat,
    );

    let mut scene = SceneState::new(width, height);
    scene.geometry.vertex_count = positions.len();
    scene.geometry.triangle_count = indices.len();
    scene.geometry.positions = positions.iter().flat_map(|p| *p).collect();
    scene.geometry.normals = normals.iter().flat_map(|n| *n).collect();
    scene.geometry.uvs = uvs.iter().flat_map(|t| *t).collect();
    scene.geometry.indices = indices.iter().flat_map(|t| *t).collect();
    scene.geometry.material_ids = material_ids.iter().map(|&m| m as u32).collect();
    // Single identity instance for T1 (multi-instance lands in T2/T3).
    scene.geometry.instance_count = 1;
    scene.geometry.instance_transforms = vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, //
    ];

    let mut params = Vec::with_capacity(materials.len() * 156);
    for m in &materials {
        params.extend_from_slice(&pack_vulkan_mesh_material(*m));
    }
    scene.materials = MaterialLayer {
        params,
        spectral_spd: Default::default(),
        material_count: materials.len(),
    };

    // Initial camera (overwritten per frame by render_camera).
    let eye = [extent, extent * 0.5, extent];
    scene.camera.view_matrix = look_at_rh(eye, [0.0, 4.0, 0.0], [0.0, 1.0, 0.0]);
    scene.camera.fov_y_radians = 60f32.to_radians();
    scene.camera.width = width;
    scene.camera.height = height;

    scene.mark_geometry_changed();
    scene.mark_materials_changed();
    scene.mark_camera_changed();
    scene
}

/// Append a unit box (centered at `center`, full `size`) to the soup with flat
/// per-face normals and the given `mat_id` on every triangle.
#[cfg(feature = "spectra-native")]
#[allow(clippy::too_many_arguments)]
fn push_box(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<[u32; 3]>,
    material_ids: &mut Vec<u8>,
    center: [f32; 3],
    size: [f32; 3],
    mat_id: u8,
) {
    let h = [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5];
    // 8 corners.
    let c = center;
    let corners = [
        [c[0] - h[0], c[1] - h[1], c[2] - h[2]], // 0
        [c[0] + h[0], c[1] - h[1], c[2] - h[2]], // 1
        [c[0] + h[0], c[1] + h[1], c[2] - h[2]], // 2
        [c[0] - h[0], c[1] + h[1], c[2] - h[2]], // 3
        [c[0] - h[0], c[1] - h[1], c[2] + h[2]], // 4
        [c[0] + h[0], c[1] - h[1], c[2] + h[2]], // 5
        [c[0] + h[0], c[1] + h[1], c[2] + h[2]], // 6
        [c[0] - h[0], c[1] + h[1], c[2] + h[2]], // 7
    ];
    // 6 faces: (corner indices, normal).
    let faces: [([usize; 4], [f32; 3]); 6] = [
        ([0, 1, 2, 3], [0.0, 0.0, -1.0]), // back
        ([5, 4, 7, 6], [0.0, 0.0, 1.0]),  // front
        ([4, 0, 3, 7], [-1.0, 0.0, 0.0]), // left
        ([1, 5, 6, 2], [1.0, 0.0, 0.0]),  // right
        ([3, 2, 6, 7], [0.0, 1.0, 0.0]),  // top
        ([4, 5, 1, 0], [0.0, -1.0, 0.0]), // bottom
    ];
    for (quad, n) in faces {
        let base = positions.len() as u32;
        for (k, &ci) in quad.iter().enumerate() {
            positions.push(corners[ci]);
            normals.push(n);
            uvs.push(match k {
                0 => [0.0, 0.0],
                1 => [1.0, 0.0],
                2 => [1.0, 1.0],
                _ => [0.0, 1.0],
            });
        }
        indices.push([base, base + 1, base + 2]);
        indices.push([base, base + 2, base + 3]);
        material_ids.push(mat_id);
        material_ids.push(mat_id);
    }
}

// ── Tiny self-contained linear algebra (avoid pulling glam into the bench's
//    public surface; matches glam's column-major to_cols_array layout). ──

#[cfg(feature = "spectra-native")]
fn look_at_rh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = norm(sub(target, eye));
    let s = norm(cross(f, up));
    let u = cross(s, f);
    // Column-major (matches glam Mat4::to_cols_array).
    [
        s[0], u[0], -f[0], 0.0, //
        s[1], u[1], -f[1], 0.0, //
        s[2], u[2], -f[2], 0.0, //
        -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0, //
    ]
}

#[cfg(feature = "spectra-native")]
fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let g = 1.0 / (fov_y * 0.5).tan();
    let k = far / (near - far);
    // Column-major, RH, depth 0..1 (matches glam perspective_rh).
    [
        g / aspect, 0.0, 0.0, 0.0, //
        0.0, g, 0.0, 0.0, //
        0.0, 0.0, k, -1.0, //
        0.0, 0.0, near * k, 0.0, //
    ]
}

#[cfg(feature = "spectra-native")]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[cfg(feature = "spectra-native")]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[cfg(feature = "spectra-native")]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[cfg(feature = "spectra-native")]
fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt();
    if l > 0.0 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        a
    }
}
