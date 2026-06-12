use super::*;
use super::super::*;


    /// M0 ACCEPTANCE: the engine-owned Spectra path tracer renders ONE real
    /// building's cooked SDF as a SOLID, continuous surface via the NATIVE
    /// SDF-volume primitive (sphere trace in megakernel.slang) — NOT the lossy
    /// triangle-quad splat bridge. Loads forge.house.craftsman's cooked 64³-class
    /// GWN-signed field from atoms.json, renders it, writes a PNG, and asserts
    /// the façade fills >= 97% of its projected silhouette bounding box (confetti
    /// would be sparse). Prints the OLD splat-bridge coverage for the same camera
    /// as the confetti→wall delta.
    ///
    /// Run alone (GPU, seconds/frame is fine):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///   SPECTRA_SLANG_DIR=$HOME/src/spectra/slang SLANG_DIR=$HOME/slang-sdk \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --lib sdf_building_renders_solid_surface -- --nocapture --test-threads=1
    #[test]
    fn sdf_building_renders_solid_surface() {
        use super::{LightRig, SdfVolumeInput, pathtrace_sdf_to_rgba};

        // --- Load the cooked SDF from the civitas craftsman asset ----------
        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms")
            .join("forge.house.craftsman.atoms.json");
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse atoms.json");
        let sdf = &json["sdf"];
        let res: Vec<u64> = sdf["resolution"]
            .as_array()
            .expect("resolution array")
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let origin: Vec<f32> = sdf["origin"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let voxel_size = sdf["voxel_size"].as_f64().unwrap() as f32;
        let narrow_band = sdf["narrow_band"].as_f64().unwrap() as f32;
        let snorm: Vec<i64> = sdf["distances_snorm16"]
            .as_array()
            .expect("distances_snorm16 array")
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        // Decode snorm16 -> metres, exactly as vox_physics::sdf::from_snorm16_grid.
        let distances: Vec<f32> = snorm
            .iter()
            .map(|&v| {
                let n = if v as i32 == i16::MIN as i32 {
                    -1.0
                } else {
                    v as f32 / i16::MAX as f32
                };
                n.clamp(-1.0, 1.0) * narrow_band
            })
            .collect();
        let resolution = [res[0] as u32, res[1] as u32, res[2] as u32];
        eprintln!(
            "[sdf_building] craftsman SDF: res={:?} voxel={voxel_size:.4} band={narrow_band:.4} \
             n_dist={} negatives(inside)={}",
            resolution,
            distances.len(),
            distances.iter().filter(|d| **d < 0.0).count()
        );
        assert!(
            distances.iter().any(|d| *d < 0.0),
            "SDF must contain interior (negative) samples — else it is not a solid"
        );

        let volume = SdfVolumeInput {
            resolution,
            origin: [origin[0], origin[1], origin[2]],
            voxel_size,
            narrow_band,
            distances,
        };

        // --- Camera framing the SDF grid (identity instance transform) -----
        // Grid world extent (scale 1, position 0): origin .. origin+(res-1)*voxel.
        let gmin = volume.origin;
        let gmax = [
            volume.origin[0] + (resolution[0] - 1) as f32 * voxel_size,
            volume.origin[1] + (resolution[1] - 1) as f32 * voxel_size,
            volume.origin[2] + (resolution[2] - 1) as f32 * voxel_size,
        ];
        let center = [
            0.5 * (gmin[0] + gmax[0]),
            0.5 * (gmin[1] + gmax[1]),
            0.5 * (gmin[2] + gmax[2]),
        ];
        let radius = {
            let dx = gmax[0] - gmin[0];
            let dy = gmax[1] - gmin[1];
            let dz = gmax[2] - gmin[2];
            0.5 * (dx * dx + dy * dy + dz * dz).sqrt()
        };
        // Front-quarter view, eye pulled back ~2.2 radii so the whole building
        // projects inside the frame.
        let dist = radius * 2.2;
        let eye = [
            center[0] + dist * 0.55,
            center[1] + dist * 0.35,
            center[2] + dist * 0.75,
        ];
        let (w, h) = (192u32, 192u32);
        let fov_y = std::f32::consts::FRAC_PI_4;
        // Dark sky / dark fills so the only lit pixels are the SDF surface — the
        // coverage gate then cleanly separates solid façade from background. The
        // SDF's view-facing fill (in megakernel.slang) lights every hit pixel.
        let rig = LightRig {
            sun_dir: [0.4, 0.7, 0.55],
            sun_intensity: 2.0,
            sky_intensity: 0.0,
            camera_fill: 0.0,
            rim_fill: 0.0,
            sky_dome_intensity: 0.0,
            sky_dome_zenith: [0.0, 0.0, 0.0],
            sky_dome_horizon: [0.0, 0.0, 0.0],
            ..Default::default()
        };

        let rgba = pathtrace_sdf_to_rgba(
            &volume,
            [0.0, 0.0, 0.0],
            1.0,
            [0.72, 0.68, 0.60],
            eye,
            center,
            fov_y,
            w,
            h,
            4,
            &rig,
        )
        .expect("SDF render should succeed");

        let out_dir = std::env::temp_dir();
        let png_path = out_dir.join("ochroma_sdf_craftsman.png");
        write_png_rgba(png_path.to_str().unwrap(), &rgba, w, h);
        eprintln!("[sdf_building] wrote {}", png_path.display());

        // --- Solidity: per-row span-fill (handles the non-rectangular house
        // silhouette). Background is the dark sky; lit pixels are the surface.
        let thr = 30.0f32; // out of 255 luma
        let (sdf_coverage, covered) = surface_solidity(&rgba, w, h, thr, 4);
        assert!(covered > 0, "render is entirely background — SDF never hit");

        // --- OLD splat-bridge coverage for the SAME asset/camera -----------
        // Build camera-facing quad splats over the SDF surface (one per inside
        // voxel boundary) and path-trace them through the lossy bridge, so the
        // confetti→wall delta is printed side by side. Skippable via
        // OCHROMA_SKIP_SPLAT_COMPARE=1 to iterate on the SDF path alone.
        let splat_coverage = if std::env::var("OCHROMA_SKIP_SPLAT_COMPARE").is_ok() {
            f64::NAN
        } else {
            use vox_core::types::GaussianSplat;
            let [nx, ny, nz] = resolution;
            let mut splats: Vec<GaussianSplat> = Vec::new();
            let dist_at = |x: u32, y: u32, z: u32| -> f32 {
                volume.distances[(x + nx * (y + ny * z)) as usize]
            };
            // Surface voxels: inside cell adjacent to an outside cell → a splat
            // at the cell centre. This is the splat sampling the bridge gets.
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        if dist_at(x, y, z) >= 0.0 {
                            continue;
                        }
                        let neighbour_outside = [
                            (x + 1 < nx).then(|| dist_at(x + 1, y, z)),
                            (x > 0).then(|| dist_at(x - 1, y, z)),
                            (y + 1 < ny).then(|| dist_at(x, y + 1, z)),
                            (y > 0).then(|| dist_at(x, y - 1, z)),
                            (z + 1 < nz).then(|| dist_at(x, y, z + 1)),
                            (z > 0).then(|| dist_at(x, y, z - 1)),
                        ]
                        .iter()
                        .flatten()
                        .any(|d| *d >= 0.0);
                        if !neighbour_outside {
                            continue;
                        }
                        let p = [
                            volume.origin[0] + x as f32 * voxel_size,
                            volume.origin[1] + y as f32 * voxel_size,
                            volume.origin[2] + z as f32 * voxel_size,
                        ];
                        splats.push(GaussianSplat::surface(
                            p,
                            [voxel_size, 0.0, 0.0],
                            [0.0, voxel_size, 0.0],
                            voxel_size,
                            1.0,
                            255,
                            std::array::from_fn(|_| half::f16::from_f32(0.7).to_bits()),
                        ));
                    }
                }
            }
            match super::pathtrace_splats_to_rgba(
                &splats, eye, center, fov_y, w, h, 16, rig.sun_dir,
            ) {
                Ok(srgba) => {
                    let (solidity, sc) = surface_solidity(&srgba, w, h, thr, 4);
                    let png2 = out_dir.join("ochroma_splat_craftsman.png");
                    write_png_rgba(png2.to_str().unwrap(), &srgba, w, h);
                    eprintln!(
                        "[sdf_building] wrote {} (splat bridge, {sc} surface px)",
                        png2.display()
                    );
                    solidity
                }
                Err(e) => {
                    eprintln!("[sdf_building] splat-bridge render failed: {e}");
                    f64::NAN
                }
            }
        };

        eprintln!(
            "[sdf_building] SOLID COVERAGE (per-row span fill — continuous wall ≈ 1.0):\n  \
             SDF native primitive : {:.4} ({covered} surface px)\n  \
             OLD splat bridge     : {:.4}   <-- confetti",
            sdf_coverage, splat_coverage
        );

        assert!(
            sdf_coverage >= 0.97,
            "SDF façade must be a continuous filled surface (span-fill solidity \
             {:.4} < 0.97). A solid building fills each scanline span; confetti \
             leaves gaps.",
            sdf_coverage
        );
    }


    /// M2 ACCEPTANCE: per-surface MATERIAL from the cooked atoms + real GLASS
    /// windows. The engine-owned Spectra path tracer renders one cooked craftsman
    /// as a native SDF (geometry only), and at each surface hit k-NN-gathers the
    /// nearest atoms to recover the per-surface channel + colour — so walls/roof/
    /// trim show real COLOUR (not one flat albedo), and WINDOWS render as real
    /// path-traced GLASS (the SDF glass hit transmits/refracts and continues the
    /// path instead of terminating opaque, so the window is see-through and shows
    /// the bright sky backdrop behind it). Proven by an A/B render: the same
    /// scene with glass reclassified as opaque facade. The pixels that DIFFER are
    /// the glass-classified window pixels, and they get BRIGHTER (sky shows
    /// through) — a wall cannot. Writes both PNGs, prints the glass-pixel count
    /// and the colour-variety + transparency verdict.
    ///
    /// Run alone (GPU, seconds/frame is fine):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///   SPECTRA_SLANG_DIR=$HOME/src/spectra/slang SLANG_DIR=$HOME/slang-sdk \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --lib sdf_craftsman_atom_material_and_glass -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn sdf_craftsman_atom_material_and_glass() {
        use super::{
            pathtrace_sdf_scene_with_atoms_to_rgba, LightRig, SdfSceneInstance,
        };

        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let asset_path = atoms_dir.join("forge.house.craftsman.atoms.json");

        let volume = load_atoms_sdf(&asset_path);
        let (atoms, raw_glass) = load_atoms_material(&asset_path, false);
        let (atoms_opaque, _) = load_atoms_material(&asset_path, true);
        eprintln!(
            "[sdf_glass] craftsman: res={:?} voxel={:.4} band={:.4} atoms={} (glass={})",
            volume.resolution,
            volume.voxel_size,
            volume.narrow_band,
            atoms.len(),
            raw_glass
        );
        assert!(raw_glass > 50, "expected many cooked glass atoms, got {raw_glass}");

        // Place the single building so its base sits on y=0, centred at origin.
        let ground_y = -volume.origin[1];
        let instance = SdfSceneInstance {
            volume_index: 0,
            position: [
                -(volume.origin[0]
                    + (volume.resolution[0] - 1) as f32 * volume.voxel_size * 0.5),
                ground_y,
                -(volume.origin[2]
                    + (volume.resolution[2] - 1) as f32 * volume.voxel_size * 0.5),
            ],
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            uniform_scale: 1.0,
            albedo: [0.7, 0.7, 0.7],
        };
        let instances = [instance];

        // Camera: look at the FRONT facade (the cooked windows are on +Z, the
        // front wall) from slightly above, close enough that windows are large.
        let (w, h) = (320u32, 320u32);
        let fov_y = std::f32::consts::FRAC_PI_4;
        let center = [0.0f32, 4.5, 0.0];
        let eye = [3.0f32, 6.0, 22.0]; // in front (+Z), slight oblique
        // Bright distinctive sky so transparent glass (which transmits to the
        // sky behind) reads visibly brighter than opaque facade.
        let rig = LightRig {
            sun_dir: [0.3, 0.6, 0.7],
            sun_intensity: 2.2,
            sky_intensity: 0.0,
            camera_fill: 0.0,
            rim_fill: 0.0,
            sky_dome_intensity: 1.0,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };

        let t0 = std::time::Instant::now();
        let rgba = pathtrace_sdf_scene_with_atoms_to_rgba(
            &volume_slice(&volume),
            &instances,
            &[atoms],
            eye,
            center,
            fov_y,
            w,
            h,
            6,
            &rig,
        )
        .expect("SDF atom-material + glass render should succeed");
        let secs = t0.elapsed().as_secs_f64();

        let rgba_opaque = pathtrace_sdf_scene_with_atoms_to_rgba(
            &volume_slice(&volume),
            &instances,
            &[atoms_opaque],
            eye,
            center,
            fov_y,
            w,
            h,
            6,
            &rig,
        )
        .expect("control (glass→opaque) render should succeed");

        let out_dir = std::env::temp_dir();
        let glass_png = out_dir.join("ochroma_sdf_craftsman_glass.png");
        let opaque_png = out_dir.join("ochroma_sdf_craftsman_opaque.png");
        write_png_rgba(glass_png.to_str().unwrap(), &rgba, w, h);
        write_png_rgba(opaque_png.to_str().unwrap(), &rgba_opaque, w, h);
        eprintln!("[sdf_glass] wrote {}", glass_png.display());
        eprintln!("[sdf_glass] wrote {} (control)", opaque_png.display());

        // The background is the smooth blue SKY DOME gradient (b clearly the max
        // channel, bright). A pixel is on the BUILDING if it is NOT that bright
        // blue sky. Glass windows can themselves be dark/blue, but they sit
        // inside the building silhouette and (critically) DIFFER between the two
        // renders — the sky is byte-identical between them.
        let is_sky = |px: &[u8]| -> bool {
            let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
            b > r + 14 && b > g + 8 && (r + g + b) > 330
        };

        let mut building_px = 0usize;
        // --- Colour variety across the building (proves per-surface material,
        // not one flat albedo): bucket each building pixel's dominant hue and
        // require several distinct buckets to be well-populated. -------------
        let mut hue_buckets = [0usize; 12];
        // --- Glass detection via A/B diff: a WINDOW pixel is a building pixel
        // whose colour changes meaningfully when glass is enabled vs. when the
        // glass atoms are reclassified to opaque facade. A wall is byte-identical
        // between the two renders; only the path-traced glass (transmit/refract/
        // reflect, blue tint) differs. Here the front windows transmit into the
        // building's darker interior so they go DARKER than the opaque facade. --
        let mut glass_pixels = 0usize;
        let mut glass_diff_sum = 0.0f64;
        for p in 0..(w * h) as usize {
            let g = &rgba[p * 4..p * 4 + 4];
            let o = &rgba_opaque[p * 4..p * 4 + 4];
            // On the building when the OPAQUE control (no glass holes) is not sky.
            let on_building = !is_sky(o);
            if on_building {
                building_px += 1;
                let (r, gg, b) = (g[0] as f32, g[1] as f32, g[2] as f32);
                if r.max(gg).max(b) - r.min(gg).min(b) > 12.0 {
                    let hue = rgb_hue(r, gg, b); // 0..360
                    let bucket = ((hue / 30.0) as usize).min(11);
                    hue_buckets[bucket] += 1;
                }
                // Per-channel absolute difference between the two renders.
                let diff = (g[0] as f32 - o[0] as f32).abs()
                    + (g[1] as f32 - o[1] as f32).abs()
                    + (g[2] as f32 - o[2] as f32).abs();
                if diff > 36.0 {
                    glass_pixels += 1;
                    glass_diff_sum += diff as f64;
                }
            }
        }

        let populated_hues = hue_buckets.iter().filter(|&&n| n >= 8).count();
        let mean_diff = if glass_pixels > 0 {
            glass_diff_sum / glass_pixels as f64
        } else {
            0.0
        };
        eprintln!(
            "[sdf_glass] M2 ATOM MATERIAL + GLASS (all native SDF):\n  \
             building pixels    : {building_px}\n  \
             glass-classified px: {glass_pixels} (window pixels that CHANGE when \
             glass is enabled — path-traced through the pane)\n  \
             mean glass A/B diff: {mean_diff:.1} (sum |ΔRGB| over the pane pixels)\n  \
             colour hue buckets : {populated_hues}/12 populated (per-surface colour \
             variety)\n  \
             hue histogram      : {hue_buckets:?}\n  \
             seconds/frame      : {secs:.2}s (x2 renders)",
        );

        // --- Acceptance gates. ----------------------------------------------
        assert!(
            building_px > 5000,
            "building barely visible ({building_px} px) — camera/placement wrong"
        );
        // Walls/roof/trim must show REAL per-surface colour, not one flat albedo:
        // multiple distinct hue buckets must be populated.
        assert!(
            populated_hues >= 3,
            "expected per-surface colour variety (>= 3 populated hue buckets), got \
             {populated_hues}. A single flat albedo would fill one bucket."
        );
        // WINDOWS must render as GLASS: a real, non-trivial set of building pixels
        // must be path-traced through the pane (differ from the opaque control).
        // A purely opaque building would have ZERO such pixels (the two renders
        // would be byte-identical everywhere on the building).
        assert!(
            glass_pixels >= 200,
            "expected >= 200 glass-classified window pixels (path-traced through \
             the pane, differing from the opaque-glass control), got {glass_pixels}. \
             Either windows are smoothed out of the SDF and the atom gather can't \
             locate them, or the glass hit is terminating opaque instead of \
             transmitting."
        );
    }


    /// Wave-1 Task 1 ACCEPTANCE: the cell-grid atom gather is a pure
    /// acceleration. The SAME craftsman frame rendered through the grid path
    /// (`pathtrace_sdf_scene_textured_to_rgba`, `u_sdf_textured = 1`,
    /// gather-only mode: no materials/textures yet, identical flat-blend
    /// shading) must match the linear-gather oracle
    /// (`pathtrace_sdf_scene_with_atoms_to_rgba`, `u_sdf_textured = 0`)
    /// byte-for-byte within 1 LSB, while visiting ~80x fewer atoms per hit.
    /// The gather cost is host-computed: avg atoms tested per probe = sum of
    /// the 3x3x3 cell-neighbourhood counts at 1,000 surface points (the atoms
    /// themselves ARE surface samples).
    ///
    /// Run alone (GPU, seconds/frame is fine):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --lib sdf_gather_grid_matches_linear -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn sdf_gather_grid_matches_linear() {
        use super::{
            pathtrace_sdf_scene_textured_to_rgba, pathtrace_sdf_scene_with_atoms_to_rgba,
        };

        // Shared craftsman harness: the M2 glass test's placement + camera +
        // rig, so the frame exercises facade, roof, trim AND glass-detect
        // gather paths.
        let hz = craftsman_harness();
        let CraftsmanHarness {
            ref asset_path,
            ref volume,
            ref atoms,
            raw_glass,
            instance,
            eye,
            center,
            fov_y,
            w,
            h,
            rig,
            ..
        } = hz;
        let atoms = atoms.clone();
        let n_atoms = atoms.len();
        eprintln!(
            "[sdf_gather_grid] craftsman: res={:?} voxel={:.4} atoms={n_atoms} (glass={raw_glass})",
            volume.resolution, volume.voxel_size
        );
        let instances = [instance];

        // Linear oracle (M2 entry point, u_sdf_textured = 0: full atom scan).
        let rgba_linear = pathtrace_sdf_scene_with_atoms_to_rgba(
            &volume_slice(volume),
            &instances,
            &[atoms.clone()],
            eye,
            center,
            fov_y,
            w,
            h,
            6,
            &rig,
        )
        .expect("linear-gather oracle render should succeed");

        // Grid path (new entry point, u_sdf_textured = 1: 3x3x3 cell gather,
        // materials empty + channel table all -1 -> identical flat-blend
        // shading, gather-only mode).
        let rgba_grid = pathtrace_sdf_scene_textured_to_rgba(
            &volume_slice(volume),
            &[load_forge_uv_params(asset_path)],
            &instances,
            &[atoms.clone()],
            &[Vec::new()],
            &[super::SdfChannelMaterials {
                material_for_channel: [-1i32; 9],
            }],
            &[],
            &[],
            eye,
            center,
            fov_y,
            w,
            h,
            6,
            0,
            &rig,
            false,
        )
        .expect("grid-gather render should succeed");

        let out_dir = std::env::temp_dir();
        let linear_png = out_dir.join("ochroma_sdf_gather_linear.png");
        let grid_png = out_dir.join("ochroma_sdf_gather_grid.png");
        write_png_rgba(linear_png.to_str().unwrap(), &rgba_linear, w, h);
        write_png_rgba(grid_png.to_str().unwrap(), &rgba_grid, w, h);
        eprintln!("[sdf_gather_grid] wrote {} (linear oracle)", linear_png.display());
        eprintln!("[sdf_gather_grid] wrote {} (cell grid)", grid_png.display());

        // --- Parity: byte-wise max |Δrgb| over the whole RGBA frame. ---------
        assert_eq!(rgba_linear.len(), rgba_grid.len());
        let mut max_d_bytes = 0u8;
        let mut diff_px = 0usize;
        for (i, (a, b)) in rgba_linear.iter().zip(rgba_grid.iter()).enumerate() {
            let d = a.abs_diff(*b);
            if d > 0 && i % 4 != 3 {
                diff_px += 1;
            }
            if d > max_d_bytes {
                max_d_bytes = d;
            }
        }
        let max_drgb = max_d_bytes as f64 / 255.0;

        // --- Gather cost, host-computed from the SAME grid the entry point
        // uploads: rebuild it over the instance's world-space atoms and sum the
        // 3x3x3 neighbourhood counts at 1,000 deterministic surface probes. ---
        let (wmin, wmax) = super::sdf_instance_world_aabb(volume, &instance);
        let q = glam::Quat::from_array(instance.rotation_xyzw).normalize();
        let pos = glam::Vec3::from(instance.position);
        let s = instance.uniform_scale;
        let world_pos: Vec<[f32; 3]> = atoms
            .iter()
            .map(|a| (pos + q * (glam::Vec3::from(a.position) * s)).to_array())
            .collect();
        // Same glass-detect-radius cell floor the entry point applies.
        let glass_detect_r = (volume.voxel_size * instance.uniform_scale * 0.6).max(0.18);
        let grid = super::build_sdf_atom_cell_grid(&world_pos, wmin, wmax, glass_detect_r);
        eprintln!(
            "[sdf_gather_grid] grid: dims={:?} cell={:.3} m cells={} table={} floats",
            grid.dims,
            grid.cell_size,
            grid.cell_table.len(),
            grid.cell_table.len() * 2
        );

        let n_probes = 1000usize;
        let mut seed: u64 = 0x243F_6A88_85A3_08D3; // fixed -> deterministic probes
        let mut total_tested: u64 = 0;
        for _ in 0..n_probes {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = ((seed >> 33) as usize) % n_atoms;
            total_tested += grid.neighbourhood_atom_count(world_pos[idx]) as u64;
        }
        let g_avg = total_tested as f64 / n_probes as f64;
        let speedup = n_atoms as f64 / g_avg;

        // The acceptance line (real measured values, no constants).
        eprintln!(
            "gather: grid={g_avg:.1} atoms/hit (avg) vs linear={n_atoms}, \
             speedup={speedup:.0}x, max|\u{394}rgb|={max_drgb:.4}"
        );
        eprintln!(
            "[sdf_gather_grid] differing RGB bytes: {diff_px} of {}",
            (w * h * 3) as usize
        );

        assert!(
            speedup >= 20.0,
            "cell-grid gather must visit >= 20x fewer atoms than the linear scan \
             (got {speedup:.1}x at {g_avg:.1} atoms/hit vs {n_atoms})"
        );
        assert!(
            max_drgb < 1.0 / 255.0,
            "grid gather must reproduce the linear-gather frame within 1 LSB \
             (max|\u{394}rgb| = {max_drgb:.4}, {diff_px} RGB bytes differ). The grid \
             neighbourhood or cell math is wrong — fix it, don't relax this gate."
        );
    }


    /// Wave-1 Task 2 ACCEPTANCE (CPU-only, no GPU): per-channel material
    /// packing for the textured SDF path. Builds `SdfChannelMaterials` from a
    /// hand-rolled cooked-material list (facade -> material 0 with a real
    /// albedo texture, roof -> material 1, glass -> -1 = keep the M2 BSDF
    /// route) plus the matching `SdfUvParams`, packs both through the SAME
    /// host packers `pathtrace_sdf_scene_textured_to_rgba` uploads with, and
    /// asserts the exact buffer layout and values the megakernel will read:
    /// `g_sdf_channel_material` is 9 floats/instance with the craftsman's
    /// facade slot resolving to a material whose `albedo_tex >= 0` and whose
    /// roughness equals the cooked `roughness_factor`;
    /// `g_sdf_volume_uv_params` is 4 floats/volume carrying the cook's
    /// forge_box_uv constants (w/2, d/2, 1/2.5).
    #[cfg(feature = "spectra-native")]
    #[test]
    fn sdf_channel_material_packing() {
        use super::{
            pack_sdf_channel_material_table, pack_sdf_uv_params, PbrMaterial,
            SdfChannelMaterials, SdfUvParams, SDF_ATOM_CH_FACADE, SDF_ATOM_CH_GLASS,
            SDF_ATOM_CH_ROOF,
        };

        // Hand-rolled cooked-material list (the craftsman shape): facade is
        // material 0 (clapboard, albedo texture 0, cooked roughness 0.6),
        // roof is material 1 (slate, albedo texture 1, cooked roughness 0.6),
        // glass stays -1 so the kernel keeps the M2 glass BSDF route.
        let cooked_facade_roughness = 0.6f32;
        let cooked_roof_roughness = 0.6f32;
        let materials = [
            PbrMaterial {
                base_color: [0.82, 0.75, 0.60],
                roughness: cooked_facade_roughness,
                albedo_tex: 0,
                roughness_tex: 2,
                ..Default::default()
            },
            PbrMaterial {
                base_color: [0.25, 0.25, 0.28],
                roughness: cooked_roof_roughness,
                albedo_tex: 1,
                roughness_tex: 3,
                ..Default::default()
            },
        ];
        let mut table = SdfChannelMaterials {
            material_for_channel: [-1i32; 9],
        };
        table.material_for_channel[SDF_ATOM_CH_FACADE as usize] = 0;
        table.material_for_channel[SDF_ATOM_CH_ROOF as usize] = 1;
        // Two instances sharing the table (a block of two craftsman).
        let per_instance = [table, table];

        let packed = pack_sdf_channel_material_table(&per_instance);
        eprintln!(
            "[sdf_channel_packing] packed channel table ({} floats / {} instances):",
            packed.len(),
            per_instance.len()
        );
        for (ii, inst) in packed.chunks_exact(9).enumerate() {
            eprintln!("  instance {ii}: {inst:?}");
        }

        // Layout: exactly 9 floats per instance, ids verbatim (as f32).
        assert_eq!(
            packed.len(),
            9 * per_instance.len(),
            "channel-material table must be 9 floats per instance"
        );
        for inst in packed.chunks_exact(9) {
            assert_eq!(inst[SDF_ATOM_CH_FACADE as usize], 0.0, "facade slot -> material 0");
            assert_eq!(inst[SDF_ATOM_CH_ROOF as usize], 1.0, "roof slot -> material 1");
            assert_eq!(inst[SDF_ATOM_CH_GLASS as usize], -1.0, "glass slot stays -1 (M2 BSDF)");
        }

        // The facade slot must resolve to a REAL textured material: the id in
        // the packed table indexes the materials slice, whose entry carries an
        // atlas albedo texture and the cooked roughness_factor.
        let facade_id = packed[SDF_ATOM_CH_FACADE as usize] as usize;
        let facade_mat = &materials[facade_id];
        eprintln!(
            "[sdf_channel_packing] facade slot -> material {facade_id}: albedo_tex={} \
             roughness={} (cooked roughness_factor={cooked_facade_roughness})",
            facade_mat.albedo_tex, facade_mat.roughness
        );
        assert!(
            facade_mat.albedo_tex >= 0,
            "facade material must carry a real atlas albedo texture (albedo_tex >= 0)"
        );
        assert_eq!(
            facade_mat.roughness, cooked_facade_roughness,
            "facade material roughness must equal the cooked roughness_factor"
        );
        let roof_id = packed[9 + SDF_ATOM_CH_ROOF as usize] as usize;
        assert_eq!(
            materials[roof_id].roughness, cooked_roof_roughness,
            "roof material roughness must equal the cooked roughness_factor"
        );

        // UV params: 4 floats per volume, the cook's forge_box_uv constants.
        let uvp = pack_sdf_uv_params(&[SdfUvParams {
            offset_x: 9.5 * 0.5,
            offset_z: 10.5 * 0.5,
            tile_recip: 1.0 / 2.5,
        }]);
        eprintln!("[sdf_channel_packing] packed uv params: {uvp:?}");
        assert_eq!(uvp.len(), 4, "uv params must be 4 floats per volume");
        assert_eq!(uvp[0], 4.75, "offset_x = forge_width * 0.5");
        assert_eq!(uvp[1], 5.25, "offset_z = forge_depth * 0.5");
        assert_eq!(uvp[2], 0.4, "tile_recip = 1 / 2.5 m per texture tile");
        assert_eq!(uvp[3], 0.0, "pad slot must be zero");
    }


    /// Wave-1 Task 3 ACCEPTANCE — the M1 textured-building gate. Renders the
    /// cooked craftsman through the textured SDF path (per-channel PolyHaven
    /// PBR materials box-projected at the hit with the cook's forge_box_uv
    /// convention) and through the M2 flat path (same camera), writes both
    /// PNGs, and measures three gates over the front facade:
    ///   (a) DETAIL: mean |∇luminance| over the facade mask, textured vs
    ///       flat-blend — texture must add >= 3.0x the gradient energy the
    ///       k-NN atom blend has (clapboard courses actually land on the wall);
    ///   (b) CONSISTENCY: mean |textured - flat| RGB over >= 2,000 facade px
    ///       < 0.12 — the texture layer sits ON the cooked appearance (same
    ///       box projection + tint normalization the cook used), it does not
    ///       repaint the building;
    ///   (c) PER-CHANNEL ROUGHNESS: the roughness AOV (returned in alpha via
    ///       aov_roughness) differs between the slate porch roof and the
    ///       clapboard wall by > 0.05 — real per-channel PBR dispatch, not one
    ///       material everywhere.
    ///
    /// Run alone (GPU, seconds/frame is fine):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --lib sdf_craftsman_textured_quality -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn sdf_craftsman_textured_quality() {
        use super::{
            pathtrace_sdf_scene_textured_to_rgba, pathtrace_sdf_scene_with_atoms_to_rgba,
        };

        let hz = craftsman_harness();
        let (materials, textures, channel_mats) = load_craftsman_pbr(&hz.asset_path);
        let uv_params = load_forge_uv_params(&hz.asset_path);
        eprintln!(
            "[sdf_textured] craftsman: atoms={} (glass={}) materials={} textures={} \
             uv_params=(w/2={}, d/2={}, 1/tile={})",
            hz.atoms.len(),
            hz.raw_glass,
            materials.len(),
            textures.len(),
            uv_params.offset_x,
            uv_params.offset_z,
            uv_params.tile_recip
        );
        assert!(
            channel_mats.material_for_channel[super::SDF_ATOM_CH_FACADE as usize] >= 0,
            "cooked payload must map the facade channel to a textured material"
        );
        assert!(
            channel_mats.material_for_channel[super::SDF_ATOM_CH_ROOF as usize] >= 0,
            "cooked payload must map the roof channel to a textured material"
        );

        let (w, h) = (hz.w, hz.h);

        // Textured render; alpha carries the primary-hit roughness AOV.
        let t0 = std::time::Instant::now();
        let rgba_tex = pathtrace_sdf_scene_textured_to_rgba(
            &volume_slice(&hz.volume),
            &[uv_params],
            &[hz.instance],
            &[hz.atoms.clone()],
            &[Vec::new()],
            &[channel_mats],
            &materials,
            &textures,
            hz.eye,
            hz.center,
            hz.fov_y,
            w,
            h,
            6,
            0,
            &hz.rig,
            true,
        )
        .expect("textured SDF render should succeed");
        let secs_tex = t0.elapsed().as_secs_f64();

        // Flat control: the M2 path (k-NN atom blend), same camera + rig.
        let rgba_flat = pathtrace_sdf_scene_with_atoms_to_rgba(
            &volume_slice(&hz.volume),
            &[hz.instance],
            &[hz.atoms.clone()],
            hz.eye,
            hz.center,
            hz.fov_y,
            w,
            h,
            6,
            &hz.rig,
        )
        .expect("flat-control (M2) render should succeed");

        // PNGs for the human eyeball line (textured alpha holds the roughness
        // AOV — force it opaque for viewing).
        let out_dir = std::env::temp_dir();
        let tex_png = out_dir.join("sdf_craftsman_textured.png");
        let flat_png = out_dir.join("sdf_craftsman_flat_control.png");
        let mut rgba_tex_view = rgba_tex.clone();
        for px in rgba_tex_view.chunks_exact_mut(4) {
            px[3] = 255;
        }
        write_png_rgba(tex_png.to_str().unwrap(), &rgba_tex_view, w, h);
        write_png_rgba(flat_png.to_str().unwrap(), &rgba_flat, w, h);
        eprintln!("[sdf_textured] wrote {} (textured)", tex_png.display());
        eprintln!("[sdf_textured] wrote {} (M2 flat control)", flat_png.display());

        let idx = |x: u32, y: u32| -> usize { ((y * w + x) * 4) as usize };
        let luma =
            |p: &[u8]| (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32)
                / 255.0;

        // --- Facade mask, classified on the FLAT control so the same pixels
        // are compared in both renders: inside the front-wall super-rect,
        // bright-but-not-blown, and WARM (r >= b rejects the slate porch roof,
        // the blue sky and the cyan glass panes). -----------------------------
        let (rx0, rx1, ry0, ry1) = (55u32, 262u32, 112u32, 232u32);
        let mut mask = vec![false; (w * h) as usize];
        let mut facade_px = 0usize;
        for y in ry0..ry1 {
            for x in rx0..rx1 {
                let p = &rgba_flat[idx(x, y)..idx(x, y) + 4];
                let l = luma(p);
                if l >= 0.45 && l <= 0.97 && p[0] >= p[2] {
                    mask[(y * w + x) as usize] = true;
                    facade_px += 1;
                }
            }
        }
        eprintln!("[sdf_textured] facade-classified pixels: {facade_px}");
        assert!(
            facade_px >= 2000,
            "facade mask too small ({facade_px} px < 2000) — camera/mask drifted"
        );

        // --- Gate (a): facade detail energy (mean |∇luminance| over in-mask
        // neighbour pairs — window/roof edges never enter the metric). --------
        let grad_energy = |rgba: &[u8]| -> f64 {
            let mut sum = 0.0f64;
            let mut pairs = 0usize;
            for y in ry0..ry1 {
                for x in rx0..rx1 {
                    if !mask[(y * w + x) as usize] {
                        continue;
                    }
                    let l = luma(&rgba[idx(x, y)..idx(x, y) + 4]);
                    if x + 1 < rx1 && mask[(y * w + x + 1) as usize] {
                        let r = luma(&rgba[idx(x + 1, y)..idx(x + 1, y) + 4]);
                        sum += (l - r).abs() as f64;
                        pairs += 1;
                    }
                    if y + 1 < ry1 && mask[((y + 1) * w + x) as usize] {
                        let d = luma(&rgba[idx(x, y + 1)..idx(x, y + 1) + 4]);
                        sum += (l - d).abs() as f64;
                        pairs += 1;
                    }
                }
            }
            sum / pairs.max(1) as f64
        };
        let energy_tex = grad_energy(&rgba_tex);
        let energy_flat = grad_energy(&rgba_flat);
        let energy_ratio = energy_tex / energy_flat.max(1e-9);
        let pass_a = energy_ratio >= 3.0;
        eprintln!(
            "M1 texture: facade detail energy {energy_ratio:.2}x flat-blend (gate >= 3.0x) \
             -> {} (textured {energy_tex:.5}, flat {energy_flat:.5})",
            if pass_a { "PASS" } else { "FAIL" }
        );

        // --- Gate (b): consistency — the texture layer must sit ON the cooked
        // appearance (same box projection + tint normalization the cook used
        // to colour the atoms), not repaint the building. ---------------------
        let mut dev_sum = 0.0f64;
        for y in ry0..ry1 {
            for x in rx0..rx1 {
                if !mask[(y * w + x) as usize] {
                    continue;
                }
                let t = &rgba_tex[idx(x, y)..idx(x, y) + 4];
                let f = &rgba_flat[idx(x, y)..idx(x, y) + 4];
                let d = (t[0] as f32 - f[0] as f32).abs()
                    + (t[1] as f32 - f[1] as f32).abs()
                    + (t[2] as f32 - f[2] as f32).abs();
                dev_sum += (d / (3.0 * 255.0)) as f64;
            }
        }
        let mean_dev = dev_sum / facade_px as f64;
        let pass_b = mean_dev < 0.12;
        eprintln!(
            "M1 consistency: mean |hit_rgb - atom_gather_rgb| = {mean_dev:.4} (gate < 0.12) \
             -> {} ({facade_px} facade px)",
            if pass_b { "PASS" } else { "FAIL" }
        );

        // --- Gate (c): per-channel roughness from the AOV (alpha channel of
        // the textured render). Roof sample = SLATE-classified pixels (cool
        // blue-grey, mid luma in the flat control) inside the porch-roof rect
        // — the gable end facing the camera is wall, the porch roof is the
        // visible slate face; facade sample = warm-wall pixels on the clean
        // clapboard right wall, off the silhouette edge. Classifying on the
        // flat control keeps both samples identical regardless of what the
        // textured shade draws.
        let mean_alpha_classified =
            |x0: u32, x1: u32, y0: u32, y1: u32, want_slate: bool| -> (f64, usize) {
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let p = &rgba_flat[idx(x, y)..idx(x, y) + 4];
                        let l = luma(p);
                        let is_slate = p[2] >= p[0] && l > 0.25 && l < 0.75;
                        let is_wall = l >= 0.45 && l <= 0.97 && p[0] >= p[2];
                        if (want_slate && is_slate) || (!want_slate && is_wall) {
                            sum += rgba_tex[idx(x, y) + 3] as f64 / 255.0;
                            count += 1;
                        }
                    }
                }
                (sum / count.max(1) as f64, count)
            };
        // Porch-roof rect (the slate face below the upper windows) and the
        // window-free right-wall strip (x < 254 stays off the sky edge).
        let (rough_roof, n_roof) = mean_alpha_classified(100, 200, 152, 185, true);
        let (rough_wall, n_wall) = mean_alpha_classified(242, 254, 152, 212, false);
        assert!(
            n_roof >= 150,
            "too few slate-classified porch-roof pixels ({n_roof}) — crop/classifier drifted"
        );
        assert!(
            n_wall >= 300,
            "too few wall-classified right-wall pixels ({n_wall}) — crop/classifier drifted"
        );
        let rough_delta = (rough_roof - rough_wall).abs();
        let pass_c = rough_delta > 0.05;
        eprintln!(
            "roughness roof={rough_roof:.3} facade={rough_wall:.3} (|delta|={rough_delta:.3}, \
             gate > 0.05, {n_roof}/{n_wall} px) -> {}",
            if pass_c { "PASS" } else { "FAIL" }
        );
        eprintln!(
            "[sdf_textured] eyeball: clapboard courses + slate porch roof + real roughness \
             split — verify {} against {} ({secs_tex:.2}s textured frame)",
            tex_png.display(),
            flat_png.display()
        );

        // --- The gates. Fix the UV/material binding, never weaken these. -----
        assert!(
            pass_a,
            "texture detail is not landing on the facade (energy ratio {energy_ratio:.2}x \
             < 3.0x; textured {energy_tex:.5} vs flat {energy_flat:.5}). The box-UV \
             projection or the channel->material binding is wrong."
        );
        assert!(
            pass_b,
            "textured facade diverges from the cooked atom appearance (mean dev \
             {mean_dev:.4} >= 0.12) — the kernel's box UV does not match the cook's \
             forge_box_uv convention."
        );
        assert!(
            pass_c,
            "roof and facade roughness do not differ (roof {rough_roof:.3} vs facade \
             {rough_wall:.3}) — per-channel material dispatch is not landing."
        );
    }


    /// M1: the multi-instance proof — a CITY BLOCK of buildings, every surface a
    /// native sphere-traced SDF instance, each building SOLID and per-instance
    /// coloured, buildings occluding each other correctly.
    #[cfg(feature = "spectra-native")]
    #[test]
    fn sdf_city_block_renders_multiple_solid_buildings() {
        use super::pathtrace_sdf_scene_to_rgba;

        let (volumes, instances, eye, center, fov_y, (w, h), rig) = city_block_scene();
        let n_instances = instances.len();

        let t0 = std::time::Instant::now();
        let rgba = pathtrace_sdf_scene_to_rgba(
            &volumes, &instances, eye, center, fov_y, w, h, 4, &rig,
        )
        .expect("SDF cluster render should succeed");
        let secs = t0.elapsed().as_secs_f64();

        let out_dir = std::env::temp_dir();
        let png_path = out_dir.join("ochroma_sdf_city_block.png");
        write_png_rgba(png_path.to_str().unwrap(), &rgba, w, h);
        eprintln!("[sdf_block] wrote {}", png_path.display());

        // --- Total lit coverage (fraction of frame that is solid building). --
        let thr = 30.0f32;
        let lit = (0..(w * h))
            .filter(|&p| luma(&rgba[(p * 4) as usize..(p * 4 + 4) as usize]) > thr)
            .count();
        let total_coverage = lit as f64 / (w * h) as f64;

        // --- Per-instance solidity: project each instance's world AABB to a
        // screen box, then measure SPAN-FILL within that box (the M0 confetti-vs-
        // solid metric, localized): for each row in the box, the lit pixels must
        // fill their span (first→last lit pixel). A SOLID building fills each
        // scanline span ≈ 1.0; confetti leaves gaps; an absent instance has no
        // lit pixels at all. This is robust to the oblique view making the
        // projected 3D-AABB box much larger than the building's silhouette (the
        // box is mostly corner air, which span-fill correctly ignores).
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::from(eye),
            glam::Vec3::from(center),
            glam::Vec3::Y,
        );
        let aspect = w as f32 / h as f32;
        let proj = glam::Mat4::perspective_rh(fov_y, aspect, 0.05, 10_000.0);
        let view_proj = proj * view;
        let project = |p: glam::Vec3| -> Option<(f32, f32)> {
            let clip = view_proj * p.extend(1.0);
            if clip.w <= 1e-4 {
                return None;
            }
            let ndc = clip.truncate() / clip.w;
            // NDC x in [-1,1] → pixel; y flipped (NDC +y up, pixel +y down).
            let px = (ndc.x * 0.5 + 0.5) * w as f32;
            let py = (1.0 - (ndc.y * 0.5 + 0.5)) * h as f32;
            Some((px, py))
        };

        let mut solid_instances = 0usize;
        let mut per_inst_fill: Vec<f64> = Vec::new();
        for inst in &instances {
            let v = &volumes[inst.volume_index as usize];
            let lmin = v.origin;
            let lmax = [
                v.origin[0] + (v.resolution[0] - 1) as f32 * v.voxel_size,
                v.origin[1] + (v.resolution[1] - 1) as f32 * v.voxel_size,
                v.origin[2] + (v.resolution[2] - 1) as f32 * v.voxel_size,
            ];
            let q = glam::Quat::from_array(inst.rotation_xyzw).normalize();
            let pos = glam::Vec3::from(inst.position);
            let (mut bx0, mut by0, mut bx1, mut by1) =
                (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
            let mut any = false;
            for cx in [lmin[0], lmax[0]] {
                for cy in [lmin[1], lmax[1]] {
                    for cz in [lmin[2], lmax[2]] {
                        let world =
                            pos + q * (glam::Vec3::new(cx, cy, cz) * inst.uniform_scale);
                        if let Some((px, py)) = project(world) {
                            bx0 = bx0.min(px);
                            by0 = by0.min(py);
                            bx1 = bx1.max(px);
                            by1 = by1.max(py);
                            any = true;
                        }
                    }
                }
            }
            if !any {
                per_inst_fill.push(0.0);
                continue;
            }
            // Clamp the screen box to the frame; span-fill within it per row.
            let x0 = bx0.floor().clamp(0.0, (w - 1) as f32) as u32;
            let y0 = by0.floor().clamp(0.0, (h - 1) as f32) as u32;
            let x1 = bx1.ceil().clamp(0.0, (w - 1) as f32) as u32;
            let y1 = by1.ceil().clamp(0.0, (h - 1) as f32) as u32;
            let (mut sum_fill, mut span_rows, mut box_lit) = (0.0f64, 0u64, 0u64);
            for y in y0..=y1 {
                let (mut first, mut last, mut count) = (None, 0u32, 0u32);
                for x in x0..=x1 {
                    let i = ((y * w + x) * 4) as usize;
                    if luma(&rgba[i..i + 4]) > thr {
                        first.get_or_insert(x);
                        last = x;
                        count += 1;
                    }
                }
                box_lit += count as u64;
                if let Some(f) = first {
                    let span = last - f + 1;
                    if span >= 3 {
                        sum_fill += count as f64 / span as f64;
                        span_rows += 1;
                    }
                }
            }
            // A building absent from the frame has no lit pixels → fill 0.
            let fill = if span_rows > 0 && box_lit > 0 {
                sum_fill / span_rows as f64
            } else {
                0.0
            };
            per_inst_fill.push(fill);
            // Solid: the building's scanline spans are well-filled (continuous
            // walls/roof) AND it actually has surface pixels in the frame.
            if fill >= 0.80 && box_lit > 20 {
                solid_instances += 1;
            }
        }

        let mean_fill = per_inst_fill.iter().sum::<f64>() / per_inst_fill.len() as f64;
        eprintln!(
            "[sdf_block] MULTI-INSTANCE SDF CLUSTER (all native sphere-traced SDF):\n  \
             instances placed   : {n_instances}\n  \
             instances solid    : {solid_instances} (per-instance span-fill >= 0.80)\n  \
             total lit coverage : {:.4} of frame\n  \
             mean span-fill     : {:.4}\n  \
             seconds/frame      : {:.2}s\n  \
             per-instance fill  : {:?}",
            total_coverage,
            mean_fill,
            secs,
            per_inst_fill
                .iter()
                .map(|f| (f * 100.0).round() / 100.0)
                .collect::<Vec<_>>(),
        );

        // --- Acceptance gates. ----------------------------------------------
        assert!(
            n_instances >= 8,
            "must render >= 8 SDF instances, placed {n_instances}"
        );
        assert!(
            lit > 0,
            "frame is entirely background — no SDF surface was hit"
        );
        // The cluster must be SOLID: the large majority of placed buildings must
        // each fill their projected footprint (multiple distinct solid masses,
        // not one building and not confetti). Allow a couple of edge instances
        // to be partly clipped by the frame.
        assert!(
            solid_instances >= 8,
            "expected >= 8 buildings to render as SOLID surfaces (per-instance \
             span-fill >= 0.80), got {solid_instances} (mean span-fill {:.3}). A \
             solid SDF cluster fills each building's scanline spans; confetti / \
             missing instances do not.",
            mean_fill
        );
        // And the block must occupy a real chunk of the frame (not a single
        // building lost in a sea of sky).
        assert!(
            total_coverage >= 0.15,
            "building cluster should cover a substantial part of the frame, got \
             {total_coverage:.4}"
        );
    }
