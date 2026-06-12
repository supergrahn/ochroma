use super::*;
use super::super::*;


    /// Mesh M0 ACCEPTANCE: render ONE cooked building — the craftsman — as a
    /// REAL, textured, lit TRIANGLE MESH through the proven single-mesh path
    /// tracer (`pathtrace_mesh_lit_to_rgba`): the pack's actual mesh
    /// (positions/normals/box-UVs/indices + per-triangle material ids), the
    /// cooked per-channel PolyHaven PBR materials loaded via the TextureCache
    /// pattern, a sun+sky+bounce rig, a 3/4-front inspection camera. Writes
    /// `mesh_craftsman_textured.png` plus a flat control (same camera/rig,
    /// textures stripped to flat albedo) and gates on measured outcomes:
    ///   (a) facade DETAIL ENERGY: the per-pixel RGB texture residual
    ///       (|Δrgb| textured-vs-flat over the CPU-rasterized facade mask)
    ///       >= 2.5x the flat control's matched same-face per-pixel RGB noise
    ///       floor — the texture paints real colour onto the wall, well above
    ///       the render-noise floor;
    ///   (b) per-part materials: wall-vs-roof AND wall-vs-door mean |Δrgb|
    ///       > 0.12, plus a wall-vs-trim split self-calibrated against the
    ///       flat control (the cooked trim albedo is near-wall by design) —
    ///       distinct per-channel materials landed;
    ///   (c) lit coverage >= 0.25 of the frame, after the CPU mask and the GPU
    ///       silhouette are shown to agree (the projection cross-check that
    ///       makes every mask-based gate trustworthy).
    /// MATERIALS: loaded via `load_building_mesh_pbr_by_forge_id` (the single
    /// canonical mesh-material loader) — the table is indexed by the FORGE
    /// channel ids the mesh's `material_ids` actually carry, so the window/
    /// door frames (forge id 4) render the TRIM material, not whatever sat at
    /// cooked slot 4 (historically ground_concrete with a red-brick diffuse —
    /// the "brick on the window frames" zoning bug). The wall-vs-trim /
    /// wall-vs-door thresholds below were RE-BASELINED 2026-06-12 on the
    /// post-zoning-fix render (trim = stucco, door = wood; they were first
    /// measured on the cross-wired render where trim was secretly brick and
    /// door was metal).
    /// GLASS: the craftsman's cooked glass channel is opaque (transmission 0
    /// from the cook), so panes render opaque here and the gates stay stable.
    /// Real mesh transmission is gated in `mesh_glass_transmits_checkerboard`,
    /// which also renders the craftsman glass demo.
    ///
    /// Run alone (GPU, seconds/frame is fine):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --profile release-fast --lib mesh_craftsman_textured_lit -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn mesh_craftsman_textured_lit() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig, PbrMaterial};

        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let asset_path = atoms_dir.join("forge.house.craftsman.atoms.json");

        // 1+2: the cooked payload's REAL mesh + per-FORGE-ID PBR materials
        // (table index == forge channel id == mesh.material_ids meaning).
        let mesh = load_craftsman_mesh(&asset_path);
        let (materials, textures, channels) = load_building_mesh_pbr_by_forge_id(&asset_path);
        eprintln!(
            "[mesh_m0] cooked mesh: {} verts, {} tris, {} materials, {} textures",
            mesh.positions.len(),
            mesh.indices.len(),
            materials.len(),
            textures.len()
        );
        let max_id = *mesh.material_ids.iter().max().unwrap() as usize;
        assert!(
            max_id < materials.len(),
            "mesh material id {max_id} out of range of {} cooked materials",
            materials.len()
        );
        // The parts the gates measure, located by cooked channel name.
        let chan_id = |name: &str| -> i32 {
            channels
                .iter()
                .position(|c| c == name)
                .unwrap_or_else(|| panic!("cooked payload has no '{name}' channel"))
                as i32
        };
        let (wall_id, roof_id, trim_id, door_id) = (
            chan_id("facade"),
            chan_id("roof"),
            chan_id("trim"),
            chan_id("door"),
        );
        for (label, id) in [("wall", wall_id), ("roof", roof_id), ("trim", trim_id)] {
            let m = &materials[id as usize];
            assert!(
                m.albedo_tex >= 0 && m.roughness_tex >= 0 && m.normal_tex >= 0,
                "{label} material must carry diffuse+roughness+normal maps \
                 (got {} {} {})",
                m.albedo_tex,
                m.roughness_tex,
                m.normal_tex
            );
        }

        // 4: 3/4-front inspection camera (front facade is +Z, windows on +Z;
        // right wall +X) from slightly above so facade, right wall, porch,
        // trim AND both roof planes are all in frame — and a sun+sky+bounce
        // rig: warm sun from high front-right, blue sky dome, soft sky/camera
        // bounce fills so shadowed faces stay readable.
        let (w, h) = (512u32, 512u32);
        let fov_y = std::f32::consts::FRAC_PI_4;
        // 3/4-front inspection framing: oblique from the front-right and
        // slightly above so facade, right wall, porch, trim, door AND both roof
        // planes are all in frame, the building filling most of the frame.
        let target = [0.0f32, 4.2, 0.6];
        let eye = [8.4f32, 5.6, 12.4];
        let rig = LightRig {
            sun_dir: [0.45, 0.65, 0.55],
            sun_intensity: 2.6,
            sky_intensity: 0.3,
            camera_fill: 0.2,
            rim_fill: 0.0,
            // Lower dome than the SDF rigs: the full-blue 1.0 dome washes the
            // beige/stucco albedo split into neutral grey.
            sky_dome_intensity: 0.65,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };
        // High spp: gate (a) divides the texture residual by the flat
        // control's render-noise floor, which falls as 1/√spp — at low spp the
        // floor is comparable to the (EWA-softened) broad-face texture and
        // starves the ratio. 256 spp drives the floor well under the texture
        // residual so the ratio reflects texture, not variance.
        let spp = 256u32;

        // 5: the textured render through the EXISTING entry point.
        let t0 = std::time::Instant::now();
        let rgba_tex = pathtrace_mesh_lit_to_rgba(
            &mesh.positions,
            &mesh.normals,
            &mesh.uvs,
            &mesh.indices,
            &mesh.material_ids,
            &materials,
            &textures,
            eye,
            target,
            fov_y,
            w,
            h,
            spp,
            &rig,
        )
        .expect("textured craftsman mesh render should succeed");
        let secs = t0.elapsed().as_secs_f64();

        // 6: FLAT control — same camera/rig/mesh, textures stripped to the
        // cooked flat albedo (the detail-energy denominator).
        let flat_materials: Vec<PbrMaterial> = materials
            .iter()
            .map(|m| PbrMaterial {
                albedo_tex: -1,
                roughness_tex: -1,
                normal_tex: -1,
                ..*m
            })
            .collect();
        let rgba_flat = pathtrace_mesh_lit_to_rgba(
            &mesh.positions,
            &mesh.normals,
            &mesh.uvs,
            &mesh.indices,
            &mesh.material_ids,
            &flat_materials,
            &[],
            eye,
            target,
            fov_y,
            w,
            h,
            spp,
            &rig,
        )
        .expect("flat-control craftsman mesh render should succeed");

        let out_dir = std::env::temp_dir();
        let tex_png = out_dir.join("mesh_craftsman_textured.png");
        let flat_png = out_dir.join("mesh_craftsman_flat_control.png");
        let opaque = |mut v: Vec<u8>| -> Vec<u8> {
            for px in v.chunks_exact_mut(4) {
                px[3] = 255;
            }
            v
        };
        write_png_rgba(tex_png.to_str().unwrap(), &opaque(rgba_tex.clone()), w, h);
        write_png_rgba(flat_png.to_str().unwrap(), &opaque(rgba_flat.clone()), w, h);

        // --- Per-material pixel masks from the CPU rasterizer (same pinhole
        // as the GPU), eroded one pixel so silhouette/part edges never enter
        // the part metrics. ---------------------------------------------------
        let (mats_px, tris_px) = rasterize_material_masks(&mesh, eye, target, fov_y, w, h);
        let eroded = |id: i32| -> Vec<bool> {
            let mut m = vec![false; (w * h) as usize];
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let i = (y * w + x) as usize;
                    if mats_px[i] == id
                        && mats_px[i - 1] == id
                        && mats_px[i + 1] == id
                        && mats_px[i - w as usize] == id
                        && mats_px[i + w as usize] == id
                    {
                        m[i] = true;
                    }
                }
            }
            m
        };

        // --- Projection cross-check: the GPU silhouette (non-sky pixels) and
        // the CPU raster coverage must agree, otherwise every mask-based gate
        // below would measure the wrong pixels. The background is the smooth
        // bright-blue sky-dome gradient; building/ground pixels are not it. --
        let is_sky = |px: &[u8]| -> bool {
            let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
            b > r + 14 && b > g + 8 && (r + g + b) > 330
        };
        let mut raster_cov = 0usize;
        let mut lit_px = 0usize;
        let mut overlap = 0usize;
        for p in 0..(w * h) as usize {
            let on_raster = mats_px[p] >= 0;
            let on_render = !is_sky(&rgba_tex[p * 4..p * 4 + 4]);
            raster_cov += on_raster as usize;
            lit_px += on_render as usize;
            overlap += (on_raster && on_render) as usize;
        }
        let agree_raster = overlap as f64 / raster_cov.max(1) as f64;
        let agree_render = overlap as f64 / lit_px.max(1) as f64;
        eprintln!(
            "[mesh_m0] projection cross-check: raster {raster_cov} px, render {lit_px} px, \
             overlap/raster = {agree_raster:.3}, overlap/render = {agree_render:.3}"
        );
        assert!(
            agree_raster >= 0.70 && agree_render >= 0.70,
            "CPU mask and GPU silhouette disagree (overlap/raster {agree_raster:.3}, \
             overlap/render {agree_render:.3}) — the mask projection does not match \
             the render; the gates below would be meaningless"
        );

        let luma = |p: &[u8]| {
            (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) / 255.0
        };
        let idx = |x: u32, y: u32| -> usize { ((y * w + x) * 4) as usize };

        // --- Gate (a): facade detail energy — the TEXTURE RESIDUAL the texture
        // layer adds to the surface, as a multiple of the flat control's own
        // within-face noise floor. ---------------------------------------------
        //
        // Why not a raw |∇luminance| ratio: the forge cook models every
        // clapboard COURSE as real geometry (a stack of beveled course prisms,
        // visible in the renders), so the dominant luminance gradients are
        // course-EDGE shading steps that appear IDENTICALLY in the textured and
        // the flat render — they cancel in a ratio and starve it toward ~1.3x
        // no matter how strong the texture is (measured across cameras/spp).
        //
        // The honest, geometry-immune signal is the per-pixel TEXTURE RESIDUAL
        // — mean |Δrgb| between the textured render and the flat control over
        // the wall mask. The flat control IS the same geometry, lighting and
        // camera with ONLY the texture removed, so this residual is exactly
        // what the texture layer paints onto the surface and nothing else (the
        // shared course-edge geometry cancels). To prove that residual is
        // texture and not Monte-Carlo noise we divide by a MATCHED noise floor:
        // the flat control's own per-pixel |Δrgb| between SAME-FACE horizontal
        // neighbours, ÷√2 (a flat Lambert face is constant shade, so its only
        // neighbour variation is uncorrelated render noise; ÷√2 converts a
        // two-sample difference to a one-sample deviation, matching the
        // residual's one-sample form). Both terms are per-pixel RGB colour
        // deviations — an apples-to-apples ratio. Texture on the surface ⇒
        // residual ≫ floor.
        let wall_mask = eroded(wall_id);
        let wall_px = wall_mask.iter().filter(|&&b| b).count();
        assert!(
            wall_px >= 3000,
            "facade mask too small ({wall_px} px < 3000) — camera drifted off the wall"
        );
        let rgb_dev = |a: &[u8], b: &[u8]| -> f64 {
            ((a[0] as f32 - b[0] as f32).abs()
                + (a[1] as f32 - b[1] as f32).abs()
                + (a[2] as f32 - b[2] as f32).abs()) as f64
                / (3.0 * 255.0)
        };
        // Texture residual: mean |Δrgb| textured-vs-flat over the wall.
        let residual = {
            let mut sum = 0.0f64;
            for (p, &on) in wall_mask.iter().enumerate() {
                if on {
                    sum += rgb_dev(&rgba_tex[p * 4..p * 4 + 4], &rgba_flat[p * 4..p * 4 + 4]);
                }
            }
            sum / wall_px as f64
        };
        // Matched per-pixel RGB noise floor: flat control, same-face horizontal
        // neighbour |Δrgb| ÷ √2.
        let noise_floor = {
            let mut sum = 0.0f64;
            let mut pairs = 0usize;
            for y in 0..h {
                for x in 0..w - 1 {
                    let p = (y * w + x) as usize;
                    if wall_mask[p] && wall_mask[p + 1] && tris_px[p + 1] == tris_px[p] {
                        sum += rgb_dev(
                            &rgba_flat[p * 4..p * 4 + 4],
                            &rgba_flat[(p + 1) * 4..(p + 1) * 4 + 4],
                        );
                        pairs += 1;
                    }
                }
            }
            (sum / pairs.max(1) as f64) / std::f64::consts::SQRT_2
        };
        let energy_ratio = residual / noise_floor.max(1e-9);
        let pass_a = energy_ratio >= 2.5;
        eprintln!(
            "[mesh_m0] facade detail energy {energy_ratio:.2}x flat-control \
             (per-pixel RGB texture residual / matched flat noise floor, gate >= 2.5x) \
             -> {} (residual {residual:.5}, noise floor {noise_floor:.5}, {wall_px} \
             facade px)",
            if pass_a { "PASS" } else { "FAIL" }
        );

        // --- Gate (b): per-part materials — wall vs roof and wall vs trim
        // mean-colour splits in the TEXTURED render, over the eroded
        // per-material masks. Distinct cooked materials must land as distinct
        // rendered parts. -----------------------------------------------------
        let mean_rgb = |rgba: &[u8], mask: &[bool]| -> ([f64; 3], usize) {
            let mut sum = [0.0f64; 3];
            let mut n = 0usize;
            for (p, &on) in mask.iter().enumerate() {
                if on {
                    for (s, &v) in sum.iter_mut().zip(&rgba[p * 4..p * 4 + 3]) {
                        *s += v as f64;
                    }
                    n += 1;
                }
            }
            ([sum[0] / n.max(1) as f64, sum[1] / n.max(1) as f64, sum[2] / n.max(1) as f64], n)
        };
        let split = |a: [f64; 3], b: [f64; 3]| -> f64 {
            ((a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs()) / (3.0 * 255.0)
        };
        let trim_mask = eroded(trim_id);
        let (wall_rgb, n_wall) = mean_rgb(&rgba_tex, &wall_mask);
        let (roof_rgb, n_roof) = mean_rgb(&rgba_tex, &eroded(roof_id));
        let (trim_rgb, n_trim) = mean_rgb(&rgba_tex, &trim_mask);
        let (door_rgb, n_door) = mean_rgb(&rgba_tex, &eroded(door_id));
        assert!(
            n_roof >= 1500,
            "roof mask too small ({n_roof} px < 1500) — roof planes not in frame"
        );
        assert!(
            n_trim >= 200,
            "trim mask too small ({n_trim} px < 200) — trim not visible"
        );
        assert!(
            n_door >= 200,
            "door mask too small ({n_door} px < 200) — door not visible"
        );
        let d_roof = split(wall_rgb, roof_rgb);
        let d_trim = split(wall_rgb, trim_rgb);
        let d_door = split(wall_rgb, door_rgb);
        // The cooked trim albedo [0.85,0.82,0.78] is INTENTIONALLY close to
        // the wall's [0.82,0.75,0.6] (painted stucco trim on a beige
        // clapboard house), so wall-vs-trim stays a self-calibrated check:
        // the FLAT control (pure cooked albedos, same lighting) shows what
        // the maximal trim/wall split looks like here, and the textured
        // render must reproduce at least 60% of it above an absolute floor.
        // Wall-vs-roof and wall-vs-door carry the hard 0.12 distinctness
        // gates. RE-BASELINED 2026-06-12 on the post-zoning-fix render
        // (materials now indexed by FORGE id: trim mask = reveal stucco, door
        // = wood — they were first measured on the cross-wired render where
        // "trim" pixels were glass geometry shaded stucco and the door was
        // detail_metal): measured d_trim 0.057 (flat-control split 0.060),
        // d_door 0.176 — trim floor pinned at 0.03 (just inside the
        // measurement, same ~2x margin as the original 0.008 floor), door
        // keeps the hard 0.12.
        let (wall_rgb_flat, _) = mean_rgb(&rgba_flat, &wall_mask);
        let (trim_rgb_flat, _) = mean_rgb(&rgba_flat, &trim_mask);
        let d_trim_flat = split(wall_rgb_flat, trim_rgb_flat);
        let pass_b1 = d_roof > 0.12;
        let pass_b2 = d_trim > 0.03 && d_trim >= 0.6 * d_trim_flat;
        let pass_b3 = d_door > 0.12;
        eprintln!(
            "[mesh_m0] per-part materials: wall vs roof |Δrgb| = {d_roof:.3} (gate > 0.12) \
             -> {} (wall rgb [{:.0},{:.0},{:.0}] {n_wall} px, roof rgb [{:.0},{:.0},{:.0}] \
             {n_roof} px)",
            if pass_b1 { "PASS" } else { "FAIL" },
            wall_rgb[0],
            wall_rgb[1],
            wall_rgb[2],
            roof_rgb[0],
            roof_rgb[1],
            roof_rgb[2]
        );
        eprintln!(
            "[mesh_m0] per-part materials: wall vs trim |Δrgb| = {d_trim:.3} \
             (gate > 0.03 and >= 0.6x the flat-control split {d_trim_flat:.3}; the \
             cooked trim albedo is near-wall by design) -> {} \
             (trim rgb [{:.0},{:.0},{:.0}] {n_trim} px)",
            if pass_b2 { "PASS" } else { "FAIL" },
            trim_rgb[0],
            trim_rgb[1],
            trim_rgb[2]
        );
        eprintln!(
            "[mesh_m0] per-part materials: wall vs door |Δrgb| = {d_door:.3} (gate > 0.12) \
             -> {} (door rgb [{:.0},{:.0},{:.0}] {n_door} px)",
            if pass_b3 { "PASS" } else { "FAIL" },
            door_rgb[0],
            door_rgb[1],
            door_rgb[2]
        );

        // --- Gate (c): lit coverage of the frame + the building is actually
        // LIT (mean luminance over its pixels well above black). --------------
        let coverage = lit_px as f64 / (w * h) as f64;
        let mut lum_sum = 0.0f64;
        for p in 0..(w * h) as usize {
            if !is_sky(&rgba_tex[p * 4..p * 4 + 4]) {
                lum_sum += luma(&rgba_tex[p * 4..p * 4 + 4]) as f64;
            }
        }
        let mean_lit_luma = lum_sum / lit_px.max(1) as f64;
        let pass_c = coverage >= 0.25;
        eprintln!(
            "[mesh_m0] lit coverage {coverage:.3} of frame (gate >= 0.25) -> {}, \
             seconds/frame {secs:.2}s (mean lit luma {mean_lit_luma:.3})",
            if pass_c { "PASS" } else { "FAIL" }
        );
        eprintln!("[mesh_m0] wrote {}", tex_png.display());
        eprintln!("[mesh_m0] wrote {} (flat control)", flat_png.display());
        eprintln!(
            "[mesh_m0] eyeball: a real textured craftsman — clapboard siding courses, \
             slate roof, stucco trim, pale-blue glass panes (opaque BY CHOICE here so \
             this test's gates stay stable; real transmission is gated in \
             mesh_glass_transmits_checkerboard) — vs the flat-colour control"
        );

        // --- The gates. Fix the UV/material/texture binding, never weaken. ---
        assert!(
            pass_a,
            "texture detail is not landing on the mesh facade (residual/floor \
             {energy_ratio:.2}x < 2.5x; per-pixel RGB texture residual {residual:.5} vs \
             matched flat noise floor {noise_floor:.5}) — the UV or atlas binding is wrong"
        );
        assert!(
            pass_b1,
            "wall and roof do not render as distinct materials \
             (|Δrgb| {d_roof:.3} <= 0.12) — per-triangle material ids are not landing"
        );
        assert!(
            pass_b2,
            "wall and trim do not render as distinct materials \
             (|Δrgb| {d_trim:.3}, floor 0.03, flat-control split {d_trim_flat:.3}) \
             — per-triangle material ids are not landing"
        );
        assert!(
            pass_b3,
            "wall and door do not render as distinct materials \
             (|Δrgb| {d_door:.3} <= 0.12) — per-triangle material ids are not landing"
        );
        assert!(
            pass_c,
            "building covers too little of the inspection frame \
             (coverage {coverage:.3} < 0.25) — camera framing drifted"
        );
        assert!(
            mean_lit_luma > 0.15,
            "building renders nearly black (mean lit luma {mean_lit_luma:.3}) — \
             the light rig is not landing"
        );
    }


    /// Mesh GLASS ISOLATION GATE: prove REAL transmission through the mesh
    /// path tracer on a TRIVIAL, CHEAP scene — a 16x16 emissive checkerboard
    /// wall at z=0 and a two-faced glass slab (front face z=2.0, back face
    /// z=1.96) in front of it, camera at z=6 looking straight through the
    /// slab at the wall. No directional lights, no sky dome: the checker's
    /// white cells are the ONLY emitters, so any pattern the glass-region
    /// pixels show is, by construction, content from BEHIND the glass.
    ///
    /// Gates (each render 256x256 @ 32 spp — SECONDS, printed):
    ///   (packing) the glass material packs a[0]=MAT_GLASS(3), a[10]=ior,
    ///       a[24..27]=absorption_color (0,0,0), a[27]=absorption_depth 1.0,
    ///       a[73]=thin_walled — the Vulkan MaterialData reflection slots,
    ///       carrying the EXACT parameter set the proven SDF window path
    ///       passes to sample_glass in megakernel.slang
    ///       (`sample_glass(wo, n, 1.5, rough, float3(0.0f), 1.0f, ...)`);
    ///       `transmission = 0` still packs MAT_LAMBERT with the historical
    ///       slots untouched;
    ///   (transmission) Pearson correlation between glass-region pixel
    ///       luminance and the KNOWN checker parity projected through each
    ///       pixel's camera ray onto the wall plane — scored against the
    ///       better of two optical models (ideal-slab straight-through, and
    ///       the engine's flipped-normal double-"entering" refraction that
    ///       magnifies the pattern; see the inline comment): transmissive
    ///       r > 0.5 AND r >= 3x the opaque control's r (identical scene, the
    ///       glass material's transmission flipped to 0 — its unlit Lambert
    ///       panes cannot show the wall pattern);
    ///   (sanity) the directly-visible wall outside the glass correlates
    ///       r > 0.8 in BOTH renders, so a transmission failure is
    ///       unambiguously the glass route, not the emissive wall.
    /// Writes `mesh_glass_checker.png` + `mesh_glass_checker_opaque.png`,
    /// then ONE craftsman demo render with the cooked glass channel flipped
    /// transmissive — `mesh_craftsman_glass.png`, eyeball-only, no gate.
    ///
    /// Run alone (GPU):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --release --lib mesh_glass_transmits_checkerboard -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn mesh_glass_transmits_checkerboard() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig, PbrMaterial};

        // --- Packer layout: the MAT_GLASS slots (CPU-only, instant). --------
        let glass_mat = PbrMaterial {
            base_color: [1.0, 1.0, 1.0],
            roughness: 0.0, // delta glass: clean refraction, f/pdf = 1
            transmission: 1.0,
            ior: 1.5,
            thin_walled: false,
            ..Default::default()
        };
        let packed = super::pack_vulkan_mesh_material(glass_mat);
        assert_eq!(packed[0].to_bits(), 3, "a[0] must pack MAT_GLASS (3)");
        assert_eq!(packed[10], 1.5, "a[10] must carry the glass ior");
        assert_eq!(
            &packed[24..28],
            &[0.0, 0.0, 0.0, 1.0],
            "a[24..28] must carry absorption_color (0,0,0) + absorption_depth \
             1.0 — the SDF window reference's sample_glass arguments"
        );
        assert_eq!(packed[73].to_bits(), 0, "a[73] thin_walled = 0 (refractive)");
        let packed_thin =
            super::pack_vulkan_mesh_material(PbrMaterial { thin_walled: true, ..glass_mat });
        assert_eq!(
            packed_thin[73].to_bits(),
            1,
            "thin_walled = true must land in a[73] (the MaterialData thin_walled slot)"
        );
        let packed_opaque = super::pack_vulkan_mesh_material(PbrMaterial::default());
        assert_eq!(
            packed_opaque[0].to_bits(),
            1,
            "transmission = 0 must still pack MAT_LAMBERT (1)"
        );
        assert_eq!(
            packed_opaque[10], 1.5,
            "default ior must reproduce the old hardcoded 1.5 byte-for-byte"
        );
        assert_eq!(
            &packed_opaque[24..28],
            &[0.0; 4],
            "opaque materials must leave the absorption slots zeroed (historical)"
        );
        eprintln!(
            "[glass] packer: type {} in a[0], ior {} in a[10], absorption \
             [{},{},{}]/{} in a[24..28], thin_walled {} in a[73]",
            packed[0].to_bits(),
            packed[10],
            packed[24],
            packed[25],
            packed[26],
            packed[27],
            packed[73].to_bits()
        );

        // --- The trivial scene: emissive checker wall + glass slab. ---------
        fn push_quad(
            mesh: &mut CookedMesh,
            corners: [[f32; 3]; 4],
            normal: [f32; 3],
            mat: u8,
        ) {
            let v0 = mesh.positions.len() as u32;
            for c in corners {
                mesh.positions.push(c);
                mesh.normals.push(normal);
                mesh.uvs.push([0.0, 0.0]);
            }
            mesh.indices.push([v0, v0 + 1, v0 + 2]);
            mesh.indices.push([v0, v0 + 2, v0 + 3]);
            mesh.material_ids.push(mat);
            mesh.material_ids.push(mat);
        }
        const CELL: f32 = 0.5; // checker cell size (m)
        const HALF: f32 = 4.0; // wall half-extent (m) -> 16x16 cells
        let mut scene_mesh = CookedMesh {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            material_ids: Vec::new(),
        };
        let ncell = (2.0 * HALF / CELL) as i32;
        for cy in 0..ncell {
            for cx in 0..ncell {
                let x0 = -HALF + cx as f32 * CELL;
                let y0 = -HALF + cy as f32 * CELL;
                push_quad(
                    &mut scene_mesh,
                    [
                        [x0, y0, 0.0],
                        [x0 + CELL, y0, 0.0],
                        [x0 + CELL, y0 + CELL, 0.0],
                        [x0, y0 + CELL, 0.0],
                    ],
                    [0.0, 0.0, 1.0],
                    ((cx + cy) % 2) as u8, // 0 = white EMITTER, 1 = black
                );
            }
        }
        // Two-faced glass slab: the craftsman panes' topology. Front face
        // normal +Z (toward camera), back face normal -Z (outward of slab).
        let (gh, zf, zb) = (0.9f32, 2.0f32, 1.96f32);
        const GLASS_ID: u8 = 2;
        push_quad(
            &mut scene_mesh,
            [[-gh, -gh, zf], [gh, -gh, zf], [gh, gh, zf], [-gh, gh, zf]],
            [0.0, 0.0, 1.0],
            GLASS_ID,
        );
        push_quad(
            &mut scene_mesh,
            [[-gh, -gh, zb], [-gh, gh, zb], [gh, gh, zb], [gh, -gh, zb]],
            [0.0, 0.0, -1.0],
            GLASS_ID,
        );
        let materials = [
            // White checker cell: the scene's ONLY light source. emission
            // packs base_color * strength (a[20..24]).
            PbrMaterial {
                base_color: [1.0, 1.0, 1.0],
                emission_strength: 3.0,
                ..Default::default()
            },
            // Black checker cell: near-zero albedo, no emission.
            PbrMaterial {
                base_color: [0.02, 0.02, 0.02],
                ..Default::default()
            },
            glass_mat,
        ];

        // --- Camera straight through the slab; lights all OFF. --------------
        let (w, h) = (256u32, 256u32);
        let fov_y = std::f32::consts::FRAC_PI_4;
        let eye = [0.0f32, 0.0, 6.0];
        let target = [0.0f32, 0.0, 0.0];
        let rig = LightRig {
            sun_dir: [0.0, 1.0, 0.0],
            sun_intensity: 0.0,
            sky_intensity: 0.0,
            camera_fill: 0.0,
            rim_fill: 0.0,
            sky_dome_intensity: 0.0,
            ..Default::default()
        };
        let spp = 32u32;

        let render = |mats: &[PbrMaterial]| -> (Vec<u8>, f64) {
            let t0 = std::time::Instant::now();
            let rgba = pathtrace_mesh_lit_to_rgba(
                &scene_mesh.positions,
                &scene_mesh.normals,
                &scene_mesh.uvs,
                &scene_mesh.indices,
                &scene_mesh.material_ids,
                mats,
                &[],
                eye,
                target,
                fov_y,
                w,
                h,
                spp,
                &rig,
            )
            .expect("checker-through-glass render should succeed");
            (rgba, t0.elapsed().as_secs_f64())
        };
        let (rgba_t, secs_t) = render(&materials);
        eprintln!("[glass] render time {secs_t:.2}s (each frame, expect < 10s)");
        let mut opaque_materials = materials;
        opaque_materials[GLASS_ID as usize].transmission = 0.0;
        let (rgba_o, secs_o) = render(&opaque_materials);
        eprintln!("[glass] render time {secs_o:.2}s (each frame, expect < 10s)");

        let out_dir = std::env::temp_dir();
        let force_alpha = |mut v: Vec<u8>| -> Vec<u8> {
            for px in v.chunks_exact_mut(4) {
                px[3] = 255;
            }
            v
        };
        let t_png = out_dir.join("mesh_glass_checker.png");
        let o_png = out_dir.join("mesh_glass_checker_opaque.png");
        write_png_rgba(t_png.to_str().unwrap(), &force_alpha(rgba_t.clone()), w, h);
        write_png_rgba(o_png.to_str().unwrap(), &force_alpha(rgba_o.clone()), w, h);
        eprintln!("[glass] wrote {} (transmissive)", t_png.display());
        eprintln!("[glass] wrote {} (opaque control)", o_png.display());

        // --- Per-pixel ground truth: the same pinhole as the GPU (proven by
        // the M0 projection cross-check), inverted to a camera ray, hit the
        // wall plane z=0, read the checker parity. TWO optical models, both
        // "the pattern behind the glass, transmitted":
        //   (straight) ideal slab physics — enter+exit refractions cancel,
        //       the ray continues straight (lateral offset sub-pixel);
        //   (refracted) the engine's CURRENT model — the megakernel flips
        //       shading normals toward the ray and `dispatch_sample` passes
        //       `mat.ior` (not the IOR-stack eta), so BOTH slab faces refract
        //       as "entering" with eta = 1/ior: the slab is a weak lens that
        //       MAGNIFIES the pattern (visible in the PNG).
        // The gate scores against the better-matching model so it certifies
        // transmission, not one refraction convention. Pixels whose first hit
        // is the glass slab (CPU rasterizer, eroded 1px) are the glass
        // region; pixels whose first hit is the wall itself are the
        // direct-view sanity region. Pixels within 0.04 m of a predicted
        // checker boundary are excluded (AA tolerance). ----------------------
        let (mats_px, _tris_px) = rasterize_material_masks(&scene_mesh, eye, target, fov_y, w, h);
        let eye_v = glam::Vec3::from(eye);
        let fwd = (glam::Vec3::from(target) - eye_v).normalize();
        let right = fwd.cross(glam::Vec3::Y).normalize();
        let up = right.cross(fwd);
        let tan_half = (fov_y * 0.5).tan();
        let aspect = w as f32 / h as f32;
        // Snell refraction of unit `d` through a plane with normal +Z,
        // ratio eta = n_i/n_t (the sample_glass convention).
        let refract_z = |d: glam::Vec3, eta: f32| -> glam::Vec3 {
            let n = glam::Vec3::Z;
            let cos_i = -n.dot(d); // d points INTO the surface (d.z < 0)
            let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
            debug_assert!(sin2_t < 1.0, "no TIR at these angles");
            let cos_t = (1.0 - sin2_t).sqrt();
            (d * eta + n * (eta * cos_i - cos_t)).normalize()
        };
        // Checker parity at the wall-plane point, None if outside the wall or
        // within 0.04 m of a cell boundary.
        let parity_at = |wx: f32, wy: f32| -> Option<f64> {
            if wx <= -HALF || wx >= HALF || wy <= -HALF || wy >= HALF {
                return None;
            }
            let fx = (wx + HALF) / CELL;
            let fy = (wy + HALF) / CELL;
            let frx = fx - fx.floor();
            let fry = fy - fy.floor();
            if frx.min(1.0 - frx).min(fry).min(1.0 - fry) * CELL < 0.04 {
                return None;
            }
            let white = ((fx.floor() as i32 + fy.floor() as i32) % 2) == 0;
            Some(if white { 1.0 } else { 0.0 })
        };
        // (pixel, expected) per model.
        let mut glass_straight: Vec<(usize, f64)> = Vec::new();
        let mut glass_refracted: Vec<(usize, f64)> = Vec::new();
        let mut wall_sel: Vec<(usize, f64)> = Vec::new();
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let i = (y * w + x) as usize;
                let ndc_x = aspect * ((x as f32 + 0.5) * 2.0 / w as f32 - 1.0);
                let ndc_y = 1.0 - (y as f32 + 0.5) * 2.0 / h as f32;
                let dir = (fwd + right * (ndc_x * tan_half) + up * (ndc_y * tan_half))
                    .normalize();
                if dir.z >= -1e-6 {
                    continue;
                }
                let m = mats_px[i];
                let eroded = |id: i32| {
                    mats_px[i - 1] == id
                        && mats_px[i + 1] == id
                        && mats_px[i - w as usize] == id
                        && mats_px[i + w as usize] == id
                };
                let at_plane = |origin: glam::Vec3, d: glam::Vec3, z: f32| -> glam::Vec3 {
                    origin + d * ((z - origin.z) / d.z)
                };
                if m == GLASS_ID as i32 && eroded(GLASS_ID as i32) {
                    // Straight-through model.
                    let p = at_plane(eye_v, dir, 0.0);
                    if let Some(e) = parity_at(p.x, p.y) {
                        glass_straight.push((i, e));
                    }
                    // Double-"entering" refraction model (the engine's).
                    let eta = 1.0 / 1.5;
                    let p1 = at_plane(eye_v, dir, zf);
                    let d1 = refract_z(dir, eta);
                    let p2 = at_plane(p1, d1, zb);
                    let d2 = refract_z(d1, eta);
                    let p3 = at_plane(p2, d2, 0.0);
                    if let Some(e) = parity_at(p3.x, p3.y) {
                        glass_refracted.push((i, e));
                    }
                } else if (m == 0 || m == 1) && eroded(m) {
                    let p = at_plane(eye_v, dir, 0.0);
                    if let Some(e) = parity_at(p.x, p.y) {
                        wall_sel.push((i, e));
                    }
                }
            }
        }
        eprintln!(
            "[glass] regions: {} glass px (straight model) / {} (refracted \
             model), {} direct-wall px (eroded, boundary-excluded)",
            glass_straight.len(),
            glass_refracted.len(),
            wall_sel.len()
        );
        assert!(
            glass_straight.len() >= 2000 && glass_refracted.len() >= 2000,
            "glass region too small ({} / {} px) — slab not covering the frame center",
            glass_straight.len(),
            glass_refracted.len()
        );
        assert!(
            wall_sel.len() >= 2000,
            "direct-wall region too small ({} px)",
            wall_sel.len()
        );

        // Pearson correlation: measured luminance vs the known checker parity.
        let pearson = |img: &[u8], sel: &[(usize, f64)]| -> f64 {
            let n = sel.len() as f64;
            let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for &(i, xv) in sel {
                let yv = 0.2126 * img[i * 4] as f64
                    + 0.7152 * img[i * 4 + 1] as f64
                    + 0.0722 * img[i * 4 + 2] as f64;
                sx += xv;
                sy += yv;
                sxx += xv * xv;
                syy += yv * yv;
                sxy += xv * yv;
            }
            let cov = sxy / n - (sx / n) * (sy / n);
            let vx = sxx / n - (sx / n) * (sx / n);
            let vy = syy / n - (sy / n) * (sy / n);
            if vx <= 1e-12 || vy <= 1e-9 {
                return 0.0; // constant image: no pattern at all
            }
            cov / (vx.sqrt() * vy.sqrt())
        };

        // Sanity: the emissive wall itself renders the pattern in BOTH
        // frames — failures below are then unambiguously the glass route.
        let rw_t = pearson(&rgba_t, &wall_sel);
        let rw_o = pearson(&rgba_o, &wall_sel);
        eprintln!(
            "[glass] direct-wall sanity: r={rw_t:.3} (transmissive frame), \
             r={rw_o:.3} (opaque frame) (gate both > 0.8)"
        );
        assert!(
            rw_t > 0.8 && rw_o > 0.8,
            "the emissive checker wall itself does not render (direct-view \
             r {rw_t:.3}/{rw_o:.3} <= 0.8) — scene/emission problem, not glass"
        );

        // --- THE gate: the pattern BEHIND the glass shows through ONLY when
        // the material transmits. r = the better-matching optical model. -----
        let model_r = |img: &[u8]| -> (f64, f64) {
            (pearson(img, &glass_straight), pearson(img, &glass_refracted))
        };
        let (rt_s, rt_r) = model_r(&rgba_t);
        let (ro_s, ro_r) = model_r(&rgba_o);
        let rt = rt_s.max(rt_r);
        let ro = ro_s.max(ro_r);
        eprintln!(
            "[glass] model correlations: transmissive straight={rt_s:.3} \
             refracted={rt_r:.3}; opaque straight={ro_s:.3} refracted={ro_r:.3}"
        );
        let pass = rt > 0.5 && rt >= 3.0 * ro;
        eprintln!(
            "[glass] checker-through-glass: transmissive correlation r={rt:.3} vs \
             opaque r={ro:.3} (gate rt > 0.5 and rt >= 3x ro) -> {}",
            if pass { "PASS" } else { "FAIL" }
        );
        assert!(
            pass,
            "checkerboard behind the glass does not show through (transmissive \
             r {rt:.3}, opaque control r {ro:.3}; need rt > 0.5 and rt >= 3x ro) \
             — check the MAT_GLASS slot layout in pack_vulkan_mesh_material \
             (type a[0], ior a[10], absorption a[24..28], thin_walled a[73]) \
             against material_types.slang's Vulkan reflection layout"
        );

        // --- ONE craftsman demo render (eyeball-only, no gate): the cooked
        // glass channel flipped transmissive — the same MAT_GLASS parameters
        // the gate above just proved. The single allowed slow render. --------
        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let asset_path = atoms_dir.join("forge.house.craftsman.atoms.json");
        let mesh = load_craftsman_mesh(&asset_path);
        let (mut cmats, ctex, channels) = load_building_mesh_pbr_by_forge_id(&asset_path);
        let glass_id = channels
            .iter()
            .position(|c| c == "glass")
            .expect("cooked payload has no 'glass' channel");
        cmats[glass_id].transmission = 1.0;
        cmats[glass_id].ior = 1.5;
        cmats[glass_id].thin_walled = false;
        // Frontal-low framing: the porch parapet's glazing and the facade
        // windows fill the view, sky and porch cavity behind them.
        let (cw, ch) = (512u32, 512u32);
        let c_target = [0.0f32, 2.6, 0.0];
        let c_eye = [0.5f32, 2.4, 14.0];
        let c_rig = LightRig {
            sun_dir: [0.45, 0.65, 0.55],
            sun_intensity: 2.6,
            sky_intensity: 0.3,
            camera_fill: 0.2,
            rim_fill: 0.0,
            sky_dome_intensity: 0.65,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let rgba_demo = pathtrace_mesh_lit_to_rgba(
            &mesh.positions,
            &mesh.normals,
            &mesh.uvs,
            &mesh.indices,
            &mesh.material_ids,
            &cmats,
            &ctex,
            c_eye,
            c_target,
            fov_y,
            cw,
            ch,
            128,
            &c_rig,
        )
        .expect("craftsman glass demo render should succeed");
        let demo_png = out_dir.join("mesh_craftsman_glass.png");
        write_png_rgba(demo_png.to_str().unwrap(), &force_alpha(rgba_demo), cw, ch);
        eprintln!(
            "[glass] wrote {} (craftsman demo, glass channel transmissive, \
             {:.2}s — the one allowed slow render)",
            demo_png.display(),
            t0.elapsed().as_secs_f64()
        );
    }


    /// CURTAIN-WALL FACADE GATE (forge facade-axis wave: the glass office).
    /// Loads the COOKED curtain-wall office tower
    /// (`city.office.l5.3x3.glass_office_tower_01`, cooked with
    /// `forge.facade_system: "curtain_wall"` into
    /// `assets/buildings/curtain_wall`) and verifies the facade family by
    /// GEOMETRY and by a CHEAP render (256² @ 32 spp — seconds):
    ///
    ///   (grid) mullion/transom counts measured as welded connected
    ///       components of the trim material, cells = Σ_walls
    ///       (mullions−1)·(transoms−1) — must equal the parametric
    ///       expectation bays·floors·3 derived from the payload's own
    ///       forge_description (width/depth/floors/floor_height) and the
    ///       forge CurtainWallParams default bay width (1.8 m);
    ///   (glass) vision panes counted from MAT_GLASS triangles == 2·bays·
    ///       floors, AND the cooked glass material arrives transmissive
    ///       (transmission > 0, thin_walled) and packs MAT_GLASS(3) —
    ///       spandrels counted as wall-plane MAT_WALL panels == bays·floors,
    ///       their material opaque (packs MAT_LAMBERT(1));
    ///   (energy) facade detail energy — masked Sobel gradient energy of the
    ///       rendered tower vs a flat Lambert box of the same dimensions,
    ///       same camera/rig/spp — ratio >= 3.0;
    ///   (time) each gate render < 10 s.
    /// Writes `curtain_wall_office.png` + `curtain_wall_flatbox.png`.
    ///
    /// Cook first (civitas repo, ~/Ochroma/projects/civitas_care):
    ///   mkdir -p /tmp/curtain_src/office && cp \
    ///     assets/source/buildings/office/glass_office_tower_01.asset.json \
    ///     /tmp/curtain_src/office/ && mkdir -p assets/buildings/curtain_wall/textures \
    ///     && cp -r assets/buildings/forge_starter/textures/polyhaven \
    ///     assets/buildings/curtain_wall/textures/
    ///   cargo build --release --bin game_asset_cook && \
    ///   GAME_FORGE_BIN=$HOME/src/forge/target/release/aetherspectra-forge \
    ///     ./target/release/game_asset_cook --no-starters \
    ///     --source /tmp/curtain_src --output assets/buildings/curtain_wall
    /// Run alone (GPU):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --profile release-fast --lib forge_curtain_wall_office -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn forge_curtain_wall_office() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig, PbrMaterial};

        let atoms_path = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(
            "Ochroma/projects/civitas_care/assets/buildings/curtain_wall/atoms/\
             city.office.l5.3x3.glass_office_tower_01.atoms.json",
        );
        let mesh = load_craftsman_mesh(&atoms_path);
        let (materials, textures, channels) = load_building_mesh_pbr_by_forge_id(&atoms_path);
        assert_eq!(channels[2], "glass", "forge id 2 must bind the glass channel");

        // ── The parametric expectation, from the payload's own cooked
        //    forge_description — never hardcoded counts. ─────────────────────
        let json: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&atoms_path)
                .unwrap_or_else(|e| panic!("read {}: {e}", atoms_path.display())),
        )
        .expect("parse atoms.json");
        let desc = &json["forge_description"];
        assert_eq!(
            desc["facade_system"].as_str(),
            Some("curtain_wall"),
            "payload was not cooked with facade_system curtain_wall"
        );
        let width = desc["footprint"]["width"].as_f64().unwrap() as f32;
        let depth = desc["footprint"]["depth"].as_f64().unwrap() as f32;
        let floors = desc["floors"].as_u64().unwrap() as usize;
        let fh = desc["floor_height"].as_f64().unwrap() as f32;
        // Forge CurtainWallParams::default().bay_width — the one cross-repo
        // parametric constant (mullion spacing target).
        const BAY_WIDTH: f32 = 1.8;
        let bays_w = (width / BAY_WIDTH).round().max(1.0) as usize;
        let bays_d = (depth / BAY_WIDTH).round().max(1.0) as usize;
        let bays_total = 2 * (bays_w + bays_d);
        let exp_mullions = bays_total + 4; // bays_e + 1 per edge
        let exp_transoms = 4 * (3 * floors + 1);
        let exp_cells = 3 * bays_total * floors;
        let exp_glass = 2 * bays_total * floors;
        let exp_spandrels = bays_total * floors;
        let total_h = floors as f32 * fh;
        let (half_w, half_d) = (width * 0.5, depth * 0.5);

        // ── GATE 1: the mullion/transom grid, measured from welded
        //    components of the trim material (forge id 4). ───────────────────
        let trim = mesh_material_components(&mesh, 4);
        // Per wall: components beyond each footprint plane (members stand
        // proud of the skin, so their centroids sit outside the plane).
        let mut per_wall: [(usize, usize); 4] = [(0, 0); 4]; // (mullions, transoms)
        for (mn, mx) in &trim {
            let c = [(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5];
            let wall = if c[0] > half_w {
                0
            } else if c[0] < -half_w {
                1
            } else if c[2] > half_d {
                2
            } else if c[2] < -half_d {
                3
            } else {
                panic!("trim component centroid {c:?} is not on any facade plane");
            };
            if mx[1] - mn[1] > 1.0 {
                per_wall[wall].0 += 1; // full-height vertical = mullion
            } else {
                per_wall[wall].1 += 1; // thin horizontal run = transom
            }
        }
        let mullions: usize = per_wall.iter().map(|w| w.0).sum();
        let transoms: usize = per_wall.iter().map(|w| w.1).sum();
        let cells: usize = per_wall
            .iter()
            .map(|&(nv, nt)| nv.saturating_sub(1) * nt.saturating_sub(1))
            .sum();
        let grid_pass = mullions == exp_mullions && transoms == exp_transoms && cells == exp_cells;
        eprintln!(
            "[curtain] grid: {mullions} mullions x {transoms} transoms = {cells} cells \
             (expect {exp_cells} == bays*floors*3: {bays_total} bays x {floors} floors x \
             3 panes_per) -> {}",
            if grid_pass { "PASS" } else { "FAIL" }
        );
        assert!(
            grid_pass,
            "curtain-wall grid is not the parametric expectation: \
             {mullions}/{exp_mullions} mullions, {transoms}/{exp_transoms} transoms, \
             {cells}/{exp_cells} cells (per-wall {per_wall:?})"
        );

        // ── GATE 2: vision glass transmissive + spandrels opaque. ────────────
        let glass_tris = mesh.material_ids.iter().filter(|&&m| m == 2).count();
        let glass_panels = glass_tris / 2;
        let glass_mat = materials[2];
        let packed_glass = super::pack_vulkan_mesh_material(glass_mat);
        let glass_pass = glass_panels == exp_glass
            && glass_mat.transmission > 0.0
            && glass_mat.thin_walled
            && packed_glass[0].to_bits() == 3;
        eprintln!(
            "[curtain] vision glass panels transmissive: {glass_panels} panels, MAT_GLASS -> {}",
            if glass_pass { "PASS" } else { "FAIL" }
        );
        assert!(
            glass_pass,
            "vision glass is not cooked transmissive: {glass_panels}/{exp_glass} panels, \
             transmission {}, thin_walled {}, packed type {} (the cook must tag the \
             curtain-wall glass channel transmission > 0)",
            glass_mat.transmission,
            glass_mat.thin_walled,
            packed_glass[0].to_bits()
        );
        // Spandrels: MAT_WALL (forge id 0) panels lying IN a facade plane
        // (every vertex within 2 cm of one wall plane, above grade) — the
        // underside seal and interior proxies live elsewhere.
        let mut spandrel_tris = 0usize;
        for (t, tri) in mesh.indices.iter().enumerate() {
            if mesh.material_ids[t] != 0 {
                continue;
            }
            let pts: Vec<[f32; 3]> = tri.iter().map(|&i| mesh.positions[i as usize]).collect();
            let in_plane = |f: &dyn Fn(&[f32; 3]) -> f32| pts.iter().all(|p| f(p).abs() < 0.02);
            let on_facade = in_plane(&|p: &[f32; 3]| p[0] - half_w)
                || in_plane(&|p: &[f32; 3]| p[0] + half_w)
                || in_plane(&|p: &[f32; 3]| p[2] - half_d)
                || in_plane(&|p: &[f32; 3]| p[2] + half_d);
            let above_grade = pts.iter().map(|p| p[1]).fold(f32::MIN, f32::max) > 0.01;
            if on_facade && above_grade {
                spandrel_tris += 1;
            }
        }
        let spandrels = spandrel_tris / 2;
        let wall_mat = materials[0];
        let packed_wall = super::pack_vulkan_mesh_material(wall_mat);
        let spandrel_pass = spandrels == exp_spandrels
            && wall_mat.transmission == 0.0
            && packed_wall[0].to_bits() == 1;
        eprintln!(
            "[curtain] spandrel panels opaque: {spandrels} -> {}",
            if spandrel_pass { "PASS" } else { "FAIL" }
        );
        assert!(
            spandrel_pass,
            "spandrels wrong: {spandrels}/{exp_spandrels} panels, wall transmission {}, \
             packed type {}",
            wall_mat.transmission,
            packed_wall[0].to_bits()
        );

        // ── GATE 3: facade detail energy vs a flat Lambert box of the same
        //    dimensions (same camera, rig, resolution, spp). CHEAP renders. ──
        let (w, h) = (256u32, 256u32);
        let spp = 32u32;
        let fov_y = std::f32::consts::FRAC_PI_4;
        let eye = [26.0f32, 16.0, 34.0];
        let target = [0.0f32, total_h * 0.45, 0.0];
        let rig = LightRig {
            sun_dir: [0.45, 0.65, 0.55],
            sun_intensity: 2.6,
            sky_intensity: 0.3,
            camera_fill: 0.2,
            rim_fill: 0.0,
            sky_dome_intensity: 0.65,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };

        let t0 = std::time::Instant::now();
        let rgba_tower = pathtrace_mesh_lit_to_rgba(
            &mesh.positions,
            &mesh.normals,
            &mesh.uvs,
            &mesh.indices,
            &mesh.material_ids,
            &materials,
            &textures,
            eye,
            target,
            fov_y,
            w,
            h,
            spp,
            &rig,
        )
        .expect("curtain-wall tower render should succeed");
        let secs_tower = t0.elapsed().as_secs_f64();

        // Flat-box control: the same envelope as ONE Lambert box (the cooked
        // facade albedo, no textures, no grid, no glass).
        let mut box_mesh = CookedMesh {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            material_ids: Vec::new(),
        };
        let mut push_quad = |corners: [[f32; 3]; 4], normal: [f32; 3]| {
            let v0 = box_mesh.positions.len() as u32;
            for c in corners {
                box_mesh.positions.push(c);
                box_mesh.normals.push(normal);
                box_mesh.uvs.push([0.0, 0.0]);
            }
            box_mesh.indices.push([v0, v0 + 1, v0 + 2]);
            box_mesh.indices.push([v0, v0 + 2, v0 + 3]);
            box_mesh.material_ids.extend_from_slice(&[0, 0]);
        };
        let (hw, hd, th) = (half_w, half_d, total_h);
        push_quad(
            [[-hw, 0.0, hd], [hw, 0.0, hd], [hw, th, hd], [-hw, th, hd]],
            [0.0, 0.0, 1.0],
        );
        push_quad(
            [[hw, 0.0, -hd], [-hw, 0.0, -hd], [-hw, th, -hd], [hw, th, -hd]],
            [0.0, 0.0, -1.0],
        );
        push_quad(
            [[hw, 0.0, hd], [hw, 0.0, -hd], [hw, th, -hd], [hw, th, hd]],
            [1.0, 0.0, 0.0],
        );
        push_quad(
            [[-hw, 0.0, -hd], [-hw, 0.0, hd], [-hw, th, hd], [-hw, th, -hd]],
            [-1.0, 0.0, 0.0],
        );
        push_quad(
            [[-hw, th, hd], [hw, th, hd], [hw, th, -hd], [-hw, th, -hd]],
            [0.0, 1.0, 0.0],
        );
        let box_materials = [PbrMaterial {
            base_color: wall_mat.base_color,
            roughness: wall_mat.roughness,
            ..Default::default()
        }];
        let t1 = std::time::Instant::now();
        let rgba_box = pathtrace_mesh_lit_to_rgba(
            &box_mesh.positions,
            &box_mesh.normals,
            &box_mesh.uvs,
            &box_mesh.indices,
            &box_mesh.material_ids,
            &box_materials,
            &[],
            eye,
            target,
            fov_y,
            w,
            h,
            spp,
            &rig,
        )
        .expect("flat-box control render should succeed");
        let secs_box = t1.elapsed().as_secs_f64();

        let coverage_mask = |m: &CookedMesh| -> (Vec<bool>, usize) {
            let (mats_px, _) = rasterize_material_masks(m, eye, target, fov_y, w, h);
            let mask: Vec<bool> = mats_px.iter().map(|&v| v >= 0).collect();
            let n = mask.iter().filter(|&&b| b).count();
            (mask, n)
        };
        let (tower_mask, tower_px) = coverage_mask(&mesh);
        let (box_mask, box_px) = coverage_mask(&box_mesh);
        assert!(
            tower_px > 5000 && box_px > 5000,
            "camera framing drifted: tower {tower_px} px, box {box_px} px coverage"
        );
        let e_tower = masked_sobel_energy(&rgba_tower, &tower_mask, w, h);
        let e_box = masked_sobel_energy(&rgba_box, &box_mask, w, h);
        let energy_ratio = e_tower / e_box.max(1e-9);
        let energy_pass = energy_ratio >= 3.0;
        eprintln!(
            "[curtain] facade detail energy {energy_ratio:.2}x a flat-box control \
             (gate >= 3.0x) -> {} (tower sobel {e_tower:.3}, box sobel {e_box:.3}, \
             {tower_px}/{box_px} px)",
            if energy_pass { "PASS" } else { "FAIL" }
        );
        eprintln!(
            "[curtain] render time {secs_tower:.2}s per frame (expect < 10s) \
             [control {secs_box:.2}s]"
        );

        let out_dir = std::env::temp_dir();
        let force_alpha = |mut v: Vec<u8>| -> Vec<u8> {
            for px in v.chunks_exact_mut(4) {
                px[3] = 255;
            }
            v
        };
        let tower_png = out_dir.join("curtain_wall_office.png");
        let box_png = out_dir.join("curtain_wall_flatbox.png");
        write_png_rgba(tower_png.to_str().unwrap(), &force_alpha(rgba_tower), w, h);
        write_png_rgba(box_png.to_str().unwrap(), &force_alpha(rgba_box), w, h);
        eprintln!("[curtain] wrote {} (curtain-wall office)", tower_png.display());
        eprintln!("[curtain] wrote {} (flat-box control)", box_png.display());
        eprintln!(
            "[curtain] eyeball: a glass office tower — vertical mullions + floor-line/\
             mid-floor transoms gridding every facade, opaque spandrel bands at the \
             slabs, transmissive vision glass showing the interior plates — vs a \
             featureless Lambert box"
        );

        assert!(
            energy_pass,
            "the curtain-wall facade does not read as structured vs a plain box \
             (sobel ratio {energy_ratio:.2}x < 3.0x)"
        );
        assert!(
            secs_tower.min(secs_box) < 10.0,
            "gate renders are not cheap: {secs_tower:.2}s / {secs_box:.2}s per frame"
        );
    }


    /// MASSING-AXIS GATES (grammar §4.2 wave): podium+tower + setback +
    /// box, all loaded from COOKED payloads (`assets/buildings/massing`,
    /// cooked by game_asset_cook through the massing-aware forge) and
    /// measured from geometry, then rendered CHEAP (256² @ 32 spp).
    ///
    ///   (volumes) podium_tower stacks exactly 2 skin-width plateaus —
    ///       podium floors / tower floors / inset / total height measured
    ///       from the mesh against the payload's own forge_description;
    ///   (watertight) podium_tower + setback + box all pass the cook's GWN
    ///       64-probe closed gate (bit-for-bit replica), and the box payload
    ///       mesh is BYTE-IDENTICAL to the pre-massing cook of the same
    ///       directive (assets/buildings/curtain_wall, cooked on master);
    ///   (facade) the tower band is glass-dominant curtain wall with the
    ///       cooked glass material transmissive (packs MAT_GLASS=3); the
    ///       podium band is wall-dominant punched window;
    ///   (shape) pairwise fitted-silhouette deltas between podium_tower,
    ///       setback and box all exceed 8% — the axis spans 3 distinct
    ///       shapes;
    ///   (render) three PNGs, each frame < 10 s.
    ///
    /// Cook first (civitas repo, ~/Ochroma/projects/civitas_care):
    ///   mkdir -p /tmp/massing_src/office && cp assets/source/buildings/office/\
    ///     {podium_tower_01,setback_tower_01,glass_office_tower_01}.asset.json \
    ///     /tmp/massing_src/office/ && mkdir -p assets/buildings/massing/textures \
    ///     && cp -r assets/buildings/forge_starter/textures/polyhaven \
    ///     assets/buildings/massing/textures/
    ///   cargo build --release --bin game_asset_cook && \
    ///   GAME_FORGE_BIN=$HOME/src/forge/target/release/aetherspectra-forge \
    ///     ./target/release/game_asset_cook --no-starters \
    ///     --source /tmp/massing_src --output assets/buildings/massing
    /// Run alone (GPU):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --profile release-fast --lib forge_massing_podium_tower -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn forge_massing_podium_tower() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig};

        let home = std::path::PathBuf::from(std::env::var("HOME").unwrap());
        let massing_atoms = home.join("Ochroma/projects/civitas_care/assets/buildings/massing/atoms");
        let pt_path = massing_atoms.join("city.office.l8.4x4.podium_tower_01.atoms.json");
        let sb_path = massing_atoms.join("city.office.l6.3x3.setback_tower_01.atoms.json");
        let box_path = massing_atoms.join("city.office.l5.3x3.glass_office_tower_01.atoms.json");
        let box_pre_path = home.join(
            "Ochroma/projects/civitas_care/assets/buildings/curtain_wall/atoms/\
             city.office.l5.3x3.glass_office_tower_01.atoms.json",
        );

        let pt = load_craftsman_mesh(&pt_path);
        let sb = load_craftsman_mesh(&sb_path);
        let bx = load_craftsman_mesh(&box_path);
        let bx_pre = load_craftsman_mesh(&box_pre_path);

        // ── GATE 1: podium_tower volumes / floors / inset / height, measured
        //    from the cooked mesh against its own forge_description. ─────────
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&pt_path).unwrap()).expect("parse atoms.json");
        let desc = &json["forge_description"];
        assert_eq!(desc["massing_mode"].as_str(), Some("podium_tower"));
        let pf = desc["podium_floors"].as_u64().unwrap() as u32;
        let tf = desc["tower_floors"].as_u64().unwrap() as u32;
        let fh = desc["floor_height"].as_f64().unwrap() as f32;
        let plot_w = desc["footprint"]["width"].as_f64().unwrap() as f32;
        let floors = desc["floors"].as_u64().unwrap() as u32;
        assert_eq!(pf + tf, floors, "floor split must tile the height");

        let slices = floors as usize;
        let (pt_widths, pt_height) = skin_width_profile(&pt, slices);
        let pt_volumes = count_width_plateaus(&pt_widths);
        let podium_w: f32 = pt_widths[..pf as usize]
            .iter()
            .fold(0.0f32, |m, &w| m.max(w));
        // Tower width away from the seam slice.
        let tower_w: f32 = pt_widths[(pf as usize + 1)..]
            .iter()
            .filter(|w| **w > 0.0)
            .fold(0.0f32, |m, &w| m.max(w));
        let inset_m = (podium_w - tower_w) * 0.5;
        let inset_pct = 100.0 * inset_m / podium_w;
        let gate1 = pt_volumes == 2
            && inset_m > 0.5
            && tf > pf
            && (podium_w - plot_w).abs() < 0.1
            && pt_height >= floors as f32 * fh
            && pt_height < floors as f32 * fh + 1.0;
        eprintln!(
            "[massing] podium_tower volumes: {pt_volumes} (podium {pf} floors, tower {tf} \
             floors); tower footprint inset = {inset_pct:.1}% of podium ({inset_m:.2} m, \
             podium {podium_w:.2} m -> tower {tower_w:.2} m) (expect inset>0, tf>pf, total \
             height {pt_height:.1}m) -> {}",
            if gate1 { "PASS" } else { "FAIL" }
        );
        assert!(
            gate1,
            "podium_tower massing wrong: {pt_volumes} volumes, inset {inset_m} m, \
             pf {pf} tf {tf}, height {pt_height} vs {} floors x {fh} m",
            floors
        );

        // ── GATE 2: watertight — the cook's GWN closed gate on all three,
        //    plus box byte-identity against the pre-massing cook. ────────────
        let gwn_label = |r: &Result<usize, f32>| match r {
            Ok(n) => format!("gwn yes ({n} interior probes)"),
            Err(w) => format!("NO (fractional |w|={w:.3})"),
        };
        let pt_gate = cooked_probe_gate(&pt);
        let sb_gate = cooked_probe_gate(&sb);
        let bx_gate = cooked_probe_gate(&bx);
        let identical = pt_eq_mesh(&bx, &bx_pre);
        let gate2 =
            pt_gate.is_ok() && sb_gate.is_ok() && bx_gate.is_ok() && identical;
        eprintln!(
            "[massing] watertight: podium_tower closed={}, setback closed={}, box-regression \
             closed={} & byte-identical to pre-massing ({} tris) = {identical} -> {}",
            gwn_label(&pt_gate),
            gwn_label(&sb_gate),
            gwn_label(&bx_gate),
            bx.indices.len(),
            if gate2 { "PASS" } else { "FAIL" }
        );
        assert!(gate2, "watertight/regression gate failed");

        // ── GATE 3: per-volume facade — transmissive curtain-wall tower over
        //    a punched podium, measured from band areas + the packed
        //    material. ─────────────────────────────────────────────────────────
        let (materials, textures, channels) = load_building_mesh_pbr_by_forge_id(&pt_path);
        assert_eq!(channels[2], "glass", "forge id 2 must bind the glass channel");
        let podium_top = pf as f32 * fh;
        let total_h = floors as f32 * fh;
        let band_area = |y0: f32, y1: f32| -> (f64, f64) {
            use glam::Vec3;
            let (mut glass, mut wall) = (0.0f64, 0.0f64);
            for (t, tri) in pt.indices.iter().enumerate() {
                let m = pt.material_ids[t];
                if m != 0 && m != 2 {
                    continue;
                }
                let p0 = Vec3::from(pt.positions[tri[0] as usize]);
                let p1 = Vec3::from(pt.positions[tri[1] as usize]);
                let p2 = Vec3::from(pt.positions[tri[2] as usize]);
                let g = (p2 - p0).cross(p1 - p0);
                if g.length() < 1e-9 || g.normalize().y.abs() > 0.5 {
                    continue;
                }
                let cy = (p0.y + p1.y + p2.y) / 3.0;
                if cy < y0 || cy > y1 {
                    continue;
                }
                let area = (g.length() * 0.5) as f64;
                if m == 2 {
                    glass += area;
                } else {
                    wall += area;
                }
            }
            (glass, wall)
        };
        let (tower_glass, tower_wall) = band_area(podium_top + 1.0, total_h - 1.0);
        let (podium_glass, podium_wall) = band_area(0.5, podium_top - 0.5);
        let tower_glass_tris = pt
            .indices
            .iter()
            .zip(&pt.material_ids)
            .filter(|(tri, m)| {
                **m == 2 && {
                    let cy = tri
                        .iter()
                        .map(|&i| pt.positions[i as usize][1])
                        .sum::<f32>()
                        / 3.0;
                    cy > podium_top
                }
            })
            .count();
        let glass_mat = materials[2];
        let packed = super::pack_vulkan_mesh_material(glass_mat);
        let gate3 = tower_glass_tris > 0
            && glass_mat.transmission > 0.0
            && glass_mat.thin_walled
            && packed[0].to_bits() == 3
            && tower_glass > tower_wall
            && podium_glass < podium_wall * 0.3;
        eprintln!(
            "[massing] tower facade = curtain_wall (transmissive glass panels \
             {} > 0, MAT_GLASS packed, band glass {tower_glass:.0} m^2 > wall \
             {tower_wall:.0} m^2), podium facade = punched (glass {podium_glass:.0} m^2 << \
             wall {podium_wall:.0} m^2) -> {}",
            tower_glass_tris / 2,
            if gate3 { "PASS" } else { "FAIL" }
        );
        assert!(
            gate3,
            "per-volume facade wrong: tower glass tris {tower_glass_tris}, transmission {}, \
             packed {}, tower {tower_glass}/{tower_wall}, podium {podium_glass}/{podium_wall}",
            glass_mat.transmission,
            packed[0].to_bits()
        );

        // ── GATE 4: the axis spans 3 distinct shapes — pairwise fitted-
        //    silhouette deltas. ───────────────────────────────────────────────
        let res = 160u32;
        let masks = [
            ("podium_tower", fitted_silhouette(&pt, res)),
            ("setback", fitted_silhouette(&sb, res)),
            ("box", fitted_silhouette(&bx, res)),
        ];
        let mut min_delta = f64::INFINITY;
        let mut deltas = Vec::new();
        for i in 0..masks.len() {
            for j in (i + 1)..masks.len() {
                let (xor, union) = masks[i]
                    .1
                    .iter()
                    .zip(&masks[j].1)
                    .fold((0usize, 0usize), |(x, u), (&a, &b)| {
                        (x + (a != b) as usize, u + (a || b) as usize)
                    });
                let delta = xor as f64 / union.max(1) as f64;
                min_delta = min_delta.min(delta);
                deltas.push(format!("{} vs {} = {:.1}%", masks[i].0, masks[j].0, delta * 100.0));
            }
        }
        let gate4 = min_delta > 0.08;
        eprintln!(
            "[massing] shape distinct: pairwise silhouette delta {} (gate > 8%) -> {}",
            deltas.join(", "),
            if gate4 { "PASS" } else { "FAIL" }
        );
        assert!(gate4, "silhouettes are not distinct: min delta {min_delta:.3}");

        // ── GATE 5: cheap renders — a human eyeballs glass-on-a-base, the
        //    ziggurat, and the plain box. ─────────────────────────────────────
        let (w, h) = (256u32, 256u32);
        let spp = 32u32;
        let fov_y = std::f32::consts::FRAC_PI_4;
        let rig = LightRig {
            sun_dir: [0.45, 0.65, 0.55],
            sun_intensity: 2.6,
            sky_intensity: 0.3,
            camera_fill: 0.2,
            rim_fill: 0.0,
            sky_dome_intensity: 0.65,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };
        let out_dir = std::env::temp_dir();
        let mut worst_secs = 0.0f64;
        for (label, mesh, path) in [
            ("massing_podium_tower", &pt, &pt_path),
            ("massing_setback", &sb, &sb_path),
            ("massing_box", &bx, &box_path),
        ] {
            let (mats, texs, _) = load_building_mesh_pbr_by_forge_id(path);
            let max_y = mesh
                .positions
                .iter()
                .map(|p| p[1])
                .fold(f32::NEG_INFINITY, f32::max);
            let max_x = mesh
                .positions
                .iter()
                .map(|p| p[0].abs())
                .fold(0.0f32, f32::max);
            let half = (max_y * 0.5).max(max_x);
            let dist = half / (fov_y * 0.5).tan() * 1.35;
            let dir = glam::Vec3::new(0.8, 0.42, 1.0).normalize();
            let target = [0.0f32, max_y * 0.48, 0.0];
            let eye = [
                target[0] + dir.x * dist,
                target[1] + dir.y * dist,
                target[2] + dir.z * dist,
            ];
            let t0 = std::time::Instant::now();
            let rgba = pathtrace_mesh_lit_to_rgba(
                &mesh.positions,
                &mesh.normals,
                &mesh.uvs,
                &mesh.indices,
                &mesh.material_ids,
                &mats,
                &texs,
                eye,
                target,
                fov_y,
                w,
                h,
                spp,
                &rig,
            )
            .unwrap_or_else(|e| panic!("{label} render failed: {e}"));
            let secs = t0.elapsed().as_secs_f64();
            worst_secs = worst_secs.max(secs);
            let mut rgba = rgba;
            for px in rgba.chunks_exact_mut(4) {
                px[3] = 255;
            }
            let png = out_dir.join(format!("{label}.png"));
            write_png_rgba(png.to_str().unwrap(), &rgba, w, h);
            eprintln!("[massing] wrote {} ({secs:.2}s)", png.display());
        }
        eprintln!(
            "[massing] render time {worst_secs:.2}s/frame (< 10s) -> {}",
            if worst_secs < 10.0 { "PASS" } else { "FAIL" }
        );
        eprintln!(
            "[massing] eyeball: a glass curtain-wall tower seated on a wider punched-window \
             podium (parapet crown), a three-band stepped ziggurat, and the plain 8-floor box"
        );
        assert!(
            worst_secs < 10.0,
            "gate renders are not cheap: {worst_secs:.2}s/frame"
        );
    }


    /// Facade elevations: straight-on front views so the facade composition
    /// (window stacks, floor hierarchy, casings, trim-zoned roof edges) is
    /// actually inspectable. Denoised eyeball output, sanity-gated non-empty.
    #[cfg(feature = "spectra-native")]
    #[test]
    fn facade_elevations() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig};
        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let rig = LightRig {
            sun_dir: [0.35, 0.55, 0.75],
            sun_intensity: 2.4,
            sky_intensity: 0.35,
            camera_fill: 0.15,
            rim_fill: 0.0,
            sky_dome_intensity: 0.7,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };
        let shots = [
            ("forge.house.craftsman", "facade_craftsman.png"),
            ("city.res_low.l1.2x2.tudor_cottage_01", "facade_tudor.png"),
            ("city.res_high.l5.2x2.highrise_point_tower_01", "facade_highrise.png"),
            ("city.com_reg.l4.3x4.modern_hotel_block_01", "facade_hotel.png"),
        ];
        for (id, png) in shots {
            let path = atoms_dir.join(format!("{id}.atoms.json"));
            let mesh = load_craftsman_mesh(&path);
            let (mats, texs, _) = load_building_mesh_pbr_by_forge_id(&path);
            let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
            for v in &mesh.positions {
                for a in 0..3 {
                    lo[a] = lo[a].min(v[a]);
                    hi[a] = hi[a].max(v[a]);
                }
            }
            // Straight-on elevation of the +Z (front) facade: pull back far
            // enough that the larger of width/height fits a 45-degree fov.
            let c = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
            let span = (hi[0] - lo[0]).max(hi[1] - lo[1]);
            let dist = span * 1.25 + (hi[2] - lo[2]) * 0.5;
            let eye = [c[0], c[1], hi[2] + dist];
            let rgba = pathtrace_mesh_lit_to_rgba(
                &mesh.positions, &mesh.normals, &mesh.uvs, &mesh.indices,
                &mesh.material_ids, &mats, &texs,
                eye, c, std::f32::consts::FRAC_PI_4, 512, 512, 32, &rig,
            )
            .expect("facade render");
            let lit = rgba
                .chunks_exact(4)
                .filter(|px| px[0] as u32 + px[1] as u32 + px[2] as u32 > 30)
                .count();
            assert!(lit > 20_000, "{id}: facade render nearly empty ({lit})");
            let mut pxv: Vec<[u8; 4]> =
                rgba.chunks_exact(4).map(|p| [p[0], p[1], p[2], 255]).collect();
            crate::denoiser::SpectralDenoiser::new(0.7).denoise(&mut pxv, 512, 512);
            let flat: Vec<u8> = pxv.into_iter().flatten().collect();
            let out = std::env::temp_dir().join(png);
            write_png_rgba(out.to_str().unwrap(), &flat, 512, 512);
            eprintln!("[facade] wrote {}", out.display());
        }
    }


    /// Showcase: render a roster of cooked buildings (each a different style
    /// family + kind) through the mesh path — visual proof of the F7 family
    /// routing (tudor/industrial textures) and the catalog breadth. No hard
    /// quality gates; sanity = each render is non-empty and distinct.
    #[cfg(feature = "spectra-native")]
    #[test]
    fn showcase_building_roster() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig};
        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let rig = LightRig {
            sun_dir: [0.45, 0.65, 0.55],
            sun_intensity: 2.6,
            sky_intensity: 0.3,
            camera_fill: 0.2,
            rim_fill: 0.0,
            sky_dome_intensity: 0.65,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };
        let shots = [
            ("city.res_low.l1.2x2.tudor_cottage_01", "showcase_tudor.png"),
            ("city.ind_heavy.l2.5x5.heavy_factory_01", "showcase_factory.png"),
            ("city.res_high.l5.2x2.highrise_point_tower_01", "showcase_highrise.png"),
        ];
        for (id, png) in shots {
            let path = atoms_dir.join(format!("{id}.atoms.json"));
            if !path.exists() {
                panic!("showcase asset missing: {}", path.display());
            }
            let mesh = load_craftsman_mesh(&path);
            let (mats, texs, _) = load_building_mesh_pbr_by_forge_id(&path);
            // 3/4 aerial framing scaled to the building's bounds.
            let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
            for v in &mesh.positions {
                for a in 0..3 {
                    lo[a] = lo[a].min(v[a]);
                    hi[a] = hi[a].max(v[a]);
                }
            }
            let c = [
                (lo[0] + hi[0]) * 0.5,
                (lo[1] + hi[1]) * 0.5,
                (lo[2] + hi[2]) * 0.5,
            ];
            let ext = ((hi[0] - lo[0]).powi(2)
                + (hi[1] - lo[1]).powi(2)
                + (hi[2] - lo[2]).powi(2))
            .sqrt();
            let eye = [c[0] + ext * 0.75, c[1] + ext * 0.45, c[2] + ext * 0.75];
            let rgba = pathtrace_mesh_lit_to_rgba(
                &mesh.positions,
                &mesh.normals,
                &mesh.uvs,
                &mesh.indices,
                &mesh.material_ids,
                &mats,
                &texs,
                eye,
                c,
                std::f32::consts::FRAC_PI_4,
                384,
                384,
                32,
                &rig,
            )
            .expect("showcase render");
            let lit = rgba
                .chunks_exact(4)
                .filter(|px| px[0] as u32 + px[1] as u32 + px[2] as u32 > 30)
                .count();
            assert!(lit > 10_000, "{id}: render nearly empty ({lit} lit px)");
            let mut pxv: Vec<[u8; 4]> = rgba
                .chunks_exact(4)
                .map(|p| [p[0], p[1], p[2], 255])
                .collect();
            crate::denoiser::SpectralDenoiser::new(0.7).denoise(&mut pxv, 384, 384);
            let flat: Vec<u8> = pxv.into_iter().flatten().collect();
            let out = std::env::temp_dir().join(png);
            write_png_rgba(out.to_str().unwrap(), &flat, 384, 384);
            eprintln!("[showcase] wrote {} ({lit} lit px)", out.display());
        }
    }



    /// Forge-facade wave 1, Task 1 gate (L/U/T cook unlock): render the cooked
    /// L-SHAPED walkup (`city.res_med.l3.3x4.lshape_walkup`) through the proven
    /// mesh path (`pathtrace_mesh_lit_to_rgba` + the cooked-mesh/PBR loaders)
    /// from a top-down inspection camera and verify the building is REALLY
    /// non-rectangular:
    ///   1. GROUND-CAP L PROOF — the cook seals every footprint with an
    ///      underside cap at y = 0 that follows the footprint polygon, so the
    ///      cap rasterized into an XZ grid must fill three bbox quadrants and
    ///      leave one (the L's removed corner) empty. This is measured from
    ///      the cooked payload itself, never assumed.
    ///   2. GPU SILHOUETTE GATE — pixels whose camera ray stays inside the
    ///      void quadrant for every height of the building must read as
    ///      BACKGROUND in the path-traced image (`void_cov < 0.02`) while the
    ///      building still fills the frame (`body_cov > 0.25`). Prints
    ///      `void_corner_empty: <bool>` and writes
    ///      `lut_lshape_nonrectangular.png` for human inspection.
    /// Background pixels are classified against the mean of the four frame
    /// corners (always sky-dome — the building never reaches them), and the
    /// classifier is cross-checked against the CPU rasterization of the same
    /// mesh with the same pinhole before any gate uses it.
    ///
    /// Run alone (GPU, seconds/frame is fine):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --release --lib forge_facade_lut_lshape -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn forge_facade_lut_lshape() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig};

        let atoms = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(
            "Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms/city.res_med.l3.3x4.lshape_walkup.atoms.json",
        );
        let mesh = load_craftsman_mesh(&atoms);
        let (materials, textures, _channels) = load_building_mesh_pbr_by_forge_id(&atoms);
        eprintln!(
            "[lut_lshape] cooked mesh: {} verts, {} tris, {} materials, {} textures",
            mesh.positions.len(),
            mesh.indices.len(),
            materials.len(),
            textures.len()
        );

        // ---- 1. Ground-cap L proof: rasterize the y≈0 underside cap into an
        // XZ grid and measure per-quadrant fill. The cap is the cook's seal of
        // the REAL footprint polygon, so an L cook leaves exactly one bbox
        // quadrant empty; a rectangular cook fills all four. -----------------
        let cap_tris: Vec<&[u32; 3]> = mesh
            .indices
            .iter()
            .filter(|tri| {
                tri.iter()
                    .all(|&i| mesh.positions[i as usize][1] <= 0.02)
            })
            .collect();
        assert!(
            !cap_tris.is_empty(),
            "cooked mesh has no ground cap at y<=0.02 — the seal pass drifted"
        );
        let (mut fx0, mut fx1, mut fz0, mut fz1) =
            (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
        for tri in &cap_tris {
            for &i in tri.iter() {
                let p = mesh.positions[i as usize];
                fx0 = fx0.min(p[0]);
                fx1 = fx1.max(p[0]);
                fz0 = fz0.min(p[2]);
                fz1 = fz1.max(p[2]);
            }
        }
        const GRID: usize = 128;
        let mut cap_grid = vec![false; GRID * GRID];
        for tri in &cap_tris {
            let p: Vec<[f32; 2]> = tri
                .iter()
                .map(|&i| {
                    let q = mesh.positions[i as usize];
                    [q[0], q[2]]
                })
                .collect();
            let den = (p[1][1] - p[2][1]) * (p[0][0] - p[2][0])
                + (p[2][0] - p[1][0]) * (p[0][1] - p[2][1]);
            if den.abs() < 1e-9 {
                continue;
            }
            for gx in 0..GRID {
                let qx = fx0 + (gx as f32 + 0.5) / GRID as f32 * (fx1 - fx0);
                for gz in 0..GRID {
                    let qz = fz0 + (gz as f32 + 0.5) / GRID as f32 * (fz1 - fz0);
                    let l1 = ((p[1][1] - p[2][1]) * (qx - p[2][0])
                        + (p[2][0] - p[1][0]) * (qz - p[2][1]))
                        / den;
                    let l2 = ((p[2][1] - p[0][1]) * (qx - p[2][0])
                        + (p[0][0] - p[2][0]) * (qz - p[2][1]))
                        / den;
                    let l3 = 1.0 - l1 - l2;
                    if l1 >= -1e-6 && l2 >= -1e-6 && l3 >= -1e-6 {
                        cap_grid[gx * GRID + gz] = true;
                    }
                }
            }
        }
        let (cx, cz) = ((fx0 + fx1) * 0.5, (fz0 + fz1) * 0.5);
        // Quadrant fill fractions, cells within 0.4 m of the quadrant border
        // excluded so shared walls never leak across.
        let mut quad_fill = [0.0f64; 4];
        for (q, fill) in quad_fill.iter_mut().enumerate() {
            let (want_xp, want_zp) = (q & 1 == 1, q & 2 == 2);
            let (mut inside, mut filled) = (0usize, 0usize);
            for gx in 0..GRID {
                let x = fx0 + (gx as f32 + 0.5) / GRID as f32 * (fx1 - fx0);
                if (x > cx + 0.4) != want_xp || (x - cx).abs() <= 0.4 {
                    continue;
                }
                for gz in 0..GRID {
                    let z = fz0 + (gz as f32 + 0.5) / GRID as f32 * (fz1 - fz0);
                    if (z > cz + 0.4) != want_zp || (z - cz).abs() <= 0.4 {
                        continue;
                    }
                    inside += 1;
                    filled += cap_grid[gx * GRID + gz] as usize;
                }
            }
            *fill = filled as f64 / inside.max(1) as f64;
        }
        let void_q = (0..4)
            .min_by(|&a, &b| quad_fill[a].total_cmp(&quad_fill[b]))
            .unwrap();
        let filled_min = (0..4)
            .filter(|&q| q != void_q)
            .map(|q| quad_fill[q])
            .fold(f64::INFINITY, f64::min);
        eprintln!(
            "[lut_lshape] ground-cap quadrant fill (x-,z-)/(x+,z-)/(x-,z+)/(x+,z+): \
             {:.3} {:.3} {:.3} {:.3} -> void quadrant {} (fill {:.3}), other min {:.3}",
            quad_fill[0], quad_fill[1], quad_fill[2], quad_fill[3], void_q,
            quad_fill[void_q], filled_min
        );
        assert!(
            quad_fill[void_q] < 0.05 && filled_min > 0.85,
            "cooked footprint does not read as an L: quadrant fills {quad_fill:?} \
             (need one ~empty quadrant and three ~full)"
        );

        // ---- 2. Path-trace top-down and gate the GPU silhouette over the
        // void quadrant. -------------------------------------------------------
        let (w, h) = (512u32, 512u32);
        let fov_y = std::f32::consts::FRAC_PI_4;
        let eye = [0.0f32, 40.0, 0.1];
        let target = [0.0f32, 0.0, 0.0];
        // Bright high dome so the below-horizon background reads clearly
        // brighter than the dark slate roof — the legacy dome's near-black
        // brown sat within the classifier's distance of the roof texture.
        let rig = LightRig {
            sun_dir: [0.4, 0.8, 0.4],
            sun_intensity: 2.6,
            sky_dome_intensity: 1.0,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.85, 0.90, 0.98],
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let rgba = pathtrace_mesh_lit_to_rgba(
            &mesh.positions,
            &mesh.normals,
            &mesh.uvs,
            &mesh.indices,
            &mesh.material_ids,
            &materials,
            &textures,
            eye,
            target,
            fov_y,
            w,
            h,
            64,
            &rig,
        )
        .expect("L-shape walkup mesh render should succeed");
        let secs = t0.elapsed().as_secs_f64();
        let png = std::env::temp_dir().join("lut_lshape_nonrectangular.png");
        let mut opaque = rgba.clone();
        for px in opaque.chunks_exact_mut(4) {
            px[3] = 255;
        }
        write_png_rgba(png.to_str().unwrap(), &opaque, w, h);
        eprintln!(
            "[lut_lshape] wrote {} ({secs:.2}s, 64 spp top-down)",
            png.display()
        );

        // Diagnostic companion view (NOT gated): a 3/4 view into the removed
        // corner so a human can inspect how the notch is actually built —
        // walls, soffit, roof. Written beside the gate PNG.
        {
            let (vx, vz) = (
                if void_q & 1 == 1 { 1.0f32 } else { -1.0 },
                if void_q & 2 == 2 { 1.0f32 } else { -1.0 },
            );
            let eye34 = [vx * 16.0, 12.0, vz * 18.0];
            let rgba34 = pathtrace_mesh_lit_to_rgba(
                &mesh.positions,
                &mesh.normals,
                &mesh.uvs,
                &mesh.indices,
                &mesh.material_ids,
                &materials,
                &textures,
                eye34,
                [0.0, 4.0, 0.0],
                fov_y,
                w,
                h,
                64,
                &rig,
            )
            .expect("3/4 diagnostic render should succeed");
            let mut o = rgba34;
            for px in o.chunks_exact_mut(4) {
                px[3] = 255;
            }
            let png34 = std::env::temp_dir().join("lut_lshape_three_quarter.png");
            write_png_rgba(png34.to_str().unwrap(), &o, w, h);
            eprintln!("[lut_lshape] wrote diagnostic 3/4 view {}", png34.display());
        }

        // GPU silhouette via a MATTE pass: the same mesh/camera rendered with
        // every material flat WHITE + emissive, no textures — geometry then
        // reads near-white regardless of sun/shadow while the background
        // stays the dome, so a luma threshold is an exact GPU silhouette.
        // (The below-horizon dome cannot be re-coloured through the rig, and
        // the slate roof texture overlaps it in both luma and hue, so
        // colour-matching the beauty frame against the frame corners is not
        // a reliable classifier — measured, not assumed.)
        let matte_materials: Vec<super::PbrMaterial> = materials
            .iter()
            .map(|_| super::PbrMaterial {
                base_color: [1.0, 1.0, 1.0],
                roughness: 1.0,
                metallic: 0.0,
                emission_strength: 4.0,
                albedo_tex: -1,
                roughness_tex: -1,
                normal_tex: -1,
                uv_scale: [1.0, 1.0],
                ..Default::default()
            })
            .collect();
        let rgba_matte = pathtrace_mesh_lit_to_rgba(
            &mesh.positions,
            &mesh.normals,
            &mesh.uvs,
            &mesh.indices,
            &mesh.material_ids,
            &matte_materials,
            &[],
            eye,
            target,
            fov_y,
            w,
            h,
            64,
            &rig,
        )
        .expect("matte silhouette render should succeed");
        let matte_luma = |p: usize| -> f64 {
            0.2126 * rgba_matte[p * 4] as f64
                + 0.7152 * rgba_matte[p * 4 + 1] as f64
                + 0.0722 * rgba_matte[p * 4 + 2] as f64
        };
        let corners = [
            0usize,
            (w - 1) as usize,
            ((h - 1) * w) as usize,
            ((h - 1) * w + (w - 1)) as usize,
        ];
        let bg_luma = corners.iter().map(|&p| matte_luma(p)).sum::<f64>() / 4.0;
        assert!(
            bg_luma < 140.0,
            "matte background luma {bg_luma:.0} too bright for a clean threshold"
        );
        let thr = bg_luma + 60.0;
        let is_background = |p: usize| -> bool { matte_luma(p) < thr };

        // Cross-check: CPU raster of the same mesh with the same pinhole must
        // agree with the GPU non-background silhouette.
        let (mats_px, _tris_px) = rasterize_material_masks(&mesh, eye, target, fov_y, w, h);
        let (mut raster_cov, mut lit_px, mut overlap) = (0usize, 0usize, 0usize);
        for p in 0..(w * h) as usize {
            let on_raster = mats_px[p] >= 0;
            let on_render = !is_background(p);
            raster_cov += on_raster as usize;
            lit_px += on_render as usize;
            overlap += (on_raster && on_render) as usize;
        }
        let agree_raster = overlap as f64 / raster_cov.max(1) as f64;
        let agree_render = overlap as f64 / lit_px.max(1) as f64;
        eprintln!(
            "[lut_lshape] silhouette cross-check: raster {raster_cov} px, render {lit_px} px, \
             overlap/raster {agree_raster:.3}, overlap/render {agree_render:.3}"
        );
        assert!(
            agree_raster > 0.9 && agree_render > 0.9,
            "GPU silhouette and CPU raster disagree (agree_raster {agree_raster:.3}, \
             agree_render {agree_render:.3}) — the background classifier or camera drifted"
        );

        // The void screen block: pixels whose ray stays inside the void
        // quadrant (inset 0.45 m, clear of walls/overhang) from the ground to
        // above the roof — geometry can cover them ONLY by occupying the
        // removed corner. The ray's XZ track is linear in height, so inside
        // at y=0 and y=ymax+0.6 means inside throughout.
        let ymax = mesh
            .positions
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let (qx0, qx1) = if void_q & 1 == 1 { (cx, fx1) } else { (fx0, cx) };
        let (qz0, qz1) = if void_q & 2 == 2 { (cz, fz1) } else { (fz0, cz) };
        let inset = 0.45f32;
        let in_void = |x: f32, z: f32| -> bool {
            x > qx0 + inset && x < qx1 - inset && z > qz0 + inset && z < qz1 - inset
        };
        // Same pinhole as rasterize_material_masks / the camera kernel.
        let eye_v = glam::Vec3::from(eye);
        let fwd = (glam::Vec3::from(target) - eye_v).normalize();
        let right = fwd.cross(glam::Vec3::Y).normalize();
        let up = right.cross(fwd);
        let tan_half = (fov_y * 0.5).tan();
        let aspect = w as f32 / h as f32;
        let mut void_px = 0usize;
        let mut void_lit = 0usize;
        for y in 0..h {
            for x in 0..w {
                let ndc_x = (2.0 * (x as f32 + 0.5) / w as f32 - 1.0) * aspect;
                let ndc_y = 1.0 - 2.0 * (y as f32 + 0.5) / h as f32;
                let dir = (fwd + (right * ndc_x + up * ndc_y) * tan_half).normalize();
                if dir.y.abs() < 1e-4 {
                    continue;
                }
                let at = |wy: f32| -> (f32, f32) {
                    let t = (wy - eye_v.y) / dir.y;
                    (eye_v.x + dir.x * t, eye_v.z + dir.z * t)
                };
                let (x0, z0) = at(0.0);
                let (x1, z1) = at(ymax + 0.6);
                if in_void(x0, z0) && in_void(x1, z1) {
                    void_px += 1;
                    void_lit += !is_background((y * w + x) as usize) as usize;
                }
            }
        }
        assert!(
            void_px >= 500,
            "void screen block too small ({void_px} px) — camera framing drifted"
        );
        let void_cov = void_lit as f64 / void_px as f64;
        let body_cov = lit_px as f64 / (w * h) as f64;
        let void_corner_empty = void_cov < 0.02 && body_cov > 0.25;
        eprintln!(
            "void_corner_empty: {void_corner_empty}  (void_cov={void_cov:.3} body_cov={body_cov:.3} \
             void_px={void_px})"
        );
        assert!(
            void_corner_empty,
            "L void corner not empty in the GPU render: void_cov={void_cov:.3} (need < 0.02), \
             body_cov={body_cov:.3} (need > 0.25) — see {}",
            png.display()
        );
    }
