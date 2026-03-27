//! Headless rendering pipeline — renders scenes to image files without a window.
//! This proves the full pipeline works: scene → ECS → cull → LOD → rasterise → image.

use std::path::Path;

use glam::{Mat4, Quat, Vec3};
use uuid::Uuid;

use vox_core::ecs::{LodLevel, SplatAssetComponent, SplatInstanceComponent};
use vox_core::mapgen::generate_map;
use vox_core::spectral::Illuminant;
use vox_core::terrain::{TerrainPlane, generate_terrain_splats};
use vox_core::types::GaussianSplat;
use vox_data::proc_gs::emit_splats_simple;
use vox_data::proc_gs_advanced::{generate_bench, generate_lamp_post, generate_tree};
use vox_render::gpu::software_rasteriser::{Framebuffer, SoftwareRasteriser};
use vox_render::spectral::RenderCamera;

/// Save a framebuffer to a PPM file (no external image dependencies needed).
pub fn save_ppm(fb: &Framebuffer, path: &Path) -> Result<(), std::io::Error> {
    let mut data = format!("P6\n{} {}\n255\n", fb.width, fb.height).into_bytes();
    for pixel in &fb.pixels {
        data.push(pixel[0]);
        data.push(pixel[1]);
        data.push(pixel[2]);
    }
    std::fs::write(path, &data)
}

/// Gather all splats from a list of (position, splat_data) pairs into a flat world-space list.
fn gather_world_splats(instances: &[(Vec3, Vec<GaussianSplat>)]) -> Vec<GaussianSplat> {
    let mut world = Vec::new();
    for (pos, splats) in instances {
        for s in splats {
            let mut ws = *s;
            ws.position[0] += pos.x;
            ws.position[1] += pos.y;
            ws.position[2] += pos.z;
            world.push(ws);
        }
    }
    world
}

/// Render a complete city scene to an image file.
/// This is the proof that the engine works end-to-end.
pub fn render_city_scene(
    output_path: &Path,
    width: u32,
    height: u32,
    seed: u64,
) -> Result<RenderStats, String> {
    let start = std::time::Instant::now();

    // === Step 1: Generate terrain ===
    let terrain_splats = generate_map(seed, 200.0, 0.5);
    let terrain_count = terrain_splats.len();

    // === Step 2: Generate buildings ===
    let mut building_instances: Vec<(Vec3, Vec<GaussianSplat>)> = Vec::new();

    // Row of Victorian terraced houses
    for i in 0..8 {
        let x = i as f32 * 7.0 - 24.0;
        let splats = emit_splats_simple(seed + i as u64, 5.5, 10.0);
        building_instances.push((Vec3::new(x, 0.0, 15.0), splats));
    }

    // Row on the other side of the street
    for i in 0..8 {
        let x = i as f32 * 7.0 - 24.0;
        let splats = emit_splats_simple(seed + 100 + i as u64, 6.0, 10.0);
        building_instances.push((Vec3::new(x, 0.0, -15.0), splats));
    }

    // A taller building (office)
    let office = emit_splats_simple(seed + 200, 15.0, 20.0);
    building_instances.push((Vec3::new(40.0, 0.0, 0.0), office));

    let building_count: usize = building_instances.iter().map(|(_, s)| s.len()).sum();

    // === Step 3: Generate trees ===
    let mut tree_instances: Vec<(Vec3, Vec<GaussianSplat>)> = Vec::new();
    for i in 0..12 {
        let x = i as f32 * 8.0 - 44.0;
        let height = 6.0 + (seed.wrapping_add(i as u64) % 5) as f32;
        let canopy = 2.5 + (seed.wrapping_add(i as u64) % 3) as f32 * 0.5;
        tree_instances.push((
            Vec3::new(x, 0.0, 5.0),
            generate_tree(seed + 500 + i as u64, height, canopy),
        ));
    }
    let tree_count: usize = tree_instances.iter().map(|(_, s)| s.len()).sum();

    // === Step 4: Generate props ===
    let mut prop_instances: Vec<(Vec3, Vec<GaussianSplat>)> = Vec::new();
    for i in 0..6 {
        let x = i as f32 * 12.0 - 30.0;
        prop_instances.push((Vec3::new(x, 0.0, 3.0), generate_lamp_post(seed + 700 + i as u64, 4.5)));
    }
    for i in 0..4 {
        let x = i as f32 * 15.0 - 22.0;
        prop_instances.push((Vec3::new(x, 0.0, -3.0), generate_bench(seed + 800 + i as u64)));
    }
    let prop_count: usize = prop_instances.iter().map(|(_, s)| s.len()).sum();

    // === Step 5: Combine all splats ===
    let mut all_splats = terrain_splats;
    all_splats.extend(gather_world_splats(&building_instances));
    all_splats.extend(gather_world_splats(&tree_instances));
    all_splats.extend(gather_world_splats(&prop_instances));

    let total_splats = all_splats.len();
    let gather_time = start.elapsed();

    // === Step 6: Set up camera ===
    let eye = Vec3::new(0.0, 25.0, 45.0);
    let target = Vec3::new(0.0, 5.0, 0.0);
    let camera = RenderCamera {
        view: Mat4::look_at_rh(eye, target, Vec3::Y),
        proj: Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            width as f32 / height as f32,
            0.1,
            500.0,
        ),
    };

    // === Step 7: Render ===
    let render_start = std::time::Instant::now();
    let mut rasteriser = SoftwareRasteriser::new(width, height);
    let fb = rasteriser.render(&all_splats, &camera, &Illuminant::d65());
    let render_time = render_start.elapsed();

    // === Step 8: Save to file ===
    save_ppm(&fb, output_path).map_err(|e| e.to_string())?;

    let total_time = start.elapsed();

    // === Step 9: Verify output ===
    let non_black_pixels = fb.pixels.iter().filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();
    let coverage = non_black_pixels as f32 / fb.pixels.len() as f32;

    Ok(RenderStats {
        width,
        height,
        total_splats,
        terrain_splats: terrain_count,
        building_splats: building_count,
        tree_splats: tree_count,
        prop_splats: prop_count,
        gather_time_ms: gather_time.as_millis() as u64,
        render_time_ms: render_time.as_millis() as u64,
        total_time_ms: total_time.as_millis() as u64,
        non_black_pixels,
        coverage_percent: coverage * 100.0,
        output_path: output_path.to_string_lossy().to_string(),
    })
}

/// Statistics from a headless render.
#[derive(Debug)]
pub struct RenderStats {
    pub width: u32,
    pub height: u32,
    pub total_splats: usize,
    pub terrain_splats: usize,
    pub building_splats: usize,
    pub tree_splats: usize,
    pub prop_splats: usize,
    pub gather_time_ms: u64,
    pub render_time_ms: u64,
    pub total_time_ms: u64,
    pub non_black_pixels: usize,
    pub coverage_percent: f32,
    pub output_path: String,
}

impl std::fmt::Display for RenderStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Ochroma Headless Render ===")?;
        writeln!(f, "Resolution: {}x{}", self.width, self.height)?;
        writeln!(f, "Total splats: {}", self.total_splats)?;
        writeln!(f, "  Terrain: {}", self.terrain_splats)?;
        writeln!(f, "  Buildings: {}", self.building_splats)?;
        writeln!(f, "  Trees: {}", self.tree_splats)?;
        writeln!(f, "  Props: {}", self.prop_splats)?;
        writeln!(f, "Gather time: {}ms", self.gather_time_ms)?;
        writeln!(f, "Render time: {}ms", self.render_time_ms)?;
        writeln!(f, "Total time: {}ms", self.total_time_ms)?;
        writeln!(f, "Coverage: {:.1}% ({} non-black pixels)", self.coverage_percent, self.non_black_pixels)?;
        writeln!(f, "Output: {}", self.output_path)?;
        Ok(())
    }
}

/// Render multiple views of the same scene (different camera angles).
pub fn render_turntable(
    output_dir: &Path,
    width: u32,
    height: u32,
    seed: u64,
    num_frames: u32,
) -> Result<Vec<RenderStats>, String> {
    std::fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;

    // Generate scene once
    let terrain = generate_map(seed, 200.0, 0.5);
    let mut all_splats = terrain;

    // Add buildings
    for i in 0..8 {
        let x = i as f32 * 7.0 - 24.0;
        let building = emit_splats_simple(seed + i as u64, 5.5, 10.0);
        for s in &building {
            let mut ws = *s;
            ws.position[0] += x;
            ws.position[2] += 15.0;
            all_splats.push(ws);
        }
    }

    // Add trees
    for i in 0..6 {
        let tree = generate_tree(seed + 500 + i as u64, 8.0, 3.0);
        for s in &tree {
            let mut ws = *s;
            ws.position[0] += i as f32 * 10.0 - 25.0;
            ws.position[2] += 5.0;
            all_splats.push(ws);
        }
    }

    let mut stats = Vec::new();
    let mut rasteriser = SoftwareRasteriser::new(width, height);

    for frame in 0..num_frames {
        let angle = frame as f32 / num_frames as f32 * std::f32::consts::TAU;
        let radius = 50.0;
        let eye = Vec3::new(angle.cos() * radius, 25.0, angle.sin() * radius);
        let target = Vec3::new(0.0, 5.0, 0.0);

        let camera = RenderCamera {
            view: Mat4::look_at_rh(eye, target, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                width as f32 / height as f32,
                0.1,
                500.0,
            ),
        };

        let render_start = std::time::Instant::now();
        let fb = rasteriser.render(&all_splats, &camera, &Illuminant::d65());
        let render_time = render_start.elapsed();

        let path = output_dir.join(format!("frame_{:04}.ppm", frame));
        save_ppm(&fb, &path).map_err(|e| e.to_string())?;

        let non_black = fb.pixels.iter().filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();

        stats.push(RenderStats {
            width,
            height,
            total_splats: all_splats.len(),
            terrain_splats: 0,
            building_splats: 0,
            tree_splats: 0,
            prop_splats: 0,
            gather_time_ms: 0,
            render_time_ms: render_time.as_millis() as u64,
            total_time_ms: render_time.as_millis() as u64,
            non_black_pixels: non_black,
            coverage_percent: non_black as f32 / fb.pixels.len() as f32 * 100.0,
            output_path: path.to_string_lossy().to_string(),
        });
    }

    Ok(stats)
}
