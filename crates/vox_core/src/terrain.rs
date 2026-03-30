use crate::types::GaussianSplat;
use half::f16;

pub struct TerrainPlane {
    pub width: f32,
    pub depth: f32,
    pub density: f32,
}

impl TerrainPlane {
    pub fn new(width: f32, depth: f32, density: f32) -> Self {
        Self { width, depth, density }
    }
}

pub fn generate_terrain_splats(terrain: &TerrainPlane, material: &str) -> Vec<GaussianSplat> {
    // 16-band SPD: 380–755 nm at 25 nm steps (USGS wavelength grid)
    let spd: [f32; 16] = match material {
        "grass"       => [0.03, 0.03, 0.04, 0.04, 0.05, 0.06, 0.10, 0.38, 0.36, 0.22, 0.14, 0.08, 0.06, 0.42, 0.46, 0.48],
        "cobblestone" => [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.17, 0.18, 0.18, 0.18, 0.17, 0.17, 0.16, 0.16, 0.15, 0.15],
        _             => [0.03, 0.03, 0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06],
    };
    let spectral: [u16; 16] = std::array::from_fn(|i| f16::from_f32(spd[i]).to_bits());

    let spacing = 1.0 / terrain.density.sqrt();
    let nx = (terrain.width / spacing).ceil() as i32;
    let nz = (terrain.depth / spacing).ceil() as i32;
    let mut splats = Vec::with_capacity((nx * nz) as usize);

    for ix in 0..nx {
        for iz in 0..nz {
            let pos = [ix as f32 * spacing - terrain.width * 0.5, 0.0, iz as f32 * spacing - terrain.depth * 0.5];
            splats.push(GaussianSplat::surface(
                pos,
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                spacing * 0.5,
                spacing * 0.5,
                250,
                spectral,
            ));
        }
    }
    splats
}
