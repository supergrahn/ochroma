// CONFIG-FIRST: the coarse Full/Reduced LOD cutoff is the runtime config value
// `scatter.lod_threshold_m` (`config/ochroma.ron`); default 200.0 m.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodLevel {
    Full,
    Reduced,
}

pub fn select_lod(distance: f32) -> LodLevel {
    if distance <= vox_config::config().scatter.lod_threshold_m {
        LodLevel::Full
    } else {
        LodLevel::Reduced
    }
}

pub fn reduce_splat_indices(indices: &[usize], fraction: f32) -> Vec<usize> {
    let target = (indices.len() as f32 * fraction.clamp(0.0, 1.0)) as usize;
    if target >= indices.len() {
        return indices.to_vec();
    }
    if target == 0 {
        return Vec::new();
    }
    let step = indices.len() as f32 / target as f32;
    (0..target)
        .map(|i| indices[(i as f32 * step) as usize])
        .collect()
}
