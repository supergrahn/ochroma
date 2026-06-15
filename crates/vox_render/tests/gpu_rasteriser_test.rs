#![cfg(feature = "legacy-raster")]
// Gated behind `legacy-raster` (THE LAW: Spectra is the only product renderer).
// This test exercises a banned rasterizer/software-rasterizer stack; it compiles
// and runs only under `--features legacy-raster`, never in the product build.

use vox_render::gpu::gpu_rasteriser::{CameraUniform, GpuSplatData};

#[test]
fn gpu_splat_data_size() {
    assert_eq!(std::mem::size_of::<GpuSplatData>(), 64);
}

#[test]
fn camera_uniform_size() {
    assert_eq!(std::mem::size_of::<CameraUniform>(), 208);
}
