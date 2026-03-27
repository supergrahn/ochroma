use vox_render::gpu::gpu_rasteriser::{CameraUniform, GpuSplatData};

#[test]
fn gpu_splat_data_size() {
    assert_eq!(std::mem::size_of::<GpuSplatData>(), 64);
}

#[test]
fn camera_uniform_size() {
    assert_eq!(std::mem::size_of::<CameraUniform>(), 208);
}
