use crate::desc::SpatialHashDesc;

pub struct SpatialHashPipelines {
    pub count: wgpu::ComputePipeline,
    pub count_bgl: wgpu::BindGroupLayout,
    pub prefix: wgpu::ComputePipeline,
    pub prefix_bgl: wgpu::BindGroupLayout,
    pub scatter: wgpu::ComputePipeline,
    pub scatter_bgl: wgpu::BindGroupLayout,
    pub su_buf: wgpu::Buffer,
    pub pu_buf: wgpu::Buffer,
}

impl SpatialHashPipelines {
    pub fn new(_device: &wgpu::Device, _desc: &SpatialHashDesc) -> Self {
        unimplemented!("placeholder")
    }
}

pub fn rebuild_spatial_hash(
    _encoder: &mut wgpu::CommandEncoder,
    _pipelines: &SpatialHashPipelines,
    _buffers: &crate::state::AgentStateBuffers,
) {
}
