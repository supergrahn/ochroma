use crate::desc::AgentStateDesc;
use crate::state::AgentStateBuffers;
use crate::uniforms::AgentUniforms;

pub enum ShaderSource {
    Wgsl(String),
}

#[derive(Debug)]
pub enum PipelineError {
    Compilation(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for PipelineError {}

pub struct AgentComputePipeline;

impl AgentComputePipeline {
    pub fn new(
        _device: &wgpu::Device,
        _source: ShaderSource,
        _desc: &AgentStateDesc,
    ) -> Result<Self, PipelineError> {
        unimplemented!("placeholder")
    }

    pub fn dispatch(
        &self,
        _device: &wgpu::Device,
        _encoder: &mut wgpu::CommandEncoder,
        _queue: &wgpu::Queue,
        _buffers: &AgentStateBuffers,
        _spectral_samples: Option<&wgpu::Buffer>,
        _uniforms: AgentUniforms,
    ) {
    }
}

pub fn layout_source(_desc: &AgentStateDesc) -> String {
    String::new()
}
