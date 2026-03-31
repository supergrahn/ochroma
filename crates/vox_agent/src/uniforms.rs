use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AgentUniforms {
    pub agent_count: u32,
    pub custom_floats: u32,
    pub dt: f32,
    pub time: f32,
    pub grid_width: u32,
    pub cell_size: f32,
    pub _pad: [f32; 2],
}
