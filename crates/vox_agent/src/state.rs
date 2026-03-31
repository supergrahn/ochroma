use crate::desc::AgentStateDesc;

pub struct AgentStateBuffers {
    desc: AgentStateDesc,
}

impl AgentStateBuffers {
    pub fn new(_device: &wgpu::Device, desc: AgentStateDesc) -> Self {
        Self { desc }
    }

    pub fn desc(&self) -> &AgentStateDesc {
        &self.desc
    }

    pub fn swap(&mut self) {}

    pub fn read_positions(&self) -> &wgpu::Buffer {
        unimplemented!("placeholder")
    }

    pub fn spatial_cells(&self) -> Option<&wgpu::Buffer> {
        None
    }

    pub fn cell_counts(&self) -> Option<&wgpu::Buffer> {
        None
    }

    pub fn cell_offsets(&self) -> Option<&wgpu::Buffer> {
        None
    }

    pub fn cell_data(&self) -> Option<&wgpu::Buffer> {
        None
    }
}
