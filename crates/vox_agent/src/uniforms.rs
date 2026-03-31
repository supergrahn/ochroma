use bytemuck::{Pod, Zeroable};

/// Uniform data uploaded to the GPU each frame.
/// Layout must match the `AgentUniforms` struct in every WGSL shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AgentUniforms {
    pub agent_count:  u32,
    pub custom_floats: u32,
    pub dt:            f32,
    pub time:          f32,
    pub grid_width:    u32,
    pub cell_size:     f32,
    pub _pad:         [f32; 2],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_are_pod_and_correct_size() {
        // bytemuck::Pod requires the type to be safely transmutable. This test
        // confirms the derive compiled successfully and the size is a multiple of 16
        // (wgpu uniform buffer alignment requirement).
        let u = AgentUniforms {
            agent_count: 1000, custom_floats: 8, dt: 0.016, time: 0.0,
            grid_width: 1024, cell_size: 10.0, _pad: [0.0; 2],
        };
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 32);
        assert_eq!(bytes.len() % 16, 0, "uniform buffer must be 16-byte aligned");
    }
}
