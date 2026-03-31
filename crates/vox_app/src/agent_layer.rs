//! Thin wrapper wiring AgentComputeLayer into the engine app.

use vox_agent::{AgentComputeLayer, AgentStateDesc};

/// Holds the AgentComputeLayer and exposes it for engine-runner wiring.
pub struct AgentComputeLayerHandle {
    pub layer: AgentComputeLayer,
}

impl AgentComputeLayerHandle {
    pub fn new(device: &wgpu::Device) -> Self {
        let desc = AgentStateDesc {
            agent_count: 0,       // starts empty; game sets agent count on scene load
            custom_floats: 8,     // 8 game-defined floats per agent
            spectral: true,       // spectral field enabled
            spatial_hash: Some(vox_agent::SpatialHashDesc {
                grid_origin_x: -20480.0,
                grid_origin_z: -20480.0,
                grid_extent:   40960.0,
                cell_size:     10.0,
            }),
        };
        let mut layer = AgentComputeLayer::new(device, desc);
        layer.load_default_shader(device).expect("default agent shader must load");
        Self { layer }
    }
}
