/// Describes the agent state layout. Drives buffer allocation and bind group layout.
#[derive(Debug, Clone)]
pub struct AgentStateDesc {
    pub agent_count: u32,
    /// Game-defined floats per agent. 0 = no custom buffer.
    pub custom_floats: u32,
    /// Include a spectral_cache[N*16] buffer.
    pub spectral: bool,
    /// Enable spatial hash. None = no spatial hash.
    pub spatial_hash: Option<SpatialHashDesc>,
}

/// Configuration for the spatial hash grid.
#[derive(Debug, Clone)]
pub struct SpatialHashDesc {
    /// World-space X origin of the grid.
    pub grid_origin_x: f32,
    /// World-space Z origin of the grid.
    pub grid_origin_z: f32,
    /// Grid covers [origin, origin + grid_extent] in X and Z.
    pub grid_extent: f32,
    /// Side length of each grid cell in world units.
    pub cell_size: f32,
}

impl SpatialHashDesc {
    /// Number of cells along one axis. grid_extent / cell_size, rounded up.
    pub fn grid_width(&self) -> u32 {
        (self.grid_extent / self.cell_size).ceil() as u32
    }

    /// Total number of cells (grid_width²).
    pub fn cell_count(&self) -> u32 {
        self.grid_width() * self.grid_width()
    }
}

impl AgentStateDesc {
    /// Bytes per agent in the positions buffer (3 × f32, no padding).
    pub fn position_stride(&self) -> u64 { 12 }

    /// Total byte size of the positions buffer (one side of ping-pong).
    pub fn positions_size(&self) -> u64 {
        self.agent_count as u64 * self.position_stride()
    }

    /// Total byte size of the custom floats buffer.
    pub fn custom_size(&self) -> u64 {
        self.agent_count as u64 * self.custom_floats as u64 * 4
    }

    /// Total byte size of the spectral cache buffer (N * 16 * 4 bytes).
    pub fn spectral_size(&self) -> u64 {
        if !self.spectral { return 0; }
        self.agent_count as u64 * 16 * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_width_rounds_up() {
        // 105.0 / 10.0 = 10.5 → ceil = 11, not 10
        let sh = SpatialHashDesc {
            grid_origin_x: 0.0,
            grid_origin_z: 0.0,
            grid_extent: 105.0,
            cell_size: 10.0,
        };
        assert_eq!(sh.grid_width(), 11);
    }

    #[test]
    fn cell_count_is_grid_width_squared() {
        let sh = SpatialHashDesc {
            grid_origin_x: 0.0, grid_origin_z: 0.0,
            grid_extent: 100.0, cell_size: 10.0,
        };
        assert_eq!(sh.cell_count(), 100);
    }

    #[test]
    fn positions_size_is_twelve_bytes_per_agent() {
        let desc = AgentStateDesc {
            agent_count: 1000,
            custom_floats: 0,
            spectral: false,
            spatial_hash: None,
        };
        assert_eq!(desc.positions_size(), 12_000);
    }

    #[test]
    fn custom_size_zero_when_no_custom_floats() {
        let desc = AgentStateDesc {
            agent_count: 500,
            custom_floats: 0,
            spectral: false,
            spatial_hash: None,
        };
        assert_eq!(desc.custom_size(), 0);
    }

    #[test]
    fn custom_size_correct_with_eight_floats() {
        let desc = AgentStateDesc {
            agent_count: 100,
            custom_floats: 8,
            spectral: false,
            spatial_hash: None,
        };
        assert_eq!(desc.custom_size(), 100 * 8 * 4);
    }

    #[test]
    fn spectral_size_zero_when_spectral_disabled() {
        let desc = AgentStateDesc {
            agent_count: 100,
            custom_floats: 0,
            spectral: false,
            spatial_hash: None,
        };
        assert_eq!(desc.spectral_size(), 0);
    }

    #[test]
    fn spectral_size_correct_when_spectral_enabled() {
        let desc = AgentStateDesc {
            agent_count: 100,
            custom_floats: 0,
            spectral: true,
            spatial_hash: None,
        };
        assert_eq!(desc.spectral_size(), 100 * 16 * 4);
    }
}
