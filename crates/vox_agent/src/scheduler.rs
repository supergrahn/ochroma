use crate::state::AgentStateBuffers;

pub struct AgentSlice<'a> {
    pub agent_ids: &'a [u32],
    pub flags: &'a [u32],
    pub custom: &'a [f32],
    pub custom_floats: u32,
}

pub struct AgentWriteQueue;

pub struct TierScheduler {
    agent_count: u32,
    custom_floats: u32,
    elapsed: f32,
    tier2: Option<Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>>,
    tier3: Option<Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>>,
}

impl TierScheduler {
    pub fn new(agent_count: u32, custom_floats: u32) -> Self {
        Self {
            agent_count,
            custom_floats,
            elapsed: 0.0,
            tier2: None,
            tier3: None,
        }
    }

    pub fn set_tier2(&mut self, cb: Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>) {
        self.tier2 = Some(cb);
    }

    pub fn set_tier3(&mut self, cb: Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>) {
        self.tier3 = Some(cb);
    }

    pub fn elapsed_time(&self) -> f32 {
        self.elapsed
    }

    pub fn flush_write_backs(&mut self, _queue: &wgpu::Queue, _buffers: &AgentStateBuffers) {}

    pub fn tick(&mut self) {
        self.elapsed += 1.0 / 60.0;
    }
}
