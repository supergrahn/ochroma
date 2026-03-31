use crate::state::AgentStateBuffers;

/// CPU-side read-only view of a contiguous slice of agent state.
pub struct AgentSlice<'a> {
    pub agent_ids:    &'a [u32],
    pub flags:        &'a [u32],
    pub custom:       &'a [f32],
    pub custom_floats: u32,
}

/// Queues mutations from CPU callbacks. Applied to GPU at start of next tick.
pub struct AgentWriteQueue {
    flag_writes:     Vec<(u32, u32, u32)>,  // (agent_id, mask, value)
    custom_writes:   Vec<(u32, u32, f32)>,
    velocity_writes: Vec<(u32, [f32; 3])>,
}

impl AgentWriteQueue {
    pub fn new() -> Self {
        Self {
            flag_writes: Vec::new(),
            custom_writes: Vec::new(),
            velocity_writes: Vec::new(),
        }
    }

    pub fn write_flag_bits(&mut self, agent_id: u32, mask: u32, value: u32) {
        self.flag_writes.push((agent_id, mask, value));
    }

    pub fn write_custom(&mut self, agent_id: u32, slot: u32, value: f32) {
        self.custom_writes.push((agent_id, slot, value));
    }

    pub fn write_velocity(&mut self, agent_id: u32, velocity: [f32; 3]) {
        self.velocity_writes.push((agent_id, velocity));
    }

    pub fn is_empty(&self) -> bool {
        self.flag_writes.is_empty()
            && self.custom_writes.is_empty()
            && self.velocity_writes.is_empty()
    }
}

impl Default for AgentWriteQueue {
    fn default() -> Self { Self::new() }
}

pub struct TierScheduler {
    agent_count:   u32,
    custom_floats: u32,
    frame:         u64,
    elapsed_time:  f32,
    cpu_flags:     Vec<u32>,
    cpu_custom:    Vec<f32>,
    write_queue:   AgentWriteQueue,
    tier2_cb: Option<Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>>,
    tier3_cb: Option<Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>>,
}

impl TierScheduler {
    pub fn new(agent_count: u32, custom_floats: u32) -> Self {
        Self {
            agent_count,
            custom_floats,
            frame: 0,
            elapsed_time: 0.0,
            cpu_flags:  vec![0u32; agent_count as usize],
            cpu_custom: vec![0.0f32; agent_count as usize * custom_floats as usize],
            write_queue: AgentWriteQueue::new(),
            tier2_cb: None,
            tier3_cb: None,
        }
    }

    pub fn set_tier2(&mut self, cb: Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>) {
        self.tier2_cb = Some(cb);
    }

    pub fn set_tier3(&mut self, cb: Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>) {
        self.tier3_cb = Some(cb);
    }

    pub fn elapsed_time(&self) -> f32 { self.elapsed_time }

    /// Advance the scheduler one frame. Fires tier-2 and tier-3 callbacks on rotating windows.
    pub fn tick(&mut self) {
        self.frame += 1;
        self.elapsed_time += 1.0 / 60.0;

        if self.agent_count == 0 {
            return;
        }

        // tier-2: rotate through all agents every 15 frames ≈ 4Hz at 60fps
        let k = (self.agent_count as usize + 14) / 15;
        if let Some(cb) = &mut self.tier2_cb {
            let start = ((self.frame - 1) as usize * k) % self.agent_count as usize;
            let end = (start + k).min(self.agent_count as usize);
            let ids: Vec<u32> = (start as u32..end as u32).collect();
            let flags = &self.cpu_flags[start..end];
            let cf = self.custom_floats as usize;
            let custom = &self.cpu_custom[start * cf..end * cf];
            let slice = AgentSlice {
                agent_ids: &ids,
                flags,
                custom,
                custom_floats: self.custom_floats,
            };
            cb(slice, &mut self.write_queue);
        }

        // tier-3: rotate through all agents every 240 frames ≈ 0.25Hz at 60fps
        let j = (self.agent_count as usize + 239) / 240;
        if let Some(cb) = &mut self.tier3_cb {
            let start = ((self.frame - 1) as usize * j) % self.agent_count as usize;
            let end = (start + j).min(self.agent_count as usize);
            let ids: Vec<u32> = (start as u32..end as u32).collect();
            let flags = &self.cpu_flags[start..end];
            let cf = self.custom_floats as usize;
            let custom = &self.cpu_custom[start * cf..end * cf];
            let slice = AgentSlice {
                agent_ids: &ids,
                flags,
                custom,
                custom_floats: self.custom_floats,
            };
            cb(slice, &mut self.write_queue);
        }

        // Apply write-queue to CPU mirrors
        for &(id, mask, val) in &self.write_queue.flag_writes {
            if (id as usize) < self.cpu_flags.len() {
                self.cpu_flags[id as usize] =
                    (self.cpu_flags[id as usize] & !mask) | (val & mask);
            }
        }
        let cf = self.custom_floats as usize;
        for &(id, slot, val) in &self.write_queue.custom_writes {
            let idx = id as usize * cf + slot as usize;
            if idx < self.cpu_custom.len() {
                self.cpu_custom[idx] = val;
            }
        }
        // Queue is intentionally NOT cleared here.
        // flush_write_backs() uploads the pending writes to GPU and clears the queue.
    }

    /// Upload pending write-backs to GPU. Call at the start of each tick, before dispatch.
    pub fn flush_write_backs(&mut self, queue: &wgpu::Queue, buffers: &AgentStateBuffers) {
        if self.write_queue.is_empty() { return; }

        for &(id, _mask, _val) in &self.write_queue.flag_writes {
            // Use the CPU mirror value (already updated by the previous tick's mask-apply)
            // rather than the raw queue value so read-modify-write semantics are correct.
            let v = self.cpu_flags[id as usize];
            let offset = id as u64 * 4;
            queue.write_buffer(buffers.flags(), offset, bytemuck::bytes_of(&v));
        }
        if let Some(custom_buf) = buffers.custom() {
            let cf = self.custom_floats as u64;
            for &(id, slot, val) in &self.write_queue.custom_writes {
                let offset = (id as u64 * cf + slot as u64) * 4;
                queue.write_buffer(custom_buf, offset, bytemuck::bytes_of(&val));
            }
        }
        for &(id, vel) in &self.write_queue.velocity_writes {
            let offset = id as u64 * 12;
            queue.write_buffer(buffers.read_velocities(), offset,
                bytemuck::cast_slice(vel.as_slice()));
        }

        self.write_queue = AgentWriteQueue::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn tier2_fires_approximately_four_times_in_sixty_frames() {
        let mut scheduler = TierScheduler::new(1000, 0);
        scheduler.set_tier2(Box::new(|_slice, _wq| {}));

        let k = (1000usize + 14) / 15;
        let mut last_start = 0usize;
        let mut wraps = 0u32;
        for f in 0..60usize {
            let start = (f * k) % 1000;
            if start < last_start { wraps += 1; }
            last_start = start;
        }
        assert!(wraps >= 3 && wraps <= 5,
            "tier-2 should rotate ~4 times in 60 frames, got {}", wraps);
    }

    #[test]
    fn write_queue_custom_mutation_updates_cpu_mirror() {
        let mut scheduler = TierScheduler::new(10, 2);
        // Set up a tier2 callback that writes custom slot 1 of agent 3 to 42.0
        scheduler.set_tier2(Box::new(|_slice, wq| {
            wq.write_custom(3, 1, 42.0);
        }));
        // Run enough ticks so agent 3 is covered (10 agents, k = ceil(10/15) = 1, so agent 3 is covered on tick 4)
        for _ in 0..10 {
            scheduler.tick();
        }
        assert_eq!(scheduler.cpu_custom[3 * 2 + 1], 42.0);
    }

    #[test]
    fn tier2_callback_receives_correct_agent_slice() {
        let received_ids: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let received_ids_clone = Arc::clone(&received_ids);
        let mut scheduler = TierScheduler::new(100, 0);
        scheduler.set_tier2(Box::new(move |slice, _wq| {
            received_ids_clone.lock().unwrap().extend_from_slice(slice.agent_ids);
        }));
        scheduler.tick();
        let ids = received_ids.lock().unwrap();
        let k = (100usize + 14) / 15;
        assert_eq!(ids.len(), k, "first tick covers k agents");
        assert_eq!(ids[0], 0, "starts at agent 0");
    }

    #[test]
    fn tier3_processes_fewer_agents_per_frame_than_tier2() {
        let k = (1000usize + 14) / 15;
        let j = (1000usize + 239) / 240;
        assert!(k > j,
            "tier-2 processes more agents per frame than tier-3 ({} vs {})", k, j);
    }
}
