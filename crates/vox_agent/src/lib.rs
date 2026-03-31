pub mod desc;
pub mod state;
pub mod uniforms;
pub mod spatial_hash;
pub mod compute;
pub mod scheduler;

#[cfg(feature = "editor")]
pub mod node_graph;
#[cfg(feature = "editor")]
pub mod codegen;
#[cfg(feature = "editor")]
pub mod editor;

#[cfg(test)]
mod gpu;

pub use desc::{AgentStateDesc, SpatialHashDesc};
pub use state::AgentStateBuffers;
pub use uniforms::AgentUniforms;
pub use spatial_hash::{SpatialHashPipelines, rebuild_spatial_hash};
pub use compute::{AgentComputePipeline, ShaderSource, PipelineError};
pub use scheduler::{TierScheduler, AgentSlice, AgentWriteQueue};

use std::sync::{Arc, Mutex};

/// Top-level GPU agent compute layer. Owned by the game's EngineApp.
pub struct AgentComputeLayer {
    buffers: AgentStateBuffers,
    pipeline: Option<AgentComputePipeline>,
    pending: Option<Arc<Mutex<Option<AgentComputePipeline>>>>,
    spatial_hash: Option<SpatialHashPipelines>,
    scheduler: TierScheduler,
    #[cfg(feature = "editor")]
    editor: editor::AgentNodeEditor,
}

impl AgentComputeLayer {
    pub fn new(device: &wgpu::Device, desc: AgentStateDesc) -> Self {
        let spatial_hash = desc.spatial_hash.as_ref()
            .map(|sh| SpatialHashPipelines::new(device, sh));
        let agent_count = desc.agent_count;
        let custom_floats = desc.custom_floats;
        Self {
            buffers: AgentStateBuffers::new(device, desc),
            pipeline: None,
            pending: None,
            spatial_hash,
            scheduler: TierScheduler::new(agent_count, custom_floats),
            #[cfg(feature = "editor")]
            editor: editor::AgentNodeEditor::new(),
        }
    }

    pub fn load_shader(
        &mut self,
        device: &wgpu::Device,
        source: ShaderSource,
    ) -> Result<(), PipelineError> {
        let pipeline = AgentComputePipeline::new(device, source, self.buffers.desc())?;
        self.pipeline = Some(pipeline);
        Ok(())
    }

    /// Load the built-in passthrough shader (integrate velocity into position).
    pub fn load_default_shader(&mut self, device: &wgpu::Device) -> Result<(), PipelineError> {
        let wgsl = include_str!("../shaders/default_agent.wgsl").to_string();
        self.load_shader(device, ShaderSource::Wgsl(wgsl))
    }

    pub fn bind_group_layout_source(&self) -> String {
        compute::layout_source(self.buffers.desc())
    }

    pub fn buffers_mut(&mut self) -> &mut AgentStateBuffers {
        &mut self.buffers
    }

    pub fn set_tier2_callback(
        &mut self,
        cb: Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>,
    ) {
        self.scheduler.set_tier2(cb);
    }

    pub fn set_tier3_callback(
        &mut self,
        cb: Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>,
    ) {
        self.scheduler.set_tier3(cb);
    }

    pub fn tick(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        spectral_samples: Option<&wgpu::Buffer>,
        dt: f32,
    ) {
        // 1. Apply write-backs from last CPU callback
        self.scheduler.flush_write_backs(queue, &self.buffers);

        // 2. Rebuild spatial hash if enabled
        if let (Some(sh_pipelines), Some(_)) = (&self.spatial_hash, self.buffers.spatial_cells()) {
            let desc = self.buffers.desc();
            let sh_desc = desc.spatial_hash.as_ref().unwrap();
            let n = desc.agent_count;

            encoder.clear_buffer(self.buffers.cell_counts().unwrap(), 0, None);

            // Count pass
            {
                use bytemuck::{Pod, Zeroable};
                #[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
                struct SU { agent_count: u32, grid_width: u32, cell_size: f32,
                            origin_x: f32, origin_z: f32, _pad: [u32; 3] }
                let su = SU {
                    agent_count: n, grid_width: sh_desc.grid_width(),
                    cell_size: sh_desc.cell_size, origin_x: sh_desc.grid_origin_x,
                    origin_z: sh_desc.grid_origin_z, _pad: [0; 3],
                };
                queue.write_buffer(&sh_pipelines.su_buf, 0, bytemuck::bytes_of(&su));

                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sh_count_bg"),
                    layout: &sh_pipelines.count_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0,
                            resource: self.buffers.read_positions().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1,
                            resource: self.buffers.cell_counts().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2,
                            resource: sh_pipelines.su_buf.as_entire_binding() },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor { label: Some("sh_count"), timestamp_writes: None });
                pass.set_pipeline(&sh_pipelines.count);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups((n + 63) / 64, 1, 1);
            }

            // Prefix sum pass
            {
                use bytemuck::{Pod, Zeroable};
                #[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
                struct PU { cell_count: u32, _pad: [u32; 3] }
                let pu = PU { cell_count: sh_desc.cell_count(), _pad: [0; 3] };
                queue.write_buffer(&sh_pipelines.pu_buf, 0, bytemuck::bytes_of(&pu));

                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sh_prefix_bg"),
                    layout: &sh_pipelines.prefix_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0,
                            resource: self.buffers.cell_counts().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1,
                            resource: self.buffers.cell_offsets().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2,
                            resource: sh_pipelines.pu_buf.as_entire_binding() },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor { label: Some("sh_prefix"), timestamp_writes: None });
                pass.set_pipeline(&sh_pipelines.prefix);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            // Reset cell_counts for scatter
            encoder.clear_buffer(self.buffers.cell_counts().unwrap(), 0, None);

            // Scatter pass
            {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sh_scatter_bg"),
                    layout: &sh_pipelines.scatter_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0,
                            resource: self.buffers.read_positions().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1,
                            resource: self.buffers.cell_counts().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2,
                            resource: self.buffers.cell_offsets().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3,
                            resource: self.buffers.cell_data().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4,
                            resource: self.buffers.spatial_cells().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5,
                            resource: sh_pipelines.su_buf.as_entire_binding() },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor { label: Some("sh_scatter"), timestamp_writes: None });
                pass.set_pipeline(&sh_pipelines.scatter);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups((n + 63) / 64, 1, 1);
            }
        }

        // 3. Dispatch behavior shader
        if let Some(pipeline) = &self.pipeline {
            let desc = self.buffers.desc();
            let uniforms = AgentUniforms {
                agent_count: desc.agent_count,
                custom_floats: desc.custom_floats,
                dt,
                time: self.scheduler.elapsed_time(),
                grid_width: desc.spatial_hash.as_ref().map(|s| s.grid_width()).unwrap_or(0),
                cell_size: desc.spatial_hash.as_ref().map(|s| s.cell_size).unwrap_or(1.0),
                _pad: [0.0; 2],
            };
            pipeline.dispatch(device, encoder, queue, &self.buffers, spectral_samples, uniforms);
        }

        // 4. Swap ping-pong buffers
        self.buffers.swap();

        // 5. Check pending hot-swap
        if let Some(pending) = self.pending.clone() {
            if let Ok(mut guard) = pending.try_lock() {
                if let Some(new_pipeline) = guard.take() {
                    self.pipeline = Some(new_pipeline);
                    self.pending = None;
                }
            }
        }

        // 6. Advance tier scheduler
        self.scheduler.tick();
    }

    #[cfg(feature = "editor")]
    pub fn show_editor(&mut self, ui: &mut egui::Ui) {
        self.editor.show(ui, self.buffers.desc());
    }
}
