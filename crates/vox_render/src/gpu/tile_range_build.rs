//! `TileRangeBuildPass` — the keystone compute pass that builds per-tile
//! `[start, end)` spans from radix-sorted tile keys, feeding `splat_raster`'s
//! `tile_ranges` binding (binding 3). It is the one pass in the tiled chain that
//! did not previously exist: `tile_assign` emits keys, `radix_sort` orders them,
//! but nothing produced the per-tile spans — that work lived only in the CPU
//! `spectra_render` path. This GPU pass mirrors that CPU boundary scan op-for-op,
//! validated bit-exact against [`cpu_tile_ranges`].

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileRangeParams {
    count: u32,
    num_tiles: u32,
    _pad0: u32,
    _pad1: u32,
}

/// CPU oracle: build per-tile `[start, end)` ranges from an array of tile ids in
/// sorted (ascending, grouped) order. Lifted op-for-op from the `spectra_render`
/// boundary scan. Empty tiles stay `[0, 0]` (`start == end` ⇒ skipped by the
/// rasteriser). This is the reference the GPU [`TileRangeBuildPass`] is validated
/// bit-exact against.
pub fn cpu_tile_ranges(sorted_tile_idx: &[u32], num_tiles: usize) -> Vec<[u32; 2]> {
    let mut ranges = vec![[0u32, 0u32]; num_tiles];
    if !sorted_tile_idx.is_empty() {
        let mut start = 0usize;
        let mut current = sorted_tile_idx[0];
        for (i, &t) in sorted_tile_idx.iter().enumerate() {
            if t != current {
                if (current as usize) < num_tiles {
                    ranges[current as usize] = [start as u32, i as u32];
                }
                start = i;
                current = t;
            }
        }
        if (current as usize) < num_tiles {
            ranges[current as usize] = [start as u32, sorted_tile_idx.len() as u32];
        }
    }
    ranges
}

/// GPU twin of [`cpu_tile_ranges`]: two compute entry points (`clear_ranges`,
/// `build_ranges`) over one bind group layout (params uniform, sorted keys,
/// ranges read-write).
pub struct TileRangeBuildPass {
    clear_pipeline: wgpu::ComputePipeline,
    build_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl TileRangeBuildPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_range_build_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("tile_range_build.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile_range_build_bgl"),
            entries: &[
                // 0 — params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1 — sorted tile keys (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2 — per-tile ranges (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile_range_build_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tile_range_clear_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("clear_ranges"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let build_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tile_range_build_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("build_ranges"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self { clear_pipeline, build_pipeline, bgl }
    }

    /// Clear then build the per-tile ranges into the caller-owned `tile_ranges`
    /// buffer (persistent, `num_tiles * 8` bytes, STORAGE). `count` is the number
    /// of sorted entries (the host-side `tile_count` readback); `num_tiles` bounds
    /// the clear. Records into `encoder`; the caller submits.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        sorted_keys_hi: &wgpu::Buffer,
        tile_ranges: &wgpu::Buffer,
        count: u32,
        num_tiles: u32,
    ) {
        let params = TileRangeParams { count, num_tiles, _pad0: 0, _pad1: 0 };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tile_range_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile_range_build_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sorted_keys_hi.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tile_ranges.as_entire_binding() },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile_range_clear"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_tiles.div_ceil(256).max(1), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile_range_build"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(256).max(1), 1, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_oracle_matches_hand_worked_example() {
        // sorted tile ids [0,0,2,2,2,5] over 6 tiles:
        //   t0 = [0,2], t1 empty, t2 = [2,5], t3,t4 empty, t5 = [5,6].
        let got = cpu_tile_ranges(&[0, 0, 2, 2, 2, 5], 6);
        assert_eq!(
            got,
            vec![[0, 2], [0, 0], [2, 5], [0, 0], [0, 0], [5, 6]],
            "boundary scan must match the hand-worked spans"
        );
    }

    #[test]
    fn cpu_oracle_empty_input_is_all_zero() {
        assert_eq!(cpu_tile_ranges(&[], 4), vec![[0, 0]; 4]);
    }
}
