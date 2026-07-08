//! GPU integer Eikonal / integration field + the cross-vendor bit-exact gate.
//!
//! # Render-input ONLY until the bit-exact gate is GREEN (rules 6 & 9)
//! The GPU Eikonal field is **ephemeral, approximate, and OUTSIDE the
//! replay-hashed sim** (brief §9 rule 6). It must NOT feed any authoritative
//! integer transfer (presence-deltas, OD outflow, meso leave-events) until the
//! cross-vendor bit-exact gate [`gpu_eikonal_bit_exact_vs_cpu`] is GREEN: the
//! `radix_sort` raster-nondeterminism landmine means floating-point or
//! reduction-order GPU work can diverge from the CPU reference per-vendor. This
//! relaxation is **integer-only** (no float anywhere in the recurrence) and uses
//! a pure `min` over the *previous* full field (Jacobi), so the result is a pure
//! function of the cost grid and the fixed iteration count — identical on the
//! CPU, on RADV/AMD-780M, and on any other vendor. The gate test is the ONLY
//! consumer in this plan; promotion into the sim is a separate, later step.
//!
//! # The integer recurrence (CPU reference == GPU kernel, bit-for-bit)
//! Cost convention (shared by [`cpu_integer_eikonal`] and the WGSL kernel):
//! - `cost[c] == 0`        → a **seed** cell: distance is pinned to `0`.
//! - `cost[c] == u32::MAX` → **impassable**: distance stays `u32::MAX` forever.
//! - otherwise             → the integer cost to *enter* cell `c`.
//!
//! One Jacobi sweep, for every non-seed, passable cell `c`:
//! `next[c] = min(prev[c], min over 4-neighbours n of prev[n] + cost[c])`
//! using **saturating** addition (so `u32::MAX + cost` cannot wrap). Seeds are
//! held at `0`; impassable cells are held at `u32::MAX`. Exactly `iters` sweeps
//! run — no early-out, no convergence check — so CPU and GPU execute the *same*
//! fixed number of identical integer relaxations.

use super::GpuContext;
use wgpu::util::DeviceExt;

/// The WGSL compute kernel: one integer Jacobi relaxation sweep. `src` is the
/// previous field, `dst` the next; `cost` and `dims` are read-only. Integer-only
/// `min`/saturating-add → bit-identical to [`cpu_integer_eikonal`] cross-vendor.
const EIKONAL_WGSL: &str = r#"
struct Dims { w: u32, h: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read>       src:  array<u32>;
@group(0) @binding(1) var<storage, read_write> dst:  array<u32>;
@group(0) @binding(2) var<storage, read>       cost: array<u32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

const UMAX: u32 = 4294967295u;

fn sat_add(a: u32, b: u32) -> u32 {
    if (a == UMAX || b == UMAX) { return UMAX; }
    // a, b are < UMAX here; their sum may still exceed UMAX, so saturate.
    if (a > UMAX - b) { return UMAX; }
    return a + b;
}

@compute @workgroup_size(8, 8, 1)
fn relax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = dims.w;
    let h = dims.h;
    if (gid.x >= w || gid.y >= h) { return; }
    let c = gid.y * w + gid.x;

    let ce = cost[c];
    // Impassable: pinned to UMAX forever.
    if (ce == UMAX) { dst[c] = UMAX; return; }
    // Seed: pinned to 0 forever.
    if (ce == 0u) { dst[c] = 0u; return; }

    var best = src[c];
    // North
    if (gid.y > 0u) {
        let n = (gid.y - 1u) * w + gid.x;
        let cand = sat_add(src[n], ce);
        if (cand < best) { best = cand; }
    }
    // South
    if (gid.y + 1u < h) {
        let n = (gid.y + 1u) * w + gid.x;
        let cand = sat_add(src[n], ce);
        if (cand < best) { best = cand; }
    }
    // East
    if (gid.x + 1u < w) {
        let n = gid.y * w + (gid.x + 1u);
        let cand = sat_add(src[n], ce);
        if (cand < best) { best = cand; }
    }
    // West
    if (gid.x > 0u) {
        let n = gid.y * w + (gid.x - 1u);
        let cand = sat_add(src[n], ce);
        if (cand < best) { best = cand; }
    }
    dst[c] = best;
}
"#;

/// CPU reference integer Eikonal: `iters` fixed Jacobi relaxation sweeps.
///
/// This is the authority the GPU kernel must match cell-for-cell. Same cost
/// convention as the module doc (`0` = seed, `u32::MAX` = impassable). No float
/// anywhere; `saturating_add` mirrors the WGSL `sat_add`.
pub fn cpu_integer_eikonal(cost: &[u32], w: u32, h: u32, iters: u32) -> Vec<u32> {
    let n = (w as usize) * (h as usize);
    assert_eq!(cost.len(), n, "cost grid must be w * h");

    // Initial field: seeds at 0, impassable at u32::MAX, everything else u32::MAX.
    let mut prev = vec![u32::MAX; n];
    for c in 0..n {
        if cost[c] == 0 {
            prev[c] = 0;
        }
    }
    let mut next = prev.clone();

    for _ in 0..iters {
        for y in 0..h {
            for x in 0..w {
                let c = (y * w + x) as usize;
                let ce = cost[c];
                if ce == u32::MAX {
                    next[c] = u32::MAX;
                    continue;
                }
                if ce == 0 {
                    next[c] = 0;
                    continue;
                }
                let mut best = prev[c];
                let relax = |ni: usize, best: &mut u32| {
                    let cand = prev[ni].saturating_add(ce);
                    if cand < *best {
                        *best = cand;
                    }
                };
                if y > 0 {
                    relax(((y - 1) * w + x) as usize, &mut best);
                }
                if y + 1 < h {
                    relax(((y + 1) * w + x) as usize, &mut best);
                }
                if x + 1 < w {
                    relax((y * w + (x + 1)) as usize, &mut best);
                }
                if x > 0 {
                    relax((y * w + (x - 1)) as usize, &mut best);
                }
                next[c] = best;
            }
        }
        std::mem::swap(&mut prev, &mut next);
    }

    // After the final swap, `prev` holds the result of the last sweep.
    prev
}

/// GPU integer Eikonal: `iters` fixed Jacobi relaxation sweeps on the device,
/// ping-ponging two storage buffers, then a read-back. Integer-only → must equal
/// [`cpu_integer_eikonal`] cell-for-cell on any vendor.
///
/// **Render-input ONLY** until the bit-exact gate is GREEN (rules 6 & 9): never
/// wire the result into a replay-hashed integer transfer.
pub fn eikonal_integer_gpu(ctx: &GpuContext, cost: &[u32], w: u32, h: u32, iters: u32) -> Vec<u32> {
    let n = (w as usize) * (h as usize);
    assert_eq!(cost.len(), n, "cost grid must be w * h");

    let device = ctx.device();
    let queue = ctx.queue();

    // Initial field: seeds at 0, everything else u32::MAX (impassable included).
    let mut init = vec![u32::MAX; n];
    for c in 0..n {
        if cost[c] == 0 {
            init[c] = 0;
        }
    }

    let buf_bytes = (n * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

    // Two ping-pong field buffers (storage, both read+write across passes).
    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("eikonal-field-a"),
        contents: bytemuck::cast_slice(&init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("eikonal-field-b"),
        contents: bytemuck::cast_slice(&init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let cost_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("eikonal-cost"),
        contents: bytemuck::cast_slice(cost),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dims = [w, h, 0u32, 0u32];
    let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("eikonal-dims"),
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("eikonal-relax"),
        source: wgpu::ShaderSource::Wgsl(EIKONAL_WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("eikonal-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("eikonal-pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("eikonal-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("relax"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // src=A → dst=B, and the reverse, so we can ping-pong.
    let bind_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("eikonal-bg-a-to-b"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cost_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dims_buf.as_entire_binding(),
            },
        ],
    });
    let bind_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("eikonal-bg-b-to-a"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cost_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dims_buf.as_entire_binding(),
            },
        ],
    });

    let groups_x = w.div_ceil(8);
    let groups_y = h.div_ceil(8);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("eikonal-enc"),
    });
    // After `iters` sweeps, the result lives in A when `iters` is even (we end on
    // a B→A pass), and in B when `iters` is odd (we end on an A→B pass).
    for i in 0..iters {
        let bg = if i % 2 == 0 {
            &bind_a_to_b
        } else {
            &bind_b_to_a
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("eikonal-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    let result_is_b = iters % 2 == 1;
    let result_buf = if result_is_b { &buf_b } else { &buf_a };

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("eikonal-readback"),
        size: buf_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(result_buf, 0, &readback, 0, buf_bytes);
    queue.submit(Some(encoder.finish()));

    // Map and read back the final field.
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .expect("map_async callback dropped")
        .expect("eikonal readback map failed");

    let data = slice.get_mapped_range();
    let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback.unmap();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A checkerboard cost grid with a single seed at the origin. Two distinct
    /// passable costs (1 and 2) so the relaxation actually propagates different
    /// integer sums; `cost==0` at cell 0 is the lone seed both backends flood from.
    fn checkerboard_cost(w: u32, h: u32) -> Vec<u32> {
        let mut cost = vec![0u32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let c = (y * w + x) as usize;
                cost[c] = if (x + y) % 2 == 0 { 1 } else { 2 };
            }
        }
        cost[0] = 0; // the single seed.
        cost
    }

    fn make_headless_context() -> Option<GpuContext> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;
            let info = adapter.get_info();
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("eikonal-test"),
                        required_features: wgpu::Features::empty(),
                        required_limits: adapter.limits(),
                        memory_hints: wgpu::MemoryHints::default(),
                    },
                    None,
                )
                .await
                .ok()?;
            Some(GpuContext::from_parts(&device, &queue, &info))
        })
    }

    /// Cross-vendor integer bit-exact gate (capability "GPU Eikonal bit-exact
    /// gate", Phase 6): the GPU integer field must equal the CPU reference field
    /// cell-for-cell at a fixed iteration count, before any value feeds the
    /// replay-hashed sim (rules 6 & 9).
    #[test]
    fn gpu_eikonal_bit_exact_vs_cpu() {
        let (w, h) = (256u32, 256u32);
        let cost = checkerboard_cost(w, h);
        let iters = 32;
        let cpu = cpu_integer_eikonal(&cost, w, h, iters);
        let Some(ctx) = make_headless_context() else {
            eprintln!("no GPU; skip");
            return;
        };
        let gpu = eikonal_integer_gpu(&ctx, &cost, w, h, iters);
        let same = cpu.iter().zip(&gpu).filter(|(a, b)| a == b).count();
        println!(
            "cpu==gpu cells {}/{} identical (fixed {} iters)",
            same,
            cpu.len(),
            iters
        );
        assert_eq!(
            cpu, gpu,
            "cross-vendor integer bit-exact gate must be GREEN before sim promotion"
        );
    }
}
