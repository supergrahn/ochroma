//! GPU mesh→SDF cook (`SdfBaker`) — SDF-pillar design §4.3 / §6.
//!
//! A WGSL port-and-upgrade of spectra's proven `physics_sdf_gen.slang`
//! three-pass structure, generalized from cube-only `u_resolution` to
//! anisotropic `[nx, ny, nz]`, with the parity sign replaced by the
//! generalized winding number (the same mathematical authority forge's
//! winding repair uses — re-implemented in `vox_core::sdf` because forge's
//! fn is private):
//!
//! 1. **Seed (scatter-min):** one thread per triangle writes exact
//!    closest-point squared distances (Ericson) into voxels within ±2 voxels
//!    of the triangle's AABB via `atomicMin` on the ordered-bits encoding of
//!    a non-negative f32 (bitcast is monotonic for d² ≥ 0); an
//!    equality-guarded second triangle pass writes the closest-point payload
//!    (benign race among equal distances, exactly the race the Slang source
//!    documents).
//! 2. **JFA:** `ceil(log2(max_axis))` ping-pong passes, 26 neighbors,
//!    carrying closest-point xyz; distance is re-derived from the carried
//!    point each step, so in-band voxels end exact.
//! 3. **Sign:** one thread per voxel sums f32 solid angles over all
//!    triangles; `|w| ≥ 0.5` ⇒ inside ⇒ negate.
//! 4. **Encode:** snorm8, saturating at ±`band_voxels · voxel`.
//!
//! Host-side, N=64 stratified CPU GWN probes over the mesh AABB gate the
//! sign (`Err(OpenMesh)` on fractional winding) and feed [`SdfValidation`];
//! the eikonal p95 validator runs on the *GPU output* field. Blocking
//! submissions throughout — this is a cook-time tool that owns its encoders;
//! it is never frame-coupled.

use crate::gpu::GpuContext;
use glam::Vec3;
use std::collections::BTreeSet;
use std::fmt;
use vox_core::sdf::{
    SdfDesc, SdfField, SdfPayload, SdfSign, SdfValidation, eikonal_p95, generalized_winding_number,
};
use wgpu::util::DeviceExt;

/// Tier-1 grid cap from the canonical `SdfDesc` contract (design §5).
const MAX_TIER1_RES: u32 = 96;
/// Stratified CPU GWN probe grid: 4×4×4 = 64 probes (design §4.9).
const PROBE_CELLS: u32 = 4;
/// |w| below this is exterior; |w| above `1 − PROBE_FRACTIONAL_MARGIN` is
/// interior; anything between is fractional ⇒ open mesh (0.5 ± 0.35).
const PROBE_FRACTIONAL_MARGIN: f32 = 0.15;
/// Deterministic seeds for probe jitter and the eikonal validator.
const PROBE_JITTER_SEED: u64 = 0xC001_BA5E;
const EIKONAL_SEED: u64 = 0x5DF_0E1C;
const EIKONAL_SAMPLES: usize = 10_000;

/// Cook target: resolution/padding/band policy, supplied by the caller
/// (the game cook passes 64 / 0.15 / 1.0 / 4.0 per the M1 plan).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdfBakeTarget {
    /// Voxels along the longest padded axis (64 | 96).
    pub max_axis_res: u32,
    /// Lower clamp on voxel size in metres.
    pub min_voxel: f32,
    /// Padding added on every side of the mesh AABB, metres.
    pub pad: f32,
    /// Narrow band half-width in voxels; the snorm8 payload saturates here.
    pub band_voxels: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SdfBakeError {
    /// Empty mesh, out-of-range indices, non-finite vertices, or a
    /// zero-extent AABB — nothing meaningful to cook.
    DegenerateMesh,
    /// CPU GWN probes measured fractional winding: the mesh is not
    /// watertight, so no trustworthy sign exists. The cook fails loudly
    /// instead of cooking a garbage sign (design §4.9).
    OpenMesh { min_abs_winding: f32 },
    /// The requested grid does not fit the device (or the tier-1 contract).
    DeviceLimits(String),
}

impl fmt::Display for SdfBakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DegenerateMesh => write!(f, "degenerate mesh: nothing to cook"),
            Self::OpenMesh { min_abs_winding } => write!(
                f,
                "open mesh: fractional generalized winding |w|={min_abs_winding:.3}"
            ),
            Self::DeviceLimits(msg) => write!(f, "device limits: {msg}"),
        }
    }
}

impl std::error::Error for SdfBakeError {}

/// GPU mesh→SDF cook. Construct once per device (pipelines are compiled in
/// the constructor), then [`Self::bake`] per asset. Blocking; cook-time only.
pub struct SdfBaker {
    ctx: GpuContext,
    grid_layout: wgpu::BindGroupLayout,
    seed_layout: wgpu::BindGroupLayout,
    init: wgpu::ComputePipeline,
    seed_min: wgpu::ComputePipeline,
    seed_payload: wgpu::ComputePipeline,
    jfa: wgpu::ComputePipeline,
    sign_resolve: wgpu::ComputePipeline,
    encode: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsRaw {
    res: [u32; 3],
    num_tris: u32,
    origin: [f32; 3],
    voxel: f32,
    band: f32,
    _pad: [f32; 3],
}

impl SdfBaker {
    /// Shared-context constructor (`GpuGi::new_with_context` precedent).
    pub fn new_with_context(ctx: &GpuContext) -> Result<Self, SdfBakeError> {
        let device = ctx.device();
        let limits = device.limits();
        // Group 0 carries 5 storage buffers + group 1 carries 2 — 7 total.
        if limits.max_storage_buffers_per_shader_stage < 7 {
            return Err(SdfBakeError::DeviceLimits(format!(
                "needs 7 storage buffers per stage, device allows {}",
                limits.max_storage_buffers_per_shader_stage
            )));
        }

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_bake"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sdf_bake.wgsl").into()),
        });

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let grid_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sdf_bake_grid"),
            entries: &[
                uniform(0),
                storage(1, true),  // positions
                storage(2, true),  // indices
                storage(3, false), // dist_bits (atomic u32)
                storage(4, false), // signed_dist
                storage(5, false), // snorm_out
                uniform(6),        // jfa step
            ],
        });
        let seed_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sdf_bake_seed"),
            entries: &[storage(0, true), storage(1, false)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sdf_bake"),
            bind_group_layouts: &[&grid_layout, &seed_layout],
            push_constant_ranges: &[],
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("sdf_bake_{entry}")),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Ok(Self {
            ctx: ctx.clone(),
            init: pipeline("init"),
            seed_min: pipeline("seed_min"),
            seed_payload: pipeline("seed_payload"),
            jfa: pipeline("jfa"),
            sign_resolve: pipeline("sign_resolve"),
            encode: pipeline("encode"),
            grid_layout,
            seed_layout,
        })
    }

    /// Mesh → `SdfField` on GPU: scatter-min seed, JFA, GWN sign, snorm8
    /// encode — gated by CPU GWN probes (open mesh fails loudly) and
    /// validated on the GPU output (inside-negative + eikonal p95).
    pub fn bake(
        &mut self,
        positions: &[[f32; 3]],
        indices: &[[u32; 3]],
        target: SdfBakeTarget,
    ) -> Result<(SdfField, SdfValidation), SdfBakeError> {
        // ── Mesh validation ────────────────────────────────────────────────
        if positions.len() < 3 || indices.is_empty() {
            return Err(SdfBakeError::DegenerateMesh);
        }
        if positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err(SdfBakeError::DegenerateMesh);
        }
        if indices
            .iter()
            .flatten()
            .any(|&i| i as usize >= positions.len())
        {
            return Err(SdfBakeError::DegenerateMesh);
        }

        let mut aabb_min = Vec3::splat(f32::INFINITY);
        let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);
        for p in positions {
            aabb_min = aabb_min.min(Vec3::from(*p));
            aabb_max = aabb_max.max(Vec3::from(*p));
        }
        let extent = aabb_max - aabb_min;
        if extent.min_element() <= 0.0 {
            return Err(SdfBakeError::DegenerateMesh);
        }

        // ── Grid policy ────────────────────────────────────────────────────
        if !(2..=MAX_TIER1_RES).contains(&target.max_axis_res) {
            return Err(SdfBakeError::DeviceLimits(format!(
                "max_axis_res {} outside tier-1 contract 2..={MAX_TIER1_RES}",
                target.max_axis_res
            )));
        }
        if !(target.pad >= 0.0 && target.min_voxel >= 0.0 && target.band_voxels > 0.0) {
            return Err(SdfBakeError::DeviceLimits(
                "pad/min_voxel must be ≥ 0 and band_voxels > 0".into(),
            ));
        }
        let padded_min = aabb_min - Vec3::splat(target.pad);
        let padded_extent = extent + Vec3::splat(2.0 * target.pad);
        let longest = padded_extent.max_element();
        let voxel = (longest / (target.max_axis_res - 1) as f32).max(target.min_voxel);
        let res_for = |e: f32| -> u32 {
            (((e / voxel).ceil() as u32) + 1).clamp(2, target.max_axis_res.min(MAX_TIER1_RES))
        };
        let res = [
            res_for(padded_extent.x),
            res_for(padded_extent.y),
            res_for(padded_extent.z),
        ];
        let band = target.band_voxels * voxel;
        let voxel_count = res[0] as usize * res[1] as usize * res[2] as usize;

        let device = self.ctx.device();
        let queue = self.ctx.queue();
        let limits = device.limits();
        let seed_bytes = voxel_count as u64 * 16;
        if seed_bytes > limits.max_storage_buffer_binding_size as u64 {
            return Err(SdfBakeError::DeviceLimits(format!(
                "{}x{}x{} needs a {seed_bytes}-byte seed buffer, device caps storage bindings at {}",
                res[0], res[1], res[2], limits.max_storage_buffer_binding_size
            )));
        }

        // ── CPU GWN gate: 64 stratified probes over the MESH AABB ──────────
        // (the mesh AABB, not the padded grid — probes inside the body are
        // the informative ones; padding is air by construction).
        let mut rng = SplitMix(PROBE_JITTER_SEED);
        let mut sign_validation_probes: Vec<Vec3> =
            Vec::with_capacity((PROBE_CELLS * PROBE_CELLS * PROBE_CELLS) as usize);
        let mut interior_probe_count = 0usize;
        let mut min_abs_winding = f32::INFINITY;
        let mut fractional = false;
        for cz in 0..PROBE_CELLS {
            for cy in 0..PROBE_CELLS {
                for cx in 0..PROBE_CELLS {
                    let u = (cx as f32 + rng.unit()) / PROBE_CELLS as f32;
                    let v = (cy as f32 + rng.unit()) / PROBE_CELLS as f32;
                    let s = (cz as f32 + rng.unit()) / PROBE_CELLS as f32;
                    let p = aabb_min + Vec3::new(u, v, s) * extent;
                    sign_validation_probes.push(p);
                    let w = generalized_winding_number(positions, indices, p).abs() as f32;
                    if w < PROBE_FRACTIONAL_MARGIN {
                        continue; // clean exterior
                    }
                    min_abs_winding = min_abs_winding.min(w);
                    if w <= 1.0 - PROBE_FRACTIONAL_MARGIN {
                        fractional = true; // |w| in 0.5 ± 0.35 ⇒ open
                    } else {
                        interior_probe_count += 1;
                    }
                }
            }
        }
        if fractional {
            return Err(SdfBakeError::OpenMesh { min_abs_winding });
        }
        if interior_probe_count == 0 {
            // No probe landed inside at all — either a degenerate sliver or a
            // shell; with no interior evidence the Closed sign is unprovable.
            return Err(SdfBakeError::OpenMesh {
                min_abs_winding: 0.0,
            });
        }

        // ── GPU buffers ────────────────────────────────────────────────────
        let positions_flat: Vec<f32> = positions.iter().flatten().copied().collect();
        let indices_flat: Vec<u32> = indices.iter().flatten().copied().collect();
        let params = ParamsRaw {
            res,
            num_tris: indices.len() as u32,
            origin: padded_min.to_array(),
            voxel,
            band,
            _pad: [0.0; 3],
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdf_bake_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let step_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdf_bake_step"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let positions_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdf_bake_positions"),
            contents: bytemuck::cast_slice(&positions_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let indices_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdf_bake_indices"),
            contents: bytemuck::cast_slice(&indices_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let storage_buf = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let dist_buf = storage_buf("sdf_bake_dist_bits", voxel_count as u64 * 4);
        let signed_buf = storage_buf("sdf_bake_signed", voxel_count as u64 * 4);
        let snorm_words = voxel_count.div_ceil(4);
        let snorm_buf = storage_buf("sdf_bake_snorm", snorm_words as u64 * 4);
        let seed_a = storage_buf("sdf_bake_seed_a", seed_bytes);
        let seed_b = storage_buf("sdf_bake_seed_b", seed_bytes);

        let grid_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sdf_bake_grid"),
            layout: &self.grid_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: signed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: snorm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: step_buf.as_entire_binding(),
                },
            ],
        });
        let seed_bind = |label: &str, src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.seed_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                ],
            })
        };
        let bind_ab = seed_bind("sdf_bake_seed_ab", &seed_a, &seed_b); // src=a → dst=b
        let bind_ba = seed_bind("sdf_bake_seed_ba", &seed_b, &seed_a); // src=b → dst=a

        let groups_voxels = voxel_count.div_ceil(256) as u32;
        let groups_tris = indices.len().div_ceil(256) as u32;
        let groups_words = snorm_words.div_ceil(256) as u32;
        let run_pass = |encoder: &mut wgpu::CommandEncoder,
                        pipeline: &wgpu::ComputePipeline,
                        seed: &wgpu::BindGroup,
                        groups: u32| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sdf_bake"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &grid_bind, &[]);
            pass.set_bind_group(1, seed, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        };

        // ── Submit: init + scatter-min seed + payload (dst = seed_a) ───────
        let mut encoder = device.create_command_encoder(&Default::default());
        run_pass(&mut encoder, &self.init, &bind_ba, groups_voxels);
        run_pass(&mut encoder, &self.seed_min, &bind_ba, groups_tris);
        run_pass(&mut encoder, &self.seed_payload, &bind_ba, groups_tris);
        queue.submit([encoder.finish()]);

        // ── JFA: next_pow2(max_axis)/2 … 1 — ceil(log2(max_axis)) passes ───
        let max_axis = res.iter().copied().max().unwrap_or(2);
        let mut step = (max_axis.next_power_of_two() / 2).max(1);
        let mut passes = 0u32;
        loop {
            queue.write_buffer(&step_buf, 0, bytemuck::cast_slice(&[step as i32, 0, 0, 0]));
            let mut encoder = device.create_command_encoder(&Default::default());
            let seed = if passes % 2 == 0 { &bind_ab } else { &bind_ba };
            run_pass(&mut encoder, &self.jfa, seed, groups_voxels);
            queue.submit([encoder.finish()]);
            passes += 1;
            if step == 1 {
                break;
            }
            step /= 2;
        }
        // Final carried points live in seed_a after an even number of passes.
        let final_seed = if passes % 2 == 0 { &bind_ab } else { &bind_ba };

        // ── Sign + encode + readback ───────────────────────────────────────
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdf_bake_staging"),
            size: snorm_words as u64 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        run_pass(&mut encoder, &self.sign_resolve, final_seed, groups_voxels);
        run_pass(&mut encoder, &self.encode, final_seed, groups_words);
        encoder.copy_buffer_to_buffer(&snorm_buf, 0, &staging, 0, snorm_words as u64 * 4);
        queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| SdfBakeError::DeviceLimits(format!("readback channel: {e}")))?
            .map_err(|e| SdfBakeError::DeviceLimits(format!("readback map: {e:?}")))?;
        let payload: Vec<i8> = slice.get_mapped_range()[..voxel_count]
            .iter()
            .map(|&b| b as i8)
            .collect();
        staging.unmap();

        // ── Field + validation (probes run on the GPU output) ──────────────
        let desc = SdfDesc::new(res, padded_min, voxel, band, SdfSign::Closed)
            .map_err(|e| SdfBakeError::DeviceLimits(e.to_string()))?;
        let field = SdfField::new(desc, SdfPayload::Snorm8(payload))
            .map_err(|e| SdfBakeError::DeviceLimits(e.to_string()))?;

        let inside_negative =
            validate_interior_lattice_signs(&field, positions, indices, &sign_validation_probes);
        let p95 = eikonal_p95(&field, EIKONAL_SAMPLES, EIKONAL_SEED);
        let validation = SdfValidation::new(inside_negative, p95, min_abs_winding, false);
        Ok((field, validation))
    }
}

/// Validate the sign actually stored on the SDF lattice against CPU-f64 GWN
/// at those exact lattice points. The closure gate's jittered probes are
/// continuous positions, while the GPU signs discrete voxels; trilinear
/// interpolation near a sharp edge can legitimately cross zero even when all
/// eight stored corner signs are correct. Comparing exact points avoids that
/// continuous-vs-discrete category error without relaxing the GWN threshold.
fn validate_interior_lattice_signs(
    field: &SdfField,
    positions: &[[f32; 3]],
    indices: &[[u32; 3]],
    probes: &[Vec3],
) -> bool {
    let desc = field.desc();
    let res = desc.resolution();
    let origin = desc.origin();
    let voxel = desc.voxel_size();
    let mut corners = BTreeSet::new();
    for &probe in probes {
        let grid = (probe - origin) / voxel;
        if !grid.is_finite() {
            return false;
        }
        let axis = |coordinate: f32, resolution: u32| {
            let max = (resolution - 1) as f32;
            let coordinate = coordinate.clamp(0.0, max);
            (coordinate.floor() as u32, coordinate.ceil() as u32)
        };
        let (x0, x1) = axis(grid.x, res[0]);
        let (y0, y1) = axis(grid.y, res[1]);
        let (z0, z1) = axis(grid.z, res[2]);
        for z in [z0, z1] {
            for y in [y0, y1] {
                for x in [x0, x1] {
                    corners.insert([x, y, z]);
                }
            }
        }
    }

    let quant_step = match field.payload() {
        SdfPayload::Snorm8(_) => desc.narrow_band() / i8::MAX as f32,
        SdfPayload::Snorm16(_) => desc.narrow_band() / i16::MAX as f32,
    };
    let flat_index = |[x, y, z]: [u32; 3]| {
        (z as usize * res[1] as usize + y as usize) * res[0] as usize + x as usize
    };
    let decode = |index: usize| {
        let normalized = match field.payload() {
            SdfPayload::Snorm8(values) => {
                let raw = values[index];
                if raw == i8::MIN {
                    -1.0
                } else {
                    raw as f32 / i8::MAX as f32
                }
            }
            SdfPayload::Snorm16(values) => {
                let raw = values[index];
                if raw == i16::MIN {
                    -1.0
                } else {
                    raw as f32 / i16::MAX as f32
                }
            }
        };
        normalized.clamp(-1.0, 1.0) * desc.narrow_band()
    };

    let mut tested_interior = 0usize;
    for corner in corners {
        let point =
            origin + Vec3::new(corner[0] as f32, corner[1] as f32, corner[2] as f32) * voxel;
        let winding = generalized_winding_number(positions, indices, point).abs() as f32;
        if winding <= 1.0 - PROBE_FRACTIONAL_MARGIN {
            continue;
        }
        tested_interior += 1;
        if decode(flat_index(corner)) > quant_step {
            return false;
        }
    }
    tested_interior > 0
}

/// splitmix64 — deterministic probe jitter.
struct SplitMix(u64);

impl SplitMix {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use vox_core::sdf::generalized_winding_number;

    /// Build a headless [`crate::gpu::GpuContext`] over a real device — the
    /// house pattern from `spectral_gi.rs::try_gpu_context`. Returns `None`
    /// (printing the skip line) without a hardware adapter so GPU-less CI
    /// stays green.
    fn try_gpu_context(label: &str) -> Option<crate::gpu::GpuContext> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));
        let adapter = match adapter {
            Some(a) => a,
            None => {
                eprintln!("[{label}] no adapter — skipping GPU test");
                return None;
            }
        };
        let info = adapter.get_info();
        if crate::gpu::adapter::ensure_hardware(&info).is_err() {
            eprintln!(
                "[{label}] software adapter ({}) — skipping GPU test",
                info.name
            );
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("sdf_bake_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("device creation on a box with a GPU");
        Some(crate::gpu::GpuContext::from_parts(&device, &queue, &info))
    }

    /// 12 outward-wound triangles of the axis-aligned box [min, max].
    fn cube_mesh(min: [f32; 3], max: [f32; 3]) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let (x0, y0, z0) = (min[0], min[1], min[2]);
        let (x1, y1, z1) = (max[0], max[1], max[2]);
        let positions = vec![
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ];
        let indices = vec![
            [0u32, 2, 1],
            [0, 3, 2], // -Z
            [4, 5, 6],
            [4, 6, 7], // +Z
            [0, 1, 5],
            [0, 5, 4], // -Y
            [3, 7, 6],
            [3, 6, 2], // +Y
            [0, 4, 7],
            [0, 7, 3], // -X
            [1, 2, 6],
            [1, 6, 5], // +X
        ];
        (positions, indices)
    }

    fn append_cube(
        positions: &mut Vec<[f32; 3]>,
        indices: &mut Vec<[u32; 3]>,
        min: [f32; 3],
        max: [f32; 3],
    ) {
        let (cube_positions, cube_indices) = cube_mesh(min, max);
        let base = positions.len() as u32;
        positions.extend(cube_positions);
        indices.extend(
            cube_indices
                .into_iter()
                .map(|triangle| [triangle[0] + base, triangle[1] + base, triangle[2] + base]),
        );
    }

    /// Watertight UV sphere: `slices·2` pole fans + `(stacks−2)·slices·2`
    /// band triangles — 8 064 tris at 64×64, the procedural ~8k-tri mesh.
    fn uv_sphere(radius: f32, stacks: u32, slices: u32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let mut positions = vec![[0.0, radius, 0.0]];
        for s in 1..stacks {
            let phi = std::f32::consts::PI * s as f32 / stacks as f32;
            for l in 0..slices {
                let theta = std::f32::consts::TAU * l as f32 / slices as f32;
                positions.push([
                    radius * phi.sin() * theta.cos(),
                    radius * phi.cos(),
                    radius * phi.sin() * theta.sin(),
                ]);
            }
        }
        positions.push([0.0, -radius, 0.0]);
        let bottom = positions.len() as u32 - 1;
        let ring = |s: u32, l: u32| 1 + (s - 1) * slices + (l % slices);

        let mut indices = Vec::new();
        for l in 0..slices {
            indices.push([0, ring(1, l + 1), ring(1, l)]); // top fan, outward
        }
        for s in 1..stacks - 1 {
            for l in 0..slices {
                let (a, b) = (ring(s, l), ring(s, l + 1));
                let (c, d) = (ring(s + 1, l), ring(s + 1, l + 1));
                indices.push([a, b, d]);
                indices.push([a, d, c]);
            }
        }
        for l in 0..slices {
            indices.push([bottom, ring(stacks - 1, l), ring(stacks - 1, l + 1)]);
        }
        (positions, indices)
    }

    /// CPU mirror of the WGSL Ericson closest-point — the oracle's distance.
    fn closest_point_on_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
        let ab = b - a;
        let ac = c - a;
        let ap = p - a;
        let d1 = ab.dot(ap);
        let d2 = ac.dot(ap);
        if d1 <= 0.0 && d2 <= 0.0 {
            return a;
        }
        let bp = p - b;
        let d3 = ab.dot(bp);
        let d4 = ac.dot(bp);
        if d3 >= 0.0 && d4 <= d3 {
            return b;
        }
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            return a + ab * (d1 / (d1 - d3));
        }
        let cp = p - c;
        let d5 = ab.dot(cp);
        let d6 = ac.dot(cp);
        if d6 >= 0.0 && d5 <= d6 {
            return c;
        }
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            return a + ac * (d2 / (d2 - d6));
        }
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
        }
        let denom = 1.0 / (va + vb + vc);
        a + ab * (vb * denom) + ac * (vc * denom)
    }

    /// Closed 2 m cube: GPU-baked field must read deeply negative at the
    /// centre and clearly positive out in the padding — the sign gate.
    #[test]
    fn sdf_bake_closed_cube_sign_gate() {
        let Some(ctx) = try_gpu_context("sdf_bake") else {
            return;
        };
        let (positions, indices) = cube_mesh([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let mut baker = SdfBaker::new_with_context(&ctx).expect("baker");
        let (field, validation) = baker
            .bake(
                &positions,
                &indices,
                SdfBakeTarget {
                    max_axis_res: 32,
                    min_voxel: 0.01,
                    pad: 1.0,
                    band_voxels: 8.0,
                },
            )
            .expect("closed cube must bake");

        let center = field.sample_local(Vec3::ZERO).expect("centre on-grid");
        let corner = field
            .sample_local(Vec3::splat(1.5))
            .expect("padded corner on-grid");
        println!("[sdf_bake] center={center:.3} corner={corner:.3}");
        assert!(center < -0.8, "cube centre must be deeply inside: {center}");
        assert!(corner > 0.4, "padded corner must be outside: {corner}");
        assert!(
            validation.inside_negative(),
            "GWN-interior probes must sample negative on the GPU output"
        );
        assert!(!validation.open_mesh());
        assert!(
            validation.winding_min_abs() > 0.85,
            "closed cube interior winding must be ≈1, got {}",
            validation.winding_min_abs()
        );
    }

    /// A continuous point just inside a sharp convex corner can interpolate
    /// positive from one negative and seven positive lattice corners. That is
    /// a reconstruction-resolution effect, not an incorrect stored sign. The
    /// validation must compare CPU GWN to the exact GPU lattice samples.
    #[test]
    fn sdf_bake_sharp_closed_component_corner_uses_lattice_aligned_sign_validation() {
        let Some(ctx) = try_gpu_context("sdf_bake_sharp_corner") else {
            return;
        };
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        append_cube(
            &mut positions,
            &mut indices,
            [-11.9, 0.0, -18.2],
            [11.9, 18.5, 18.2],
        );
        // An independently closed upper component sits across a sub-voxel air
        // gap. The fixed 4^3 probe stream places one point only centimetres
        // inside all three planes of its sharp corner at the 64-voxel target.
        append_cube(
            &mut positions,
            &mut indices,
            [-6.87, 19.015, -5.0],
            [-2.0, 22.8, 14.825],
        );

        let target = SdfBakeTarget {
            max_axis_res: 64,
            min_voxel: 0.15,
            pad: 1.0,
            band_voxels: 4.0,
        };
        let mut baker = SdfBaker::new_with_context(&ctx).expect("baker");
        let (field, validation) = baker
            .bake(&positions, &indices, target)
            .expect("closed compound mesh must bake");
        assert!(
            validation.inside_negative(),
            "every crisp CPU-interior lattice point must carry a negative GPU sign"
        );

        // Prove the fixture exercises the old category error: continuous CPU
        // probes sampled through trilinear reconstruction do not all remain
        // negative at this finite grid resolution.
        let aabb_min = Vec3::new(-11.9, 0.0, -18.2);
        let extent = Vec3::new(23.8, 22.8, 36.4);
        let quant_step = field.desc().narrow_band() / i8::MAX as f32;
        let mut rng = SplitMix(PROBE_JITTER_SEED);
        let mut old_continuous_check = true;
        for cz in 0..PROBE_CELLS {
            for cy in 0..PROBE_CELLS {
                for cx in 0..PROBE_CELLS {
                    let point = aabb_min
                        + Vec3::new(
                            (cx as f32 + rng.unit()) / PROBE_CELLS as f32,
                            (cy as f32 + rng.unit()) / PROBE_CELLS as f32,
                            (cz as f32 + rng.unit()) / PROBE_CELLS as f32,
                        ) * extent;
                    let winding =
                        generalized_winding_number(&positions, &indices, point).abs() as f32;
                    if winding > 1.0 - PROBE_FRACTIONAL_MARGIN
                        && !field
                            .sample_local(point)
                            .is_some_and(|distance| distance <= quant_step)
                    {
                        old_continuous_check = false;
                    }
                }
            }
        }
        assert!(
            !old_continuous_check,
            "fixture must expose the continuous-vs-lattice false negative"
        );
    }

    /// GPU cook ≡ CPU oracle: brute-force point-triangle distance + the
    /// vox_core GWN sign at every voxel centre of a 24³ bake.
    #[test]
    fn sdf_bake_matches_cpu_oracle() {
        let Some(ctx) = try_gpu_context("sdf_bake") else {
            return;
        };
        let (positions, indices) = cube_mesh([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let mut baker = SdfBaker::new_with_context(&ctx).expect("baker");
        let (field, _validation) = baker
            .bake(
                &positions,
                &indices,
                SdfBakeTarget {
                    max_axis_res: 24,
                    min_voxel: 0.01,
                    pad: 1.0,
                    band_voxels: 4.0,
                },
            )
            .expect("closed cube must bake");

        let desc = field.desc();
        let res = desc.resolution();
        let voxel = desc.voxel_size();
        let band = desc.narrow_band();
        let origin = desc.origin();
        let tri = |t: &[u32; 3]| {
            (
                Vec3::from(positions[t[0] as usize]),
                Vec3::from(positions[t[1] as usize]),
                Vec3::from(positions[t[2] as usize]),
            )
        };

        let mut max_dev = 0.0f32;
        let mut in_band = 0usize;
        for z in 0..res[2] {
            for y in 0..res[1] {
                for x in 0..res[0] {
                    let p = origin + Vec3::new(x as f32, y as f32, z as f32) * voxel;
                    let mut d = f32::INFINITY;
                    for t in &indices {
                        let (a, b, c) = tri(t);
                        d = d.min(p.distance(closest_point_on_tri(p, a, b, c)));
                    }
                    let w = generalized_winding_number(&positions, &indices, p);
                    if w.abs() >= 0.5 {
                        d = -d;
                    }
                    if d.abs() >= 0.9 * band {
                        continue; // saturation region — encoding clamps here
                    }
                    in_band += 1;
                    let gpu = field.sample_local(p).expect("voxel centre on-grid");
                    max_dev = max_dev.max((gpu - d.clamp(-band, band)).abs());
                }
            }
        }
        println!(
            "[sdf_bake] gpu-vs-cpu max_abs_dev={max_dev:.5} over {in_band} in-band voxels (voxel={voxel:.4})"
        );
        assert!(in_band > 500, "oracle must actually cover the band");
        assert!(
            max_dev < voxel * 0.5,
            "GPU cook must match the CPU oracle in-band: dev={max_dev} voxel={voxel}"
        );
    }

    /// The baked field must satisfy the eikonal metric gate — validation runs
    /// on the GPU output (the sphere has no edge singularities, so a clean
    /// bake genuinely reads ≪ 0.15).
    #[test]
    fn sdf_bake_eikonal_p95() {
        let Some(ctx) = try_gpu_context("sdf_bake") else {
            return;
        };
        let (positions, indices) = uv_sphere(1.0, 64, 64);
        let mut baker = SdfBaker::new_with_context(&ctx).expect("baker");
        let (_field, validation) = baker
            .bake(
                &positions,
                &indices,
                SdfBakeTarget {
                    max_axis_res: 48,
                    min_voxel: 0.01,
                    pad: 1.0,
                    band_voxels: 4.0,
                },
            )
            .expect("sphere must bake");
        let p95 = validation.eikonal_p95();
        println!("[sdf_bake] eikonal p95={p95:.4}");
        assert!(p95 < 0.15, "baked sphere must pass the eikonal gate: {p95}");
        assert!(p95 > 0.0, "a literally-zero p95 means nothing was sampled");
    }

    /// Cube minus one face → `Err(OpenMesh)` carrying the fractional |w| the
    /// CPU GWN probes measured — the loud cook failure of design §4.9.
    #[test]
    fn sdf_bake_open_mesh_fails() {
        let Some(ctx) = try_gpu_context("sdf_bake") else {
            return;
        };
        let (positions, indices) = cube_mesh([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        // Remove the +Z face (two triangles).
        let holed: Vec<[u32; 3]> = indices[..2]
            .iter()
            .chain(indices[4..].iter())
            .copied()
            .collect();
        assert_eq!(holed.len(), 10);
        let mut baker = SdfBaker::new_with_context(&ctx).expect("baker");
        let err = baker
            .bake(
                &positions,
                &holed,
                SdfBakeTarget {
                    max_axis_res: 32,
                    min_voxel: 0.01,
                    pad: 1.0,
                    band_voxels: 4.0,
                },
            )
            .expect_err("holed mesh must fail the bake");
        match err {
            SdfBakeError::OpenMesh { min_abs_winding } => {
                println!("[sdf_bake] open mesh |w|={min_abs_winding:.3}");
                assert!(
                    min_abs_winding > 0.05 && min_abs_winding < 0.95,
                    "open-mesh winding must be fractional: {min_abs_winding}"
                );
            }
            other => panic!("expected OpenMesh, got {other:?}"),
        }
    }

    /// The speed finding: ~8k-tri mesh at 64³ — record the ms (the only
    /// assert is the sanity ceiling; CPU brute force at this res is minutes).
    #[test]
    fn sdf_bake_speed_64cubed() {
        let Some(ctx) = try_gpu_context("sdf_bake") else {
            return;
        };
        let (positions, indices) = uv_sphere(1.0, 64, 64);
        assert!(
            (7000..10000).contains(&indices.len()),
            "fixture should be ~8k tris, got {}",
            indices.len()
        );
        let mut baker = SdfBaker::new_with_context(&ctx).expect("baker");
        let start = std::time::Instant::now();
        let (field, _validation) = baker
            .bake(
                &positions,
                &indices,
                SdfBakeTarget {
                    max_axis_res: 64,
                    min_voxel: 0.01,
                    pad: 1.0,
                    band_voxels: 4.0,
                },
            )
            .expect("sphere must bake");
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let res = field.desc().resolution();
        println!(
            "[sdf_bake] 8k-tri 64^3 cook={ms:.1} ms ({} tris, {}x{}x{})",
            indices.len(),
            res[0],
            res[1],
            res[2]
        );
        assert_eq!(res, [64, 64, 64], "isotropic mesh + pad ⇒ 64³");
        assert!(ms < 10_000.0, "cook took {ms} ms — over the sanity ceiling");
    }
}
