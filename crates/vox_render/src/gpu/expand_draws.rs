//! `ExpandDrawsPass` — GPU expand into the persistent tiled buffers (M2 of
//! virtualized splat rendering).
//!
//! Takes the CPU [`InstancedSelector`](crate::atom_instances::InstancedSelector)
//! output (`Vec<ClusterDraw>` with prefix-summed `atom_offset`s) and runs one
//! compute thread per selected atom: each thread binary-searches its owning
//! draw, fetches the asset-LOCAL packed library atom through the cooked flat
//! tables, applies the instance transform (`pos' = iq * p + it`,
//! `quat' = iq * aq` — glam-exact math, see `expand_draws.wgsl`), and writes a
//! [`GpuSplatFull`] plus the two transform vec4s DIRECTLY into the tiled
//! chain's persistent `splat_buf` / `transform_buf` — the same
//! writes-into-`splat_buf` contract as [`crate::gpu::gi_combine`], on the same
//! shared [`GpuContext`], with zero renderer reconstruction.
//!
//! The flattened library (packed 64 B atoms + concatenated u32 index lists) is
//! uploaded ONCE at construction; per [`ExpandDrawsPass::encode`] only the draw
//! list (32 B/draw POD) and the instance transform pairs (2 × vec4/instance)
//! are re-uploaded. `conic` and `position_depth.w` are written as 0 — the
//! frozen `tile_assign` pass fills them in-device each frame.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::atom_instances::{AssetAtomLibrary, AtomInstance, DrawUnit, InstancedSelection};
use crate::gpu::GpuContext;
use crate::gpu::instanced_select_gpu::InstancedSelectGpu;
use crate::hierarchical_lod::LOD_LEVEL_COUNT;

/// Bytes per output splat slot (`GpuSplatFull`).
const SPLAT_SLOT_BYTES: u64 = 80;
/// Bytes per output transform pair (2 × vec4).
const TRANSFORM_SLOT_BYTES: u64 = 32;

const _: () = assert!(
    SPLAT_SLOT_BYTES as usize == std::mem::size_of::<crate::gpu::splat_buffer::GpuSplatFull>(),
    "expand_draws.wgsl GpuSplatFull must stay byte-identical to the Rust struct"
);

/// Error returned when the expand pass cannot be created. Mirrors
/// [`crate::gpu::atom_budget_gpu::AtomBudgetGpuError::ExceedsDeviceLimits`]:
/// limits are validated UP FRONT so wgpu never raises an uncaptured
/// validation error that aborts the process.
#[derive(Debug, Clone)]
pub enum ExpandDrawsError {
    /// A required GPU buffer would exceed a hard device limit
    /// (`max_storage_buffer_binding_size` / `max_buffer_size`). `what` names
    /// the offending buffer, `requested` the byte size, `limit` the cap.
    ExceedsDeviceLimits {
        what: &'static str,
        requested: u64,
        limit: u64,
    },
}

impl std::fmt::Display for ExpandDrawsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpandDrawsError::ExceedsDeviceLimits {
                what,
                requested,
                limit,
            } => write!(
                f,
                "GPU resource '{what}' requires {requested} bytes, exceeding device limit {limit}"
            ),
        }
    }
}

impl std::error::Error for ExpandDrawsError {}

/// `expand_draws.wgsl` binding-0 `ExpandParams` (16 B uniform).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ExpandParams {
    total: u32,
    draw_count: u32,
    _pad0: u32,
    _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<ExpandParams>() == 16);

/// `expand_draws.wgsl` `ExpandDraw` — 32 B POD, one per selected draw unit.
/// `pub(crate)` since M3.2: [`ExpandDrawsPass::lower`] returns it as the
/// draw-byte oracle for the GPU-resident walk's equality suite.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ExpandDraw {
    atom_offset: u32,
    atom_count: u32,
    index_offset: u32,
    atom_base: u32,
    instance: u32,
    opacity_scale: f32,
    _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<ExpandDraw>() == 32);

/// Host snapshot of one asset's flat-table ranges, taken from the library at
/// construction so [`ExpandDrawsPass::encode`] can resolve each
/// `ClusterDraw` to its `(index_offset, len)` without re-borrowing the library.
struct AssetRanges {
    atom_base: u32,
    /// `cluster_ranges[id][lod]` = `(offset, len)` into the index list.
    cluster_ranges: Vec<[(u32, u32); LOD_LEVEL_COUNT]>,
    /// `(offset, len)` for the I0 / I1 imposter units.
    imposter_ranges: [(u32, u32); 2],
}

/// GPU expand of an [`InstancedSelection`] into the tiled renderer's
/// persistent buffers. Library tables upload ONCE in
/// [`Self::new_with_context`]; per-frame [`Self::encode`] uploads only the
/// draw list and instance transforms, then dispatches
/// `ceil(total/256)` workgroups of 256 — one thread per selected atom,
/// disjoint output slots (the selector's prefix sums), deterministic contents.
///
/// One `encode` per submit: the draw/instance/params uploads go through
/// `Queue::write_buffer`, which lands before the NEXT submit — encoding twice
/// into unsubmitted encoders would clobber the first upload.
pub struct ExpandDrawsPass {
    ctx: GpuContext,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,

    // Library tables — uploaded once, never rewritten.
    atoms_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,

    // Per-encode uploads.
    draw_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,

    assets: Vec<AssetRanges>,
    max_instances: u32,
    budget_cap: u32,

    // Reused host scratch (the `&mut self` ownership model).
    draw_scratch: Vec<ExpandDraw>,
    xform_scratch: Vec<[f32; 4]>,
}

impl ExpandDrawsPass {
    /// Build the pass on the shared context: validate every buffer against
    /// `max_storage_buffer_binding_size` / `max_buffer_size` UP FRONT, upload
    /// the flattened library exactly once, and size the per-frame draw /
    /// instance buffers from `budget_cap` (a selection of `n` atoms holds at
    /// most `n` draws — every emitted draw carries ≥ 1 atom) and
    /// `max_instances`.
    ///
    /// Returns `Result` (unlike `GpuGi::new_with_context`) because the
    /// budget-derived sizes can genuinely exceed device limits — the
    /// documented [`ExpandDrawsError::ExceedsDeviceLimits`] class.
    pub fn new_with_context(
        ctx: &GpuContext,
        library: &AssetAtomLibrary,
        max_instances: u32,
        budget_cap: u32,
    ) -> Result<Self, ExpandDrawsError> {
        let device = ctx.device();

        let atoms_bytes = (library.packed_atoms().len() as u64 * 64).max(64);
        let index_bytes = (library.atom_index_list().len() as u64 * 4).max(4);
        let draws_bytes =
            (budget_cap as u64 * std::mem::size_of::<ExpandDraw>() as u64).max(32);
        let instances_bytes = (max_instances as u64 * TRANSFORM_SLOT_BYTES).max(32);

        let limits = device.limits();
        let max_storage = limits.max_storage_buffer_binding_size as u64;
        let max_buffer = limits.max_buffer_size;
        for (what, bytes) in [
            ("expand_lib_atoms", atoms_bytes),
            ("expand_atom_indices", index_bytes),
            ("expand_draws", draws_bytes),
            ("expand_instances", instances_bytes),
        ] {
            if bytes > max_storage {
                return Err(ExpandDrawsError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_storage,
                });
            }
            if bytes > max_buffer {
                return Err(ExpandDrawsError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_buffer,
                });
            }
        }

        // ── Library upload (ONCE) ───────────────────────────────────────────
        let atoms_buf = if library.packed_atoms().is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("expand_lib_atoms"),
                size: atoms_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("expand_lib_atoms"),
                contents: bytemuck::cast_slice(library.packed_atoms()),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let index_buf = if library.atom_index_list().is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("expand_atom_indices"),
                size: index_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("expand_atom_indices"),
                contents: bytemuck::cast_slice(library.atom_index_list()),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        // ── Per-frame buffers ───────────────────────────────────────────────
        let draw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("expand_draws"),
            size: draws_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("expand_instances"),
            size: instances_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("expand_params"),
            size: std::mem::size_of::<ExpandParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Range snapshot — resolves draws without the library at encode ───
        let assets = (0..library.asset_count() as u32)
            .map(|a| AssetRanges {
                atom_base: library.asset_atom_base(a),
                cluster_ranges: (0..library.cluster_count(a) as u32)
                    .map(|id| {
                        std::array::from_fn(|lod| {
                            library.unit_range(
                                a,
                                DrawUnit::Cluster {
                                    id,
                                    lod: lod as u8,
                                },
                            )
                        })
                    })
                    .collect(),
                imposter_ranges: [
                    library.unit_range(a, DrawUnit::Imposter { level: 0 }),
                    library.unit_range(a, DrawUnit::Imposter { level: 1 }),
                ],
            })
            .collect();

        // ── Pipeline ────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("expand_draws_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("expand_draws.wgsl").into()),
        });

        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("expand_draws_bgl"),
            entries: &[
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
                storage_entry(1, true),  // lib atoms
                storage_entry(2, true),  // index list
                storage_entry(3, true),  // draws
                storage_entry(4, true),  // instance transforms
                storage_entry(5, false), // out splats (the renderer's splat_buf)
                storage_entry(6, false), // out transforms (transform_buf)
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("expand_draws_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("expand_draws_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            ctx: ctx.clone(),
            pipeline,
            bgl,
            atoms_buf,
            index_buf,
            draw_buf,
            instance_buf,
            params_buf,
            assets,
            max_instances,
            budget_cap,
            draw_scratch: Vec::new(),
            xform_scratch: Vec::new(),
        })
    }

    /// Lower an [`InstancedSelection`] into the `ExpandDraw` POD array
    /// `encode` uploads — the EXACT loop extracted from `encode` (M3.2 Task 1,
    /// zero behavior change; one lowering, two consumers: `encode`'s upload
    /// and the GPU-resident walk's draw-byte equality oracle). Returns the
    /// lowered slice (in `self.draw_scratch`) and the expanded atom total,
    /// with `encode`'s exact defensive truncation: draws past `budget_cap`,
    /// referencing unknown assets/instances, or disagreeing with the cooked
    /// ranges truncate the prefix.
    pub(crate) fn lower(
        &mut self,
        selection: &InstancedSelection,
        instance_count: usize,
    ) -> (&[ExpandDraw], u32) {
        self.draw_scratch.clear();
        let mut total = 0u32;
        for d in selection.draws() {
            if d.atom_count == 0 {
                continue;
            }
            let end = d.atom_offset.saturating_add(d.atom_count);
            if d.atom_offset != total || end > self.budget_cap {
                break;
            }
            if d.instance >= self.max_instances || (d.instance as usize) >= instance_count {
                break;
            }
            let Some(asset) = self.assets.get(d.asset as usize) else {
                break;
            };
            let (index_offset, len) = match d.unit {
                DrawUnit::Cluster { id, lod } => {
                    let Some(ranges) = asset.cluster_ranges.get(id as usize) else {
                        break;
                    };
                    let Some(&r) = ranges.get(lod as usize) else {
                        break;
                    };
                    r
                }
                DrawUnit::Imposter { level } => {
                    let Some(&r) = asset.imposter_ranges.get(level as usize) else {
                        break;
                    };
                    r
                }
            };
            if len != d.atom_count {
                break;
            }
            self.draw_scratch.push(ExpandDraw {
                atom_offset: d.atom_offset,
                atom_count: d.atom_count,
                index_offset,
                atom_base: asset.atom_base,
                instance: d.instance,
                opacity_scale: d.opacity_scale,
                _pad: [0; 2],
            });
            total = end;
        }
        (&self.draw_scratch, total)
    }

    /// Record the expand into `encoder`: resolve each draw to its flat
    /// `(index_offset, len)` on the host, upload the POD draw array + the
    /// instance transform pairs, and dispatch one thread per selected atom
    /// writing `splat_buf[atom_offset + k]` / `transform_buf[(atom_offset+k)*2..]`
    /// — the renderer's REAL persistent buffers, bound read-write. The caller
    /// submits, then calls
    /// [`set_active_splat_count`](crate::gpu::tiled_splat_renderer::TiledSplatRenderer::set_active_splat_count)
    /// with the returned count and renders.
    ///
    /// Returns the expanded atom count — `selection.atom_count()` for every
    /// well-formed selection. Defensive (never panics): draws past
    /// `budget_cap`, referencing unknown assets/instances, or disagreeing with
    /// the cooked ranges TRUNCATE the prefix (slots stay disjoint and
    /// contiguous from 0, so the returned count is always safe to render).
    pub fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        selection: &InstancedSelection,
        instances: &[AtomInstance],
        splat_buf: &wgpu::Buffer,
        transform_buf: &wgpu::Buffer,
    ) -> u32 {
        let total = self.lower(selection, instances.len()).1;

        if total == 0 || self.draw_scratch.is_empty() {
            return 0;
        }

        let queue = self.ctx.queue();
        queue.write_buffer(&self.draw_buf, 0, bytemuck::cast_slice(&self.draw_scratch));

        self.xform_scratch.clear();
        for inst in instances.iter().take(self.max_instances as usize) {
            self.xform_scratch
                .push([inst.position[0], inst.position[1], inst.position[2], 0.0]);
            self.xform_scratch.push(inst.rotation);
        }
        queue.write_buffer(
            &self.instance_buf,
            0,
            bytemuck::cast_slice(&self.xform_scratch),
        );

        let params = ExpandParams {
            total,
            draw_count: self.draw_scratch.len() as u32,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // Per-encode bind group: the output buffers belong to the renderer and
        // can vary between calls (the gi_combine per-dispatch pattern).
        let device = self.ctx.device();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("expand_draws_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.atoms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.index_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.draw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.instance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: splat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: transform_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("expand_draws_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(total.div_ceil(256), 1, 1);

        total
    }

    /// Record the GPU-resident expand into `encoder` (M3.2 Task 2): bind the
    /// `ExpandDraw` buffer and `ExpandParams` that
    /// [`InstancedSelectGpu::select_resident`] emitted ON DEVICE and dispatch
    /// via `dispatch_workgroups_indirect` off `src.expand_args_buf()` — the
    /// host never sees scores, WorkUnits, or draws, and `expand_draws.wgsl`
    /// is unchanged (its binding layout already fits: binding 0 =
    /// `src.expand_params_buf()` as the 16 B uniform, binding 3 =
    /// `src.draws_buf()`). Only the instance transform pairs are uploaded,
    /// exactly as [`Self::encode`] does (the recorded ~320 KB/frame at 10k
    /// instances is measured noise — do not "fix").
    ///
    /// Caller contract: the `select_resident` `budget` must be
    /// `<= budget_cap` (the destination `splat_buf` / `transform_buf` must
    /// hold every emitted atom slot), and `src` must be built over the SAME
    /// [`AssetAtomLibrary`] as this pass — the device-emitted draws'
    /// `index_offset` / `atom_base` index this pass's once-uploaded flat
    /// tables. On [`SelectPath::CpuFallback`](crate::gpu::instanced_select_gpu::SelectPath)
    /// (or an empty instance set) `select_resident` zeroed the indirect args
    /// and `ExpandParams`, so this dispatch is an honest no-op (0 workgroups,
    /// 0 atoms) — route the fallback selection through the host
    /// [`Self::encode`] instead. The expanded atom count comes from the
    /// select stats (`stats.selected`), not a return value — there is no
    /// host-visible draw list to count.
    pub fn encode_indirect(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        src: &InstancedSelectGpu,
        instances: &[AtomInstance],
        splat_buf: &wgpu::Buffer,
        transform_buf: &wgpu::Buffer,
    ) {
        let queue = self.ctx.queue();

        self.xform_scratch.clear();
        for inst in instances.iter().take(self.max_instances as usize) {
            self.xform_scratch
                .push([inst.position[0], inst.position[1], inst.position[2], 0.0]);
            self.xform_scratch.push(inst.rotation);
        }
        if !self.xform_scratch.is_empty() {
            queue.write_buffer(
                &self.instance_buf,
                0,
                bytemuck::cast_slice(&self.xform_scratch),
            );
        }

        // Per-encode bind group, as `encode`: the output buffers belong to the
        // renderer and the draws/params to the selector — both caller-owned.
        // `src.expand_args_buf()` is NOT bound (the k2_args usage rule:
        // indirect-args buffers never appear in their own dispatch's layout).
        let device = self.ctx.device();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("expand_draws_indirect_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.expand_params_buf().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.atoms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.index_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: src.draws_buf().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.instance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: splat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: transform_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("expand_draws_indirect_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups_indirect(src.expand_args_buf(), 0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// In-module tests — M2 of the virtualized splat rendering plan: the GPU expand
// must reproduce the CPU transform oracle to < 1e-5 over ≥ 100k atoms, carry
// the LOD crossfade through opacity_color.w, and serve two frames through ONE
// `new_with_capacity` renderer with the expand pass as the only buffer writer.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom_instances::{AssetAtomLibrary, AtomInstance, InstancedSelector};
    use crate::gpu::instanced_select_gpu::{InstancedSelectGpu, SelectPath};
    use crate::gpu::splat_buffer::GpuSplatFull;
    use crate::gpu::tiled_splat_renderer::TiledSplatRenderer;
    use crate::spectral::RenderCamera;
    use glam::{Mat4, Quat, Vec3};
    use half::f16;
    use std::sync::Arc;
    use vox_core::spectral::Illuminant;
    use vox_core::types::GaussianSplat;

    const W: u32 = 320;
    const H: u32 = 180;

    /// House `try_gpu` pattern (shape from `tiled_splat_renderer.rs` /
    /// `atom_budget_gpu.rs`): hardware adapter or printed skip + green exit.
    fn try_gpu_context(label: &str) -> Option<GpuContext> {
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
                label: Some("expand_draws_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("device creation on a box with a GPU");
        Some(GpuContext::from_parts(&device, &queue, &info))
    }

    /// splitmix64-style scramble (house pattern, same as `atom_instances`).
    fn hash_u64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    fn hash01(a: u64, b: u64) -> f32 {
        let h = hash_u64(a ^ hash_u64(b.wrapping_mul(0x100_0000_01B3)));
        (h >> 40) as f32 / (1u64 << 24) as f32
    }

    fn spectral_level(level: f32) -> [u16; 16] {
        std::array::from_fn(|i| {
            let v = if i < 8 { level } else { level * 0.55 };
            f16::from_f32(v).to_bits()
        })
    }

    /// Deterministic hash-jittered blob asset (cylinder `radius` × `height`
    /// rooted at the local origin), with NON-identity per-atom rotations so the
    /// quat product `iq * aq` is exercised for real.
    fn synthetic_asset(seed: u64, n: usize, radius: f32, height: f32) -> Vec<GaussianSplat> {
        (0..n)
            .map(|i| {
                let ii = i as u64;
                let ang = hash01(seed ^ ii, 10) * std::f32::consts::TAU;
                let rr = hash01(seed ^ ii, 11).sqrt() * radius;
                let hh = hash01(seed ^ ii, 12) * height;
                let op = 160 + (hash_u64(seed ^ ii.wrapping_mul(2_654_435_761)) % 90) as u8;
                let level = 0.35 + hash01(seed ^ ii, 13) * 0.5;
                let rot = Quat::from_euler(
                    glam::EulerRot::YXZ,
                    hash01(seed ^ ii, 14) * std::f32::consts::TAU,
                    hash01(seed ^ ii, 15) * 0.8,
                    0.0,
                );
                GaussianSplat::volume(
                    [rr * ang.cos(), hh, rr * ang.sin()],
                    [0.25, 0.25, 0.25],
                    rot,
                    op,
                    spectral_level(level),
                )
            })
            .collect()
    }

    fn camera(eye: Vec3, target: Vec3, fovy: f32) -> RenderCamera {
        RenderCamera {
            view: Mat4::look_at_rh(eye, target, Vec3::Y),
            proj: Mat4::perspective_rh(fovy, W as f32 / H as f32, 0.1, 2000.0),
        }
    }

    /// The M3 city geometry (shape copied from
    /// `instanced_select_gpu.rs::tests::city_scene` — reuse, don't re-derive):
    /// 4 × 5,000-atom synthetic assets, 10,000 instances on the 100×100 × 4 m
    /// grid with quarter-turn yaws.
    fn city_scene() -> (Arc<AssetAtomLibrary>, Vec<AtomInstance>) {
        let assets: Vec<Vec<GaussianSplat>> = (0..4)
            .map(|a| synthetic_asset(0xC17 + a as u64, 5000, 4.0, 12.0))
            .collect();
        let lib = Arc::new(AssetAtomLibrary::build(&assets, 128));
        let instances: Vec<AtomInstance> = (0..10_000u32)
            .map(|i| {
                let gx = (i % 100) as f32 - 49.5;
                let gz = (i / 100) as f32 - 49.5;
                let q = Quat::from_rotation_y((i % 4) as f32 * std::f32::consts::FRAC_PI_2);
                AtomInstance::new(i % 4, i, [gx * 4.0, 0.0, gz * 4.0], [q.x, q.y, q.z, q.w])
            })
            .collect();
        (lib, instances)
    }

    /// Copy `bytes` out of `buf` at `offset` and return them (blocking, proof-
    /// only readback).
    fn read_back(ctx: &GpuContext, buf: &wgpu::Buffer, offset: u64, bytes: u64) -> Vec<u8> {
        let device = ctx.device();
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("expand_test_readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("expand_test_readback_copy"),
        });
        enc.copy_buffer_to_buffer(buf, offset, &readback, 0, bytes);
        ctx.queue().submit(Some(enc.finish()));
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        assert!(matches!(rx.recv(), Ok(Ok(()))), "readback map failed");
        let data = slice.get_mapped_range().to_vec();
        readback.unmap();
        data
    }

    /// ≥ 100k atoms expanded on-device must match the CPU oracle
    /// (`pos' = iq * p + it`, `quat' = iq * aq` from `decoded_rotation()`,
    /// opacity × `opacity_scale`) to < 1e-5 absolute, with the spectral bins
    /// BIT-equal to the f16 round-trip and the device-filled fields zeroed —
    /// then the same buffers render through the frozen tiled chain.
    #[test]
    fn expand_matches_cpu_oracle() {
        let Some(ctx) = try_gpu_context("expand") else {
            return;
        };

        let assets: Vec<Vec<GaussianSplat>> = (0..5)
            .map(|a| synthetic_asset(0xE84A + a as u64, 5000, 4.0, 12.0))
            .collect();
        // target 512 → fewer, larger clusters (~2.8 m radius) so the whole
        // 8×8 grid sits at L1 from this camera: 64 × 5000 × 0.4 ≈ 128k atoms.
        let lib = Arc::new(AssetAtomLibrary::build(&assets, 512));

        let instances: Vec<AtomInstance> = (0..64u32)
            .map(|i| {
                let gx = (i % 8) as f32 - 3.5;
                let gz = (i / 8) as f32 - 3.5;
                let q = Quat::from_rotation_y(i as f32 * 0.37);
                AtomInstance::new(i % 5, i, [gx * 5.0, 0.0, gz * 5.0], [q.x, q.y, q.z, q.w])
            })
            .collect();
        let cam = camera(Vec3::new(0.0, 18.0, 28.0), Vec3::new(0.0, 5.0, 0.0), 1.4);

        const BUDGET: u32 = 150_000;
        let mut renderer = TiledSplatRenderer::new_with_capacity(ctx.clone(), BUDGET, W, H)
            .expect("150k capacity fits default device limits");
        let mut pass = ExpandDrawsPass::new_with_context(&ctx, &lib, 64, BUDGET)
            .expect("expand pass fits default device limits");

        let mut selector = InstancedSelector::new(lib.clone());
        selector.set_instances(&instances);
        let mut sel = InstancedSelection::new();
        let stats = selector.select(&cam, BUDGET as usize, &mut sel);
        let total = sel.atom_count();
        assert!(
            total >= 100_000,
            "oracle scene must select >= 100k atoms (selected {} of {} visible instances)",
            total,
            stats.instances_visible
        );

        let mut enc = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("expand_oracle_encode"),
            });
        let n = pass.encode(
            &mut enc,
            &sel,
            &instances,
            renderer.splat_buf(),
            renderer.transform_buf(),
        );
        ctx.queue().submit(Some(enc.finish()));
        assert_eq!(n as usize, total, "encode must expand the whole selection");

        // Read back BEFORE rendering — tile_assign rewrites depth/conic in place.
        let splat_bytes = read_back(&ctx, renderer.splat_buf(), 0, n as u64 * SPLAT_SLOT_BYTES);
        let splats: &[GpuSplatFull] = bytemuck::cast_slice(&splat_bytes);
        let xform_bytes = read_back(
            &ctx,
            renderer.transform_buf(),
            0,
            n as u64 * TRANSFORM_SLOT_BYTES,
        );
        let xforms: &[[f32; 4]] = bytemuck::cast_slice(&xform_bytes);

        let packed = lib.packed_atoms();
        let mut max_dev = 0.0f32;
        let mut spectral_checked = 0usize;
        for d in sel.draws() {
            let inst = &instances[d.instance as usize];
            let iq = Quat::from_xyzw(
                inst.rotation[0],
                inst.rotation[1],
                inst.rotation[2],
                inst.rotation[3],
            );
            let it = Vec3::from(inst.position);
            let base = lib.asset_atom_base(d.asset);
            let indices = lib.unit_indices(d.asset, d.unit);
            assert_eq!(indices.len() as u32, d.atom_count);
            for (k, &li) in indices.iter().enumerate() {
                let slot = d.atom_offset as usize + k;
                let a = &packed[(base + li) as usize];
                let apos = Vec3::new(a.pos_opacity[0], a.pos_opacity[1], a.pos_opacity[2]);
                let aq = Quat::from_xyzw(a.quat[0], a.quat[1], a.quat[2], a.quat[3]);
                // The pinned expand math (glam::Quat::mul_vec3 / mul_quat,
                // instance applied AFTER atom, atom quat already decoded).
                let want_pos = iq.mul_vec3(apos) + it;
                let want_quat = iq * aq;
                let want_opacity = a.pos_opacity[3] * d.opacity_scale;

                let got = &splats[slot];
                for c in 0..3 {
                    max_dev = max_dev.max((got.position_depth[c] - want_pos[c]).abs());
                }
                // Device-filled fields must arrive zeroed for tile_assign.
                max_dev = max_dev.max(got.position_depth[3].abs());
                for c in 0..3 {
                    max_dev = max_dev.max(got.conic[c].abs());
                    max_dev = max_dev.max(got.opacity_color[c].abs());
                }
                max_dev = max_dev.max((got.opacity_color[3] - want_opacity).abs());

                let t0 = xforms[slot * 2];
                let t1 = xforms[slot * 2 + 1];
                for c in 0..3 {
                    max_dev = max_dev.max((t0[c] - a.scale[c]).abs());
                }
                max_dev = max_dev.max(t0[3].abs());
                let wq = [want_quat.x, want_quat.y, want_quat.z, want_quat.w];
                for c in 0..4 {
                    max_dev = max_dev.max((t1[c] - wq[c]).abs());
                }

                for b in 0..8 {
                    let bits = (a.spectral[b / 2] >> ((b % 2) * 16)) as u16;
                    let want = f16::from_bits(bits).to_f32();
                    assert_eq!(
                        got.spectral[b].to_bits(),
                        want.to_bits(),
                        "slot {slot} bin {b}: spectral must be bit-equal to the f16 round-trip \
                         (got {} want {want})",
                        got.spectral[b]
                    );
                    spectral_checked += 1;
                }
            }
        }

        println!("[expand] atoms={n} max_abs_dev={max_dev:e}");
        assert!(
            max_dev < 1e-5,
            "GPU expand deviates {max_dev:e} (>= 1e-5) from the CPU oracle"
        );

        // Drive the full seam on the SAME shared context: expand was the only
        // splat-buffer writer; the frozen chain must light pixels from it.
        renderer.set_active_splat_count(n);
        let frame = renderer.render(&cam).expect("render after expand");
        let (_, non_black) = frame.resolve_to_srgb(&ctx, &Illuminant::d65());
        assert!(
            non_black > 0,
            "expanded splats must rasterize ({non_black} non-black pixels)"
        );
        println!(
            "[expand] oracle seam: non_black={non_black} spectral_bins_bit_equal={spectral_checked}"
        );
    }

    /// A single-cluster asset placed so the cluster centroid sits ~140 m out
    /// (the L1 130–150 m crossfade band): the draw's fractional opacity_scale
    /// must come back through `opacity_color.w` as `full_opacity × scale`,
    /// strictly between 0 and the full opacity.
    #[test]
    fn expand_carries_crossfade() {
        let Some(ctx) = try_gpu_context("expand") else {
            return;
        };

        // target_cluster_size ≥ atom count ⇒ clas cooks exactly ONE cluster;
        // its AABB half-diagonal (≈ 17.7 m) keeps the projected size ≥ 50 px
        // at 140 m, which holds the distance LOD at L1 — whose crossfade band
        // is exactly 130–150 m.
        let atoms = synthetic_asset(0xFADE, 2000, 12.0, 10.0);
        let lib = Arc::new(AssetAtomLibrary::build(std::slice::from_ref(&atoms), 4096));
        let instances = [AtomInstance::new(
            0,
            0,
            [0.0, 0.0, 140.0],
            [0.0, 0.0, 0.0, 1.0],
        )];

        let cam = camera(Vec3::ZERO, Vec3::new(0.0, 5.0, 140.0), 0.9);
        let mut selector = InstancedSelector::new(lib.clone());
        selector.set_instances(&instances);
        let mut sel = InstancedSelection::new();
        let stats = selector.select(&cam, usize::MAX, &mut sel);
        assert_eq!(stats.instances_far, 0, "140 m instance must classify NEAR");
        // Atoms exactly on the AABB max face spill into corner grid cells, so
        // clas cooks a dominant cluster plus a few one-atom satellites — the
        // dominant draw is the only one big enough to sit in the L1 band.
        let d = *sel
            .draws()
            .iter()
            .max_by_key(|d| d.atom_count)
            .expect("selection must emit draws");
        assert!(
            d.atom_count > 100,
            "dominant draw must carry the cluster body, got {} atoms",
            d.atom_count
        );
        assert!(
            matches!(d.unit, DrawUnit::Cluster { lod: 1, .. }),
            "dominant cluster at ~140 m must sit at L1 (its crossfade band), got {:?}",
            d.unit
        );
        assert!(
            d.opacity_scale > 0.0 && d.opacity_scale < 1.0,
            "selector must emit a fractional crossfade scale, got {}",
            d.opacity_scale
        );

        let renderer = TiledSplatRenderer::new_with_capacity(ctx.clone(), 2048, W, H)
            .expect("2k capacity fits");
        let mut pass =
            ExpandDrawsPass::new_with_context(&ctx, &lib, 1, 2048).expect("expand pass fits");
        let mut enc = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("expand_crossfade_encode"),
            });
        let n = pass.encode(
            &mut enc,
            &sel,
            &instances,
            renderer.splat_buf(),
            renderer.transform_buf(),
        );
        ctx.queue().submit(Some(enc.finish()));
        assert_eq!(n as usize, sel.atom_count());
        assert!(n >= d.atom_count);

        // Read back exactly the dominant draw's output slot range.
        let bytes = read_back(
            &ctx,
            renderer.splat_buf(),
            d.atom_offset as u64 * SPLAT_SLOT_BYTES,
            d.atom_count as u64 * SPLAT_SLOT_BYTES,
        );
        let splats: &[GpuSplatFull] = bytemuck::cast_slice(&bytes);
        let li = lib.unit_indices(0, d.unit)[0];
        let full = lib.packed_atoms()[(lib.asset_atom_base(0) + li) as usize].pos_opacity[3];
        let o = splats[0].opacity_color[3];
        println!("[expand] crossfade opacity={o} full={full}");
        assert!(
            o > 0.0 && o < full,
            "crossfaded opacity must be strictly between 0 and full ({o} vs {full})"
        );
        assert!(
            (o - full * d.opacity_scale).abs() < 1e-6,
            "opacity must be full × opacity_scale ({} expected, got {o})",
            full * d.opacity_scale
        );
    }

    /// ONE `new_with_capacity` renderer + ONE expand pass serve two frames
    /// from two cameras with the expand pass as the ONLY splat-buffer writer;
    /// a tail-slot sentinel survives both frames bit-exactly (same persistent
    /// buffer, no rebuild, nothing written outside the active range).
    #[test]
    fn expand_two_frames_no_rebuild() {
        let Some(ctx) = try_gpu_context("expand") else {
            return;
        };

        let assets: Vec<Vec<GaussianSplat>> = (0..3)
            .map(|a| synthetic_asset(0x2F00 + a as u64, 3000, 4.0, 12.0))
            .collect();
        let lib = Arc::new(AssetAtomLibrary::build(&assets, 256));
        let instances: Vec<AtomInstance> = (0..25u32)
            .map(|i| {
                let gx = (i % 5) as f32 - 2.0;
                let gz = (i / 5) as f32 - 2.0;
                let q = Quat::from_rotation_y(i as f32 * 0.61);
                AtomInstance::new(i % 3, i, [gx * 10.0, 0.0, gz * 10.0], [q.x, q.y, q.z, q.w])
            })
            .collect();

        const CAPACITY: u32 = 64_000;
        const BUDGET: usize = 60_000;
        // Constructed ONCE — the whole point of capacity + expand (M1/M2).
        let mut renderer = TiledSplatRenderer::new_with_capacity(ctx.clone(), CAPACITY, W, H)
            .expect("64k capacity fits");
        let mut pass = ExpandDrawsPass::new_with_context(&ctx, &lib, 25, CAPACITY)
            .expect("expand pass fits");
        let mut selector = InstancedSelector::new(lib.clone());
        selector.set_instances(&instances);

        // Sentinel bit pattern in the LAST capacity slot, written before
        // frame 1, read back after frame 2 (the Task-3 technique — but here
        // the GPU expand pass, not the host, writes every active slot).
        let sentinel: [f32; 20] = std::array::from_fn(|i| -77.5 - 3.25 * i as f32);
        let tail_off = (CAPACITY as u64 - 1) * SPLAT_SLOT_BYTES;
        ctx.queue()
            .write_buffer(renderer.splat_buf(), tail_off, bytemuck::cast_slice(&sentinel));

        let cams = [
            camera(Vec3::new(0.0, 20.0, 38.0), Vec3::new(0.0, 5.0, 0.0), 1.2),
            camera(Vec3::new(16.0, 26.0, -34.0), Vec3::new(0.0, 5.0, 0.0), 1.2),
        ];
        let mut non_black = [0usize; 2];
        let mut sel = InstancedSelection::new();
        for (f, cam) in cams.iter().enumerate() {
            selector.select(cam, BUDGET, &mut sel);
            assert!(
                sel.atom_count() > 0,
                "frame {f}: selection must be non-empty"
            );
            let mut enc = ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("expand_two_frames_encode"),
                });
            let n = pass.encode(
                &mut enc,
                &sel,
                &instances,
                renderer.splat_buf(),
                renderer.transform_buf(),
            );
            ctx.queue().submit(Some(enc.finish()));
            assert_eq!(n as usize, sel.atom_count());
            renderer.set_active_splat_count(n);
            let frame = renderer.render(cam).expect("render");
            let (_, nb) = frame.resolve_to_srgb(&ctx, &Illuminant::d65());
            non_black[f] = nb;
        }

        let bytes = read_back(&ctx, renderer.splat_buf(), tail_off, SPLAT_SLOT_BYTES);
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let intact = got.len() == sentinel.len()
            && got
                .iter()
                .zip(sentinel.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());

        assert!(non_black[0] > 0, "frame 1 must light pixels");
        assert!(non_black[1] > 0, "frame 2 must light pixels");
        assert!(
            intact,
            "tail-slot sentinel must survive both expand+render frames bit-exactly"
        );
        println!(
            "[expand] frame1 non_black={} frame2 non_black={} sentinel=intact constructions=1",
            non_black[0], non_black[1]
        );
    }

    /// THE M3.2 TASK 2 SEAM: the same city selection driven through (A) the
    /// host path — CPU `InstancedSelector::select` → `encode` — and (B) the
    /// resident path — `select_resident` → `encode_indirect` (the exact order
    /// the Task 4 harness uses) — must leave `splat_buf` / `transform_buf`
    /// BYTE-equal over every expanded atom, and the indirect renderer must
    /// then light pixels through the frozen tiled chain. The host never sees
    /// path B's draws: they stay in `src.draws_buf()`, consumed via
    /// `dispatch_workgroups_indirect`.
    #[test]
    fn resident_expand_indirect_matches_host_encode() {
        let Some(ctx) = try_gpu_context("resident_expand") else {
            return;
        };
        let (lib, instances) = city_scene();
        // The M3 oblique aerial city camera (aspect 1.0, verbatim from
        // `instanced_select_gpu.rs::tests::city_cameras`).
        let cam = RenderCamera {
            view: Mat4::look_at_rh(Vec3::new(80.0, 80.0, 80.0), Vec3::ZERO, Vec3::Y),
            proj: Mat4::perspective_rh(1.2, 1.0, 0.1, 2000.0),
        };
        const BUDGET: u32 = 250_000;

        // ONE pass serves both paths — same pipeline, same once-uploaded
        // library tables; only the draw/params sources differ.
        let mut pass =
            ExpandDrawsPass::new_with_context(&ctx, &lib, instances.len() as u32, BUDGET)
                .expect("expand pass fits default device limits");

        // ── Path A: host encode off the CPU selector ─────────────────────────
        let mut selector = InstancedSelector::new(lib.clone());
        selector.set_instances(&instances);
        let mut sel = InstancedSelection::new();
        selector.select(&cam, BUDGET as usize, &mut sel);
        let total = sel.atom_count();
        assert!(total > 0, "city selection must be non-trivial");

        let host_renderer = TiledSplatRenderer::new_with_capacity(ctx.clone(), BUDGET, W, H)
            .expect("250k capacity fits default device limits");
        let mut enc = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident_expand_host_encode"),
            });
        let n = pass.encode(
            &mut enc,
            &sel,
            &instances,
            host_renderer.splat_buf(),
            host_renderer.transform_buf(),
        );
        ctx.queue().submit(Some(enc.finish()));
        assert_eq!(n as usize, total, "host encode must take the whole selection");
        // Read back BEFORE any render — tile_assign rewrites depth/conic in place.
        let host_splats = read_back(
            &ctx,
            host_renderer.splat_buf(),
            0,
            n as u64 * SPLAT_SLOT_BYTES,
        );
        let host_xforms = read_back(
            &ctx,
            host_renderer.transform_buf(),
            0,
            n as u64 * TRANSFORM_SLOT_BYTES,
        );

        // ── Path B: GPU-resident draws, no host round trip ───────────────────
        let mut gpu = InstancedSelectGpu::new_with_context(
            &ctx,
            lib.clone(),
            instances.len() as u32,
            262_144,
        )
        .expect("city scene fits default device limits");
        gpu.set_instances(&instances);
        let mut fb = InstancedSelection::new();
        let (stats, path) = gpu
            .select_resident(&cam, BUDGET as usize, &mut fb)
            .expect("resident select");
        assert_eq!(path, SelectPath::GpuResident, "city must stay on the GPU path");
        assert_eq!(
            stats.selected, total,
            "resident select must agree with the CPU selector on the atom total"
        );

        let mut ind_renderer = TiledSplatRenderer::new_with_capacity(ctx.clone(), BUDGET, W, H)
            .expect("250k capacity fits default device limits");
        let mut enc = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident_expand_indirect_encode"),
            });
        pass.encode_indirect(
            &mut enc,
            &gpu,
            &instances,
            ind_renderer.splat_buf(),
            ind_renderer.transform_buf(),
        );
        ctx.queue().submit(Some(enc.finish()));

        let ind_splats = read_back(
            &ctx,
            ind_renderer.splat_buf(),
            0,
            stats.selected as u64 * SPLAT_SLOT_BYTES,
        );
        let ind_xforms = read_back(
            &ctx,
            ind_renderer.transform_buf(),
            0,
            stats.selected as u64 * TRANSFORM_SLOT_BYTES,
        );
        assert!(
            host_splats == ind_splats,
            "indirect expand splat_buf bytes diverge from the host encode"
        );
        assert!(
            host_xforms == ind_xforms,
            "indirect expand transform_buf bytes diverge from the host encode"
        );

        // The GPU-written ExpandParams the indirect dispatch consumed.
        let pbytes = read_back(&ctx, gpu.expand_params_buf(), 0, 16);
        let p: &[u32] = bytemuck::cast_slice(&pbytes);
        assert_eq!(p[0] as usize, total, "ExpandParams.total");
        let draws = p[1];
        assert!(draws > 0, "city selection must emit draws");
        println!(
            "[resident_expand] indirect seam: splat_buf+transform_buf byte-equal to host encode over {total} atoms | draws={draws}"
        );

        // The full Task-4 harness order, finished: render off the indirect
        // expand on the SAME shared context.
        ind_renderer.set_active_splat_count(stats.selected as u32);
        let frame = ind_renderer.render(&cam).expect("render after indirect expand");
        let (_, non_black) = frame.resolve_to_srgb(&ctx, &Illuminant::d65());
        assert!(
            non_black > 0,
            "indirect-expanded city must rasterize ({non_black} non-black pixels)"
        );
        println!("[resident_expand] indirect render: non_black={non_black}");
    }
}
