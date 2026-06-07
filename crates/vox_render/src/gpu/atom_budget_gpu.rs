//! Atom-budget LOD selector — GPU compute port of the CPU oracle
//! [`crate::atom_budget::AtomBudgetSelector`] ("Nanite for splats").
//!
//! GPU is the foundation; the CPU [`crate::atom_budget::AtomBudgetSelector::select`]
//! is the correctness ORACLE this mirrors. The output is a discrete
//! [`crate::atom_budget::Selection`] (chosen splat indices + crossfade opacity
//! multipliers); exactness — same clusters, same LODs, same totals — is the
//! contract, so this port reproduces the oracle's `Selection` BYTE-FOR-BYTE.
//!
//! Follows the [`crate::gpu::many_light_gpu::ManyLightGpu`] /
//! [`crate::gpu::splat_rt_gpu::SplatRtGpu`] house pattern: construct-with-its-own-
//! device, adapter-gated (never panics on a missing GPU — returns
//! [`AtomBudgetGpuError::NoAdapter`]), measured-then-asserted tolerances, bit-
//! identical determinism, and the [`AtomBudgetGpuError::ExceedsDeviceLimits`]
//! buffer validation that wave-13 made the contract.
//!
//! ## Self-owned scene data
//!
//! The CPU oracle keeps its `clusters` / `bvh` / `lods` private and read-only, so
//! this GPU port builds its OWN copy from the same splat array using the same
//! `pub` builders (`build_clusters`, `build_cluster_bvh`) and the same
//! opacity-sorted LOD prefixes the oracle uses internally — exactly as the other
//! GPU modules own their own scene data rather than borrowing the oracle's
//! internals. Built once via [`AtomBudgetGpu::build_scene`]; the oracle remains
//! the validation reference (tests call `oracle.select()` and compare).
//!
//! ## The GPU / host split (and why)
//!
//! The flagship insight is that the per-cluster SCORING — projected solid angle
//! × opacity × view-angle, plus the distance-driven LOD and the leaf frustum
//! test — is embarrassingly parallel and rounding-sensitive, while the BUDGET
//! SELECTION (demote lowest-score, promote highest-score, shed under pressure)
//! is a tiny sequential heap walk over a few HUNDRED visible clusters.
//!
//! So:
//!
//! * **GPU, one thread per cluster** (`atom_budget_gpu.wgsl`): computes, in the
//!   EXACT op order of the oracle, `radius`, scoring `distance` (centroid-based),
//!   `screen`, `distance_lod`, `score`, and a leaf-level `passes_frustum` flag.
//!   This is the heavy "Nanite scoring" core; no fast-math means RADV rounds
//!   identically to the CPU (the BARY_EPS lesson, commit dae84d8).
//! * **Host**: walks the cluster BVH structurally to know which leaves are even
//!   reachable (internal-node pruning — pointer-chasing a GPU thread would only
//!   diverge on), intersects that with the GPU `passes_frustum` flags to
//!   reproduce the oracle's `visible` set EXACTLY, then runs the oracle's own
//!   demote/promote/shed sequence over the GPU-computed scores and emits the
//!   identical [`Selection`]. A GPU-side sort/shed buys nothing here and risks
//!   divergence, so it stays on the host.
//!
//! The host budget logic is mirrored 1:1 from the oracle's `select()` body (its
//! shedding logic is inline there, not a `pub(crate)` helper, so it is
//! reproduced verbatim — same heap tie-breaks, same op order). `Selection`'s
//! `indices`/`opacity_scale` fields are `pub(crate)`, so this same-crate module
//! writes them directly and the output is the exact oracle `Selection`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4Swizzles};

use vox_core::types::GaussianSplat;

use crate::atom_budget::{Selection, SelectionStats};
use crate::clas::{build_cluster_bvh, build_clusters, ClusterBVHNode, SplatCluster};
use crate::frustum::Frustum;
use crate::hierarchical_lod::{crossfade_factor, LOD_LEVEL_COUNT};
use crate::spectral::RenderCamera;

/// Fraction of original splat count kept at each LOD level — IDENTICAL to the
/// oracle's private `atom_budget::LOD_FRACTIONS`.
const LOD_FRACTIONS: [f32; LOD_LEVEL_COUNT] = [1.0, 0.4, 0.1, 0.0];

/// GPU-side static cluster geometry. Mirrors `Cluster` in the shader (three
/// `vec4<f32>` = 48 bytes): `aabb_min.xyz + total_opacity`, `aabb_max.xyz`,
/// `center.xyz` (the centroid).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCluster {
    aabb_min_op: [f32; 4],
    aabb_max: [f32; 4],
    center: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuCluster>() == 48);

/// One host-normalized frustum plane (`normal.xyz + d`). Matches `Plane` in the
/// shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPlane {
    n_d: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuPlane>() == 16);

/// Dispatch params uniform (mirrors `Params` in the shader).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    cluster_count: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    eye: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<Params>() == 32);

/// Per-cluster GPU scoring result — the readback POD, matching `ClusterScore`
/// in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct GpuClusterScore {
    /// Importance score = `total_opacity * radius² / distance²`.
    pub score: f32,
    /// Scoring distance: `(centroid - eye).length().max(1e-3)`.
    pub distance: f32,
    /// Leaf-level `frustum.contains_sphere` result: `1` = inside, `0` = culled.
    pub passes_frustum: u32,
    /// Distance-driven LOD ceiling (`0`..=`3`).
    pub distance_lod: u32,
}

const _: () = assert!(std::mem::size_of::<GpuClusterScore>() == 16);

/// Error returned when the GPU selector cannot be created or run. Never panics
/// on a missing/inadequate GPU — the caller can fall back to the CPU oracle.
#[derive(Debug, Clone)]
pub enum AtomBudgetGpuError {
    /// No wgpu adapter (no GPU / no driver) could be found.
    NoAdapter,
    /// An adapter was found but device creation failed.
    DeviceCreation(String),
    /// Mapping the readback buffer failed.
    Readback(String),
    /// A required storage buffer would exceed a hard device limit
    /// (`max_storage_buffer_binding_size` / `max_buffer_size`). Returned instead
    /// of letting wgpu raise an uncaptured Validation Error that aborts the
    /// process — so the caller can fall back to the CPU oracle. Same no-panic
    /// contract class as
    /// [`crate::gpu::splat_rt_gpu::SplatRtGpuError::ExceedsDeviceLimits`].
    ExceedsDeviceLimits {
        what: &'static str,
        requested: u64,
        limit: u64,
    },
}

impl std::fmt::Display for AtomBudgetGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomBudgetGpuError::NoAdapter => write!(f, "no GPU adapter available"),
            AtomBudgetGpuError::DeviceCreation(e) => {
                write!(f, "GPU device creation failed: {e}")
            }
            AtomBudgetGpuError::Readback(e) => write!(f, "GPU readback failed: {e}"),
            AtomBudgetGpuError::ExceedsDeviceLimits {
                what,
                requested,
                limit,
            } => write!(
                f,
                "GPU buffer '{what}' requires {requested} bytes, exceeding device limit {limit}"
            ),
        }
    }
}

impl std::error::Error for AtomBudgetGpuError {}

/// Per-cluster precomputed LOD index lists — mirror of the oracle's private
/// `ClusterLod` (built by the same opacity-sorted prefixes).
struct ClusterLod {
    levels: [Vec<u32>; LOD_LEVEL_COUNT],
}

/// Headless GPU atom-budget LOD selector. Owns its own wgpu device/queue (no
/// window/surface) AND its own copy of the cluster/BVH/LOD scene data.
pub struct AtomBudgetGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    cluster_buffer: wgpu::Buffer,
    plane_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    out_readback: wgpu::Buffer,
    max_clusters: u32,
    /// `device.limits().max_storage_buffer_binding_size`, captured at creation so
    /// `score()` can validate each storage binding range and return
    /// [`AtomBudgetGpuError::ExceedsDeviceLimits`] instead of letting wgpu raise
    /// an uncaptured, process-aborting Validation Error.
    max_storage_binding: u64,
    /// Adapter human name, for diagnostics / benches.
    pub adapter_name: String,

    // --- Self-owned scene data (built via `build_scene`). ---
    clusters: Vec<SplatCluster>,
    bvh: Option<ClusterBVHNode>,
    lods: Vec<ClusterLod>,
    resident: Vec<bool>,
}

impl AtomBudgetGpu {
    /// Create a headless GPU selector sized for up to `max_clusters` clusters.
    /// Returns [`AtomBudgetGpuError`] (never panics) if no adapter is found or
    /// device creation fails. Call [`build_scene`](Self::build_scene) before
    /// [`select`](Self::select).
    pub fn new(max_clusters: u32) -> Result<Self, AtomBudgetGpuError> {
        pollster::block_on(Self::new_async(max_clusters))
    }

    async fn new_async(max_clusters: u32) -> Result<Self, AtomBudgetGpuError> {
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
            .await
            .ok_or(AtomBudgetGpuError::NoAdapter)?;
        let info = adapter.get_info();
        crate::gpu::adapter::ensure_hardware(&info).map_err(|_| AtomBudgetGpuError::NoAdapter)?;
        let adapter_name = info.name;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("atom_budget_gpu_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| AtomBudgetGpuError::DeviceCreation(e.to_string()))?;

        let max_clusters = max_clusters.max(1);

        let cluster_bytes = max_clusters as u64 * std::mem::size_of::<GpuCluster>() as u64;
        let plane_bytes = 6 * std::mem::size_of::<GpuPlane>() as u64;
        let out_bytes = max_clusters as u64 * std::mem::size_of::<GpuClusterScore>() as u64;

        // Validate every STORAGE buffer against the device's binding/total limits
        // BEFORE creating + binding them. Same no-panic-contract hardening as
        // splat_rt_gpu / many_light_gpu: a bound range over
        // `max_storage_buffer_binding_size` triggers an uncaptured,
        // process-aborting wgpu Validation Error, so we surface it as a returned
        // error here instead.
        let limits = device.limits();
        let max_storage_binding = limits.max_storage_buffer_binding_size as u64;
        let max_buffer = limits.max_buffer_size;
        for (what, bytes) in [
            ("cluster_buffer", cluster_bytes),
            ("plane_buffer", plane_bytes),
            ("out_buffer", out_bytes),
        ] {
            if bytes > max_storage_binding {
                return Err(AtomBudgetGpuError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_storage_binding,
                });
            }
            if bytes > max_buffer {
                return Err(AtomBudgetGpuError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_buffer,
                });
            }
        }

        let cluster_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atom_budget_clusters"),
            size: cluster_bytes.max(48),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let plane_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atom_budget_planes"),
            size: plane_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atom_budget_params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atom_budget_out"),
            size: out_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atom_budget_out_readback"),
            size: out_bytes.max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("atom_budget_gpu.wgsl"));

        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atom_budget_bgl"),
            entries: &[
                storage_ro(0), // clusters
                storage_ro(1), // planes
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("atom_budget_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("atom_budget_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bgl,
            cluster_buffer,
            plane_buffer,
            params_buffer,
            out_buffer,
            out_readback,
            max_clusters,
            max_storage_binding,
            adapter_name,
            clusters: Vec::new(),
            bvh: None,
            lods: Vec::new(),
            resident: Vec::new(),
        })
    }

    /// Build the cluster/BVH/LOD scene data from `splats`, IDENTICALLY to the CPU
    /// oracle's `AtomBudgetSelector::build` (same `pub` builders, same
    /// opacity-sorted LOD prefixes). Call once before [`select`](Self::select).
    pub fn build_scene(&mut self, splats: &[GaussianSplat], target_cluster_size: usize) {
        let clusters = build_clusters(splats, target_cluster_size.max(1));
        let bvh = build_cluster_bvh(&clusters);
        let lods: Vec<ClusterLod> = clusters
            .iter()
            .map(|c| build_cluster_lod(c, splats))
            .collect();
        let resident = vec![true; clusters.len()];
        self.clusters = clusters;
        self.bvh = bvh;
        self.lods = lods;
        self.resident = resident;
    }

    /// Number of clusters built. Matches the oracle's `cluster_count` for the
    /// same `(splats, target_cluster_size)`.
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// Streaming hook — non-resident clusters are skipped by `select()`. Mirrors
    /// the oracle's `set_cluster_resident`.
    pub fn set_cluster_resident(&mut self, cluster_id: u32, resident: bool) {
        if let Some(pos) = self.clusters.iter().position(|c| c.id == cluster_id) {
            self.resident[pos] = resident;
        }
    }

    fn bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atom_budget_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.cluster_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.plane_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.out_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Score every built cluster for `camera` on the GPU (one thread per
    /// cluster). Returns one [`GpuClusterScore`] per cluster, indexed by cluster
    /// id (clusters are ids `0..cluster_count`, contiguous). The host budget pass
    /// in [`select`](Self::select) consumes this; exposed for cross-checks.
    pub fn score(
        &self,
        camera: &RenderCamera,
    ) -> Result<Vec<GpuClusterScore>, AtomBudgetGpuError> {
        let n = self.clusters.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        // No-panic contract: an over-`max_clusters` scene goes through the error
        // channel, never a panic (mirrors splat_rt_gpu / many_light_gpu).
        if n as u32 > self.max_clusters {
            return Err(AtomBudgetGpuError::ExceedsDeviceLimits {
                what: "out_buffer (scene exceeds max_clusters)",
                requested: n as u64 * std::mem::size_of::<GpuClusterScore>() as u64,
                limit: self.max_clusters as u64 * std::mem::size_of::<GpuClusterScore>() as u64,
            });
        }
        let out_range = n as u64 * std::mem::size_of::<GpuClusterScore>() as u64;
        if out_range > self.max_storage_binding {
            return Err(AtomBudgetGpuError::ExceedsDeviceLimits {
                what: "out_buffer",
                requested: out_range,
                limit: self.max_storage_binding,
            });
        }

        // Upload cluster geometry.
        let gpu_clusters: Vec<GpuCluster> = self
            .clusters
            .iter()
            .map(|c| GpuCluster {
                aabb_min_op: [c.aabb_min.x, c.aabb_min.y, c.aabb_min.z, c.total_opacity],
                aabb_max: [c.aabb_max.x, c.aabb_max.y, c.aabb_max.z, 0.0],
                center: [c.center.x, c.center.y, c.center.z, 0.0],
            })
            .collect();
        self.queue
            .write_buffer(&self.cluster_buffer, 0, bytemuck::cast_slice(&gpu_clusters));

        // Host-normalize the 6 frustum planes IDENTICALLY to `frustum.rs`'s
        // private `Plane::from_vec4`, so the uploaded plane floats are bit-equal
        // to the oracle's.
        let planes = frustum_planes(camera.view_proj());
        self.queue
            .write_buffer(&self.plane_buffer, 0, bytemuck::cast_slice(&planes));

        let eye = camera_eye(camera);
        let params = Params {
            cluster_count: n as u32,
            _p0: 0,
            _p1: 0,
            _p2: 0,
            eye: [eye.x, eye.y, eye.z, 0.0],
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group = self.bind_group();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("atom_budget_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("atom_budget_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        let copy_bytes = n as u64 * std::mem::size_of::<GpuClusterScore>() as u64;
        encoder.copy_buffer_to_buffer(&self.out_buffer, 0, &self.out_readback, 0, copy_bytes);
        self.queue.submit(Some(encoder.finish()));

        let data = self.map_read(&self.out_readback, copy_bytes)?;
        let out: Vec<GpuClusterScore> =
            bytemuck::cast_slice::<u8, GpuClusterScore>(&data)[..n].to_vec();
        self.out_readback.unmap();
        Ok(out)
    }

    /// GPU-scored equivalent of [`AtomBudgetSelector::select`](crate::atom_budget::AtomBudgetSelector::select).
    /// Scores every cluster on the GPU, then runs the oracle's EXACT budget pass
    /// on the host to fill `out` with a [`Selection`] byte-identical to the CPU
    /// oracle's. Returns the same [`SelectionStats`]. Never panics — buffer
    /// overruns return [`AtomBudgetGpuError::ExceedsDeviceLimits`].
    pub fn select(
        &self,
        camera: &RenderCamera,
        budget: usize,
        out: &mut Selection,
    ) -> Result<SelectionStats, AtomBudgetGpuError> {
        let start = std::time::Instant::now();
        // `Selection`'s fields are pub(crate); clear them directly (the oracle's
        // private `clear()` does exactly this).
        out.indices.clear();
        out.opacity_scale.clear();

        let total_clusters = self.clusters.len();
        let scores = self.score(camera)?;

        // --- 1. Reproduce the oracle's `visible` set: structural BVH walk
        //        (internal-node pruning) AND the GPU leaf-level frustum flag.
        //        The oracle's BVH leaf accept IS contains_sphere on the leaf's
        //        own AABB sphere — exactly what the GPU computed — so a leaf is
        //        visible iff it is BVH-reachable and its GPU flag is set. The
        //        internal-node test is recomputed on the host with the same
        //        host-normalized planes the GPU used. ---
        let frustum = Frustum::from_view_proj(camera.view_proj());
        let mut visible: Vec<u32> = Vec::new();
        if let Some(bvh) = &self.bvh {
            collect_visible_gpu(bvh, &self.clusters, &self.resident, &frustum, &scores, &mut visible);
        }
        visible.sort_unstable();

        // --- 2/3. Build per-visible-cluster working state from GPU scores. ---
        let mut work: Vec<ClusterWork> = Vec::with_capacity(visible.len());
        for &cid in &visible {
            let s = &scores[cid as usize];
            let distance_lod = s.distance_lod as u8;
            let lod = distance_lod;
            let count = self.lods[cid as usize].levels[lod as usize].len();
            work.push(ClusterWork {
                cluster_id: cid,
                distance: s.distance,
                score: s.score,
                distance_lod,
                lod,
                count,
            });
        }

        // --- 4. Drive the summed splat count toward the budget. EXACT mirror
        //        of the oracle's `select()` body (heap tie-breaks included). ---
        let mut total: usize = work.iter().map(|w| w.count).sum();

        if total > budget {
            let mut heap: BinaryHeap<DemoteEntry> = BinaryHeap::with_capacity(work.len());
            for (i, w) in work.iter().enumerate() {
                if (w.lod as usize) < LOD_LEVEL_COUNT - 1 {
                    heap.push(DemoteEntry {
                        score: w.score,
                        work_idx: i,
                    });
                }
            }
            while total > budget {
                let Some(entry) = heap.pop() else { break };
                let w = &mut work[entry.work_idx];
                if (w.lod as usize) >= LOD_LEVEL_COUNT - 1 {
                    continue;
                }
                let old = w.count;
                w.lod += 1;
                w.count = self.lods[w.cluster_id as usize].levels[w.lod as usize].len();
                total = total - old + w.count;
                if (w.lod as usize) < LOD_LEVEL_COUNT - 1 {
                    heap.push(DemoteEntry {
                        score: w.score,
                        work_idx: entry.work_idx,
                    });
                }
            }
            if total > budget {
                let mut by_score: Vec<usize> = (0..work.len()).collect();
                by_score.sort_by(|&a, &b| {
                    work[a]
                        .score
                        .partial_cmp(&work[b].score)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| work[a].cluster_id.cmp(&work[b].cluster_id))
                });
                for idx in by_score {
                    if total <= budget {
                        break;
                    }
                    let w = &mut work[idx];
                    total -= w.count;
                    w.count = 0;
                }
            }
        } else if total < budget {
            let mut heap: BinaryHeap<PromoteEntry> = BinaryHeap::with_capacity(work.len());
            for (i, w) in work.iter().enumerate() {
                if w.lod > w.distance_lod {
                    heap.push(PromoteEntry {
                        score: w.score,
                        work_idx: i,
                    });
                }
            }
            while let Some(entry) = heap.peek().copied() {
                let w = work[entry.work_idx];
                if w.lod <= w.distance_lod {
                    heap.pop();
                    continue;
                }
                let next_lod = w.lod - 1;
                let next_count = self.lods[w.cluster_id as usize].levels[next_lod as usize].len();
                let delta = next_count - w.count;
                if total + delta > budget {
                    break;
                }
                heap.pop();
                let wm = &mut work[entry.work_idx];
                wm.lod = next_lod;
                wm.count = next_count;
                total += delta;
                if wm.lod > wm.distance_lod {
                    heap.push(PromoteEntry {
                        score: wm.score,
                        work_idx: entry.work_idx,
                    });
                }
            }
        }

        // --- 5. Emit indices + crossfade opacity multipliers. EXACT mirror. ---
        out.indices.reserve(total);
        out.opacity_scale.reserve(total);
        let mut histogram = [0usize; LOD_LEVEL_COUNT];
        for w in work.iter() {
            if w.count == 0 {
                continue;
            }
            histogram[w.lod as usize] += 1;
            let level = &self.lods[w.cluster_id as usize].levels[w.lod as usize];
            let fade = if w.lod == w.distance_lod {
                crossfade_factor(w.distance, w.lod as u32)
            } else {
                0.0
            };
            let scale = 1.0 - fade;
            out.indices.extend_from_slice(level);
            if scale == 1.0 {
                out.opacity_scale.resize(out.indices.len(), 1.0);
            } else {
                out.opacity_scale
                    .extend(std::iter::repeat_n(scale, level.len()));
            }
        }

        let selected = out.indices.len();
        let clusters_visible = work.len();
        let clusters_culled = total_clusters - clusters_visible;
        let select_us = start.elapsed().as_micros() as u64;

        Ok(SelectionStats {
            budget,
            selected,
            clusters_visible,
            clusters_culled,
            lod_histogram: histogram,
            select_us,
        })
    }

    fn map_read(
        &self,
        buffer: &wgpu::Buffer,
        bytes: u64,
    ) -> Result<Vec<u8>, AtomBudgetGpuError> {
        let slice = buffer.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(AtomBudgetGpuError::Readback(e.to_string())),
            Err(e) => return Err(AtomBudgetGpuError::Readback(e.to_string())),
        }
        let data = slice.get_mapped_range().to_vec();
        Ok(data)
    }
}

/// Build the 4-level per-cluster LOD index table — IDENTICAL to the oracle's
/// private `atom_budget::build_cluster_lod` (opacity-descending prefixes, stable
/// id tie-break; L3 = the single brightest splat).
fn build_cluster_lod(cluster: &SplatCluster, splats: &[GaussianSplat]) -> ClusterLod {
    let mut sorted: Vec<u32> = cluster.splat_indices.clone();
    sorted.sort_by(|&a, &b| {
        let oa = splats[a as usize].opacity();
        let ob = splats[b as usize].opacity();
        ob.cmp(&oa).then_with(|| a.cmp(&b))
    });

    let n = sorted.len();
    let l0 = sorted.clone();
    let l1_len = ((n as f32 * LOD_FRACTIONS[1]).round() as usize)
        .clamp(if n > 0 { 1 } else { 0 }, n);
    let l2_len = ((n as f32 * LOD_FRACTIONS[2]).round() as usize)
        .clamp(if n > 0 { 1 } else { 0 }, n);
    let l1 = sorted[..l1_len].to_vec();
    let l2 = sorted[..l2_len].to_vec();
    let l3 = if n > 0 { vec![sorted[0]] } else { Vec::new() };

    ClusterLod {
        levels: [l0, l1, l2, l3],
    }
}

/// Working LOD-selection state for one visible cluster — mirror of the oracle's
/// private `ClusterWork`.
#[derive(Clone, Copy)]
struct ClusterWork {
    cluster_id: u32,
    distance: f32,
    score: f32,
    distance_lod: u8,
    lod: u8,
    count: usize,
}

/// Heap entry for budget demotion — mirror of the oracle's `DemoteEntry`
/// (min-heap on score; ties broken by `work_idx`).
#[derive(Clone, Copy)]
struct DemoteEntry {
    score: f32,
    work_idx: usize,
}
impl PartialEq for DemoteEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.work_idx == other.work_idx
    }
}
impl Eq for DemoteEntry {}
impl Ord for DemoteEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.work_idx.cmp(&self.work_idx))
    }
}
impl PartialOrd for DemoteEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Heap entry for budget promotion — mirror of the oracle's `PromoteEntry`
/// (max-heap on score; ties broken by `work_idx`).
#[derive(Clone, Copy)]
struct PromoteEntry {
    score: f32,
    work_idx: usize,
}
impl PartialEq for PromoteEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.work_idx == other.work_idx
    }
}
impl Eq for PromoteEntry {}
impl Ord for PromoteEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.work_idx.cmp(&self.work_idx))
    }
}
impl PartialOrd for PromoteEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Walk the cluster BVH structurally, collecting visible cluster ids. Reproduces
/// the oracle's `collect_visible` EXACTLY: an internal node is entered iff its
/// AABB sphere is inside the frustum, and a leaf is accepted iff resident AND its
/// own AABB sphere is inside — but here the leaf accept reuses the GPU-computed
/// `passes_frustum` flag (bit-identical to the oracle's leaf `contains_sphere`),
/// while the INTERNAL-node test is recomputed on the host with the same
/// host-normalized planes the GPU used.
fn collect_visible_gpu(
    node: &ClusterBVHNode,
    clusters: &[SplatCluster],
    resident: &[bool],
    frustum: &Frustum,
    scores: &[GpuClusterScore],
    out: &mut Vec<u32>,
) {
    match node {
        ClusterBVHNode::Leaf { cluster_id } => {
            let cid = *cluster_id;
            if !resident[cid as usize] {
                return;
            }
            // Leaf accept == GPU leaf-level contains_sphere flag (bit-identical to
            // the oracle's per-leaf test).
            if scores[cid as usize].passes_frustum != 0 {
                out.push(cid);
            }
            let _ = clusters; // clusters only needed for the internal-node arms
        }
        ClusterBVHNode::Internal {
            aabb_min,
            aabb_max,
            left,
            right,
        } => {
            let centre = (*aabb_min + *aabb_max) * 0.5;
            let radius = ((*aabb_max - *aabb_min) * 0.5).length().max(1e-4);
            if !frustum.contains_sphere(centre, radius) {
                return;
            }
            collect_visible_gpu(left, clusters, resident, frustum, scores, out);
            collect_visible_gpu(right, clusters, resident, frustum, scores, out);
        }
    }
}

/// Camera eye position — IDENTICAL to the oracle's private `camera_eye`.
fn camera_eye(camera: &RenderCamera) -> Vec3 {
    camera.view.inverse().col(3).truncate()
}

/// The 6 normalized frustum planes, computed IDENTICALLY to `frustum.rs`'s
/// `Frustum::from_view_proj` + `Plane::from_vec4`, so the GPU evaluates the same
/// plane floats the oracle's CPU `contains_sphere` does. Plane order matches the
/// oracle (left, right, bottom, top, near, far); the all-planes AND test is
/// order-independent, but the order is kept for clarity.
fn frustum_planes(vp: glam::Mat4) -> [GpuPlane; 6] {
    let row0 = vp.row(0);
    let row1 = vp.row(1);
    let row2 = vp.row(2);
    let row3 = vp.row(3);
    let raw = [
        row3 + row0,
        row3 - row0,
        row3 + row1,
        row3 - row1,
        row3 + row2,
        row3 - row2,
    ];
    raw.map(|v| {
        let len = v.xyz().length();
        let normal = v.xyz() / len;
        let d = v.w / len;
        GpuPlane {
            n_d: [normal.x, normal.y, normal.z, d],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom_budget::AtomBudgetSelector;
    use crate::hierarchical_lod::select_lod_level;
    use glam::{Mat4, Quat};
    use std::f32::consts::FRAC_PI_4;

    /// Skip a GPU test gracefully if this box truly has no GPU (CI without one).
    fn try_gpu(max_clusters: u32) -> Option<AtomBudgetGpu> {
        match AtomBudgetGpu::new(max_clusters) {
            Ok(g) => {
                eprintln!("[atom_budget_gpu test] adapter: {}", g.adapter_name);
                Some(g)
            }
            Err(AtomBudgetGpuError::NoAdapter) => {
                eprintln!("[atom_budget_gpu test] no adapter — skipping GPU test");
                None
            }
            Err(e) => panic!("unexpected GPU init error on a box with a GPU: {e}"),
        }
    }

    fn splat_at(pos: [f32; 3], opacity: u8) -> GaussianSplat {
        GaussianSplat::volume(pos, [0.3, 0.3, 0.3], Quat::IDENTITY, opacity, [0u16; 16])
    }

    /// The oracle's exact `grid_scene` test generator: ~64k splats, deterministic
    /// opacities (40^3 grid). Reconstructed identically.
    fn grid_scene() -> Vec<GaussianSplat> {
        let mut v = Vec::new();
        let n = 40; // 40^3 = 64000
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let op = (((x * 7 + y * 13 + z * 17) % 200) + 55) as u8;
                    v.push(splat_at(
                        [x as f32 * 1.0 - 20.0, y as f32 * 0.5, z as f32 * 1.0 - 20.0],
                        op,
                    ));
                }
            }
        }
        v
    }

    /// A larger deterministic grid (hundreds of clusters) at a higher atom count
    /// than `grid_scene`, to exercise the GPU on the largest in-test scene.
    fn big_grid_scene(side: i32) -> Vec<GaussianSplat> {
        let mut v = Vec::new();
        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    let op = (((x * 11 + y * 5 + z * 23) % 200) + 55) as u8;
                    v.push(splat_at(
                        [
                            x as f32 * 1.5 - side as f32 * 0.75,
                            y as f32 * 0.5,
                            z as f32 * 1.5 - side as f32 * 0.75,
                        ],
                        op,
                    ));
                }
            }
        }
        v
    }

    fn camera_at(eye: Vec3, target: Vec3) -> RenderCamera {
        RenderCamera {
            view: Mat4::look_at_rh(eye, target, Vec3::Y),
            proj: Mat4::perspective_rh(FRAC_PI_4, 1.0, 0.1, 2000.0),
        }
    }

    /// Build a CPU oracle + a GPU selector over the SAME scene/params.
    fn build_pair(
        scene: &[GaussianSplat],
        target: usize,
    ) -> (AtomBudgetSelector, Option<AtomBudgetGpu>) {
        let cpu = AtomBudgetSelector::build(scene, target);
        let gpu = try_gpu(cpu.cluster_count().max(1) as u32).map(|mut g| {
            g.build_scene(scene, target);
            g
        });
        (cpu, gpu)
    }

    /// THE VALIDATION: the GPU-scored selection must equal the CPU oracle's
    /// `Selection` EXACTLY — same indices, same opacity multipliers, same stats —
    /// across a spread of budgets and cameras. Exactness is the contract (the
    /// output is discrete).
    #[test]
    fn gpu_selection_exactly_equals_cpu_oracle() {
        let scene = grid_scene();
        let (mut cpu, gpu) = build_pair(&scene, 128);
        let Some(gpu) = gpu else { return };
        // Precondition: the two builders agree on cluster count.
        assert_eq!(
            cpu.cluster_count(),
            gpu.cluster_count(),
            "CPU and GPU must build identical cluster sets"
        );

        let cameras = [
            camera_at(Vec3::new(0.0, 10.0, -25.0), Vec3::new(0.0, 10.0, 25.0)),
            camera_at(Vec3::new(2.0, 8.0, -20.0), Vec3::new(1.0, 9.0, 30.0)),
            camera_at(Vec3::new(-15.0, 5.0, -15.0), Vec3::new(15.0, 5.0, 15.0)),
        ];
        let budgets = [100usize, 2000, 8000, 32_000, usize::MAX];

        let mut max_score_abs = 0.0f32;
        let mut max_score_rel = 0.0f32;

        for cam in &cameras {
            // Cross-check the raw GPU scores vs the oracle's CPU scoring first.
            let gpu_scores = gpu.score(cam).expect("gpu score");
            let cpu_scores = cpu_cluster_scores(&gpu, cam);
            for (cid, (g, c)) in gpu_scores.iter().zip(cpu_scores.iter()).enumerate() {
                // distance_lod and frustum flag are discrete → must be exact.
                assert_eq!(
                    g.distance_lod, c.distance_lod,
                    "cluster {cid}: distance_lod GPU {} != CPU {}",
                    g.distance_lod, c.distance_lod
                );
                assert_eq!(
                    g.passes_frustum, c.passes_frustum,
                    "cluster {cid}: frustum flag GPU {} != CPU {}",
                    g.passes_frustum, c.passes_frustum
                );
                let d = (g.score - c.score).abs();
                max_score_abs = max_score_abs.max(d);
                max_score_rel = max_score_rel.max(d / c.score.abs().max(1e-9));
            }

            for &budget in &budgets {
                let mut cpu_out = Selection::new();
                let cpu_stats = cpu.select(cam, budget, &mut cpu_out);

                let mut gpu_out = Selection::new();
                let gpu_stats = gpu.select(cam, budget, &mut gpu_out).expect("gpu select");

                assert_eq!(
                    gpu_out.indices(),
                    cpu_out.indices(),
                    "budget {budget}: GPU indices must EXACTLY equal CPU oracle"
                );
                // opacity_scale is f32 but produced by the IDENTICAL crossfade op
                // chain over the SAME (matched-exact) distance — assert bit-equal.
                let cpu_op = cpu_out.opacity_scale();
                let gpu_op = gpu_out.opacity_scale();
                assert_eq!(gpu_op.len(), cpu_op.len(), "opacity length mismatch");
                for (i, (a, b)) in gpu_op.iter().zip(cpu_op.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "budget {budget}: opacity[{i}] GPU {a} != CPU {b} (bit)"
                    );
                }
                assert_eq!(gpu_stats.selected, cpu_stats.selected, "selected mismatch");
                assert_eq!(
                    gpu_stats.clusters_visible, cpu_stats.clusters_visible,
                    "clusters_visible mismatch"
                );
                assert_eq!(
                    gpu_stats.clusters_culled, cpu_stats.clusters_culled,
                    "clusters_culled mismatch"
                );
                assert_eq!(
                    gpu_stats.lod_histogram, cpu_stats.lod_histogram,
                    "lod_histogram mismatch"
                );
            }
        }
        eprintln!(
            "[validate] GPU<->CPU score: max_abs={max_score_abs:e} max_rel={max_score_rel:e}"
        );
        // Identical f32 op order + no fast-math → scores are ULP-tight on RADV.
        // Measured well below 1e-5 relative; assert a generous, above-measured
        // bound. (Selection equality above is the HARD contract; this guards the
        // continuous score channel.)
        assert!(
            max_score_rel < 1e-4,
            "GPU<->CPU max relative score deviation {max_score_rel:e} exceeds 1e-4"
        );
    }

    /// A larger deterministic scene (hundreds of clusters) — selection must still
    /// match the oracle EXACTLY, and we record GPU-vs-CPU timing (informational).
    #[test]
    fn gpu_matches_cpu_on_large_scene_with_timing() {
        let scene = big_grid_scene(48); // 48^3 = 110_592 splats
        let (mut cpu, gpu) = build_pair(&scene, 128);
        let Some(gpu) = gpu else { return };
        eprintln!(
            "[large] {} splats, {} clusters",
            scene.len(),
            gpu.cluster_count()
        );
        assert_eq!(cpu.cluster_count(), gpu.cluster_count());

        let cam = camera_at(Vec3::new(0.0, 12.0, -40.0), Vec3::new(0.0, 12.0, 40.0));
        let budget = 50_000usize;

        let mut cpu_out = Selection::new();
        let t0 = std::time::Instant::now();
        let cpu_stats = cpu.select(&cam, budget, &mut cpu_out);
        let cpu_us = t0.elapsed().as_micros();

        let mut gpu_out = Selection::new();
        let t1 = std::time::Instant::now();
        let gpu_stats = gpu.select(&cam, budget, &mut gpu_out).expect("gpu select");
        let gpu_us = t1.elapsed().as_micros();

        eprintln!(
            "[timing] {} clusters: CPU select={cpu_us}µs  GPU select(incl dispatch+readback)={gpu_us}µs  selected={}",
            gpu.cluster_count(),
            cpu_stats.selected
        );

        assert_eq!(
            gpu_out.indices(),
            cpu_out.indices(),
            "large scene: GPU indices must EXACTLY equal CPU oracle"
        );
        assert_eq!(
            gpu_out.opacity_scale(),
            cpu_out.opacity_scale(),
            "large scene: opacity multipliers must match"
        );
        assert_eq!(gpu_stats.lod_histogram, cpu_stats.lod_histogram);
        assert_eq!(gpu_stats.selected, cpu_stats.selected);
    }

    /// Bit-identical determinism: two GPU selects of the same scene/camera/budget
    /// produce byte-identical `Selection`s.
    #[test]
    fn gpu_is_deterministic() {
        let scene = grid_scene();
        let (_cpu, gpu) = build_pair(&scene, 128);
        let Some(gpu) = gpu else { return };
        let cam = camera_at(Vec3::new(1.0, 9.0, -22.0), Vec3::new(0.0, 9.0, 30.0));

        let mut a = Selection::new();
        let mut b = Selection::new();
        gpu.select(&cam, 8000, &mut a).expect("select a");
        gpu.select(&cam, 8000, &mut b).expect("select b");
        assert_eq!(a.indices(), b.indices(), "indices not deterministic");
        for (x, y) in a.opacity_scale().iter().zip(b.opacity_scale().iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "opacity not bit-deterministic");
        }
    }

    /// Edge cases must match the CPU oracle EXACTLY: empty scene, everything
    /// culled (camera looking away), and budget 0.
    #[test]
    fn edge_cases_match_cpu_oracle() {
        // --- Empty scene: zero clusters. score() returns empty, select() empty. ---
        {
            let empty: Vec<GaussianSplat> = Vec::new();
            let (mut cpu, gpu) = build_pair(&empty, 128);
            let Some(gpu) = gpu else { return };
            assert_eq!(cpu.cluster_count(), 0);
            assert_eq!(gpu.cluster_count(), 0);
            let cam = camera_at(Vec3::new(0.0, 0.0, -5.0), Vec3::ZERO);

            let mut cpu_out = Selection::new();
            let cpu_stats = cpu.select(&cam, 1000, &mut cpu_out);
            let mut gpu_out = Selection::new();
            let gpu_stats = gpu.select(&cam, 1000, &mut gpu_out).expect("empty select");
            assert_eq!(gpu_out.indices(), cpu_out.indices());
            assert_eq!(gpu_out.indices().len(), 0, "empty scene emits nothing");
            assert_eq!(gpu_stats.selected, cpu_stats.selected);
            assert_eq!(gpu_stats.lod_histogram, cpu_stats.lod_histogram);
        }

        // --- All culled: splats behind the camera (camera looks away). ---
        {
            let mut scene = Vec::new();
            for i in 0..2000 {
                let f = i as f32;
                scene.push(splat_at(
                    [(f % 10.0) - 5.0, (f * 0.01) % 5.0, -10.0 - (f * 0.1)],
                    200,
                ));
            }
            let (mut cpu, gpu) = build_pair(&scene, 64);
            let Some(gpu) = gpu else { return };
            let looking_away = camera_at(Vec3::ZERO, Vec3::new(0.0, 0.0, 50.0));
            let mut cpu_out = Selection::new();
            let cpu_stats = cpu.select(&looking_away, 100_000, &mut cpu_out);
            let mut gpu_out = Selection::new();
            let gpu_stats = gpu
                .select(&looking_away, 100_000, &mut gpu_out)
                .expect("all-culled select");
            assert_eq!(gpu_out.indices(), cpu_out.indices());
            assert_eq!(gpu_out.indices().len(), 0, "all behind camera → culled");
            assert_eq!(
                gpu_stats.clusters_culled, cpu_stats.clusters_culled,
                "cull count must match oracle"
            );
            assert!(gpu_stats.clusters_culled > 0);

            // --- Budget 0: in-frustum scene, but no atoms may be emitted. ---
            let looking_at = camera_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -50.0));
            let mut cpu0 = Selection::new();
            let cpu0_stats = cpu.select(&looking_at, 0, &mut cpu0);
            let mut gpu0 = Selection::new();
            let gpu0_stats = gpu.select(&looking_at, 0, &mut gpu0).expect("budget-0 select");
            assert_eq!(
                gpu0.indices(),
                cpu0.indices(),
                "budget 0: GPU must shed identically to the oracle"
            );
            assert_eq!(gpu0_stats.selected, cpu0_stats.selected);
            assert_eq!(gpu0_stats.lod_histogram, cpu0_stats.lod_histogram);
        }
    }

    /// CONTRACT (class-consistency with splat_rt_gpu / many_light_gpu): a scene
    /// with MORE clusters than the GPU was sized for returns
    /// [`AtomBudgetGpuError::ExceedsDeviceLimits`], NOT a panic, and the GPU stays
    /// usable afterward.
    #[test]
    fn oversized_returns_error_not_panic() {
        let scene = grid_scene();
        // Size the GPU for only 4 clusters, then build the full ~hundreds-cluster
        // scene into it.
        let Some(mut gpu) = try_gpu(4) else { return };
        gpu.build_scene(&scene, 128);
        assert!(gpu.cluster_count() > 4, "precondition: many clusters");
        let cam = camera_at(Vec3::new(0.0, 10.0, -25.0), Vec3::new(0.0, 10.0, 25.0));

        let err = gpu
            .score(&cam)
            .expect_err("oversized scene must return an error, not abort");
        match err {
            AtomBudgetGpuError::ExceedsDeviceLimits {
                what,
                requested,
                limit,
            } => {
                assert_eq!(what, "out_buffer (scene exceeds max_clusters)");
                assert!(requested > limit, "requested {requested} must exceed limit {limit}");
            }
            other => panic!("expected ExceedsDeviceLimits, got {other:?}"),
        }
        // select() must also surface the error (not abort).
        let mut out = Selection::new();
        let err2 = gpu
            .select(&cam, 1000, &mut out)
            .expect_err("oversized select must return an error, not abort");
        assert!(matches!(err2, AtomBudgetGpuError::ExceedsDeviceLimits { .. }));

        // GPU still usable: a tiny scene that fits still works (no abort).
        let mut small = AtomBudgetGpu::new(4).expect("re-init small gpu");
        small.build_scene(&scene[..200], 256);
        assert!(small.cluster_count() <= 4, "tiny scene must fit");
        let mut out2 = Selection::new();
        let _ = small
            .select(&cam, 1000, &mut out2)
            .expect("valid select on a fitting scene");
    }

    // --- CPU reference: the oracle's exact per-cluster scoring, recomputed here
    //     to cross-check the GPU score channel. Mirrors `atom_budget::select`'s
    //     scoring block op-for-op, reading the GPU selector's own (identical)
    //     cluster set. ---

    struct CpuScore {
        score: f32,
        distance_lod: u32,
        passes_frustum: u32,
    }

    fn cpu_cluster_scores(gpu: &AtomBudgetGpu, camera: &RenderCamera) -> Vec<CpuScore> {
        let eye = camera_eye(camera);
        let planes = frustum_planes(camera.view_proj());
        gpu.clusters
            .iter()
            .map(|c| {
                let radius = ((c.aabb_max - c.aabb_min) * 0.5).length().max(1e-4);
                let d = (c.center - eye).length().max(1e-3);
                let screen = 1000.0 * radius / d;
                let distance_lod = select_lod_level(d, screen);
                let score = c.total_opacity * (radius * radius) / (d * d);
                let mid = (c.aabb_min + c.aabb_max) * 0.5;
                let mut passes = 1u32;
                for p in &planes {
                    let dist =
                        p.n_d[0] * mid.x + p.n_d[1] * mid.y + p.n_d[2] * mid.z + p.n_d[3];
                    if dist < -radius {
                        passes = 0;
                        break;
                    }
                }
                CpuScore {
                    score,
                    distance_lod,
                    passes_frustum: passes,
                }
            })
            .collect()
    }
}
