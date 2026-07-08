//! Tier-1 resident SDF atlas (`SdfAtlasGpu`) — SDF-pillar design §4.2 / §5 / §6.
//!
//! One `R8Snorm` 3D texture holds every resident per-asset distance field;
//! hardware trilinear filtering samples it directly (R8Snorm is
//! core-filterable in wgpu 24 — sample returns [-1, 1], decode is
//! `value * narrow_band` metres). Boxes are placed by a 3D shelf allocator
//! over a 512×512 cross-section whose depth is scaled to the byte quota
//! (the design's 64 MB ceiling = 512×512×256 voxels at 1 B/voxel), each box
//! carrying a 1-voxel replicated border so filtering never bleeds a
//! neighbouring asset.
//!
//! Eviction is least-recently-inserted — deliberately the trivial case of
//! design §4.9's degradation order (wounded/hero residency classes arrive in
//! later waves). Freed boxes go on a first-fit free list, so same-recipe
//! cooks (the common case: one cook target per game) reuse slots exactly.
//!
//! [`SdfAtlasProbe`] is the atlas's first consumer: a tiny compute kernel
//! that binds [`SdfAtlasGpu::texture`] + [`SdfAtlasGpu::header_buf`] exactly
//! the way later passes (clipmap composite, contact AO, far imposters) will,
//! and is how the hardware-trilinear parity gate is measured.

use crate::gpu::GpuContext;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use vox_core::sdf::{SdfField, SdfPayload, SdfSign};
use vox_physics::sdf::SdfAssetId;
use wgpu::util::DeviceExt;

/// Atlas cross-section: the design's 64 MB ceiling is 512×512×256 voxels at
/// 1 B/voxel; depth scales with the quota, X/Y stay fixed at 512.
const ATLAS_XY: u32 = 512;
/// Minimum useful depth — below this no tier-1 box (plus border) fits.
const MIN_DEPTH: u32 = 4;
/// Header slot capacity. 1024 × 48 B = 48 KB of headers; eviction frees
/// slots, so this bounds *concurrent* residency, not total inserts.
const HEADER_SLOTS: u32 = 1024;
const HEADER_BYTES: u64 = std::mem::size_of::<GpuSdfFieldHeader>() as u64;

/// GPU-side per-asset field header — 48 bytes, std430-compatible (the WGSL
/// mirror uses scalar fields, so the array stride is exactly 48).
///
/// Design §5 reconciliation: the prose pins 48 B; the sketch's two `_pad`
/// words would make it 64, so they are dropped and `band_sign` carries only
/// its two live floats — `narrow_band` metres and the sign convention
/// (+1 = `SdfSign::Closed`, −1 = `SdfSign::Shell`).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq)]
pub struct GpuSdfFieldHeader {
    /// Atlas texel of the field's interior voxel (0,0,0) — past the border.
    atlas_offset_voxels: [u32; 3],
    resolution: [u32; 3],
    /// origin.xyz (asset-local metres), voxel_size.
    origin_voxel: [f32; 4],
    /// narrow_band metres, sign (+1 Closed / −1 Shell).
    band_sign: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<GpuSdfFieldHeader>() == 48);

impl GpuSdfFieldHeader {
    pub fn atlas_offset_voxels(&self) -> [u32; 3] {
        self.atlas_offset_voxels
    }

    pub fn resolution(&self) -> [u32; 3] {
        self.resolution
    }

    pub fn origin(&self) -> [f32; 3] {
        [
            self.origin_voxel[0],
            self.origin_voxel[1],
            self.origin_voxel[2],
        ]
    }

    pub fn voxel_size(&self) -> f32 {
        self.origin_voxel[3]
    }

    pub fn narrow_band(&self) -> f32 {
        self.band_sign[0]
    }

    /// +1.0 for `SdfSign::Closed`, −1.0 for `SdfSign::Shell`.
    pub fn sign(&self) -> f32 {
        self.band_sign[1]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SdfAtlasError {
    /// The quota is too small to host even one minimal box.
    QuotaTooSmall {
        quota_bytes: u64,
        min_bytes: u64,
    },
    /// A single field's payload exceeds the whole quota — no eviction
    /// sequence can ever make it resident.
    QuotaExceeded {
        needed_bytes: u64,
        quota_bytes: u64,
    },
    /// The field's bordered box does not fit the atlas extents at this quota.
    FieldTooLarge {
        resolution: [u32; 3],
        atlas: [u32; 3],
    },
    DeviceLimits(String),
}

impl fmt::Display for SdfAtlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QuotaTooSmall {
                quota_bytes,
                min_bytes,
            } => write!(
                f,
                "atlas quota {quota_bytes} B is below the {min_bytes} B minimum"
            ),
            Self::QuotaExceeded {
                needed_bytes,
                quota_bytes,
            } => write!(
                f,
                "field needs {needed_bytes} B, the whole quota is {quota_bytes} B"
            ),
            Self::FieldTooLarge { resolution, atlas } => write!(
                f,
                "field {resolution:?} (+2 border) does not fit atlas {atlas:?}"
            ),
            Self::DeviceLimits(msg) => write!(f, "device limits: {msg}"),
        }
    }
}

impl std::error::Error for SdfAtlasError {}

/// One open shelf inside a slab: boxes packed left-to-right along X.
struct Shelf {
    y0: u32,
    height: u32,
    cursor_x: u32,
}

/// One Z-slab: shelves stacked along Y, slab depth fixed by its first box.
struct Slab {
    z0: u32,
    depth: u32,
    cursor_y: u32,
    shelves: Vec<Shelf>,
}

/// A box returned to the allocator by eviction. First-fit reuse; any
/// remainder is wasted — exact for same-recipe cooks, which share one
/// `SdfBakeTarget` and therefore one box shape per asset class.
struct FreeBox {
    origin: [u32; 3],
    dims: [u32; 3],
}

struct Resident {
    slot: u32,
    box_origin: [u32; 3],
    box_dims: [u32; 3],
    payload_bytes: u64,
}

/// Owns the `R8Snorm` 3D atlas texture + the mirrored header buffer.
/// Render-thread-owned; inserts are tool/cook-paced (each is one
/// `write_texture` + one `write_buffer`), never frame-coupled.
pub struct SdfAtlasGpu {
    ctx: GpuContext,
    texture: wgpu::Texture,
    header_buf: wgpu::Buffer,
    dims: [u32; 3],
    quota_bytes: u64,
    resident_bytes: u64,
    evicted: u64,
    slabs: Vec<Slab>,
    cursor_z: u32,
    free_boxes: Vec<FreeBox>,
    free_slots: Vec<u32>,
    next_slot: u32,
    residents: HashMap<SdfAssetId, Resident>,
    /// Insertion order, front = least-recently-inserted = first evicted.
    order: VecDeque<SdfAssetId>,
}

impl SdfAtlasGpu {
    /// Shared-context constructor (`GpuGi::new_with_context` precedent).
    /// Allocates the atlas texture at `512 × 512 × (quota / 512²)` so the
    /// spatial ceiling and the byte quota agree.
    pub fn new_with_context(ctx: &GpuContext, quota_bytes: u64) -> Result<Self, SdfAtlasError> {
        let device = ctx.device();
        let limits = device.limits();
        if limits.max_texture_dimension_3d < ATLAS_XY {
            return Err(SdfAtlasError::DeviceLimits(format!(
                "atlas needs {ATLAS_XY}³-capable 3D textures, device caps at {}",
                limits.max_texture_dimension_3d
            )));
        }
        let depth = ((quota_bytes / (ATLAS_XY as u64 * ATLAS_XY as u64)) as u32)
            .min(limits.max_texture_dimension_3d);
        if depth < MIN_DEPTH {
            return Err(SdfAtlasError::QuotaTooSmall {
                quota_bytes,
                min_bytes: ATLAS_XY as u64 * ATLAS_XY as u64 * MIN_DEPTH as u64,
            });
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sdf_atlas_r8snorm"),
            size: wgpu::Extent3d {
                width: ATLAS_XY,
                height: ATLAS_XY,
                depth_or_array_layers: depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Snorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let header_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdf_atlas_headers"),
            size: HEADER_SLOTS as u64 * HEADER_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            ctx: ctx.clone(),
            texture,
            header_buf,
            dims: [ATLAS_XY, ATLAS_XY, depth],
            quota_bytes,
            resident_bytes: 0,
            evicted: 0,
            slabs: Vec::new(),
            cursor_z: 0,
            free_boxes: Vec::new(),
            free_slots: Vec::new(),
            next_slot: 0,
            residents: HashMap::new(),
            order: VecDeque::new(),
        })
    }

    /// Upload the field (or return its resident slot). Both `Snorm8` and
    /// `Snorm16` payloads are accepted — snorm16 is re-quantized to snorm8 on
    /// upload (the atlas is the snorm8 tier; disk payloads stay snorm16).
    /// Evicts least-recently-inserted entries when the byte quota or the
    /// spatial ceiling is hit. Returns the header slot index.
    pub fn insert(&mut self, asset: SdfAssetId, field: &SdfField) -> Result<u32, SdfAtlasError> {
        if let Some(resident) = self.residents.get(&asset) {
            return Ok(resident.slot);
        }

        let desc = field.desc();
        let res = desc.resolution();
        let padded = [res[0] + 2, res[1] + 2, res[2] + 2];
        if padded[0] > self.dims[0] || padded[1] > self.dims[1] || padded[2] > self.dims[2] {
            return Err(SdfAtlasError::FieldTooLarge {
                resolution: res,
                atlas: self.dims,
            });
        }
        // Quota accounting counts the payload voxels (1 B each as snorm8) —
        // the design §4.2 memory math; border + shelf waste live inside the
        // texture's own quota-scaled ceiling.
        let payload_bytes = desc.sample_count() as u64;
        if payload_bytes > self.quota_bytes {
            return Err(SdfAtlasError::QuotaExceeded {
                needed_bytes: payload_bytes,
                quota_bytes: self.quota_bytes,
            });
        }
        while self.resident_bytes + payload_bytes > self.quota_bytes {
            if !self.evict_least_recently_inserted() {
                return Err(SdfAtlasError::QuotaExceeded {
                    needed_bytes: payload_bytes,
                    quota_bytes: self.quota_bytes,
                });
            }
        }
        let box_origin = loop {
            if let Some(origin) = self.try_alloc(padded) {
                break origin;
            }
            if !self.evict_least_recently_inserted() {
                // Empty atlas and still no box — fragmentation-free by
                // construction, so the field genuinely cannot fit.
                return Err(SdfAtlasError::FieldTooLarge {
                    resolution: res,
                    atlas: self.dims,
                });
            }
        };

        // 1-voxel replicated border so trilinear taps never bleed neighbours.
        let data = padded_snorm8_volume(field);
        self.ctx.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: box_origin[0],
                    y: box_origin[1],
                    z: box_origin[2],
                },
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded[0]),
                rows_per_image: Some(padded[1]),
            },
            wgpu::Extent3d {
                width: padded[0],
                height: padded[1],
                depth_or_array_layers: padded[2],
            },
        );

        let slot = match self.free_slots.pop() {
            Some(slot) => slot,
            None if self.next_slot < HEADER_SLOTS => {
                let slot = self.next_slot;
                self.next_slot += 1;
                slot
            }
            None => {
                // Headers exhausted before space was: evict one entry for its
                // slot (its freed box simply joins the free list).
                if !self.evict_least_recently_inserted() {
                    return Err(SdfAtlasError::DeviceLimits(
                        "header slots exhausted with nothing to evict".into(),
                    ));
                }
                self.free_slots.pop().expect("eviction frees a header slot")
            }
        };
        let header = GpuSdfFieldHeader {
            atlas_offset_voxels: [box_origin[0] + 1, box_origin[1] + 1, box_origin[2] + 1],
            resolution: res,
            origin_voxel: [
                desc.origin().x,
                desc.origin().y,
                desc.origin().z,
                desc.voxel_size(),
            ],
            band_sign: [
                desc.narrow_band(),
                match desc.sign() {
                    SdfSign::Closed => 1.0,
                    SdfSign::Shell => -1.0,
                },
            ],
        };
        self.ctx.queue().write_buffer(
            &self.header_buf,
            slot as u64 * HEADER_BYTES,
            bytemuck::bytes_of(&header),
        );

        self.residents.insert(
            asset,
            Resident {
                slot,
                box_origin,
                box_dims: padded,
                payload_bytes,
            },
        );
        self.order.push_back(asset);
        self.resident_bytes += payload_bytes;
        Ok(slot)
    }

    /// Payload bytes currently resident (sum of unpadded voxel counts — the
    /// quota-governing measure).
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Total evictions since construction.
    pub fn evicted_count(&self) -> u64 {
        self.evicted
    }

    /// Resident asset count.
    pub fn resident_count(&self) -> usize {
        self.residents.len()
    }

    /// The `R8Snorm` 3D atlas — bind in consumer passes (filterable float).
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// The `GpuSdfFieldHeader` storage buffer, indexed by insert's slot.
    pub fn header_buf(&self) -> &wgpu::Buffer {
        &self.header_buf
    }

    /// Evict the least-recently-inserted resident — the trivial case of
    /// design §4.9's degradation order (wounded/hero classes: later waves).
    fn evict_least_recently_inserted(&mut self) -> bool {
        let Some(asset) = self.order.pop_front() else {
            return false;
        };
        let resident = self
            .residents
            .remove(&asset)
            .expect("order and resident map stay in sync");
        self.free_boxes.push(FreeBox {
            origin: resident.box_origin,
            dims: resident.box_dims,
        });
        self.free_slots.push(resident.slot);
        self.resident_bytes -= resident.payload_bytes;
        self.evicted += 1;
        // Zero the header so a stale slot can never alias a live field.
        self.ctx.queue().write_buffer(
            &self.header_buf,
            resident.slot as u64 * HEADER_BYTES,
            &[0u8; HEADER_BYTES as usize],
        );
        true
    }

    /// Shelf allocation: freed boxes first (first-fit), then open shelves,
    /// then a new shelf in an existing slab, then a new slab.
    fn try_alloc(&mut self, padded: [u32; 3]) -> Option<[u32; 3]> {
        if let Some(i) = self.free_boxes.iter().position(|b| {
            b.dims[0] >= padded[0] && b.dims[1] >= padded[1] && b.dims[2] >= padded[2]
        }) {
            return Some(self.free_boxes.swap_remove(i).origin);
        }
        for slab in &mut self.slabs {
            if padded[2] > slab.depth {
                continue;
            }
            for shelf in &mut slab.shelves {
                if padded[1] <= shelf.height && shelf.cursor_x + padded[0] <= self.dims[0] {
                    let origin = [shelf.cursor_x, shelf.y0, slab.z0];
                    shelf.cursor_x += padded[0];
                    return Some(origin);
                }
            }
            if slab.cursor_y + padded[1] <= self.dims[1] {
                let y0 = slab.cursor_y;
                slab.cursor_y += padded[1];
                slab.shelves.push(Shelf {
                    y0,
                    height: padded[1],
                    cursor_x: padded[0],
                });
                return Some([0, y0, slab.z0]);
            }
        }
        if self.cursor_z + padded[2] <= self.dims[2] {
            let z0 = self.cursor_z;
            self.cursor_z += padded[2];
            self.slabs.push(Slab {
                z0,
                depth: padded[2],
                cursor_y: padded[1],
                shelves: vec![Shelf {
                    y0: 0,
                    height: padded[1],
                    cursor_x: padded[0],
                }],
            });
            return Some([0, 0, z0]);
        }
        None
    }
}

/// Snorm8 payload with a 1-voxel replicated border on every face, X-fastest
/// layout matching the field. Snorm16 payloads are re-quantized here
/// (decode i16 → renormalize → round to i8); snorm8 passes through.
fn padded_snorm8_volume(field: &SdfField) -> Vec<u8> {
    let res = field.desc().resolution();
    let (nx, ny, nz) = (res[0] as usize, res[1] as usize, res[2] as usize);
    let src: Vec<u8> = match field.payload() {
        SdfPayload::Snorm8(v) => bytemuck::cast_slice(v).to_vec(),
        SdfPayload::Snorm16(v) => v
            .iter()
            .map(|&raw| {
                let normalized = if raw == i16::MIN {
                    -1.0
                } else {
                    raw as f32 / i16::MAX as f32
                };
                ((normalized * i8::MAX as f32)
                    .round()
                    .clamp(i8::MIN as f32 + 1.0, i8::MAX as f32) as i8) as u8
            })
            .collect(),
    };
    let (px, py, pz) = (nx + 2, ny + 2, nz + 2);
    let mut out = vec![0u8; px * py * pz];
    for z in 0..pz {
        let sz = (z as isize - 1).clamp(0, nz as isize - 1) as usize;
        for y in 0..py {
            let sy = (y as isize - 1).clamp(0, ny as isize - 1) as usize;
            let src_row = (sz * ny + sy) * nx;
            let dst_row = (z * py + y) * px;
            for x in 0..px {
                let sx = (x as isize - 1).clamp(0, nx as isize - 1) as usize;
                out[dst_row + x] = src[src_row + sx];
            }
        }
    }
    out
}

/// One probe for [`SdfAtlasProbe`]: an asset-local position plus the header
/// slot returned by [`SdfAtlasGpu::insert`]. 16 B, matches the WGSL
/// `struct Probe { pos: vec3f, slot: u32 }` layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SdfAtlasProbePoint {
    pub position: [f32; 3],
    pub header_slot: u32,
}

/// The atlas probe kernel: hardware-trilinear samples through a real
/// `FilterMode::Linear` sampler. This is the atlas's first consumer and the
/// canonical binding example — it binds [`SdfAtlasGpu::texture`] +
/// [`SdfAtlasGpu::header_buf`] exactly the way later passes (clipmap
/// composite, contact AO, far imposters) will.
const PROBE_WGSL: &str = r#"
// Mirrors GpuSdfFieldHeader: 12 scalars, 48 B, stride 48 in storage arrays.
struct Header {
    off_x: u32, off_y: u32, off_z: u32,
    res_x: u32, res_y: u32, res_z: u32,
    origin_x: f32, origin_y: f32, origin_z: f32,
    voxel: f32,
    band: f32,
    sign: f32,
}

struct Probe {
    pos: vec3<f32>,
    slot: u32,
}

@group(0) @binding(0) var atlas_tex: texture_3d<f32>;
@group(0) @binding(1) var atlas_samp: sampler;
@group(0) @binding(2) var<storage, read> headers: array<Header>;
@group(0) @binding(3) var<storage, read> probes: array<Probe>;
@group(0) @binding(4) var<storage, read_write> out_d: array<f32>;

@compute @workgroup_size(64)
fn probe_atlas(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&probes)) {
        return;
    }
    let pr = probes[i];
    let h = headers[pr.slot];
    let origin = vec3<f32>(h.origin_x, h.origin_y, h.origin_z);
    // Asset-local metres -> field grid coords -> atlas texel centre -> uvw.
    let g = (pr.pos - origin) / h.voxel;
    let off = vec3<f32>(f32(h.off_x), f32(h.off_y), f32(h.off_z));
    let dims = vec3<f32>(textureDimensions(atlas_tex));
    let uvw = (off + g + vec3<f32>(0.5)) / dims;
    // R8Snorm sample is [-1, 1]; decode = value * narrow_band metres.
    out_d[i] = textureSampleLevel(atlas_tex, atlas_samp, uvw, 0.0).r * h.band;
}
"#;

pub struct SdfAtlasProbe {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    sampler: wgpu::Sampler,
}

impl SdfAtlasProbe {
    pub fn new_with_context(ctx: &GpuContext) -> Result<Self, SdfAtlasError> {
        let device = ctx.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_atlas_probe"),
            source: wgpu::ShaderSource::Wgsl(PROBE_WGSL.into()),
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sdf_atlas_probe"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                storage(2, true),
                storage(3, true),
                storage(4, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sdf_atlas_probe"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sdf_atlas_probe"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("probe_atlas"),
            compilation_options: Default::default(),
            cache: None,
        });
        // THE hardware-trilinear sampler the parity gate measures.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sdf_atlas_trilinear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Ok(Self {
            layout,
            pipeline,
            sampler,
        })
    }

    /// Sample every probe's asset-local position through the atlas with
    /// hardware trilinear filtering; returns decoded metres per probe.
    /// Blocking (tool/test-paced) — owns its encoder, never frame-coupled.
    pub fn sample_blocking(
        &self,
        ctx: &GpuContext,
        atlas: &SdfAtlasGpu,
        probes: &[SdfAtlasProbePoint],
    ) -> Result<Vec<f32>, SdfAtlasError> {
        if probes.is_empty() {
            return Ok(Vec::new());
        }
        let device = ctx.device();
        let queue = ctx.queue();
        let out_bytes = probes.len() as u64 * 4;

        let probes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdf_atlas_probe_points"),
            contents: bytemuck::cast_slice(probes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdf_atlas_probe_out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdf_atlas_probe_staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view = atlas.texture().create_view(&Default::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sdf_atlas_probe"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: atlas.header_buf().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: probes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sdf_atlas_probe"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(probes.len().div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| SdfAtlasError::DeviceLimits(format!("readback channel: {e}")))?
            .map_err(|e| SdfAtlasError::DeviceLimits(format!("readback map: {e:?}")))?;
        let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;
    use glam::Vec3;
    use vox_core::sdf::{SdfDesc, SdfField, SdfPayload, SdfSign};
    use vox_physics::sdf::SdfAssetId;

    /// Build a headless [`GpuContext`] over a real device — the house pattern
    /// from `spectral_gi.rs::try_gpu_context`. Returns `None` (printing the
    /// skip line) without a hardware adapter so GPU-less CI stays green.
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
                label: Some("sdf_atlas_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("device creation on a box with a GPU");
        Some(GpuContext::from_parts(&device, &queue, &info))
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

    enum Shape {
        Sphere(f32),
        Box(Vec3),
    }

    impl Shape {
        fn distance(&self, p: Vec3) -> f32 {
            match self {
                Self::Sphere(r) => p.length() - r,
                Self::Box(half) => {
                    let q = p.abs() - *half;
                    q.max(Vec3::ZERO).length() + q.x.max(q.y).max(q.z).min(0.0)
                }
            }
        }
    }

    enum Quant {
        Snorm8,
        Snorm16,
    }

    /// Analytic SDF on an anisotropic grid centred on the shape — Task 3's
    /// types built CPU-side (no baker dependency). Band = 4 voxels.
    fn analytic_field(shape: Shape, res: [u32; 3], voxel: f32, quant: Quant) -> SdfField {
        let band = voxel * 4.0;
        let origin = Vec3::new(
            -((res[0] - 1) as f32) * 0.5,
            -((res[1] - 1) as f32) * 0.5,
            -((res[2] - 1) as f32) * 0.5,
        ) * voxel;
        let count = res[0] as usize * res[1] as usize * res[2] as usize;
        let mut s8 = Vec::new();
        let mut s16 = Vec::new();
        match quant {
            Quant::Snorm8 => s8.reserve(count),
            Quant::Snorm16 => s16.reserve(count),
        }
        for z in 0..res[2] {
            for y in 0..res[1] {
                for x in 0..res[0] {
                    let p = origin + Vec3::new(x as f32, y as f32, z as f32) * voxel;
                    let n = (shape.distance(p) / band).clamp(-1.0, 1.0);
                    match quant {
                        Quant::Snorm8 => s8.push((n * i8::MAX as f32).round() as i8),
                        Quant::Snorm16 => s16.push((n * i16::MAX as f32).round() as i16),
                    }
                }
            }
        }
        let payload = match quant {
            Quant::Snorm8 => SdfPayload::Snorm8(s8),
            Quant::Snorm16 => SdfPayload::Snorm16(s16),
        };
        let desc =
            SdfDesc::new(res, origin, voxel, band, SdfSign::Closed).expect("valid test desc");
        SdfField::new(desc, payload).expect("payload count matches")
    }

    /// 5 mixed-resolution fields through the real R8Snorm atlas + a hardware
    /// `FilterMode::Linear` sampler in the probe kernel, against
    /// `SdfField::sample_local` over 100k deterministic points. The kernel is
    /// the atlas's first consumer — it binds `texture()` + `header_buf()`
    /// exactly as later passes will.
    #[test]
    fn sdf_atlas_parity_hw_trilinear() {
        let Some(ctx) = try_gpu_context("sdf_atlas") else {
            return;
        };
        // Mixed resolutions ≤ 64³, both payload quantizations accepted.
        let fields = [
            analytic_field(Shape::Sphere(1.0), [64, 64, 64], 0.05, Quant::Snorm8),
            analytic_field(
                Shape::Box(Vec3::new(0.6, 0.4, 0.3)),
                [48, 40, 56],
                0.04,
                Quant::Snorm8,
            ),
            analytic_field(Shape::Sphere(0.5), [33, 57, 41], 0.035, Quant::Snorm16),
            analytic_field(
                Shape::Box(Vec3::splat(0.45)),
                [24, 24, 24],
                0.06,
                Quant::Snorm16,
            ),
            analytic_field(Shape::Sphere(0.8), [64, 32, 48], 0.045, Quant::Snorm8),
        ];

        let mut atlas = SdfAtlasGpu::new_with_context(&ctx, 64 << 20).expect("atlas");
        let mut slots = Vec::new();
        for (i, field) in fields.iter().enumerate() {
            slots.push(
                atlas
                    .insert(SdfAssetId(1000 + i as u64), field)
                    .expect("insert under quota"),
            );
        }
        let mut unique = slots.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            fields.len(),
            "each asset gets its own header slot"
        );

        // 100k deterministic hash-jittered points, 20k per field, inside the
        // sampleable local bounds so the CPU oracle always answers.
        const PER_FIELD: usize = 20_000;
        let mut rng = SplitMix(0xA71A5_BA5E);
        let mut probes = Vec::with_capacity(PER_FIELD * fields.len());
        for (fi, field) in fields.iter().enumerate() {
            let (lo, hi) = field.desc().local_bounds();
            let extent = hi - lo;
            for _ in 0..PER_FIELD {
                let p = lo + Vec3::new(rng.unit(), rng.unit(), rng.unit()) * extent;
                probes.push(SdfAtlasProbePoint {
                    position: p.to_array(),
                    header_slot: slots[fi],
                });
            }
        }

        let probe = SdfAtlasProbe::new_with_context(&ctx).expect("probe kernel");
        let gpu_d = probe
            .sample_blocking(&ctx, &atlas, &probes)
            .expect("probe dispatch + readback");
        assert_eq!(gpu_d.len(), probes.len());

        let mut max_dev = 0.0f32;
        for (i, pt) in probes.iter().enumerate() {
            let field = &fields[i / PER_FIELD];
            let cpu = field
                .sample_local(Vec3::from(pt.position))
                .expect("probe inside local bounds");
            max_dev = max_dev.max((gpu_d[i] - cpu).abs());
        }

        let mb = atlas.resident_bytes() as f64 / 1.0e6;
        println!(
            "[sdf_atlas] {} assets packed, {mb:.1} MB | gpu-vs-cpu max |Δd| = {max_dev:.6} m over {} samples",
            fields.len(),
            probes.len()
        );

        let min_voxel = fields
            .iter()
            .map(|f| f.desc().voxel_size())
            .fold(f32::INFINITY, f32::min);
        assert!(
            max_dev < min_voxel * 0.5,
            "hardware trilinear must match SdfField::sample_local within half the \
             smallest voxel: dev={max_dev} voxel={min_voxel}"
        );
        assert!(
            max_dev > 0.0,
            "a 0.0 deviation means the GPU path was not actually sampled"
        );
    }

    /// The design §4.2 memory math, measured: 200 unique ~64×56×51 snorm8
    /// assets ≈ 36 MB resident under the 64 MB quota with zero evictions;
    /// re-run at quota 24 MB the atlas must evict (least-recently-inserted)
    /// and stay under quota.
    #[test]
    fn sdf_atlas_quota_eviction() {
        let Some(ctx) = try_gpu_context("sdf_atlas") else {
            return;
        };
        // ~183 KB per asset: 64×56×51 snorm8 (the design's projection shape).
        let field = analytic_field(Shape::Sphere(4.0), [64, 56, 51], 0.2, Quant::Snorm8);
        assert_eq!(field.desc().sample_count(), 182_784, "≈183 KB fixture");

        let mut atlas = SdfAtlasGpu::new_with_context(&ctx, 64 << 20).expect("atlas");
        let mut slots = std::collections::HashSet::new();
        for i in 0..200u64 {
            slots.insert(
                atlas
                    .insert(SdfAssetId(i), &field)
                    .expect("insert under the 64 MB quota"),
            );
        }
        assert_eq!(slots.len(), 200, "no eviction ⇒ 200 distinct header slots");
        // Re-insert of a resident asset returns its slot without growing.
        let bytes_before = atlas.resident_bytes();
        let again = atlas.insert(SdfAssetId(7), &field).expect("re-insert");
        assert!(
            slots.contains(&again),
            "re-insert returns the resident slot"
        );
        assert_eq!(atlas.resident_bytes(), bytes_before, "re-insert is a no-op");

        let resident = atlas.resident_bytes();
        println!(
            "[sdf_atlas] 200 assets requested | resident={:.1} MB (quota {}) | evicted={}",
            resident as f64 / 1.0e6,
            (64u64 << 20) >> 20,
            atlas.evicted_count()
        );
        assert_eq!(atlas.evicted_count(), 0, "200 × 183 KB fits a 64 MB quota");
        assert!(
            (34.0..38.0).contains(&(resident as f64 / 1.0e6)),
            "design §4.2: 200 assets ≈ 36 MB, got {:.1} MB",
            resident as f64 / 1.0e6
        );

        // Forced-eviction phase: the same 200 assets through a 24 MB quota.
        let mut small = SdfAtlasGpu::new_with_context(&ctx, 24 << 20).expect("small atlas");
        for i in 0..200u64 {
            small
                .insert(SdfAssetId(i), &field)
                .expect("insert with forced eviction");
        }
        let resident = small.resident_bytes();
        println!(
            "[sdf_atlas] quota 24 MB: resident={:.1} MB | evicted={}",
            resident as f64 / 1.0e6,
            small.evicted_count()
        );
        assert!(
            small.evicted_count() > 0,
            "200 × 183 KB cannot all stay resident under 24 MB"
        );
        assert!(
            resident <= 24 << 20,
            "resident bytes must respect the quota: {resident}"
        );
    }
}
