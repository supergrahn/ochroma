//! Headless GPU proof for the material compiler.
//!
//! Takes a [`MaterialDag`](crate::material_compiler::MaterialDag)'s compiled
//! WGSL, runs `material_eval` for all 16 bands at a set of cos_theta values on a
//! real compute device, and reads the results back. The equivalence test
//! compares this against [`MaterialDag::eval_cpu`] within f32 epsilon — this is
//! the "compiles to the spectral pipeline" proof.
//!
//! Mirrors the headless-device pattern in `spectral_gi.rs` (instance → adapter →
//! device → compute pass → map_async readback). Returns `Err(NoAdapter)` when no
//! GPU is available so tests can skip cleanly.

use crate::material_compiler::N_BANDS;
use vox_core::spectral::BAND_WAVELENGTHS;

/// Errors from the headless material-eval harness.
#[derive(Debug)]
pub enum GpuMaterialError {
    /// No GPU adapter available (tests should skip).
    NoAdapter,
    /// Device creation failed.
    DeviceCreation(String),
    /// Shader module creation / validation failed.
    Shader(String),
    /// Buffer readback failed.
    Readback(String),
}

impl std::fmt::Display for GpuMaterialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no GPU adapter available"),
            Self::DeviceCreation(e) => write!(f, "device creation failed: {e}"),
            Self::Shader(e) => write!(f, "shader error: {e}"),
            Self::Readback(e) => write!(f, "readback failed: {e}"),
        }
    }
}

impl std::error::Error for GpuMaterialError {}

/// A headless compute device that runs compiled material WGSL.
pub struct GpuMaterialEval {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub adapter_name: String,
}

impl GpuMaterialEval {
    /// Create the headless device, or `Err(NoAdapter)` if none is present.
    pub fn new() -> Result<Self, GpuMaterialError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuMaterialError> {
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
            .ok_or(GpuMaterialError::NoAdapter)?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("material_eval_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuMaterialError::DeviceCreation(e.to_string()))?;
        Ok(Self {
            device,
            queue,
            adapter_name,
        })
    }

    /// Wrap compiled `material_eval` WGSL with a compute entry point and run it
    /// for all 16 bands at `cos_theta`, returning the per-band output.
    pub fn eval_spd(
        &self,
        material_wgsl: &str,
        cos_theta: f32,
    ) -> Result<[f32; N_BANDS], GpuMaterialError> {
        let lambdas = BAND_WAVELENGTHS;
        let lambda_lits: Vec<String> = lambdas.iter().map(|l| format!("{l:?}")).collect();

        let full_wgsl = format!(
            r#"{material}

const BAND_LAMBDAS = array<f32, 16>({lambdas});

struct Params {{ cos_theta: f32, _pad0: f32, _pad1: f32, _pad2: f32 }};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_spd: array<f32, 16>;

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let band = gid.x;
    if (band >= 16u) {{ return; }}
    out_spd[band] = material_eval(band, BAND_LAMBDAS[band], params.cos_theta);
}}
"#,
            material = material_wgsl,
            lambdas = lambda_lits.join(", "),
        );

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("material_eval_module"),
                source: wgpu::ShaderSource::Wgsl(full_wgsl.into()),
            });

        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("material_eval_bgl"),
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
                ],
            });

        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("material_eval_pl"),
                    bind_group_layouts: &[&bgl],
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("material_eval_pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Params uniform: cos_theta + padding to 16 bytes.
        let params = [cos_theta, 0.0, 0.0, 0.0];
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

        let out_size = (N_BANDS * 4) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_spd"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_eval_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("material_eval_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("material_eval_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(GpuMaterialError::Readback(e.to_string())),
            Err(e) => return Err(GpuMaterialError::Readback(e.to_string())),
        }

        let result: [f32; N_BANDS] = {
            let data = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&data);
            let mut out = [0.0f32; N_BANDS];
            out.copy_from_slice(&floats[..N_BANDS]);
            out
        };
        readback.unmap();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material_compiler::{BsdfNode, MaterialDag};

    fn distinct_graphs() -> Vec<(&'static str, MaterialDag)> {
        vec![
            (
                "layer_red_over_grey",
                MaterialDag {
                    nodes: vec![
                        BsdfNode::RgbUplift { rgb: [0.9, 0.1, 0.1] },
                        BsdfNode::SpectralConstant { spd: [0.18; 16] },
                        BsdfNode::Layer { coat: 0, base: 1, f0: 0.04 },
                    ],
                    output: 2,
                },
            ),
            (
                "blackbody_4000k_scaled",
                MaterialDag {
                    nodes: vec![
                        BsdfNode::BlackbodyEmitter { kelvin: 4000.0 },
                        BsdfNode::Scalar { value: 0.7 },
                        BsdfNode::Multiply { a: 0, b: 1 },
                    ],
                    output: 2,
                },
            ),
            (
                "fresnel_mix_wavelength",
                MaterialDag {
                    nodes: vec![
                        BsdfNode::SpectralConstant {
                            spd: [
                                0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.62, 0.63,
                                0.64, 0.64, 0.65, 0.65,
                            ],
                        },
                        BsdfNode::Fresnel { base: 0, power: 5.0 },
                        BsdfNode::RgbUplift { rgb: [0.2, 0.6, 0.9] },
                        BsdfNode::Mix { a: 1, b: 2, factor: 0.35 },
                    ],
                    output: 3,
                },
            ),
        ]
    }

    /// GPU equivalence: compiled WGSL output == CPU eval for all 16 bands across
    /// 3 cos_theta values and 3 distinct graphs, within 1e-5. Skips without an
    /// adapter. Prints the real per-band values.
    #[test]
    fn gpu_equals_cpu_per_band() {
        let gpu = match GpuMaterialEval::new() {
            Ok(g) => g,
            Err(GpuMaterialError::NoAdapter) => {
                eprintln!("SKIP gpu_equals_cpu_per_band: no GPU adapter");
                return;
            }
            Err(e) => panic!("unexpected GPU init error: {e}"),
        };
        eprintln!("GPU adapter: {}", gpu.adapter_name);

        let cosines = [1.0f32, 0.5, 0.05];
        let mut max_err = 0.0f32;
        for (name, dag) in distinct_graphs() {
            let wgsl = dag.compile_to_wgsl().expect("compile");
            for &cos in &cosines {
                let cpu = dag.eval_cpu_spd(cos);
                let gpu_spd = gpu.eval_spd(&wgsl, cos).expect("gpu eval");
                eprintln!("--- graph={name} cos_theta={cos} ---");
                for band in 0..N_BANDS {
                    let err = (cpu[band] - gpu_spd[band]).abs();
                    max_err = max_err.max(err);
                    eprintln!(
                        "  band {band:>2} (λ={:>5.1}nm): cpu={:.8} gpu={:.8} |Δ|={:.2e}",
                        BAND_WAVELENGTHS[band],
                        cpu[band],
                        gpu_spd[band],
                        err
                    );
                    assert!(
                        err < 1e-5,
                        "band {band} graph {name} cos {cos}: cpu={} gpu={} err={}",
                        cpu[band],
                        gpu_spd[band],
                        err
                    );
                }
            }
        }
        eprintln!("max per-band |Δ| across all graphs/angles = {max_err:.3e}");
    }
}
