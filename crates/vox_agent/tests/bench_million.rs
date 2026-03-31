//! Integration test: smoke + optional GPU benchmark.
//! Smoke test: cargo test -p vox_agent --test bench_million
//! Benchmark:  cargo test -p vox_agent --test bench_million -- --nocapture --ignored

fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        // Use Vulkan-only to avoid the WSL2 DX12/WARP driver crash on adapter
        // enumeration. Falls back gracefully (returns None) when no Vulkan GPU.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("vox_agent_bench"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .ok()?;
        Some((device, queue))
    })
}

#[test]
fn layer_dispatches_without_panic() {
    let Some((device, queue)) = test_device() else { return; };

    let desc = vox_agent::AgentStateDesc {
        agent_count: 1000,
        custom_floats: 0,
        spectral: false,
        spatial_hash: None,
    };
    let mut layer = vox_agent::AgentComputeLayer::new(&device, desc);
    layer.load_default_shader(&device).expect("default shader");
    layer.buffers_mut().mark_all_alive(&queue);
    queue.submit([]);

    let mut encoder = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("smoke") });
    layer.tick(&device, &mut encoder, &queue, None, 1.0 / 60.0);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);
}

#[test]
#[ignore = "requires GPU; run with: cargo test --test bench_million -- --ignored --nocapture"]
fn bench_million() {
    let Some((device, queue)) = test_device() else {
        println!("SKIP: no GPU available");
        return;
    };

    let n = 1_000_000u32;
    let desc = vox_agent::AgentStateDesc {
        agent_count: n,
        custom_floats: 0,
        spectral: false,
        spatial_hash: None,
    };

    let mut layer = vox_agent::AgentComputeLayer::new(&device, desc);
    layer.load_default_shader(&device).expect("default shader must load");

    let positions = vec![[0.0f32; 3]; n as usize];
    let velocities = vec![[0.01f32, 0.0, 0.0]; n as usize];
    layer.buffers_mut().upload_positions(&queue, &positions);
    layer.buffers_mut().upload_velocities(&queue, &velocities);
    layer.buffers_mut().mark_all_alive(&queue);
    queue.submit([]);

    let frames = 120;
    let mut total_ms = 0.0f64;

    for _ in 0..frames {
        let t0 = std::time::Instant::now();
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("bench") });
        layer.tick(&device, &mut encoder, &queue, None, 1.0 / 60.0);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }

    let avg_ms = total_ms / frames as f64;
    println!("agents: {:>10}  avg_frame_ms: {:.2}  min_fps: {:.0}  dispatch: GPU",
        n, avg_ms, 1000.0 / avg_ms);

    assert!(avg_ms <= 16.0,
        "avg frame time {:.2}ms exceeds 16ms budget for {}M agents", avg_ms, n / 1_000_000);
}
