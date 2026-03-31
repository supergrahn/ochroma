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
        // Use Limits::default() (not downlevel_defaults()) — the behavior pipeline uses 5
        // storage buffers (positions_in/out, velocities_in/out, flags), which exceeds the
        // 4-buffer cap in downlevel_defaults.
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

    // Upload non-zero velocities so agent 0 will actually move during the tick.
    let velocities = vec![[0.01f32, 0.0, 0.0]; 1000];
    layer.buffers_mut().upload_velocities(&queue, &velocities);
    layer.buffers_mut().mark_all_alive(&queue);
    queue.submit([]);

    let mut encoder = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("smoke") });
    layer.tick(&device, &mut encoder, &queue, None, 1.0 / 60.0);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);

    // Verify agent 0 moved: read back from the current read buffer (after swap).
    // After tick(), buffers have been swapped — read_positions() holds the written output.
    let readback_size = 3 * 4u64; // 3 f32 for one agent
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("smoke_readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder2.copy_buffer_to_buffer(layer.buffers_mut().read_positions(), 0, &readback, 0, readback_size);
    queue.submit([encoder2.finish()]);
    device.poll(wgpu::Maintain::Wait);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    // Agent 0 had velocity=[0.01, 0, 0] and dt=1/60, so x should be ~0.01/60 ≈ 0.000167
    assert!(
        data[0].abs() > 1e-6,
        "agent 0 x position must have changed after dispatch, got {}",
        data[0]
    );
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

    // Warm-up: one frame to trigger any JIT compilation or driver lazy-init; not counted.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("warmup") });
    layer.tick(&device, &mut encoder, &queue, None, 1.0 / 60.0);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);

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
