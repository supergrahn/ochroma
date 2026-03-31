//! Boids flocking simulation using AgentComputeLayer.
//! Run: cargo run --example flocking -p vox_agent

fn main() {
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no GPU adapter found");
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("flocking"),
                    required_features: wgpu::Features::empty(),
                    // Limits::default() required: spatial hash adds bindings 6-8 (3 storage),
                    // plus positions_in/out + velocities_in/out + flags = 8 storage total,
                    // which exceeds downlevel_defaults() cap of 4.
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("device creation failed")
    });

    let n = 10_000u32;
    let desc = vox_agent::AgentStateDesc {
        agent_count: n,
        custom_floats: 0,
        spectral: false,
        spatial_hash: Some(vox_agent::SpatialHashDesc {
            grid_origin_x: -500.0,
            grid_origin_z: -500.0,
            grid_extent: 1000.0,
            cell_size: 25.0,
        }),
    };

    let mut layer = vox_agent::AgentComputeLayer::new(&device, desc.clone());

    // Boids shader: separation + alignment + cohesion using spatial hash
    let boids_wgsl = format!(
        "{}\n{}",
        layer.bind_group_layout_source(),
        include_str!("boids_behavior.wgsl"),
    );
    layer
        .load_shader(&device, vox_agent::ShaderSource::Wgsl(boids_wgsl))
        .expect("boids shader failed to load");

    // Initialize agents in a 100x100 grid
    let mut positions = Vec::with_capacity(n as usize);
    let mut velocities = Vec::with_capacity(n as usize);
    for i in 0..n {
        let x = (i % 100) as f32 * 1.0 - 50.0;
        let z = (i / 100) as f32 * 1.0 - 50.0;
        positions.push([x, 0.0, z]);
        velocities.push([0.1, 0.0, 0.0]);
    }
    layer.buffers_mut().upload_positions(&queue, &positions);
    layer.buffers_mut().upload_velocities(&queue, &velocities);
    layer.buffers_mut().mark_all_alive(&queue);
    queue.submit([]);

    println!("Running Boids with {} agents for 300 frames...", n);
    let mut total_ms = 0.0f64;

    for frame in 0..300usize {
        let t0 = std::time::Instant::now();
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("boids") },
        );
        layer.tick(&device, &mut encoder, &queue, None, 1.0 / 60.0);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_ms += elapsed_ms;

        if frame % 60 == 0 {
            println!("frame {:>3}: {:.2}ms", frame, elapsed_ms);
        }
    }

    println!("avg: {:.2}ms over 300 frames ({} agents)", total_ms / 300.0, n);
}
