//! AAA Spec 05, Step 1 — the keystone tile-range-build pass is bit-exact against
//! its CPU oracle on real hardware. Skips gracefully on a no-GPU / software-only
//! lane so CI stays green.

use vox_render::gpu::tile_range_build::{cpu_tile_ranges, TileRangeBuildPass};
use wgpu::util::DeviceExt;

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    // Use local GPU: a real hardware adapter only (skip llvmpipe).
    if vox_render::gpu::adapter::ensure_hardware(&adapter.get_info()).is_err() {
        eprintln!("[tile_range_build_test] software adapter — SKIPPED");
        return None;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("tile_range_build_test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;
    Some((device, queue))
}

#[test]
fn tile_range_matches_cpu_oracle() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[tile_range_build_test] no adapter — SKIPPED");
        return;
    };

    // Radix-sorted tile ids: tile 0 twice, tile 2 thrice, tile 5 once, over 6 tiles.
    let sorted: [u32; 6] = [0, 0, 2, 2, 2, 5];
    let num_tiles = 6u32;

    let keys = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sorted_keys_hi"),
        contents: bytemuck::cast_slice(&sorted),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let ranges_bytes = (num_tiles as u64) * 8; // vec2<u32>
    let ranges = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile_ranges"),
        size: ranges_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile_ranges_readback"),
        size: ranges_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pass = TileRangeBuildPass::new(&device);
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
    pass.dispatch(&device, &mut encoder, &keys, &ranges, sorted.len() as u32, num_tiles);
    encoder.copy_buffer_to_buffer(&ranges, 0, &readback, 0, ranges_bytes);
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().expect("map channel").expect("map ok");
    let data = slice.get_mapped_range();
    let flat: &[u32] = bytemuck::cast_slice(&data);
    let gpu: Vec<[u32; 2]> = flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    drop(data);
    readback.unmap();

    let cpu = cpu_tile_ranges(&sorted, num_tiles as usize);
    eprintln!("[tile_range_build_test] gpu={gpu:?} cpu={cpu:?}");

    // Bit-exact vs the oracle AND the hand-worked literal.
    assert_eq!(gpu, cpu, "GPU tile ranges must match the CPU oracle exactly");
    assert_eq!(
        gpu,
        vec![[0, 2], [0, 0], [2, 5], [0, 0], [0, 0], [5, 6]],
        "GPU tile ranges must match the hand-worked spans"
    );
}
