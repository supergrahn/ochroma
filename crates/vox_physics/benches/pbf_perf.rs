use criterion::{criterion_group, criterion_main, Criterion};
use vox_physics::pbf::{PbfFluidSim, WATER_SPECTRAL};

fn bench_pbf_cpu_1k(c: &mut Criterion) {
    let mut sim = PbfFluidSim::new(1000.0, 0.08);
    for i in 0..1000 {
        sim.spawn(
            [
                (i % 10) as f32 * 0.1,
                (i / 100) as f32 * 0.1 + 1.0,
                (i / 10 % 10) as f32 * 0.1,
            ],
            [0.0; 3],
            WATER_SPECTRAL,
        );
    }
    c.bench_function("pbf_cpu_1k_particles", |b| b.iter(|| sim.cpu_step()));
}

fn bench_spectral_fluid_500(c: &mut Criterion) {
    use vox_physics::fluid::{SpectralFluid, SpectralFluidKind};
    let mut fluid = SpectralFluid::new(SpectralFluidKind::Water);
    for i in 0..500 {
        fluid.spawn(
            [(i % 10) as f32 * 0.1, (i / 10) as f32 * 0.1 + 1.0, 0.0],
            [0.0; 3],
        );
    }
    c.bench_function("spectral_fluid_500_particles", |b| b.iter(|| fluid.step()));
}

criterion_group!(benches, bench_pbf_cpu_1k, bench_spectral_fluid_500);
criterion_main!(benches);
