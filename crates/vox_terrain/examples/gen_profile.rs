//! Profiling + determinism-witness harness for the ASSEMBLY/GEN hot paths.
//!
//! Run: `cargo run --release --example gen_profile -p vox_terrain`
//!
//! Prints per-path wall-clock timing + an FNV-1a hash of the byte-exact output.
//! The hashes are the DETERMINISM WITNESS — they must be byte-identical before
//! and after any optimization. Floats are hashed via their raw bits (bytemuck /
//! to_bits), so any change in the produced numbers changes the hash.

use std::time::Instant;
use vox_terrain::foliage::{default_foliage_rules, scatter_foliage};
use vox_terrain::heightmap::{default_zones, generate_test_heightmap};
use vox_terrain::volume::{default_volume_materials, generate_demo_volume, volume_to_splats};

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Hash the byte-exact contents of a slice of Pod values.
fn hash_pod<T: bytemuck::Pod>(v: &[T]) -> u64 {
    fnv1a(bytemuck::cast_slice(v))
}

/// Hash foliage instances by their float bits + lengths of the names (the
/// numeric placement is what determinism guards; names are stable strings).
fn hash_foliage(instances: &[vox_terrain::foliage::FoliageInstance]) -> u64 {
    let mut h = FNV_OFFSET;
    for inst in instances {
        for f in inst.position {
            h ^= f.to_bits() as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h ^= inst.rotation_y.to_bits() as u64;
        h = h.wrapping_mul(FNV_PRIME);
        h ^= inst.scale.to_bits() as u64;
        h = h.wrapping_mul(FNV_PRIME);
        // include the resolved asset path so a name regression is caught too
        for b in inst.asset_path.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

fn bench<F: FnMut() -> R, R>(name: &str, iters: u32, mut f: F) -> R {
    // warm
    let mut last = f();
    let t = Instant::now();
    for _ in 0..iters {
        last = f();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("  {name:<34} {ms:>9.3} ms/iter  ({iters} iters)");
    last
}

fn main() {
    println!("== gen_profile (release) ==");

    // ---- #1 scatter_foliage --------------------------------------------
    // Map-representative: 300x300 cells @ 3.3m, default 4 rules, seed 42.
    println!("\n[scatter_foliage] 300x300 @3.3m, default rules, seed 42");
    let hm_fol = generate_test_heightmap(300, 300, 3.3, 11);
    let rules = default_foliage_rules();
    let inst = bench("scatter_foliage", 5, || {
        scatter_foliage(&hm_fol, &rules, 42)
    });
    println!(
        "  -> n={}  WITNESS scatter_foliage = {:#018x}",
        inst.len(),
        hash_foliage(&inst)
    );

    // ---- #3 generate_test_heightmap ------------------------------------
    println!("\n[generate_test_heightmap] 2048x2048 @1.0m seed 7");
    let hm = bench("generate_test_heightmap", 5, || {
        generate_test_heightmap(2048, 2048, 1.0, 7)
    });
    println!(
        "  -> cells={}  WITNESS heightmap_data = {:#018x}",
        hm.data.len(),
        fnv1a(bytemuck::cast_slice(&hm.data))
    );

    // ---- #2 to_splats --------------------------------------------------
    println!("\n[to_splats] 512x512 @1.0m, default zones, 1 spc");
    let hm_sp = generate_test_heightmap(512, 512, 1.0, 7);
    let zones = default_zones();
    let splats = bench("to_splats", 5, || hm_sp.to_splats(&zones, 1));
    println!(
        "  -> splats={}  WITNESS to_splats = {:#018x}",
        splats.len(),
        hash_pod(&splats)
    );

    // ---- #4 volume_to_splats -------------------------------------------
    println!("\n[volume_to_splats] demo volume, default materials, seed 42");
    let vol = generate_demo_volume(0);
    let mats = default_volume_materials();
    let vsplats = bench("volume_to_splats", 5, || volume_to_splats(&vol, &mats, 42));
    println!(
        "  -> splats={}  WITNESS volume_to_splats = {:#018x}",
        vsplats.len(),
        hash_pod(&vsplats)
    );
}
