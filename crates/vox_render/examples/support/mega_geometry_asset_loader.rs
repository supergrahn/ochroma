//! Loader for cooked `ReadyAssetPayload` assets (`.atoms.json` / `.ready.json`)
//! into the bench's `BlasDesc` + `PbrMaterial` types, so the MegaGeometry
//! fixtures render the REAL Forge-cooked geometry (with its LOD chain) instead
//! of a synthetic unit cube.
//!
//! Parsed generically via `serde_json::Value` so this vox_render example does
//! NOT depend on the `urban_horizon` `ReadyAssetPayload` type (that would be a
//! circular dependency). Only the fields the bench needs are read; the rest of
//! the (large) payload is ignored.
//!
//! Asset id → file resolution (root from `MEGAGEOMETRY_ASSET_ROOT`, default
//! `assets`):
//!   - `city.*`  → `<root>/buildings/forge_starter/atoms/<asset_id>.atoms.json`
//!   - `ph.*`    → `<root>/ready_prototypes/<asset_id>.ready.json`
//!   - anything else (e.g. `urban_horizon.live_city`) → no file (caller falls back).

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use vox_render::splat_backend::{BlasDesc, PbrMaterial};

/// A cooked asset resolved to bench geometry: the base (full-detail) mesh, its
/// LOD chain (progressively coarser meshes), the material palette, and a content
/// hash for fixture authorization.
pub struct CookedAsset {
    /// Full-detail mesh (LOD 0 / base).
    pub base: BlasDesc,
    /// Progressively coarser LOD meshes, in cook order (`mesh_lods`).
    pub lods: Vec<BlasDesc>,
    /// Material palette (`materials[*]`), indexed by each triangle's material id.
    pub materials: Vec<PbrMaterial>,
    /// Hex of the cooked content hash (`mega_geometry.content_hash`, else an FNV
    /// of the base positions) — the fixture's `asset_content_hash`.
    pub content_hash: String,
}

impl CookedAsset {
    /// The mesh to trace at a given LOD level, clamped to the available chain.
    /// `level == 0` is the base (full detail); higher levels are coarser.
    pub fn lod(&self, level: usize) -> &BlasDesc {
        if level == 0 || self.lods.is_empty() {
            &self.base
        } else {
            &self.lods[(level - 1).min(self.lods.len() - 1)]
        }
    }

    /// Triangle count of the full-detail base mesh.
    pub fn base_triangles(&self) -> usize {
        self.base.indices.len()
    }
}

fn f3(v: &serde_json::Value) -> [f32; 3] {
    let a = v.as_array();
    let g = |i: usize| {
        a.and_then(|a| a.get(i))
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0) as f32
    };
    [g(0), g(1), g(2)]
}

fn f2(v: &serde_json::Value) -> [f32; 2] {
    let a = v.as_array();
    let g = |i: usize| {
        a.and_then(|a| a.get(i))
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0) as f32
    };
    [g(0), g(1)]
}

fn u3(v: &serde_json::Value) -> [u32; 3] {
    let a = v.as_array();
    let g = |i: usize| {
        a.and_then(|a| a.get(i))
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as u32
    };
    [g(0), g(1), g(2)]
}

fn parse_mesh(m: &serde_json::Value, proto_id: u64) -> Result<BlasDesc, String> {
    let arr = |k: &str| m.get(k).and_then(|v| v.as_array());
    let positions: Vec<[f32; 3]> = arr("positions")
        .ok_or("mesh.positions missing")?
        .iter()
        .map(f3)
        .collect();
    if positions.is_empty() {
        return Err("mesh has no positions".into());
    }
    let indices: Vec<[u32; 3]> = arr("indices")
        .ok_or("mesh.indices missing")?
        .iter()
        .map(u3)
        .collect();
    if indices.is_empty() {
        return Err("mesh has no indices".into());
    }
    // Normals / UVs default to a per-vertex fill when absent or a mismatched
    // length (a flat lit surface + zero UV is fine for the geometry benchmark).
    let normals: Vec<[f32; 3]> = arr("normals")
        .map(|a| a.iter().map(f3).collect::<Vec<_>>())
        .filter(|n| n.len() == positions.len())
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
    let uvs: Vec<[f32; 2]> = arr("uvs")
        .map(|a| a.iter().map(f2).collect::<Vec<_>>())
        .filter(|u| u.len() == positions.len())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
    // One material id per triangle (relative to the per-instance base).
    let material_ids: Vec<u32> = arr("material_ids")
        .map(|a| {
            a.iter()
                .map(|x| x.as_u64().unwrap_or(0) as u32)
                .collect::<Vec<_>>()
        })
        .filter(|m| m.len() == indices.len())
        .unwrap_or_default();

    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in &positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    Ok(BlasDesc {
        proto_id,
        positions,
        normals,
        uvs,
        indices,
        material_ids,
        aabb_min: lo,
        aabb_max: hi,
        weathering_masks: Vec::new(),
    })
}

fn parse_materials(v: &serde_json::Value) -> Vec<PbrMaterial> {
    let Some(arr) = v.as_array() else {
        return vec![PbrMaterial {
            base_color: [0.6, 0.6, 0.6],
            roughness: 0.6,
            ..Default::default()
        }];
    };
    let out: Vec<PbrMaterial> = arr
        .iter()
        .map(|m| {
            let bc = m.get("base_color_factor").and_then(|v| v.as_array());
            let g = |i: usize| {
                bc.and_then(|a| a.get(i))
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.6) as f32
            };
            PbrMaterial {
                base_color: [g(0), g(1), g(2)],
                roughness: m
                    .get("roughness_factor")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.6) as f32,
                metallic: m
                    .get("metallic_factor")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0) as f32,
                ..Default::default()
            }
        })
        .collect();
    if out.is_empty() {
        vec![PbrMaterial {
            base_color: [0.6, 0.6, 0.6],
            roughness: 0.6,
            ..Default::default()
        }]
    } else {
        out
    }
}

/// Root directory for cooked assets (`MEGAGEOMETRY_ASSET_ROOT`, default `assets`).
fn asset_root() -> String {
    std::env::var("MEGAGEOMETRY_ASSET_ROOT").unwrap_or_else(|_| "assets".to_string())
}

/// Resolve a fixture `asset_id` to its cooked file path, or `None` when the id
/// has no single cooked file (e.g. the externally-authorized `mixed_city`).
pub fn resolve_asset_path(asset_id: &str) -> Option<String> {
    let root = asset_root();
    if let Some(rest) = asset_id.strip_prefix("ph.") {
        Some(format!("{root}/ready_prototypes/ph.{rest}.ready.json"))
    } else if asset_id.starts_with("city.") {
        Some(format!(
            "{root}/buildings/forge_starter/atoms/{asset_id}.atoms.json"
        ))
    } else {
        None
    }
}

fn parse_cooked(bytes: &[u8], proto_id: u64) -> Result<CookedAsset, String> {
    let v: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| format!("json parse: {e}"))?;
    let base = parse_mesh(v.get("mesh").ok_or("payload has no .mesh")?, proto_id)?;
    let mut lods = Vec::new();
    if let Some(a) = v.get("mesh_lods").and_then(|x| x.as_array()) {
        for (i, lod) in a.iter().enumerate() {
            // mesh_lods entries wrap the mesh under `.mesh`; tolerate a bare mesh.
            let mesh = lod.get("mesh").unwrap_or(lod);
            lods.push(parse_mesh(mesh, proto_id.wrapping_mul(1000) + i as u64 + 1)?);
        }
    }
    let materials = parse_materials(v.get("materials").unwrap_or(&serde_json::Value::Null));
    // Prefer the cooked mega_geometry content hash ([u8;32]); else FNV the base
    // positions so the fixture still carries a real content certification.
    let content_hash = v
        .get("mega_geometry")
        .and_then(|mg| mg.get("content_hash"))
        .and_then(|h| h.as_array())
        .filter(|a| !a.is_empty())
        .map(|a| {
            a.iter()
                .map(|x| format!("{:02x}", x.as_u64().unwrap_or(0) as u8))
                .collect::<String>()
        })
        .unwrap_or_else(|| {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for p in &base.positions {
                for c in p {
                    h ^= c.to_bits() as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            format!("{h:016x}")
        });
    Ok(CookedAsset {
        base,
        lods,
        materials,
        content_hash,
    })
}

fn cache() -> &'static Mutex<HashMap<String, Arc<CookedAsset>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<CookedAsset>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load (and cache) the cooked asset for a fixture `asset_id`. The large JSON is
/// parsed once per id and shared across the three modes / all instances. Returns
/// `Err` when the id has no cooked file or the file is missing/malformed.
pub fn load_fixture_asset(asset_id: &str, proto_id: u64) -> Result<Arc<CookedAsset>, String> {
    if let Some(hit) = cache().lock().unwrap().get(asset_id).cloned() {
        return Ok(hit);
    }
    let path = resolve_asset_path(asset_id)
        .ok_or_else(|| format!("no cooked file maps to asset id '{asset_id}'"))?;
    let bytes = std::fs::read(&path).map_err(|e| format!("read '{path}': {e}"))?;
    let asset = Arc::new(parse_cooked(&bytes, proto_id)?);
    cache()
        .lock()
        .unwrap()
        .insert(asset_id.to_string(), asset.clone());
    Ok(asset)
}
