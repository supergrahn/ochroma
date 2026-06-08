use serde::{Serialize, Deserialize};
use std::path::Path;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldSave {
    pub version: u32,
    pub engine_version: String,
    pub timestamp: String,
    pub scene_name: String,
    pub entities: Vec<SavedEntity>,
    pub resources: SavedResources,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedEntity {
    pub name: String,
    pub position: [f32; 3],
    pub rotation: [f32; 4],     // quaternion xyzw
    pub scale: [f32; 3],
    pub asset_path: Option<String>,
    pub scripts: Vec<String>,
    pub tags: Vec<String>,
    pub custom_data: HashMap<String, serde_json::Value>,
    pub collider: Option<SavedCollider>,
    pub audio: Option<SavedAudio>,
    pub light: Option<SavedLight>,
    /// Spectral Gaussian splats baked onto this entity (in entity-local space).
    /// Each splat carries the full 16-band spectral signature (380–755 nm).
    /// `#[serde(default)]` keeps older saves (without splats) loadable.
    #[serde(default)]
    pub splats: Vec<SavedSplat>,
    /// AAA Spec 06 — the LOSSLESS Gaussian-splat geometry this entity owns in the
    /// editor overlay, stored as raw quantized records ([`SavedSplatGeom`]) so the
    /// 16-band `u16` spectral and `i16` rotation round-trip BIT-IDENTICALLY (the
    /// legacy `splats` field above stores f32 reflectance for the game layer; this
    /// stores the exact in-memory splat for the editor save/load). Mapped through
    /// [`crate::splat_codec::to_saved_geom`] / [`from_saved_geom`]. `#[serde(default)]`
    /// keeps older saves (without it) loadable.
    #[serde(default)]
    pub geom_splats: Vec<SavedSplatGeom>,
    /// If this entity was spawned from a prefab, the reference back to it.
    /// `None` for hand-placed entities. `#[serde(default)]` keeps older saves loadable.
    #[serde(default)]
    pub prefab_ref: Option<SavedPrefabRef>,
}

/// Number of spectral bands per saved splat: 380–755 nm at 25 nm steps.
/// Matches `vox_core::types::GaussianSplat::BANDS`.
pub const SAVED_SPLAT_BANDS: usize = 16;

/// A single spectral Gaussian splat persisted inside a [`SavedEntity`].
///
/// Stored in plain `f32` (not f16) so the on-disk JSON round-trips bit-exactly
/// — load(save(x)) == x for every field, which is what the world round-trip asserts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedSplat {
    /// Entity-local centroid.
    pub position: [f32; 3],
    /// 16-band spectral reflectance/emission, 380–755 nm (USGS grid).
    pub spectral: [f32; SAVED_SPLAT_BANDS],
    /// Alpha in [0, 1].
    pub opacity: f32,
}

/// A single Gaussian splat persisted LOSSLESSLY for the editor save/load (AAA
/// Spec 06).
///
/// Unlike [`SavedSplat`] (which stores f32 reflectance for the game layer), this
/// mirrors the runtime [`vox_core::types::GaussianSplat`] layout: the 16-band
/// `spectral` is the RAW `u16` (f16 bits) and `rotation` is the RAW `i16`
/// quantization, so a save → load round-trip reproduces every splat bit-for-bit.
/// Built and consumed only through [`crate::splat_codec::to_saved_geom`] /
/// [`from_saved_geom`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedSplatGeom {
    /// World-space centroid.
    pub position: [f32; 3],
    /// `0` = 2DGS surface, `1` = 3DGS volume (matches `GaussianSplat::kind`).
    pub kind: u32,
    /// 2DGS disk u-axis (zero for volumes).
    pub tangent_u: [f32; 3],
    /// 2DGS u-radius / 3DGS x half-axis.
    pub scale_u: f32,
    /// 2DGS disk v-axis (zero for volumes).
    pub tangent_v: [f32; 3],
    /// 2DGS v-radius / 3DGS y half-axis.
    pub scale_v: f32,
    /// RAW quantized quaternion XYZW (each `/ 32767`); identity for surfaces.
    pub rotation: [i16; 4],
    /// 2DGS unused (0) / 3DGS z half-axis.
    pub scale_w: f32,
    /// Alpha (0..=255).
    pub opacity: u8,
    /// RAW 16-band f16-as-`u16` spectral signature, 380–755 nm — copied verbatim
    /// so it round-trips bit-identically.
    pub spectral: [u16; SAVED_SPLAT_BANDS],
}

/// A reference recording that an entity (or group of entities) originated from
/// a prefab. Persisted so a load can re-link instances to their source prefab.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedPrefabRef {
    /// Name of the prefab this instance was spawned from.
    pub prefab_name: String,
    /// World-space position the prefab was instantiated at.
    pub instance_position: [f32; 3],
    /// Stable per-instance id (distinguishes multiple instances of one prefab).
    pub instance_id: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedCollider {
    pub shape_type: String,  // "box", "sphere", "capsule"
    pub dimensions: Vec<f32>, // half_extents for box, [radius] for sphere, [radius, height] for capsule
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedAudio {
    pub clip_path: String,
    pub volume: f32,
    pub looping: bool,
    pub spatial: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedLight {
    pub light_type: String,  // "point", "directional"
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedResources {
    pub time_of_day: f32,
    pub camera_position: [f32; 3],
    pub camera_rotation: [f32; 4],
    pub game_state: String,
}

impl SavedEntity {
    /// Construct a minimal entity at `position` with identity rotation, unit
    /// scale, and every optional component empty. Set fields after construction.
    pub fn new(name: &str, position: [f32; 3]) -> Self {
        Self {
            name: name.to_string(),
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            asset_path: None,
            scripts: Vec::new(),
            tags: Vec::new(),
            custom_data: HashMap::new(),
            collider: None,
            audio: None,
            light: None,
            splats: Vec::new(),
            geom_splats: Vec::new(),
            prefab_ref: None,
        }
    }
}

impl WorldSave {
    pub fn new(scene_name: &str) -> Self {
        Self {
            version: 1,
            engine_version: "0.1.0".to_string(),
            timestamp: chrono_lite_timestamp(),
            scene_name: scene_name.to_string(),
            entities: Vec::new(),
            resources: SavedResources {
                time_of_day: 12.0,
                camera_position: [0.0, 10.0, 30.0],
                camera_rotation: [0.0, 0.0, 0.0, 1.0],
                game_state: "playing".to_string(),
            },
        }
    }

    pub fn add_entity(&mut self, entity: SavedEntity) {
        self.entities.push(entity);
    }

    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, &json).map_err(|e| e.to_string())?;
        println!("[save] Saved {} entities to {}", self.entities.len(), path.display());
        Ok(())
    }

    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let save: Self = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        println!("[save] Loaded {} entities from {}", save.entities.len(), path.display());
        Ok(save)
    }

    pub fn entity_count(&self) -> usize { self.entities.len() }

    /// Quick save to default location
    pub fn quick_save_path() -> std::path::PathBuf {
        std::path::PathBuf::from("saves/quicksave.json")
    }

    pub fn from_entities(
        entities: Vec<SavedEntity>,
        camera_position: [f32; 3],
        camera_rotation: [f32; 4],
        time_of_day: f32,
    ) -> Self {
        WorldSave {
            version: 1,
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono_lite_timestamp(),
            scene_name: "scene".into(),
            entities,
            resources: SavedResources {
                time_of_day,
                camera_position,
                camera_rotation,
                game_state: "playing".into(),
            },
        }
    }

    /// Auto save path with timestamp
    pub fn auto_save_path() -> std::path::PathBuf {
        let dir = dirs_next::data_dir().unwrap_or_else(|| std::path::PathBuf::from(".")).join("ochroma/saves");
        std::fs::create_dir_all(&dir).ok();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        dir.join(format!("autosave_{}.ochroma_save", timestamp))
    }
}

fn chrono_lite_timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}", secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn world_save_roundtrip() {
        let ws = WorldSave {
            version: 1,
            engine_version: "test".into(),
            timestamp: "0".into(),
            scene_name: "test".into(),
            entities: vec![SavedEntity {
                name: "cube".into(),
                position: [1.0, 2.0, 3.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
                asset_path: Some("assets/cube.vxm".into()),
                scripts: vec![],
                tags: vec![],
                custom_data: HashMap::new(),
                collider: None,
                audio: None,
                light: None,
                splats: vec![],
                geom_splats: vec![],
                prefab_ref: None,
            }],
            resources: SavedResources {
                time_of_day: 12.0,
                camera_position: [0.0, 5.0, -10.0],
                camera_rotation: [0.0, 0.0, 0.0, 1.0],
                game_state: "playing".into(),
            },
        };
        let f = tempfile::NamedTempFile::new().unwrap();
        ws.save_to_file(f.path()).unwrap();
        let loaded = WorldSave::load_from_file(f.path()).unwrap();
        assert_eq!(loaded.entities.len(), 1);
        assert_eq!(loaded.entities[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(loaded.entities[0].name, "cube");
        assert_eq!(loaded.resources.time_of_day, 12.0);
        assert_eq!(loaded.resources.camera_position, [0.0, 5.0, -10.0]);
    }

    #[test]
    fn world_save_from_entities_sets_version() {
        let ws = WorldSave::from_entities(vec![], [0.0; 3], [0.0, 0.0, 0.0, 1.0], 6.0);
        assert_eq!(ws.version, 1);
        assert_eq!(ws.resources.time_of_day, 6.0);
        assert_eq!(ws.scene_name, "scene");
    }
}
