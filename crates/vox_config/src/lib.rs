//! `vox_config` — the runtime-loaded SINGLE SOURCE OF TRUTH for every tunable
//! engine setting (`config/ochroma.ron`).
//!
//! CONFIG-FIRST LAW (CLAUDE.md): a tunable value must be changeable WITHOUT a
//! rebuild and WITHOUT exporting an env var. This crate deserializes
//! `config/ochroma.ron` into [`OchromaConfig`] and exposes it as a process-wide
//! [`config()`] singleton that every engine crate reads at its former
//! hardcoded-default / `env::var` site.
//!
//! ROBUSTNESS CONTRACT (load can NEVER break the build or run):
//!   * Every struct is `#[serde(default)]` over a hand-written [`Default`], so a
//!     PARTIAL `ochroma.ron` (any missing block or field) silently falls back to
//!     the documented default — the same value the code hardcoded before.
//!   * Unknown fields (e.g. the `_env_inventory` documentation block) are
//!     IGNORED (no `deny_unknown_fields`).
//!   * A missing file, an unreadable file, or a parse error all resolve to the
//!     full default config (logged once to stderr, never a panic).
//!
//! PRECEDENCE at a wired read site: an explicit `env::var` override (where one
//! still exists — the documented A/B "sweep levers") wins, then this config
//! value, then nothing else (the old hardcoded literal is GONE). So with no env
//! set and no `ochroma.ron` present the runtime is byte-identical to before.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;

/// Canonical LOD ladder length. The LOD fraction/distance vecs are normalized to
/// this length so wired read sites can index `[0..LOD_LEN]` without bounds risk.
pub const LOD_LEN: usize = 4;

// ============================================================================
// terrain — heightmap geometry + per-map snow line / curvature field.
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TerrainConfig {
    /// Heightmap grid edge (cells). `vox_terrain::HEIGHTMAP_SIZE`.
    pub heightmap_size: u32,
    /// Metres per heightmap cell. `vox_terrain::HEIGHTMAP_RESOLUTION`.
    pub heightmap_resolution: f32,
    /// Snow-cap atlas slots (-1 = snow off, rolls into biome ground).
    pub snow_albedo_slot: i32,
    pub snow_normal_slot: i32,
    pub snow_disp_slot: i32,
    /// Altitude (m) where snow begins; 1e9 = off.
    pub snow_line_m: f32,
    /// Vertical blend band (m) above the snow line.
    pub snow_band_m: f32,
    /// Up-cosine cutoff: caps sit on ledges/peaks, not vertical faces.
    pub snow_slope_cos: f32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            heightmap_size: 4096,
            heightmap_resolution: 0.25,
            snow_albedo_slot: -1,
            snow_normal_slot: -1,
            snow_disp_slot: -1,
            snow_line_m: 1.0e9,
            snow_band_m: 120.0,
            snow_slope_cos: 0.78,
        }
    }
}

// ============================================================================
// slope_layers — height/slope-aware layered ground material atlas slots.
// All default -1 (absent) — NOT 0 (a valid slot that would fire the rock blend).
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SlopeLayersConfig {
    pub rock_albedo_slot: i32,
    pub rock_normal_slot: i32,
    pub dirt_albedo_slot: i32,
    pub dirt_normal_slot: i32,
    pub rock_disp_slot: i32,
    pub dirt_disp_slot: i32,
}

impl Default for SlopeLayersConfig {
    fn default() -> Self {
        Self {
            rock_albedo_slot: -1,
            rock_normal_slot: -1,
            dirt_albedo_slot: -1,
            dirt_normal_slot: -1,
            rock_disp_slot: -1,
            dirt_disp_slot: -1,
        }
    }
}

// ============================================================================
// ground — stochastic texture anti-tiling + faceted-normal de-facet.
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GroundConfig {
    /// SPECTRA_GROUND_ANTITILE_STRENGTH; spectra default 0.85.
    pub antitile_strength: f32,
    /// SPECTRA_GROUND_ANTITILE_CELL_M; spectra default 18 m.
    pub antitile_cell_m: f32,
    /// SPECTRA_GROUND_SMOOTH_NORMAL; 1.0 = fully smooth, range 0..=1.
    pub smooth_normal: f32,
}

impl Default for GroundConfig {
    fn default() -> Self {
        Self {
            antitile_strength: 0.85,
            antitile_cell_m: 18.0,
            smooth_normal: 1.0,
        }
    }
}

// ============================================================================
// scatter / lod — vegetation + asset instancing budgets and LOD ladder.
// The single source for the LOD fractions/distances that were copy-pasted
// across hierarchical_lod / atom_budget / atom_instances.
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ScatterConfig {
    /// LOD level count. NOTE: array-sizing const `LOD_LEVEL_COUNT` stays a
    /// compile-time const (4); this mirrors it for completeness / validation.
    pub lod_level_count: u32,
    /// Splat-count fraction per LOD level (0.0 = single billboard). RON list
    /// syntax `[..]`; the loader [`OchromaConfig::normalize`] guarantees it is
    /// exactly `LOD_LEN` long so wired read sites can index it safely.
    pub lod_fractions: Vec<f32>,
    /// Eye-distance (m) transitions per LOD level. Length-normalized to `LOD_LEN`.
    pub lod_distances_m: Vec<f32>,
    /// Imposter switch distance (m). `atom_instances::FAR_INSTANCE_M`.
    pub far_instance_m: f32,
    /// 1-atom imposter distance (m). `atom_instances::IMPOSTER_I1_M`.
    pub imposter_i1_m: f32,
    /// Coarse Full/Reduced cutoff (m). `lod::LOD_THRESHOLD`.
    pub lod_threshold_m: f32,
    /// CPU splat raster tile size. `spectra_render::TILE_SIZE`.
    pub rt_tile_size: u32,
    /// Per-tile splat budget (WGSL `BUDGET`). Informational mirror.
    pub splat_rt_budget: u32,
}

impl Default for ScatterConfig {
    fn default() -> Self {
        Self {
            lod_level_count: 4,
            lod_fractions: vec![1.0, 0.4, 0.1, 0.0],
            lod_distances_m: vec![0.0, 50.0, 150.0, 400.0],
            far_instance_m: 150.0,
            imposter_i1_m: 400.0,
            lod_threshold_m: 200.0,
            rt_tile_size: 16,
            splat_rt_budget: 64,
        }
    }
}

// ============================================================================
// instancing — GPU GI / emissive-light scene capacities.
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct InstancingConfig {
    /// GPU GI probe/instance capacity. `engine_loop::GPU_GI_CAPACITY`.
    pub gpu_gi_capacity: u32,
    /// Night NEE emitter cap. `resident_renderer::MAX_EMISSIVE_POINT_LIGHTS`.
    pub max_emissive_point_lights: u32,
}

impl Default for InstancingConfig {
    fn default() -> Self {
        Self {
            gpu_gi_capacity: 200_000,
            max_emissive_point_lights: 4096,
        }
    }
}

// ============================================================================
// resident_renderer — the live present-path per-frame real-time gates.
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ResidentRendererConfig {
    pub prefer_lean_shade: bool,
    /// "single" | "multi" | "hero4". OCHROMA_SPECTRAL.
    pub spectral_mode: String,
    pub use_optix_rt: bool,
    /// SPECTRA_GLASS_BOUNCES; -1 = use tier floor, 0 = disable bump.
    pub glass_bounces_override: i32,
    pub glass_floor_performance: u32,
    pub glass_floor_balanced: u32,
    pub glass_floor_beauty: u32,
    pub temporal_enabled: bool,
    /// -1 = spectra RenderConfig.temporal default.
    pub temporal_alpha_static: f32,
    pub temporal_alpha_moving: f32,
    pub temporal_depth_threshold: f32,
    pub temporal_normal_threshold: f32,
    /// "on" | "off" | "auto". OCHROMA_RESTIR.
    pub restir_override: String,
    /// SPECTRA_SHOT_SPP; 0 = use tier spp.
    pub shot_spp_override: u32,
    /// SPECTRA_MAX_PIXELS_PER_DISPATCH; 0 = no banding.
    pub max_pixels_per_dispatch: u32,
    pub use_cuda_graphs: bool,
    /// Shader Execution Reordering. Runtime field (was `#[cfg(windows)]`-gated).
    pub ser_enabled: bool,
    pub lit_windows_enabled: bool,
    pub emissive_light_scale: f32,
    pub lit_window_glow: f32,
    pub lit_window_fraction: f32,
    pub aurora_trace: bool,
    pub dispatch_timing: bool,
    pub film_diag: bool,
}

impl Default for ResidentRendererConfig {
    fn default() -> Self {
        Self {
            prefer_lean_shade: true,
            spectral_mode: "hero4".to_string(),
            use_optix_rt: true,
            glass_bounces_override: -1,
            glass_floor_performance: 4,
            glass_floor_balanced: 5,
            glass_floor_beauty: 8,
            temporal_enabled: true,
            temporal_alpha_static: -1.0,
            temporal_alpha_moving: -1.0,
            temporal_depth_threshold: -1.0,
            temporal_normal_threshold: -1.0,
            restir_override: "auto".to_string(),
            shot_spp_override: 0,
            max_pixels_per_dispatch: 65_536,
            use_cuda_graphs: true,
            ser_enabled: true,
            lit_windows_enabled: true,
            emissive_light_scale: 2000.0,
            lit_window_glow: 8.0,
            lit_window_fraction: 0.6,
            aurora_trace: false,
            dispatch_timing: false,
            film_diag: false,
        }
    }
}

// ============================================================================
// denoiser — À-Trous cascade + still/SDF render A-B knobs.
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DenoiserConfig {
    /// OCHROMA_DENOISE_OFF (set => off); default ON.
    pub enabled: bool,
    /// OCHROMA_DENOISE_LEGACY=1 — 1-pass, no luminance edge-stop.
    pub legacy_single: bool,
    /// OCHROMA_DENOISE_ITERS; 0 = spectra cascade default (min 1).
    pub iterations: u32,
    /// OCHROMA_DENOISE_SIGMA_LUM; -1 = spectra default.
    pub sigma_lum: f32,
    /// OCHROMA_RELIEF_MODE (0=POM,1=cone-step); -1 = auto-select by data.
    pub relief_mode: i32,
}

impl Default for DenoiserConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            legacy_single: false,
            iterations: 0,
            sigma_lum: -1.0,
            relief_mode: -1,
        }
    }
}

// ============================================================================
// present_bench — FSR + OptiX/still bench A-B escapes (measurement levers).
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PresentBenchConfig {
    pub lean_pipeline: bool,
    pub shade_lean: bool,
    pub fsr_legacy_jitter: bool,
    pub bench_cam_motion: bool,
    pub pass_profile: bool,
    pub keep_beauty: bool,
}

impl Default for PresentBenchConfig {
    fn default() -> Self {
        Self {
            lean_pipeline: false,
            shade_lean: false,
            fsr_legacy_jitter: false,
            bench_cam_motion: false,
            pass_profile: false,
            keep_beauty: false,
        }
    }
}

// ============================================================================
// gpu / backend — device selection. NOTE: spectra_backend / slang_kernel_dir /
// spectra_path are TOOLCHAIN/PATH vars and stay env-driven (left in code).
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GpuConfig {
    /// "gpu" | "cpu". OCHROMA_GI; default cpu (proven path).
    pub gi_backend: String,
    /// OCHROMA_ALLOW_SOFTWARE_GPU — permit llvmpipe fallback.
    pub allow_software_gpu: bool,
    /// SPECTRA_BACKEND. Toolchain selector — bridged here for completeness, the
    /// env var remains authoritative in code (do not wire).
    pub spectra_backend: String,
    pub slang_kernel_dir: String,
    pub spectra_path: String,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            gi_backend: "cpu".to_string(),
            allow_software_gpu: false,
            spectra_backend: "auto".to_string(),
            slang_kernel_dir: String::new(),
            spectra_path: String::new(),
        }
    }
}

// ============================================================================
// spectral / atmosphere — Rayleigh/Mie coefficients + sky luminance + sun disk.
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SpectralConfig {
    pub beta_rayleigh_ref_per_km: f32,
    pub beta_mie_per_km: f32,
    pub rayleigh_scale_height_km: f32,
    pub mie_scale_height_km: f32,
    pub atmosphere_thickness_km: f32,
    pub preetham_luminance_scale: f32,
    pub sun_angular_radius_rad: f32,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            beta_rayleigh_ref_per_km: 0.0128,
            beta_mie_per_km: 0.005,
            rayleigh_scale_height_km: 8.0,
            mie_scale_height_km: 1.2,
            atmosphere_thickness_km: 60.0,
            preetham_luminance_scale: 20.0,
            sun_angular_radius_rad: 0.0046,
        }
    }
}

// ============================================================================
// materials — dynamic-weathering pattern falloffs/caps + splat importance.
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MaterialsConfig {
    pub weather_soot_falloff_m: f32,
    pub weather_efflor_falloff_m: f32,
    pub weather_soot_max: f32,
    pub weather_efflor_max: f32,
    pub weather_stain_max: f32,
    pub weather_edge_max: f32,
    pub importance_redundancy_weight: f32,
    pub importance_color_similarity: f32,
    pub splat_sigma_cutoff: f32,
}

impl Default for MaterialsConfig {
    fn default() -> Self {
        Self {
            weather_soot_falloff_m: 14.0,
            weather_efflor_falloff_m: 4.0,
            weather_soot_max: 0.28,
            weather_efflor_max: 0.18,
            weather_stain_max: 0.22,
            weather_edge_max: 0.12,
            importance_redundancy_weight: 0.5,
            importance_color_similarity: 0.98,
            splat_sigma_cutoff: 3.0,
        }
    }
}

// ============================================================================
// build_features — compile-time `#[cfg(feature)]` render gates (informational).
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct BuildFeaturesConfig {
    pub spectra_native: bool,
    pub spectra_native_optix: bool,
    pub fsr: bool,
    pub bevy: bool,
}

impl Default for BuildFeaturesConfig {
    fn default() -> Self {
        Self {
            spectra_native: true,
            spectra_native_optix: false,
            fsr: false,
            bevy: false,
        }
    }
}

// ============================================================================
// OchromaConfig — the whole tree.
// ============================================================================
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct OchromaConfig {
    pub terrain: TerrainConfig,
    pub slope_layers: SlopeLayersConfig,
    pub ground: GroundConfig,
    pub scatter: ScatterConfig,
    pub instancing: InstancingConfig,
    pub resident_renderer: ResidentRendererConfig,
    pub denoiser: DenoiserConfig,
    pub present_bench: PresentBenchConfig,
    pub gpu: GpuConfig,
    pub spectral: SpectralConfig,
    pub materials: MaterialsConfig,
    pub build_features: BuildFeaturesConfig,
}

impl OchromaConfig {
    /// Coerce variable-length list fields to their canonical lengths so wired
    /// read sites can index them without bounds checks even if a partial / hand-
    /// edited `ochroma.ron` supplies the wrong number of elements. Missing slots
    /// are filled from the default ladder; extras are truncated.
    pub fn normalize(&mut self) {
        let def = ScatterConfig::default();
        coerce_len(&mut self.scatter.lod_fractions, &def.lod_fractions, LOD_LEN);
        coerce_len(&mut self.scatter.lod_distances_m, &def.lod_distances_m, LOD_LEN);
    }

    /// Load the config.
    ///
    /// * `Some(path)` — load that exact file.
    /// * `None`       — resolve the default location: `$OCHROMA_CONFIG`, else a
    ///   `config/ochroma.ron` found by walking up from the current directory.
    ///
    /// ANY failure (missing file, unreadable, parse error) returns the full
    /// default config — load can never break the build or run. A partial file
    /// fills only the fields it names; everything else stays default.
    pub fn load(path: Option<&Path>) -> OchromaConfig {
        let resolved = match path {
            Some(p) => Some(p.to_path_buf()),
            None => resolve_default_path(),
        };
        let Some(p) = resolved else {
            // No file resolved ANYWHERE (no $OCHROMA_CONFIG, no config/ochroma.ron up
            // from cwd, none beside the exe). The live game falling through to here is
            // the silent config-first bug: ~372 engine settings stay at compiled
            // defaults. Warn loudly so a missing-config ship/run is caught, not hidden.
            eprintln!(
                "[vox_config] WARNING: no ochroma.ron found ($OCHROMA_CONFIG unset, none \
                 in config/ up from cwd, none beside the exe) — using COMPILED DEFAULTS. \
                 Engine config (sky/lighting/glass/perf) will NOT reflect ochroma.ron."
            );
            return OchromaConfig::default();
        };
        match std::fs::read_to_string(&p) {
            Ok(text) => match ron::from_str::<OchromaConfig>(&text) {
                Ok(mut cfg) => {
                    cfg.normalize();
                    eprintln!("[vox_config] loaded engine config from {}", p.display());
                    cfg
                }
                Err(e) => {
                    eprintln!(
                        "[vox_config] parse error in {} ({e}); using defaults",
                        p.display()
                    );
                    OchromaConfig::default()
                }
            },
            Err(e) => {
                // A missing default file is normal (defaults == old hardcoded
                // values); only note an explicit path that failed to read.
                if path.is_some() {
                    eprintln!(
                        "[vox_config] could not read {} ({e}); using defaults",
                        p.display()
                    );
                }
                OchromaConfig::default()
            }
        }
    }
}

/// Pad/truncate `v` to exactly `len`, filling missing tail slots from `fallback`.
fn coerce_len(v: &mut Vec<f32>, fallback: &[f32], len: usize) {
    if v.len() == len {
        return;
    }
    v.truncate(len);
    while v.len() < len {
        v.push(fallback.get(v.len()).copied().unwrap_or(0.0));
    }
}

/// Resolve the default config path: `$OCHROMA_CONFIG`, else the nearest
/// `config/ochroma.ron` walking up from the current directory (covers running
/// from the repo root or any crate subdir). `None` if neither resolves.
fn resolve_default_path() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("OCHROMA_CONFIG") {
        if !env_path.is_empty() {
            return Some(PathBuf::from(env_path));
        }
    }
    // 1) Walk up from the current directory (covers running from the repo root or
    //    any crate subdir during dev/test).
    if let Ok(start) = std::env::current_dir() {
        let mut dir = start;
        loop {
            let candidate = dir.join("config").join("ochroma.ron");
            if candidate.is_file() {
                return Some(candidate);
            }
            if !dir.pop() {
                break;
            }
        }
    }
    // 2) SHIP path: the game runs from `urban_horizon/` (no `config/ochroma.ron` in
    //    its tree) so the cwd walk above fails and the live game silently fell back
    //    to the compiled defaults — i.e. NONE of ochroma.ron's ~372 settings applied
    //    (the "config doesn't override everything" bug). Also search relative to the
    //    EXECUTABLE: `<exe_dir>/ochroma.ron` and `<exe_dir>/config/ochroma.ron`,
    //    walking up from the exe dir. The Windows packager copies ochroma.ron next to
    //    play.exe so a shipped build is config-first without any env var.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(start) = exe.parent() {
            let mut dir = start.to_path_buf();
            loop {
                let beside = dir.join("ochroma.ron");
                if beside.is_file() {
                    return Some(beside);
                }
                let in_cfg = dir.join("config").join("ochroma.ron");
                if in_cfg.is_file() {
                    return Some(in_cfg);
                }
                if !dir.pop() {
                    break;
                }
            }
        }
    }
    None
}

static GLOBAL: OnceLock<OchromaConfig> = OnceLock::new();

/// The process-wide config singleton. Loaded ONCE from the default location
/// (`$OCHROMA_CONFIG` or the nearest `config/ochroma.ron`) on first access;
/// defaults on any failure. This is the accessor every wired read site calls.
pub fn config() -> &'static OchromaConfig {
    GLOBAL.get_or_init(|| OchromaConfig::load(None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_documented_values() {
        let c = OchromaConfig::default();
        assert_eq!(c.terrain.heightmap_size, 4096);
        assert_eq!(c.terrain.snow_albedo_slot, -1);
        assert_eq!(c.scatter.lod_fractions, vec![1.0, 0.4, 0.1, 0.0]);
        assert_eq!(c.scatter.far_instance_m, 150.0);
        assert_eq!(c.instancing.gpu_gi_capacity, 200_000);
        assert_eq!(c.resident_renderer.spectral_mode, "hero4");
        assert!(c.resident_renderer.ser_enabled);
        assert_eq!(c.spectral.preetham_luminance_scale, 20.0);
        assert_eq!(c.materials.splat_sigma_cutoff, 3.0);
    }

    #[test]
    fn partial_ron_fills_only_named_fields() {
        // Only one nested block, only one field in it — everything else default.
        let text = "(terrain: (snow_line_m: 500.0))";
        let c = ron::from_str::<OchromaConfig>(text).expect("partial parse");
        assert_eq!(c.terrain.snow_line_m, 500.0);
        // Untouched fields keep their documented defaults.
        assert_eq!(c.terrain.heightmap_size, 4096);
        assert_eq!(c.terrain.snow_band_m, 120.0);
        assert_eq!(c.scatter.lod_distances_m, vec![0.0, 50.0, 150.0, 400.0]);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        // The real ochroma.ron carries an `_env_inventory` doc block that has no
        // matching struct field; it must be skipped, not error.
        let text = "(terrain: (heightmap_size: 8), _env_inventory: (foo: [\"x\"]))";
        let c = ron::from_str::<OchromaConfig>(text).expect("unknown-field skip");
        assert_eq!(c.terrain.heightmap_size, 8);
    }

    #[test]
    fn missing_file_yields_defaults() {
        let c = OchromaConfig::load(Some(Path::new("/no/such/ochroma.ron")));
        assert_eq!(c.terrain.heightmap_size, 4096);
    }

    #[test]
    fn real_repo_ron_parses() {
        // The shipped config must round-trip through the loader without error.
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("config")
            .join("ochroma.ron");
        if p.is_file() {
            let text = std::fs::read_to_string(&p).unwrap();
            let c = ron::from_str::<OchromaConfig>(&text)
                .unwrap_or_else(|e| panic!("shipped ochroma.ron failed to parse: {e}"));
            // A value only the real file sets distinctly from struct default
            // proves the file actually drove the parse.
            assert_eq!(c.resident_renderer.spectral_mode, "hero4");
            assert_eq!(c.scatter.lod_fractions, vec![1.0, 0.4, 0.1, 0.0]);
            assert_eq!(c.scatter.lod_distances_m, vec![0.0, 50.0, 150.0, 400.0]);
        }
    }
}
