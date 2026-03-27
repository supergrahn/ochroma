use serde::{Deserialize, Serialize};
use vox_core::types::GaussianSplat;

/// Predefined benchmark scenes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BenchmarkScene {
    SplatGrid1M,
    SplatGrid5M,
    SplatGrid10M,
    CityBlock,
    StressTest,
}

impl std::fmt::Display for BenchmarkScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SplatGrid1M => write!(f, "SplatGrid1M"),
            Self::SplatGrid5M => write!(f, "SplatGrid5M"),
            Self::SplatGrid10M => write!(f, "SplatGrid10M"),
            Self::CityBlock => write!(f, "CityBlock"),
            Self::StressTest => write!(f, "StressTest"),
        }
    }
}

/// Result of running a single benchmark scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub scene: BenchmarkScene,
    pub resolution: (u32, u32),
    pub splat_count: usize,
    pub avg_frame_ms: f64,
    pub min_frame_ms: f64,
    pub max_frame_ms: f64,
    pub p99_frame_ms: f64,
    pub fps: f64,
    pub sort_time_ms: f64,
    pub render_time_ms: f64,
}

impl BenchmarkResult {
    /// Create a new result from frame time samples.
    pub fn from_samples(
        scene: BenchmarkScene,
        resolution: (u32, u32),
        splat_count: usize,
        frame_times_ms: &[f64],
        sort_time_ms: f64,
        render_time_ms: f64,
    ) -> Self {
        let n = frame_times_ms.len().max(1);
        let avg = frame_times_ms.iter().sum::<f64>() / n as f64;
        let min = frame_times_ms
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = frame_times_ms
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // p99: sort and take 99th percentile.
        let mut sorted = frame_times_ms.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99_index = ((n as f64 * 0.99).ceil() as usize).min(n) - 1;
        let p99 = sorted.get(p99_index).copied().unwrap_or(avg);

        let fps = if avg > 0.0 { 1000.0 / avg } else { 0.0 };

        Self {
            scene,
            resolution,
            splat_count,
            avg_frame_ms: avg,
            min_frame_ms: min,
            max_frame_ms: max,
            p99_frame_ms: p99,
            fps,
            sort_time_ms,
            render_time_ms,
        }
    }
}

/// A regression detected between baseline and current results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regression {
    pub scene: BenchmarkScene,
    pub metric: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub percent_change: f64,
}

/// Benchmark suite that runs scenes and collects results.
#[derive(Debug, Default)]
pub struct BenchmarkSuite {
    pub results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Record a benchmark result.
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Get result for a specific scene.
    pub fn result_for(&self, scene: BenchmarkScene) -> Option<&BenchmarkResult> {
        self.results.iter().find(|r| r.scene == scene)
    }
}

/// Generate a deterministic grid of splats for benchmarking.
pub fn generate_benchmark_splats(count: usize) -> Vec<GaussianSplat> {
    let side = (count as f64).cbrt().ceil() as usize;
    let mut splats = Vec::with_capacity(count);

    for i in 0..count {
        let ix = i % side;
        let iy = (i / side) % side;
        let iz = i / (side * side);

        let spacing = 1.0f32;
        let splat = GaussianSplat {
            position: [
                ix as f32 * spacing,
                iy as f32 * spacing,
                iz as f32 * spacing,
            ],
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767], // identity quaternion w=1
            opacity: 255,
            _pad: [0; 3],
            spectral: [0; 8],
        };
        splats.push(splat);
    }

    splats
}

/// Compare baseline results with current results, returning any regressions (>5% slower).
pub fn compare_results(baseline: &[BenchmarkResult], current: &[BenchmarkResult]) -> Vec<Regression> {
    let mut regressions = Vec::new();

    for cur in current {
        if let Some(base) = baseline.iter().find(|b| b.scene == cur.scene) {
            let metrics: Vec<(&str, f64, f64)> = vec![
                ("avg_frame_ms", base.avg_frame_ms, cur.avg_frame_ms),
                ("max_frame_ms", base.max_frame_ms, cur.max_frame_ms),
                ("p99_frame_ms", base.p99_frame_ms, cur.p99_frame_ms),
                ("sort_time_ms", base.sort_time_ms, cur.sort_time_ms),
                ("render_time_ms", base.render_time_ms, cur.render_time_ms),
            ];

            for (name, base_val, cur_val) in metrics {
                if base_val > 0.0 {
                    let pct = ((cur_val - base_val) / base_val) * 100.0;
                    // Positive pct means slower (worse) for time metrics.
                    if pct > 5.0 {
                        regressions.push(Regression {
                            scene: cur.scene,
                            metric: name.to_string(),
                            baseline_value: base_val,
                            current_value: cur_val,
                            percent_change: pct,
                        });
                    }
                }
            }
        }
    }

    regressions
}

/// Export benchmark results as JSON.
pub fn export_results_json(results: &[BenchmarkResult]) -> String {
    serde_json::to_string_pretty(results).unwrap_or_else(|_| "[]".to_string())
}
