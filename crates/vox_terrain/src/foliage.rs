use rand::prelude::*;
use rand::SeedableRng;

/// A foliage placement rule.
#[derive(Debug, Clone)]
pub struct FoliageRule {
    pub name: String,
    pub asset_path: String,   // .ply or .vxm file to instance
    pub density: f32,          // instances per 100 square metres
    pub min_height: f32,       // only place above this height
    pub max_height: f32,
    pub max_slope: f32,        // degrees — don't place on steep slopes
    pub min_scale: f32,
    pub max_scale: f32,
    pub random_rotation: bool, // randomise Y rotation
    pub cluster_radius: f32,   // 0 = uniform, >0 = cluster tendency
}

/// A placed foliage instance.
#[derive(Debug, Clone)]
pub struct FoliageInstance {
    pub rule_name: String,
    pub asset_path: String,
    pub position: [f32; 3],
    pub rotation_y: f32,
    pub scale: f32,
}

/// Scatter foliage on a terrain according to rules.
pub fn scatter_foliage(
    heightmap: &super::heightmap::Heightmap,
    rules: &[FoliageRule],
    seed: u64,
) -> Vec<FoliageInstance> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut instances = Vec::new();
    let (min_bound, max_bound) = heightmap.bounds();
    let area = heightmap.area();

    for rule in rules {
        let count = (area / 100.0 * rule.density) as usize;

        for _ in 0..count {
            let x = rng.random_range(min_bound[0]..max_bound[0]);
            let z = rng.random_range(min_bound[1]..max_bound[1]);
            let y = heightmap.sample(x, z);

            // Height check
            if y < rule.min_height || y > rule.max_height {
                continue;
            }

            // Slope check
            let slope = heightmap.slope_at(x, z);
            if slope > rule.max_slope {
                continue;
            }

            // Cluster offset
            let (cx, cz) = if rule.cluster_radius > 0.0 {
                let angle = rng.random::<f32>() * std::f32::consts::TAU;
                let dist = rng.random::<f32>() * rule.cluster_radius;
                (x + angle.cos() * dist, z + angle.sin() * dist)
            } else {
                (x, z)
            };

            let final_y = heightmap.sample(cx, cz);
            let scale = rng.random_range(rule.min_scale..=rule.max_scale);
            let rotation = if rule.random_rotation {
                rng.random::<f32>() * std::f32::consts::TAU
            } else {
                0.0
            };

            instances.push(FoliageInstance {
                rule_name: rule.name.clone(),
                asset_path: rule.asset_path.clone(),
                position: [cx, final_y, cz],
                rotation_y: rotation,
                scale,
            });
        }
    }

    instances
}

/// Default foliage rules for a temperate environment.
pub fn default_foliage_rules() -> Vec<FoliageRule> {
    vec![
        FoliageRule {
            name: "Oak Tree".into(),
            asset_path: "assets/trees/oak.ply".into(),
            density: 2.0,
            min_height: 0.5,
            max_height: 10.0,
            max_slope: 30.0,
            min_scale: 0.8,
            max_scale: 1.3,
            random_rotation: true,
            cluster_radius: 5.0,
        },
        FoliageRule {
            name: "Pine Tree".into(),
            asset_path: "assets/trees/pine.ply".into(),
            density: 3.0,
            min_height: 5.0,
            max_height: 15.0,
            max_slope: 40.0,
            min_scale: 0.7,
            max_scale: 1.5,
            random_rotation: true,
            cluster_radius: 3.0,
        },
        FoliageRule {
            name: "Grass Tuft".into(),
            asset_path: "assets/foliage/grass.ply".into(),
            density: 30.0,
            min_height: 0.0,
            max_height: 8.0,
            max_slope: 45.0,
            min_scale: 0.5,
            max_scale: 1.0,
            random_rotation: true,
            cluster_radius: 0.0,
        },
        FoliageRule {
            name: "Bush".into(),
            asset_path: "assets/foliage/bush.ply".into(),
            density: 5.0,
            min_height: 0.5,
            max_height: 6.0,
            max_slope: 25.0,
            min_scale: 0.6,
            max_scale: 1.2,
            random_rotation: true,
            cluster_radius: 2.0,
        },
    ]
}
