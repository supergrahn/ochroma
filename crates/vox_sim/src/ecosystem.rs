use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeInstance {
    pub id: u32,
    pub position: [f32; 2],
    pub species: TreeSpecies,
    pub age_years: f32,
    pub height: f32,
    pub canopy_radius: f32,
    pub health: f32, // 0.0-1.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TreeSpecies {
    Oak,
    Pine,
    Birch,
    Willow,
}

impl TreeSpecies {
    pub fn max_height(&self) -> f32 {
        match self {
            Self::Oak => 15.0,
            Self::Pine => 20.0,
            Self::Birch => 12.0,
            Self::Willow => 10.0,
        }
    }

    pub fn max_canopy(&self) -> f32 {
        match self {
            Self::Oak => 6.0,
            Self::Pine => 3.0,
            Self::Birch => 4.0,
            Self::Willow => 5.0,
        }
    }

    pub fn growth_rate(&self) -> f32 {
        match self {
            Self::Oak => 0.3,
            Self::Pine => 0.5,
            Self::Birch => 0.4,
            Self::Willow => 0.35,
        }
    }
}

pub struct EcosystemManager {
    pub trees: Vec<TreeInstance>,
    next_id: u32,
}

impl EcosystemManager {
    pub fn new() -> Self {
        Self {
            trees: Vec::new(),
            next_id: 0,
        }
    }

    pub fn plant_tree(&mut self, position: [f32; 2], species: TreeSpecies) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.trees.push(TreeInstance {
            id,
            position,
            species,
            age_years: 0.0,
            height: 0.5,
            canopy_radius: 0.2,
            health: 1.0,
        });
        id
    }

    /// Tick: grow trees, check health.
    pub fn tick(
        &mut self,
        dt_years: f32,
        pollution_at: impl Fn([f32; 2]) -> f32,
        crop_growth_rate: f32,
    ) {
        for tree in &mut self.trees {
            let pollution = pollution_at(tree.position);
            tree.health = (tree.health - pollution * dt_years * 0.1).max(0.0);

            let growth = tree.species.growth_rate() * dt_years * tree.health * crop_growth_rate;
            tree.age_years += dt_years;
            tree.height = (tree.height + growth).min(tree.species.max_height());
            tree.canopy_radius =
                (tree.canopy_radius + growth * 0.5).min(tree.species.max_canopy());
        }

        // Dead trees
        self.trees.retain(|t| t.health > 0.01);
    }

    pub fn count(&self) -> usize {
        self.trees.len()
    }

    /// Natural spread: healthy mature trees produce saplings nearby.
    pub fn spread(&mut self, seed: u64) {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut new_trees = Vec::new();

        for tree in &self.trees {
            if tree.age_years > 10.0 && tree.health > 0.5 && rng.random::<f32>() < 0.05 {
                let offset_x = (rng.random::<f32>() - 0.5) * 20.0;
                let offset_z = (rng.random::<f32>() - 0.5) * 20.0;
                new_trees.push((
                    [tree.position[0] + offset_x, tree.position[1] + offset_z],
                    tree.species,
                ));
            }
        }

        for (pos, species) in new_trees {
            self.plant_tree(pos, species);
        }
    }
}

impl Default for EcosystemManager {
    fn default() -> Self {
        Self::new()
    }
}
