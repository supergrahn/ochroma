use vox_sim::ecosystem::*;

#[test]
fn tree_grows_over_time() {
    let mut eco = EcosystemManager::new();
    eco.plant_tree([0.0, 0.0], TreeSpecies::Oak);
    let initial_height = eco.trees[0].height;
    eco.tick(5.0, |_| 0.0, 1.0);
    assert!(eco.trees[0].height > initial_height);
}

#[test]
fn pollution_kills_trees() {
    let mut eco = EcosystemManager::new();
    eco.plant_tree([0.0, 0.0], TreeSpecies::Birch);
    // Heavy pollution
    for _ in 0..100 {
        eco.tick(1.0, |_| 0.9, 1.0);
    }
    assert_eq!(eco.count(), 0, "Tree should die from heavy pollution");
}

#[test]
fn tree_capped_at_max_height() {
    let mut eco = EcosystemManager::new();
    eco.plant_tree([0.0, 0.0], TreeSpecies::Pine);
    for _ in 0..200 {
        eco.tick(1.0, |_| 0.0, 1.0);
    }
    assert!(eco.trees[0].height <= TreeSpecies::Pine.max_height());
}

#[test]
fn mature_trees_spread() {
    let mut eco = EcosystemManager::new();
    eco.plant_tree([0.0, 0.0], TreeSpecies::Oak);
    eco.trees[0].age_years = 20.0;
    eco.trees[0].health = 1.0;
    // Try spreading many times (probabilistic)
    for i in 0..100 {
        eco.spread(i);
    }
    assert!(eco.count() > 1, "Mature tree should produce saplings");
}
