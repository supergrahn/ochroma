use std::collections::HashSet;
use vox_sim::sharding::{ShardManager, TileCoord};

fn tile_set(coords: &[(i32, i32)]) -> HashSet<TileCoord> {
    coords.iter().map(|&(x, y)| TileCoord { x, y }).collect()
}

#[test]
fn create_shard() {
    let mut mgr = ShardManager::new();
    let tiles = tile_set(&[(0, 0), (1, 0), (0, 1)]);
    let id = mgr.create_shard(tiles);
    assert_eq!(mgr.shard_count(), 1);
    assert!(mgr.shard(id).is_some());
}

#[test]
fn assign_entities() {
    let mut mgr = ShardManager::new();
    let id = mgr.create_shard(tile_set(&[(0, 0)]));
    assert!(mgr.assign_entity(1, id));
    assert!(mgr.assign_entity(2, id));
    assert_eq!(mgr.shard(id).unwrap().entity_count(), 2);
    assert!(mgr.shard(id).unwrap().contains_entity(1));
}

#[test]
fn migrate_between_shards() {
    let mut mgr = ShardManager::new();
    let a = mgr.create_shard(tile_set(&[(0, 0)]));
    let b = mgr.create_shard(tile_set(&[(1, 0)]));
    mgr.assign_entity(42, a);

    let record = mgr.migrate_entity(42, a, b).expect("migration should succeed");
    assert_eq!(record.entity_id, 42);
    assert_eq!(record.from_shard, a);
    assert_eq!(record.to_shard, b);
    assert!(!mgr.shard(a).unwrap().contains_entity(42));
    assert!(mgr.shard(b).unwrap().contains_entity(42));
}

#[test]
fn migrate_nonexistent_entity_returns_none() {
    let mut mgr = ShardManager::new();
    let a = mgr.create_shard(tile_set(&[(0, 0)]));
    let b = mgr.create_shard(tile_set(&[(1, 0)]));
    assert!(mgr.migrate_entity(999, a, b).is_none());
}

#[test]
fn rebalance_splits_large_shard() {
    let mut mgr = ShardManager::new();
    let tiles: HashSet<TileCoord> = (0..100).map(|x| TileCoord { x, y: 0 }).collect();
    let id = mgr.create_shard(tiles);

    for i in 0..10_500u64 {
        mgr.assign_entity(i, id);
    }
    assert_eq!(mgr.shard_count(), 1);

    let records = mgr.rebalance();
    assert!(mgr.shard_count() >= 2, "large shard should be split");
    assert!(!records.is_empty());
    assert_eq!(mgr.total_entities(), 10_500);
}

#[test]
fn rebalance_merges_small_shards() {
    let mut mgr = ShardManager::new();
    let a = mgr.create_shard(tile_set(&[(0, 0)]));
    let b = mgr.create_shard(tile_set(&[(1, 0)]));

    // Add fewer than 100 entities to each
    for i in 0..10u64 {
        mgr.assign_entity(i, a);
    }
    for i in 10..20u64 {
        mgr.assign_entity(i, b);
    }
    assert_eq!(mgr.shard_count(), 2);

    let records = mgr.rebalance();
    assert_eq!(mgr.shard_count(), 1, "small shards should be merged");
    assert!(!records.is_empty());
    assert_eq!(mgr.total_entities(), 20);
}

#[test]
fn migration_preserves_entity() {
    let mut mgr = ShardManager::new();
    let a = mgr.create_shard(tile_set(&[(0, 0)]));
    let b = mgr.create_shard(tile_set(&[(1, 0)]));
    mgr.assign_entity(7, a);

    let before = mgr.total_entities();
    mgr.migrate_entity(7, a, b);
    let after = mgr.total_entities();

    assert_eq!(before, after, "migration must not lose or duplicate entities");
    assert_eq!(after, 1);
}

#[test]
fn shard_lookup_by_tile() {
    let mut mgr = ShardManager::new();
    let id = mgr.create_shard(tile_set(&[(5, 5), (6, 5)]));

    assert_eq!(mgr.shard_for_tile(TileCoord { x: 5, y: 5 }), Some(id));
    assert_eq!(mgr.shard_for_tile(TileCoord { x: 6, y: 5 }), Some(id));
    assert!(mgr.shard_for_tile(TileCoord { x: 99, y: 99 }).is_none());
}

#[test]
fn total_entity_count() {
    let mut mgr = ShardManager::new();
    let a = mgr.create_shard(tile_set(&[(0, 0)]));
    let b = mgr.create_shard(tile_set(&[(1, 0)]));

    for i in 0..50u64 {
        mgr.assign_entity(i, a);
    }
    for i in 50..80u64 {
        mgr.assign_entity(i, b);
    }

    assert_eq!(mgr.total_entities(), 80);
}

#[test]
fn tick_advances_all_shards() {
    let mut mgr = ShardManager::new();
    let a = mgr.create_shard(tile_set(&[(0, 0)]));
    let b = mgr.create_shard(tile_set(&[(1, 0)]));

    mgr.tick();
    mgr.tick();

    assert_eq!(mgr.shard(a).unwrap().current_tick(), 2);
    assert_eq!(mgr.shard(b).unwrap().current_tick(), 2);
}
