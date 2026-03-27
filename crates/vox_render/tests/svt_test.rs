use vox_render::svt::*;

#[test]
fn touch_and_load_tile() {
    let mut cache = TileCache::new(10);
    let id = TileId { x: 0, y: 0, mip_level: 0 };
    cache.touch(id);
    assert_eq!(cache.tile_count(), 1);
    assert!(!cache.is_loaded(id));
    cache.mark_loaded(id);
    assert!(cache.is_loaded(id));
}

#[test]
fn evicts_lru_when_over_budget() {
    let mut cache = TileCache::new(2);
    let t0 = TileId { x: 0, y: 0, mip_level: 0 };
    let t1 = TileId { x: 1, y: 0, mip_level: 0 };
    let t2 = TileId { x: 2, y: 0, mip_level: 0 };
    cache.touch(t0);
    cache.touch(t1);
    cache.touch(t2);
    let evicted = cache.evict_to_budget();
    assert_eq!(evicted.len(), 1);
    assert_eq!(evicted[0], t0); // least recently used
    assert_eq!(cache.tile_count(), 2);
}

#[test]
fn touch_refreshes_access_order() {
    let mut cache = TileCache::new(2);
    let t0 = TileId { x: 0, y: 0, mip_level: 0 };
    let t1 = TileId { x: 1, y: 0, mip_level: 0 };
    let t2 = TileId { x: 2, y: 0, mip_level: 0 };
    cache.touch(t0);
    cache.touch(t1);
    cache.touch(t0); // refresh t0
    cache.touch(t2);
    let evicted = cache.evict_to_budget();
    assert_eq!(evicted[0], t1); // t1 is now LRU because t0 was refreshed
}
