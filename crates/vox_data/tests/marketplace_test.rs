use uuid::Uuid;
use vox_data::marketplace::*;

fn make_listing(
    name: &str,
    category: MarketplaceCategory,
    rating: f32,
    downloads: u64,
) -> AssetListing {
    AssetListing {
        id: Uuid::new_v4(),
        name: name.to_string(),
        description: format!("A {} asset", name),
        author: "Test".to_string(),
        version: "1.0".to_string(),
        category,
        tags: vec![name.to_lowercase()],
        price: AssetPrice::Free,
        downloads,
        rating,
        rating_count: 10,
        splat_count: 1000,
        file_size_bytes: 50000,
        spectral_validated: true,
        created_at: "2026-03-27".to_string(),
        updated_at: "2026-03-27".to_string(),
    }
}

#[test]
fn search_by_name() {
    let mut cache = MarketplaceCache::new();
    cache.add_listing(make_listing(
        "Victorian House",
        MarketplaceCategory::Building,
        4.5,
        100,
    ));
    cache.add_listing(make_listing(
        "Modern Lamp",
        MarketplaceCategory::Prop,
        4.0,
        50,
    ));
    let results = cache.search("victorian");
    assert_eq!(results.len(), 1);
    assert!(results[0].name.contains("Victorian"));
}

#[test]
fn filter_by_category() {
    let mut cache = MarketplaceCache::new();
    cache.add_listing(make_listing(
        "House",
        MarketplaceCategory::Building,
        4.0,
        100,
    ));
    cache.add_listing(make_listing(
        "Tree",
        MarketplaceCategory::Vegetation,
        4.5,
        200,
    ));
    assert_eq!(cache.by_category(MarketplaceCategory::Building).len(), 1);
    assert_eq!(cache.by_category(MarketplaceCategory::Vegetation).len(), 1);
}

#[test]
fn top_rated() {
    let mut cache = MarketplaceCache::new();
    cache.add_listing(make_listing("A", MarketplaceCategory::Building, 3.0, 100));
    cache.add_listing(make_listing("B", MarketplaceCategory::Building, 5.0, 50));
    cache.add_listing(make_listing("C", MarketplaceCategory::Building, 4.0, 75));
    let top = cache.top_rated(2);
    assert_eq!(top[0].name, "B");
    assert_eq!(top[1].name, "C");
}

#[test]
fn price_display() {
    assert_eq!(AssetPrice::Free.usd_display(), "Free");
    assert_eq!(
        AssetPrice::Paid { usd_cents: 499 }.usd_display(),
        "$4.99"
    );
}
