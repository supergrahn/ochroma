use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A marketplace listing for a shared asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetListing {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub author: String,
    pub version: String,
    pub category: MarketplaceCategory,
    pub tags: Vec<String>,
    pub price: AssetPrice,
    pub downloads: u64,
    pub rating: f32,
    pub rating_count: u32,
    pub splat_count: u32,
    pub file_size_bytes: u64,
    pub spectral_validated: bool, // passes spectral pipeline validation
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketplaceCategory {
    Building,
    Prop,
    Vehicle,
    Vegetation,
    Terrain,
    Character,
    ProcGSRule,
    MaterialPack,
    ModPlugin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetPrice {
    Free,
    Paid { usd_cents: u32 },
}

impl AssetPrice {
    pub fn is_free(&self) -> bool {
        matches!(self, Self::Free)
    }

    pub fn usd_display(&self) -> String {
        match self {
            Self::Free => "Free".to_string(),
            Self::Paid { usd_cents } => format!("${:.2}", *usd_cents as f64 / 100.0),
        }
    }
}

/// Local marketplace cache.
pub struct MarketplaceCache {
    pub listings: Vec<AssetListing>,
}

impl MarketplaceCache {
    pub fn new() -> Self {
        Self {
            listings: Vec::new(),
        }
    }

    pub fn add_listing(&mut self, listing: AssetListing) {
        self.listings.push(listing);
    }

    pub fn search(&self, query: &str) -> Vec<&AssetListing> {
        let q = query.to_lowercase();
        self.listings
            .iter()
            .filter(|l| {
                l.name.to_lowercase().contains(&q)
                    || l.description.to_lowercase().contains(&q)
                    || l.tags.iter().any(|t| t.to_lowercase().contains(&q))
            })
            .collect()
    }

    pub fn by_category(&self, category: MarketplaceCategory) -> Vec<&AssetListing> {
        self.listings
            .iter()
            .filter(|l| l.category == category)
            .collect()
    }

    pub fn top_rated(&self, count: usize) -> Vec<&AssetListing> {
        let mut sorted: Vec<&AssetListing> = self.listings.iter().collect();
        sorted.sort_by(|a, b| {
            b.rating
                .partial_cmp(&a.rating)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(count).collect()
    }

    pub fn most_downloaded(&self, count: usize) -> Vec<&AssetListing> {
        let mut sorted: Vec<&AssetListing> = self.listings.iter().collect();
        sorted.sort_by(|a, b| b.downloads.cmp(&a.downloads));
        sorted.into_iter().take(count).collect()
    }

    pub fn count(&self) -> usize {
        self.listings.len()
    }
}
