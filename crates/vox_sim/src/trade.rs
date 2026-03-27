use serde::{Deserialize, Serialize};

use crate::economy::ResourceType;

/// A trade partner (external city/region).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradePartner {
    pub name: String,
    pub buy_prices: Vec<(ResourceType, f32)>,  // what they'll pay for our exports
    pub sell_prices: Vec<(ResourceType, f32)>,  // what they charge for imports
    pub trust_level: f32,                        // 0.0-1.0, improves with trade volume
}

/// Dynamic pricing based on supply and demand.
#[derive(Debug, Clone)]
pub struct MarketPrice {
    pub resource: ResourceType,
    pub base_price: f32,
    pub current_price: f32,
    pub supply: f32,
    pub demand: f32,
}

impl MarketPrice {
    pub fn new(resource: ResourceType, base_price: f32) -> Self {
        Self {
            resource,
            base_price,
            current_price: base_price,
            supply: 0.0,
            demand: 0.0,
        }
    }

    /// Recalculate price based on supply/demand ratio.
    pub fn update(&mut self) {
        if self.supply <= 0.0 {
            self.current_price = self.base_price * 3.0; // scarcity premium
        } else {
            let ratio = self.demand / self.supply;
            // Price increases when demand > supply, decreases when supply > demand
            self.current_price = self.base_price * ratio.clamp(0.3, 3.0);
        }
    }
}

pub struct TradeSystem {
    pub partners: Vec<TradePartner>,
    pub market_prices: Vec<MarketPrice>,
    pub trade_balance: f64, // total exports - imports
}

impl TradeSystem {
    pub fn new() -> Self {
        Self {
            partners: vec![
                TradePartner {
                    name: "Riverside".into(),
                    buy_prices: vec![
                        (ResourceType::Timber, 10.0),
                        (ResourceType::Iron, 25.0),
                    ],
                    sell_prices: vec![
                        (ResourceType::Wheat, 5.0),
                        (ResourceType::Stone, 8.0),
                    ],
                    trust_level: 0.5,
                },
                TradePartner {
                    name: "Mountain Hold".into(),
                    buy_prices: vec![
                        (ResourceType::Wheat, 8.0),
                        (ResourceType::Bread, 15.0),
                    ],
                    sell_prices: vec![
                        (ResourceType::Stone, 6.0),
                        (ResourceType::Iron, 20.0),
                    ],
                    trust_level: 0.3,
                },
            ],
            market_prices: vec![
                MarketPrice::new(ResourceType::Timber, 8.0),
                MarketPrice::new(ResourceType::Stone, 10.0),
                MarketPrice::new(ResourceType::Iron, 20.0),
                MarketPrice::new(ResourceType::Wheat, 4.0),
                MarketPrice::new(ResourceType::Bread, 12.0),
            ],
            trade_balance: 0.0,
        }
    }

    /// Execute an export trade.
    pub fn export(
        &mut self,
        partner_name: &str,
        resource: ResourceType,
        quantity: f32,
    ) -> f64 {
        if let Some(partner) = self.partners.iter_mut().find(|p| p.name == partner_name) {
            if let Some((_, price)) = partner.buy_prices.iter().find(|(r, _)| *r == resource) {
                let revenue = quantity as f64 * *price as f64;
                partner.trust_level = (partner.trust_level + 0.01).min(1.0);
                self.trade_balance += revenue;
                return revenue;
            }
        }
        0.0
    }

    /// Execute an import trade.
    pub fn import(
        &mut self,
        partner_name: &str,
        resource: ResourceType,
        quantity: f32,
    ) -> f64 {
        if let Some(partner) = self.partners.iter_mut().find(|p| p.name == partner_name) {
            if let Some((_, price)) = partner.sell_prices.iter().find(|(r, _)| *r == resource) {
                let cost = quantity as f64 * *price as f64;
                partner.trust_level = (partner.trust_level + 0.01).min(1.0);
                self.trade_balance -= cost;
                return cost;
            }
        }
        0.0
    }

    /// Update all market prices.
    pub fn update_prices(&mut self) {
        for price in &mut self.market_prices {
            price.update();
        }
    }

    pub fn price_of(&self, resource: ResourceType) -> f32 {
        self.market_prices
            .iter()
            .find(|p| p.resource == resource)
            .map(|p| p.current_price)
            .unwrap_or(0.0)
    }
}

impl Default for TradeSystem {
    fn default() -> Self {
        Self::new()
    }
}
