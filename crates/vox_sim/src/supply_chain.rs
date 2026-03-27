use crate::economy::{ResourceType, ResourceStock};
use crate::buildings::{BuildingManager, BuildingType};

/// Manages resource production, processing, and distribution.
pub struct SupplyChainManager {
    pub stocks: Vec<ResourceStock>,
}

impl SupplyChainManager {
    pub fn new() -> Self {
        let mut stocks = Vec::new();
        for rt in [
            ResourceType::Timber,
            ResourceType::Stone,
            ResourceType::Iron,
            ResourceType::Wheat,
            ResourceType::Planks,
            ResourceType::Tools,
            ResourceType::Bread,
        ] {
            stocks.push(ResourceStock {
                resource_type: rt,
                capacity: 1000.0,
                current: 0.0,
                production_rate: 0.0,
                consumption_rate: 0.0,
            });
        }
        Self { stocks }
    }

    /// Tick: industrial buildings produce, commercial buildings distribute.
    pub fn tick(&mut self, buildings: &BuildingManager, dt: f32) {
        for building in &buildings.buildings {
            if !building.operational {
                continue;
            }
            match building.building_type {
                BuildingType::Industrial => {
                    self.produce(ResourceType::Timber, dt * 10.0);
                    self.produce(ResourceType::Wheat, dt * 8.0);
                }
                BuildingType::Commercial => {
                    let timber = self.consume(ResourceType::Timber, dt * 5.0);
                    self.produce_amount(ResourceType::Planks, timber * 0.8);
                    let wheat = self.consume(ResourceType::Wheat, dt * 4.0);
                    self.produce_amount(ResourceType::Bread, wheat * 0.9);
                }
                _ => {}
            }
        }
    }

    fn produce(&mut self, resource: ResourceType, amount: f32) {
        if let Some(stock) = self.stocks.iter_mut().find(|s| s.resource_type == resource) {
            stock.current = (stock.current + amount).min(stock.capacity);
        }
    }

    fn produce_amount(&mut self, resource: ResourceType, amount: f32) {
        self.produce(resource, amount);
    }

    fn consume(&mut self, resource: ResourceType, amount: f32) -> f32 {
        if let Some(stock) = self.stocks.iter_mut().find(|s| s.resource_type == resource) {
            let consumed = amount.min(stock.current);
            stock.current -= consumed;
            consumed
        } else {
            0.0
        }
    }

    pub fn stock_level(&self, resource: ResourceType) -> f32 {
        self.stocks
            .iter()
            .find(|s| s.resource_type == resource)
            .map(|s| s.current)
            .unwrap_or(0.0)
    }
}

impl Default for SupplyChainManager {
    fn default() -> Self {
        Self::new()
    }
}
