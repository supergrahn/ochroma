use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CityBudget {
    pub funds: f64,
    pub residential_tax_rate: f32,
    pub commercial_tax_rate: f32,
    pub industrial_tax_rate: f32,
    pub residential_income: f64,
    pub commercial_income: f64,
    pub industrial_income: f64,
    pub infrastructure_expenses: f64,
    pub services_expenses: f64,
    pub maintenance_expenses: f64,
}

impl Default for CityBudget {
    fn default() -> Self {
        Self {
            funds: 50_000.0,
            residential_tax_rate: 0.08,
            commercial_tax_rate: 0.10,
            industrial_tax_rate: 0.12,
            residential_income: 0.0,
            commercial_income: 0.0,
            industrial_income: 0.0,
            infrastructure_expenses: 0.0,
            services_expenses: 0.0,
            maintenance_expenses: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetReport {
    pub total_income: f64,
    pub total_expenses: f64,
    pub net: f64,
    pub funds: f64,
    pub citizen_count: u32,
}

impl CityBudget {
    pub fn generate_report(&self, citizen_count: u32) -> BudgetReport {
        BudgetReport {
            total_income: self.total_income(),
            total_expenses: self.total_expenses(),
            net: self.net(),
            funds: self.funds,
            citizen_count,
        }
    }

    pub fn total_income(&self) -> f64 {
        self.residential_income + self.commercial_income + self.industrial_income
    }

    pub fn total_expenses(&self) -> f64 {
        self.infrastructure_expenses + self.services_expenses + self.maintenance_expenses
    }

    pub fn net(&self) -> f64 {
        self.total_income() - self.total_expenses()
    }

    /// Simulate one budget tick. Updates income based on citizen count and zone data.
    pub fn tick(&mut self, citizen_count: u32, commercial_count: u32, industrial_count: u32) {
        self.residential_income = citizen_count as f64 * self.residential_tax_rate as f64 * 100.0;
        self.commercial_income = commercial_count as f64 * self.commercial_tax_rate as f64 * 500.0;
        self.industrial_income = industrial_count as f64 * self.industrial_tax_rate as f64 * 800.0;
        self.funds += self.net();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    Power,
    Water,
    Sewage,
    Waste,
    NaturalGas,
    Broadband,
    Timber,
    Wheat,
    Steel,
    Coal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStock {
    pub resource_type: ResourceType,
    pub capacity: f32,
    pub current: f32,
    pub production_rate: f32,
    pub consumption_rate: f32,
}

impl ResourceStock {
    pub fn surplus(&self) -> f32 {
        self.production_rate - self.consumption_rate
    }

    pub fn fill_ratio(&self) -> f32 {
        if self.capacity > 0.0 {
            self.current / self.capacity
        } else {
            0.0
        }
    }
}

/// Manages resource flow from producers to consumers.
pub struct SupplyChain {
    pub stocks: Vec<ResourceStock>,
}

impl SupplyChain {
    pub fn new() -> Self {
        Self { stocks: Vec::new() }
    }

    pub fn add_stock(&mut self, resource: ResourceType, capacity: f32) {
        self.stocks.push(ResourceStock {
            resource_type: resource,
            capacity,
            current: 0.0,
            production_rate: 0.0,
            consumption_rate: 0.0,
        });
    }

    pub fn produce(&mut self, resource: ResourceType, amount: f32) {
        if let Some(stock) = self.stocks.iter_mut().find(|s| s.resource_type == resource) {
            stock.current = (stock.current + amount).min(stock.capacity);
        }
    }

    pub fn consume(&mut self, resource: ResourceType, amount: f32) -> f32 {
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

impl Default for SupplyChain {
    fn default() -> Self {
        Self::new()
    }
}
