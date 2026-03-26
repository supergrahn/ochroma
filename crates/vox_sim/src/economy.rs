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

impl CityBudget {
    pub fn total_income(&self) -> f64 {
        self.residential_income + self.commercial_income + self.industrial_income
    }

    pub fn total_expenses(&self) -> f64 {
        self.infrastructure_expenses + self.services_expenses + self.maintenance_expenses
    }

    pub fn net(&self) -> f64 {
        self.total_income() - self.total_expenses()
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
