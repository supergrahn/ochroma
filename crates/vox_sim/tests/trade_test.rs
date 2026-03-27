use vox_sim::economy::ResourceType;
use vox_sim::trade::*;

#[test]
fn export_generates_revenue() {
    let mut trade = TradeSystem::new();
    let revenue = trade.export("Riverside", ResourceType::Timber, 100.0);
    assert!(revenue > 0.0);
    assert!(trade.trade_balance > 0.0);
}

#[test]
fn import_costs_money() {
    let mut trade = TradeSystem::new();
    let cost = trade.import("Mountain Hold", ResourceType::Stone, 50.0);
    assert!(cost > 0.0);
    assert!(trade.trade_balance < 0.0);
}

#[test]
fn scarcity_increases_price() {
    let mut price = MarketPrice::new(ResourceType::Iron, 20.0);
    price.supply = 0.0;
    price.demand = 100.0;
    price.update();
    assert!(
        price.current_price > 20.0,
        "Scarce resource should be expensive"
    );
}

#[test]
fn surplus_decreases_price() {
    let mut price = MarketPrice::new(ResourceType::Wheat, 4.0);
    price.supply = 1000.0;
    price.demand = 100.0;
    price.update();
    assert!(price.current_price < 4.0, "Surplus should lower price");
}

#[test]
fn trade_improves_trust() {
    let mut trade = TradeSystem::new();
    let initial_trust = trade.partners[0].trust_level;
    trade.export("Riverside", ResourceType::Timber, 10.0);
    assert!(trade.partners[0].trust_level > initial_trust);
}
