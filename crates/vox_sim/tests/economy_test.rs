use vox_sim::economy::{CityBudget, ResourceType, SupplyChain};

#[test]
fn budget_tick_increases_funds() {
    let mut budget = CityBudget::default();
    let initial = budget.funds;
    budget.tick(1000, 50, 20);
    assert!(budget.funds > initial, "Budget should increase with citizens");
}

#[test]
fn supply_chain_produce_consume() {
    let mut chain = SupplyChain::new();
    chain.add_stock(ResourceType::Timber, 100.0);
    chain.produce(ResourceType::Timber, 50.0);
    assert!((chain.stock_level(ResourceType::Timber) - 50.0).abs() < 0.01);

    let consumed = chain.consume(ResourceType::Timber, 30.0);
    assert!((consumed - 30.0).abs() < 0.01);
    assert!((chain.stock_level(ResourceType::Timber) - 20.0).abs() < 0.01);
}

#[test]
fn consume_more_than_available() {
    let mut chain = SupplyChain::new();
    chain.add_stock(ResourceType::Wheat, 10.0);
    chain.produce(ResourceType::Wheat, 5.0);
    let consumed = chain.consume(ResourceType::Wheat, 20.0);
    assert!((consumed - 5.0).abs() < 0.01); // can only consume what's available
}
