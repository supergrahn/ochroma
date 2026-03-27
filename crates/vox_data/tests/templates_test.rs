use vox_data::templates::*;

#[test]
fn has_city_builder_template() {
    let templates = available_templates();
    assert!(templates.iter().any(|t| t.genre == GameGenre::CityBuilder));
}

#[test]
fn all_templates_have_features() {
    let templates = available_templates();
    for t in &templates {
        assert!(
            !t.features.is_empty(),
            "Template '{}' should have features",
            t.name
        );
    }
}

#[test]
fn city_builder_has_zoning() {
    let templates = available_templates();
    let cb = templates
        .iter()
        .find(|t| t.genre == GameGenre::CityBuilder)
        .unwrap();
    assert!(cb.uses_feature("zoning"));
}

#[test]
fn at_least_5_templates() {
    assert!(available_templates().len() >= 5);
}

#[test]
fn complexity_ordering() {
    assert!(Complexity::Beginner < Complexity::Expert);
}
