use vox_data::templates::*;

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
fn at_least_4_templates() {
    assert!(available_templates().len() >= 4);
}

#[test]
fn complexity_ordering() {
    assert!(Complexity::Beginner < Complexity::Expert);
}

#[test]
fn no_city_builder_genre() {
    // Engine must not ship a game-specific city-builder template.
    let templates = available_templates();
    for t in &templates {
        assert_ne!(
            t.name, "City Builder",
            "engine must not ship a city-builder template"
        );
    }
}
