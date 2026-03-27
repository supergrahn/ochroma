use vox_core::i18n::{I18nManager, Locale, TranslationBundle};

fn setup_manager() -> I18nManager {
    let mut mgr = I18nManager::new(Locale::En);

    let mut en = TranslationBundle::new();
    en.insert("greeting", "Hello");
    en.insert("welcome", "Welcome to {city_name}");
    en.insert("population", "Population: {count}");
    en.insert("english_only", "Only in English");
    mgr.load_bundle(Locale::En, en);

    let mut de = TranslationBundle::new();
    de.insert("greeting", "Hallo");
    de.insert("welcome", "Willkommen in {city_name}");
    mgr.load_bundle(Locale::De, de);

    let mut fr = TranslationBundle::new();
    fr.insert("greeting", "Bonjour");
    mgr.load_bundle(Locale::Fr, fr);

    mgr
}

#[test]
fn english_translation() {
    let mgr = setup_manager();
    assert_eq!(mgr.t("greeting"), "Hello");
}

#[test]
fn german_translation() {
    let mut mgr = setup_manager();
    mgr.set_locale(Locale::De);
    assert_eq!(mgr.t("greeting"), "Hallo");
}

#[test]
fn fallback_to_english() {
    let mut mgr = setup_manager();
    mgr.set_locale(Locale::De);
    // "english_only" not in German bundle, should fall back to English.
    assert_eq!(mgr.t("english_only"), "Only in English");
}

#[test]
fn missing_key_returns_key() {
    let mgr = setup_manager();
    assert_eq!(mgr.t("nonexistent_key"), "nonexistent_key");
}

#[test]
fn template_substitution() {
    let mgr = setup_manager();
    let result = mgr.t_with_args("welcome", &[("city_name", "Oakford")]);
    assert_eq!(result, "Welcome to Oakford");
}

#[test]
fn template_multiple_args() {
    let mgr = setup_manager();
    let result = mgr.t_with_args("population", &[("count", "42000")]);
    assert_eq!(result, "Population: 42000");
}

#[test]
fn rtl_detection() {
    assert!(Locale::Ar.is_rtl());
    assert!(!Locale::En.is_rtl());
    assert!(!Locale::De.is_rtl());
    assert!(!Locale::Ja.is_rtl());
}

#[test]
fn locale_tag() {
    assert_eq!(Locale::En.tag(), "en");
    assert_eq!(Locale::Ja.tag(), "ja");
    assert_eq!(Locale::Ar.tag(), "ar");
}

#[test]
fn german_template_substitution() {
    let mut mgr = setup_manager();
    mgr.set_locale(Locale::De);
    let result = mgr.t_with_args("welcome", &[("city_name", "Berlin")]);
    assert_eq!(result, "Willkommen in Berlin");
}
