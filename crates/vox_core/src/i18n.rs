use std::collections::HashMap;

/// Supported locales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Locale {
    En,
    De,
    Fr,
    Es,
    Ja,
    Zh,
    Ar,
    Pt,
    Ko,
    Ru,
}

impl Locale {
    /// Returns true if this locale uses right-to-left text direction.
    pub fn is_rtl(&self) -> bool {
        matches!(self, Locale::Ar)
    }

    /// BCP 47 language tag.
    pub fn tag(&self) -> &'static str {
        match self {
            Locale::En => "en",
            Locale::De => "de",
            Locale::Fr => "fr",
            Locale::Es => "es",
            Locale::Ja => "ja",
            Locale::Zh => "zh",
            Locale::Ar => "ar",
            Locale::Pt => "pt",
            Locale::Ko => "ko",
            Locale::Ru => "ru",
        }
    }
}

/// A bundle of translated strings for one locale.
#[derive(Debug, Clone, Default)]
pub struct TranslationBundle {
    pub messages: HashMap<String, String>,
}

impl TranslationBundle {
    pub fn new() -> Self {
        Self {
            messages: HashMap::new(),
        }
    }

    /// Insert a message.
    pub fn insert(&mut self, key: &str, value: &str) {
        self.messages.insert(key.to_string(), value.to_string());
    }

    /// Look up a message by key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.messages.get(key).map(|s| s.as_str())
    }
}

/// Internationalisation manager. Resolves translations with fallback to English.
#[derive(Debug)]
pub struct I18nManager {
    bundles: HashMap<Locale, TranslationBundle>,
    current_locale: Locale,
}

impl I18nManager {
    pub fn new(locale: Locale) -> Self {
        Self {
            bundles: HashMap::new(),
            current_locale: locale,
        }
    }

    /// Set the active locale.
    pub fn set_locale(&mut self, locale: Locale) {
        self.current_locale = locale;
    }

    /// Get the active locale.
    pub fn locale(&self) -> Locale {
        self.current_locale
    }

    /// Load a translation bundle for a locale.
    pub fn load_bundle(&mut self, locale: Locale, bundle: TranslationBundle) {
        self.bundles.insert(locale, bundle);
    }

    /// Translate a key. Falls back to English, then returns the key itself.
    pub fn t<'a>(&'a self, key: &'a str) -> &'a str {
        // Try current locale first.
        if let Some(bundle) = self.bundles.get(&self.current_locale)
            && let Some(msg) = bundle.get(key) {
                return msg;
            }

        // Fallback to English.
        if self.current_locale != Locale::En
            && let Some(bundle) = self.bundles.get(&Locale::En)
                && let Some(msg) = bundle.get(key) {
                    return msg;
                }

        // Return the key itself.
        key
    }

    /// Translate with argument substitution. Placeholders use `{name}` syntax.
    pub fn t_with_args(&self, key: &str, args: &[(&str, &str)]) -> String {
        let template = self.t(key);
        let mut result = template.to_string();
        for (name, value) in args {
            result = result.replace(&format!("{{{}}}", name), value);
        }
        result
    }
}

impl Default for I18nManager {
    fn default() -> Self {
        Self::new(Locale::En)
    }
}
