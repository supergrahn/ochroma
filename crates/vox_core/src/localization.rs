use std::collections::HashMap;
use std::path::Path;

/// Localization manager — loads string tables and resolves keys.
pub struct Localization {
    pub current_locale: String,
    pub fallback_locale: String,
    tables: HashMap<String, HashMap<String, String>>, // locale -> (key -> translated)
}

impl Localization {
    pub fn new(default_locale: &str) -> Self {
        Self {
            current_locale: default_locale.to_string(),
            fallback_locale: default_locale.to_string(),
            tables: HashMap::new(),
        }
    }

    /// Load translations from a CSV file. Format: `key,text` per line.
    /// Returns number of entries loaded.
    pub fn load_csv(&mut self, locale: &str, path: &Path) -> Result<usize, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
        Ok(self.load_from_string(locale, &content))
    }

    /// Load translations from a CSV string. Format: `key,text` per line.
    /// Returns number of entries loaded.
    pub fn load_from_string(&mut self, locale: &str, csv: &str) -> usize {
        let table = self
            .tables
            .entry(locale.to_string())
            .or_default();
        let mut count = 0;
        for line in csv.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, value)) = line.split_once(',') {
                table.insert(key.trim().to_string(), value.trim().to_string());
                count += 1;
            }
        }
        count
    }

    /// Switch the active locale.
    pub fn set_locale(&mut self, locale: &str) {
        self.current_locale = locale.to_string();
    }

    /// Get a translated string. Falls back to fallback locale, then returns the key itself.
    pub fn get<'a>(&'a self, key: &'a str) -> &'a str {
        // Try current locale
        if let Some(table) = self.tables.get(&self.current_locale)
            && let Some(val) = table.get(key) {
                return val.as_str();
            }
        // Try fallback
        if self.current_locale != self.fallback_locale
            && let Some(table) = self.tables.get(&self.fallback_locale)
                && let Some(val) = table.get(key) {
                    return val.as_str();
                }
        // Return key itself
        key
    }

    /// Get a translated string with argument substitution.
    /// Args are `(placeholder, value)` pairs. Placeholder format in string: `{name}`.
    pub fn get_with_args(&self, key: &str, args: &[(&str, &str)]) -> String {
        let template = self.get(key).to_string();
        let mut result = template;
        for (name, value) in args {
            let placeholder = format!("{{{}}}", name);
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// List available locales.
    pub fn available_locales(&self) -> Vec<&str> {
        self.tables.keys().map(|s| s.as_str()).collect()
    }

    /// Number of keys in a specific locale.
    pub fn key_count(&self, locale: &str) -> usize {
        self.tables
            .get(locale)
            .map(|t| t.len())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_csv_string() {
        let mut loc = Localization::new("en");
        let csv = "menu.play,Play Game\nmenu.quit,Quit\nhud.health,Health: {value}";
        let count = loc.load_from_string("en", csv);
        assert_eq!(count, 3);
        assert_eq!(loc.key_count("en"), 3);
    }

    #[test]
    fn get_returns_translated() {
        let mut loc = Localization::new("en");
        loc.load_from_string("en", "menu.play,Play Game\nmenu.quit,Quit");
        assert_eq!(loc.get("menu.play"), "Play Game");
        assert_eq!(loc.get("menu.quit"), "Quit");
    }

    #[test]
    fn missing_key_returns_key() {
        let loc = Localization::new("en");
        assert_eq!(loc.get("nonexistent.key"), "nonexistent.key");
    }

    #[test]
    fn get_with_args_substitutes() {
        let mut loc = Localization::new("en");
        loc.load_from_string("en", "hud.health,Health: {value}/{max}");
        let result = loc.get_with_args("hud.health", &[("value", "80"), ("max", "100")]);
        assert_eq!(result, "Health: 80/100");
    }

    #[test]
    fn set_locale_switches() {
        let mut loc = Localization::new("en");
        loc.load_from_string("en", "menu.play,Play Game");
        loc.load_from_string("es", "menu.play,Jugar");

        assert_eq!(loc.get("menu.play"), "Play Game");

        loc.set_locale("es");
        assert_eq!(loc.get("menu.play"), "Jugar");

        // Falls back to en for missing keys in es
        loc.load_from_string("en", "menu.quit,Quit");
        assert_eq!(loc.get("menu.quit"), "Quit"); // fallback
    }
}
