use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub severity: LogSeverity,
    pub message: String,
    pub timestamp: f64,
    pub source: String,
}

pub struct OutputLog {
    pub entries: VecDeque<LogEntry>,
    pub max_entries: usize,
    pub filter_severity: Option<LogSeverity>,
    pub filter_text: String,
    pub auto_scroll: bool,
}

impl OutputLog {
    pub fn new(max: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries: max,
            filter_severity: None,
            filter_text: String::new(),
            auto_scroll: true,
        }
    }

    pub fn log(&mut self, severity: LogSeverity, source: &str, message: &str, time: f64) {
        let entry = LogEntry {
            severity,
            message: message.to_string(),
            timestamp: time,
            source: source.to_string(),
        };
        self.entries.push_back(entry);
        while self.entries.len() > self.max_entries {
            self.entries.pop_front();
        }
    }

    pub fn info(&mut self, source: &str, msg: &str, time: f64) {
        self.log(LogSeverity::Info, source, msg, time);
    }

    pub fn warn(&mut self, source: &str, msg: &str, time: f64) {
        self.log(LogSeverity::Warning, source, msg, time);
    }

    pub fn error(&mut self, source: &str, msg: &str, time: f64) {
        self.log(LogSeverity::Error, source, msg, time);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn filtered_entries(&self) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| {
                if let Some(sev) = self.filter_severity
                    && e.severity != sev {
                        return false;
                    }
                if !self.filter_text.is_empty()
                    && !e.message.to_lowercase().contains(&self.filter_text.to_lowercase())
                    && !e.source.to_lowercase().contains(&self.filter_text.to_lowercase())
                {
                    return false;
                }
                true
            })
            .collect()
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn error_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.severity == LogSeverity::Error)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.severity == LogSeverity::Warning)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_adds_entries() {
        let mut log = OutputLog::new(100);
        log.info("engine", "started", 0.0);
        log.warn("physics", "slow frame", 1.0);
        log.error("script", "null ref", 2.0);
        assert_eq!(log.entry_count(), 3);
    }

    #[test]
    fn filter_by_severity() {
        let mut log = OutputLog::new(100);
        log.info("engine", "info msg", 0.0);
        log.warn("engine", "warn msg", 1.0);
        log.error("engine", "error msg", 2.0);
        log.filter_severity = Some(LogSeverity::Error);
        let filtered = log.filtered_entries();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].message, "error msg");
    }

    #[test]
    fn filter_by_text() {
        let mut log = OutputLog::new(100);
        log.info("engine", "loading assets", 0.0);
        log.info("engine", "player spawned", 1.0);
        log.info("physics", "collision detected", 2.0);
        log.filter_text = "player".to_string();
        let filtered = log.filtered_entries();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].message, "player spawned");
    }

    #[test]
    fn max_entries_evicts_old() {
        let mut log = OutputLog::new(3);
        for i in 0..5 {
            log.info("engine", &format!("msg {i}"), i as f64);
        }
        assert_eq!(log.entry_count(), 3);
        assert_eq!(log.entries[0].message, "msg 2");
        assert_eq!(log.entries[2].message, "msg 4");
    }

    #[test]
    fn clear_empties() {
        let mut log = OutputLog::new(100);
        log.info("engine", "a", 0.0);
        log.error("engine", "b", 1.0);
        log.clear();
        assert_eq!(log.entry_count(), 0);
        assert_eq!(log.error_count(), 0);
    }

    #[test]
    fn error_count_tracks() {
        let mut log = OutputLog::new(100);
        log.info("engine", "ok", 0.0);
        log.error("engine", "bad", 1.0);
        log.error("script", "worse", 2.0);
        log.warn("physics", "meh", 3.0);
        assert_eq!(log.error_count(), 2);
        assert_eq!(log.warning_count(), 1);
    }
}
