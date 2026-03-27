use std::fmt;

/// Engine-wide error type for graceful error handling.
#[derive(Debug)]
pub enum EngineError {
    /// Asset file is corrupted or unreadable.
    AssetCorrupted { path: String, reason: String },
    /// A required asset is missing.
    AssetMissing { uuid: String },
    /// GPU initialization or rendering failed.
    RenderError { reason: String },
    /// Save file is corrupted.
    SaveCorrupted { path: String, reason: String },
    /// System resource exhausted (VRAM, RAM, etc.).
    ResourceExhausted { resource: String },
    /// Generic IO error.
    IoError(std::io::Error),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AssetCorrupted { path, reason } => write!(f, "Corrupted asset '{}': {}", path, reason),
            Self::AssetMissing { uuid } => write!(f, "Missing asset: {}", uuid),
            Self::RenderError { reason } => write!(f, "Render error: {}", reason),
            Self::SaveCorrupted { path, reason } => write!(f, "Corrupted save '{}': {}", path, reason),
            Self::ResourceExhausted { resource } => write!(f, "Resource exhausted: {}", resource),
            Self::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<std::io::Error> for EngineError {
    fn from(e: std::io::Error) -> Self { Self::IoError(e) }
}

/// Log an error without crashing. Returns a default value.
pub fn recover<T: Default>(error: EngineError) -> T {
    eprintln!("[ochroma-error] {}", error);
    T::default()
}

/// Log an error with context and return a default.
pub fn recover_with_context<T: Default>(error: EngineError, context: &str) -> T {
    eprintln!("[ochroma-error] {}: {}", context, error);
    T::default()
}
