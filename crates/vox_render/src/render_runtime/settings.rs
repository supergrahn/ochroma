//! Engine-facing renderer settings types.
//!
//! These are lightweight config structs/enums used by games and tools without
//! pulling the full native Spectra renderer feature graph into non-render builds.

pub use spectra_types::{TierEntry, TierTable, UpscalerMode, UpscalerQuality};
