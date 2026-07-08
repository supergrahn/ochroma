//! Reusable render runtime — engine-side render service so games don't re-implement optimizations.

#[cfg(feature = "spectra-native")]
pub mod atlas;
pub mod color;
pub mod env;
#[cfg(feature = "spectra-native")]
pub mod frame;
pub mod geometry;
pub mod hash;
pub mod illuminant;
#[cfg(feature = "spectra-native")]
pub mod present;
#[cfg(feature = "spectra-native")]
pub mod runtime;
pub mod settings;
pub mod terrain;

pub use terrain::TerrainUpload;

#[cfg(feature = "spectra-native")]
pub use runtime::{GpuPresentResult, PresentResult, RenderRuntime};
