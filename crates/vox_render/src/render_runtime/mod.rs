//! Reusable render runtime — engine-side render service so games don't re-implement optimizations.

pub mod env;
#[cfg(feature = "spectra-native")]
pub mod atlas;
#[cfg(feature = "spectra-native")]
pub mod frame;
#[cfg(feature = "spectra-native")]
pub mod runtime;
pub mod color;
pub mod geometry;
pub mod hash;
pub mod illuminant;
pub mod terrain;

pub use terrain::TerrainUpload;

#[cfg(feature = "spectra-native")]
pub use runtime::{GpuPresentResult, PresentResult, RenderRuntime};
