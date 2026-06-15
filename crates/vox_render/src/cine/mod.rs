//! Cinematic camera & sequencer engine.
//!
//! A pure [`CameraSequence`] (integer-frame clock, centripetal Catmull-Rom
//! position spline, cubic-Hermite scalar channels, SQUAD rotation, arc-length
//! constant-speed playback) is the single source of truth for keyframed
//! cinematics, trailers, and deterministic offline frame export.
//!
//! This module is built up task-by-task. Currently it contains the atomic
//! scalar [`Channel`].

mod channel;
#[cfg(feature = "spectra-native")]
mod pose_adapt;
mod rotation;
mod sequence;
mod spline;

pub use channel::{Channel, FrameRate, Interp, Key};
#[cfg(feature = "spectra-native")]
pub use pose_adapt::CineOrbit;
pub use rotation::QuatTrack;
pub use sequence::{BokehShape, CameraSequence, CinePose, CinematicConfig, FocusMode};
pub use spline::{ArcLengthLut, Spline3};
