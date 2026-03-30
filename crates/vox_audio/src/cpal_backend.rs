//! CPAL device backend — cross-platform audio output.
//! Full implementation in Task 5; this file establishes the dep import.

#[cfg(feature = "audio-backend")]
use cpal::traits::HostTrait as _;

/// Builder for the CPAL audio backend.
pub struct CpalBackendBuilder {
    preferred_sample_rate: Option<u32>,
}

impl CpalBackendBuilder {
    pub fn new() -> Self {
        Self { preferred_sample_rate: None }
    }

    pub fn sample_rate(mut self, hz: u32) -> Self {
        self.preferred_sample_rate = Some(hz);
        self
    }
}

impl Default for CpalBackendBuilder {
    fn default() -> Self { Self::new() }
}
