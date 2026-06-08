//! `GpuTimers` — a small, game-agnostic GPU timestamp harness.
//!
//! "GPU is Alfa omega": a 2026 AAA engine's frame budget must be MEASURED, not
//! asserted. This wraps wgpu 24's `TIMESTAMP_QUERY` into a reusable harness that
//! brackets one or more passes (`pairs` begin/end slots) and resolves each into a
//! real GPU-millisecond reading. It is the measurement floor every later GPU gap
//! (Spec 05 tiled raster, Spec 11 residency) regresses against — instrument once,
//! reuse with no API change.
//!
//! No-panic contract (the house pattern): when the device was not granted
//! `TIMESTAMP_QUERY`, [`GpuTimers::new`] returns the same all-`None` state as
//! [`GpuTimers::disabled`] — it creates NO query set and NO buffers, and every
//! method degrades gracefully (writes return `None`, `resolve` is a no-op,
//! `resolve_ms` returns `None`). Nothing here ever `unwrap`s a query/map result.

/// Bytes per timestamp value (`u64` ticks).
const TS_BYTES: u64 = 8;
/// `resolve_query_set` destination offset alignment (wgpu `QUERY_RESOLVE_BUFFER_ALIGNMENT`).
const RESOLVE_ALIGN: u64 = 256;

/// A reusable GPU timestamp harness covering `pairs` begin/end timestamp slots.
pub struct GpuTimers {
    enabled: bool,
    query_set: Option<wgpu::QuerySet>,
    /// Destination of `resolve_query_set` (QUERY_RESOLVE | COPY_SRC).
    resolve_buf: Option<wgpu::Buffer>,
    /// CPU-mappable copy of `resolve_buf` (MAP_READ | COPY_DST).
    readback_buf: Option<wgpu::Buffer>,
    /// Nanoseconds per timestamp tick (`Queue::get_timestamp_period`); `0.0` ⇒ unusable.
    period_ns: f32,
    pairs: u32,
}

impl GpuTimers {
    /// Build a harness for `pairs` begin/end slots, gated on the GRANTED device
    /// features. If `features` lacks `TIMESTAMP_QUERY`, this is exactly
    /// [`Self::disabled`] — no query set, no buffers, no panic.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, features: wgpu::Features, pairs: u32) -> Self {
        let pairs = pairs.max(1);
        if !features.contains(wgpu::Features::TIMESTAMP_QUERY) || pairs == 0 {
            return Self::disabled();
        }
        let count = pairs * 2;
        let raw = count as u64 * TS_BYTES;
        // resolve dest must be 256-aligned; map both buffers at the same padded size.
        let size = raw.next_multiple_of(RESOLVE_ALIGN);

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_timers_qset"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timers_resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timers_readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let period_ns = queue.get_timestamp_period();

        Self {
            enabled: period_ns > 0.0,
            query_set: Some(query_set),
            resolve_buf: Some(resolve_buf),
            readback_buf: Some(readback_buf),
            period_ns,
            pairs,
        }
    }

    /// The inert harness: no GPU resources, every method a graceful no-op. Used on
    /// adapters without `TIMESTAMP_QUERY`, when `OCHROMA_NO_TIMESTAMP=1`, and as
    /// the timer the untimed `relight()` path passes so it stays byte-identical.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            query_set: None,
            resolve_buf: None,
            readback_buf: None,
            period_ns: 0.0,
            pairs: 0,
        }
    }

    /// Is timestamp measurement live (feature granted AND a usable period)?
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// `timestamp_writes` for a compute pass measuring `slot` (begin → end).
    /// `None` when disabled or `slot` is out of range — the pass then runs untimed.
    pub fn compute_writes(&self, slot: u32) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let qs = self.query_set.as_ref()?;
        if !self.enabled || slot >= self.pairs {
            return None;
        }
        Some(wgpu::ComputePassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(slot * 2),
            end_of_pass_write_index: Some(slot * 2 + 1),
        })
    }

    /// `timestamp_writes` for a render pass measuring `slot` (begin → end).
    pub fn render_writes(&self, slot: u32) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let qs = self.query_set.as_ref()?;
        if !self.enabled || slot >= self.pairs {
            return None;
        }
        Some(wgpu::RenderPassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(slot * 2),
            end_of_pass_write_index: Some(slot * 2 + 1),
        })
    }

    /// Resolve all slots' timestamps into the readback buffer. Encode this AFTER
    /// the measured passes and BEFORE `queue.submit`. No-op when disabled.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled {
            return;
        }
        let (Some(qs), Some(resolve), Some(readback)) =
            (self.query_set.as_ref(), self.resolve_buf.as_ref(), self.readback_buf.as_ref())
        else {
            return;
        };
        let count = self.pairs * 2;
        encoder.resolve_query_set(qs, 0..count, resolve, 0);
        encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, count as u64 * TS_BYTES);
    }

    /// Read back `slot`'s elapsed GPU time in milliseconds. Call after the submit
    /// whose encoder ran [`Self::resolve`]. Returns `None` (never panics) when
    /// disabled, out of range, the map fails, the period is unusable, or the
    /// result is non-finite. Does one blocking `poll(Wait)` — the same stall class
    /// as the existing relight/GI readbacks (Spec 11 later defers this).
    pub fn resolve_ms(&self, device: &wgpu::Device, slot: u32) -> Option<f32> {
        if !self.enabled || slot >= self.pairs || self.period_ns <= 0.0 {
            return None;
        }
        let readback = self.readback_buf.as_ref()?;

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        let mapped_ok = matches!(rx.recv(), Ok(Ok(())));
        if !mapped_ok {
            // map failed: nothing to unmap (mapping did not succeed).
            return None;
        }

        let ms = {
            let data = slice.get_mapped_range();
            let ticks: &[u64] = bytemuck::cast_slice(&data);
            let i = (slot * 2) as usize;
            if i + 1 >= ticks.len() {
                None
            } else {
                let start = ticks[i];
                let end = ticks[i + 1];
                let ms = (end.wrapping_sub(start) as f64 * self.period_ns as f64 / 1.0e6) as f32;
                if ms.is_finite() { Some(ms) } else { None }
            }
        };
        // ALWAYS unmap (no-panic contract), even if the slice was empty/short.
        readback.unmap();
        ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_timer_is_inert_and_never_panics() {
        let t = GpuTimers::disabled();
        assert!(!t.is_enabled(), "disabled timer must report disabled");
        assert!(t.compute_writes(0).is_none(), "no compute writes when disabled");
        assert!(t.render_writes(0).is_none(), "no render writes when disabled");
        // resolve_ms needs a device; covered by the GPU-gated relight test. Here we
        // assert the pure no-GPU surface degrades to None without a query set.
        assert_eq!(t.pairs, 0);
        assert_eq!(t.period_ns, 0.0);
    }
}
