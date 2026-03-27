/// A buffer handle from the pool.
#[derive(Debug, Clone)]
pub struct PooledBuffer {
    pub id: usize,
    pub size: usize,
    pub in_use: bool,
}

/// Memory usage statistics.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub total_used: usize,
    pub pool_count: usize,
    pub ring_buffer_usage: usize,
}

/// Pre-allocated pool of fixed-size GPU buffers.
/// No allocation per frame -- acquire and release only.
pub struct BufferPool {
    buffer_size: usize,
    buffers: Vec<PooledBuffer>,
}

impl BufferPool {
    /// Create a pool with `count` buffers, each of `buffer_size` bytes.
    pub fn new(count: usize, buffer_size: usize) -> Self {
        let buffers = (0..count)
            .map(|id| PooledBuffer {
                id,
                size: buffer_size,
                in_use: false,
            })
            .collect();

        Self {
            buffer_size,
            buffers,
        }
    }

    /// Acquire a free buffer from the pool. Returns `None` if all buffers are in use.
    pub fn acquire(&mut self) -> Option<PooledBuffer> {
        for buf in &mut self.buffers {
            if !buf.in_use {
                buf.in_use = true;
                return Some(buf.clone());
            }
        }
        None
    }

    /// Release a buffer back to the pool.
    pub fn release(&mut self, buffer: &PooledBuffer) {
        if let Some(buf) = self.buffers.iter_mut().find(|b| b.id == buffer.id) {
            buf.in_use = false;
        }
    }

    /// Number of currently available (free) buffers.
    pub fn available(&self) -> usize {
        self.buffers.iter().filter(|b| !b.in_use).count()
    }

    /// Total number of buffers in the pool.
    pub fn total(&self) -> usize {
        self.buffers.len()
    }

    /// Get memory stats for this pool.
    pub fn stats(&self) -> MemoryStats {
        let in_use_count = self.buffers.iter().filter(|b| b.in_use).count();
        MemoryStats {
            total_allocated: self.buffers.len() * self.buffer_size,
            total_used: in_use_count * self.buffer_size,
            pool_count: self.buffers.len(),
            ring_buffer_usage: 0,
        }
    }
}

/// Circular ring buffer for per-frame uploads.
pub struct RingBuffer {
    capacity: usize,
    data: Vec<u8>,
    write_head: usize,
    bytes_written: usize,
}

impl RingBuffer {
    /// Create a ring buffer with the given capacity in bytes.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: vec![0u8; capacity],
            write_head: 0,
            bytes_written: 0,
        }
    }

    /// Write data into the ring buffer, returning the offset where data was written.
    /// Wraps around when reaching the end.
    pub fn write(&mut self, data: &[u8]) -> usize {
        let offset = self.write_head;

        for &byte in data {
            self.data[self.write_head] = byte;
            self.write_head = (self.write_head + 1) % self.capacity;
        }

        self.bytes_written += data.len();
        offset
    }

    /// Reset the ring buffer at frame end.
    pub fn reset(&mut self) {
        self.write_head = 0;
        self.bytes_written = 0;
    }

    /// Current usage in bytes since last reset.
    pub fn usage(&self) -> usize {
        self.bytes_written
    }

    /// Total capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Read data at a given offset (for testing).
    pub fn read_at(&self, offset: usize, len: usize) -> &[u8] {
        &self.data[offset..offset + len]
    }
}
