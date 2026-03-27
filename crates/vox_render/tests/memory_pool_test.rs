use vox_render::memory_pool::*;

#[test]
fn acquire_and_release() {
    let mut pool = BufferPool::new(4, 1024);
    assert_eq!(pool.available(), 4);

    let buf = pool.acquire().expect("should acquire");
    assert!(buf.in_use);
    assert_eq!(pool.available(), 3);

    pool.release(&buf);
    assert_eq!(pool.available(), 4);
}

#[test]
fn pool_exhaustion_returns_none() {
    let mut pool = BufferPool::new(2, 512);
    let _b1 = pool.acquire().unwrap();
    let _b2 = pool.acquire().unwrap();

    // Pool exhausted.
    assert!(pool.acquire().is_none());
}

#[test]
fn release_makes_buffer_available_again() {
    let mut pool = BufferPool::new(1, 256);
    let buf = pool.acquire().unwrap();
    assert!(pool.acquire().is_none());

    pool.release(&buf);
    let buf2 = pool.acquire();
    assert!(buf2.is_some());
    assert_eq!(buf2.unwrap().id, buf.id);
}

#[test]
fn stats_tracking() {
    let mut pool = BufferPool::new(4, 1024);
    let stats = pool.stats();
    assert_eq!(stats.total_allocated, 4 * 1024);
    assert_eq!(stats.total_used, 0);
    assert_eq!(stats.pool_count, 4);

    let _b1 = pool.acquire().unwrap();
    let _b2 = pool.acquire().unwrap();
    let stats = pool.stats();
    assert_eq!(stats.total_used, 2 * 1024);
}

#[test]
fn ring_buffer_write_and_read() {
    let mut ring = RingBuffer::new(256);
    let data = [1u8, 2, 3, 4];
    let offset = ring.write(&data);
    assert_eq!(offset, 0);
    assert_eq!(ring.usage(), 4);
    assert_eq!(ring.read_at(0, 4), &[1, 2, 3, 4]);
}

#[test]
fn ring_buffer_wraps() {
    let mut ring = RingBuffer::new(8);

    // Write 6 bytes.
    ring.write(&[1, 2, 3, 4, 5, 6]);
    assert_eq!(ring.usage(), 6);

    // Write 4 more — wraps around.
    let offset = ring.write(&[7, 8, 9, 10]);
    assert_eq!(offset, 6); // Starts at position 6.
    assert_eq!(ring.usage(), 10);

    // Data at positions 0-1 was overwritten by wrap.
    assert_eq!(ring.read_at(0, 2), &[9, 10]);
}

#[test]
fn ring_buffer_reset() {
    let mut ring = RingBuffer::new(64);
    ring.write(&[1, 2, 3]);
    assert_eq!(ring.usage(), 3);

    ring.reset();
    assert_eq!(ring.usage(), 0);
}
