use vox_render::gpu::entity_buffer::EntityIdBuffer;

#[test]
fn buffer_initialises_to_zero() {
    let buf = EntityIdBuffer::new(64, 64);
    assert_eq!(buf.pick(32, 32), 0);
}

#[test]
fn write_and_pick() {
    let mut buf = EntityIdBuffer::new(64, 64);
    buf.write(10, 20, 42);
    assert_eq!(buf.pick(10, 20), 42);
    assert_eq!(buf.pick(11, 20), 0);
}
