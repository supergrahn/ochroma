//! WorldChunkGrid replication — generic spatial simulation state sync (server→clients only).
//!
//! Grid chunked into CHUNK_SIZE×CHUNK_SIZE chunks.
//! Per-cell values quantized to u8. Chunk packet = 4-byte header + 5*256 u8 bytes.

pub const CHUNK_SIZE: usize = 16;
pub const CELLS_PER_CHUNK: usize = CHUNK_SIZE * CHUNK_SIZE;

#[derive(Clone, Copy, Default)]
pub struct WorldCellF32 {
    pub channel_a: f32,
    pub channel_b: f32,
    pub channel_c: f32,
    pub channel_d: f32,
    pub channel_e: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldCellPacked {
    pub channel_a: u8,
    pub channel_b: u8,
    pub channel_c: u8,
    pub channel_d: u8,
    pub channel_e: u8,
}

impl WorldCellPacked {
    pub fn from_f32(c: &WorldCellF32) -> Self {
        Self {
            channel_a: (c.channel_a.clamp(0.0, 1.0) * 255.0) as u8,
            channel_b: (c.channel_b.clamp(0.0, 1.0) * 255.0) as u8,
            channel_c: (c.channel_c.clamp(0.0, 1.0) * 255.0) as u8,
            channel_d: (c.channel_d.clamp(0.0, 1.0) * 255.0) as u8,
            channel_e: (c.channel_e.clamp(0.0, 1.0) * 255.0) as u8,
        }
    }

    pub fn to_f32(&self) -> WorldCellF32 {
        WorldCellF32 {
            channel_a: self.channel_a as f32 / 255.0,
            channel_b: self.channel_b as f32 / 255.0,
            channel_c: self.channel_c as f32 / 255.0,
            channel_d: self.channel_d as f32 / 255.0,
            channel_e: self.channel_e as f32 / 255.0,
        }
    }
}

pub struct WorldChunk {
    pub chunk_x: u16,
    pub chunk_z: u16,
    pub cells: [WorldCellPacked; CELLS_PER_CHUNK],
}

impl WorldChunk {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + CELLS_PER_CHUNK * 5);
        buf.extend_from_slice(&self.chunk_x.to_le_bytes());
        buf.extend_from_slice(&self.chunk_z.to_le_bytes());
        for cell in &self.cells {
            buf.push(cell.channel_a);
            buf.push(cell.channel_b);
            buf.push(cell.channel_c);
            buf.push(cell.channel_d);
            buf.push(cell.channel_e);
        }
        buf
    }

    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 4 + CELLS_PER_CHUNK * 5 { return None; }
        let chunk_x = u16::from_le_bytes([data[0], data[1]]);
        let chunk_z = u16::from_le_bytes([data[2], data[3]]);
        let mut cells = [WorldCellPacked::default(); CELLS_PER_CHUNK];
        for (i, cell) in cells.iter_mut().enumerate() {
            let base = 4 + i * 5;
            cell.channel_a = data[base];
            cell.channel_b = data[base + 1];
            cell.channel_c = data[base + 2];
            cell.channel_d = data[base + 3];
            cell.channel_e = data[base + 4];
        }
        Some(WorldChunk { chunk_x, chunk_z, cells })
    }
}

pub struct WorldChunkGridNet {
    pub cells: Vec<WorldCellF32>,
    pub width: u32,
    pub height: u32,
}

impl WorldChunkGridNet {
    pub fn to_chunks(&self, chunk_size: usize) -> Vec<WorldChunk> {
        assert_eq!(
            chunk_size, CHUNK_SIZE,
            "to_chunks: chunk_size must equal CHUNK_SIZE={} (WorldChunk::cells is a fixed-size array)",
            CHUNK_SIZE
        );
        let chunks_x = (self.width as usize).div_ceil(chunk_size);
        let chunks_z = (self.height as usize).div_ceil(chunk_size);
        let w = self.width as usize;
        let mut chunks = Vec::with_capacity(chunks_x * chunks_z);
        for cz in 0..chunks_z {
            for cx in 0..chunks_x {
                let mut packed_cells = [WorldCellPacked::default(); CELLS_PER_CHUNK];
                for lz in 0..chunk_size {
                    for lx in 0..chunk_size {
                        let gx = cx * chunk_size + lx;
                        let gz = cz * chunk_size + lz;
                        let local_idx = lz * chunk_size + lx;
                        if gx < self.width as usize && gz < self.height as usize {
                            let g_idx = gz * w + gx;
                            packed_cells[local_idx] = WorldCellPacked::from_f32(&self.cells[g_idx]);
                        }
                    }
                }
                chunks.push(WorldChunk { chunk_x: cx as u16, chunk_z: cz as u16, cells: packed_cells });
            }
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_chunk_encode_decode_roundtrip() {
        let chunk = WorldChunk {
            chunk_x: 2, chunk_z: 3,
            cells: [WorldCellPacked { channel_a: 128, channel_b: 50, channel_c: 200, channel_d: 30, channel_e: 180 }; 256],
        };
        let encoded = chunk.encode();
        let decoded = WorldChunk::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.chunk_x, 2);
        assert_eq!(decoded.chunk_z, 3);
        assert_eq!(decoded.cells[0].channel_a, 128);
        assert_eq!(decoded.cells[127].channel_e, 180);
    }

    #[test]
    fn test_world_chunk_quantization() {
        let original = 0.73f32;
        let quantized = (original * 255.0) as u8;
        let restored = quantized as f32 / 255.0;
        assert!((restored - original).abs() < 1.0 / 255.0 + f32::EPSILON);
    }

    #[test]
    fn test_world_grid_to_chunks() {
        let cells = vec![WorldCellF32 { channel_a: 0.5, channel_b: 0.1, channel_c: 0.8, channel_d: 0.3, channel_e: 0.4 }; 32 * 32];
        let grid = WorldChunkGridNet { cells, width: 32, height: 32 };
        let chunks = grid.to_chunks(16);
        assert_eq!(chunks.len(), 4, "32×32 / 16×16 = 4 chunks");
        assert_eq!(chunks[0].cells.len(), 256);
    }
}
