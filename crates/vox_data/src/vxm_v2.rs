use std::io::{Read, Write};

use bytemuck::{Pod, Zeroable, bytes_of, cast_slice, try_cast_slice};
use uuid::Uuid;

use crate::vxm::VxmError;

const MAGIC: &[u8; 4] = b"VXMF";
const VERSION_V2: u16 = 2;

/// A single Gaussian splat for .vxm v0.2 (52 bytes).
///
/// Layout: position [f32;3](12) + scale [f32;3](12) + rotation [i16;4](8)
///       + opacity u8(1) + semantic_zone u8(1) + entity_id u16(2)
///       + spectral [u16;8](16) = 52 bytes
///
/// Replaces the original GaussianSplat's `_pad: [u8;3]` with
/// `semantic_zone: u8` + `entity_id: u16`, keeping total at 52 bytes.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GaussianSplatV2 {
    pub position: [f32; 3],
    pub scale: [f32; 3],
    pub rotation: [i16; 4],
    pub opacity: u8,
    pub semantic_zone: u8,
    pub entity_id: u16,
    pub spectral: [u16; 8],
}

const _: () = assert!(std::mem::size_of::<GaussianSplatV2>() == 52);

/// 64-byte header for .vxm v0.2 files.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct VxmHeaderV2 {
    pub magic: [u8; 4],
    pub version: u16,
    pub flags: u16,
    pub asset_uuid: [u8; 16],
    pub splat_count: u32,
    pub material_type: u8,
    pub _pad0: [u8; 3],
    pub _pad1: [u8; 32],
}

const _: () = assert!(std::mem::size_of::<VxmHeaderV2>() == 64);

impl VxmHeaderV2 {
    pub fn new(uuid: Uuid, splat_count: u32) -> Self {
        let mut header = VxmHeaderV2::zeroed();
        header.magic = *MAGIC;
        header.version = VERSION_V2;
        header.flags = 0;
        header.asset_uuid = *uuid.as_bytes();
        header.splat_count = splat_count;
        header
    }

    pub fn uuid(&self) -> Uuid {
        Uuid::from_bytes(self.asset_uuid)
    }
}

pub struct VxmFileV2 {
    pub header: VxmHeaderV2,
    pub splats: Vec<GaussianSplatV2>,
}

impl VxmFileV2 {
    pub fn write<W: Write>(&self, mut writer: W) -> Result<(), VxmError> {
        writer.write_all(bytes_of(&self.header))?;

        let splat_bytes: &[u8] = cast_slice(&self.splats);
        let compressed = zstd::encode_all(splat_bytes, 0)
            .map_err(|e| VxmError::Compress(e.to_string()))?;

        let compressed_size = compressed.len() as u64;
        writer.write_all(&compressed_size.to_le_bytes())?;
        writer.write_all(&compressed)?;

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> Result<Self, VxmError> {
        let mut header_bytes = [0u8; 64];
        reader.read_exact(&mut header_bytes)?;
        let header: VxmHeaderV2 = *bytemuck::from_bytes(&header_bytes);

        if &header.magic != MAGIC {
            return Err(VxmError::InvalidMagic);
        }
        if header.version != VERSION_V2 {
            return Err(VxmError::UnsupportedVersion(header.version));
        }

        let mut size_bytes = [0u8; 8];
        reader.read_exact(&mut size_bytes)?;
        let compressed_size = u64::from_le_bytes(size_bytes) as usize;

        let mut compressed = vec![0u8; compressed_size];
        reader.read_exact(&mut compressed)?;

        let decompressed = zstd::decode_all(&compressed[..])
            .map_err(|e| VxmError::Decompress(e.to_string()))?;

        let splats: Vec<GaussianSplatV2> =
            try_cast_slice::<u8, GaussianSplatV2>(&decompressed)
                .map_err(|_| VxmError::InvalidAlignment)?
                .to_vec();

        Ok(VxmFileV2 { header, splats })
    }
}
