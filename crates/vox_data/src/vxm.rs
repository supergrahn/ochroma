use std::io::{Read, Write};

use bytemuck::{Pod, Zeroable, bytes_of, cast_slice, try_cast_slice};
use thiserror::Error;
use uuid::Uuid;
use vox_core::types::GaussianSplat;

const MAGIC: &[u8; 4] = b"VXMF";
const VERSION: u16 = 1;

#[derive(Debug, Error)]
pub enum VxmError {
    #[error("invalid magic bytes")]
    InvalidMagic,
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u16),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("decompression error: {0}")]
    Decompress(String),
    #[error("compression error: {0}")]
    Compress(String),
    #[error("invalid alignment")]
    InvalidAlignment,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialType {
    Generic = 0,
    Concrete = 1,
    Glass = 2,
    Vegetation = 3,
    Metal = 4,
    Water = 5,
}

impl From<u8> for MaterialType {
    fn from(v: u8) -> Self {
        match v {
            1 => MaterialType::Concrete,
            2 => MaterialType::Glass,
            3 => MaterialType::Vegetation,
            4 => MaterialType::Metal,
            5 => MaterialType::Water,
            _ => MaterialType::Generic,
        }
    }
}

/// 64-byte header for .vxm v0.1 files.
///
/// Layout: magic(4) + version(2) + flags(2) + asset_uuid(16) + splat_count(4) + material_type(1) + _pad0(3) + _pad1(32) = 64
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct VxmHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub flags: u16,
    pub asset_uuid: [u8; 16],
    pub splat_count: u32,
    pub material_type: u8,
    pub _pad0: [u8; 3],
    pub _pad1: [u8; 32],
}

const _: () = assert!(std::mem::size_of::<VxmHeader>() == 64);

impl VxmHeader {
    pub fn new(uuid: Uuid, splat_count: u32, material_type: MaterialType) -> Self {
        let mut header = VxmHeader::zeroed();
        header.magic = *MAGIC;
        header.version = VERSION;
        header.flags = 0;
        header.asset_uuid = *uuid.as_bytes();
        header.splat_count = splat_count;
        header.material_type = material_type as u8;
        header
    }

    pub fn uuid(&self) -> Uuid {
        Uuid::from_bytes(self.asset_uuid)
    }
}

pub struct VxmFile {
    pub header: VxmHeader,
    pub splats: Vec<GaussianSplat>,
}

impl VxmFile {
    pub fn write<W: Write>(&self, mut writer: W) -> Result<(), VxmError> {
        // Write 64-byte header
        writer.write_all(bytes_of(&self.header))?;

        // Compress splat data
        let splat_bytes: &[u8] = cast_slice(&self.splats);
        let compressed = zstd::encode_all(splat_bytes, 0)
            .map_err(|e| VxmError::Compress(e.to_string()))?;

        // Write compressed size as u64 le
        let compressed_size = compressed.len() as u64;
        writer.write_all(&compressed_size.to_le_bytes())?;

        // Write compressed data
        writer.write_all(&compressed)?;

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> Result<Self, VxmError> {
        // Read 64-byte header
        let mut header_bytes = [0u8; 64];
        reader.read_exact(&mut header_bytes)?;

        let header: VxmHeader = *bytemuck::from_bytes(&header_bytes);

        // Validate magic
        if &header.magic != MAGIC {
            return Err(VxmError::InvalidMagic);
        }

        // Validate version
        if header.version != VERSION {
            return Err(VxmError::UnsupportedVersion(header.version));
        }

        // Read compressed size
        let mut size_bytes = [0u8; 8];
        reader.read_exact(&mut size_bytes)?;
        let compressed_size = u64::from_le_bytes(size_bytes) as usize;

        // Read compressed data
        let mut compressed = vec![0u8; compressed_size];
        reader.read_exact(&mut compressed)?;

        // Decompress
        let decompressed = zstd::decode_all(&compressed[..])
            .map_err(|e| VxmError::Decompress(e.to_string()))?;

        // Cast bytes to splats
        let splats: Vec<GaussianSplat> = try_cast_slice::<u8, GaussianSplat>(&decompressed)
            .map_err(|_| VxmError::InvalidAlignment)?
            .to_vec();

        Ok(VxmFile { header, splats })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_is_64_bytes() {
        assert_eq!(std::mem::size_of::<VxmHeader>(), 64);
    }

    #[test]
    fn round_trip_write_read() {
        let uuid = Uuid::new_v4();
        let splat = GaussianSplat {
            position: [1.0, 2.0, 3.0],
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity: 200,
            _pad: [0; 3],
            spectral: [0; 8],
        };
        let file = VxmFile {
            header: VxmHeader::new(uuid, 1, MaterialType::Concrete),
            splats: vec![splat],
        };

        let mut buf = Vec::new();
        file.write(&mut buf).expect("write should succeed");

        let read_back = VxmFile::read(&buf[..]).expect("read should succeed");
        assert_eq!(read_back.header.splat_count, 1);
        assert_eq!(read_back.header.uuid(), uuid);
        assert_eq!(read_back.splats.len(), 1);
        assert_eq!(read_back.splats[0].position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn invalid_magic_rejected() {
        let bad_data = vec![0u8; 128];
        let result = VxmFile::read(&bad_data[..]);
        assert!(matches!(result, Err(VxmError::InvalidMagic)));
    }

    #[test]
    fn material_type_from_u8() {
        assert_eq!(MaterialType::from(1), MaterialType::Concrete);
        assert_eq!(MaterialType::from(2), MaterialType::Glass);
        assert_eq!(MaterialType::from(99), MaterialType::Generic);
    }
}
