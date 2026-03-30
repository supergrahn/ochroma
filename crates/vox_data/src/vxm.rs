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

const VERSION_V3: u16 = 3;
/// flags bit: material_ids section present
const FLAG_MATERIAL_IDS: u16 = 0x0001;

/// VXM v3: splats + optional material_ids + spectral_level.
pub struct VxmFileV3 {
    pub splats: Vec<GaussianSplat>,
    /// Per-splat material ID (0 = unassigned, 1-8 = SpectralMaterialDb index).
    /// May be empty (len=0 means no material data).
    pub material_ids: Vec<u16>,
    /// 1 = Smits uplift, 2 = capture-approximate, 3 = measured from 3-photo.
    pub spectral_level: u8,
}

impl VxmFileV3 {
    pub fn write<W: Write>(&self, mut w: W) -> Result<(), VxmError> {
        let has_mats = !self.material_ids.is_empty();
        let flags: u16 = if has_mats { FLAG_MATERIAL_IDS } else { 0 };

        let mut hdr = VxmHeader::zeroed();
        hdr.magic = *MAGIC;
        hdr.version = VERSION_V3;
        hdr.flags = flags;
        hdr.splat_count = self.splats.len() as u32;
        hdr._pad0[0] = self.spectral_level;
        w.write_all(bytemuck::bytes_of(&hdr))?;

        // Compressed splat block
        let splat_bytes: &[u8] = bytemuck::cast_slice(&self.splats);
        let compressed = zstd::encode_all(splat_bytes, 0)
            .map_err(|e| VxmError::Compress(e.to_string()))?;
        w.write_all(&(compressed.len() as u64).to_le_bytes())?;
        w.write_all(&compressed)?;

        // Optional material_ids section
        if has_mats {
            let ids_bytes: &[u8] = bytemuck::cast_slice(&self.material_ids);
            let ids_compressed = zstd::encode_all(ids_bytes, 0)
                .map_err(|e| VxmError::Compress(e.to_string()))?;
            w.write_all(&(self.material_ids.len() as u32).to_le_bytes())?;
            w.write_all(&(ids_compressed.len() as u64).to_le_bytes())?;
            w.write_all(&ids_compressed)?;
        }

        Ok(())
    }

    pub fn read<R: Read>(mut r: R) -> Result<Self, VxmError> {
        let mut hdr_bytes = [0u8; 64];
        r.read_exact(&mut hdr_bytes)?;
        let hdr: VxmHeader = *bytemuck::from_bytes(&hdr_bytes);

        if &hdr.magic != MAGIC {
            return Err(VxmError::InvalidMagic);
        }
        if hdr.version != VERSION_V3 {
            return Err(VxmError::UnsupportedVersion(hdr.version));
        }

        let spectral_level = hdr._pad0[0];

        // Read compressed splat block
        let mut size_bytes = [0u8; 8];
        r.read_exact(&mut size_bytes)?;
        let compressed_size = u64::from_le_bytes(size_bytes) as usize;
        let mut compressed = vec![0u8; compressed_size];
        r.read_exact(&mut compressed)?;
        let decompressed = zstd::decode_all(&compressed[..])
            .map_err(|e| VxmError::Decompress(e.to_string()))?;
        let splats: Vec<GaussianSplat> = bytemuck::cast_slice(&decompressed).to_vec();

        // Optional material_ids section
        let mut material_ids = Vec::new();
        if hdr.flags & FLAG_MATERIAL_IDS != 0 {
            let mut count_bytes = [0u8; 4];
            r.read_exact(&mut count_bytes)?;
            let count = u32::from_le_bytes(count_bytes) as usize;
            let mut ids_size_bytes = [0u8; 8];
            r.read_exact(&mut ids_size_bytes)?;
            let ids_compressed_size = u64::from_le_bytes(ids_size_bytes) as usize;
            let mut ids_compressed = vec![0u8; ids_compressed_size];
            r.read_exact(&mut ids_compressed)?;
            let ids_bytes = zstd::decode_all(&ids_compressed[..])
                .map_err(|e| VxmError::Decompress(e.to_string()))?;
            let ids_slice: &[u16] = bytemuck::cast_slice(&ids_bytes);
            material_ids = ids_slice[..count].to_vec();
        }

        Ok(Self { splats, material_ids, spectral_level })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    #[test]
    fn header_size_is_64_bytes() {
        assert_eq!(std::mem::size_of::<VxmHeader>(), 64);
    }

    #[test]
    fn round_trip_write_read() {
        let uuid = Uuid::new_v4();
        let splat = GaussianSplat::volume(
            [1.0, 2.0, 3.0],
            [0.1, 0.1, 0.1],
            glam::Quat::IDENTITY,
            200,
            [0u16; 16],
        );
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
        assert_eq!(read_back.splats[0].position(), [1.0, 2.0, 3.0]);
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

    mod v3_tests {
        use super::*;

        fn make_splat(pos: [f32; 3]) -> GaussianSplat {
            GaussianSplat::surface(
                pos,
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                0.01, 0.01,
                200,
                [0u16; GaussianSplat::BANDS],
            )
        }

        #[test]
        fn vxm_v3_header_still_64_bytes() {
            assert_eq!(std::mem::size_of::<VxmHeader>(), 64);
        }

        #[test]
        fn material_ids_roundtrip() {
            let splats = vec![make_splat([0.0, 0.0, 0.0]), make_splat([1.0, 0.0, 0.0])];
            let material_ids = vec![3u16, 7u16];

            let mut buf = Vec::new();
            let file = VxmFileV3 {
                splats: splats.clone(),
                material_ids: material_ids.clone(),
                spectral_level: 1,
            };
            file.write(&mut buf).unwrap();

            let loaded = VxmFileV3::read(std::io::Cursor::new(&buf)).unwrap();
            println!("loaded.material_ids = {:?}, expected {:?}", loaded.material_ids, material_ids);
            assert_eq!(loaded.splats.len(), 2);
            assert_eq!(loaded.material_ids, material_ids,
                "loaded.material_ids = {:?}, expected {:?}", loaded.material_ids, material_ids);
            assert_eq!(loaded.spectral_level, 1);
        }

        #[test]
        fn empty_material_ids_roundtrip() {
            let splats = vec![make_splat([0.0, 1.0, 0.0])];
            let file = VxmFileV3 { splats, material_ids: vec![], spectral_level: 2 };
            let mut buf = Vec::new();
            file.write(&mut buf).unwrap();
            let loaded = VxmFileV3::read(std::io::Cursor::new(&buf)).unwrap();
            assert!(loaded.material_ids.is_empty());
        }
    }
}
