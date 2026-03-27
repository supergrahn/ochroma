use std::io::Read;
use std::path::Path;
use vox_core::types::GaussianSplat;
use half::f16;

#[derive(Debug)]
pub enum PlyError {
    IoError(std::io::Error),
    ParseError(String),
    UnsupportedFormat(String),
}

impl From<std::io::Error> for PlyError {
    fn from(e: std::io::Error) -> Self { Self::IoError(e) }
}

impl std::fmt::Display for PlyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::ParseError(s) => write!(f, "Parse error: {}", s),
            Self::UnsupportedFormat(s) => write!(f, "Unsupported format: {}", s),
        }
    }
}

/// Properties found in a PLY Gaussian splat file.
#[derive(Debug, Default)]
struct PlyHeader {
    vertex_count: usize,
    is_binary_le: bool,
    properties: Vec<PlyProperty>,
}

#[derive(Debug)]
struct PlyProperty {
    name: String,
    data_type: PlyDataType,
}

#[derive(Debug, Clone, Copy)]
enum PlyDataType {
    Float,
    Double,
    UChar,
    Short,
    Int,
}

impl PlyDataType {
    fn byte_size(&self) -> usize {
        match self { Self::Float => 4, Self::Double => 8, Self::UChar => 1, Self::Short => 2, Self::Int => 4 }
    }
}

fn parse_data_type(s: &str) -> Result<PlyDataType, PlyError> {
    match s {
        "float" | "float32" => Ok(PlyDataType::Float),
        "double" | "float64" => Ok(PlyDataType::Double),
        "uchar" | "uint8" => Ok(PlyDataType::UChar),
        "short" | "int16" => Ok(PlyDataType::Short),
        "int" | "int32" => Ok(PlyDataType::Int),
        _ => Err(PlyError::UnsupportedFormat(format!("Unknown data type: {}", s))),
    }
}

/// Parse PLY header.
fn parse_header(reader: &mut impl Read) -> Result<(PlyHeader, Vec<u8>), PlyError> {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    let mut header_text = String::new();

    // Read until "end_header\n"
    loop {
        reader.read_exact(&mut byte)?;
        buf.push(byte[0]);
        header_text.push(byte[0] as char);
        if header_text.ends_with("end_header\n") {
            break;
        }
        if buf.len() > 100_000 {
            return Err(PlyError::ParseError("Header too large".into()));
        }
    }

    let mut header = PlyHeader::default();

    for line in header_text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() { continue; }

        match parts[0] {
            "format" => {
                if parts.len() >= 2 {
                    header.is_binary_le = parts[1] == "binary_little_endian";
                    if parts[1] == "ascii" {
                        return Err(PlyError::UnsupportedFormat("ASCII PLY not supported, use binary_little_endian".into()));
                    }
                }
            }
            "element" if parts.len() >= 3 && parts[1] == "vertex" => {
                header.vertex_count = parts[2].parse().map_err(|_| PlyError::ParseError("Invalid vertex count".into()))?;
            }
            "property" if parts.len() >= 3 => {
                let dt = parse_data_type(parts[1])?;
                header.properties.push(PlyProperty { name: parts[2].to_string(), data_type: dt });
            }
            _ => {}
        }
    }

    // Read all vertex data
    let vertex_size: usize = header.properties.iter().map(|p| p.data_type.byte_size()).sum();
    let total_bytes = vertex_size * header.vertex_count;
    let mut data = vec![0u8; total_bytes];
    reader.read_exact(&mut data)?;

    Ok((header, data))
}

/// Find property index by name.
fn find_prop(header: &PlyHeader, name: &str) -> Option<usize> {
    header.properties.iter().position(|p| p.name == name)
}

/// Read a float property from vertex data.
fn read_float(data: &[u8], header: &PlyHeader, vertex: usize, prop_idx: usize) -> f32 {
    let vertex_size: usize = header.properties.iter().map(|p| p.data_type.byte_size()).sum();
    let prop_offset: usize = header.properties[..prop_idx].iter().map(|p| p.data_type.byte_size()).sum();
    let offset = vertex * vertex_size + prop_offset;

    match header.properties[prop_idx].data_type {
        PlyDataType::Float => f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]),
        PlyDataType::Double => f64::from_le_bytes(data[offset..offset+8].try_into().unwrap()) as f32,
        PlyDataType::UChar => data[offset] as f32 / 255.0,
        PlyDataType::Short => i16::from_le_bytes([data[offset], data[offset+1]]) as f32 / 32767.0,
        PlyDataType::Int => i32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as f32,
    }
}

/// Sigmoid activation (for logit-space opacity).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Load a .ply Gaussian splat file and convert to Ochroma GaussianSplat format.
pub fn load_ply(path: &Path) -> Result<Vec<GaussianSplat>, PlyError> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    load_ply_from_reader(&mut reader)
}

/// Load from any reader (useful for testing with in-memory data).
pub fn load_ply_from_reader(reader: &mut impl Read) -> Result<Vec<GaussianSplat>, PlyError> {
    let (header, data) = parse_header(reader)?;

    // Find required properties
    let ix = find_prop(&header, "x").ok_or(PlyError::ParseError("Missing 'x' property".into()))?;
    let iy = find_prop(&header, "y").ok_or(PlyError::ParseError("Missing 'y' property".into()))?;
    let iz = find_prop(&header, "z").ok_or(PlyError::ParseError("Missing 'z' property".into()))?;

    // Optional properties with fallbacks
    let i_scale0 = find_prop(&header, "scale_0");
    let i_scale1 = find_prop(&header, "scale_1");
    let i_scale2 = find_prop(&header, "scale_2");
    let i_rot0 = find_prop(&header, "rot_0");
    let i_rot1 = find_prop(&header, "rot_1");
    let i_rot2 = find_prop(&header, "rot_2");
    let i_rot3 = find_prop(&header, "rot_3");
    let i_opacity = find_prop(&header, "opacity");
    let i_fdc0 = find_prop(&header, "f_dc_0");
    let i_fdc1 = find_prop(&header, "f_dc_1");
    let i_fdc2 = find_prop(&header, "f_dc_2");

    let sh_c0: f32 = 0.28209479177; // 1 / (2 * sqrt(pi))

    let mut splats = Vec::with_capacity(header.vertex_count);

    for v in 0..header.vertex_count {
        let x = read_float(&data, &header, v, ix);
        let y = read_float(&data, &header, v, iy);
        let z = read_float(&data, &header, v, iz);

        // Scales (log-space in PLY)
        let sx = i_scale0.map(|i| read_float(&data, &header, v, i).exp()).unwrap_or(0.01);
        let sy = i_scale1.map(|i| read_float(&data, &header, v, i).exp()).unwrap_or(0.01);
        let sz = i_scale2.map(|i| read_float(&data, &header, v, i).exp()).unwrap_or(0.01);

        // Rotation (quaternion w,x,y,z in PLY -> x,y,z,w as i16 in Ochroma)
        let rw = i_rot0.map(|i| read_float(&data, &header, v, i)).unwrap_or(1.0);
        let rx = i_rot1.map(|i| read_float(&data, &header, v, i)).unwrap_or(0.0);
        let ry = i_rot2.map(|i| read_float(&data, &header, v, i)).unwrap_or(0.0);
        let rz = i_rot3.map(|i| read_float(&data, &header, v, i)).unwrap_or(0.0);

        // Normalize quaternion
        let len = (rw*rw + rx*rx + ry*ry + rz*rz).sqrt().max(1e-8);
        let rotation = [
            (rx / len * 32767.0) as i16,
            (ry / len * 32767.0) as i16,
            (rz / len * 32767.0) as i16,
            (rw / len * 32767.0) as i16,
        ];

        // Opacity (logit-space in PLY)
        let opacity_raw = i_opacity.map(|i| read_float(&data, &header, v, i)).unwrap_or(0.0);
        let opacity = (sigmoid(opacity_raw) * 255.0) as u8;

        // Colour: SH DC -> linear RGB -> approximate spectral
        let r = i_fdc0.map(|i| (0.5 + sh_c0 * read_float(&data, &header, v, i)).clamp(0.0, 1.0)).unwrap_or(0.5);
        let g = i_fdc1.map(|i| (0.5 + sh_c0 * read_float(&data, &header, v, i)).clamp(0.0, 1.0)).unwrap_or(0.5);
        let b = i_fdc2.map(|i| (0.5 + sh_c0 * read_float(&data, &header, v, i)).clamp(0.0, 1.0)).unwrap_or(0.5);

        // Convert RGB to approximate spectral bands
        let spectral = rgb_to_approximate_spectral(r, g, b);

        splats.push(GaussianSplat {
            position: [x, y, z],
            scale: [sx, sy, sz],
            rotation,
            opacity,
            _pad: [0; 3],
            spectral,
        });
    }

    Ok(splats)
}

/// Convert linear RGB to approximate 8-band spectral reflectance.
/// Uses a simple primary decomposition: R peaks at 620nm, G at 540nm, B at 460nm.
fn rgb_to_approximate_spectral(r: f32, g: f32, b: f32) -> [u16; 8] {
    // Band centres: 380, 420, 460, 500, 540, 580, 620, 660nm
    let bands = [
        b * 0.3,                          // 380nm -- blue tail
        b * 0.7,                          // 420nm -- blue
        b * 1.0,                          // 460nm -- blue peak
        g * 0.4 + b * 0.2,               // 500nm -- cyan
        g * 1.0,                          // 540nm -- green peak
        r * 0.4 + g * 0.3,               // 580nm -- yellow
        r * 1.0,                          // 620nm -- red peak
        r * 0.6,                          // 660nm -- red tail
    ];

    std::array::from_fn(|i| f16::from_f32(bands[i].clamp(0.0, 1.0)).to_bits())
}

/// Create a minimal binary PLY file in memory (for testing).
pub fn create_test_ply(positions: &[[f32; 3]]) -> Vec<u8> {
    let mut data = Vec::new();
    let header = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nproperty float opacity\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nend_header\n",
        positions.len()
    );
    data.extend_from_slice(header.as_bytes());

    for pos in positions {
        // x, y, z
        data.extend_from_slice(&pos[0].to_le_bytes());
        data.extend_from_slice(&pos[1].to_le_bytes());
        data.extend_from_slice(&pos[2].to_le_bytes());
        // scale_0, scale_1, scale_2 (log-space, ln(0.01) ~ -4.6)
        data.extend_from_slice(&(-4.6f32).to_le_bytes());
        data.extend_from_slice(&(-4.6f32).to_le_bytes());
        data.extend_from_slice(&(-4.6f32).to_le_bytes());
        // rot_0 (w), rot_1 (x), rot_2 (y), rot_3 (z) -- identity
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        // opacity (logit space, sigmoid(2.0) ~ 0.88)
        data.extend_from_slice(&2.0f32.to_le_bytes());
        // f_dc_0, f_dc_1, f_dc_2 (SH DC, will map to ~neutral grey)
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    data
}
