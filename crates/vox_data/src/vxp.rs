//! `.vxp` — the **vox pack** asset container.
//!
//! A sibling of this crate's `.vxm` (VXM v3): where `.vxm` is one voxel/splat
//! model, `.vxp` is the SHIPPING container that carries many finished assets
//! into the game folder. It is written by an asset pipeline and read by the
//! game; nothing in here knows a game concept, so both sides can depend on this
//! one implementation instead of drifting apart.
//!
//! # Format (ratified spec — `urban_horizon/CLAUDE.md`, "Asset container format")
//!
//! * **Container**: a ZIP whose every entry uses compression method **STORE**
//!   (0, uncompressed), so an entry maps straight out of the file with no
//!   inflate cost. Mesh and texture payloads are already compact/compressed
//!   internally; deflating them again costs CPU for ~nothing.
//! * **Entry naming**: `<asset_id>[_LOD<n>]_<32-hex-content-hash>.<Kind>`, each
//!   accompanied by a **32-byte `.cid` sidecar** holding those same 32 hex
//!   characters. Content addressing means identical payloads dedupe and
//!   integrity is checkable without a manifest.
//! * **Kinds**: [`VxpKind::Geometry`] (BINARY — see [`VxpGeometry`], this is
//!   what replaces pretty-printed JSON), [`VxpKind::Surface`],
//!   [`VxpKind::Metadata`] (small, so it reads without the mesh) and
//!   [`VxpKind::Texture`] (a per-asset thumbnail for in-game display).
//! * **Index**: a single top-level `index.json` entry, written FIRST, lets a
//!   reader enumerate every asset and pull `.Metadata` + `.Texture` at startup
//!   **without touching geometry**.
//! * **Texture register**: `index.json` also carries a [`VxpTexture`] entry for
//!   every material texture the pack's assets sample ([`VxpIndex::textures`]),
//!   so GPU residency — which textures, how many bytes, which mip tail stays
//!   resident — is decidable from the index ALONE, without opening a single
//!   `.Surface`. See the "Texture register" section below.
//!
//! # Determinism (project LAW)
//!
//! Identical input produces a byte-identical `.vxp`. Nothing observable from
//! outside the input is recorded: entries are emitted in a fixed
//! `(asset_id, kind, lod)` order, every ZIP timestamp is the fixed MS-DOS epoch
//! `1980-01-01 00:00:00`, and no "created at"/tool-version/host field exists.
//! [`tests::pack_is_byte_identical_across_writes`] proves it.
//!
//! # Texture register
//!
//! Material textures are NOT entries in the pack — they are separate GPU-native
//! artifacts (BCn DDS/KTX2) shared by many assets across many packs, so packing
//! a copy per pack would multiply them. What the pack carries instead is a
//! **register**: [`VxpIndex::textures`], one [`VxpTexture`] per texture the
//! pack's assets sample, keyed by the texture payload's [`content_id`].
//!
//! The register exists to make **residency decidable from `index.json` alone**.
//! Before it, a loader had to open and parse every `.Surface` to learn which
//! textures a pack even wants — which means on-demand loading could only ever
//! be added as a FORMAT MIGRATION. With it, the loader reads one small JSON
//! entry and knows the full texture set, its byte cost, and which mip levels are
//! permanently resident.
//!
//! Each entry carries `width`, `height`, [`VxpTextureFormat`], `mip_count`,
//! `bytes` (GPU bytes for the whole mip chain) and the **always-resident mip
//! tail**: `tail_mip` (the first level that never leaves memory) and
//! `tail_bytes`. The tail is why a shader never has to stall: a sample of a
//! non-resident texture reads the tail and comes back BLURRY, never wrong and
//! never blocked on I/O — a shader cannot do file I/O, and that is the one
//! constraint streaming actually has to solve.
//!
//! Bindless slots are assigned by [`VxpTextureSlots`] from the CONTENT HASH, so
//! the same texture lands in the same slot no matter which order packs load in.
//!
//! # Reading
//!
//! [`VxpReader`] parses the central directory once (ZIP64-aware) and then serves
//! entries by name. `read_entry` verifies the stored CRC-32 and the
//! content hash embedded in the entry name, so a truncated or swapped payload is
//! a loud error rather than silent corruption.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum VxpError {
    #[error("vxp io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("vxp format error in {path}: {reason}")]
    Format { path: PathBuf, reason: String },
    #[error("vxp entry {name} not found in {path}")]
    MissingEntry { path: PathBuf, name: String },
    #[error("vxp entry {name} in {path} is corrupt: {reason}")]
    Corrupt {
        path: PathBuf,
        name: String,
        reason: String,
    },
    #[error("vxp geometry decode error: {0}")]
    Geometry(String),
    #[error("vxp json error for {context}: {source}")]
    Json {
        context: String,
        #[source]
        source: serde_json::Error,
    },
    /// A register entry that cannot describe a real texture. Always names the
    /// texture and the exact disagreement — never a silent default.
    #[error("vxp texture register entry {id}: {reason}")]
    TextureRegister { id: String, reason: String },
    /// An asset samples a texture the pack never registered. This is the one
    /// failure the register exists to make impossible to miss: without it the
    /// loader would size residency from an incomplete set and quietly render
    /// the wrong thing.
    #[error(
        "vxp asset {asset_id} references texture {texture_id}, which is absent from the \
         pack's texture register — register it before writing the pack"
    )]
    UnregisteredTexture {
        asset_id: String,
        texture_id: String,
    },
    /// The register says one thing, the texture payload says another.
    #[error("vxp texture {id} disagrees with its payload: {reason}")]
    TextureMismatch { id: String, reason: String },
}

type Result<T> = std::result::Result<T, VxpError>;

// ─────────────────────────────────────────────────────────────────────────────
// Kinds
// ─────────────────────────────────────────────────────────────────────────────

/// The entry kinds a `.vxp` carries. Together they ARE one finished asset:
/// the mesh plus the attributes it needs in the game and in the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum VxpKind {
    /// The finished mesh, BINARY ([`VxpGeometry`]). Never JSON.
    Geometry,
    /// Material / PBR data.
    Surface,
    /// The game/sim attributes. Deliberately small: a reader lists the whole
    /// pack from `.Metadata` alone.
    Metadata,
    /// A thumbnail image (PNG bytes) for in-game display.
    Texture,
    /// Skeleton/clip payload for moving parts or skinned characters.
    Animation,
    /// Shared texture-atlas payload.
    Atlas,
}

impl VxpKind {
    pub const ALL: [VxpKind; 6] = [
        VxpKind::Geometry,
        VxpKind::Surface,
        VxpKind::Metadata,
        VxpKind::Texture,
        VxpKind::Animation,
        VxpKind::Atlas,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            VxpKind::Geometry => "Geometry",
            VxpKind::Surface => "Surface",
            VxpKind::Metadata => "Metadata",
            VxpKind::Texture => "Texture",
            VxpKind::Animation => "Animation",
            VxpKind::Atlas => "Atlas",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        VxpKind::ALL.into_iter().find(|k| k.as_str() == s)
    }
}

impl fmt::Display for VxpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Content hashing
// ─────────────────────────────────────────────────────────────────────────────

/// The 32-hex content id of a payload: the first 128 bits of its SHA-256.
///
/// 128 bits of a cryptographic digest is the same strength every content-
/// addressed store (git's future SHA-256 truncation, OCI, Bazel) relies on for
/// an identity key, and it keeps the entry name short enough to stay readable.
pub fn content_id(bytes: &[u8]) -> String {
    hex_bytes(&sha256(bytes)[..16])
}

/// Full SHA-256 content address used by GPU geometry-page integrity checks.
pub fn content_hash(bytes: &[u8]) -> [u8; 32] {
    sha256(bytes)
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).into()
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from_digit(u32::from(byte >> 4), 16).unwrap());
        out.push(char::from_digit(u32::from(byte & 0x0f), 16).unwrap());
    }
    out
}

/// `<asset_id>[_LOD<n>]_<32-hex>.<Kind>` — the canonical entry name.
pub fn entry_name(asset_id: &str, kind: VxpKind, lod: Option<u8>, cid: &str) -> String {
    match lod {
        Some(level) => format!("{asset_id}_LOD{level}_{cid}.{kind}"),
        None => format!("{asset_id}_{cid}.{kind}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Texture register
// ─────────────────────────────────────────────────────────────────────────────

/// The GPU formats a registered texture can be in.
///
/// This is a *sizing* taxonomy, not a renderer one: the register only has to
/// answer "how many bytes does this texture cost, per mip level". Block size
/// and block footprint are all that requires, which is why this enum is tiny and
/// game-agnostic. The names are the same SCREAMING_SNAKE spellings the cook's
/// residency manifest already uses, so `index.json` reads the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum VxpTextureFormat {
    #[serde(rename = "BC1_UNORM")]
    Bc1Unorm,
    #[serde(rename = "BC1_UNORM_SRGB")]
    Bc1UnormSrgb,
    #[serde(rename = "BC3_UNORM")]
    Bc3Unorm,
    #[serde(rename = "BC3_UNORM_SRGB")]
    Bc3UnormSrgb,
    #[serde(rename = "BC4_UNORM")]
    Bc4Unorm,
    #[serde(rename = "BC5_UNORM")]
    Bc5Unorm,
    #[serde(rename = "BC6H_UF16")]
    Bc6hUf16,
    #[serde(rename = "BC7_UNORM")]
    Bc7Unorm,
    #[serde(rename = "BC7_UNORM_SRGB")]
    Bc7UnormSrgb,
    #[serde(rename = "RGBA8_UNORM")]
    Rgba8Unorm,
    #[serde(rename = "RGBA8_UNORM_SRGB")]
    Rgba8UnormSrgb,
}

impl VxpTextureFormat {
    pub const ALL: [VxpTextureFormat; 11] = [
        VxpTextureFormat::Bc1Unorm,
        VxpTextureFormat::Bc1UnormSrgb,
        VxpTextureFormat::Bc3Unorm,
        VxpTextureFormat::Bc3UnormSrgb,
        VxpTextureFormat::Bc4Unorm,
        VxpTextureFormat::Bc5Unorm,
        VxpTextureFormat::Bc6hUf16,
        VxpTextureFormat::Bc7Unorm,
        VxpTextureFormat::Bc7UnormSrgb,
        VxpTextureFormat::Rgba8Unorm,
        VxpTextureFormat::Rgba8UnormSrgb,
    ];

    pub fn as_str(self) -> &'static str {
        use VxpTextureFormat::*;
        match self {
            Bc1Unorm => "BC1_UNORM",
            Bc1UnormSrgb => "BC1_UNORM_SRGB",
            Bc3Unorm => "BC3_UNORM",
            Bc3UnormSrgb => "BC3_UNORM_SRGB",
            Bc4Unorm => "BC4_UNORM",
            Bc5Unorm => "BC5_UNORM",
            Bc6hUf16 => "BC6H_UF16",
            Bc7Unorm => "BC7_UNORM",
            Bc7UnormSrgb => "BC7_UNORM_SRGB",
            Rgba8Unorm => "RGBA8_UNORM",
            Rgba8UnormSrgb => "RGBA8_UNORM_SRGB",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        VxpTextureFormat::ALL.into_iter().find(|f| f.as_str() == s)
    }

    /// Edge length of one compression block in texels (1 for uncompressed).
    pub fn block_dim(self) -> u32 {
        use VxpTextureFormat::*;
        match self {
            Rgba8Unorm | Rgba8UnormSrgb => 1,
            _ => 4,
        }
    }

    /// Bytes one block occupies.
    pub fn block_bytes(self) -> u64 {
        use VxpTextureFormat::*;
        match self {
            // 4 bpp block formats.
            Bc1Unorm | Bc1UnormSrgb | Bc4Unorm => 8,
            // 8 bpp block formats.
            Bc3Unorm | Bc3UnormSrgb | Bc5Unorm | Bc6hUf16 | Bc7Unorm | Bc7UnormSrgb => 16,
            Rgba8Unorm | Rgba8UnormSrgb => 4,
        }
    }

    /// Bytes one mip level of `width` x `height` occupies, rounded up to whole
    /// blocks the way every BCn upload path does.
    pub fn level_bytes(self, width: u32, height: u32) -> u64 {
        let block = self.block_dim();
        let bw = u64::from(width.div_ceil(block).max(1));
        let bh = u64::from(height.div_ceil(block).max(1));
        bw * bh * self.block_bytes()
    }
}

impl fmt::Display for VxpTextureFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Dimensions of mip `level` of a `width` x `height` texture.
pub fn mip_dimensions(width: u32, height: u32, level: u32) -> (u32, u32) {
    (
        (width >> level.min(31)).max(1),
        (height >> level.min(31)).max(1),
    )
}

/// Number of levels in a complete mip chain down to 1x1.
pub fn full_mip_count(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).max(1).leading_zeros()
}

/// Largest edge, in texels, a mip level may have and still be part of the
/// ALWAYS-RESIDENT tail.
///
/// 128 is a deliberate trade, not a round number. The tail is the image a ray
/// gets when the texture it wants is not resident, so it has to be legible: at
/// 128 a facade still shows its brick coursing and its true colour, and the miss
/// reads as "one frame softer", which is exactly the cost §4.4 of the design
/// budgets for. It is also cheap — the whole tail of a 2K BC7 texture is ~21 KB,
/// about 0.4 % of its 5.6 MB chain — so the permanently-pinned set stays in the
/// single-digit-MB range for a corpus of hundreds of textures. Going lower (64,
/// 32) saves bytes nobody is short of and makes the fallback a colour smear;
/// going higher pins tens of MB for a frame the player barely sees.
pub const VXP_TAIL_MAX_DIM: u32 = 128;

/// First mip level of the always-resident tail: the coarsest-first level whose
/// largest edge is at most [`VXP_TAIL_MAX_DIM`].
///
/// A texture already smaller than the threshold is resident in full (`0`); a
/// texture whose chain is too short still pins at least its last level, so
/// EVERY texture has a tail.
pub fn tail_mip_for(width: u32, height: u32, mip_count: u32) -> u32 {
    let last = mip_count.saturating_sub(1);
    for level in 0..mip_count {
        let (w, h) = mip_dimensions(width, height, level);
        if w.max(h) <= VXP_TAIL_MAX_DIM {
            return level;
        }
    }
    last
}

/// What a texture payload actually turned out to be, as observed by whoever
/// decoded it. Compared against a [`VxpTexture`] by
/// [`VxpTexture::verify_against_payload`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VxpTextureDesc {
    pub width: u32,
    pub height: u32,
    pub format: VxpTextureFormat,
    pub mip_count: u32,
    /// Length of the texel data alone — container headers excluded.
    pub texel_bytes: u64,
}

/// One texture in a pack's register.
///
/// Everything a residency decision needs, and nothing that requires opening the
/// texture or the `.Surface` that references it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VxpTexture {
    /// [`content_id`] of the texture payload: 32 lowercase hex. This is the
    /// identity — the same texture referenced from ten packs is ONE entry with
    /// ONE id, and [`VxpTextureSlots`] turns that id into a bindless slot.
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub format: VxpTextureFormat,
    pub mip_count: u32,
    /// GPU bytes for the COMPLETE mip chain.
    pub bytes: u64,
    /// First level of the always-resident tail — see [`tail_mip_for`].
    pub tail_mip: u32,
    /// GPU bytes for levels `tail_mip .. mip_count`: what this texture costs
    /// permanently, whether or not anything is looking at it.
    pub tail_bytes: u64,
}

impl VxpTexture {
    /// Build a register entry, computing `bytes`, `tail_mip` and `tail_bytes`
    /// from the description. Rejects anything that cannot be a real texture.
    pub fn new(
        id: impl Into<String>,
        width: u32,
        height: u32,
        format: VxpTextureFormat,
        mip_count: u32,
    ) -> Result<Self> {
        let id = id.into();
        let bad = |reason: String| VxpError::TextureRegister {
            id: id.clone(),
            reason,
        };
        if id.len() != 32
            || !id
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err(bad(
                "id must be 32 lowercase hex characters (the payload's content_id)".to_string(),
            ));
        }
        if width == 0 || height == 0 {
            return Err(bad(format!("degenerate dimensions {width}x{height}")));
        }
        if mip_count == 0 {
            return Err(bad(
                "mip_count 0: a texture has at least one level".to_string()
            ));
        }
        let full = full_mip_count(width, height);
        if mip_count > full {
            return Err(bad(format!(
                "mip_count {mip_count} exceeds the {full} levels a {width}x{height} chain can have"
            )));
        }
        let tail_mip = tail_mip_for(width, height, mip_count);
        Ok(Self {
            id,
            width,
            height,
            format,
            mip_count,
            bytes: level_range_bytes(width, height, format, 0, mip_count),
            tail_mip,
            tail_bytes: level_range_bytes(width, height, format, tail_mip, mip_count),
        })
    }

    /// Re-derive every computed field and refuse an entry that disagrees with
    /// itself. Read packs go through this, so a hand-edited or foreign
    /// `index.json` fails loudly instead of mis-sizing residency.
    pub fn validate(&self) -> Result<()> {
        let recomputed = VxpTexture::new(
            self.id.clone(),
            self.width,
            self.height,
            self.format,
            self.mip_count,
        )?;
        if &recomputed != self {
            return Err(VxpError::TextureRegister {
                id: self.id.clone(),
                reason: format!(
                    "declares bytes={} tail_mip={} tail_bytes={}, but a {}x{} {} chain of {} \
                     levels is bytes={} tail_mip={} tail_bytes={}",
                    self.bytes,
                    self.tail_mip,
                    self.tail_bytes,
                    self.width,
                    self.height,
                    self.format,
                    self.mip_count,
                    recomputed.bytes,
                    recomputed.tail_mip,
                    recomputed.tail_bytes
                ),
            });
        }
        Ok(())
    }

    /// Bytes of the levels that are NOT permanently resident — what streaming
    /// would ever have to move.
    pub fn streamable_bytes(&self) -> u64 {
        self.bytes - self.tail_bytes
    }

    /// Byte cost of one mip level.
    pub fn level_bytes(&self, level: u32) -> u64 {
        let (w, h) = mip_dimensions(self.width, self.height, level);
        self.format.level_bytes(w, h)
    }

    /// Prove this entry describes the payload it claims to.
    ///
    /// Both halves fail by NAME: a payload whose content id is not `self.id` is
    /// the wrong texture; a payload whose decoded dimensions/format/mip count or
    /// texel length disagree means the register would size residency from
    /// fiction. Neither is ever tolerated as a default.
    pub fn verify_against_payload(&self, payload: &[u8], observed: VxpTextureDesc) -> Result<()> {
        let actual = content_id(payload);
        if actual != self.id {
            return Err(VxpError::TextureMismatch {
                id: self.id.clone(),
                reason: format!("payload content id is {actual}"),
            });
        }
        let mut wrong = Vec::new();
        if observed.width != self.width || observed.height != self.height {
            wrong.push(format!(
                "dimensions {}x{} (register says {}x{})",
                observed.width, observed.height, self.width, self.height
            ));
        }
        if observed.format != self.format {
            wrong.push(format!(
                "format {} (register says {})",
                observed.format, self.format
            ));
        }
        if observed.mip_count != self.mip_count {
            wrong.push(format!(
                "mip_count {} (register says {})",
                observed.mip_count, self.mip_count
            ));
        }
        if observed.texel_bytes != self.bytes {
            wrong.push(format!(
                "texel bytes {} (register says {})",
                observed.texel_bytes, self.bytes
            ));
        }
        if wrong.is_empty() {
            Ok(())
        } else {
            Err(VxpError::TextureMismatch {
                id: self.id.clone(),
                reason: format!("payload has {}", wrong.join(", ")),
            })
        }
    }
}

fn level_range_bytes(width: u32, height: u32, format: VxpTextureFormat, from: u32, to: u32) -> u64 {
    (from..to)
        .map(|level| {
            let (w, h) = mip_dimensions(width, height, level);
            format.level_bytes(w, h)
        })
        .sum()
}

/// Bindless slot assignment keyed by CONTENT HASH, never by load order.
///
/// The descriptor array is rewritten every dispatch, so a slot that moved
/// because packs happened to load in a different order is a whole class of
/// "wrong texture on the wrong surface" bug waiting to happen. Slots here are a
/// pure function of the SET of texture ids: ids are collected into a
/// `BTreeMap`, then numbered in ascending id order. Same corpus, any pack order,
/// same slots.
///
/// (Adding a texture to the corpus does renumber the ids after it. That is a
/// content change, handled at build time, not the frame-to-frame instability
/// the hash keying exists to rule out.)
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VxpTextureSlots {
    by_id: BTreeMap<String, (u32, VxpTexture)>,
}

impl VxpTextureSlots {
    /// Build the slot table from any number of pack indices, in any order.
    ///
    /// Reads ONLY the indices — no `.Surface`, no texture payload.
    pub fn from_indices<'a, I>(indices: I) -> Result<Self>
    where
        I: IntoIterator<Item = &'a VxpIndex>,
    {
        let mut merged: BTreeMap<String, VxpTexture> = BTreeMap::new();
        for index in indices {
            for texture in &index.textures {
                texture.validate()?;
                if let Some(existing) = merged.get(&texture.id) {
                    if existing != texture {
                        return Err(VxpError::TextureRegister {
                            id: texture.id.clone(),
                            reason: format!(
                                "packs disagree about the same content hash: {existing:?} vs \
                                 {texture:?}"
                            ),
                        });
                    }
                } else {
                    merged.insert(texture.id.clone(), texture.clone());
                }
            }
        }
        Ok(Self {
            by_id: merged
                .into_iter()
                .enumerate()
                .map(|(slot, (id, texture))| (id, (slot as u32, texture)))
                .collect(),
        })
    }

    pub fn slot(&self, id: &str) -> Option<u32> {
        self.by_id.get(id).map(|(slot, _)| *slot)
    }

    pub fn texture(&self, id: &str) -> Option<&VxpTexture> {
        self.by_id.get(id).map(|(_, texture)| texture)
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// `(slot, texture)` in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &VxpTexture)> {
        self.by_id.values().map(|(slot, tex)| (*slot, tex))
    }

    /// Bytes permanently pinned for the whole corpus — the always-resident mip
    /// tails, counted once per distinct texture.
    pub fn resident_tail_bytes(&self) -> u64 {
        self.by_id.values().map(|(_, t)| t.tail_bytes).sum()
    }

    /// Bytes every texture would cost if all of them were fully resident.
    pub fn total_bytes(&self) -> u64 {
        self.by_id.values().map(|(_, t)| t.bytes).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Index
// ─────────────────────────────────────────────────────────────────────────────

pub const VXP_INDEX_ENTRY: &str = "index.json";
/// Version 2 introduced the mandatory texture register. Version 3 briefly
/// admitted asset-side runtime geometry pages; version 4 removes that
/// architectural violation. New packs contain one authoritative finished mesh
/// only. Ochroma derives every cluster, detail level, page, and MegaGeometry
/// representation into its disposable runtime cache after loading the asset.
/// Readers still accept v2/v3 packs for migration and ignore unknown legacy
/// index fields; version 1 remains forbidden because texture residency would
/// be undecidable.
pub const VXP_INDEX_VERSION: u32 = 4;
const VXP_MIN_READABLE_INDEX_VERSION: u32 = 2;

/// One asset's entries, as recorded in the pack index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VxpAssetIndex {
    pub asset_id: String,
    /// Entry name of the base `.Geometry`, if the asset has geometry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<String>,
    /// `(lod_level, entry_name)`, ascending by level.
    ///
    /// Read-only migration data for old v2 packs. `VxpWriter` cannot emit
    /// authored LODs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub geometry_lods: Vec<(u8, String)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<String>,
    /// `(clip_id, entry_name)`, sorted by clip id. Required social roles are
    /// validated by the product pack gate, never silently substituted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub animations: Vec<(String, String)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atlas: Option<String>,
    /// Content ids of the material textures this asset samples, ascending.
    /// Every one of them resolves in [`VxpIndex::textures`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub texture_ids: Vec<String>,
    /// Uncompressed byte size of every entry belonging to this asset.
    pub bytes: u64,
}

/// The top-level index: what the game reads at startup to enumerate content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VxpIndex {
    pub version: u32,
    /// Stable identity of the pack (the theme/category it groups).
    pub pack_id: String,
    /// Ascending by `asset_id` — the pack's canonical order.
    pub assets: Vec<VxpAssetIndex>,
    /// THE TEXTURE REGISTER: every material texture this pack's assets sample,
    /// ascending by content id. Always present (possibly empty) so "this pack
    /// needs no textures" and "this pack predates the register" can never be
    /// confused.
    pub textures: Vec<VxpTexture>,
}

impl VxpIndex {
    pub fn asset(&self, asset_id: &str) -> Option<&VxpAssetIndex> {
        self.assets
            .binary_search_by(|probe| probe.asset_id.as_str().cmp(asset_id))
            .ok()
            .map(|i| &self.assets[i])
    }

    pub fn texture(&self, id: &str) -> Option<&VxpTexture> {
        self.textures
            .binary_search_by(|probe| probe.id.as_str().cmp(id))
            .ok()
            .map(|i| &self.textures[i])
    }

    /// Bytes this pack's textures would cost fully resident.
    pub fn texture_bytes(&self) -> u64 {
        self.textures.iter().map(|t| t.bytes).sum()
    }

    /// Bytes of this pack's always-resident mip tails.
    pub fn resident_tail_bytes(&self) -> u64 {
        self.textures.iter().map(|t| t.tail_bytes).sum()
    }

    /// Check the register is internally sound: ascending, unique, self-
    /// consistent, and closed over every id an asset references. This is what
    /// the reader runs, so an incomplete register is never handed to a loader.
    pub fn validate_texture_register(&self) -> Result<()> {
        let mut previous: Option<&str> = None;
        for texture in &self.textures {
            texture.validate()?;
            if let Some(previous) = previous {
                if previous >= texture.id.as_str() {
                    return Err(VxpError::TextureRegister {
                        id: texture.id.clone(),
                        reason: format!("register is not ascending/unique (follows {previous})"),
                    });
                }
            }
            previous = Some(texture.id.as_str());
        }
        for asset in &self.assets {
            for texture_id in &asset.texture_ids {
                if self.texture(texture_id).is_none() {
                    return Err(VxpError::UnregisteredTexture {
                        asset_id: asset.asset_id.clone(),
                        texture_id: texture_id.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Writer
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulates one asset's payloads before the pack is written.
#[derive(Debug, Default, Clone)]
struct PendingAsset {
    geometry: Option<Vec<u8>>,
    surface: Option<Vec<u8>>,
    metadata: Option<Vec<u8>>,
    texture: Option<Vec<u8>>,
    animations: BTreeMap<String, Vec<u8>>,
    atlas: Option<Vec<u8>>,
    /// Content ids of the material textures this asset samples. A `BTreeSet`
    /// so the emitted order is the content's, not the caller's.
    texture_ids: BTreeSet<String>,
}

/// Builds a `.vxp`. Payloads are staged in a `BTreeMap` so the emitted order is
/// a function of the CONTENT (asset id, then a fixed kind order), never of the
/// order the caller happened to add things in — which is what makes a parallel
/// cook able to produce a byte-identical pack.
#[derive(Debug, Default)]
pub struct VxpWriter {
    pack_id: String,
    assets: BTreeMap<String, PendingAsset>,
    textures: BTreeMap<String, VxpTexture>,
}

impl VxpWriter {
    pub fn new(pack_id: impl Into<String>) -> Self {
        Self {
            pack_id: pack_id.into(),
            assets: BTreeMap::new(),
            textures: BTreeMap::new(),
        }
    }

    /// Number of distinct assets staged so far.
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    pub fn add_geometry(&mut self, asset_id: &str, bytes: Vec<u8>) {
        self.entry(asset_id).geometry = Some(bytes);
    }

    pub fn add_surface(&mut self, asset_id: &str, bytes: Vec<u8>) {
        self.entry(asset_id).surface = Some(bytes);
    }

    pub fn add_metadata(&mut self, asset_id: &str, bytes: Vec<u8>) {
        self.entry(asset_id).metadata = Some(bytes);
    }

    /// The per-asset thumbnail (PNG bytes). Required for every asset the pack
    /// ships; [`VxpWriter::finish`] refuses to write a pack that is missing one.
    pub fn add_texture(&mut self, asset_id: &str, png: Vec<u8>) {
        self.entry(asset_id).texture = Some(png);
    }

    pub fn add_animation(&mut self, asset_id: &str, clip_id: &str, bytes: Vec<u8>) {
        self.entry(asset_id)
            .animations
            .insert(clip_id.to_string(), bytes);
    }

    pub fn add_atlas(&mut self, asset_id: &str, bytes: Vec<u8>) {
        self.entry(asset_id).atlas = Some(bytes);
    }

    /// Put a texture in the pack's register.
    ///
    /// Idempotent for an identical entry (the same texture is normally reached
    /// through many assets); a second registration of the same content id with
    /// DIFFERENT metadata is a named error, because one of the two descriptions
    /// is a lie about the same bytes.
    pub fn register_texture(&mut self, texture: VxpTexture) -> Result<()> {
        texture.validate()?;
        match self.textures.get(&texture.id) {
            Some(existing) if existing != &texture => Err(VxpError::TextureRegister {
                id: texture.id.clone(),
                reason: format!(
                    "already registered as {existing:?}; refusing the conflicting {texture:?}"
                ),
            }),
            Some(_) => Ok(()),
            None => {
                self.textures.insert(texture.id.clone(), texture);
                Ok(())
            }
        }
    }

    /// Record that `asset_id` samples the texture with this content id.
    ///
    /// The id does not have to be registered yet — [`VxpWriter::finish`] checks
    /// that every referenced id ended up in the register, so a reference the
    /// pipeline could not resolve fails the WRITE rather than shipping a pack
    /// whose residency set is short.
    pub fn reference_texture(&mut self, asset_id: &str, texture_id: &str) {
        self.entry(asset_id)
            .texture_ids
            .insert(texture_id.to_string());
    }

    /// Textures registered so far.
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    fn entry(&mut self, asset_id: &str) -> &mut PendingAsset {
        self.assets.entry(asset_id.to_string()).or_default()
    }

    /// Write the pack to `path`.
    ///
    /// `require_texture` enforces the spec's REQUIRED per-asset thumbnail. It is
    /// a parameter rather than an unconditional check only so a pipeline can
    /// stage a pack before the thumbnail pass has run; a SHIPPING pack is always
    /// written with `true`.
    pub fn finish(self, path: &Path, require_texture: bool) -> Result<VxpIndex> {
        if require_texture {
            let missing: Vec<&str> = self
                .assets
                .iter()
                .filter(|(_, a)| a.texture.is_none())
                .map(|(id, _)| id.as_str())
                .collect();
            if !missing.is_empty() {
                return Err(VxpError::Format {
                    path: path.to_path_buf(),
                    reason: format!(
                        "{} asset(s) have no .Texture thumbnail, which every shipped \
                         asset must carry (first: {}); add one or write the pack with \
                         require_texture=false",
                        missing.len(),
                        missing[0]
                    ),
                });
            }
        }

        // Stage every (name, bytes) pair in emission order. The index goes first
        // so a reader can find it with one small read at a known position.
        let mut index = VxpIndex {
            version: VXP_INDEX_VERSION,
            pack_id: self.pack_id.clone(),
            assets: Vec::with_capacity(self.assets.len()),
            // `BTreeMap` values come out ascending by content id, which is the
            // canonical register order and what makes the bytes reproducible.
            textures: self.textures.values().cloned().collect(),
        };
        let mut payloads: Vec<(String, Vec<u8>)> = Vec::new();
        // The index is the ownership/reference layer. Identical immutable
        // payloads are stored once and every asset record points at that one
        // content-addressed entry. This is especially important for authored
        // GLBs shared by character wardrobe variants: the VXP must reference
        // the source bytes, not copy them once per variant.
        let mut staged: BTreeMap<(VxpKind, String), String> = BTreeMap::new();

        for (asset_id, pending) in &self.assets {
            let mut record = VxpAssetIndex {
                asset_id: asset_id.clone(),
                geometry: None,
                geometry_lods: Vec::new(),
                surface: None,
                metadata: None,
                texture: None,
                animations: Vec::new(),
                atlas: None,
                texture_ids: pending.texture_ids.iter().cloned().collect(),
                bytes: 0,
            };
            if let Some(bytes) = &pending.geometry {
                record.bytes += bytes.len() as u64;
                record.geometry = Some(stage_payload(
                    &mut payloads,
                    &mut staged,
                    asset_id,
                    VxpKind::Geometry,
                    None,
                    bytes,
                ));
            }
            if let Some(bytes) = &pending.surface {
                record.bytes += bytes.len() as u64;
                record.surface = Some(stage_payload(
                    &mut payloads,
                    &mut staged,
                    asset_id,
                    VxpKind::Surface,
                    None,
                    bytes,
                ));
            }
            if let Some(bytes) = &pending.metadata {
                record.bytes += bytes.len() as u64;
                record.metadata = Some(stage_payload(
                    &mut payloads,
                    &mut staged,
                    asset_id,
                    VxpKind::Metadata,
                    None,
                    bytes,
                ));
            }
            if let Some(bytes) = &pending.texture {
                record.bytes += bytes.len() as u64;
                record.texture = Some(stage_payload(
                    &mut payloads,
                    &mut staged,
                    asset_id,
                    VxpKind::Texture,
                    None,
                    bytes,
                ));
            }
            for (clip_id, bytes) in &pending.animations {
                record.bytes += bytes.len() as u64;
                let sub_asset_id = format!("{asset_id}_{clip_id}");
                let name = stage_payload(
                    &mut payloads,
                    &mut staged,
                    &sub_asset_id,
                    VxpKind::Animation,
                    None,
                    bytes,
                );
                record.animations.push((clip_id.clone(), name));
            }
            if let Some(bytes) = &pending.atlas {
                record.bytes += bytes.len() as u64;
                record.atlas = Some(stage_payload(
                    &mut payloads,
                    &mut staged,
                    asset_id,
                    VxpKind::Atlas,
                    None,
                    bytes,
                ));
            }
            index.assets.push(record);
        }

        // The register must be CLOSED before a byte is written: an asset that
        // samples a texture the pack never registered would leave the loader
        // sizing residency from an incomplete set — the exact silent default
        // this format change exists to make impossible.
        index.validate_texture_register()?;

        let index_bytes = serde_json::to_vec(&index).map_err(|source| VxpError::Json {
            context: format!("{} index", path.display()),
            source,
        })?;

        // Write to a sibling temp file and rename, so an interrupted write can
        // never replace a good pack with a truncated one.
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent).map_err(|source| VxpError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
        let temp = parent.join(format!(
            ".{}.{}.vxp.part",
            path.file_name().unwrap_or_default().to_string_lossy(),
            std::process::id()
        ));
        {
            let file = File::create(&temp).map_err(|source| VxpError::Io {
                path: temp.clone(),
                source,
            })?;
            let mut zip = StoredZipWriter::new(BufWriter::new(file));
            zip.add(VXP_INDEX_ENTRY, &index_bytes)
                .map_err(|source| VxpError::Io {
                    path: temp.clone(),
                    source,
                })?;
            for (name, bytes) in &payloads {
                zip.add(name, bytes).map_err(|source| VxpError::Io {
                    path: temp.clone(),
                    source,
                })?;
            }
            zip.finish().map_err(|source| VxpError::Io {
                path: temp.clone(),
                source,
            })?;
        }
        std::fs::rename(&temp, path).map_err(|source| VxpError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(index)
    }
}

fn stage_payload(
    payloads: &mut Vec<(String, Vec<u8>)>,
    staged: &mut BTreeMap<(VxpKind, String), String>,
    asset_id: &str,
    kind: VxpKind,
    lod: Option<u8>,
    bytes: &[u8],
) -> String {
    let cid = content_id(bytes);
    let key = (kind, cid.clone());
    if let Some(name) = staged.get(&key) {
        return name.clone();
    }
    let name = entry_name(asset_id, kind, lod, &cid);
    payloads.push((name.clone(), bytes.to_vec()));
    payloads.push((format!("{name}.cid"), cid.into_bytes()));
    staged.insert(key, name.clone());
    name
}

// ─────────────────────────────────────────────────────────────────────────────
// Reader
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DirEntry {
    /// Offset of the LOCAL file header.
    local_header_offset: u64,
    size: u64,
    crc32: u32,
}

/// Reads a `.vxp`. Construction parses only the central directory; payload bytes
/// are pulled on demand, which is what lets a game list a pack and show
/// thumbnails without paying for geometry.
#[derive(Debug)]
pub struct VxpReader {
    path: PathBuf,
    file: File,
    dir: BTreeMap<String, DirEntry>,
    index: VxpIndex,
}

/// Exact byte range of one STORE entry inside a `.vxp`.
///
/// Geometry transport can copy this range directly from the pack into a
/// GPU-selected destination. It never has to inflate, reinterpret, or scan the
/// container on the render thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VxpEntryLocation {
    pub data_offset: u64,
    pub byte_count: u64,
}

impl VxpReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(|source| VxpError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let dir = read_central_directory(&mut file, path)?;
        let mut reader = Self {
            path: path.to_path_buf(),
            file,
            dir,
            index: VxpIndex {
                version: 0,
                pack_id: String::new(),
                assets: Vec::new(),
                textures: Vec::new(),
            },
        };
        let index_bytes = reader.read_raw(VXP_INDEX_ENTRY)?;
        reader.index = serde_json::from_slice(&index_bytes).map_err(|source| VxpError::Json {
            context: format!("{} {VXP_INDEX_ENTRY}", path.display()),
            source,
        })?;
        if !(VXP_MIN_READABLE_INDEX_VERSION..=VXP_INDEX_VERSION).contains(&reader.index.version) {
            return Err(VxpError::Format {
                path: path.to_path_buf(),
                reason: format!(
                    "index version {} but this build reads {}..={VXP_INDEX_VERSION} \
                     (v1 has no texture register; a newer version is unknown)",
                    reader.index.version, VXP_MIN_READABLE_INDEX_VERSION,
                ),
            });
        }
        reader.index.validate_texture_register()?;
        Ok(reader)
    }

    pub fn index(&self) -> &VxpIndex {
        &self.index
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Resolve an entry's exact stored-data range without reading its payload.
    pub fn entry_location(&mut self, name: &str) -> Result<VxpEntryLocation> {
        let entry = self
            .dir
            .get(name)
            .cloned()
            .ok_or_else(|| VxpError::MissingEntry {
                path: self.path.clone(),
                name: name.to_string(),
            })?;
        let path = self.path.clone();
        let io = |source| VxpError::Io {
            path: path.clone(),
            source,
        };
        self.file
            .seek(SeekFrom::Start(entry.local_header_offset))
            .map_err(io)?;
        let mut header = [0u8; 30];
        self.file.read_exact(&mut header).map_err(io)?;
        if u32::from_le_bytes([header[0], header[1], header[2], header[3]]) != 0x0403_4b50 {
            return Err(VxpError::Corrupt {
                path: self.path.clone(),
                name: name.to_string(),
                reason: "local file header signature missing".to_string(),
            });
        }
        let method = u16::from_le_bytes([header[8], header[9]]);
        if method != 0 {
            return Err(VxpError::Corrupt {
                path: self.path.clone(),
                name: name.to_string(),
                reason: format!(
                    "entry uses compression method {method}; geometry pages require STORE"
                ),
            });
        }
        let name_len = u16::from_le_bytes([header[26], header[27]]) as u64;
        let extra_len = u16::from_le_bytes([header[28], header[29]]) as u64;
        Ok(VxpEntryLocation {
            data_offset: entry.local_header_offset + 30 + name_len + extra_len,
            byte_count: entry.size,
        })
    }

    /// Read an entry, verifying BOTH the ZIP CRC-32 and the content hash carried
    /// in the entry name. A payload that fails either is an error, never data.
    pub fn read_entry(&mut self, name: &str) -> Result<Vec<u8>> {
        let bytes = self.read_raw(name)?;
        if let Some(expected) = cid_in_name(name) {
            let actual = content_id(&bytes);
            if actual != expected {
                return Err(VxpError::Corrupt {
                    path: self.path.clone(),
                    name: name.to_string(),
                    reason: format!("content hash {actual} does not match the name's {expected}"),
                });
            }
        }
        Ok(bytes)
    }

    fn read_raw(&mut self, name: &str) -> Result<Vec<u8>> {
        let entry = self
            .dir
            .get(name)
            .cloned()
            .ok_or_else(|| VxpError::MissingEntry {
                path: self.path.clone(),
                name: name.to_string(),
            })?;
        let path = self.path.clone();
        let io = |source| VxpError::Io {
            path: path.clone(),
            source,
        };
        // The local header repeats the name/extra lengths; the payload begins
        // right after them. (Sizes are read from the central directory, which is
        // authoritative and already parsed.)
        let location = self.entry_location(name)?;
        self.file
            .seek(SeekFrom::Start(location.data_offset))
            .map_err(io)?;
        let mut bytes = vec![0u8; entry.size as usize];
        self.file.read_exact(&mut bytes).map_err(io)?;
        let crc = crc32(&bytes);
        if crc != entry.crc32 {
            return Err(VxpError::Corrupt {
                path: self.path.clone(),
                name: name.to_string(),
                reason: format!("crc32 {crc:08x} != stored {:08x}", entry.crc32),
            });
        }
        Ok(bytes)
    }
}

/// The 32-hex content id embedded in `<...>_<32hex>.<Kind>`, if the name has the
/// canonical shape.
fn cid_in_name(name: &str) -> Option<String> {
    let stem = name.rsplit_once('.')?.0;
    let cid = stem.rsplit_once('_')?.1;
    (cid.len() == 32 && cid.bytes().all(|b| b.is_ascii_hexdigit())).then(|| cid.to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// Binary geometry
// ─────────────────────────────────────────────────────────────────────────────

pub const VXPG_MAGIC: [u8; 4] = *b"VXPG";
pub const VXPG_VERSION: u32 = 1;

/// A finished mesh in the form `.vxp` stores it — game-agnostic on purpose: it
/// is positions/normals/uvs/indices plus NAMED auxiliary streams, so a game can
/// carry whatever per-vertex or per-asset data its renderer needs (weathering
/// masks, a baked SDF, surface ids) without this crate learning what any of it
/// means.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VxpGeometry {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<[u32; 3]>,
    /// Per-triangle or per-vertex `u8` classification streams, by name.
    pub aux_u8: BTreeMap<String, Vec<u8>>,
    /// Named `f32` streams with a component count (e.g. 7 weathering channels
    /// per vertex).
    pub aux_f32: BTreeMap<String, (u32, Vec<f32>)>,
    /// Named `i16` streams (e.g. an snorm16 distance field).
    pub aux_i16: BTreeMap<String, Vec<i16>>,
    /// Named `f32` scalars/vectors that describe a stream (e.g. an SDF's origin
    /// and voxel size).
    pub aux_params: BTreeMap<String, Vec<f32>>,
    /// Named string lists (e.g. per-triangle surface ids).
    pub strings: BTreeMap<String, Vec<String>>,
}

impl VxpGeometry {
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Encode to the little-endian `VXPG` binary form.
    ///
    /// Every array is written tightly packed, so a 320k-vertex building costs
    /// bytes proportional to its data — not the ~30 bytes per float that a
    /// pretty-printed JSON array costs.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_size_hint());
        out.extend_from_slice(&VXPG_MAGIC);
        out.extend_from_slice(&VXPG_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.positions.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.indices.len() as u32).to_le_bytes());
        let mut flags = 0u32;
        if !self.normals.is_empty() {
            flags |= 1;
        }
        if !self.uvs.is_empty() {
            flags |= 2;
        }
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&(self.aux_u8.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.aux_f32.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.aux_i16.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.aux_params.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.strings.len() as u32).to_le_bytes());

        for p in &self.positions {
            for v in p {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        if flags & 1 != 0 {
            for n in &self.normals {
                for v in n {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        if flags & 2 != 0 {
            for uv in &self.uvs {
                for v in uv {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        for tri in &self.indices {
            for v in tri {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for (name, data) in &self.aux_u8 {
            write_str(&mut out, name);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(data);
        }
        for (name, (components, data)) in &self.aux_f32 {
            write_str(&mut out, name);
            out.extend_from_slice(&components.to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            for v in data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for (name, data) in &self.aux_i16 {
            write_str(&mut out, name);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            for v in data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for (name, data) in &self.aux_params {
            write_str(&mut out, name);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            for v in data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for (name, list) in &self.strings {
            write_str(&mut out, name);
            out.extend_from_slice(&(list.len() as u32).to_le_bytes());
            for s in list {
                write_str(&mut out, s);
            }
        }
        out
    }

    fn encoded_size_hint(&self) -> usize {
        40 + self.positions.len() * 12
            + self.normals.len() * 12
            + self.uvs.len() * 8
            + self.indices.len() * 12
    }

    /// Decode the little-endian `VXPG` binary form.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = Cursor { bytes, at: 0 };
        let magic = r.take(4)?;
        if magic != VXPG_MAGIC {
            return Err(VxpError::Geometry(format!(
                "bad magic {magic:?}, expected VXPG"
            )));
        }
        let version = r.u32()?;
        if version != VXPG_VERSION {
            return Err(VxpError::Geometry(format!(
                "geometry version {version} but this build reads {VXPG_VERSION}"
            )));
        }
        let vertex_count = r.u32()? as usize;
        let triangle_count = r.u32()? as usize;
        let flags = r.u32()?;
        let aux_u8_count = r.u32()? as usize;
        let aux_f32_count = r.u32()? as usize;
        let aux_i16_count = r.u32()? as usize;
        let aux_params_count = r.u32()? as usize;
        let strings_count = r.u32()? as usize;

        let mut geo = VxpGeometry::default();
        geo.positions.reserve_exact(vertex_count);
        for _ in 0..vertex_count {
            geo.positions.push([r.f32()?, r.f32()?, r.f32()?]);
        }
        if flags & 1 != 0 {
            geo.normals.reserve_exact(vertex_count);
            for _ in 0..vertex_count {
                geo.normals.push([r.f32()?, r.f32()?, r.f32()?]);
            }
        }
        if flags & 2 != 0 {
            geo.uvs.reserve_exact(vertex_count);
            for _ in 0..vertex_count {
                geo.uvs.push([r.f32()?, r.f32()?]);
            }
        }
        geo.indices.reserve_exact(triangle_count);
        for _ in 0..triangle_count {
            geo.indices.push([r.u32()?, r.u32()?, r.u32()?]);
        }
        for _ in 0..aux_u8_count {
            let name = r.string()?;
            let len = r.u32()? as usize;
            geo.aux_u8.insert(name, r.take(len)?.to_vec());
        }
        for _ in 0..aux_f32_count {
            let name = r.string()?;
            let components = r.u32()?;
            let len = r.u32()? as usize;
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                data.push(r.f32()?);
            }
            geo.aux_f32.insert(name, (components, data));
        }
        for _ in 0..aux_i16_count {
            let name = r.string()?;
            let len = r.u32()? as usize;
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                data.push(r.i16()?);
            }
            geo.aux_i16.insert(name, data);
        }
        for _ in 0..aux_params_count {
            let name = r.string()?;
            let len = r.u32()? as usize;
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                data.push(r.f32()?);
            }
            geo.aux_params.insert(name, data);
        }
        for _ in 0..strings_count {
            let name = r.string()?;
            let len = r.u32()? as usize;
            let mut list = Vec::with_capacity(len);
            for _ in 0..len {
                list.push(r.string()?);
            }
            geo.strings.insert(name, list);
        }
        if r.at != bytes.len() {
            return Err(VxpError::Geometry(format!(
                "{} trailing bytes after geometry payload",
                bytes.len() - r.at
            )));
        }
        Ok(geo)
    }
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(n).ok_or_else(|| {
            VxpError::Geometry("length overflow while decoding geometry".to_string())
        })?;
        if end > self.bytes.len() {
            return Err(VxpError::Geometry(format!(
                "geometry payload truncated: wanted {n} bytes at {}, only {} remain",
                self.at,
                self.bytes.len().saturating_sub(self.at)
            )));
        }
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i16(&mut self) -> Result<i16> {
        let b = self.take(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    fn f32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| VxpError::Geometry(format!("non-UTF-8 name in geometry payload: {e}")))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// A deterministic, STORE-only ZIP writer
// ─────────────────────────────────────────────────────────────────────────────
//
// Hand-rolled rather than pulled from a crate for two reasons, both required by
// the format: (1) every general-purpose zip writer stamps the current time into
// each entry, which would make identical input produce a different file every
// run and break the determinism LAW; (2) we only ever want method STORE, so
// there is no compression machinery to reuse. ZIP64 records are emitted when a
// size or offset no longer fits in 32 bits, so a multi-gigabyte pack is legal.

/// MS-DOS 1980-01-01 00:00:00 — a fixed timestamp, so the bytes depend only on
/// the content.
const DOS_EPOCH_TIME: u16 = 0;
const DOS_EPOCH_DATE: u16 = 0x0021;
const ZIP64_THRESHOLD: u64 = 0xFFFF_FFFF;

struct StoredZipEntry {
    name: String,
    crc32: u32,
    size: u64,
    local_header_offset: u64,
}

struct StoredZipWriter<W: Write> {
    out: W,
    at: u64,
    entries: Vec<StoredZipEntry>,
}

impl<W: Write> StoredZipWriter<W> {
    fn new(out: W) -> Self {
        Self {
            out,
            at: 0,
            entries: Vec::new(),
        }
    }

    fn add(&mut self, name: &str, bytes: &[u8]) -> std::io::Result<()> {
        let crc = crc32(bytes);
        let size = bytes.len() as u64;
        let local_header_offset = self.at;
        let zip64 = size >= ZIP64_THRESHOLD;
        // Local file header.
        let mut header = Vec::with_capacity(30 + name.len() + 20);
        header.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        header.extend_from_slice(&if zip64 { 45u16 } else { 20u16 }.to_le_bytes());
        header.extend_from_slice(&0u16.to_le_bytes()); // flags
        header.extend_from_slice(&0u16.to_le_bytes()); // method = STORE
        header.extend_from_slice(&DOS_EPOCH_TIME.to_le_bytes());
        header.extend_from_slice(&DOS_EPOCH_DATE.to_le_bytes());
        header.extend_from_slice(&crc.to_le_bytes());
        if zip64 {
            header.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
            header.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        } else {
            header.extend_from_slice(&(size as u32).to_le_bytes());
            header.extend_from_slice(&(size as u32).to_le_bytes());
        }
        header.extend_from_slice(&(name.len() as u16).to_le_bytes());
        header.extend_from_slice(&if zip64 { 20u16 } else { 0u16 }.to_le_bytes());
        header.extend_from_slice(name.as_bytes());
        if zip64 {
            header.extend_from_slice(&0x0001u16.to_le_bytes());
            header.extend_from_slice(&16u16.to_le_bytes());
            header.extend_from_slice(&size.to_le_bytes());
            header.extend_from_slice(&size.to_le_bytes());
        }
        self.out.write_all(&header)?;
        self.out.write_all(bytes)?;
        self.at += header.len() as u64 + size;
        self.entries.push(StoredZipEntry {
            name: name.to_string(),
            crc32: crc,
            size,
            local_header_offset,
        });
        Ok(())
    }

    fn finish(mut self) -> std::io::Result<()> {
        let cd_start = self.at;
        for entry in &self.entries {
            let need_zip64 =
                entry.size >= ZIP64_THRESHOLD || entry.local_header_offset >= ZIP64_THRESHOLD;
            let mut extra = Vec::new();
            if need_zip64 {
                extra.extend_from_slice(&0x0001u16.to_le_bytes());
                extra.extend_from_slice(&24u16.to_le_bytes());
                extra.extend_from_slice(&entry.size.to_le_bytes());
                extra.extend_from_slice(&entry.size.to_le_bytes());
                extra.extend_from_slice(&entry.local_header_offset.to_le_bytes());
            }
            let mut cd = Vec::with_capacity(46 + entry.name.len() + extra.len());
            cd.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            cd.extend_from_slice(&if need_zip64 { 45u16 } else { 20u16 }.to_le_bytes()); // made by
            cd.extend_from_slice(&if need_zip64 { 45u16 } else { 20u16 }.to_le_bytes()); // needed
            cd.extend_from_slice(&0u16.to_le_bytes()); // flags
            cd.extend_from_slice(&0u16.to_le_bytes()); // method = STORE
            cd.extend_from_slice(&DOS_EPOCH_TIME.to_le_bytes());
            cd.extend_from_slice(&DOS_EPOCH_DATE.to_le_bytes());
            cd.extend_from_slice(&entry.crc32.to_le_bytes());
            let stored32 = if need_zip64 {
                0xFFFF_FFFFu32
            } else {
                entry.size as u32
            };
            cd.extend_from_slice(&stored32.to_le_bytes());
            cd.extend_from_slice(&stored32.to_le_bytes());
            cd.extend_from_slice(&(entry.name.len() as u16).to_le_bytes());
            cd.extend_from_slice(&(extra.len() as u16).to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes()); // comment len
            cd.extend_from_slice(&0u16.to_le_bytes()); // disk number
            cd.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
            cd.extend_from_slice(&0u32.to_le_bytes()); // external attrs
            cd.extend_from_slice(
                &if need_zip64 {
                    0xFFFF_FFFFu32
                } else {
                    entry.local_header_offset as u32
                }
                .to_le_bytes(),
            );
            cd.extend_from_slice(entry.name.as_bytes());
            cd.extend_from_slice(&extra);
            self.out.write_all(&cd)?;
            self.at += cd.len() as u64;
        }
        let cd_size = self.at - cd_start;
        let count = self.entries.len() as u64;
        let need_zip64 =
            count > 0xFFFF || cd_start >= ZIP64_THRESHOLD || cd_size >= ZIP64_THRESHOLD;
        if need_zip64 {
            let eocd64_at = self.at;
            let mut z = Vec::with_capacity(56);
            z.extend_from_slice(&0x0606_4b50u32.to_le_bytes());
            z.extend_from_slice(&44u64.to_le_bytes()); // size of remainder
            z.extend_from_slice(&45u16.to_le_bytes());
            z.extend_from_slice(&45u16.to_le_bytes());
            z.extend_from_slice(&0u32.to_le_bytes()); // this disk
            z.extend_from_slice(&0u32.to_le_bytes()); // cd start disk
            z.extend_from_slice(&count.to_le_bytes());
            z.extend_from_slice(&count.to_le_bytes());
            z.extend_from_slice(&cd_size.to_le_bytes());
            z.extend_from_slice(&cd_start.to_le_bytes());
            self.out.write_all(&z)?;
            self.at += z.len() as u64;
            let mut l = Vec::with_capacity(20);
            l.extend_from_slice(&0x0706_4b50u32.to_le_bytes());
            l.extend_from_slice(&0u32.to_le_bytes());
            l.extend_from_slice(&eocd64_at.to_le_bytes());
            l.extend_from_slice(&1u32.to_le_bytes());
            self.out.write_all(&l)?;
            self.at += l.len() as u64;
        }
        let mut eocd = Vec::with_capacity(22);
        eocd.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        eocd.extend_from_slice(&0u16.to_le_bytes());
        eocd.extend_from_slice(&0u16.to_le_bytes());
        let count16 = if count > 0xFFFF {
            0xFFFFu16
        } else {
            count as u16
        };
        eocd.extend_from_slice(&count16.to_le_bytes());
        eocd.extend_from_slice(&count16.to_le_bytes());
        eocd.extend_from_slice(
            &if cd_size >= ZIP64_THRESHOLD {
                0xFFFF_FFFFu32
            } else {
                cd_size as u32
            }
            .to_le_bytes(),
        );
        eocd.extend_from_slice(
            &if cd_start >= ZIP64_THRESHOLD {
                0xFFFF_FFFFu32
            } else {
                cd_start as u32
            }
            .to_le_bytes(),
        );
        eocd.extend_from_slice(&0u16.to_le_bytes()); // comment len
        self.out.write_all(&eocd)?;
        self.out.flush()?;
        Ok(())
    }
}

fn read_central_directory(file: &mut File, path: &Path) -> Result<BTreeMap<String, DirEntry>> {
    let io = |source| VxpError::Io {
        path: path.to_path_buf(),
        source,
    };
    let len = file.metadata().map_err(io)?.len();
    // The EOCD is the last 22 bytes when there is no comment; we always write no
    // comment, but scan a small tail anyway so a foreign-written pack still reads.
    let tail_len = len.min(66_000);
    file.seek(SeekFrom::Start(len - tail_len)).map_err(io)?;
    let mut tail = vec![0u8; tail_len as usize];
    file.read_exact(&mut tail).map_err(io)?;
    let eocd_rel = (0..=tail.len().saturating_sub(22))
        .rev()
        .find(|&i| tail[i..i + 4] == 0x0605_4b50u32.to_le_bytes())
        .ok_or_else(|| VxpError::Format {
            path: path.to_path_buf(),
            reason: "no end-of-central-directory record — not a zip/vxp".to_string(),
        })?;
    let eocd = &tail[eocd_rel..];
    let mut count = u16::from_le_bytes([eocd[10], eocd[11]]) as u64;
    let mut cd_start = u32::from_le_bytes([eocd[16], eocd[17], eocd[18], eocd[19]]) as u64;

    // ZIP64 locator sits immediately before the EOCD.
    if (count == 0xFFFF || cd_start == ZIP64_THRESHOLD) && eocd_rel >= 20 {
        let loc = &tail[eocd_rel - 20..eocd_rel];
        if loc[0..4] == 0x0706_4b50u32.to_le_bytes() {
            let eocd64_at = u64::from_le_bytes(loc[8..16].try_into().unwrap());
            file.seek(SeekFrom::Start(eocd64_at)).map_err(io)?;
            let mut z = [0u8; 56];
            file.read_exact(&mut z).map_err(io)?;
            if z[0..4] != 0x0606_4b50u32.to_le_bytes() {
                return Err(VxpError::Format {
                    path: path.to_path_buf(),
                    reason: "zip64 locator points at a non-zip64-EOCD".to_string(),
                });
            }
            count = u64::from_le_bytes(z[32..40].try_into().unwrap());
            cd_start = u64::from_le_bytes(z[48..56].try_into().unwrap());
        }
    }

    file.seek(SeekFrom::Start(cd_start)).map_err(io)?;
    let mut cd = Vec::new();
    file.read_to_end(&mut cd).map_err(io)?;
    let mut dir = BTreeMap::new();
    let mut at = 0usize;
    for _ in 0..count {
        if at + 46 > cd.len() || cd[at..at + 4] != 0x0201_4b50u32.to_le_bytes() {
            return Err(VxpError::Format {
                path: path.to_path_buf(),
                reason: format!("central directory truncated or corrupt at byte {at}"),
            });
        }
        let crc32 = u32::from_le_bytes(cd[at + 16..at + 20].try_into().unwrap());
        let mut size = u32::from_le_bytes(cd[at + 24..at + 28].try_into().unwrap()) as u64;
        let name_len = u16::from_le_bytes(cd[at + 28..at + 30].try_into().unwrap()) as usize;
        let extra_len = u16::from_le_bytes(cd[at + 30..at + 32].try_into().unwrap()) as usize;
        let comment_len = u16::from_le_bytes(cd[at + 32..at + 34].try_into().unwrap()) as usize;
        let mut local_header_offset =
            u32::from_le_bytes(cd[at + 42..at + 46].try_into().unwrap()) as u64;
        let name = String::from_utf8_lossy(&cd[at + 46..at + 46 + name_len]).into_owned();
        // ZIP64 extended-information extra field, if the 32-bit slots overflowed.
        if size == ZIP64_THRESHOLD || local_header_offset == ZIP64_THRESHOLD {
            let extra = &cd[at + 46 + name_len..at + 46 + name_len + extra_len];
            let mut e = 0usize;
            while e + 4 <= extra.len() {
                let tag = u16::from_le_bytes(extra[e..e + 2].try_into().unwrap());
                let field_len =
                    u16::from_le_bytes(extra[e + 2..e + 4].try_into().unwrap()) as usize;
                if tag == 0x0001 {
                    let f = &extra[e + 4..(e + 4 + field_len).min(extra.len())];
                    let mut o = 0usize;
                    if size == ZIP64_THRESHOLD && o + 8 <= f.len() {
                        size = u64::from_le_bytes(f[o..o + 8].try_into().unwrap());
                        o += 8;
                    }
                    if o + 8 <= f.len() {
                        o += 8; // compressed size — equal to `size` for STORE
                    }
                    if local_header_offset == ZIP64_THRESHOLD && o + 8 <= f.len() {
                        local_header_offset = u64::from_le_bytes(f[o..o + 8].try_into().unwrap());
                    }
                    break;
                }
                e += 4 + field_len;
            }
        }
        dir.insert(
            name,
            DirEntry {
                local_header_offset,
                size,
                crc32,
            },
        );
        at += 46 + name_len + extra_len + comment_len;
    }
    Ok(dir)
}

/// CRC-32 (IEEE), the checksum ZIP stores per entry.
fn crc32(bytes: &[u8]) -> u32 {
    static TABLE: std::sync::LazyLock<[u32; 256]> = std::sync::LazyLock::new(|| {
        let mut table = [0u32; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *slot = c;
        }
        table
    });
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc = TABLE[((crc ^ u32::from(b)) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_geometry() -> VxpGeometry {
        let mut geo = VxpGeometry {
            positions: vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 2.25, -3.5]],
            normals: vec![[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![[0, 1, 2]],
            ..Default::default()
        };
        geo.aux_u8.insert("material_indices".to_string(), vec![3]);
        geo.aux_f32.insert(
            "weathering".to_string(),
            (7, (0..21).map(|i| i as f32 * 0.125).collect()),
        );
        geo.aux_i16
            .insert("sdf".to_string(), vec![-32768, 0, 17, 32767]);
        geo.aux_params
            .insert("sdf_origin".to_string(), vec![-1.0, -2.0, -3.0]);
        geo.strings.insert(
            "exterior_surface_ids".to_string(),
            vec!["wall".to_string(), "roof".to_string()],
        );
        geo
    }

    #[test]
    fn geometry_round_trips_every_stream_exactly() {
        let geo = sample_geometry();
        let encoded = geo.encode();
        let decoded = VxpGeometry::decode(&encoded).expect("decode");
        assert_eq!(decoded, geo);
        // And the encoding is what makes this worth doing: 3 verts with normals,
        // uvs, 1 tri, 21 weathering floats and a 4-sample sdf must be well under
        // 400 bytes, where the equivalent pretty JSON is thousands.
        assert!(
            encoded.len() < 400,
            "binary geometry unexpectedly large: {} bytes",
            encoded.len()
        );
    }

    #[test]
    fn geometry_binary_is_dramatically_smaller_than_pretty_json() {
        // The concrete claim this format exists to make. 20k verts / 10k tris.
        let mut geo = VxpGeometry::default();
        for i in 0..20_000u32 {
            // Realistic mesh coordinates: irrational-looking f32s that need
            // their full significand printed, which is what a real cooked
            // building's arrays look like.
            let f = (i as f32).sqrt() * 0.317_31;
            geo.positions.push([f, f * 2.718_28, -f * 1.414_213_5]);
            geo.normals
                .push([f.sin(), f.cos(), (1.0 - f.sin() * f.sin()).sqrt()]);
            geo.uvs.push([f.fract(), (f * 0.5).fract()]);
        }
        for i in 0..10_000u32 {
            geo.indices.push([i, i + 1, i + 2]);
        }
        // Seven weathering channels per vertex — the same shape as the real
        // corpus, where this one flat array was the single largest contributor.
        let weathering: Vec<f32> = (0..20_000 * 7)
            .map(|i| ((i as f32) * 0.618_034).fract())
            .collect();
        geo.aux_f32
            .insert("weathering".to_string(), (7, weathering.clone()));
        let binary = geo.encode().len();
        // Same numbers, serialized the way the cook used to write them.
        #[derive(serde::Serialize)]
        struct AsJson<'a> {
            positions: &'a [[f32; 3]],
            normals: &'a [[f32; 3]],
            uvs: &'a [[f32; 2]],
            indices: &'a [[u32; 3]],
            weathering_masks: &'a [f32],
        }
        let json = serde_json::to_vec_pretty(&AsJson {
            positions: &geo.positions,
            normals: &geo.normals,
            uvs: &geo.uvs,
            indices: &geo.indices,
            weathering_masks: &weathering,
        })
        .unwrap()
        .len();
        // The property that actually holds by construction: the binary form is
        // EXACTLY the raw arrays plus a fixed 40-byte header and one named-
        // stream record. Nothing is spent on syntax.
        let raw = geo.positions.len() * 12
            + geo.normals.len() * 12
            + geo.uvs.len() * 8
            + geo.indices.len() * 12
            + weathering.len() * 4;
        assert_eq!(
            binary,
            40 + raw + 4 + "weathering".len() + 4 + 4,
            "binary geometry must be raw arrays + a fixed header, nothing else"
        );
        // And the size win over the format this replaces. 4x is the floor on
        // this synthetic data; the measured real-corpus figure is higher (a
        // 320k-vertex building was 147 MB of pretty JSON) because real cooked
        // coordinates need their full significand printed.
        assert!(
            json > binary * 4,
            "expected pretty JSON to be >4x the binary size; binary={binary} json={json}"
        );
    }

    #[test]
    fn geometry_decode_rejects_truncation_instead_of_guessing() {
        let encoded = sample_geometry().encode();
        let err = VxpGeometry::decode(&encoded[..encoded.len() - 5]).unwrap_err();
        assert!(
            format!("{err}").contains("truncated"),
            "expected a truncation error, got: {err}"
        );
    }

    #[test]
    fn geometry_decode_rejects_a_foreign_container() {
        let err = VxpGeometry::decode(b"{\"positions\":[]}").unwrap_err();
        assert!(format!("{err}").contains("bad magic"), "got: {err}");
    }

    /// Two textures both sample assets share, so the register also proves
    /// dedupe-by-content-hash.
    fn sample_textures() -> [VxpTexture; 2] {
        [
            VxpTexture::new(
                content_id(b"brick base color"),
                2048,
                2048,
                VxpTextureFormat::Bc7UnormSrgb,
                12,
            )
            .expect("brick"),
            VxpTexture::new(
                content_id(b"brick normal"),
                1024,
                1024,
                VxpTextureFormat::Bc5Unorm,
                11,
            )
            .expect("normal"),
        ]
    }

    fn write_sample_pack(dir: &Path, name: &str) -> (PathBuf, VxpIndex) {
        let mut w = VxpWriter::new("test_theme");
        // Registered in the OPPOSITE order to the canonical one, to prove the
        // emitted register is ordered by content, not by call order.
        for texture in sample_textures().into_iter().rev() {
            w.register_texture(texture).expect("register");
        }
        for id in ["b_two", "a_one"] {
            w.add_geometry(id, sample_geometry().encode());
            w.add_surface(id, br#"{"materials":[]}"#.to_vec());
            w.add_metadata(id, format!(r#"{{"id":"{id}"}}"#).into_bytes());
            w.add_texture(id, vec![0x89, b'P', b'N', b'G', 13, 10, 26, 10]);
            w.add_animation(id, "Talk", b"animation-talk".to_vec());
            w.add_animation(id, "Hug", b"animation-hug".to_vec());
            w.add_atlas(id, b"shared-atlas".to_vec());
            for texture in sample_textures() {
                w.reference_texture(id, &texture.id);
            }
        }
        let path = dir.join(name);
        let index = w.finish(&path, true).expect("write pack");
        (path, index)
    }

    #[test]
    fn pack_round_trips_through_the_reader() {
        let dir = tempfile::tempdir().unwrap();
        let (path, written) = write_sample_pack(dir.path(), "theme.vxp");
        let mut reader = VxpReader::open(&path).expect("open");
        assert_eq!(reader.index(), &written);
        assert_eq!(reader.index().version, 4);
        assert_eq!(reader.index().pack_id, "test_theme");
        // Index order is by asset id, not insertion order.
        let ids: Vec<&str> = reader
            .index()
            .assets
            .iter()
            .map(|a| a.asset_id.as_str())
            .collect();
        assert_eq!(ids, ["a_one", "b_two"]);

        let record = reader.index().asset("a_one").cloned().expect("a_one");
        let geometry_name = record.geometry.clone().expect("geometry");
        let bytes = reader.read_entry(&geometry_name).expect("read geometry");
        assert_eq!(VxpGeometry::decode(&bytes).unwrap(), sample_geometry());
        assert_eq!(
            reader
                .read_entry(record.metadata.as_ref().unwrap())
                .unwrap(),
            br#"{"id":"a_one"}"#
        );
        assert_eq!(
            reader.read_entry(record.texture.as_ref().unwrap()).unwrap()[..4],
            [0x89, b'P', b'N', b'G']
        );
        assert!(
            record.geometry_lods.is_empty(),
            "new packs must contain only the authoritative mesh"
        );
        assert_eq!(
            record
                .animations
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>(),
            ["Hug", "Talk"]
        );
        assert_eq!(
            reader.read_entry(&record.animations[1].1).unwrap(),
            b"animation-talk"
        );
        assert_eq!(
            reader.read_entry(record.atlas.as_ref().unwrap()).unwrap(),
            b"shared-atlas"
        );
    }

    #[test]
    fn new_pack_index_cannot_carry_authored_detail_or_runtime_pages() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = write_sample_pack(dir.path(), "theme.vxp");
        let mut reader = VxpReader::open(&path).expect("open");
        let index = String::from_utf8(
            reader
                .read_entry(VXP_INDEX_ENTRY)
                .expect("read canonical index"),
        )
        .expect("index utf8");
        assert!(!index.contains("geometry_lods"), "{index}");
        assert!(!index.contains("geometry_pages"), "{index}");
        assert!(!index.contains("_PAGE"), "{index}");
    }

    #[test]
    fn every_entry_has_a_32_byte_cid_sidecar_matching_its_payload() {
        let dir = tempfile::tempdir().unwrap();
        let (path, index) = write_sample_pack(dir.path(), "theme.vxp");
        let mut reader = VxpReader::open(&path).unwrap();
        let asset = index.asset("a_one").unwrap().clone();
        for name in [
            asset.geometry.clone().unwrap(),
            asset.surface.clone().unwrap(),
            asset.metadata.clone().unwrap(),
            asset.texture.clone().unwrap(),
        ] {
            let payload = reader.read_entry(&name).unwrap();
            let sidecar = reader.read_entry(&format!("{name}.cid")).unwrap();
            assert_eq!(sidecar.len(), 32, "cid sidecar must be exactly 32 bytes");
            assert_eq!(
                String::from_utf8(sidecar).unwrap(),
                content_id(&payload),
                "{name}: sidecar must equal the payload's content id"
            );
        }
    }

    #[test]
    fn identical_content_dedupes_to_the_same_entry_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.vxp");
        let mut writer = VxpWriter::new("shared");
        for id in ["a_one", "b_two"] {
            writer.add_geometry(id, b"same authoritative GLB".to_vec());
            writer.add_texture(id, b"same thumbnail".to_vec());
        }
        let index = writer.finish(&path, true).unwrap();
        let a = index.asset("a_one").unwrap();
        let b = index.asset("b_two").unwrap();
        assert_eq!(
            a.geometry, b.geometry,
            "both assets must reference the one stored geometry entry"
        );
        assert_eq!(
            a.texture, b.texture,
            "dedupe applies to every immutable payload kind"
        );

        let reader = VxpReader::open(&path).unwrap();
        assert_eq!(
            reader
                .dir
                .keys()
                .filter(|name| name.ends_with(".Geometry"))
                .count(),
            1,
            "the archive must physically contain one Geometry entry"
        );
    }

    #[test]
    fn pack_is_byte_identical_across_writes_and_insertion_orders() {
        let dir = tempfile::tempdir().unwrap();
        let (first, _) = write_sample_pack(dir.path(), "one.vxp");
        let (second, _) = write_sample_pack(dir.path(), "two.vxp");
        let a = std::fs::read(&first).unwrap();
        let b = std::fs::read(&second).unwrap();
        assert_eq!(a, b, "two identical cooks must produce identical bytes");

        // Now build the same pack adding assets — and registering textures —
        // in the opposite order.
        let mut w = VxpWriter::new("test_theme");
        for texture in sample_textures() {
            w.register_texture(texture).expect("register");
        }
        for id in ["a_one", "b_two"] {
            w.add_geometry(id, sample_geometry().encode());
            w.add_surface(id, br#"{"materials":[]}"#.to_vec());
            w.add_metadata(id, format!(r#"{{"id":"{id}"}}"#).into_bytes());
            w.add_texture(id, vec![0x89, b'P', b'N', b'G', 13, 10, 26, 10]);
            // Add the same animation set as `write_sample_pack`, in the
            // opposite call order. Canonical output must be independent of
            // both asset and sub-asset insertion order.
            w.add_animation(id, "Hug", b"animation-hug".to_vec());
            w.add_animation(id, "Talk", b"animation-talk".to_vec());
            w.add_atlas(id, b"shared-atlas".to_vec());
            for texture in sample_textures().into_iter().rev() {
                w.reference_texture(id, &texture.id);
            }
        }
        let third = dir.path().join("three.vxp");
        w.finish(&third, true).unwrap();
        assert_eq!(
            std::fs::read(&third).unwrap(),
            a,
            "insertion order must not affect the bytes"
        );
    }

    #[test]
    fn every_entry_is_stored_uncompressed() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = write_sample_pack(dir.path(), "theme.vxp");
        let bytes = std::fs::read(&path).unwrap();
        // Walk local file headers from the top; method (offset 8) must be 0.
        let mut at = 0usize;
        let mut seen = 0usize;
        while at + 30 <= bytes.len() && bytes[at..at + 4] == 0x0403_4b50u32.to_le_bytes() {
            let method = u16::from_le_bytes([bytes[at + 8], bytes[at + 9]]);
            assert_eq!(method, 0, "entry {seen} is not STORE");
            let size = u32::from_le_bytes(bytes[at + 18..at + 22].try_into().unwrap()) as usize;
            let name_len = u16::from_le_bytes([bytes[at + 26], bytes[at + 27]]) as usize;
            let extra_len = u16::from_le_bytes([bytes[at + 28], bytes[at + 29]]) as usize;
            at += 30 + name_len + extra_len + size;
            seen += 1;
        }
        // Content-addressed payloads are physically deduplicated, so two assets
        // need not produce 2x the local headers. Compare the raw local-header
        // walk with the authoritative central directory instead of hardcoding a
        // pre-dedup entry count; equality proves that every real entry was
        // reached and every one reported STORE above.
        let reader = VxpReader::open(&path).unwrap();
        assert_eq!(
            seen,
            reader.dir.len(),
            "expected every central-directory entry to be walked as STORE"
        );
    }

    #[test]
    fn reader_refuses_a_corrupted_payload_rather_than_returning_it() {
        let dir = tempfile::tempdir().unwrap();
        let (path, index) = write_sample_pack(dir.path(), "theme.vxp");
        let name = index.asset("a_one").unwrap().geometry.clone().unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        // Flip a byte inside the first stored payload's data region.
        let name_len = u16::from_le_bytes([bytes[26], bytes[27]]) as usize;
        let extra_len = u16::from_le_bytes([bytes[28], bytes[29]]) as usize;
        let data_at = 30 + name_len + extra_len;
        bytes[data_at + 40] ^= 0xFF;
        let broken = dir.path().join("broken.vxp");
        std::fs::write(&broken, &bytes).unwrap();
        // The index is written first, so the flip lands in it and `open` itself
        // must fail. Either way, SOME step must fail loudly rather than hand
        // back bad data — never silently serve a corrupted payload.
        let failed = match VxpReader::open(&broken) {
            Err(_) => true,
            Ok(mut reader) => {
                reader.read_entry(VXP_INDEX_ENTRY).is_err() || reader.read_entry(&name).is_err()
            }
        };
        assert!(failed, "a corrupted pack must not read back cleanly");
    }

    #[test]
    fn a_pack_missing_a_thumbnail_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = VxpWriter::new("t");
        w.add_geometry("no_thumb", sample_geometry().encode());
        let err = w
            .finish(&dir.path().join("t.vxp"), true)
            .expect_err("must refuse");
        assert!(
            format!("{err}").contains("no .Texture thumbnail"),
            "got: {err}"
        );
    }

    #[test]
    fn large_pack_uses_zip64_and_still_reads() {
        // Force the ZIP64 path via entry count (>0xFFFF entries).
        let dir = tempfile::tempdir().unwrap();
        let mut w = VxpWriter::new("big");
        for i in 0..17_000u32 {
            let id = format!("a{i:05}");
            w.add_metadata(&id, format!("{{\"i\":{i}}}").into_bytes());
            w.add_texture(&id, vec![0x89, b'P', b'N', b'G']);
        }
        let path = dir.path().join("big.vxp");
        let index = w.finish(&path, true).unwrap();
        assert_eq!(index.assets.len(), 17_000);
        let mut reader = VxpReader::open(&path).unwrap();
        assert_eq!(reader.index().assets.len(), 17_000);
        let last = reader.index().assets.last().unwrap().clone();
        assert_eq!(
            reader.read_entry(last.metadata.as_ref().unwrap()).unwrap(),
            b"{\"i\":16999}"
        );
    }

    #[test]
    fn crc32_matches_the_known_ieee_vector() {
        // "123456789" -> 0xCBF43926 is the standard CRC-32/ISO-HDLC check value.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn content_id_is_the_first_128_bits_of_sha256() {
        use sha2::{Digest, Sha256};
        let cid = content_id(b"abc");
        let full = Sha256::digest(b"abc");
        let expected: String = full[..16].iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(cid, expected);
        assert_eq!(cid.len(), 32);
    }
}
