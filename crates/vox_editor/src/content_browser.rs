//! Content Browser — a UE-style asset panel for the Ochroma editor.
//!
//! Closes row 10 of `docs/spec/unreal-gap-analysis.md` ("No content browser").
//!
//! The browser roots at a directory (e.g. `assets/`), scans it recursively for
//! splat/scene assets (`.vxm`, `.spz`, `.ply`, `.gltf`/`.glb`, `.rhai`, `.wgsl`)
//! and extracts *real* per-type metadata — splat count, file size, format
//! version — using cheap header paths wherever possible, never a full load.
//!
//! It is engine-agnostic: it produces [`LoadedAsset`] values (splats or script
//! paths) via [`ContentBrowser::load_selected`] and surfaces UI interactions as
//! a drainable [`BrowserEvent`] queue, mirroring how the node-editor widgets in
//! this crate surface their state. The editor *shell* decides what to do with a
//! loaded asset (e.g. drop the splats into the scene); the browser only exposes
//! the hook.
//!
//! Search ranking mirrors [`crate::registry`]: Prefix > Substring > Subsequence,
//! ties broken on shorter name then alphabetical, so listings are deterministic.

use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use vox_core::types::GaussianSplat;
use vox_data::ply_loader::{load_ply, PlyError};
use vox_data::spz::{load_spz, SpzError};
use vox_data::vxm::{VxmError, VxmFile};

// ---------------------------------------------------------------------------
// Asset typing
// ---------------------------------------------------------------------------

/// The kind of asset, derived from a file's extension. Drives filter chips,
/// metadata extraction and the load dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AssetKind {
    /// Native Ochroma splat container (`.vxm`).
    Vxm,
    /// Niantic SPZ compressed splats (`.spz`).
    Spz,
    /// PLY Gaussian-splat point cloud (`.ply`).
    Ply,
    /// glTF / GLB scene (`.gltf`, `.glb`).
    Gltf,
    /// Rhai gameplay script (`.rhai`).
    Rhai,
    /// WGSL shader (`.wgsl`).
    Wgsl,
}

impl AssetKind {
    /// Classify a path by its (case-insensitive) extension. Returns `None` for
    /// files that are not recognised browser assets.
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        Some(match ext.as_str() {
            "vxm" => Self::Vxm,
            "spz" => Self::Spz,
            "ply" => Self::Ply,
            "gltf" | "glb" => Self::Gltf,
            "rhai" => Self::Rhai,
            "wgsl" => Self::Wgsl,
            _ => return None,
        })
    }

    /// Short human label for the type badge / filter chip.
    pub fn label(self) -> &'static str {
        match self {
            Self::Vxm => "VXM",
            Self::Spz => "SPZ",
            Self::Ply => "PLY",
            Self::Gltf => "glTF",
            Self::Rhai => "Rhai",
            Self::Wgsl => "WGSL",
        }
    }

    /// Does this asset kind carry Gaussian splats that the browser can load into
    /// the scene (as opposed to a script/shader/scene reference)?
    pub fn is_splats(self) -> bool {
        matches!(self, Self::Vxm | Self::Spz | Self::Ply)
    }

    /// All asset kinds, in stable display order (used to build filter chips).
    pub fn all() -> [AssetKind; 6] {
        [Self::Vxm, Self::Spz, Self::Ply, Self::Gltf, Self::Rhai, Self::Wgsl]
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Cheaply-extracted, type-specific metadata about an asset. Populated from
/// header reads only — never a full splat decode.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AssetMeta {
    /// Number of Gaussian splats (vxm/spz/ply), if known from the header.
    pub splat_count: Option<u32>,
    /// Format/container version, if the header exposes one (vxm version,
    /// spz container version, glTF asset version string).
    pub version: Option<String>,
    /// Free-form extra detail surfaced as a badge (e.g. glТF asset version).
    pub detail: Option<String>,
}

/// One scanned asset: its path, kind, size, modification time and metadata.
#[derive(Debug, Clone)]
pub struct AssetEntry {
    /// Absolute (or root-relative as scanned) path to the file.
    pub path: PathBuf,
    /// Just the file name, e.g. `cube.vxm`, cached for display & search.
    pub name: String,
    /// Asset kind derived from extension.
    pub kind: AssetKind,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Last-modified time, used for the mtime-based dirty check.
    pub modified: Option<SystemTime>,
    /// Cheap header metadata.
    pub meta: AssetMeta,
}

impl AssetEntry {
    /// Splat count if this asset exposed one in its header.
    pub fn splat_count(&self) -> Option<u32> {
        self.meta.splat_count
    }
}

// ---------------------------------------------------------------------------
// Loaded asset (the engine-agnostic load result)
// ---------------------------------------------------------------------------

/// The result of loading a selected asset. Engine-agnostic: the editor shell
/// decides what to do (drop splats into the scene, open a script, etc.).
#[derive(Debug)]
pub enum LoadedAsset {
    /// Gaussian splats decoded from a vxm/spz/ply asset.
    Splats(Vec<GaussianSplat>),
    /// A script asset — the browser hands back the path; the shell loads it.
    Script(PathBuf),
    /// A shader asset — path handed back for the shell to compile/inspect.
    Shader(PathBuf),
    /// A glTF/GLB scene — path handed back for the shell's import pipeline.
    Scene(PathBuf),
}

/// Errors from scanning or loading assets.
#[derive(Debug)]
pub enum BrowserError {
    /// No asset is currently selected.
    NoSelection,
    /// I/O failure reading a file.
    Io(std::io::Error),
    /// A `.vxm` failed to decode.
    Vxm(VxmError),
    /// A `.spz` failed to decode.
    Spz(SpzError),
    /// A `.ply` failed to decode.
    Ply(PlyError),
}

impl std::fmt::Display for BrowserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSelection => write!(f, "no asset selected"),
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Vxm(e) => write!(f, "vxm error: {e}"),
            Self::Spz(e) => write!(f, "spz error: {e}"),
            Self::Ply(e) => write!(f, "ply error: {e}"),
        }
    }
}

impl std::error::Error for BrowserError {}

impl From<std::io::Error> for BrowserError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Cheap header metadata extraction
// ---------------------------------------------------------------------------

/// Read just the 64-byte `.vxm` header to learn `splat_count` and `version`
/// without decompressing the splat block.
fn vxm_meta(path: &Path) -> AssetMeta {
    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return AssetMeta::default(),
    };
    let mut hdr = [0u8; 64];
    if file.read_exact(&mut hdr).is_err() {
        return AssetMeta::default();
    }
    // Layout: magic(4) version(2) flags(2) uuid(16) splat_count(4) ...
    if &hdr[0..4] != b"VXMF" {
        return AssetMeta::default();
    }
    let version = u16::from_le_bytes([hdr[4], hdr[5]]);
    let splat_count = u32::from_le_bytes([hdr[24], hdr[25], hdr[26], hdr[27]]);
    AssetMeta {
        splat_count: Some(splat_count),
        version: Some(format!("v{version}")),
        detail: None,
    }
}

/// Read just the 16-byte SPZ header (decompressing only the gzip prefix needed)
/// to learn `numPoints` and the container version, without decoding splats.
fn spz_meta(path: &Path) -> AssetMeta {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return AssetMeta::default(),
    };
    let mut gz = flate2::read::GzDecoder::new(std::io::BufReader::new(file));
    let mut hdr = [0u8; 16];
    if gz.read_exact(&mut hdr).is_err() {
        return AssetMeta::default();
    }
    // magic(4) version(4) numPoints(4) shDegree(1) fracBits(1) flags(1) reserved(1)
    let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
    if magic != 0x5053_474e {
        return AssetMeta::default();
    }
    let version = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
    let num_points = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]);
    AssetMeta {
        splat_count: Some(num_points),
        version: Some(format!("v{version}")),
        detail: None,
    }
}

/// Parse only the PLY ASCII header to learn the vertex (splat) count without
/// reading the binary vertex block.
fn ply_meta(path: &Path) -> AssetMeta {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return AssetMeta::default(),
    };
    let mut reader = std::io::BufReader::new(file);
    let mut header_text = String::new();
    let mut byte = [0u8; 1];
    // Read until "end_header\n", bounded so a non-PLY file can't run away.
    loop {
        if reader.read_exact(&mut byte).is_err() {
            return AssetMeta::default();
        }
        header_text.push(byte[0] as char);
        // PLY headers may be LF or CRLF terminated; accept both.
        if header_text.ends_with("end_header\n") || header_text.ends_with("end_header\r\n") {
            break;
        }
        if header_text.len() > 64_000 {
            return AssetMeta::default();
        }
    }
    let mut vertex_count: Option<u32> = None;
    let mut format: Option<String> = None;
    for line in header_text.lines() {
        // Prefix-tolerant: extra trailing tokens (or a stray \r, which
        // split_whitespace also strips) must not hide the count.
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 && parts[0] == "element" && parts[1] == "vertex" {
            vertex_count = parts[2].parse().ok();
        } else if parts.len() >= 3 && parts[0] == "format" {
            format = Some(format!("{} {}", parts[1], parts[2]));
        }
    }
    AssetMeta { splat_count: vertex_count, version: None, detail: format }
}

/// Extract the glTF `asset.version` string. Handles both raw `.gltf` JSON and
/// binary `.glb` (12-byte header + JSON chunk) by reading only the JSON chunk.
fn gltf_meta(path: &Path) -> AssetMeta {
    // Header-peek contract: NEVER read the whole file — a .glb scene can be
    // hundreds of megabytes and this runs for every browser entry. Bound the
    // peek; the `asset` block sits at/near the top of well-formed glTF JSON.
    const MAX_JSON_PEEK: u64 = 256 * 1024;
    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return AssetMeta::default(),
    };
    let mut head = [0u8; 20];
    let n = match file.read(&mut head) {
        Ok(n) => n,
        Err(_) => return AssetMeta::default(),
    };
    let mut json_bytes: Vec<u8>;
    if n >= 12 && &head[0..4] == b"glTF" {
        // GLB: header(12) then first chunk = [len(4) type(4) data]. Read only
        // min(chunk_len, peek cap) bytes of the JSON chunk; a truncated file
        // yields a short read and we scan whatever arrived.
        if n < 20 {
            return AssetMeta::default();
        }
        let chunk_len =
            u64::from(u32::from_le_bytes([head[12], head[13], head[14], head[15]]));
        json_bytes = Vec::new();
        if file
            .take(chunk_len.min(MAX_JSON_PEEK))
            .read_to_end(&mut json_bytes)
            .is_err()
        {
            return AssetMeta::default();
        }
    } else {
        // Raw .gltf JSON: scan a bounded prefix only.
        json_bytes = head[..n].to_vec();
        if file
            .take(MAX_JSON_PEEK - n as u64)
            .read_to_end(&mut json_bytes)
            .is_err()
        {
            return AssetMeta::default();
        }
    }
    let version = extract_gltf_asset_version(&json_bytes);
    AssetMeta {
        splat_count: None,
        version: version.clone(),
        detail: version.map(|v| format!("glTF {v}")),
    }
}

/// Pull the `"version"` field out of the glTF JSON `"asset"` block using a tiny
/// scanner (no JSON dep needed for one field). Returns `None` if absent.
fn extract_gltf_asset_version(json: &[u8]) -> Option<String> {
    // Lossy: a bounded peek may cut a multibyte char at the tail; the asset
    // block we scan for is pure ASCII either way.
    let text = String::from_utf8_lossy(json);
    let text: &str = &text;
    let asset_at = text.find("\"asset\"")?;
    let rest = &text[asset_at..];
    let ver_at = rest.find("\"version\"")?;
    let after = &rest[ver_at + "\"version\"".len()..];
    let colon = after.find(':')?;
    let after_colon = &after[colon + 1..];
    let q1 = after_colon.find('"')?;
    let tail = &after_colon[q1 + 1..];
    let q2 = tail.find('"')?;
    Some(tail[..q2].to_string())
}

/// Dispatch metadata extraction by asset kind. Scripts/shaders have no header
/// metadata, so they get an empty [`AssetMeta`].
fn extract_meta(kind: AssetKind, path: &Path) -> AssetMeta {
    match kind {
        AssetKind::Vxm => vxm_meta(path),
        AssetKind::Spz => spz_meta(path),
        AssetKind::Ply => ply_meta(path),
        AssetKind::Gltf => gltf_meta(path),
        AssetKind::Rhai | AssetKind::Wgsl => AssetMeta::default(),
    }
}

// ---------------------------------------------------------------------------
// Search ranking (mirrors crate::registry)
// ---------------------------------------------------------------------------

/// Tier of a search match — lower ranks first. Mirrors `crate::registry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatchTier {
    Prefix = 0,
    Substring = 1,
    Subsequence = 2,
}

/// Is `needle` a subsequence of `haystack` (chars in order, gaps allowed)?
fn is_subsequence(needle: &str, haystack: &str) -> bool {
    let mut hay = haystack.chars();
    for nc in needle.chars() {
        loop {
            match hay.next() {
                Some(hc) if hc == nc => break,
                Some(_) => continue,
                None => return false,
            }
        }
    }
    true
}

/// Classify `name` against `query` into a match tier, or `None` for no match.
/// Case-insensitive. Empty query matches everything at the `Prefix` tier.
fn match_tier(name: &str, query: &str) -> Option<MatchTier> {
    if query.is_empty() {
        return Some(MatchTier::Prefix);
    }
    let name_lc = name.to_ascii_lowercase();
    if name_lc.starts_with(query) {
        Some(MatchTier::Prefix)
    } else if name_lc.contains(query) {
        Some(MatchTier::Substring)
    } else if is_subsequence(query, &name_lc) {
        Some(MatchTier::Subsequence)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// ContentBrowser state
// ---------------------------------------------------------------------------

/// The content browser model: scans a root directory, holds the asset listing,
/// current filter/search/selection, and an mtime fingerprint for cheap
/// change-detection.
pub struct ContentBrowser {
    root: PathBuf,
    entries: Vec<AssetEntry>,
    /// Active type filter — `None` means "all types".
    pub filter: Option<AssetKind>,
    /// Substring search query (lower-cased on apply).
    pub search: String,
    /// Index into `entries` of the selected asset, if any.
    selected: Option<usize>,
    /// Sum of (path-hash, mtime) used to detect external changes cheaply.
    fingerprint: u64,
}

impl ContentBrowser {
    /// Create a browser rooted at `root` and perform the initial scan.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let mut b = Self {
            root: root.into(),
            entries: Vec::new(),
            filter: None,
            search: String::new(),
            selected: None,
            fingerprint: 0,
        };
        b.refresh();
        b
    }

    /// The root directory being browsed.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// All scanned entries (unfiltered), in scan order.
    pub fn entries(&self) -> &[AssetEntry] {
        &self.entries
    }

    /// Re-scan the root directory from scratch, rebuilding the listing and the
    /// change-detection fingerprint. Selection is preserved by path if the
    /// selected asset still exists.
    pub fn refresh(&mut self) {
        let prev_selected_path = self.selected_entry().map(|e| e.path.clone());

        let mut entries = Vec::new();
        let mut fingerprint: u64 = 0;
        scan_dir(&self.root, &mut entries, &mut fingerprint);
        // Stable order: by path, so listings are deterministic across runs.
        entries.sort_by(|a, b| a.path.cmp(&b.path));

        self.entries = entries;
        self.fingerprint = fingerprint;

        // Re-resolve the selection against the new listing.
        self.selected = prev_selected_path
            .and_then(|p| self.entries.iter().position(|e| e.path == p));
    }

    /// Cheap mtime-based dirty check: re-walks the tree computing only the
    /// fingerprint (no header reads) and reports whether it differs from the
    /// last scan. Call [`refresh`](Self::refresh) when this returns `true`.
    pub fn is_dirty(&self) -> bool {
        let mut fingerprint: u64 = 0;
        fingerprint_dir(&self.root, &mut fingerprint);
        fingerprint != self.fingerprint
    }

    /// The set of distinct folders (relative to root) containing assets — the
    /// data behind a folder tree. Always includes the root itself ("").
    pub fn folders(&self) -> Vec<PathBuf> {
        let mut set: BTreeSet<PathBuf> = BTreeSet::new();
        set.insert(PathBuf::new());
        for e in &self.entries {
            if let Some(parent) = e.path.parent() {
                if let Ok(rel) = parent.strip_prefix(&self.root) {
                    // Insert this folder and every ancestor up to the root.
                    let mut acc = PathBuf::new();
                    for comp in rel.components() {
                        acc.push(comp);
                        set.insert(acc.clone());
                    }
                }
            }
        }
        set.into_iter().collect()
    }

    /// Breadcrumb components for `path` relative to the root, e.g.
    /// `assets/props/cube.vxm` under root `assets` -> `["props", "cube.vxm"]`.
    pub fn breadcrumbs(&self, path: &Path) -> Vec<String> {
        let rel = path.strip_prefix(&self.root).unwrap_or(path);
        rel.components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect()
    }

    /// Entries passing the current type filter and search query, ranked
    /// best-match first (Prefix > Substring > Subsequence; ties: shorter name
    /// then alphabetical). Returns references into [`entries`](Self::entries).
    pub fn visible(&self) -> Vec<&AssetEntry> {
        let q = self.search.to_ascii_lowercase();
        let mut hits: Vec<(&AssetEntry, MatchTier)> = self
            .entries
            .iter()
            .filter(|e| self.filter.is_none_or(|f| e.kind == f))
            .filter_map(|e| match_tier(&e.name, &q).map(|t| (e, t)))
            .collect();
        hits.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.0.name.len().cmp(&b.0.name.len()))
                .then_with(|| a.0.name.cmp(&b.0.name))
        });
        hits.into_iter().map(|(e, _)| e).collect()
    }

    /// Set the active type filter (or clear it with `None`). Clears selection if
    /// the previously-selected asset no longer passes the filter.
    pub fn set_filter(&mut self, filter: Option<AssetKind>) {
        self.filter = filter;
        if let (Some(idx), Some(f)) = (self.selected, filter) {
            if self.entries[idx].kind != f {
                self.selected = None;
            }
        }
    }

    /// Set the substring search query.
    pub fn set_search(&mut self, query: impl Into<String>) {
        self.search = query.into();
    }

    /// Select the asset at the given path. Returns `true` if it was found.
    pub fn select_path(&mut self, path: &Path) -> bool {
        match self.entries.iter().position(|e| e.path == path) {
            Some(idx) => {
                self.selected = Some(idx);
                true
            }
            None => false,
        }
    }

    /// The currently-selected entry, if any.
    pub fn selected_entry(&self) -> Option<&AssetEntry> {
        self.selected.and_then(|i| self.entries.get(i))
    }

    /// Load the currently-selected asset into an engine-agnostic [`LoadedAsset`].
    ///
    /// Splat assets (vxm/spz/ply) decode to [`LoadedAsset::Splats`] via
    /// `vox_data`'s real loaders; scripts/shaders/scenes hand back their path.
    pub fn load_selected(&self) -> Result<LoadedAsset, BrowserError> {
        let entry = self.selected_entry().ok_or(BrowserError::NoSelection)?;
        load_asset(&entry.path, entry.kind)
    }
}

/// Load any asset by path + kind into a [`LoadedAsset`] (free function so the
/// shell can load an event's path without a live selection).
pub fn load_asset(path: &Path, kind: AssetKind) -> Result<LoadedAsset, BrowserError> {
    match kind {
        AssetKind::Vxm => {
            let file = fs::File::open(path)?;
            let vxm = VxmFile::read(std::io::BufReader::new(file)).map_err(BrowserError::Vxm)?;
            Ok(LoadedAsset::Splats(vxm.splats))
        }
        AssetKind::Spz => {
            let splats = load_spz(path).map_err(BrowserError::Spz)?;
            Ok(LoadedAsset::Splats(splats))
        }
        AssetKind::Ply => {
            let splats = load_ply(path).map_err(BrowserError::Ply)?;
            Ok(LoadedAsset::Splats(splats))
        }
        AssetKind::Gltf => Ok(LoadedAsset::Scene(path.to_path_buf())),
        AssetKind::Rhai => Ok(LoadedAsset::Script(path.to_path_buf())),
        AssetKind::Wgsl => Ok(LoadedAsset::Shader(path.to_path_buf())),
    }
}

// ---------------------------------------------------------------------------
// Directory scanning
// ---------------------------------------------------------------------------

/// Combine a path and an mtime into the running fingerprint. A cheap, order-
/// independent hash (xor of per-file hashes) so the walk order doesn't matter.
fn mix_fingerprint(acc: &mut u64, path: &Path, modified: Option<SystemTime>) {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut h);
    if let Some(m) = modified {
        if let Ok(dur) = m.duration_since(SystemTime::UNIX_EPOCH) {
            dur.as_nanos().hash(&mut h);
        }
    }
    *acc ^= h.finish();
}

/// Recursively scan `dir` for asset files, pushing entries and mixing each into
/// the fingerprint. Non-asset files are skipped.
fn scan_dir(dir: &Path, out: &mut Vec<AssetEntry>, fingerprint: &mut u64) {
    let Ok(read) = fs::read_dir(dir) else { return };
    for entry in read.flatten() {
        let path = entry.path();
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_dir() {
            scan_dir(&path, out, fingerprint);
            continue;
        }
        let Some(kind) = AssetKind::from_path(&path) else { continue };
        let metadata = entry.metadata().ok();
        let size_bytes = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
        let modified = metadata.as_ref().and_then(|m| m.modified().ok());
        mix_fingerprint(fingerprint, &path, modified);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let meta = extract_meta(kind, &path);
        out.push(AssetEntry { path, name, kind, size_bytes, modified, meta });
    }
}

/// Like [`scan_dir`] but only mixes the fingerprint (no header reads / alloc) —
/// the cheap path for [`ContentBrowser::is_dirty`].
fn fingerprint_dir(dir: &Path, fingerprint: &mut u64) {
    let Ok(read) = fs::read_dir(dir) else { return };
    for entry in read.flatten() {
        let path = entry.path();
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_dir() {
            fingerprint_dir(&path, fingerprint);
            continue;
        }
        if AssetKind::from_path(&path).is_none() {
            continue;
        }
        let modified = entry.metadata().ok().and_then(|m| m.modified().ok());
        mix_fingerprint(fingerprint, &path, modified);
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// A UI interaction surfaced by [`ContentBrowserPanel::ui`] for the caller to
/// drain and act on (mirrors how the node-editor widgets surface state).
#[derive(Debug, Clone, PartialEq)]
pub enum BrowserEvent {
    /// An asset was single-clicked (now the selection).
    Selected(PathBuf),
    /// An asset was double-clicked / activated — the shell should load+place it.
    Activated { path: PathBuf, kind: AssetKind },
    /// The user toggled the type filter.
    FilterChanged(Option<AssetKind>),
}

// ---------------------------------------------------------------------------
// egui panel
// ---------------------------------------------------------------------------

/// View mode for the asset listing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    /// Vertical list with metadata badges per row.
    List,
    /// Wrapped grid of compact cards.
    Grid,
}

/// egui panel wrapping a [`ContentBrowser`]. Renders breadcrumbs, filter chips,
/// a search box, a list/grid toggle and the asset listing with metadata badges.
/// UI interactions are pushed into [`events`](Self::events) for the caller to
/// drain via [`take_events`](Self::take_events).
pub struct ContentBrowserPanel {
    pub browser: ContentBrowser,
    pub view: ViewMode,
    events: Vec<BrowserEvent>,
}

impl ContentBrowserPanel {
    /// Wrap a browser rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            browser: ContentBrowser::new(root),
            view: ViewMode::List,
            events: Vec::new(),
        }
    }

    /// Drain queued UI events. The caller acts on these (e.g. on `Activated`,
    /// call [`load_asset`] and drop the splats into the scene).
    pub fn take_events(&mut self) -> Vec<BrowserEvent> {
        std::mem::take(&mut self.events)
    }

    /// Human-readable byte size, e.g. `12.0 KB`.
    fn human_size(bytes: u64) -> String {
        const KB: f64 = 1024.0;
        const MB: f64 = KB * 1024.0;
        let b = bytes as f64;
        if b >= MB {
            format!("{:.1} MB", b / MB)
        } else if b >= KB {
            format!("{:.1} KB", b / KB)
        } else {
            format!("{bytes} B")
        }
    }

    /// One line of metadata badges for an entry.
    fn badges(entry: &AssetEntry) -> String {
        let mut parts = vec![entry.kind.label().to_string()];
        if let Some(n) = entry.meta.splat_count {
            parts.push(format!("{n} splats"));
        }
        if let Some(v) = &entry.meta.version {
            parts.push(v.clone());
        }
        parts.push(Self::human_size(entry.size_bytes));
        parts.join("  ·  ")
    }

    /// Render the content browser panel into `ui`.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Content Browser");
        ui.label(self.browser.root().display().to_string());
        ui.separator();

        // --- Type filter chips -------------------------------------------------
        ui.horizontal_wrapped(|ui| {
            let all_selected = self.browser.filter.is_none();
            if ui.selectable_label(all_selected, "All").clicked() {
                self.browser.set_filter(None);
                self.events.push(BrowserEvent::FilterChanged(None));
            }
            for kind in AssetKind::all() {
                let sel = self.browser.filter == Some(kind);
                if ui.selectable_label(sel, kind.label()).clicked() {
                    let new = if sel { None } else { Some(kind) };
                    self.browser.set_filter(new);
                    self.events.push(BrowserEvent::FilterChanged(new));
                }
            }
        });

        // --- Search + view toggle ---------------------------------------------
        ui.horizontal(|ui| {
            ui.label("Search:");
            let mut q = self.browser.search.clone();
            if ui.text_edit_singleline(&mut q).changed() {
                self.browser.set_search(q);
            }
            ui.separator();
            if ui.selectable_label(self.view == ViewMode::List, "List").clicked() {
                self.view = ViewMode::List;
            }
            if ui.selectable_label(self.view == ViewMode::Grid, "Grid").clicked() {
                self.view = ViewMode::Grid;
            }
            if ui.button("Refresh").clicked() {
                self.browser.refresh();
            }
        });
        ui.separator();

        // Gather what we need before borrowing self mutably for events.
        let selected_path = self.browser.selected_entry().map(|e| e.path.clone());
        let visible: Vec<(PathBuf, String, AssetKind, String)> = self
            .browser
            .visible()
            .into_iter()
            .map(|e| (e.path.clone(), e.name.clone(), e.kind, Self::badges(e)))
            .collect();

        let mut clicked: Option<PathBuf> = None;
        let mut activated: Option<(PathBuf, AssetKind)> = None;

        egui::ScrollArea::vertical().show(ui, |ui| match self.view {
            ViewMode::List => {
                for (path, name, kind, badges) in &visible {
                    let is_sel = selected_path.as_deref() == Some(path.as_path());
                    let resp = ui.selectable_label(is_sel, format!("{name}\n{badges}"));
                    if resp.clicked() {
                        clicked = Some(path.clone());
                    }
                    if resp.double_clicked() {
                        activated = Some((path.clone(), *kind));
                    }
                }
            }
            ViewMode::Grid => {
                ui.horizontal_wrapped(|ui| {
                    for (path, name, kind, badges) in &visible {
                        let is_sel = selected_path.as_deref() == Some(path.as_path());
                        let resp = ui.selectable_label(is_sel, format!("{name}\n{badges}"));
                        if resp.clicked() {
                            clicked = Some(path.clone());
                        }
                        if resp.double_clicked() {
                            activated = Some((path.clone(), *kind));
                        }
                    }
                });
            }
        });

        if let Some(path) = clicked {
            self.browser.select_path(&path);
            self.events.push(BrowserEvent::Selected(path));
        }
        if let Some((path, kind)) = activated {
            self.browser.select_path(&path);
            self.events.push(BrowserEvent::Activated { path, kind });
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;
    use std::io::Write;
    use vox_data::spz::write_spz;
    use vox_data::vxm::VxmFileV3;

    /// Build a deterministic volume splat at a known position.
    fn splat_at(pos: [f32; 3]) -> GaussianSplat {
        GaussianSplat::volume(pos, [0.1, 0.1, 0.1], Quat::IDENTITY, 200, [0u16; 16])
    }

    /// Write a real `.vxm` (V3) with `n` splats; the first is at `first_pos`.
    fn write_vxm(path: &Path, n: usize, first_pos: [f32; 3]) {
        let mut splats = vec![splat_at(first_pos)];
        for i in 1..n {
            splats.push(splat_at([i as f32, i as f32 * 2.0, i as f32 * 3.0]));
        }
        let file = VxmFileV3 { splats, material_ids: Vec::new(), spectral_level: 1 };
        let mut buf = Vec::new();
        file.write(&mut buf).expect("vxm write");
        fs::write(path, buf).expect("vxm to disk");
    }

    /// A unique temp dir for a test (no external tempfile dep).
    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ochroma_cb_{tag}_{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn scan_finds_only_known_assets_with_real_metadata() {
        let dir = temp_dir("scan");
        write_vxm(&dir.join("cube.vxm"), 7, [1.5, -2.0, 3.25]);
        write_spz(&dir.join("cloud.spz"), &[splat_at([0.0, 0.0, 0.0]); 4]).expect("spz");
        fs::write(dir.join("notes.txt"), b"junk, not an asset").unwrap();

        let browser = ContentBrowser::new(&dir);
        let entries = browser.entries();

        // Exactly the two real assets; the .txt is absent.
        assert_eq!(entries.len(), 2, "should find exactly the vxm and spz");
        assert!(
            !entries.iter().any(|e| e.name == "notes.txt"),
            "the .txt junk file must not be in the listing"
        );

        let vxm = entries.iter().find(|e| e.name == "cube.vxm").expect("vxm entry");
        assert_eq!(vxm.kind, AssetKind::Vxm);
        assert_eq!(vxm.splat_count(), Some(7), "vxm header must report 7 splats");

        let spz = entries.iter().find(|e| e.name == "cloud.spz").expect("spz entry");
        assert_eq!(spz.kind, AssetKind::Spz);
        assert_eq!(spz.splat_count(), Some(4), "spz header must report 4 splats");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_cub_ranks_prefix_first_subsequence_last() {
        let dir = temp_dir("search");
        // cube.vxm (prefix "cub"), cubemap.spz (prefix "cub"), scuba.ply (subseq).
        write_vxm(&dir.join("cube.vxm"), 3, [0.0, 0.0, 0.0]);
        write_spz(&dir.join("cubemap.spz"), &[splat_at([0.0, 0.0, 0.0]); 2]).expect("spz");
        // A minimal binary-LE PLY with a header carrying 1 vertex, x/y/z floats.
        write_min_ply(&dir.join("scuba.ply"), 1);

        let mut browser = ContentBrowser::new(&dir);
        browser.set_search("cub");
        let names: Vec<&str> = browser.visible().iter().map(|e| e.name.as_str()).collect();

        assert_eq!(names.len(), 3, "all three names match 'cub' (cube/cubemap prefix, scuba subseq)");
        assert_eq!(names[0], "cube.vxm", "shortest prefix match ranks first");
        assert_eq!(names[1], "cubemap.spz", "longer prefix match ranks second");
        assert_eq!(names[2], "scuba.ply", "subsequence match ranks last");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_selected_vxm_returns_splats_with_roundtripped_position() {
        let dir = temp_dir("load");
        let first = [1.5_f32, -2.0, 3.25];
        write_vxm(&dir.join("cube.vxm"), 7, first);

        let mut browser = ContentBrowser::new(&dir);
        assert!(browser.select_path(&dir.join("cube.vxm")), "selecting the vxm");

        match browser.load_selected().expect("load should succeed") {
            LoadedAsset::Splats(splats) => {
                assert_eq!(splats.len(), 7, "loaded splat count must equal written count");
                let p = splats[0].position();
                // Native vxm splats round-trip position exactly (no quantization).
                assert_eq!(p, first, "first splat position must round-trip exactly");
            }
            other => panic!("expected Splats, got {other:?}"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn filter_by_spz_keeps_only_spz() {
        let dir = temp_dir("filter");
        write_vxm(&dir.join("cube.vxm"), 5, [0.0, 0.0, 0.0]);
        write_spz(&dir.join("cloud.spz"), &[splat_at([0.0, 0.0, 0.0]); 3]).expect("spz");

        let mut browser = ContentBrowser::new(&dir);
        browser.set_filter(Some(AssetKind::Spz));
        let visible: Vec<&AssetEntry> = browser.visible();

        assert_eq!(visible.len(), 1, "only the spz passes a Spz filter");
        assert_eq!(visible[0].name, "cloud.spz");
        assert_eq!(visible[0].kind, AssetKind::Spz);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn recursive_scan_finds_nested_assets() {
        let dir = temp_dir("nested");
        let sub = dir.join("props/small");
        fs::create_dir_all(&sub).unwrap();
        write_vxm(&dir.join("top.vxm"), 2, [0.0, 0.0, 0.0]);
        write_vxm(&sub.join("deep.vxm"), 9, [4.0, 5.0, 6.0]);

        let browser = ContentBrowser::new(&dir);
        assert_eq!(browser.entries().len(), 2, "scan must recurse into subfolders");
        let deep = browser
            .entries()
            .iter()
            .find(|e| e.name == "deep.vxm")
            .expect("nested asset found");
        assert_eq!(deep.splat_count(), Some(9));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dirty_check_detects_new_asset_then_refresh_clears_it() {
        let dir = temp_dir("dirty");
        write_vxm(&dir.join("a.vxm"), 1, [0.0, 0.0, 0.0]);
        let mut browser = ContentBrowser::new(&dir);
        assert!(!browser.is_dirty(), "fresh scan is not dirty");

        write_vxm(&dir.join("b.vxm"), 2, [0.0, 0.0, 0.0]);
        assert!(browser.is_dirty(), "adding a file must make the browser dirty");

        browser.refresh();
        assert!(!browser.is_dirty(), "refresh re-syncs the fingerprint");
        assert_eq!(browser.entries().len(), 2, "refresh picks up the new asset");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gltf_metadata_reports_asset_version() {
        let dir = temp_dir("gltf");
        let json = br#"{"asset":{"version":"2.0","generator":"ochroma-test"},"scenes":[]}"#;
        fs::write(dir.join("scene.gltf"), json).unwrap();

        let browser = ContentBrowser::new(&dir);
        let e = browser.entries().iter().find(|e| e.name == "scene.gltf").expect("gltf entry");
        assert_eq!(e.kind, AssetKind::Gltf);
        assert_eq!(e.meta.version.as_deref(), Some("2.0"), "glTF asset version must be parsed");

        fs::remove_dir_all(&dir).ok();
    }

    /// CRLF-terminated PLY headers (Windows-authored files) must still yield
    /// the vertex count — the terminator and line parsing accept both endings.
    #[test]
    fn crlf_ply_header_reports_vertex_count() {
        let dir = temp_dir("plycrlf");
        let header = "ply\r\nformat binary_little_endian 1.0\r\nelement vertex 42\r\nproperty float x\r\nend_header\r\n";
        fs::write(dir.join("win.ply"), header.as_bytes()).unwrap();

        let browser = ContentBrowser::new(&dir);
        let e = browser.entries().iter().find(|e| e.name == "win.ply").expect("ply entry");
        assert_eq!(e.meta.splat_count, Some(42), "CRLF header must still parse the count");
        assert_eq!(
            e.meta.detail.as_deref(),
            Some("binary_little_endian 1.0"),
            "CRLF format line must still parse"
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// gltf_meta is a bounded header peek: a raw .gltf far larger than the
    /// 256 KiB peek window still yields the version (asset block is at the
    /// top), proving the scan does not require — and the parse does not choke
    /// on — the full file.
    #[test]
    fn huge_gltf_version_found_via_bounded_peek() {
        let dir = temp_dir("gltfbig");
        let mut json = String::from(r#"{"asset":{"version":"2.0"},"scenes":[{"name":""#);
        json.push_str(&"x".repeat(600 * 1024)); // body far beyond the peek cap
        json.push_str(r#""}]}"#);
        fs::write(dir.join("big.gltf"), json.as_bytes()).unwrap();

        let browser = ContentBrowser::new(&dir);
        let e = browser.entries().iter().find(|e| e.name == "big.gltf").expect("entry");
        assert_eq!(e.meta.version.as_deref(), Some("2.0"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_selected_without_selection_errors() {
        let dir = temp_dir("nosel");
        let browser = ContentBrowser::new(&dir);
        assert!(
            matches!(browser.load_selected(), Err(BrowserError::NoSelection)),
            "loading with nothing selected must report NoSelection"
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// Write a minimal binary-little-endian PLY with `n` vertices (x,y,z floats).
    fn write_min_ply(path: &Path, n: usize) {
        let mut f = fs::File::create(path).unwrap();
        let header = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex {n}\n\
             property float x\nproperty float y\nproperty float z\nend_header\n"
        );
        f.write_all(header.as_bytes()).unwrap();
        for i in 0..n {
            for c in [i as f32, i as f32, i as f32] {
                f.write_all(&c.to_le_bytes()).unwrap();
            }
        }
    }
}
