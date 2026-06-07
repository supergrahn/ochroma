//! Disk tile streamer — the service layer that actually loads world-partition
//! tiles off disk on background threads.
//!
//! `vox_render::world_partition::WorldPartition` decides *which* tiles to load
//! each frame (`compute_streaming_state` → to_load / to_hlod / to_unload, plus a
//! `CellLoadRequest` priority queue) and stores a `splat_tile_path` per cell — but
//! it never touches the disk. This module is that missing half, and is engine-
//! generic: tiles are keyed by an opaque [`TileKey`] (an `(i32, i32, i32)` tuple,
//! so a `CellCoord` adapts trivially) so nothing here depends on vox_render.
//!
//! Generation / cancellation discipline (how a camera teleport cancels obsolete
//! work): every [`TileStreamer::request`] carries a `generation`. When the camera
//! jumps, the driver bumps [`TileStreamer::set_min_generation`]; any queued
//! request older than that is reported as [`StreamError::Stale`] *without ever
//! opening the file*, and any in-flight completion from a stale generation is
//! dropped in [`TileStreamer::drain_completed`]. So obsolete loads cost no I/O.
//!
//! Expected driver loop (in vox_render's `WorldPartition`):
//!   * for each coord in `compute_streaming_state(..).0` (to_load): call
//!     `streamer.request(coord.into(), cell.splat_tile_path.clone(), priority, gen)`;
//!   * on a camera teleport: `streamer.set_min_generation(new_gen)`;
//!   * each frame: feed `drain_completed()` into `WorldPartition::complete_load`
//!     and run [`ResidencyCache::evict_to_budget`] with `SplatBudget.max_gpu_mb`.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::vxm::VxmError;
use crate::vxm_v2::{GaussianSplatV2, VxmFileV2};

// ---------------------------------------------------------------------------
// TileKey
// ---------------------------------------------------------------------------

/// Opaque key identifying a tile. Engine-generic on purpose: a vox_render
/// `CellCoord(i32, i32, i32)` maps onto it directly so the streamer never has to
/// know about cells, worlds, or partitions.
pub type TileKey = (i32, i32, i32);

// ---------------------------------------------------------------------------
// StreamError
// ---------------------------------------------------------------------------

/// Why a tile load did not produce a usable [`VxmFileV2`].
///
/// This is a self-contained, `Clone`-able error (unlike `VxmError`, which wraps a
/// non-`Clone` `io::Error`) so it can be copied into observability counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    /// Request was older than the streamer's minimum generation — never read.
    Stale,
    /// Disk I/O / open failed after all retry attempts (message is the last error).
    Io(String),
    /// File opened but failed to parse as a v2 tile.
    Decode(String),
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::Stale => write!(f, "load cancelled: stale generation"),
            StreamError::Io(m) => write!(f, "io error: {m}"),
            StreamError::Decode(m) => write!(f, "decode error: {m}"),
        }
    }
}

impl std::error::Error for StreamError {}

// ---------------------------------------------------------------------------
// LoadedTile
// ---------------------------------------------------------------------------

/// Result of a single tile-load attempt, handed back via
/// [`TileStreamer::drain_completed`].
pub struct LoadedTile {
    pub key: TileKey,
    pub generation: u64,
    /// Number of open attempts actually made (1 + retries on I/O error; 0 if
    /// skipped as stale). Exposed for observability and to prove retry behavior.
    pub attempts: u32,
    pub result: Result<VxmFileV2, StreamError>,
}

// ---------------------------------------------------------------------------
// StreamerConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`TileStreamer`].
#[derive(Debug, Clone, Copy)]
pub struct StreamerConfig {
    /// Number of background worker threads. `0` is a valid, test-only value:
    /// no threads are spawned and work is processed via [`TileStreamer::step_one`].
    pub workers: usize,
    /// Number of *retries* after the first failed open (so total attempts on a
    /// hard I/O failure = `max_retries + 1`).
    pub max_retries: u32,
}

impl Default for StreamerConfig {
    fn default() -> Self {
        Self { workers: 2, max_retries: 1 }
    }
}

// ---------------------------------------------------------------------------
// Request — internal priority-queue entry
// ---------------------------------------------------------------------------

/// Internal queue entry. Priority is stored as `f32::to_bits()` (mirroring
/// `world_partition::CellLoadRequest`) so a `BinaryHeap` orders highest-priority
/// first without pulling in `ordered_float`.
struct Request {
    key: TileKey,
    path: PathBuf,
    priority_bits: u32,
    generation: u64,
    /// Tie-break: lower sequence (enqueued earlier) wins among equal priorities.
    seq: u64,
}

impl PartialEq for Request {
    fn eq(&self, o: &Self) -> bool {
        self.priority_bits == o.priority_bits && self.seq == o.seq
    }
}
impl Eq for Request {}
impl PartialOrd for Request {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Request {
    fn cmp(&self, o: &Self) -> Ordering {
        // Higher priority first; on a tie, lower seq first (so flip seq order).
        self.priority_bits
            .cmp(&o.priority_bits)
            .then_with(|| o.seq.cmp(&self.seq))
    }
}

// ---------------------------------------------------------------------------
// Shared queue state
// ---------------------------------------------------------------------------

struct Shared {
    queue: Mutex<QueueState>,
    /// Signalled when a request is enqueued or shutdown is requested.
    cvar: Condvar,
    /// Min generation gate; requests older than this are skipped without I/O.
    min_generation: AtomicU64,
    /// Completed loads, ready to be drained. Separate lock from the queue so a
    /// worker finishing a load never blocks an enqueue.
    completed: Mutex<Vec<LoadedTile>>,
}

struct QueueState {
    heap: BinaryHeap<Request>,
    next_seq: u64,
    shutdown: bool,
}

// ---------------------------------------------------------------------------
// TileStreamer
// ---------------------------------------------------------------------------

/// Background, priority-ordered, cancellable disk loader for `.vxm` v2 tiles.
pub struct TileStreamer {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    config: StreamerConfig,
}

impl TileStreamer {
    /// Create a streamer and spawn `config.workers` background loader threads.
    pub fn new(config: StreamerConfig) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(QueueState {
                heap: BinaryHeap::new(),
                next_seq: 0,
                shutdown: false,
            }),
            cvar: Condvar::new(),
            min_generation: AtomicU64::new(0),
            completed: Mutex::new(Vec::new()),
        });

        let mut workers = Vec::with_capacity(config.workers);
        for _ in 0..config.workers {
            let shared = Arc::clone(&shared);
            let max_retries = config.max_retries;
            workers.push(std::thread::spawn(move || worker_loop(shared, max_retries)));
        }

        Self { shared, workers, config }
    }

    /// Enqueue a tile load. Workers pick the highest-`priority` request first.
    /// `generation` participates in the cancellation discipline (see module docs).
    pub fn request(&self, key: TileKey, path: PathBuf, priority: f32, generation: u64) {
        let mut q = self.shared.queue.lock().unwrap();
        let seq = q.next_seq;
        q.next_seq += 1;
        q.heap.push(Request {
            key,
            path,
            priority_bits: priority.to_bits(),
            generation,
            seq,
        });
        drop(q);
        self.shared.cvar.notify_one();
    }

    /// Raise the minimum generation. Queued requests older than `gen` are reported
    /// as [`StreamError::Stale`] without any disk I/O; already-completed loads from
    /// older generations are dropped on the next [`Self::drain_completed`].
    pub fn set_min_generation(&self, generation: u64) {
        self.shared.min_generation.store(generation, AtomicOrdering::SeqCst);
    }

    /// Current minimum generation gate.
    pub fn min_generation(&self) -> u64 {
        self.shared.min_generation.load(AtomicOrdering::SeqCst)
    }

    /// Non-blocking: take all completed loads. An *obsolete success* — a tile that
    /// finished loading before a teleport bumped the generation past it — is
    /// downgraded in place to `Err(StreamError::Stale)` so the driver never applies
    /// stale geometry, while still seeing the slot resolve. Entries already failed
    /// (including those skipped as `Stale` by a worker) pass through unchanged so
    /// cancellation is observable.
    pub fn drain_completed(&self) -> Vec<LoadedTile> {
        let min_gen = self.shared.min_generation.load(AtomicOrdering::SeqCst);
        let mut done = self.shared.completed.lock().unwrap();
        let drained = std::mem::take(&mut *done);
        drop(done);
        drained
            .into_iter()
            .map(|mut t| {
                if t.generation < min_gen && t.result.is_ok() {
                    t.result = Err(StreamError::Stale);
                }
                t
            })
            .collect()
    }

    /// Test-only synchronous hook: process exactly one queued request inline on the
    /// calling thread (highest priority first), pushing its result into the
    /// completed buffer. Returns `false` if the queue was empty. Intended for
    /// deterministic `workers == 0` tests of priority ordering and staleness.
    pub fn step_one(&self) -> bool {
        let req = {
            let mut q = self.shared.queue.lock().unwrap();
            q.heap.pop()
        };
        match req {
            Some(req) => {
                let tile = process_request(&self.shared, req, self.config.max_retries);
                self.shared.completed.lock().unwrap().push(tile);
                true
            }
            None => false,
        }
    }
}

impl Drop for TileStreamer {
    fn drop(&mut self) {
        // Signal shutdown and wake every worker, then join — no detached threads.
        {
            let mut q = self.shared.queue.lock().unwrap();
            q.shutdown = true;
        }
        self.shared.cvar.notify_all();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Worker loop + request processing
// ---------------------------------------------------------------------------

fn worker_loop(shared: Arc<Shared>, max_retries: u32) {
    loop {
        let req = {
            let mut q = shared.queue.lock().unwrap();
            loop {
                if q.shutdown {
                    return;
                }
                if let Some(req) = q.heap.pop() {
                    break req;
                }
                // No work: wait until notified (enqueue or shutdown).
                q = shared.cvar.wait(q).unwrap();
            }
        };

        let tile = process_request(&shared, req, max_retries);
        shared.completed.lock().unwrap().push(tile);
    }
}

/// Load a single request, honoring the generation gate and retry policy.
fn process_request(shared: &Shared, req: Request, max_retries: u32) -> LoadedTile {
    // Generation gate: skip stale requests without touching disk.
    let min_gen = shared.min_generation.load(AtomicOrdering::SeqCst);
    if req.generation < min_gen {
        return LoadedTile {
            key: req.key,
            generation: req.generation,
            attempts: 0,
            result: Err(StreamError::Stale),
        };
    }

    let (attempts, result) = load_tile_with_retries(&req.path, max_retries);
    LoadedTile { key: req.key, generation: req.generation, attempts, result }
}

/// Open + decode a tile, retrying on I/O errors up to `max_retries` extra times.
/// Returns the number of attempts actually made alongside the result.
///
/// Retry policy by failure class:
///   * an `open()` failure is *transient* I/O — retried;
///   * a read failure mid-stream (e.g. a truncated / mid-write / partially
///     copied file → `VxmError::Io(UnexpectedEof)`) is *transient* I/O too —
///     it is retried, and a final failure is reported as [`StreamError::Io`];
///   * a genuine *content* error (bad magic, unsupported version, bad
///     decompression / alignment → any non-`Io` `VxmError`) is terminal and
///     reported as [`StreamError::Decode`] — retrying a corrupt file would only
///     waste I/O.
fn load_tile_with_retries(
    path: &Path,
    max_retries: u32,
) -> (u32, Result<VxmFileV2, StreamError>) {
    let total_attempts = max_retries + 1;
    let mut last_io_err = String::new();

    for attempt in 1..=total_attempts {
        match std::fs::File::open(path) {
            Ok(file) => {
                match VxmFileV2::read(std::io::BufReader::new(file)) {
                    Ok(tile) => return (attempt, Ok(tile)),
                    // Truncated / partial read is transient I/O: record it and
                    // fall through to retry like an open() failure.
                    Err(VxmError::Io(e)) => last_io_err = e.to_string(),
                    // Real content error: terminal, no retry.
                    Err(e) => return (attempt, Err(StreamError::Decode(e.to_string()))),
                }
            }
            Err(e) => {
                last_io_err = e.to_string();
            }
        }
    }

    (total_attempts, Err(StreamError::Io(last_io_err)))
}

// ---------------------------------------------------------------------------
// ResidencyCache — synchronous LRU
// ---------------------------------------------------------------------------

/// A loaded tile resident in the cache, with its exact byte size.
struct CacheEntry {
    tile: VxmFileV2,
    size_bytes: usize,
    /// Monotonic touch stamp; higher = more recently used.
    last_touch: u64,
}

/// Synchronous LRU cache over loaded tiles, with a byte budget.
///
/// Size accounting is exact (`splats.len() * size_of::<GaussianSplatV2>()`), not
/// estimated, so [`Self::evict_to_budget`] reflects real resident bytes. The
/// driver runs `evict_to_budget(SplatBudget.max_gpu_mb * 1MB)` once per frame
/// after applying completed loads.
pub struct ResidencyCache {
    entries: HashMap<TileKey, CacheEntry>,
    clock: u64,
}

impl Default for ResidencyCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ResidencyCache {
    pub fn new() -> Self {
        Self { entries: HashMap::new(), clock: 0 }
    }

    /// Exact resident byte size of a tile.
    pub fn tile_size_bytes(tile: &VxmFileV2) -> usize {
        tile.splats.len() * std::mem::size_of::<GaussianSplatV2>()
    }

    /// Insert (or replace) a tile. `size_bytes` should come from
    /// [`Self::tile_size_bytes`]; it's taken explicitly so callers that already
    /// computed it don't recompute. Counts as a touch (most-recently-used).
    pub fn insert(&mut self, key: TileKey, tile: VxmFileV2, size_bytes: usize) {
        let last_touch = self.tick();
        self.entries.insert(key, CacheEntry { tile, size_bytes, last_touch });
    }

    /// Get a tile, touching it so it becomes most-recently-used.
    pub fn get(&mut self, key: &TileKey) -> Option<&VxmFileV2> {
        let stamp = self.tick();
        let entry = self.entries.get_mut(key)?;
        entry.last_touch = stamp;
        Some(&entry.tile)
    }

    /// Peek without affecting LRU order.
    pub fn peek(&self, key: &TileKey) -> Option<&VxmFileV2> {
        self.entries.get(key).map(|e| &e.tile)
    }

    /// Total resident bytes.
    pub fn total_bytes(&self) -> usize {
        self.entries.values().map(|e| e.size_bytes).sum()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, key: &TileKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Evict least-recently-used tiles until total resident bytes ≤ `budget_bytes`.
    /// Returns the evicted keys in eviction order (least-recently-used first).
    /// Evicts nothing if already within budget.
    pub fn evict_to_budget(&mut self, budget_bytes: usize) -> Vec<TileKey> {
        let mut evicted = Vec::new();
        while self.total_bytes() > budget_bytes {
            // Find the least-recently-used entry (smallest touch stamp).
            let victim = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_touch)
                .map(|(k, _)| *k);
            match victim {
                Some(key) => {
                    self.entries.remove(&key);
                    evicted.push(key);
                }
                None => break, // empty but still over budget (budget < 0 logically)
            }
        }
        evicted
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vxm_v2::VxmHeaderV2;
    use bytemuck::Zeroable;
    use std::time::{Duration, Instant};
    use uuid::Uuid;

    /// Build a tile with `n` splats; splat[0] sits at the given position so the
    /// round-trip test can assert an exact recovered value.
    fn make_tile(n: usize, first_pos: [f32; 3]) -> VxmFileV2 {
        let mut splats = Vec::with_capacity(n);
        for i in 0..n {
            let mut s = GaussianSplatV2::zeroed();
            s.position = if i == 0 { first_pos } else { [i as f32, 0.0, 0.0] };
            s.opacity = 255;
            splats.push(s);
        }
        let header = VxmHeaderV2::new(Uuid::nil(), n as u32);
        VxmFileV2 { header, splats }
    }

    fn write_tile(dir: &Path, name: &str, tile: &VxmFileV2) -> PathBuf {
        let path = dir.join(name);
        let f = std::fs::File::create(&path).unwrap();
        tile.write(std::io::BufWriter::new(f)).unwrap();
        path
    }

    /// Spin draining until `want` tiles arrive or we time out (prevents a hang
    /// from becoming an infinite loop in CI).
    fn drain_until(streamer: &TileStreamer, want: usize) -> Vec<LoadedTile> {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut out = Vec::new();
        while out.len() < want {
            out.extend(streamer.drain_completed());
            if out.len() >= want {
                break;
            }
            assert!(Instant::now() < deadline, "timed out waiting for {want} tiles");
            std::thread::sleep(Duration::from_millis(2));
        }
        out
    }

    // --- Round-trip: real data integrity through the streamer ----------------

    #[test]
    fn roundtrip_three_tiles() {
        let dir = tempfile::tempdir().unwrap();
        let specs = [
            ((0, 0, 0), 10usize, [1.0f32, 2.0, 3.0]),
            ((1, 0, 0), 100, [4.0, 5.0, 6.0]),
            ((2, 0, 0), 1000, [7.0, 8.0, 9.0]),
        ];
        let mut expected: HashMap<TileKey, (usize, [f32; 3])> = HashMap::new();
        let streamer = TileStreamer::new(StreamerConfig::default());

        for (i, (key, count, pos)) in specs.iter().enumerate() {
            let tile = make_tile(*count, *pos);
            let path = write_tile(dir.path(), &format!("tile_{i}.vxm"), &tile);
            expected.insert(*key, (*count, *pos));
            // Distinct priorities so the heap has to order them.
            streamer.request(*key, path, *count as f32, 1);
        }

        let loaded = drain_until(&streamer, 3);
        assert_eq!(loaded.len(), 3);
        for lt in loaded {
            let tile = match lt.result {
                Ok(t) => t,
                Err(e) => panic!("tile {:?} should load, got {e}", lt.key),
            };
            let (exp_count, exp_pos) = expected[&lt.key];
            assert_eq!(tile.splats.len(), exp_count, "splat count for {:?}", lt.key);
            assert_eq!(tile.splats[0].position, exp_pos, "splat[0] pos for {:?}", lt.key);
        }
    }

    // --- Priority order: deterministic via workers=0 + step_one --------------

    #[test]
    fn priority_order_is_descending() {
        let dir = tempfile::tempdir().unwrap();
        let streamer = TileStreamer::new(StreamerConfig { workers: 0, max_retries: 0 });

        // Enqueue in a deliberately non-sorted order with distinct priorities.
        // key.0 is set equal to the priority so we can read order off the result.
        let prios = [3.0f32, 9.0, 1.0, 7.0, 5.0];
        for (i, p) in prios.iter().enumerate() {
            let tile = make_tile(1, [*p, 0.0, 0.0]);
            let path = write_tile(dir.path(), &format!("p_{i}.vxm"), &tile);
            streamer.request((*p as i32, 0, 0), path, *p, 1);
        }

        // No worker threads exist, so nothing has been processed yet.
        let mut order = Vec::new();
        while streamer.step_one() {
            for lt in streamer.drain_completed() {
                order.push(lt.key.0);
            }
        }
        // Highest priority processed first → strictly descending.
        assert_eq!(order, vec![9, 7, 5, 3, 1]);
    }

    // --- Stale generation: skipped with zero disk I/O ------------------------

    #[test]
    fn stale_generation_skips_disk_io() {
        let streamer = TileStreamer::new(StreamerConfig { workers: 0, max_retries: 3 });
        // Path that does NOT exist. If the streamer opened it we'd get Io, not Stale.
        let missing = PathBuf::from("/nonexistent/definitely/not/here_stale.vxm");
        streamer.request((42, 0, 0), missing, 1.0, 1);
        streamer.set_min_generation(2);

        assert!(streamer.step_one());
        let done = streamer.drain_completed();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].result.as_ref().err(), Some(&StreamError::Stale));
        // Proof no disk I/O happened: zero attempts (an Io error would be >= 1).
        assert_eq!(done[0].attempts, 0);
    }

    // --- Missing file: Io error after exactly max_retries+1 attempts ---------

    #[test]
    fn missing_file_retries_then_reports_io() {
        let missing = PathBuf::from("/nonexistent/definitely/not/here_missing.vxm");

        // max_retries = 0 → exactly 1 attempt.
        let s0 = TileStreamer::new(StreamerConfig { workers: 0, max_retries: 0 });
        s0.request((1, 0, 0), missing.clone(), 1.0, 1);
        assert!(s0.step_one());
        let d0 = s0.drain_completed();
        assert_eq!(d0.len(), 1);
        assert!(matches!(d0[0].result, Err(StreamError::Io(_))));
        assert_eq!(d0[0].attempts, 1);

        // max_retries = 2 → exactly 3 attempts.
        let s2 = TileStreamer::new(StreamerConfig { workers: 0, max_retries: 2 });
        s2.request((2, 0, 0), missing, 1.0, 1);
        assert!(s2.step_one());
        let d2 = s2.drain_completed();
        assert_eq!(d2.len(), 1);
        assert!(matches!(d2[0].result, Err(StreamError::Io(_))));
        assert_eq!(d2[0].attempts, 3);
    }

    // --- Truncated file: transient I/O, retried, reported as Io --------------

    #[test]
    fn truncated_file_retries_then_reports_io() {
        let dir = tempfile::tempdir().unwrap();
        // Write a valid tile, then truncate it to half its length so it opens
        // fine but read_exact hits UnexpectedEof mid-stream → VxmError::Io.
        let tile = make_tile(100, [1.0, 2.0, 3.0]);
        let path = write_tile(dir.path(), "truncated.vxm", &tile);
        let full_len = std::fs::metadata(&path).unwrap().len();
        assert!(full_len > 1, "tile should have nonzero length");
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(full_len / 2).unwrap();
        drop(f);

        // max_retries = 2 → exactly 3 attempts, all transient I/O failures.
        let s = TileStreamer::new(StreamerConfig { workers: 0, max_retries: 2 });
        s.request((7, 0, 0), path, 1.0, 1);
        assert!(s.step_one());
        let done = s.drain_completed();
        assert_eq!(done.len(), 1);
        assert!(
            matches!(done[0].result, Err(StreamError::Io(_))),
            "truncated file must be classified Io (transient), got err={:?}",
            done[0].result.as_ref().err()
        );
        assert_eq!(done[0].attempts, 3, "transient I/O must exhaust all retries");
    }

    // --- LRU eviction: exact victims, remaining within budget ----------------

    #[test]
    fn lru_evicts_exact_keys() {
        let mut cache = ResidencyCache::new();
        // Three tiles of known sizes. size_of::<GaussianSplatV2>() == 52, so we
        // pass explicit byte sizes to make the budget arithmetic exact.
        let a = (0, 0, 0);
        let b = (1, 0, 0);
        let c = (2, 0, 0);
        cache.insert(a, make_tile(1, [0.0; 3]), 100); // inserted first (oldest)
        cache.insert(b, make_tile(1, [0.0; 3]), 200); // middle
        cache.insert(c, make_tile(1, [0.0; 3]), 300); // newest
        assert_eq!(cache.total_bytes(), 600);

        // Touch `a` so it is now most-recently-used; b is the LRU.
        assert!(cache.get(&a).is_some());

        // Budget 350: must drop b (200) then c (300)? total 600 -> remove b -> 400
        // still > 350 -> remove next LRU which is now c (a was just touched) -> 100.
        let evicted = cache.evict_to_budget(350);
        assert_eq!(evicted, vec![b, c]);
        assert_eq!(cache.total_bytes(), 100);
        assert!(cache.contains(&a));
        assert!(!cache.contains(&b));
        assert!(!cache.contains(&c));
        assert!(cache.total_bytes() <= 350);
    }

    #[test]
    fn evict_within_budget_is_noop() {
        let mut cache = ResidencyCache::new();
        cache.insert((0, 0, 0), make_tile(1, [0.0; 3]), 100);
        let evicted = cache.evict_to_budget(1000);
        assert!(evicted.is_empty());
        assert_eq!(cache.total_bytes(), 100);
    }

    #[test]
    fn exact_size_accounting_from_splats() {
        let tile = make_tile(10, [0.0; 3]);
        assert_eq!(ResidencyCache::tile_size_bytes(&tile), 10 * 52);
    }

    // --- Shutdown: drop must join, no hang -----------------------------------

    #[test]
    fn drop_joins_workers_without_hanging() {
        let dir = tempfile::tempdir().unwrap();
        let tile = make_tile(50, [1.0, 1.0, 1.0]);
        let path = write_tile(dir.path(), "shutdown.vxm", &tile);

        let start = Instant::now();
        {
            let streamer = TileStreamer::new(StreamerConfig::default());
            streamer.request((0, 0, 0), path, 1.0, 1);
            // Drop here: must signal + join all workers, not detach them.
        }
        // If a worker were left spinning/blocked the drop would hang; bound it.
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "drop should join workers promptly"
        );
    }
}
