//! Bounded asynchronous Ochroma geometry admission.
//!
//! Finished meshes enter this service as ordinary geometry. A single engine
//! worker derives exact programs, continuous detail, correspondence, and cache
//! pages without blocking the render thread. The worker never makes camera,
//! residency, eviction, or native-backend decisions; Spectra owns those
//! frame-time choices after admission.

use crate::mesh_detail::{DetailMeshInput, RuntimeDetailPreparation, prepare_runtime_detail};
use crate::mesh_program::{MeshProgramDerivation, MeshProgramInput, prepare_mesh_programs};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const RUNTIME_GEOMETRY_SOURCE_SCHEMA: u32 = 1;

/// Static read-only views into an authoritative source allocation owned by a
/// game-object cache. The admission worker may outlive the caller's stack but
/// never the cache, so callers must provide genuinely process/catalog-lifetime
/// storage. This prevents a second city-scale copy while pages are derived.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeGeometryBorrowedSource {
    pub positions: &'static [[f32; 3]],
    pub normals: &'static [[f32; 3]],
    pub uvs: &'static [[f32; 2]],
    pub indices: &'static [[u32; 3]],
    pub weathering_masks: &'static [[f32; 7]],
}

#[derive(Clone, Debug)]
pub struct RuntimeGeometrySource {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<[u32; 3]>,
    pub material_ids: Vec<u32>,
    pub weathering_masks: Vec<[f32; 7]>,
    /// Shared immutable source backing when admission originates from a frozen
    /// `HybridMesh` prototype. Owned vectors stay empty on this path, so the
    /// bounded worker does not clone the full source before deriving pages.
    pub shared_prototype: Option<Arc<crate::hybrid_compose::HybridPrototypeGeometry>>,
    /// Shared source owned by a non-renderer game-object cache (for example a
    /// lazily materialised VXP payload). Mutually exclusive with
    /// `shared_prototype`; owned geometry vectors remain empty.
    pub borrowed_source: Option<RuntimeGeometryBorrowedSource>,
    /// Exact structural programs are valuable for architectural meshes but
    /// make organic geometry pay for a three-edge BTreeMap per triangle only
    /// to discover that it has no repeated planar lattice. Organic prototypes
    /// request continuous detail directly.
    pub derivation: RuntimeGeometryDerivation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeGeometryDerivation {
    ProgramsAndDetail,
    DetailOnly,
}

impl RuntimeGeometrySource {
    fn source_positions(&self) -> &[[f32; 3]] {
        if let Some(source) = self.borrowed_source {
            source.positions
        } else {
            self.shared_prototype
                .as_ref()
                .map_or(self.positions.as_slice(), |geometry| geometry.positions())
        }
    }

    fn source_normals(&self) -> &[[f32; 3]] {
        if let Some(source) = self.borrowed_source {
            source.normals
        } else {
            self.shared_prototype
                .as_ref()
                .map_or(self.normals.as_slice(), |geometry| geometry.normals())
        }
    }

    fn source_uvs(&self) -> &[[f32; 2]] {
        if let Some(source) = self.borrowed_source {
            source.uvs
        } else {
            self.shared_prototype
                .as_ref()
                .map_or(self.uvs.as_slice(), |geometry| geometry.uvs())
        }
    }

    fn source_indices(&self) -> &[[u32; 3]] {
        if let Some(source) = self.borrowed_source {
            source.indices
        } else {
            self.shared_prototype
                .as_ref()
                .map_or(self.indices.as_slice(), |geometry| geometry.indices())
        }
    }

    fn source_material_ids(&self) -> &[u32] {
        self.shared_prototype
            .as_ref()
            .map_or(self.material_ids.as_slice(), |geometry| {
                if geometry.material_ids().is_empty() {
                    self.material_ids.as_slice()
                } else {
                    geometry.material_ids()
                }
            })
    }

    fn source_weathering_masks(&self) -> &[[f32; 7]] {
        if let Some(source) = self.borrowed_source {
            source.weathering_masks
        } else {
            self.shared_prototype
                .as_ref()
                .map_or(self.weathering_masks.as_slice(), |geometry| {
                    geometry.weathering_masks()
                })
        }
    }

    /// Normalize an immutable scene prototype for the shared runtime-geometry
    /// admission path. This is used by small derived prototypes such as a
    /// progressive-reveal frontier; the returned content key deduplicates one
    /// source BLAS across every active instance of the authored section.
    pub fn from_proto_mesh(
        mesh: &vox_scene::ProtoMesh,
    ) -> Result<Self, RuntimeGeometryServiceError> {
        let source = Self {
            positions: mesh.positions.clone(),
            normals: mesh.normals.clone(),
            uvs: mesh.uvs.clone(),
            indices: mesh.indices.clone(),
            material_ids: mesh.material_ids.clone(),
            weathering_masks: Vec::new(),
            shared_prototype: None,
            borrowed_source: None,
            derivation: RuntimeGeometryDerivation::ProgramsAndDetail,
        };
        source.validate()?;
        Ok(source)
    }

    fn validate_hybrid_mesh(
        mesh: &crate::hybrid_compose::HybridMesh,
    ) -> Result<usize, RuntimeGeometryServiceError> {
        let positions = mesh.positions();
        let indices = mesh.indices();
        let normals = mesh.normals();
        let uvs = mesh.uvs();
        let weathering_masks = mesh.weathering_masks();
        let material_ids = mesh.material_ids();
        let triangle_count = indices.len() / 3;
        if positions.is_empty()
            || triangle_count == 0
            || indices.len() % 3 != 0
            || normals.len() != positions.len()
            || (!uvs.is_empty() && uvs.len() != positions.len())
            || (!weathering_masks.is_empty() && weathering_masks.len() != positions.len() * 7)
            || (!material_ids.is_empty() && material_ids.len() != triangle_count)
            || mesh
                .positions()
                .iter()
                .flatten()
                .chain(normals.iter().flatten())
                .chain(uvs.iter().flatten())
                .chain(weathering_masks.iter())
                .any(|value| !value.is_finite())
            || indices
                .iter()
                .any(|index| *index as usize >= positions.len())
        {
            return Err(RuntimeGeometryServiceError::InvalidSource);
        }
        Ok(triangle_count)
    }

    /// Hash an ordinary finished renderer mesh without copying its potentially
    /// city-scale geometry streams. The result is byte-identical to converting
    /// with [`Self::from_hybrid_mesh`] and then calling [`Self::key`].
    pub fn key_from_hybrid_mesh(
        mesh: &crate::hybrid_compose::HybridMesh,
    ) -> Result<RuntimeGeometryKey, RuntimeGeometryServiceError> {
        Self::key_from_hybrid_mesh_with_derivation(
            mesh,
            RuntimeGeometryDerivation::ProgramsAndDetail,
        )
    }

    pub fn key_from_hybrid_mesh_detail_only(
        mesh: &crate::hybrid_compose::HybridMesh,
    ) -> Result<RuntimeGeometryKey, RuntimeGeometryServiceError> {
        Self::key_from_hybrid_mesh_with_derivation(mesh, RuntimeGeometryDerivation::DetailOnly)
    }

    fn key_from_hybrid_mesh_with_derivation(
        mesh: &crate::hybrid_compose::HybridMesh,
        derivation: RuntimeGeometryDerivation,
    ) -> Result<RuntimeGeometryKey, RuntimeGeometryServiceError> {
        let triangle_count = Self::validate_hybrid_mesh(mesh)?;
        let mut hash = Sha256::new();
        hash.update(b"OCHROMA_RUNTIME_GEOMETRY_SOURCE");
        hash.update(RUNTIME_GEOMETRY_SOURCE_SCHEMA.to_le_bytes());
        hash.update([match derivation {
            RuntimeGeometryDerivation::ProgramsAndDetail => 0,
            RuntimeGeometryDerivation::DetailOnly => 1,
        }]);
        hash_f32_arrays(&mut hash, mesh.positions());
        hash_f32_arrays(&mut hash, mesh.normals());
        hash_f32_arrays(&mut hash, mesh.uvs());
        hash.update((triangle_count as u64).to_le_bytes());
        for index in mesh.indices() {
            hash.update(index.to_le_bytes());
        }
        hash.update((triangle_count as u64).to_le_bytes());
        if mesh.material_ids().is_empty() {
            for _ in 0..triangle_count {
                hash.update(u32::from(mesh.material_channel).to_le_bytes());
            }
        } else {
            for material in mesh.material_ids() {
                hash.update(material.to_le_bytes());
            }
        }
        hash.update(
            (if mesh.weathering_masks().is_empty() {
                0
            } else {
                mesh.positions().len()
            } as u64)
                .to_le_bytes(),
        );
        for value in mesh.weathering_masks() {
            hash.update(value.to_bits().to_le_bytes());
        }
        Ok(RuntimeGeometryKey(hash.finalize().into()))
    }

    /// Copy one ordinary finished renderer mesh into the canonical admission
    /// source. The mesh remains the authority; this only normalizes its flat
    /// index stream and material/weathering streams for the engine worker.
    ///
    /// Missing normals are rejected rather than synthesized because exact
    /// source-surface correspondence includes the shading frame.
    pub fn from_hybrid_mesh(
        mesh: &crate::hybrid_compose::HybridMesh,
    ) -> Result<Self, RuntimeGeometryServiceError> {
        let triangle_count = Self::validate_hybrid_mesh(mesh)?;
        if let Some(shared_prototype) = mesh.shared_prototype_geometry() {
            let material_ids = if shared_prototype.material_ids().is_empty() {
                vec![u32::from(mesh.material_channel); triangle_count]
            } else {
                Vec::new()
            };
            let source = Self {
                positions: Vec::new(),
                normals: Vec::new(),
                uvs: Vec::new(),
                indices: Vec::new(),
                material_ids,
                weathering_masks: Vec::new(),
                shared_prototype: Some(shared_prototype),
                borrowed_source: None,
                derivation: RuntimeGeometryDerivation::ProgramsAndDetail,
            };
            source.validate()?;
            debug_assert_eq!(source.key(), Self::key_from_hybrid_mesh(mesh)?);
            return Ok(source);
        }
        let indices = mesh
            .indices()
            .chunks_exact(3)
            .map(|triangle| [triangle[0], triangle[1], triangle[2]])
            .collect::<Vec<_>>();
        let weathering_masks = mesh
            .weathering_masks()
            .chunks_exact(7)
            .map(|channels| {
                <[f32; 7]>::try_from(channels).expect("seven-channel chunks are validated")
            })
            .collect::<Vec<_>>();
        let material_ids = if mesh.material_ids().is_empty() {
            vec![u32::from(mesh.material_channel); triangle_count]
        } else {
            mesh.material_ids().to_vec()
        };
        let source = Self {
            positions: mesh.positions().to_vec(),
            normals: mesh.normals().to_vec(),
            uvs: mesh.uvs().to_vec(),
            material_ids,
            indices,
            weathering_masks,
            shared_prototype: None,
            borrowed_source: None,
            derivation: RuntimeGeometryDerivation::ProgramsAndDetail,
        };
        source.validate()?;
        debug_assert_eq!(source.key(), Self::key_from_hybrid_mesh(mesh)?);
        Ok(source)
    }

    pub fn validate(&self) -> Result<(), RuntimeGeometryServiceError> {
        if self.borrowed_source.is_some() && self.shared_prototype.is_some()
            || self.borrowed_source.is_some()
                && (!self.positions.is_empty()
                    || !self.normals.is_empty()
                    || !self.uvs.is_empty()
                    || !self.indices.is_empty()
                    || !self.weathering_masks.is_empty())
        {
            return Err(RuntimeGeometryServiceError::InvalidSource);
        }
        let positions = self.source_positions();
        let normals = self.source_normals();
        let uvs = self.source_uvs();
        let indices = self.source_indices();
        let material_ids = self.source_material_ids();
        let weathering_masks = self.source_weathering_masks();
        if positions.is_empty()
            || indices.is_empty()
            || normals.len() != positions.len()
            || (!uvs.is_empty() && uvs.len() != positions.len())
            || (!weathering_masks.is_empty() && weathering_masks.len() != positions.len())
            || material_ids.len() != indices.len()
            || self
                .source_positions()
                .iter()
                .flatten()
                .chain(normals.iter().flatten())
                .chain(uvs.iter().flatten())
                .chain(weathering_masks.iter().flatten())
                .any(|value| !value.is_finite())
            || indices
                .iter()
                .flatten()
                .any(|index| *index as usize >= positions.len())
        {
            return Err(RuntimeGeometryServiceError::InvalidSource);
        }
        Ok(())
    }

    #[must_use]
    pub fn key(&self) -> RuntimeGeometryKey {
        let mut hash = Sha256::new();
        hash.update(b"OCHROMA_RUNTIME_GEOMETRY_SOURCE");
        hash.update(RUNTIME_GEOMETRY_SOURCE_SCHEMA.to_le_bytes());
        hash.update([match self.derivation {
            RuntimeGeometryDerivation::ProgramsAndDetail => 0,
            RuntimeGeometryDerivation::DetailOnly => 1,
        }]);
        hash_f32_arrays(&mut hash, self.source_positions());
        hash_f32_arrays(&mut hash, self.source_normals());
        hash_f32_arrays(&mut hash, self.source_uvs());
        hash_u32_arrays(&mut hash, self.source_indices());
        hash_u32s(&mut hash, self.source_material_ids());
        hash_f32_arrays(&mut hash, self.source_weathering_masks());
        RuntimeGeometryKey(hash.finalize().into())
    }

    #[must_use]
    pub fn detail_only(mut self) -> Self {
        self.derivation = RuntimeGeometryDerivation::DetailOnly;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RuntimeGeometryKey(pub [u8; 32]);

#[derive(Clone, Debug)]
pub struct RuntimeGeometryProduct {
    pub programs: MeshProgramDerivation,
    pub detail: RuntimeDetailPreparation,
}

#[derive(Clone, Debug)]
pub enum RuntimeGeometryStatus {
    Missing,
    Queued,
    Running,
    Ready(Arc<RuntimeGeometryProduct>),
    Failed(Arc<str>),
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum RuntimeGeometryServiceError {
    #[error("finished mesh has invalid or inconsistent streams")]
    InvalidSource,
    #[error("runtime geometry admission queue is full (capacity {capacity})")]
    QueueFull { capacity: usize },
    #[error("deferred runtime geometry requires immutable shared source storage")]
    DeferredSourceNotShared,
    #[error("runtime geometry service has shut down")]
    Shutdown,
}

enum Job {
    Queued(RuntimeGeometrySource),
    Running,
    Ready(Arc<RuntimeGeometryProduct>),
    Failed(Arc<str>),
}

struct State {
    queue: VecDeque<RuntimeGeometryKey>,
    jobs: BTreeMap<RuntimeGeometryKey, Job>,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    shutdown: AtomicBool,
    completion_epoch: AtomicU64,
    capacity: usize,
}

pub struct RuntimeGeometryService {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for RuntimeGeometryService {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeGeometryService")
            .field("capacity", &self.shared.capacity)
            .field("completion_epoch", &self.completion_epoch())
            .finish_non_exhaustive()
    }
}

impl RuntimeGeometryService {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "runtime geometry capacity must be nonzero");
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                queue: VecDeque::with_capacity(capacity),
                jobs: BTreeMap::new(),
            }),
            wake: Condvar::new(),
            shutdown: AtomicBool::new(false),
            completion_epoch: AtomicU64::new(0),
            capacity,
        });
        let worker_shared = Arc::clone(&shared);
        let worker = std::thread::Builder::new()
            .name("ochroma-geometry-admission".into())
            .spawn(move || worker_main(&worker_shared))
            .expect("spawn bounded Ochroma geometry-admission worker");
        Self {
            shared,
            worker: Some(worker),
        }
    }

    pub fn submit(
        &self,
        source: RuntimeGeometrySource,
    ) -> Result<RuntimeGeometryKey, RuntimeGeometryServiceError> {
        source.validate()?;
        if self.shared.shutdown.load(Ordering::Acquire) {
            return Err(RuntimeGeometryServiceError::Shutdown);
        }
        let key = source.key();
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.jobs.contains_key(&key) {
            return Ok(key);
        }
        // Terminal products still own the complete derived hierarchy until a
        // consumer atomically takes them. Counting only queued/running work
        // lets an unpolled city build accumulate an unbounded number of Ready
        // products even though the service advertises a bounded capacity.
        if state.jobs.len() >= self.shared.capacity {
            return Err(RuntimeGeometryServiceError::QueueFull {
                capacity: self.shared.capacity,
            });
        }
        state.jobs.insert(key, Job::Queued(source));
        state.queue.push_back(key);
        drop(state);
        self.shared.wake.notify_one();
        Ok(key)
    }

    #[must_use]
    pub fn has_capacity(&self) -> bool {
        let state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.jobs.len() < self.shared.capacity
    }

    /// True only when no queued, running, or completed-but-unconsumed product
    /// remains. Callers use this to coalesce many prototype promotions into
    /// one structural scene update.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .jobs
            .is_empty()
    }

    #[must_use]
    pub fn poll(&self, key: RuntimeGeometryKey) -> RuntimeGeometryStatus {
        let state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match state.jobs.get(&key) {
            None => RuntimeGeometryStatus::Missing,
            Some(Job::Queued(_)) => RuntimeGeometryStatus::Queued,
            Some(Job::Running) => RuntimeGeometryStatus::Running,
            Some(Job::Ready(product)) => RuntimeGeometryStatus::Ready(Arc::clone(product)),
            Some(Job::Failed(error)) => RuntimeGeometryStatus::Failed(Arc::clone(error)),
        }
    }

    /// Remove and return a terminal result. Queued/running work remains owned
    /// by the service. Consumers call this when atomically promoting a product
    /// so the worker does not retain a duplicate runtime hierarchy forever.
    pub fn take(&self, key: RuntimeGeometryKey) -> RuntimeGeometryStatus {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match state.jobs.remove(&key) {
            None => RuntimeGeometryStatus::Missing,
            Some(Job::Queued(source)) => {
                state.jobs.insert(key, Job::Queued(source));
                RuntimeGeometryStatus::Queued
            }
            Some(Job::Running) => {
                state.jobs.insert(key, Job::Running);
                RuntimeGeometryStatus::Running
            }
            Some(Job::Ready(product)) => RuntimeGeometryStatus::Ready(product),
            Some(Job::Failed(error)) => RuntimeGeometryStatus::Failed(error),
        }
    }

    #[must_use]
    pub fn completion_epoch(&self) -> u64 {
        self.shared.completion_epoch.load(Ordering::Acquire)
    }

    /// Diagnostic/test wait only. Product callers poll the completion epoch and
    /// continue rendering the authoritative triangle fallback.
    pub fn wait(&self, key: RuntimeGeometryKey, timeout: Duration) -> RuntimeGeometryStatus {
        let deadline = Instant::now() + timeout;
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            let status = match state.jobs.get(&key) {
                None => RuntimeGeometryStatus::Missing,
                Some(Job::Queued(_)) => RuntimeGeometryStatus::Queued,
                Some(Job::Running) => RuntimeGeometryStatus::Running,
                Some(Job::Ready(product)) => RuntimeGeometryStatus::Ready(Arc::clone(product)),
                Some(Job::Failed(error)) => RuntimeGeometryStatus::Failed(Arc::clone(error)),
            };
            if matches!(
                status,
                RuntimeGeometryStatus::Missing
                    | RuntimeGeometryStatus::Ready(_)
                    | RuntimeGeometryStatus::Failed(_)
            ) {
                return status;
            }
            let now = Instant::now();
            if now >= deadline {
                return status;
            }
            let (next, _) = self
                .shared
                .wake
                .wait_timeout(state, deadline - now)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next;
        }
    }
}

impl Default for RuntimeGeometryService {
    fn default() -> Self {
        // Bound decoded source duplication while a city build discovers many
        // prototypes. Urban Horizon retains excess ids and submits them as
        // slots complete; the renderable triangle source never waits.
        Self::new(8)
    }
}

impl Drop for RuntimeGeometryService {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.wake.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Content-addressed owner for arbitrary finished engine meshes.
///
/// This registry is intentionally unaware of assets, maps, terrain, roads, or
/// games. Callers retain and render their authoritative triangles while the
/// bounded service derives a disposable product, then look it up by the exact
/// finished-mesh key. Queue saturation is explicit: callers retry on a later
/// structural pass and no unbounded duplicate source queue is retained here.
#[derive(Debug)]
pub struct RuntimeGeometryRegistry {
    service: RuntimeGeometryService,
    active: Mutex<BTreeSet<RuntimeGeometryKey>>,
    waiting: Mutex<BTreeMap<RuntimeGeometryKey, RuntimeGeometrySource>>,
    products: Mutex<BTreeMap<RuntimeGeometryKey, Arc<RuntimeGeometryProduct>>>,
    failures: AtomicU64,
}

impl RuntimeGeometryRegistry {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            service: RuntimeGeometryService::new(capacity),
            active: Mutex::new(BTreeSet::new()),
            waiting: Mutex::new(BTreeMap::new()),
            products: Mutex::new(BTreeMap::new()),
            failures: AtomicU64::new(0),
        }
    }

    /// Whether a new owned scene source can enter the bounded derivation
    /// island. Callers check this before normalising/cloning a finished mesh;
    /// constructing a city-scale source merely to return `QueueFull` defeats
    /// the bound and was the source of multi-gigabyte transient allocations.
    #[must_use]
    pub fn has_capacity(&self) -> bool {
        self.service.has_capacity()
    }

    /// Admit a finished mesh without waiting. Returns its stable content key.
    ///
    /// `QueueFull` means the authoritative triangle mesh remains renderable and
    /// should be retried after the completion epoch advances.
    pub fn admit(
        &self,
        source: RuntimeGeometrySource,
    ) -> Result<RuntimeGeometryKey, RuntimeGeometryServiceError> {
        source.validate()?;
        let key = source.key();
        let active = self
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if active.contains(&key)
            || self
                .products
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains_key(&key)
        {
            return Ok(key);
        }
        drop(active);
        let submitted = self.service.submit(source)?;
        debug_assert_eq!(submitted, key);
        self.active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key);
        Ok(key)
    }

    /// Request eventual admission for a source already held in immutable
    /// shared storage. Saturation retains only an Arc/borrowed view, never a
    /// second geometry allocation. Pending keys are submitted in content-key
    /// order as worker slots complete, so a one-shot scene build cannot lose
    /// every prototype after the first bounded batch.
    pub fn admit_or_defer_shared(
        &self,
        source: RuntimeGeometrySource,
    ) -> Result<RuntimeGeometryKey, RuntimeGeometryServiceError> {
        source.validate()?;
        if source.shared_prototype.is_none() && source.borrowed_source.is_none() {
            return Err(RuntimeGeometryServiceError::DeferredSourceNotShared);
        }
        let key = source.key();
        if self.product(key).is_some()
            || self
                .active
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains(&key)
            || self
                .waiting
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains_key(&key)
        {
            return Ok(key);
        }
        if self.service.has_capacity() {
            let submitted = self.service.submit(source)?;
            self.active
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(key);
            debug_assert_eq!(submitted, key);
        } else {
            self.waiting
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(key, source);
        }
        Ok(key)
    }

    /// Promote every terminal worker result. This never waits or derives.
    pub fn refresh(&self) -> Result<usize, String> {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let keys = active.iter().copied().collect::<Vec<_>>();
        let mut promoted = 0;
        for key in keys {
            match self.service.take(key) {
                RuntimeGeometryStatus::Ready(product) => {
                    self.products
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .insert(key, product);
                    active.remove(&key);
                    promoted += 1;
                }
                RuntimeGeometryStatus::Failed(error) => {
                    active.remove(&key);
                    self.failures.fetch_add(1, Ordering::Relaxed);
                    return Err(format!(
                        "Ochroma runtime geometry derivation failed for {}: {error}",
                        hex_key(key)
                    ));
                }
                RuntimeGeometryStatus::Missing => {
                    active.remove(&key);
                    self.failures.fetch_add(1, Ordering::Relaxed);
                    return Err(format!(
                        "Ochroma runtime geometry job disappeared for {}",
                        hex_key(key)
                    ));
                }
                RuntimeGeometryStatus::Queued | RuntimeGeometryStatus::Running => {}
            }
        }
        drop(active);
        self.admit_waiting()?;
        Ok(promoted)
    }

    fn admit_waiting(&self) -> Result<(), String> {
        loop {
            if !self.service.has_capacity() {
                return Ok(());
            }
            let next = self
                .waiting
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .pop_first();
            let Some((key, source)) = next else {
                return Ok(());
            };
            match self.service.submit(source) {
                Ok(submitted) => {
                    debug_assert_eq!(submitted, key);
                    self.active
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .insert(key);
                }
                Err(RuntimeGeometryServiceError::QueueFull { .. }) => {
                    return Err(
                        "runtime geometry capacity changed while admitting deferred source".into(),
                    );
                }
                Err(error) => {
                    self.failures.fetch_add(1, Ordering::Relaxed);
                    return Err(format!(
                        "Ochroma deferred runtime geometry admission failed for {}: {error}",
                        hex_key(key)
                    ));
                }
            }
        }
    }

    #[must_use]
    pub fn product(&self, key: RuntimeGeometryKey) -> Option<Arc<RuntimeGeometryProduct>> {
        self.products
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
            .cloned()
    }

    #[must_use]
    pub fn completion_epoch(&self) -> u64 {
        self.service.completion_epoch()
    }

    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.service.is_idle()
            && self
                .active
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_empty()
            && self
                .waiting
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_empty()
    }

    #[must_use]
    pub fn failure_count(&self) -> u64 {
        self.failures.load(Ordering::Relaxed)
    }
}

impl Default for RuntimeGeometryRegistry {
    fn default() -> Self {
        // Ordinary scene meshes are owned vectors, not borrowed VXP mappings.
        // One running/ready source bounds duplicate live geometry to one exact
        // prototype; completed products are promoted before the next admission.
        Self::new(1)
    }
}

fn worker_main(shared: &Shared) {
    loop {
        let (key, source) = {
            let mut state = shared
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            loop {
                if shared.shutdown.load(Ordering::Acquire) {
                    return;
                }
                if let Some(key) = state.queue.pop_front() {
                    let source = match state.jobs.insert(key, Job::Running) {
                        Some(Job::Queued(source)) => source,
                        _ => continue,
                    };
                    break (key, source);
                }
                state = shared
                    .wake
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        };
        let source_triangles = source.source_indices().len();
        let started = Instant::now();
        let result = derive_product(&source)
            .map(Arc::new)
            .map_err(|error| Arc::<str>::from(error));
        if let Ok(product) = &result {
            eprintln!(
                "[mega-geometry] derived key={} triangles={} pages={} cache_hit={} elapsed_ms={:.1}",
                &hex_key(key)[..12],
                source_triangles,
                product.detail.page_count,
                product.detail.cache_hit,
                started.elapsed().as_secs_f64() * 1_000.0,
            );
        }
        // The product contains only derived programs plus page descriptors; it
        // does not borrow the authoritative source. Release that potentially
        // city-scale clone before publishing readiness, otherwise a renderer
        // reacting to the completion epoch can upload the resident scene while
        // this dead second mesh is still live on the worker stack.
        drop(source);
        trim_background_derivation_allocator();
        let mut state = shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.jobs.insert(
            key,
            match result {
                Ok(product) => Job::Ready(product),
                Err(error) => Job::Failed(error),
            },
        );
        shared.completion_epoch.fetch_add(1, Ordering::AcqRel);
        drop(state);
        shared.wake.notify_all();
    }
}

#[cfg(target_os = "linux")]
fn trim_background_derivation_allocator() {
    unsafe extern "C" {
        fn malloc_trim(pad: usize) -> i32;
    }
    // SAFETY: process-global allocator maintenance takes no application
    // pointers. This runs only at a bulk-job lifetime boundary, after every
    // source and temporary hierarchy allocation owned by the job was dropped.
    unsafe {
        malloc_trim(0);
    }
}

#[cfg(not(target_os = "linux"))]
fn trim_background_derivation_allocator() {}

fn hex_key(key: RuntimeGeometryKey) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(64);
    for byte in key.0 {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn derive_product(source: &RuntimeGeometrySource) -> Result<RuntimeGeometryProduct, String> {
    let positions = source.source_positions();
    let normals = source.source_normals();
    let uvs = source.source_uvs();
    let indices = source.source_indices();
    let material_ids = source.source_material_ids();
    let weathering_masks = source.source_weathering_masks();
    let programs = match source.derivation {
        RuntimeGeometryDerivation::ProgramsAndDetail => {
            prepare_mesh_programs(MeshProgramInput {
                positions,
                indices,
                uvs,
                material_ids,
            })
            .map_err(|error| format!("derive exact programs: {error}"))?
            .derivation
        }
        RuntimeGeometryDerivation::DetailOnly => MeshProgramDerivation {
            payload: None,
            exact_quad_count: 0,
            program_count: 0,
            covered_triangle_count: 0,
            residual_triangle_count: u32::try_from(indices.len())
                .map_err(|_| "detail-only source exceeds u32 triangles".to_string())?,
            rejected_quad_candidates: 0,
        },
    };
    let covered = programs
        .payload
        .as_ref()
        .map(|payload| payload.covered_triangle_indices())
        .unwrap_or_default();
    let mut residual_indices = Vec::new();
    let mut residual_materials = Vec::new();
    let mut residual_sources = Vec::new();
    for (triangle, indices) in indices.iter().copied().enumerate() {
        if covered.binary_search(&(triangle as u32)).is_ok() {
            continue;
        }
        residual_indices.push(indices);
        residual_materials.push(material_ids[triangle]);
        residual_sources.push(triangle as u32);
    }
    let mut detail = prepare_runtime_detail(
        positions,
        indices,
        DetailMeshInput {
            positions,
            normals,
            uvs,
            indices: &residual_indices,
            material_ids: &residual_materials,
            weathering_masks,
        },
        &residual_sources,
        matches!(
            source.derivation,
            RuntimeGeometryDerivation::ProgramsAndDetail
        ),
        &[0.125, 0.25, 0.5, 1.0],
    )
    .map_err(|error| format!("derive continuous detail: {error}"))?;
    if matches!(source.derivation, RuntimeGeometryDerivation::DetailOnly)
        && let Some(root) = detail.prototype_root.as_ref()
    {
        // Keep the material-preserving root available as one shared resident
        // prototype. The loaded asset is still the sole product authority: a
        // derived representation keeps its indexed geometry, UVs, materials,
        // and authored shading inputs instead of inventing proxy cards.
        detail.prototype_resident_cut = Some(Arc::clone(root));
    }
    Ok(RuntimeGeometryProduct { programs, detail })
}

fn hash_f32_arrays<const N: usize>(hash: &mut Sha256, values: &[[f32; N]]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values.iter().flatten() {
        hash.update(value.to_bits().to_le_bytes());
    }
}

fn hash_u32_arrays<const N: usize>(hash: &mut Sha256, values: &[[u32; N]]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values.iter().flatten() {
        hash.update(value.to_le_bytes());
    }
}

fn hash_u32s(hash: &mut Sha256, values: &[u32]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(value.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> RuntimeGeometrySource {
        RuntimeGeometrySource {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, -1.0]; 6],
            uvs: vec![
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ],
            indices: vec![[0, 2, 1], [0, 3, 2], [1, 5, 4], [1, 2, 5]],
            material_ids: vec![0; 4],
            weathering_masks: vec![[0.0; 7]; 6],
            shared_prototype: None,
            borrowed_source: None,
            derivation: RuntimeGeometryDerivation::ProgramsAndDetail,
        }
    }

    fn hybrid_source() -> crate::hybrid_compose::HybridMesh {
        let mut mesh = crate::hybrid_compose::HybridMesh::from_rgb(
            source().positions,
            vec![0, 2, 1, 0, 3, 2, 1, 5, 4, 1, 2, 5],
            [0.4, 0.4, 0.4],
            1,
        );
        mesh.normals = source().normals;
        mesh.uvs = source().uvs;
        mesh.weathering_masks = vec![0.0; mesh.positions.len() * 7];
        mesh
    }

    #[test]
    fn frozen_hybrid_admission_shares_source_geometry() {
        let mesh = hybrid_source().freeze_prototype_geometry();
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
        assert_eq!(mesh.positions().len(), 6);
        let before = mesh.prototype_geometry_strong_count();
        let source = RuntimeGeometrySource::from_hybrid_mesh(&mesh).unwrap();
        assert!(source.positions.is_empty());
        assert!(source.indices.is_empty());
        assert!(source.weathering_masks.is_empty());
        assert!(source.shared_prototype.is_some());
        assert_eq!(mesh.prototype_geometry_strong_count(), before + 1);
        assert_eq!(source.source_positions().len(), 6);
        assert_eq!(source.source_indices().len(), 4);
        assert_eq!(source.source_weathering_masks().len(), 6);
        assert_eq!(
            source.key(),
            RuntimeGeometrySource::key_from_hybrid_mesh(&mesh).unwrap()
        );
    }

    #[test]
    fn borrowed_game_object_admission_keeps_one_source_allocation() {
        let owned = source();
        let positions = Box::leak(owned.positions.into_boxed_slice());
        let normals = Box::leak(owned.normals.into_boxed_slice());
        let uvs = Box::leak(owned.uvs.into_boxed_slice());
        let indices = Box::leak(owned.indices.into_boxed_slice());
        let weathering = Box::leak(owned.weathering_masks.into_boxed_slice());
        let positions_ptr = positions.as_ptr();
        let borrowed = RuntimeGeometrySource {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            material_ids: owned.material_ids,
            weathering_masks: Vec::new(),
            shared_prototype: None,
            borrowed_source: Some(RuntimeGeometryBorrowedSource {
                positions,
                normals,
                uvs,
                indices,
                weathering_masks: weathering,
            }),
            derivation: RuntimeGeometryDerivation::ProgramsAndDetail,
        };
        borrowed.validate().unwrap();
        assert_eq!(borrowed.source_positions().as_ptr(), positions_ptr);
        assert!(borrowed.positions.capacity() == 0 && borrowed.indices.capacity() == 0);
        let service = RuntimeGeometryService::new(1);
        let key = service.submit(borrowed).unwrap();
        assert!(matches!(
            service.wait(key, Duration::from_secs(10)),
            RuntimeGeometryStatus::Ready(_)
        ));
    }

    #[test]
    fn bounded_worker_deduplicates_and_completes_off_thread() {
        let service = RuntimeGeometryService::new(2);
        let key = service.submit(source()).unwrap();
        assert_eq!(service.submit(source()).unwrap(), key);
        let status = service.wait(key, Duration::from_secs(10));
        let RuntimeGeometryStatus::Ready(product) = status else {
            panic!("runtime geometry did not complete: {status:?}");
        };
        assert_eq!(service.completion_epoch(), 1);
        assert_eq!(product.programs.covered_triangle_count, 4);
        assert_eq!(product.detail.page_count, 0);
    }

    #[test]
    fn organic_detail_only_request_skips_exact_program_extraction() {
        let service = RuntimeGeometryService::new(1);
        let mut organic = source();
        organic.positions[5][2] = 0.37;
        let key = service.submit(organic.detail_only()).unwrap();
        let RuntimeGeometryStatus::Ready(product) = service.wait(key, Duration::from_secs(10))
        else {
            panic!("detail-only runtime geometry did not complete");
        };
        assert!(product.programs.payload.is_none());
        assert_eq!(product.programs.program_count, 0);
        assert_eq!(product.programs.residual_triangle_count, 4);
        assert!(product.detail.page_count > 0);
    }

    #[test]
    fn invalid_source_is_rejected_before_queue_admission() {
        let service = RuntimeGeometryService::new(1);
        let mut invalid = source();
        invalid.material_ids.pop();
        assert_eq!(
            service.submit(invalid),
            Err(RuntimeGeometryServiceError::InvalidSource)
        );
        assert_eq!(service.completion_epoch(), 0);
    }

    #[test]
    fn borrowed_hybrid_key_matches_owned_admission_source() {
        let mesh = hybrid_source();
        let borrowed = RuntimeGeometrySource::key_from_hybrid_mesh(&mesh).unwrap();
        let owned = RuntimeGeometrySource::from_hybrid_mesh(&mesh).unwrap();
        assert_eq!(borrowed, owned.key());
    }

    #[test]
    fn completed_products_remain_inside_the_bounded_capacity_until_taken() {
        let service = RuntimeGeometryService::new(1);
        let first = service.submit(source()).unwrap();
        assert!(matches!(
            service.wait(first, Duration::from_secs(10)),
            RuntimeGeometryStatus::Ready(_)
        ));
        assert!(!service.has_capacity());

        let mut distinct = source();
        distinct.positions[0][0] = -1.0;
        assert_eq!(
            service.submit(distinct.clone()),
            Err(RuntimeGeometryServiceError::QueueFull { capacity: 1 })
        );

        assert!(matches!(
            service.take(first),
            RuntimeGeometryStatus::Ready(_)
        ));
        assert!(service.has_capacity());
        assert!(service.submit(distinct).is_ok());
    }

    #[test]
    fn service_is_not_idle_until_terminal_product_is_consumed() {
        let service = RuntimeGeometryService::new(1);
        assert!(service.is_idle());
        let key = service.submit(source()).unwrap();
        assert!(!service.is_idle());
        assert!(matches!(
            service.wait(key, Duration::from_secs(10)),
            RuntimeGeometryStatus::Ready(_)
        ));
        assert!(!service.is_idle());
        assert!(matches!(service.take(key), RuntimeGeometryStatus::Ready(_)));
        assert!(service.is_idle());
    }

    #[test]
    fn registry_promotes_by_finished_mesh_content_without_blocking() {
        let registry = RuntimeGeometryRegistry::new(1);
        let key = registry.admit(source()).unwrap();
        assert_eq!(registry.admit(source()).unwrap(), key);
        let deadline = Instant::now() + Duration::from_secs(10);
        while registry.product(key).is_none() && Instant::now() < deadline {
            registry.refresh().unwrap();
            std::thread::yield_now();
        }
        let product = registry
            .product(key)
            .expect("content-addressed runtime product");
        assert_eq!(product.programs.covered_triangle_count, 4);
        assert!(registry.is_idle());
        assert_eq!(registry.failure_count(), 0);
    }

    #[test]
    fn registry_saturation_retains_no_unbounded_waiting_sources() {
        let registry = RuntimeGeometryRegistry::new(1);
        let first = registry.admit(source()).unwrap();
        let mut distinct = source();
        distinct.positions[0][0] = -1.0;
        assert_eq!(
            registry.admit(distinct.clone()),
            Err(RuntimeGeometryServiceError::QueueFull { capacity: 1 })
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        while registry.product(first).is_none() && Instant::now() < deadline {
            registry.refresh().unwrap();
            std::thread::yield_now();
        }
        registry
            .admit(distinct)
            .expect("caller can retry after a bounded slot drains");
    }

    #[test]
    fn registry_defers_shared_sources_and_drains_them() {
        let registry = RuntimeGeometryRegistry::new(1);
        let first_mesh = hybrid_source().freeze_prototype_geometry();
        let mut second_mesh = hybrid_source();
        second_mesh.positions[0][2] = 0.25;
        let second_mesh = second_mesh.freeze_prototype_geometry();
        let first_key = registry
            .admit_or_defer_shared(RuntimeGeometrySource::from_hybrid_mesh(&first_mesh).unwrap())
            .unwrap();
        let second_key = registry
            .admit_or_defer_shared(RuntimeGeometrySource::from_hybrid_mesh(&second_mesh).unwrap())
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(10);
        while !registry.is_idle() && Instant::now() < deadline {
            registry.refresh().unwrap();
            std::thread::sleep(Duration::from_millis(5));
        }
        registry.refresh().unwrap();
        assert!(registry.is_idle());
        assert!(registry.product(first_key).is_some());
        assert!(registry.product(second_key).is_some());
    }

    #[test]
    fn deferred_admission_rejects_owned_duplicate_source() {
        let registry = RuntimeGeometryRegistry::new(1);
        assert_eq!(
            registry.admit_or_defer_shared(source()),
            Err(RuntimeGeometryServiceError::DeferredSourceNotShared)
        );
    }
}
