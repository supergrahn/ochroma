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

#[derive(Clone, Debug)]
pub struct RuntimeGeometrySource {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<[u32; 3]>,
    pub material_ids: Vec<u32>,
    pub weathering_masks: Vec<[f32; 7]>,
}

impl RuntimeGeometrySource {
    fn validate_hybrid_mesh(
        mesh: &crate::hybrid_compose::HybridMesh,
    ) -> Result<usize, RuntimeGeometryServiceError> {
        let triangle_count = mesh.indices.len() / 3;
        if mesh.positions.is_empty()
            || triangle_count == 0
            || mesh.indices.len() % 3 != 0
            || mesh.normals.len() != mesh.positions.len()
            || (!mesh.uvs.is_empty() && mesh.uvs.len() != mesh.positions.len())
            || (!mesh.weathering_masks.is_empty()
                && mesh.weathering_masks.len() != mesh.positions.len() * 7)
            || (!mesh.material_ids.is_empty() && mesh.material_ids.len() != triangle_count)
            || mesh
                .positions
                .iter()
                .flatten()
                .chain(mesh.normals.iter().flatten())
                .chain(mesh.uvs.iter().flatten())
                .chain(mesh.weathering_masks.iter())
                .any(|value| !value.is_finite())
            || mesh
                .indices
                .iter()
                .any(|index| *index as usize >= mesh.positions.len())
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
        let triangle_count = Self::validate_hybrid_mesh(mesh)?;
        let mut hash = Sha256::new();
        hash.update(b"OCHROMA_RUNTIME_GEOMETRY_SOURCE");
        hash.update(RUNTIME_GEOMETRY_SOURCE_SCHEMA.to_le_bytes());
        hash_f32_arrays(&mut hash, &mesh.positions);
        hash_f32_arrays(&mut hash, &mesh.normals);
        hash_f32_arrays(&mut hash, &mesh.uvs);
        hash.update((triangle_count as u64).to_le_bytes());
        for index in &mesh.indices {
            hash.update(index.to_le_bytes());
        }
        hash.update((triangle_count as u64).to_le_bytes());
        if mesh.material_ids.is_empty() {
            for _ in 0..triangle_count {
                hash.update(u32::from(mesh.material_channel).to_le_bytes());
            }
        } else {
            for material in &mesh.material_ids {
                hash.update(material.to_le_bytes());
            }
        }
        hash.update(
            (if mesh.weathering_masks.is_empty() {
                0
            } else {
                mesh.positions.len()
            } as u64)
                .to_le_bytes(),
        );
        for value in &mesh.weathering_masks {
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
        let indices = mesh
            .indices
            .chunks_exact(3)
            .map(|triangle| [triangle[0], triangle[1], triangle[2]])
            .collect::<Vec<_>>();
        let weathering_masks = mesh
            .weathering_masks
            .chunks_exact(7)
            .map(|channels| {
                <[f32; 7]>::try_from(channels).expect("seven-channel chunks are validated")
            })
            .collect::<Vec<_>>();
        let material_ids = if mesh.material_ids.is_empty() {
            vec![u32::from(mesh.material_channel); triangle_count]
        } else {
            mesh.material_ids.clone()
        };
        let source = Self {
            positions: mesh.positions.clone(),
            normals: mesh.normals.clone(),
            uvs: mesh.uvs.clone(),
            material_ids,
            indices,
            weathering_masks,
        };
        source.validate()?;
        debug_assert_eq!(source.key(), Self::key_from_hybrid_mesh(mesh)?);
        Ok(source)
    }

    pub fn validate(&self) -> Result<(), RuntimeGeometryServiceError> {
        if self.positions.is_empty()
            || self.indices.is_empty()
            || self.normals.len() != self.positions.len()
            || (!self.uvs.is_empty() && self.uvs.len() != self.positions.len())
            || (!self.weathering_masks.is_empty()
                && self.weathering_masks.len() != self.positions.len())
            || self.material_ids.len() != self.indices.len()
            || self
                .positions
                .iter()
                .flatten()
                .chain(self.normals.iter().flatten())
                .chain(self.uvs.iter().flatten())
                .chain(self.weathering_masks.iter().flatten())
                .any(|value| !value.is_finite())
            || self
                .indices
                .iter()
                .flatten()
                .any(|index| *index as usize >= self.positions.len())
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
        hash_f32_arrays(&mut hash, &self.positions);
        hash_f32_arrays(&mut hash, &self.normals);
        hash_f32_arrays(&mut hash, &self.uvs);
        hash_u32_arrays(&mut hash, &self.indices);
        hash_u32s(&mut hash, &self.material_ids);
        hash_f32_arrays(&mut hash, &self.weathering_masks);
        RuntimeGeometryKey(hash.finalize().into())
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
    products: Mutex<BTreeMap<RuntimeGeometryKey, Arc<RuntimeGeometryProduct>>>,
    failures: AtomicU64,
}

impl RuntimeGeometryRegistry {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            service: RuntimeGeometryService::new(capacity),
            active: Mutex::new(BTreeSet::new()),
            products: Mutex::new(BTreeMap::new()),
            failures: AtomicU64::new(0),
        }
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
        Ok(promoted)
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
    }

    #[must_use]
    pub fn failure_count(&self) -> u64 {
        self.failures.load(Ordering::Relaxed)
    }
}

impl Default for RuntimeGeometryRegistry {
    fn default() -> Self {
        Self::new(8)
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
        let result = derive_product(&source)
            .map(Arc::new)
            .map_err(|error| Arc::<str>::from(error));
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

fn hex_key(key: RuntimeGeometryKey) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(64);
    for byte in key.0 {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn derive_product(source: &RuntimeGeometrySource) -> Result<RuntimeGeometryProduct, String> {
    let programs = prepare_mesh_programs(MeshProgramInput {
        positions: &source.positions,
        indices: &source.indices,
        uvs: &source.uvs,
        material_ids: &source.material_ids,
    })
    .map_err(|error| format!("derive exact programs: {error}"))?
    .derivation;
    let covered = programs
        .payload
        .as_ref()
        .map(|payload| payload.covered_triangle_indices())
        .unwrap_or_default();
    let mut residual_indices = Vec::new();
    let mut residual_materials = Vec::new();
    let mut residual_sources = Vec::new();
    for (triangle, indices) in source.indices.iter().copied().enumerate() {
        if covered.binary_search(&(triangle as u32)).is_ok() {
            continue;
        }
        residual_indices.push(indices);
        residual_materials.push(source.material_ids[triangle]);
        residual_sources.push(triangle as u32);
    }
    let detail = prepare_runtime_detail(
        &source.positions,
        &source.indices,
        DetailMeshInput {
            positions: &source.positions,
            normals: &source.normals,
            uvs: &source.uvs,
            indices: &residual_indices,
            material_ids: &residual_materials,
            weathering_masks: &source.weathering_masks,
        },
        &residual_sources,
        &[0.125, 0.25, 0.5, 1.0],
    )
    .map_err(|error| format!("derive continuous detail: {error}"))?;
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
        assert!(matches!(
            service.take(key),
            RuntimeGeometryStatus::Ready(_)
        ));
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
}
