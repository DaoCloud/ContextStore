//! Multi-rail descriptor reads for a single client Worker.

use crate::pb;
use crate::rdma::{RdmaClient, RdmaClientConfig, RdmaReadOutcome};
use std::collections::HashSet;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// One local RDMA device and one listener on the node advertised by placement.
#[derive(Clone, Debug)]
pub struct RailRoute {
    /// Stable name used in metrics and errors.
    pub id: String,
    /// Owning node's endpoint returned by `LookupObject`.
    pub advertised_endpoint: String,
    /// Local device and the listener actually dialed over this rail.
    pub connection: RdmaClientConfig,
    /// Initial availability; `RailReader::set_enabled` can change it later.
    pub enabled: bool,
    /// Relative capacity hint used by byte-weighted scheduling.
    pub weight: u32,
}

/// Local RDMA device placement reported by sysfs when available.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RailTopology {
    /// NUMA node from sysfs, when the host reports one.
    pub numa_node: Option<i32>,
    /// PCI bus/device/function address of the local HCA.
    pub pci_bdf: Option<String>,
}

fn read_topology_from(base: &Path, device: &str) -> RailTopology {
    let path = base.join(device).join("device");
    let numa_node = std::fs::read_to_string(path.join("numa_node"))
        .ok()
        .and_then(|text| text.trim().parse::<i32>().ok())
        .filter(|node| *node >= 0);
    let pci_bdf = std::fs::read_link(&path).ok().and_then(|target| {
        target
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .filter(|name| {
                let bytes = name.as_bytes();
                bytes.len() == 12
                    && bytes[4] == b':'
                    && bytes[7] == b':'
                    && bytes[10] == b'.'
                    && bytes.iter().enumerate().all(|(index, byte)| {
                        matches!(index, 4 | 7 | 10) || byte.is_ascii_hexdigit()
                    })
            })
    });
    RailTopology { numa_node, pci_bdf }
}

impl RailRoute {
    /// Configure one local/remote path for an advertised storage endpoint.
    pub fn new(
        id: impl Into<String>,
        advertised_endpoint: impl Into<String>,
        connection: RdmaClientConfig,
    ) -> Self {
        Self {
            id: id.into(),
            advertised_endpoint: advertised_endpoint.into(),
            connection,
            enabled: true,
            weight: 1,
        }
    }

    /// Keep a configured path visible while excluding it from read plans.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Set a positive relative scheduling weight for this path.
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight.max(1);
        self
    }
}

/// Bounds all allocations and transfers made by one reader instance.
#[derive(Clone, Debug)]
pub struct RailLimits {
    /// Maximum concurrent object reads accepted by this reader.
    pub max_active_reads: usize,
    /// Maximum concurrent transfer tasks using one configured rail.
    pub max_active_reads_per_rail: usize,
    /// Maximum final and per-rail staging allocation reserved at once.
    pub max_staging_bytes: u64,
    /// Maximum total MR lengths reserved across active reads.
    pub max_registered_bytes: u64,
    /// Maximum object payload bytes in flight at once.
    pub max_inflight_bytes: u64,
    /// Maximum aggregate payload bytes in flight on one rail.
    pub max_inflight_bytes_per_rail: u64,
    /// TCP control deadline for connect, send, and reply operations.
    pub io_timeout: Duration,
    /// Delay before a failed rail is eligible for a later request.
    pub rail_cooldown: Duration,
}

impl Default for RailLimits {
    fn default() -> Self {
        Self {
            max_active_reads: 8,
            max_active_reads_per_rail: 8,
            max_staging_bytes: 4 * 1024 * 1024 * 1024,
            max_registered_bytes: 4 * 1024 * 1024 * 1024,
            max_inflight_bytes: 4 * 1024 * 1024 * 1024,
            max_inflight_bytes_per_rail: 2 * 1024 * 1024 * 1024,
            io_timeout: Duration::from_secs(30),
            rail_cooldown: Duration::from_secs(10),
        }
    }
}

/// Typed failure returned without publishing a partial object.
#[derive(Debug)]
pub enum RailReadError {
    InvalidPlacement(String),
    ResourceExhausted(String),
    BufferTooSmall { need: usize, have: usize },
    Transport(String),
    StaleDescriptor,
    Incomplete { expected: usize, actual: usize },
    Checksum { stripe: u32 },
    VersionChanged,
    Cancelled,
    WorkerPanic,
}

impl fmt::Display for RailReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlacement(reason) => write!(f, "invalid placement: {reason}"),
            Self::ResourceExhausted(reason) => write!(f, "rail resource limit: {reason}"),
            Self::BufferTooSmall { need, have } => {
                write!(f, "read buffer needs {need} bytes, has {have}")
            }
            Self::Transport(reason) => write!(f, "rail transfer failed: {reason}"),
            Self::StaleDescriptor => write!(f, "object descriptor became stale"),
            Self::Incomplete { expected, actual } => {
                write!(
                    f,
                    "incomplete rail read: expected {expected}, received {actual}"
                )
            }
            Self::Checksum { stripe } => write!(f, "checksum mismatch on stripe {stripe}"),
            Self::VersionChanged => write!(f, "object version changed during rail read"),
            Self::Cancelled => write!(f, "rail read cancelled"),
            Self::WorkerPanic => write!(f, "rail worker panicked"),
        }
    }
}

impl std::error::Error for RailReadError {}

pub(crate) fn same_descriptor_identity(
    left: &pb::ObjectDescriptor,
    right: &pb::ObjectDescriptor,
) -> bool {
    left.key == right.key
        && left.object_handle == right.object_handle
        && left.object_generation == right.object_generation
        && left.content_etag == right.content_etag
        && left.layout_version == right.layout_version
        && left.size == right.size
        && left.is_striped == right.is_striped
        && left.stripe_count == right.stripe_count
        && left.chunk_size == right.chunk_size
}

pub(crate) fn commit_if_unchanged(
    initial: &crate::ObjectLookup,
    current: &crate::ObjectLookup,
    payload: &[u8],
    destination: &mut [u8],
    cancel: Option<&RailCancel>,
) -> Result<usize, RailReadError> {
    if !same_descriptor_identity(&initial.descriptor, &current.descriptor)
        || initial.placement != current.placement
    {
        return Err(RailReadError::VersionChanged);
    }
    let size = usize::try_from(initial.descriptor.size)
        .map_err(|_| RailReadError::InvalidPlacement("object size exceeds address space".into()))?;
    if payload.len() != size {
        return Err(RailReadError::Incomplete {
            expected: size,
            actual: payload.len(),
        });
    }
    if destination.len() < size {
        return Err(RailReadError::BufferTooSmall {
            need: size,
            have: destination.len(),
        });
    }
    if let Some(cancel) = cancel {
        cancel.publish_if_live(|| destination[..size].copy_from_slice(payload))?;
    } else {
        destination[..size].copy_from_slice(payload);
    }
    Ok(size)
}

/// Cooperative request cancellation shared with the caller.
#[derive(Default)]
struct CancelState {
    cancelled: AtomicBool,
    publish_gate: Mutex<()>,
}

#[derive(Clone, Default)]
pub struct RailCancel(Arc<CancelState>);

impl RailCancel {
    /// Request cancellation; already started rail workers still quiesce.
    pub fn cancel(&self) {
        let _gate = self.0.publish_gate.lock().unwrap();
        self.0.cancelled.store(true, Ordering::Release);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::Acquire)
    }

    fn publish_if_live<T>(&self, publish: impl FnOnce() -> T) -> Result<T, RailReadError> {
        let _gate = self.0.publish_gate.lock().unwrap();
        if self.is_cancelled() {
            return Err(RailReadError::Cancelled);
        }
        Ok(publish())
    }
}

#[derive(Clone, Debug)]
struct StripeRead {
    index: u32,
    object_offset: usize,
    length: usize,
    packed_offset: usize,
}

#[derive(Clone, Debug)]
struct RailTask {
    route_index: usize,
    stripes: Vec<StripeRead>,
    packed_len: usize,
    dummy_len: usize,
}

struct RailPlan {
    size: usize,
    chunk_size: usize,
    checksums: Vec<String>,
    tasks: Vec<RailTask>,
}

impl RailPlan {
    fn build(
        descriptor: &pb::ObjectDescriptor,
        placement: &pb::PlacementDescriptor,
        routes: &[RailRoute],
    ) -> Result<Self, RailReadError> {
        let size = usize::try_from(descriptor.size).map_err(|_| {
            RailReadError::InvalidPlacement("object size exceeds address space".into())
        })?;
        if descriptor.key.is_none()
            || placement.key != descriptor.key
            || descriptor.object_handle.is_empty()
            || descriptor.object_generation == 0
            || descriptor.layout_version == 0
        {
            return Err(RailReadError::InvalidPlacement(
                "descriptor identity and placement key do not match".into(),
            ));
        }
        if size == 0 {
            return Err(RailReadError::InvalidPlacement("empty RDMA object".into()));
        }
        let (stripe_count, chunk_size) = if descriptor.is_striped {
            let chunk_size = usize::try_from(descriptor.chunk_size).map_err(|_| {
                RailReadError::InvalidPlacement("chunk size exceeds address space".into())
            })?;
            if chunk_size == 0 || descriptor.stripe_count as usize != size.div_ceil(chunk_size) {
                return Err(RailReadError::InvalidPlacement(
                    "descriptor stripe count and chunk size disagree".into(),
                ));
            }
            (descriptor.stripe_count as usize, chunk_size)
        } else {
            (1, size)
        };
        if stripe_count > u16::MAX as usize || placement.chunks.len() != stripe_count {
            return Err(RailReadError::InvalidPlacement(
                "placement stripe count exceeds the wire limit or is incomplete".into(),
            ));
        }
        let mut ordered = vec![None; stripe_count];
        for chunk in &placement.chunks {
            let index = chunk.stripe_index as usize;
            if index >= stripe_count || ordered[index].is_some() {
                return Err(RailReadError::InvalidPlacement(format!(
                    "stripe {} is duplicate or out of bounds",
                    chunk.stripe_index
                )));
            }
            let offset = index * chunk_size;
            let length = (size - offset).min(chunk_size);
            if chunk.offset != offset as u64
                || chunk.length != length as u64
                || chunk.rdma_endpoint.is_empty()
            {
                return Err(RailReadError::InvalidPlacement(format!(
                    "stripe {index} has a wrong offset, length, or endpoint"
                )));
            }
            ordered[index] = Some(chunk);
        }
        let mut tasks: Vec<RailTask> = routes
            .iter()
            .enumerate()
            .filter(|(_, route)| route.enabled)
            .map(|(route_index, _)| RailTask {
                route_index,
                stripes: Vec::new(),
                packed_len: 0,
                dummy_len: 0,
            })
            .collect();
        let mut checksums = Vec::with_capacity(stripe_count);
        for (index, chunk) in ordered.into_iter().enumerate() {
            let chunk = chunk.ok_or_else(|| {
                RailReadError::InvalidPlacement(format!("missing stripe {index}"))
            })?;
            let offset = index * chunk_size;
            let length = (size - offset).min(chunk_size);
            let selected = tasks
                .iter()
                .enumerate()
                .filter(|(_, task)| {
                    routes[task.route_index].advertised_endpoint == chunk.rdma_endpoint
                })
                .min_by(|(_, left), (_, right)| {
                    let left_weight = routes[left.route_index].weight as u128;
                    let right_weight = routes[right.route_index].weight as u128;
                    ((left.packed_len as u128) * right_weight)
                        .cmp(&((right.packed_len as u128) * left_weight))
                        .then_with(|| right_weight.cmp(&left_weight))
                })
                .map(|(position, _)| position)
                .ok_or_else(|| {
                    RailReadError::InvalidPlacement(format!(
                        "no enabled rail for {}",
                        chunk.rdma_endpoint
                    ))
                })?;
            let task = &mut tasks[selected];
            task.stripes.push(StripeRead {
                index: index as u32,
                object_offset: offset,
                length,
                packed_offset: task.packed_len,
            });
            task.packed_len += length;
            checksums.push(chunk.checksum.clone());
        }
        tasks.retain(|task| !task.stripes.is_empty());
        let checksum_count = checksums
            .iter()
            .filter(|checksum| !checksum.is_empty())
            .count();
        if checksum_count != 0 && checksum_count != stripe_count {
            return Err(RailReadError::InvalidPlacement(
                "placement has only some stripe checksums".into(),
            ));
        }
        if tasks.is_empty() {
            return Err(RailReadError::InvalidPlacement("no rail selected".into()));
        }
        for task in &mut tasks {
            task.dummy_len = if task.stripes.len() == stripe_count {
                0
            } else {
                chunk_size.min(size)
            };
        }
        Ok(Self {
            size,
            chunk_size,
            checksums,
            tasks,
        })
    }
}

trait RailTransport: Sync {
    fn fetch(
        &self,
        route: &RailRoute,
        task: &RailTask,
        descriptor: &pb::ObjectDescriptor,
        timeout: Duration,
    ) -> Result<Vec<u8>, RailReadError>;
}

#[derive(Default)]
struct RailCounters {
    enabled: AtomicBool,
    cooldown_until: Mutex<Option<Instant>>,
    reads_ok: AtomicU64,
    reads_err: AtomicU64,
    bytes: AtomicU64,
    inflight_requests: AtomicU64,
    inflight_bytes: AtomicU64,
    peak_inflight_bytes: AtomicU64,
    registered_bytes: AtomicU64,
    peak_registered_bytes: AtomicU64,
    duration_us: AtomicU64,
}

impl RailCounters {
    fn cooldown_remaining(&self) -> Duration {
        self.cooldown_until
            .lock()
            .unwrap()
            .map(|until| until.saturating_duration_since(Instant::now()))
            .unwrap_or_default()
    }

    fn cooldown_ms(&self) -> u64 {
        self.cooldown_remaining().as_millis() as u64
    }

    fn is_available(&self) -> bool {
        self.enabled.load(Ordering::Acquire) && self.cooldown_remaining().is_zero()
    }
}

struct RailActivityGuard<'a> {
    counters: &'a RailCounters,
    bytes: u64,
    registered: u64,
}

impl<'a> RailActivityGuard<'a> {
    fn new(counters: &'a RailCounters, bytes: u64, registered: u64) -> Self {
        counters.inflight_requests.fetch_add(1, Ordering::Relaxed);
        let inflight = counters.inflight_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        counters
            .peak_inflight_bytes
            .fetch_max(inflight, Ordering::Relaxed);
        let pinned = counters
            .registered_bytes
            .fetch_add(registered, Ordering::Relaxed)
            + registered;
        counters
            .peak_registered_bytes
            .fetch_max(pinned, Ordering::Relaxed);
        Self {
            counters,
            bytes,
            registered,
        }
    }
}

impl Drop for RailActivityGuard<'_> {
    fn drop(&mut self) {
        self.counters
            .inflight_requests
            .fetch_sub(1, Ordering::Relaxed);
        self.counters
            .inflight_bytes
            .fetch_sub(self.bytes, Ordering::Relaxed);
        self.counters
            .registered_bytes
            .fetch_sub(self.registered, Ordering::Relaxed);
    }
}

/// One rail's cumulative and current metrics.
#[derive(Clone, Debug)]
pub struct RailSnapshot {
    /// Configured rail name.
    pub id: String,
    /// Dialed control listener.
    pub listener: String,
    /// Local Verbs device name.
    pub device: String,
    /// Cached NUMA and PCIe placement for the local device.
    pub topology: RailTopology,
    /// Whether this rail is enabled and outside its failure cooldown.
    pub healthy: bool,
    /// Milliseconds until a failed rail becomes selectable again.
    pub cooldown_ms: u64,
    /// Successful stripe-subset requests.
    pub reads_ok: u64,
    /// Failed stripe-subset requests.
    pub reads_err: u64,
    /// Verified payload bytes returned by this rail.
    pub bytes: u64,
    /// Requests currently executing on this rail.
    pub inflight_requests: u64,
    /// Payload bytes currently assigned to executing requests.
    pub inflight_bytes: u64,
    /// Maximum observed in-flight payload bytes.
    pub peak_inflight_bytes: u64,
    /// Reserved MR bytes while a task is active (not a hardware pin counter).
    pub registered_bytes: u64,
    /// Maximum observed reserved MR length.
    pub peak_registered_bytes: u64,
    /// Cumulative task duration in microseconds.
    pub duration_us: u64,
}

#[derive(Default)]
struct BudgetState {
    active_reads: usize,
    staging_bytes: u64,
    registered_bytes: u64,
    inflight_bytes: u64,
    rail_active_reads: Vec<usize>,
    rail_inflight_bytes: Vec<u64>,
}

/// Bounded multi-rail reader; one active request uses one QP per chosen rail.
pub struct RailReader {
    routes: Vec<RailRoute>,
    topologies: Vec<RailTopology>,
    limits: RailLimits,
    counters: Vec<RailCounters>,
    budget: Arc<Mutex<BudgetState>>,
}

struct BudgetGuard {
    state: Arc<Mutex<BudgetState>>,
    staging: u64,
    registered: u64,
    inflight: u64,
    rail_reservations: Vec<(usize, u64)>,
}

impl BudgetGuard {
    fn finish_transfer(&mut self) {
        let mut budget = self.state.lock().unwrap();
        budget.staging_bytes -= self.registered;
        budget.registered_bytes -= self.registered;
        budget.inflight_bytes -= self.inflight;
        for (index, bytes) in self.rail_reservations.drain(..) {
            budget.rail_active_reads[index] -= 1;
            budget.rail_inflight_bytes[index] -= bytes;
        }
        self.staging -= self.registered;
        self.registered = 0;
        self.inflight = 0;
    }
}

impl Drop for BudgetGuard {
    fn drop(&mut self) {
        let mut budget = self.state.lock().unwrap();
        budget.active_reads -= 1;
        budget.staging_bytes -= self.staging;
        budget.registered_bytes -= self.registered;
        budget.inflight_bytes -= self.inflight;
        for (index, bytes) in &self.rail_reservations {
            budget.rail_active_reads[*index] -= 1;
            budget.rail_inflight_bytes[*index] -= *bytes;
        }
    }
}

/// A verified private payload that retains its staging and active-read budget
/// until the caller publishes or discards the bytes.
pub struct StagedRailRead {
    bytes: Vec<u8>,
    _budget: BudgetGuard,
}

impl StagedRailRead {
    /// Complete object bytes in descriptor order.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl RailReader {
    /// Validate routes and create an RDMA reader without opening connections.
    pub fn new(routes: Vec<RailRoute>, limits: RailLimits) -> Result<Self, RailReadError> {
        if routes.is_empty()
            || limits.max_active_reads == 0
            || limits.max_active_reads_per_rail == 0
        {
            return Err(RailReadError::InvalidPlacement(
                "at least one rail and one active read slot are required".into(),
            ));
        }
        let mut ids = HashSet::new();
        let mut paths = HashSet::new();
        let mut local_ports = HashSet::new();
        for route in &routes {
            if route.id.is_empty()
                || route.advertised_endpoint.is_empty()
                || route.connection.endpoint.is_empty()
                || route.weight == 0
                || !ids.insert(route.id.clone())
                || !paths.insert((
                    route.advertised_endpoint.clone(),
                    route.connection.endpoint.clone(),
                ))
                || !local_ports.insert((
                    route.advertised_endpoint.clone(),
                    route.connection.device.clone(),
                    route.connection.port,
                ))
            {
                return Err(RailReadError::InvalidPlacement(
                    "rail IDs, listeners, and local ports must identify distinct paths".into(),
                ));
            }
        }
        let topologies = routes
            .iter()
            .map(|route| {
                read_topology_from(Path::new("/sys/class/infiniband"), &route.connection.device)
            })
            .collect();
        let counters = routes
            .iter()
            .map(|route| RailCounters {
                enabled: AtomicBool::new(route.enabled),
                ..RailCounters::default()
            })
            .collect();
        let route_count = routes.len();
        Ok(Self {
            routes,
            topologies,
            limits,
            counters,
            budget: Arc::new(Mutex::new(BudgetState {
                rail_active_reads: vec![0; route_count],
                rail_inflight_bytes: vec![0; route_count],
                ..BudgetState::default()
            })),
        })
    }

    /// Read per-rail counters without blocking in-flight transfers.
    pub fn snapshots(&self) -> Vec<RailSnapshot> {
        self.routes
            .iter()
            .zip(&self.counters)
            .enumerate()
            .map(|(index, (route, counters))| RailSnapshot {
                id: route.id.clone(),
                listener: route.connection.endpoint.clone(),
                device: route.connection.device.clone(),
                topology: self.topologies[index].clone(),
                healthy: counters.is_available(),
                cooldown_ms: counters.cooldown_ms(),
                reads_ok: counters.reads_ok.load(Ordering::Relaxed),
                reads_err: counters.reads_err.load(Ordering::Relaxed),
                bytes: counters.bytes.load(Ordering::Relaxed),
                inflight_requests: counters.inflight_requests.load(Ordering::Relaxed),
                inflight_bytes: counters.inflight_bytes.load(Ordering::Relaxed),
                peak_inflight_bytes: counters.peak_inflight_bytes.load(Ordering::Relaxed),
                registered_bytes: counters.registered_bytes.load(Ordering::Relaxed),
                peak_registered_bytes: counters.peak_registered_bytes.load(Ordering::Relaxed),
                duration_us: counters.duration_us.load(Ordering::Relaxed),
            })
            .collect()
    }

    /// Enable or disable one Rail without modifying object placement.
    pub fn set_enabled(&self, id: &str, enabled: bool) -> bool {
        let Some(index) = self.routes.iter().position(|route| route.id == id) else {
            return false;
        };
        let counters = &self.counters[index];
        counters.enabled.store(enabled, Ordering::Release);
        if enabled {
            *counters.cooldown_until.lock().unwrap() = None;
        }
        true
    }

    fn reserve(&self, plan: &RailPlan) -> Result<BudgetGuard, RailReadError> {
        let registered = plan
            .tasks
            .iter()
            .try_fold(0u64, |sum, task| {
                task.packed_len
                    .checked_add(task.dummy_len)
                    .and_then(|bytes| sum.checked_add(bytes as u64))
            })
            .ok_or_else(|| RailReadError::ResourceExhausted("registration size overflow".into()))?;
        let staging = registered
            .checked_add(plan.size as u64)
            .ok_or_else(|| RailReadError::ResourceExhausted("staging size overflow".into()))?;
        let inflight = plan.size as u64;
        let rail_reservations: Vec<(usize, u64)> = plan
            .tasks
            .iter()
            .map(|task| (task.route_index, task.packed_len as u64))
            .collect();
        let mut budget = self.budget.lock().unwrap();
        if budget.active_reads >= self.limits.max_active_reads
            || budget.staging_bytes.saturating_add(staging) > self.limits.max_staging_bytes
            || budget.registered_bytes.saturating_add(registered) > self.limits.max_registered_bytes
            || budget.inflight_bytes.saturating_add(inflight) > self.limits.max_inflight_bytes
        {
            return Err(RailReadError::ResourceExhausted(
                "active reads, staging, registration, or in-flight bytes".into(),
            ));
        }
        if rail_reservations.iter().any(|(index, bytes)| {
            budget.rail_active_reads[*index] >= self.limits.max_active_reads_per_rail
                || budget.rail_inflight_bytes[*index].saturating_add(*bytes)
                    > self.limits.max_inflight_bytes_per_rail
        }) {
            return Err(RailReadError::ResourceExhausted(
                "per-rail active tasks or aggregate in-flight bytes".into(),
            ));
        }
        budget.active_reads += 1;
        budget.staging_bytes += staging;
        budget.registered_bytes += registered;
        budget.inflight_bytes += inflight;
        for (index, bytes) in &rail_reservations {
            budget.rail_active_reads[*index] += 1;
            budget.rail_inflight_bytes[*index] += *bytes;
        }
        Ok(BudgetGuard {
            state: Arc::clone(&self.budget),
            staging,
            registered,
            inflight,
            rail_reservations,
        })
    }

    fn read_staged_with<T: RailTransport>(
        &self,
        descriptor: &pb::ObjectDescriptor,
        placement: &pb::PlacementDescriptor,
        transport: &T,
        cancel: Option<&RailCancel>,
    ) -> Result<StagedRailRead, RailReadError> {
        let mut available = self.routes.clone();
        for (route, counters) in available.iter_mut().zip(&self.counters) {
            route.enabled = counters.is_available();
        }
        let plan = RailPlan::build(descriptor, placement, &available)?;
        let mut budget = self.reserve(&plan)?;
        if cancel.is_some_and(RailCancel::is_cancelled) {
            return Err(RailReadError::Cancelled);
        }
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = plan
                .tasks
                .iter()
                .map(|task| {
                    let route = &self.routes[task.route_index];
                    let counters = &self.counters[task.route_index];
                    scope.spawn(move || {
                        let started = Instant::now();
                        let _activity = RailActivityGuard::new(
                            counters,
                            task.packed_len as u64,
                            (task.packed_len + task.dummy_len) as u64,
                        );
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            transport.fetch(route, task, descriptor, self.limits.io_timeout)
                        }))
                        .unwrap_or(Err(RailReadError::WorkerPanic));
                        counters
                            .duration_us
                            .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                        if result.is_ok() {
                            counters.reads_ok.fetch_add(1, Ordering::Relaxed);
                            counters
                                .bytes
                                .fetch_add(task.packed_len as u64, Ordering::Relaxed);
                        } else {
                            counters.reads_err.fetch_add(1, Ordering::Relaxed);
                            if matches!(
                                &result,
                                Err(RailReadError::Transport(_)
                                    | RailReadError::Incomplete { .. }
                                    | RailReadError::WorkerPanic)
                            ) {
                                *counters.cooldown_until.lock().unwrap() =
                                    Some(Instant::now() + self.limits.rail_cooldown);
                            }
                        }
                        result
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap_or(Err(RailReadError::WorkerPanic)))
                .collect::<Vec<_>>()
        });
        if cancel.is_some_and(RailCancel::is_cancelled) {
            return Err(RailReadError::Cancelled);
        }
        let mut staged = Vec::new();
        staged
            .try_reserve_exact(plan.size)
            .map_err(|error| RailReadError::ResourceExhausted(error.to_string()))?;
        staged.resize(plan.size, 0);
        for (task, result) in plan.tasks.iter().zip(results) {
            let packed = result?;
            if packed.len() != task.packed_len {
                return Err(RailReadError::Incomplete {
                    expected: task.packed_len,
                    actual: packed.len(),
                });
            }
            for stripe in &task.stripes {
                staged[stripe.object_offset..stripe.object_offset + stripe.length].copy_from_slice(
                    &packed[stripe.packed_offset..stripe.packed_offset + stripe.length],
                );
            }
        }
        for (index, checksum) in plan.checksums.iter().enumerate() {
            if checksum.is_empty() {
                continue;
            }
            let start = index * plan.chunk_size;
            let end = (start + plan.chunk_size).min(plan.size);
            let actual = format!("{:016x}", twox_hash::xxh3::hash64(&staged[start..end]));
            if !checksum.eq_ignore_ascii_case(&actual) {
                return Err(RailReadError::Checksum {
                    stripe: index as u32,
                });
            }
        }
        budget.finish_transfer();
        Ok(StagedRailRead {
            bytes: staged,
            _budget: budget,
        })
    }

    fn read_into_with<T: RailTransport>(
        &self,
        descriptor: &pb::ObjectDescriptor,
        placement: &pb::PlacementDescriptor,
        destination: &mut [u8],
        transport: &T,
        cancel: Option<&RailCancel>,
    ) -> Result<usize, RailReadError> {
        let size = usize::try_from(descriptor.size).map_err(|_| {
            RailReadError::InvalidPlacement("object size exceeds address space".into())
        })?;
        if destination.len() < size {
            return Err(RailReadError::BufferTooSmall {
                need: size,
                have: destination.len(),
            });
        }
        let staged = self.read_staged_with(descriptor, placement, transport, cancel)?;
        if let Some(cancel) = cancel {
            cancel.publish_if_live(|| destination[..size].copy_from_slice(staged.as_bytes()))?;
        } else {
            destination[..size].copy_from_slice(staged.as_bytes());
        }
        Ok(size)
    }

    /// Return a complete private object buffer after all rails and checksums pass.
    /// Callers that require a stable version must perform a post-read lookup
    /// before publishing it, as `KvClient::read_multi_rail_into` does.
    pub fn read_staged(
        &self,
        descriptor: &pb::ObjectDescriptor,
        placement: &pb::PlacementDescriptor,
        cancel: Option<&RailCancel>,
    ) -> Result<StagedRailRead, RailReadError> {
        self.read_staged_with(descriptor, placement, &VerbsTransport, cancel)
    }

    /// Copy a completed read into the caller's buffer. This low-level method
    /// validates each server-side descriptor request but does not re-lookup the
    /// object after transfer; use `KvClient::read_multi_rail_into` for that.
    pub fn read_into(
        &self,
        descriptor: &pb::ObjectDescriptor,
        placement: &pb::PlacementDescriptor,
        destination: &mut [u8],
        cancel: Option<&RailCancel>,
    ) -> Result<usize, RailReadError> {
        self.read_into_with(descriptor, placement, destination, &VerbsTransport, cancel)
    }
}

fn build_sge_segments(
    task: &RailTask,
    descriptor: &pb::ObjectDescriptor,
    base: u64,
    rkey: u32,
) -> Result<Vec<(u64, u32, u64)>, RailReadError> {
    let registered_len = task
        .packed_len
        .checked_add(task.dummy_len)
        .ok_or_else(|| RailReadError::InvalidPlacement("registered size overflow".into()))?;
    let registered_end = base
        .checked_add(registered_len as u64)
        .ok_or_else(|| RailReadError::InvalidPlacement("registered address overflow".into()))?;
    let mut segments = Vec::with_capacity(descriptor.stripe_count as usize);
    let mut owned = task.stripes.iter().peekable();
    for index in 0..descriptor.stripe_count as usize {
        let offset = index
            .checked_mul(descriptor.chunk_size as usize)
            .ok_or_else(|| RailReadError::InvalidPlacement("stripe offset overflow".into()))?;
        let length = (descriptor.size as usize - offset).min(descriptor.chunk_size as usize);
        let local_offset = if owned
            .peek()
            .is_some_and(|stripe| stripe.index as usize == index)
        {
            owned.next().expect("owned stripe").packed_offset
        } else {
            task.packed_len
        };
        let address = base
            .checked_add(local_offset as u64)
            .ok_or_else(|| RailReadError::InvalidPlacement("SGE address overflow".into()))?;
        if address
            .checked_add(length as u64)
            .is_none_or(|end| end > registered_end)
        {
            return Err(RailReadError::InvalidPlacement(
                "SGE maps outside registered memory".into(),
            ));
        }
        segments.push((address, rkey, length as u64));
    }
    Ok(segments)
}

struct VerbsTransport;

impl RailTransport for VerbsTransport {
    fn fetch(
        &self,
        route: &RailRoute,
        task: &RailTask,
        descriptor: &pb::ObjectDescriptor,
        timeout: Duration,
    ) -> Result<Vec<u8>, RailReadError> {
        let mut client = RdmaClient::connect(route.connection.clone().with_io_timeout(timeout))
            .map_err(|error| RailReadError::Transport(error.to_string()))?;
        let capacity = task
            .packed_len
            .checked_add(task.dummy_len)
            .ok_or_else(|| RailReadError::ResourceExhausted("rail buffer overflow".into()))?;
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(capacity)
            .map_err(|error| RailReadError::ResourceExhausted(error.to_string()))?;
        packed.resize(capacity, 0u8);
        let registered = client
            .register_buffer(&mut packed)
            .map_err(|error| RailReadError::Transport(error.to_string()))?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if descriptor.is_striped {
                let view = registered.view();
                let segments = build_sge_segments(task, descriptor, view.addr(), view.rkey())
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                let indices: Vec<u32> = task.stripes.iter().map(|stripe| stripe.index).collect();
                client.get_descriptor_stripes_sge_detailed(descriptor, &indices, &segments)
            } else {
                client
                    .get_descriptor_into(descriptor, &registered, 0)
                    .map(|outcome| outcome.map(|bytes| RdmaReadOutcome { bytes, chunks: 1 }))
            }
        }));
        drop(client);
        drop(registered);
        let result = match result {
            Ok(result) => result.map_err(|error| RailReadError::Transport(error.to_string()))?,
            Err(panic) => std::panic::resume_unwind(panic),
        };
        let outcome = result.ok_or(RailReadError::StaleDescriptor)?;
        if outcome.bytes != task.packed_len || outcome.chunks != task.stripes.len() as u32 {
            return Err(RailReadError::Incomplete {
                expected: task.packed_len,
                actual: outcome.bytes,
            });
        }
        packed.truncate(task.packed_len);
        Ok(packed)
    }
}

#[cfg(test)]
mod tests;
