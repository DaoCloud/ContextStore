//! Multi-rail RDMA read path: one client Worker reads a single striped object
//! in parallel over several local HCAs ("rails").
//!
//! # Model
//!
//! A *rail* is one local RDMA device (HCA port) plus its slice of per-rail
//! resources: verbs context, PD, CQ, QPs and cached memory registrations. The
//! remote side is addressed by the per-stripe RDMA endpoints carried in a
//! `PlacementDescriptor` — the server already exposes one listener per NIC
//! (`CS_RDMA_DEVICES`), so a rail↔endpoint pairing is simply a connection.
//!
//! Reads keep the existing data path untouched: every connection sends one
//! stripe-subset descriptor GET over its TCP control stream and the server
//! RDMA-WRITEs the requested stripes at `base + stripe_index * chunk_size` of
//! the client-registered destination. Because stripes land in disjoint
//! regions, rails and endpoints transfer concurrently with no client-side
//! reassembly.
//!
//! The destination buffer is registered **once per rail device** (an rkey is
//! only meaningful to the device it was registered on), so a read through N
//! rails holds N MRs of the same memory.
//!
//! # Memory-safety contract (late-write protection)
//!
//! RDMA WRITEs on the GET path are server-initiated; a failed or timed-out
//! control exchange does not prove the WRITE never happened. This module
//! guarantees that [`MultiRailClient::read_object_into`] (and the `_raw`
//! variant) returns only when no work request targeting the caller's buffer
//! can still place data:
//!
//! * success: every task received its tag-3 response, which the server only
//!   sends after its last RDMA WRITE completed — the RC transport has already
//!   placed all bytes;
//! * failure/timeout: the affected connections are quiesced — `Stop` is
//!   queued behind any in-flight command, the worker thread is joined (so the
//!   QP is destroyed and the cached MRs are deregistered, in that order),
//!   and only then does the call return. Stray retransmissions hit a
//!   destroyed QP / invalid rkey and are dropped by the transport instead of
//!   DMA-ing into freed or reused memory.
//!
//! This is the "safe failure, no in-request transparent retry" posture: the
//! first version fails the whole read when a rail breaks; the rail enters a
//! cooldown window and later reads recover by planning around it.
//!
//! # Backpressure
//!
//! [`RailLimits`] bounds total connections, per-rail connections (queue
//! depth), and in-flight bytes (per rail and total). Planning groups tasks
//! into waves so a wave never exceeds the connection caps, and dispatch
//! blocks — with a deadline — until byte headroom exists.
//!
//! # Integrity
//!
//! Every task carries the full object descriptor (handle / generation / ETag
//! / layout version); the server rejects mismatches by answering
//! `found=false`, which surfaces here as [`MultiRailError::StaleDescriptor`].
//! After the bytes arrive the client re-derives the expected per-stripe byte
//! count from the placement, cross-checks the server-reported `num_chunks`,
//! and, when the placement carries per-stripe xxh3-64 checksums, verifies
//! every stripe it received. Missing, duplicated, or out-of-bounds stripes
//! are detected while planning, before any network I/O is issued.

use crate::pb;
use crate::rdma::{self, RdmaClient, RdmaClientConfig};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Cooldown applied to a rail after a failure before it is considered for
/// planning again (path recovery window).
const DEFAULT_RAIL_COOLDOWN: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// One local rail: an RDMA device + port + GID to use for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RailConfig {
    /// Verbs device name, e.g. `mlx5_0` or `rxe0`.
    pub device: String,
    /// HCA port number (almost always 1).
    pub port: u8,
    /// GID index for the address handle.
    pub gid_index: u8,
    /// Relative share of stripes this rail receives (load-balancing weight).
    pub weight: u32,
    /// RC path MTU in bytes; 4096 on jumbo-frame fabrics, 1024 (default) on
    /// standard 1500-byte networks.
    pub mtu: u16,
    /// GRH hop limit; routed RoCE fabrics need more than the default 1.
    pub hop_limit: u8,
    /// Optional endpoint whitelist (`host:port` or bare `host`). Empty means
    /// the rail may serve any endpoint. Rail-optimized fabrics pin rail k to
    /// fabric k; soft-RoCE cross-wired testbeds use it the same way.
    pub endpoints: Vec<String>,
}

impl RailConfig {
    pub fn new(device: impl Into<String>) -> Self {
        Self {
            device: device.into(),
            port: 1,
            gid_index: 3,
            weight: 1,
            mtu: 1024,
            hop_limit: 1,
            endpoints: Vec::new(),
        }
    }

    pub fn with_port(mut self, port: u8) -> Self {
        self.port = port;
        self
    }

    pub fn with_gid_index(mut self, gid_index: u8) -> Self {
        self.gid_index = gid_index;
        self
    }

    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight.max(1);
        self
    }

    /// Set the RC path MTU (bytes; 4096 for jumbo-frame fabrics).
    pub fn with_mtu(mut self, bytes: u16) -> Self {
        self.mtu = bytes;
        self
    }

    /// Set the GRH hop limit (routed RoCE needs more than 1).
    pub fn with_hop_limit(mut self, hops: u8) -> Self {
        self.hop_limit = hops;
        self
    }

    /// Restrict this rail to the given endpoints (`host:port` or bare host).
    pub fn with_endpoints(mut self, endpoints: Vec<String>) -> Self {
        self.endpoints = endpoints;
        self
    }

    /// Whether this rail may serve `endpoint` (exact `host:port` or host-only
    /// match; an empty whitelist allows everything).
    fn allows_endpoint(&self, endpoint: &str) -> bool {
        if self.endpoints.is_empty() {
            return true;
        }
        let host = endpoint
            .rsplit_once(':')
            .map(|(h, _)| h)
            .unwrap_or(endpoint);
        self.endpoints
            .iter()
            .any(|allowed| allowed == endpoint || *allowed == host)
    }

    /// Parse `device[:port[:gid[:weight[:mtu]]]][@ep[;ep...]]` (omitted
    /// parts keep defaults). The optional `@` suffix pins the rail to an
    /// endpoint whitelist (`host:port` or bare host, `;`-separated).
    pub fn parse(spec: &str) -> Option<Self> {
        let (spec, endpoints) = match spec.split_once('@') {
            Some((left, right)) => (
                left,
                right
                    .split(';')
                    .map(|ep| ep.trim().to_string())
                    .filter(|ep| !ep.is_empty())
                    .collect::<Vec<_>>(),
            ),
            None => (spec, Vec::new()),
        };
        let mut parts = spec.split(':');
        let device = parts.next()?.trim();
        if device.is_empty() {
            return None;
        }
        let mut config = Self::new(device);
        if let Some(port) = parts.next().and_then(|p| p.trim().parse().ok()) {
            config = config.with_port(port);
        }
        if let Some(gid) = parts.next().and_then(|g| g.trim().parse().ok()) {
            config = config.with_gid_index(gid);
        }
        if let Some(weight) = parts.next().and_then(|w| w.trim().parse().ok()) {
            config = config.with_weight(weight);
        }
        if let Some(mtu) = parts.next().and_then(|m| m.trim().parse().ok()) {
            config = config.with_mtu(mtu);
        }
        if !endpoints.is_empty() {
            config = config.with_endpoints(endpoints);
        }
        Some(config)
    }
}

/// Resource and backpressure limits for multi-rail reads.
#[derive(Clone, Debug)]
pub struct RailLimits {
    /// Maximum simultaneous connections (queue depth) one rail may hold.
    pub max_connections_per_rail: usize,
    /// Maximum simultaneous connections across all rails.
    pub max_connections_total: usize,
    /// In-flight bytes one rail may accumulate before dispatch blocks.
    pub max_inflight_bytes_per_rail: u64,
    /// In-flight bytes across all rails before dispatch blocks.
    pub max_inflight_bytes_total: u64,
    /// Cap on stripes per task: an endpoint's stripes assigned to one rail
    /// are split into tasks of at most this many stripes, each on its own
    /// connection (per-rail concurrency / queue depth). 0 = unlimited.
    pub task_max_stripes: usize,
    /// Per-operation TCP control-channel timeout; also bounds connect.
    pub io_timeout: Duration,
    /// How long a failed rail is skipped before it is retried.
    pub rail_cooldown: Duration,
}

impl Default for RailLimits {
    fn default() -> Self {
        Self {
            max_connections_per_rail: 8,
            max_connections_total: 32,
            max_inflight_bytes_per_rail: 8 * 1024 * 1024 * 1024u64,
            max_inflight_bytes_total: 32 * 1024 * 1024 * 1024u64,
            task_max_stripes: 0,
            io_timeout: Duration::from_secs(30),
            rail_cooldown: DEFAULT_RAIL_COOLDOWN,
        }
    }
}

/// Stripe→rail assignment strategy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RailSelectPolicy {
    /// Spread every endpoint's stripes over all healthy rails, weighted by
    /// rail weight and current assignment load (default; maximum fan-out).
    #[default]
    LeastLoaded,
    /// Pin each endpoint to a single rail — prefer a subnet match between the
    /// rail GID and the endpoint IP, then the least-loaded rail. Fewer
    /// connections, cleaner topology, no fan-out within one endpoint.
    EndpointAffinity,
    /// Pure weighted round-robin over healthy rails.
    WeightedRoundRobin,
}

impl RailSelectPolicy {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "least-loaded" | "least_loaded" | "leastloaded" => Some(Self::LeastLoaded),
            "endpoint-affinity" | "endpoint_affinity" | "affinity" => Some(Self::EndpointAffinity),
            "weighted-round-robin" | "round-robin" | "rr" => Some(Self::WeightedRoundRobin),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Typed failures of a multi-rail read. The `StaleDescriptor` variant means
/// the server rejected the descriptor identity (generation / ETag / layout
/// changed) and the caller should re-run the gRPC lookup before retrying.
#[derive(Debug)]
pub enum MultiRailError {
    NoRails,
    NoHealthyRails,
    BufferTooSmall {
        need: u64,
        have: usize,
    },
    InvalidPlacement(String),
    MissingStripes(Vec<u32>),
    DuplicateStripes(Vec<u32>),
    OutOfBoundsStripe {
        stripe: u32,
        stripe_count: u32,
    },
    StaleDescriptor {
        endpoint: String,
    },
    TaskFailed {
        rail: String,
        endpoint: String,
        source: String,
    },
    Timeout {
        rail: String,
        endpoint: String,
        after_ms: u128,
    },
    ByteCountMismatch {
        rail: String,
        endpoint: String,
        expected: u64,
        actual: u64,
    },
    ChunkCountMismatch {
        rail: String,
        endpoint: String,
        expected: u32,
        actual: u32,
    },
    ChecksumMismatch {
        stripe: u32,
        expected: String,
        actual: String,
    },
    WorkerPanic {
        rail: String,
        endpoint: String,
    },
    Cancelled,
}

impl fmt::Display for MultiRailError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRails => write!(f, "multi-rail client has no rails configured"),
            Self::NoHealthyRails => write!(f, "all rails are unhealthy (in cooldown)"),
            Self::BufferTooSmall { need, have } => write!(
                f,
                "destination buffer too small: need {need} bytes, have {have}"
            ),
            Self::InvalidPlacement(reason) => write!(f, "invalid placement: {reason}"),
            Self::MissingStripes(stripes) => {
                write!(f, "placement is missing stripes {stripes:?}")
            }
            Self::DuplicateStripes(stripes) => {
                write!(f, "placement duplicates stripes {stripes:?}")
            }
            Self::OutOfBoundsStripe { stripe, stripe_count } => write!(
                f,
                "stripe index {stripe} out of bounds for stripe count {stripe_count}"
            ),
            Self::StaleDescriptor { endpoint } => write!(
                f,
                "server at {endpoint} rejected the descriptor (stale generation/etag/layout); re-lookup required"
            ),
            Self::TaskFailed {
                rail,
                endpoint,
                source,
            } => write!(f, "rail {rail} failed reading from {endpoint}: {source}"),
            Self::Timeout {
                rail,
                endpoint,
                after_ms,
            } => write!(f, "rail {rail} timed out reading from {endpoint} after {after_ms}ms"),
            Self::ByteCountMismatch {
                rail,
                endpoint,
                expected,
                actual,
            } => write!(
                f,
                "rail {rail} got {actual} bytes from {endpoint}, expected {expected}"
            ),
            Self::ChunkCountMismatch {
                rail,
                endpoint,
                expected,
                actual,
            } => write!(
                f,
                "rail {rail}: server at {endpoint} served {actual} chunks, expected {expected}"
            ),
            Self::ChecksumMismatch {
                stripe,
                expected,
                actual,
            } => write!(
                f,
                "checksum mismatch on stripe {stripe}: expected {expected}, got {actual}"
            ),
            Self::WorkerPanic { rail, endpoint } => {
                write!(f, "rail {rail} worker for {endpoint} panicked")
            }
            Self::Cancelled => {
                write!(f, "read cancelled by caller")
            }
        }
    }
}

impl std::error::Error for MultiRailError {}

/// Cooperative cancellation for multi-rail reads.
///
/// Clones share the same flag. Cancellation is observed at wave boundaries,
/// during backpressure waits, and between reply-collection cycles; the
/// in-flight tasks of the current wave are still drained (bounded by the
/// read deadline) and every participating connection is quiesced before the
/// read returns, so cancelling never violates the memory-safety contract.
#[derive(Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    fn check(&self) -> Result<(), MultiRailError> {
        if self.is_cancelled() {
            Err(MultiRailError::Cancelled)
        } else {
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------

/// Host topology attributes of a rail, read from sysfs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RailTopology {
    /// NUMA node the device is attached to (-1 when unknown).
    pub numa_node: i32,
    /// PCIe BDF, e.g. `0000:60:00.0` (empty when unknown).
    pub pci_slot: String,
}

impl fmt::Display for RailTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "numa={}",
            if self.numa_node < 0 {
                "?".to_string()
            } else {
                self.numa_node.to_string()
            }
        )?;
        if !self.pci_slot.is_empty() {
            write!(f, " pci={}", self.pci_slot)?;
        }
        Ok(())
    }
}

/// Read a rail's topology from `<base>/<device>/device`.
fn topology_from(base: &Path, device: &str) -> RailTopology {
    let dev_dir = base.join(device);
    let numa = std::fs::read_to_string(dev_dir.join("device/numa_node"))
        .ok()
        .and_then(|text| text.trim().parse::<i32>().ok())
        .unwrap_or(-1);
    let pci_slot = dev_dir
        .canonicalize()
        .ok()
        .and_then(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .unwrap_or_default();
    RailTopology {
        numa_node: numa,
        pci_slot,
    }
}

/// Read a rail's topology from the real sysfs tree.
pub fn read_topology(device: &str) -> RailTopology {
    topology_from(Path::new("/sys/class/infiniband"), device)
}

// ---------------------------------------------------------------------------
// Per-rail stats & state
// ---------------------------------------------------------------------------

#[derive(Default)]
struct RailStats {
    requests_ok: AtomicU64,
    requests_err: AtomicU64,
    bytes_read: AtomicU64,
    timeouts: AtomicU64,
    connections_created: AtomicU64,
    connections_quiesced: AtomicU64,
    inflight_requests: AtomicU64,
    inflight_bytes: AtomicU64,
    /// Sum and max of per-request latencies (µs); avg = sum / requests_ok.
    latency_us_sum: AtomicU64,
    latency_us_max: AtomicU64,
    /// Bytes currently pinned by cached memory registrations on this rail.
    registered_bytes: AtomicU64,
}

/// Point-in-time view of one rail for observability.
#[derive(Clone, Debug)]
pub struct RailSnapshot {
    pub index: usize,
    pub device: String,
    pub topology: RailTopology,
    pub healthy: bool,
    pub cooldown_ms_remaining: u64,
    pub requests_ok: u64,
    pub requests_err: u64,
    pub bytes_read: u64,
    pub timeouts: u64,
    pub connections_created: u64,
    pub connections_quiesced: u64,
    pub inflight_requests: u64,
    pub inflight_bytes: u64,
    pub latency_avg_us: u64,
    pub latency_max_us: u64,
    pub registered_bytes: u64,
}

impl fmt::Display for RailSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "rail[{}] {} {} healthy={} cooldown={}ms ok={} err={} tmo={} read={}MiB conns=+{}/~{} inflight=(req:{},B:{}) lat=(avg:{}us,max:{}us) reg={}MiB",
            self.index,
            self.device,
            self.topology,
            self.healthy,
            self.cooldown_ms_remaining,
            self.requests_ok,
            self.requests_err,
            self.timeouts,
            self.bytes_read / (1024 * 1024),
            self.connections_created,
            self.connections_quiesced,
            self.inflight_requests,
            self.inflight_bytes,
            self.latency_avg_us,
            self.latency_max_us,
            self.registered_bytes >> 20,
        )
    }
}

struct Rail {
    index: usize,
    config: RailConfig,
    topology: RailTopology,
    stats: RailStats,
    healthy: AtomicBool,
    unhealthy_until: Mutex<Option<Instant>>,
    /// IPv4 address embedded in the rail GID, when the GID is an IPv4-mapped
    /// RoCE v2 GID (used for endpoint affinity).
    gid_v4: Option<std::net::Ipv4Addr>,
}

impl Rail {
    fn mark_unhealthy(&self, cooldown: Duration) {
        self.healthy.store(false, Ordering::Release);
        *self.unhealthy_until.lock().unwrap() = Some(Instant::now() + cooldown);
    }

    fn cooldown_remaining(&self) -> Duration {
        self.unhealthy_until
            .lock()
            .unwrap()
            .map(|until| until.saturating_duration_since(Instant::now()))
            .unwrap_or_default()
    }

    /// A rail is selectable when healthy, or when its cooldown expired.
    fn selectable(&self) -> bool {
        if self.healthy.load(Ordering::Acquire) {
            return true;
        }
        if self.cooldown_remaining().is_zero() {
            // Cooldown over: allow retry and restore health optimistically;
            // a new failure puts it straight back into cooldown.
            *self.unhealthy_until.lock().unwrap() = None;
            self.healthy.store(true, Ordering::Release);
            return true;
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Placement validation & read planning
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct StripeInfo {
    endpoint: Arc<str>,
    offset: u64,
    length: u64,
    checksum: Option<String>,
}

#[derive(Debug)]
struct ValidatedPlacement {
    is_striped: bool,
    stripe_count: u32,
    object_size: u64,
    /// Stripe index → placement record. Covers exactly `0..stripe_count` for
    /// striped objects; a single implicit stripe 0 for plain objects.
    stripes: BTreeMap<u32, StripeInfo>,
    expected_bytes: u64,
}

/// Byte length of `stripe` under a striped layout (last stripe is short).
fn stripe_length(stripe: u32, stripe_count: u32, chunk_size: u64, size: u64) -> Option<u64> {
    if chunk_size == 0 || stripe >= stripe_count {
        return None;
    }
    let base = stripe as u64 * chunk_size;
    if base >= size {
        return None;
    }
    Some((size - base).min(chunk_size))
}

/// Validate descriptor ↔ placement consistency: every stripe present exactly
/// once, indices in range, offsets/lengths matching the striped layout, and
/// every stripe owned by a non-empty RDMA endpoint. Detects the missing /
/// duplicated / out-of-bounds cases before any network I/O happens.
fn validate_placement(
    descriptor: &pb::ObjectDescriptor,
    chunks: &[pb::PlacementChunk],
) -> Result<ValidatedPlacement, MultiRailError> {
    let object_size = descriptor.size;
    if chunks.is_empty() {
        return Err(MultiRailError::InvalidPlacement(
            "placement contains no chunks".into(),
        ));
    }
    if descriptor.is_striped && (descriptor.chunk_size == 0 || descriptor.stripe_count == 0) {
        return Err(MultiRailError::InvalidPlacement(
            "striped descriptor with zero chunk_size/stripe_count".into(),
        ));
    }

    let stripe_count = if descriptor.is_striped {
        descriptor.stripe_count
    } else {
        1
    };
    let chunk_size = if descriptor.is_striped {
        descriptor.chunk_size
    } else {
        object_size
    };

    let mut stripes: BTreeMap<u32, StripeInfo> = BTreeMap::new();
    let mut expected_bytes = 0u64;
    for chunk in chunks {
        let stripe = chunk.stripe_index;
        if stripe >= stripe_count {
            return Err(MultiRailError::OutOfBoundsStripe {
                stripe,
                stripe_count,
            });
        }
        if chunk.rdma_endpoint.is_empty() {
            return Err(MultiRailError::InvalidPlacement(format!(
                "stripe {stripe} has no RDMA endpoint"
            )));
        }
        let expected_len = if descriptor.is_striped {
            stripe_length(stripe, stripe_count, chunk_size, object_size).ok_or(
                MultiRailError::OutOfBoundsStripe {
                    stripe,
                    stripe_count,
                },
            )?
        } else {
            object_size
        };
        if chunk.length != 0 && chunk.length != expected_len {
            return Err(MultiRailError::InvalidPlacement(format!(
                "stripe {stripe} length {} does not match layout expectation {expected_len}",
                chunk.length
            )));
        }
        let expected_offset = stripe as u64 * chunk_size;
        if chunk.offset != 0 && chunk.offset != expected_offset {
            return Err(MultiRailError::InvalidPlacement(format!(
                "stripe {stripe} offset {} does not match natural position {expected_offset}",
                chunk.offset
            )));
        }
        if stripes
            .insert(
                stripe,
                StripeInfo {
                    endpoint: Arc::from(chunk.rdma_endpoint.as_str()),
                    offset: expected_offset,
                    length: expected_len,
                    checksum: if chunk.checksum.is_empty() {
                        None
                    } else {
                        Some(chunk.checksum.clone())
                    },
                },
            )
            .is_some()
        {
            return Err(MultiRailError::DuplicateStripes(vec![stripe]));
        }
        expected_bytes += expected_len;
    }

    let missing: Vec<u32> = (0..stripe_count)
        .filter(|index| !stripes.contains_key(index))
        .collect();
    if !missing.is_empty() {
        return Err(MultiRailError::MissingStripes(missing));
    }

    Ok(ValidatedPlacement {
        is_striped: descriptor.is_striped,
        stripe_count,
        object_size,
        stripes,
        expected_bytes,
    })
}

/// One unit of dispatch: a stripe subset read from one endpoint over one rail.
#[derive(Debug)]
struct TaskSpec {
    rail_index: usize,
    endpoint: Arc<str>,
    stripes: Vec<u32>,
    bytes: u64,
}

/// The planned distribution of a read across rails.
#[derive(Debug)]
pub struct ReadPlan {
    tasks: Vec<TaskSpec>,
    expected_bytes: u64,
    stripe_count: u32,
}

impl ReadPlan {
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    pub fn expected_bytes(&self) -> u64 {
        self.expected_bytes
    }

    pub fn stripe_count(&self) -> u32 {
        self.stripe_count
    }
}

/// Extract the IPv4 host part of an `host:port` endpoint string.
fn endpoint_v4(endpoint: &str) -> Option<std::net::Ipv4Addr> {
    let host = endpoint
        .rsplit_once(':')
        .map(|(host, _)| host)
        .unwrap_or(endpoint);
    host.parse().ok()
}

/// If the GID bytes are an IPv4-mapped RoCE v2 GID, return the v4 address.
fn gid_to_v4(raw: [u8; 16]) -> Option<std::net::Ipv4Addr> {
    if raw[..10].iter().all(|b| *b == 0) && raw[10] == 0xff && raw[11] == 0xff {
        return Some(std::net::Ipv4Addr::new(raw[12], raw[13], raw[14], raw[15]));
    }
    None
}

/// Pick the candidate rail with the smallest `assigned / weight` fraction,
/// comparing by cross-multiplication (no integer division rounding) and
/// breaking ties toward the higher-weight rail.
fn least_loaded(candidates: &[usize], rails: &[Arc<Rail>], assigned: &[u64]) -> usize {
    let mut best = candidates[0];
    for &candidate in &candidates[1..] {
        // assigned ≤ 2^40 and weight ≤ 2^32, so the products fit u128.
        let lhs = (assigned[candidate] as u128) * (rails[best].config.weight as u128);
        let rhs = (assigned[best] as u128) * (rails[candidate].config.weight as u128);
        if lhs < rhs || (lhs == rhs && rails[candidate].config.weight > rails[best].config.weight) {
            best = candidate;
        }
    }
    best
}

/// Choose the rail for one stripe (or one endpoint, under `EndpointAffinity`)
/// out of `candidates` (indices into `rails`, pre-filtered by endpoint
/// whitelists).
fn pick_rail(
    endpoint: &str,
    rails: &[Arc<Rail>],
    candidates: &[usize],
    assigned: &[u64],
    policy: RailSelectPolicy,
    rr: &mut u64,
) -> usize {
    match policy {
        RailSelectPolicy::WeightedRoundRobin => {
            let total: u32 = candidates.iter().map(|&i| rails[i].config.weight).sum();
            if total == 0 || candidates.is_empty() {
                return candidates.first().copied().unwrap_or(0);
            }
            let mut slot = *rr % total as u64;
            *rr += 1;
            for &index in candidates {
                let weight = rails[index].config.weight as u64;
                if slot < weight {
                    return index;
                }
                slot -= weight;
            }
            candidates[0]
        }
        RailSelectPolicy::EndpointAffinity => {
            // Prefer rails on the same /24 as the endpoint, then the least
            // loaded relative to weight.
            let mut shortlist: Vec<usize> = candidates.to_vec();
            if let Some(ip) = endpoint_v4(endpoint) {
                let matched: Vec<usize> = candidates
                    .iter()
                    .copied()
                    .filter(|&index| {
                        rails[index]
                            .gid_v4
                            .is_some_and(|rail_ip| rail_ip.octets()[..3] == ip.octets()[..3])
                    })
                    .collect();
                if !matched.is_empty() {
                    shortlist = matched;
                }
            }
            least_loaded(&shortlist, rails, assigned)
        }
        RailSelectPolicy::LeastLoaded => least_loaded(candidates, rails, assigned),
    }
}

/// Build the task list for a validated placement: group stripes by endpoint,
/// then distribute each endpoint's stripes over the rails allowed to reach
/// that endpoint, according to the policy, keeping one task per (rail,
/// endpoint) pair so the byte count per task can be verified afterwards.
/// Total bytes of one stripe batch according to the validated placement.
fn stripe_batch_bytes(stripes: &[u32], placement: &ValidatedPlacement) -> u64 {
    stripes
        .iter()
        .filter_map(|stripe| placement.stripes.get(stripe))
        .map(|info| info.length)
        .sum()
}

fn build_plan(
    placement: &ValidatedPlacement,
    rails: &[Arc<Rail>],
    policy: RailSelectPolicy,
    limits: &RailLimits,
) -> Result<ReadPlan, MultiRailError> {
    // Group stripes by endpoint first.
    let mut by_endpoint: BTreeMap<Arc<str>, Vec<(u32, u64)>> = BTreeMap::new();
    for (stripe, info) in &placement.stripes {
        by_endpoint
            .entry(Arc::clone(&info.endpoint))
            .or_default()
            .push((*stripe, info.length));
    }

    let mut assigned = vec![0u64; rails.len()];
    let mut rr = 0u64;
    let mut tasks: Vec<TaskSpec> = Vec::new();
    for (endpoint, mut stripes) in by_endpoint {
        let candidates: Vec<usize> = rails
            .iter()
            .enumerate()
            .filter(|(_, rail)| rail.config.allows_endpoint(&endpoint))
            .map(|(index, _)| index)
            .collect();
        if candidates.is_empty() {
            return Err(MultiRailError::InvalidPlacement(format!(
                "no rail is allowed to reach endpoint {endpoint} (check rail endpoint whitelists)"
            )));
        }
        stripes.sort_unstable();
        match policy {
            // Non-striped objects are read whole: the server's stripe-subset
            // path rejects them, so the task carries an empty stripe list and
            // falls back to the full-object descriptor GET.
            _ if !placement.is_striped => {
                tasks.push(TaskSpec {
                    rail_index: candidates[0],
                    endpoint,
                    stripes: Vec::new(),
                    bytes: placement.expected_bytes,
                });
            }
            // Split this endpoint's stripes over every allowed rail.
            RailSelectPolicy::LeastLoaded | RailSelectPolicy::WeightedRoundRobin => {
                let mut per_rail: Vec<Vec<u32>> = vec![Vec::new(); rails.len()];
                let mut bytes_per_rail = vec![0u64; rails.len()];
                for (stripe, length) in stripes {
                    let rail = pick_rail(&endpoint, rails, &candidates, &assigned, policy, &mut rr);
                    per_rail[rail].push(stripe);
                    bytes_per_rail[rail] += length;
                    assigned[rail] += length;
                }
                for (rail_index, (stripe_list, _bytes)) in
                    per_rail.into_iter().zip(bytes_per_rail).enumerate()
                {
                    if stripe_list.is_empty() {
                        continue;
                    }
                    // Split into per-connection batches for intra-rail
                    // concurrency (bounded queue depth per rail).
                    let batch = if limits.task_max_stripes == 0 {
                        stripe_list.len()
                    } else {
                        limits.task_max_stripes.max(1)
                    };
                    for chunk in stripe_list.chunks(batch) {
                        tasks.push(TaskSpec {
                            rail_index,
                            endpoint: Arc::clone(&endpoint),
                            stripes: chunk.to_vec(),
                            bytes: stripe_batch_bytes(chunk, placement),
                        });
                    }
                }
            }
            // Pin the whole endpoint to one allowed rail.
            RailSelectPolicy::EndpointAffinity => {
                let rail = pick_rail(&endpoint, rails, &candidates, &assigned, policy, &mut rr);
                let bytes: u64 = stripes.iter().map(|(_, length)| *length).sum();
                assigned[rail] += bytes;
                tasks.push(TaskSpec {
                    rail_index: rail,
                    endpoint,
                    stripes: stripes.into_iter().map(|(stripe, _)| stripe).collect(),
                    bytes,
                });
            }
        }
    }

    Ok(ReadPlan {
        tasks,
        expected_bytes: placement.expected_bytes,
        stripe_count: placement.stripe_count,
    })
}

/// Group tasks into dispatch waves that respect the connection caps. Tasks
/// never share a (rail, endpoint) pair within one plan, so every task in a
/// wave is a distinct connection.
fn plan_waves(tasks: Vec<TaskSpec>, limits: &RailLimits) -> Vec<Vec<TaskSpec>> {
    let mut waves: Vec<Vec<TaskSpec>> = Vec::new();
    let mut current: Vec<TaskSpec> = Vec::new();
    let mut rail_conns: HashMap<usize, usize> = HashMap::new();
    for task in tasks {
        let rail_ok = rail_conns.get(&task.rail_index).copied().unwrap_or(0)
            < limits.max_connections_per_rail.max(1);
        let total_ok = current.len() < limits.max_connections_total.max(1);
        if !rail_ok || !total_ok {
            waves.push(std::mem::take(&mut current));
            rail_conns.clear();
        }
        *rail_conns.entry(task.rail_index).or_insert(0) += 1;
        current.push(task);
    }
    if !current.is_empty() {
        waves.push(current);
    }
    if waves.is_empty() {
        waves.push(Vec::new());
    }
    waves
}

// ---------------------------------------------------------------------------
// Connection workers
// ---------------------------------------------------------------------------

enum Command {
    Read {
        descriptor: Arc<pb::ObjectDescriptor>,
        stripes: Arc<Vec<u32>>,
        dst_base: usize,
        dst_len: usize,
        reply: mpsc::Sender<TaskReply>,
    },
    EvictRegistration {
        base: usize,
        ack: mpsc::Sender<()>,
    },
    Stop,
}

/// Result of one dispatched task. The worker computes the *expected* byte and
/// chunk counts from the descriptor + requested stripes so the dispatcher can
/// verify completeness without external bookkeeping.
struct TaskReply {
    rail_index: usize,
    endpoint: Arc<str>,
    expected_bytes: u64,
    expected_chunks: u32,
    outcome: Result<Option<rdma::GetOutcome>, String>,
}

impl TaskReply {
    /// Best-effort classification of transport errors caused by timeouts.
    fn is_timeout(&self) -> bool {
        match &self.outcome {
            Err(message) => {
                message.contains("timed out")
                    || message.contains("WouldBlock")
                    || message.contains("etimedout")
                    || message.contains("ETIMEDOUT")
                    || message.contains("os error 110")
            }
            Ok(_) => false,
        }
    }
}

struct ConnEntry {
    rail_index: usize,
    tx: mpsc::Sender<Command>,
    handle: JoinHandle<()>,
}

/// Expected bytes/chunks for one stripe-subset request against `descriptor`.
fn expected_task_outcome(descriptor: &pb::ObjectDescriptor, stripes: &[u32]) -> (u64, u32) {
    if stripes.is_empty() {
        // Non-striped (or whole-object) GET: one implicit chunk.
        return (descriptor.size, 1);
    }
    let mut total = 0u64;
    for &stripe in stripes {
        let length = if descriptor.is_striped {
            stripe_length(
                stripe,
                descriptor.stripe_count,
                descriptor.chunk_size,
                descriptor.size,
            )
            .unwrap_or(0)
        } else {
            descriptor.size
        };
        total += length;
    }
    (total, stripes.len() as u32)
}

fn connect_rail_client(
    rail: &Rail,
    endpoint: &str,
    limits: &RailLimits,
) -> Result<RdmaClient, String> {
    let config = RdmaClientConfig::new(endpoint.to_string(), rail.config.device.clone())
        .with_port(rail.config.port)
        .with_gid_index(rail.config.gid_index)
        .with_io_timeout(limits.io_timeout)
        .with_connect_timeout(limits.io_timeout);
    RdmaClient::connect(config).map_err(|error| error.to_string())
}

/// Worker body for one (rail, endpoint) connection. Owns the `RdmaClient`
/// exclusively; commands arrive serialized, so the TCP control channel
/// invariant of one request in flight per connection holds.
///
/// Teardown ordering is the memory-safety core: `Stop` exits the loop, then
/// `RdmaClient` drops — BYE + `ibv_destroy_qp` first, cached MRs
/// deregistered after (each MR's `Arc<RdmaResources>` keeps the verbs context
/// alive until the last registration is gone).
fn conn_worker(
    rail: Arc<Rail>,
    endpoint: Arc<str>,
    limits: RailLimits,
    rx: mpsc::Receiver<Command>,
    setup_tx: mpsc::Sender<Result<(), String>>,
) {
    let mut client = match connect_rail_client(&rail, &endpoint, &limits) {
        Ok(client) => {
            rail.stats
                .connections_created
                .fetch_add(1, Ordering::Relaxed);
            let _ = setup_tx.send(Ok(()));
            client
        }
        Err(error) => {
            let _ = setup_tx.send(Err(error));
            // Drain until Stop so the dispatcher's join never blocks on a
            // command nobody will read.
            for command in rx.try_iter() {
                if matches!(command, Command::Stop) {
                    break;
                }
            }
            return;
        }
    };

    while let Ok(command) = rx.recv() {
        match command {
            Command::Read {
                descriptor,
                stripes,
                dst_base,
                dst_len,
                reply,
            } => {
                let (expected_bytes, expected_chunks) =
                    expected_task_outcome(&descriptor, &stripes);
                let started = Instant::now();
                let outcome = (|| -> Result<Option<rdma::GetOutcome>, String> {
                    // SAFETY: dst_base..dst_base+dst_len stays valid and
                    // unmoved for the whole read call — the dispatcher joins
                    // every worker before returning, and registrations for
                    // non-sticky buffers are evicted synchronously at the end
                    // of each read.
                    let before = client.registered_cache_bytes();
                    let view = unsafe {
                        client
                            .register_raw_buffer_cached(dst_base as *mut u8, dst_len)
                            .map_err(|error| error.to_string())?
                    };
                    let registered_now = client.registered_cache_bytes();
                    if registered_now > before {
                        rail.stats
                            .registered_bytes
                            .fetch_add((registered_now - before) as u64, Ordering::Relaxed);
                    }
                    client
                        .get_descriptor_stripes_into_view_detailed(&descriptor, &stripes, view, 0)
                        .map_err(|error| error.to_string())
                })();
                rail.stats
                    .latency_us_sum
                    .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                rail.stats
                    .latency_us_max
                    .fetch_max(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                let _ = reply.send(TaskReply {
                    rail_index: rail.index,
                    endpoint: Arc::clone(&endpoint),
                    expected_bytes,
                    expected_chunks,
                    outcome,
                });
            }
            Command::EvictRegistration { base, ack } => {
                let freed = client.evict_registrations_for(base);
                rail.stats
                    .registered_bytes
                    .fetch_sub(freed as u64, Ordering::Relaxed);
                let _ = ack.send(());
            }
            Command::Stop => break,
        }
    }

    // Teardown bookkeeping: whatever this connection still had registered
    // goes away with the dropped client — keep the rail counter honest.
    rail.stats
        .registered_bytes
        .fetch_sub(client.registered_cache_bytes() as u64, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// MultiRailClient
// ---------------------------------------------------------------------------

/// A multi-rail RDMA reader for one client Worker.
///
/// Holds one connection per (rail, endpoint) pair used so far. Connections
/// persist across reads; a failed connection is quiesced (worker joined, QP
/// destroyed) before the failing read returns, and the rail enters a cooldown
/// window that later reads plan around.
pub struct MultiRailClient {
    rails: Vec<Arc<Rail>>,
    limits: RailLimits,
    policy: RailSelectPolicy,
    conns: Mutex<HashMap<(usize, Arc<str>), ConnEntry>>,
}

impl MultiRailClient {
    /// Create a client over the given rails. Fails when no rail is given.
    pub fn new(rails: Vec<RailConfig>) -> Result<Self, MultiRailError> {
        if rails.is_empty() {
            return Err(MultiRailError::NoRails);
        }
        let rails = rails
            .into_iter()
            .enumerate()
            .map(|(index, config)| {
                let topology = read_topology(&config.device);
                let gid_v4 = rdma::query_gid_raw(&config.device, config.port, config.gid_index)
                    .and_then(gid_to_v4);
                Arc::new(Rail {
                    index,
                    topology,
                    gid_v4,
                    config,
                    stats: RailStats::default(),
                    healthy: AtomicBool::new(true),
                    unhealthy_until: Mutex::new(None),
                })
            })
            .collect();
        Ok(Self {
            rails,
            limits: RailLimits::default(),
            policy: RailSelectPolicy::default(),
            conns: Mutex::new(HashMap::new()),
        })
    }

    /// Override resource/backpressure limits.
    pub fn with_limits(mut self, limits: RailLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Override the stripe→rail selection policy.
    pub fn with_policy(mut self, policy: RailSelectPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn rail_count(&self) -> usize {
        self.rails.len()
    }

    pub fn limits(&self) -> &RailLimits {
        &self.limits
    }

    /// Snapshot of per-rail state for observability (health, throughput,
    /// errors, in-flight counters, topology).
    pub fn rails_snapshot(&self) -> Vec<RailSnapshot> {
        let mut live_conns: HashMap<usize, u64> = HashMap::new();
        for key in self.conns.lock().unwrap().keys() {
            *live_conns.entry(key.0).or_insert(0) += 1;
        }
        self.rails
            .iter()
            .map(|rail| {
                let s = &rail.stats;
                RailSnapshot {
                    index: rail.index,
                    device: rail.config.device.clone(),
                    topology: rail.topology.clone(),
                    healthy: rail.healthy.load(Ordering::Acquire),
                    cooldown_ms_remaining: rail.cooldown_remaining().as_millis() as u64,
                    requests_ok: s.requests_ok.load(Ordering::Relaxed),
                    requests_err: s.requests_err.load(Ordering::Relaxed),
                    bytes_read: s.bytes_read.load(Ordering::Relaxed),
                    timeouts: s.timeouts.load(Ordering::Relaxed),
                    connections_created: s.connections_created.load(Ordering::Relaxed),
                    connections_quiesced: s.connections_quiesced.load(Ordering::Relaxed),
                    inflight_requests: s.inflight_requests.load(Ordering::Relaxed),
                    inflight_bytes: s.inflight_bytes.load(Ordering::Relaxed),
                    latency_avg_us: s
                        .latency_us_sum
                        .load(Ordering::Relaxed)
                        .checked_div(s.requests_ok.load(Ordering::Relaxed))
                        .unwrap_or(0),
                    latency_max_us: s.latency_us_max.load(Ordering::Relaxed),
                    registered_bytes: s.registered_bytes.load(Ordering::Relaxed),
                }
            })
            .collect()
    }

    /// Read an object described by `descriptor` + `placement chunks` into
    /// `buffer`, distributing stripes across all healthy rails.
    ///
    /// Returns the number of bytes placed in `buffer` (== descriptor.size on
    /// success). On error the whole read fails safely: every connection that
    /// had work in flight is quiesced before this function returns, so no
    /// late RDMA WRITE can touch `buffer` after the call.
    pub fn read_object_into(
        &self,
        descriptor: &pb::ObjectDescriptor,
        chunks: &[pb::PlacementChunk],
        buffer: &mut [u8],
    ) -> Result<usize, MultiRailError> {
        let base = buffer.as_mut_ptr() as usize;
        self.read_impl(descriptor, chunks, base, buffer.len(), false, None)
    }

    /// [`Self::read_object_into`] with cooperative cancellation.
    pub fn read_object_into_cancelled(
        &self,
        descriptor: &pb::ObjectDescriptor,
        chunks: &[pb::PlacementChunk],
        buffer: &mut [u8],
        cancel: &CancelToken,
    ) -> Result<usize, MultiRailError> {
        let base = buffer.as_mut_ptr() as usize;
        self.read_impl(descriptor, chunks, base, buffer.len(), false, Some(cancel))
    }

    /// Convenience wrapper taking a gRPC [`crate::ObjectLookup`] result.
    pub fn read_lookup_into(
        &self,
        lookup: &crate::ObjectLookup,
        buffer: &mut [u8],
    ) -> Result<usize, MultiRailError> {
        let placement = lookup.placement.as_ref().ok_or_else(|| {
            MultiRailError::InvalidPlacement("lookup returned no placement".into())
        })?;
        self.read_object_into(&lookup.descriptor, &placement.chunks, buffer)
    }

    /// `read_object_into` for FFI-owned / pinned memory.
    ///
    /// # Safety
    /// `ptr..ptr+len` must be valid writable memory that stays alive, unmoved
    /// and unmodified for the duration of this call. With
    /// `sticky_registration = true` the per-rail registrations of this buffer
    /// are kept cached in the connections (skipping ~1.5 ms `ibv_reg_mr` per
    /// rail on subsequent reads); the caller then guarantees the buffer is a
    /// long-lived pool region that is never freed nor reused for non-RDMA
    /// purposes while this client lives.
    pub unsafe fn read_object_into_raw(
        &self,
        descriptor: &pb::ObjectDescriptor,
        chunks: &[pb::PlacementChunk],
        ptr: *mut u8,
        len: usize,
        sticky_registration: bool,
    ) -> Result<usize, MultiRailError> {
        if ptr.is_null() || len == 0 {
            return Err(MultiRailError::BufferTooSmall {
                need: descriptor.size,
                have: len,
            });
        }
        self.read_impl(
            descriptor,
            chunks,
            ptr as usize,
            len,
            sticky_registration,
            None,
        )
    }

    fn read_impl(
        &self,
        descriptor: &pb::ObjectDescriptor,
        chunks: &[pb::PlacementChunk],
        base: usize,
        len: usize,
        sticky: bool,
        cancel: Option<&CancelToken>,
    ) -> Result<usize, MultiRailError> {
        let placement = validate_placement(descriptor, chunks)?;
        if placement.object_size > len as u64 {
            return Err(MultiRailError::BufferTooSmall {
                need: placement.object_size,
                have: len,
            });
        }

        // Plan around healthy rails only.
        let rails: Vec<Arc<Rail>> = self
            .rails
            .iter()
            .filter(|rail| rail.selectable())
            .cloned()
            .collect();
        if rails.is_empty() {
            return Err(MultiRailError::NoHealthyRails);
        }
        let rail_of: HashMap<usize, Arc<Rail>> = rails
            .iter()
            .map(|rail| (rail.index, Arc::clone(rail)))
            .collect();

        let plan = build_plan(&placement, &rails, self.policy, &self.limits)?;
        let waves = plan_waves(plan.tasks, &self.limits);

        // Global deadline: every wave gets a full io_timeout, plus one extra
        // for connect phases.
        let deadline = Instant::now()
            + self.limits.io_timeout.mul_f64(waves.len().max(1) as f64)
            + self.limits.io_timeout;

        let (reply_tx, reply_rx) = mpsc::channel::<TaskReply>();
        let descriptor = Arc::new(descriptor.clone());
        let mut participated: HashSet<(usize, Arc<str>)> = HashSet::new();
        let mut failure: Option<MultiRailError> = None;
        let mut failed_rail: Option<usize> = None;
        let mut verified_bytes = 0u64;

        let mut dispatched_total = 0usize;
        let mut collected_total = 0usize;
        'waves: for wave in waves {
            if let Some(cancel) = cancel {
                cancel.check()?;
            }
            // ---- dispatch this wave ----
            let mut dispatched = 0usize;
            for task in &wave {
                let Some(rail) = rail_of.get(&task.rail_index) else {
                    continue;
                };
                // Backpressure: wait for per-rail and total byte headroom.
                if !self.await_headroom(rail, task.bytes, deadline, cancel) {
                    failure = Some(MultiRailError::Timeout {
                        rail: rail.config.device.clone(),
                        endpoint: task.endpoint.to_string(),
                        after_ms: self.limits.io_timeout.as_millis(),
                    });
                    failed_rail = Some(rail.index);
                    break 'waves;
                }
                match self.get_or_create_conn(rail, Arc::clone(&task.endpoint), deadline) {
                    Ok(tx) => {
                        participated.insert((rail.index, Arc::clone(&task.endpoint)));
                        rail.stats.inflight_requests.fetch_add(1, Ordering::Relaxed);
                        rail.stats
                            .inflight_bytes
                            .fetch_add(task.bytes, Ordering::Relaxed);
                        let sent = tx.send(Command::Read {
                            descriptor: Arc::clone(&descriptor),
                            stripes: Arc::new(task.stripes.clone()),
                            dst_base: base,
                            dst_len: len,
                            reply: reply_tx.clone(),
                        });
                        if sent.is_err() {
                            rail.stats.inflight_requests.fetch_sub(1, Ordering::Relaxed);
                            rail.stats
                                .inflight_bytes
                                .fetch_sub(task.bytes, Ordering::Relaxed);
                            failure = Some(MultiRailError::TaskFailed {
                                rail: rail.config.device.clone(),
                                endpoint: task.endpoint.to_string(),
                                source: "worker exited unexpectedly".into(),
                            });
                            failed_rail = Some(rail.index);
                            break 'waves;
                        }
                        dispatched += 1;
                        dispatched_total += 1;
                    }
                    Err(error) => {
                        failed_rail = Some(rail.index);
                        failure = Some(error);
                        break 'waves;
                    }
                }
            }

            // ---- collect this wave's replies (one per dispatched task) ----
            let mut replies: Vec<TaskReply> = Vec::with_capacity(dispatched);
            while replies.len() < dispatched {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    failure = Some(MultiRailError::Timeout {
                        rail: "any".into(),
                        endpoint: "any".into(),
                        after_ms: self.limits.io_timeout.as_millis(),
                    });
                    break;
                }
                if let Some(cancel) = cancel {
                    if let Err(error) = cancel.check() {
                        // Drain with a short grace period, then quiesce via
                        // the failure path below (join still guarantees that
                        // no work request outlives this call).
                        failure = Some(error);
                        break;
                    }
                }
                // Bound each wait so cancellation is observed promptly.
                let slice = remaining.min(Duration::from_millis(100));
                match reply_rx.recv_timeout(slice) {
                    Ok(reply) => {
                        replies.push(reply);
                        collected_total += 1;
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if deadline <= Instant::now() {
                            failure = Some(MultiRailError::Timeout {
                                rail: "any".into(),
                                endpoint: "any".into(),
                                after_ms: self.limits.io_timeout.as_millis(),
                            });
                            break;
                        }
                        continue;
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        failure = Some(MultiRailError::WorkerPanic {
                            rail: "any".into(),
                            endpoint: "any".into(),
                        });
                        break;
                    }
                }
            }

            // ---- verify replies, release in-flight accounting ----
            for reply in replies {
                let rail = &self.rails[reply.rail_index];
                rail.stats.inflight_requests.fetch_sub(1, Ordering::Relaxed);
                rail.stats
                    .inflight_bytes
                    .fetch_sub(reply.expected_bytes, Ordering::Relaxed);
                match &reply.outcome {
                    Ok(Some(outcome)) => {
                        if outcome.bytes as u64 != reply.expected_bytes {
                            rail.stats.requests_err.fetch_add(1, Ordering::Relaxed);
                            failure = Some(MultiRailError::ByteCountMismatch {
                                rail: rail.config.device.clone(),
                                endpoint: reply.endpoint.to_string(),
                                expected: reply.expected_bytes,
                                actual: outcome.bytes as u64,
                            });
                            failed_rail = Some(rail.index);
                        } else if outcome.num_chunks != reply.expected_chunks {
                            rail.stats.requests_err.fetch_add(1, Ordering::Relaxed);
                            failure = Some(MultiRailError::ChunkCountMismatch {
                                rail: rail.config.device.clone(),
                                endpoint: reply.endpoint.to_string(),
                                expected: reply.expected_chunks,
                                actual: outcome.num_chunks,
                            });
                            failed_rail = Some(rail.index);
                        } else {
                            rail.stats.requests_ok.fetch_add(1, Ordering::Relaxed);
                            rail.stats
                                .bytes_read
                                .fetch_add(outcome.bytes as u64, Ordering::Relaxed);
                            verified_bytes += outcome.bytes as u64;
                        }
                    }
                    Ok(None) => {
                        // The server answers found=false both for absent
                        // objects and for descriptor mismatches; with a valid
                        // placement in hand this means a stale descriptor.
                        rail.stats.requests_err.fetch_add(1, Ordering::Relaxed);
                        failure = Some(MultiRailError::StaleDescriptor {
                            endpoint: reply.endpoint.to_string(),
                        });
                        failed_rail = Some(rail.index);
                    }
                    Err(message) => {
                        rail.stats.requests_err.fetch_add(1, Ordering::Relaxed);
                        if reply.is_timeout() {
                            rail.stats.timeouts.fetch_add(1, Ordering::Relaxed);
                            failure = Some(MultiRailError::Timeout {
                                rail: rail.config.device.clone(),
                                endpoint: reply.endpoint.to_string(),
                                after_ms: self.limits.io_timeout.as_millis(),
                            });
                        } else {
                            failure = Some(MultiRailError::TaskFailed {
                                rail: rail.config.device.clone(),
                                endpoint: reply.endpoint.to_string(),
                                source: message.clone(),
                            });
                        }
                        failed_rail = Some(rail.index);
                    }
                }
            }

            if failure.is_some() {
                break 'waves;
            }
        }

        // ---- failure path: drain outstanding replies, then quiesce ----
        // Workers send exactly one reply per Read, and the quiesce Stop is
        // queued behind any in-flight command — so every outstanding reply
        // arrives once the worker finishes. Drain them (bounded by the read
        // deadline) to release the in-flight counters; otherwise aborted
        // reads would leak inflight_requests/inflight_bytes and eventually
        // starve await_headroom.
        while collected_total < dispatched_total {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match reply_rx.recv_timeout(remaining.min(Duration::from_millis(500))) {
                Ok(reply) => {
                    collected_total += 1;
                    let rail = &self.rails[reply.rail_index];
                    rail.stats.inflight_requests.fetch_sub(1, Ordering::Relaxed);
                    rail.stats
                        .inflight_bytes
                        .fetch_sub(reply.expected_bytes, Ordering::Relaxed);
                }
                Err(_) => break,
            }
        }

        if let Some(error) = failure {
            if let Some(index) = failed_rail {
                self.rails[index].mark_unhealthy(self.limits.rail_cooldown);
            }
            self.quiesce(participated.into_iter().collect());
            return Err(error);
        }

        // Full-coverage invariant: the per-task byte checks must add up to
        // the whole object, otherwise stripes went missing.
        if verified_bytes != placement.expected_bytes {
            if !sticky {
                self.evict_participated_registrations(&participated, base);
            }
            return Err(MultiRailError::ByteCountMismatch {
                rail: "aggregate".into(),
                endpoint: "aggregate".into(),
                expected: placement.expected_bytes,
                actual: verified_bytes,
            });
        }

        // ---- per-stripe checksum verification (when the server set them) ----
        for (stripe, info) in &placement.stripes {
            if let Some(expected) = &info.checksum {
                let start = info.offset as usize;
                let end = start + info.length as usize;
                if end > len {
                    return Err(MultiRailError::OutOfBoundsStripe {
                        stripe: *stripe,
                        stripe_count: placement.stripe_count,
                    });
                }
                // SAFETY: `base..base+len` is the caller buffer, valid for
                // this whole call; the slice below stays inside it.
                let view: &[u8] = unsafe {
                    std::slice::from_raw_parts((base + start) as *const u8, info.length as usize)
                };
                let actual = format!("{:016x}", twox_hash::xxh3::hash64(view));
                if !expected.eq_ignore_ascii_case(&actual) {
                    if !sticky {
                        self.evict_participated_registrations(&participated, base);
                    }
                    return Err(MultiRailError::ChecksumMismatch {
                        stripe: *stripe,
                        expected: expected.clone(),
                        actual,
                    });
                }
            }
        }

        // ---- success path: synchronous MR eviction for non-sticky buffers ----
        if !sticky {
            self.evict_participated_registrations(&participated, base);
        }

        Ok(placement.expected_bytes as usize)
    }

    /// Wait until `bytes` more in-flight traffic fits the per-rail and total
    /// budgets, or the deadline passes.
    fn await_headroom(
        &self,
        rail: &Rail,
        bytes: u64,
        deadline: Instant,
        cancel: Option<&CancelToken>,
    ) -> bool {
        loop {
            if let Some(cancel) = cancel {
                if cancel.is_cancelled() {
                    return false;
                }
            }
            let rail_inflight = rail.stats.inflight_bytes.load(Ordering::Relaxed);
            if rail_inflight + bytes <= self.limits.max_inflight_bytes_per_rail {
                let total: u64 = self
                    .rails
                    .iter()
                    .map(|r| r.stats.inflight_bytes.load(Ordering::Relaxed))
                    .sum();
                if total + bytes <= self.limits.max_inflight_bytes_total {
                    return true;
                }
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Get an existing connection's command channel or spawn a worker for
    /// (rail, endpoint), waiting for setup completion (bounded by `deadline`).
    fn get_or_create_conn(
        &self,
        rail: &Arc<Rail>,
        endpoint: Arc<str>,
        deadline: Instant,
    ) -> Result<mpsc::Sender<Command>, MultiRailError> {
        if let Some(entry) = self
            .conns
            .lock()
            .unwrap()
            .get(&(rail.index, Arc::clone(&endpoint)))
        {
            return Ok(entry.tx.clone());
        }

        let (setup_tx, setup_rx) = mpsc::channel();
        let (command_tx, command_rx) = mpsc::channel();
        let worker_rail = Arc::clone(rail);
        let worker_endpoint = Arc::clone(&endpoint);
        let limits = self.limits.clone();
        let handle = std::thread::Builder::new()
            .name(format!("mrail-{}-{}", rail.config.device, endpoint))
            .spawn(move || conn_worker(worker_rail, worker_endpoint, limits, command_rx, setup_tx))
            .map_err(|error| MultiRailError::TaskFailed {
                rail: rail.config.device.clone(),
                endpoint: endpoint.to_string(),
                source: format!("spawn worker: {error}"),
            })?;

        let remaining = deadline.saturating_duration_since(Instant::now());
        match setup_rx.recv_timeout(remaining) {
            Ok(Ok(())) => {}
            Ok(Err(reason)) => {
                rail.mark_unhealthy(self.limits.rail_cooldown);
                let _ = handle.join();
                return Err(MultiRailError::TaskFailed {
                    rail: rail.config.device.clone(),
                    endpoint: endpoint.to_string(),
                    source: format!("connect failed: {reason}"),
                });
            }
            Err(_) => {
                rail.mark_unhealthy(self.limits.rail_cooldown);
                let _ = command_tx.send(Command::Stop);
                let _ = handle.join();
                return Err(MultiRailError::Timeout {
                    rail: rail.config.device.clone(),
                    endpoint: endpoint.to_string(),
                    after_ms: self.limits.io_timeout.as_millis(),
                });
            }
        }

        let mut conns = self.conns.lock().unwrap();
        // Another read may have created the same pair concurrently.
        if let Some(existing) = conns.get(&(rail.index, Arc::clone(&endpoint))) {
            let _ = command_tx.send(Command::Stop);
            let _ = handle.join();
            return Ok(existing.tx.clone());
        }
        conns.insert(
            (rail.index, Arc::clone(&endpoint)),
            ConnEntry {
                rail_index: rail.index,
                tx: command_tx.clone(),
                handle,
            },
        );
        Ok(command_tx)
    }

    /// Stop and join the given connections, removing them from the map. Join
    /// completion implies: worker loop exited → `RdmaClient` dropped → BYE
    /// sent, QP destroyed, MRs deregistered. This is the quiesce barrier that
    /// makes returning a failed read safe.
    /// Evict the caller-buffer registrations from all participating
    /// connections (non-sticky reads). Runs on the success path AND on
    /// post-dispatch error returns — a cached MR must never outlive the
    /// caller's buffer and pin its freed pages.
    fn evict_participated_registrations(
        &self,
        participated: &std::collections::HashSet<(usize, Arc<str>)>,
        base: usize,
    ) {
        for key in participated {
            let ack_rx = {
                let conns = self.conns.lock().unwrap();
                match conns.get(key) {
                    Some(entry) => {
                        let (ack_tx, ack_rx) = mpsc::channel();
                        if entry
                            .tx
                            .send(Command::EvictRegistration { base, ack: ack_tx })
                            .is_ok()
                        {
                            Some(ack_rx)
                        } else {
                            None
                        }
                    }
                    None => None,
                }
            };
            if let Some(ack_rx) = ack_rx {
                let _ = ack_rx.recv_timeout(self.limits.io_timeout);
            }
        }
    }

    fn quiesce(&self, keys: Vec<(usize, Arc<str>)>) {
        let entries: Vec<ConnEntry> = {
            let mut conns = self.conns.lock().unwrap();
            keys.into_iter()
                .filter_map(|key| conns.remove(&key))
                .collect()
        };
        for entry in &entries {
            let _ = entry.tx.send(Command::Stop);
        }
        for entry in entries {
            if entry.handle.join().is_ok() {
                if let Some(rail) = self.rails.get(entry.rail_index) {
                    rail.stats
                        .connections_quiesced
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

impl Drop for MultiRailClient {
    fn drop(&mut self) {
        let keys: Vec<(usize, Arc<str>)> = {
            let conns = self.conns.lock().unwrap();
            conns.keys().cloned().collect()
        };
        self.quiesce(keys);
    }
}

/// Verify the xxh3-64 checksum of one received stripe against the placement
/// checksum (lowercase hex, matching the server's `twox-hash` encoding).
pub fn verify_stripe_checksum(
    buffer: &[u8],
    stripe: u32,
    chunk_size: u64,
    expected_hex: &str,
) -> Result<(), MultiRailError> {
    let offset = stripe as u64 * chunk_size;
    let end = (offset + chunk_size).min(buffer.len() as u64);
    if offset >= buffer.len() as u64 || end <= offset {
        return Err(MultiRailError::OutOfBoundsStripe {
            stripe,
            stripe_count: stripe.saturating_add(1),
        });
    }
    let actual = twox_hash::xxh3::hash64(&buffer[offset as usize..end as usize]);
    let actual_hex = format!("{actual:016x}");
    if !expected_hex.eq_ignore_ascii_case(&actual_hex) {
        return Err(MultiRailError::ChecksumMismatch {
            stripe,
            expected: expected_hex.to_string(),
            actual: actual_hex,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests (hardware-independent)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(size: u64, stripe_count: u32, chunk_size: u64) -> pb::ObjectDescriptor {
        pb::ObjectDescriptor {
            key: Some(pb::ObjectKey {
                namespace: "ns".into(),
                object_key: "k".into(),
            }),
            object_handle: "h".into(),
            object_generation: 1,
            content_etag: "e".into(),
            layout_version: 1,
            size,
            is_striped: stripe_count > 1,
            stripe_count,
            chunk_size,
        }
    }

    fn chunk(stripe: u32, endpoint: &str, length: u64, checksum: &str) -> pb::PlacementChunk {
        pb::PlacementChunk {
            stripe_index: stripe,
            node_id: "node".into(),
            grpc_endpoint: endpoint.into(),
            rdma_endpoint: format!("{endpoint}:50053"),
            device_id: 0,
            storage_handle: "s".into(),
            offset: stripe as u64 * 8,
            length,
            checksum: checksum.into(),
        }
    }

    fn test_rails(count: usize) -> Vec<Arc<Rail>> {
        (0..count)
            .map(|index| {
                Arc::new(Rail {
                    index,
                    config: RailConfig::new(format!("dev{index}")),
                    topology: RailTopology::default(),
                    stats: RailStats::default(),
                    healthy: AtomicBool::new(true),
                    unhealthy_until: Mutex::new(None),
                    gid_v4: None,
                })
            })
            .collect()
    }

    #[test]
    fn rail_config_parses_optional_fields() {
        assert_eq!(RailConfig::parse("mlx5_0").unwrap().port, 1);
        let full = RailConfig::parse("irdma0:1:5:2:4096").unwrap();
        assert_eq!(full.device, "irdma0");
        assert_eq!(full.port, 1);
        assert_eq!(full.gid_index, 5);
        assert_eq!(full.weight, 2);
        assert_eq!(full.mtu, 4096);
        assert!(RailConfig::parse("  ").is_none());
    }

    #[test]
    fn placement_rejects_missing_and_duplicate_stripes() {
        let desc = descriptor(24, 3, 8);
        let chunks = vec![chunk(0, "10.0.0.1", 8, ""), chunk(2, "10.0.0.1", 8, "")];
        match validate_placement(&desc, &chunks) {
            Err(MultiRailError::MissingStripes(stripes)) => assert_eq!(stripes, vec![1]),
            other => panic!("expected missing stripes, got {other:?}"),
        }
        let dup = vec![
            chunk(0, "10.0.0.1", 8, ""),
            chunk(0, "10.0.0.2", 8, ""),
            chunk(1, "10.0.0.1", 8, ""),
            chunk(2, "10.0.0.1", 8, ""),
        ];
        match validate_placement(&desc, &dup) {
            Err(MultiRailError::DuplicateStripes(stripes)) => assert_eq!(stripes, vec![0]),
            other => panic!("expected duplicate stripes, got {other:?}"),
        }
    }

    #[test]
    fn placement_rejects_out_of_bounds_and_bad_lengths() {
        let desc = descriptor(24, 3, 8);
        let oob = vec![
            chunk(0, "10.0.0.1", 8, ""),
            chunk(1, "10.0.0.1", 8, ""),
            chunk(2, "10.0.0.1", 8, ""),
            chunk(3, "10.0.0.1", 8, ""),
        ];
        assert!(matches!(
            validate_placement(&desc, &oob),
            Err(MultiRailError::OutOfBoundsStripe { stripe: 3, .. })
        ));
        let bad_len = vec![
            chunk(0, "10.0.0.1", 8, ""),
            chunk(1, "10.0.0.1", 7, ""),
            chunk(2, "10.0.0.1", 8, ""),
        ];
        assert!(matches!(
            validate_placement(&desc, &bad_len),
            Err(MultiRailError::InvalidPlacement(_))
        ));
        // Last stripe is short: a 20-byte object with 8-byte chunks ends in 4.
        let desc2 = descriptor(20, 3, 8);
        let ok = vec![
            chunk(0, "10.0.0.1", 8, ""),
            chunk(1, "10.0.0.1", 8, ""),
            chunk(2, "10.0.0.1", 4, ""),
        ];
        let validated = validate_placement(&desc2, &ok).expect("short last stripe accepted");
        assert_eq!(validated.expected_bytes, 20);
    }

    #[test]
    fn placement_accepts_non_striped_single_chunk() {
        let desc = descriptor(100, 0, 0);
        let chunks = vec![chunk(0, "10.0.0.1", 0, "")];
        let validated = validate_placement(&desc, &chunks).expect("valid");
        assert_eq!(validated.expected_bytes, 100);
        assert_eq!(validated.stripe_count, 1);
    }

    #[test]
    fn plan_balances_stripes_over_rails_by_bytes() {
        let desc = descriptor(64, 8, 8);
        let chunks: Vec<_> = (0..8).map(|i| chunk(i, "10.0.0.1", 8, "")).collect();
        let placement = validate_placement(&desc, &chunks).unwrap();
        let rails = test_rails(2);
        let plan = build_plan(
            &placement,
            &rails,
            RailSelectPolicy::LeastLoaded,
            &RailLimits::default(),
        )
        .unwrap();
        assert_eq!(plan.task_count(), 2);
        let total: u64 = plan.tasks.iter().map(|t| t.bytes).sum();
        assert_eq!(total, 64);
        // 8 equal stripes over 2 rails → 4/4 per rail.
        for task in &plan.tasks {
            assert_eq!(task.bytes, 32);
            assert_eq!(task.stripes.len(), 4);
        }
        // Disjoint stripe sets covering 0..8.
        let mut seen: HashSet<u32> = HashSet::new();
        for task in &plan.tasks {
            for stripe in &task.stripes {
                assert!(seen.insert(*stripe), "stripe {stripe} assigned twice");
            }
        }
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn plan_respects_relative_weights() {
        let desc = descriptor(80, 10, 8);
        let chunks: Vec<_> = (0..10).map(|i| chunk(i, "10.0.0.1", 8, "")).collect();
        let placement = validate_placement(&desc, &chunks).unwrap();
        let mut rails = test_rails(2);
        rails[1] = Arc::new(Rail {
            index: 1,
            config: RailConfig::new("dev1").with_weight(4),
            topology: RailTopology::default(),
            stats: RailStats::default(),
            healthy: AtomicBool::new(true),
            unhealthy_until: Mutex::new(None),
            gid_v4: None,
        });
        let plan = build_plan(
            &placement,
            &rails,
            RailSelectPolicy::LeastLoaded,
            &RailLimits::default(),
        )
        .unwrap();
        let rail0: u64 = plan
            .tasks
            .iter()
            .filter(|t| t.rail_index == 0)
            .map(|t| t.bytes)
            .sum();
        let rail1: u64 = plan
            .tasks
            .iter()
            .filter(|t| t.rail_index == 1)
            .map(|t| t.bytes)
            .sum();
        // 1:4 weights over 10 stripes of 8 bytes: 16 vs 64.
        assert_eq!((rail0, rail1), (16, 64));
    }

    #[test]
    fn rail_endpoint_whitelist_restricts_and_errors() {
        let desc = descriptor(16, 2, 8);
        let chunks = vec![chunk(0, "10.0.0.1", 8, ""), chunk(1, "10.0.0.2", 8, "")];
        let placement = validate_placement(&desc, &chunks).unwrap();
        let mut rails = test_rails(2);
        rails[0] = Arc::new(Rail {
            index: 0,
            config: RailConfig::new("dev0").with_endpoints(vec!["10.0.0.2".into()]),
            topology: RailTopology::default(),
            stats: RailStats::default(),
            healthy: AtomicBool::new(true),
            unhealthy_until: Mutex::new(None),
            gid_v4: None,
        });
        let plan = build_plan(
            &placement,
            &rails,
            RailSelectPolicy::LeastLoaded,
            &RailLimits::default(),
        )
        .expect("plan builds");
        // dev0 pinned to 10.0.0.2; 10.0.0.1 falls to dev1.
        assert_eq!(plan.tasks.len(), 2);
        for task in &plan.tasks {
            let expected_rail = if task.endpoint.contains("10.0.0.2") {
                0
            } else {
                1
            };
            assert_eq!(task.rail_index, expected_rail);
        }
        // No rail allowed for 10.0.0.1 → typed planning error.
        let mut strict = test_rails(1);
        strict[0] = Arc::new(Rail {
            index: 0,
            config: RailConfig::new("dev0").with_endpoints(vec!["10.0.0.2".into()]),
            topology: RailTopology::default(),
            stats: RailStats::default(),
            healthy: AtomicBool::new(true),
            unhealthy_until: Mutex::new(None),
            gid_v4: None,
        });
        assert!(build_plan(
            &placement,
            &strict,
            RailSelectPolicy::LeastLoaded,
            &RailLimits::default()
        )
        .is_err());
    }

    #[test]
    fn plan_pins_endpoint_with_affinity_policy() {
        let desc = descriptor(64, 8, 8);
        let chunks: Vec<_> = (0..8).map(|i| chunk(i, "10.0.0.1", 8, "")).collect();
        let placement = validate_placement(&desc, &chunks).unwrap();
        let rails = test_rails(2);
        let plan = build_plan(
            &placement,
            &rails,
            RailSelectPolicy::EndpointAffinity,
            &RailLimits::default(),
        )
        .unwrap();
        assert_eq!(plan.task_count(), 1);
        assert_eq!(plan.tasks[0].bytes, 64);
        assert_eq!(plan.tasks[0].stripes.len(), 8);
    }

    #[test]
    fn affinity_prefers_same_subnet_rail() {
        let desc = descriptor(16, 2, 8);
        let chunks = vec![chunk(0, "10.0.0.9", 8, ""), chunk(1, "10.0.0.9", 8, "")];
        let placement = validate_placement(&desc, &chunks).unwrap();
        let mut rails = test_rails(2);
        rails[1] = Arc::new(Rail {
            index: 1,
            config: RailConfig::new("dev1"),
            topology: RailTopology::default(),
            stats: RailStats::default(),
            healthy: AtomicBool::new(true),
            unhealthy_until: Mutex::new(None),
            gid_v4: Some("10.0.0.5".parse().unwrap()),
        });
        let plan = build_plan(
            &placement,
            &rails,
            RailSelectPolicy::EndpointAffinity,
            &RailLimits::default(),
        )
        .unwrap();
        assert_eq!(plan.tasks.len(), 1);
        assert_eq!(plan.tasks[0].rail_index, 1);
    }

    #[test]
    fn waves_respect_connection_caps() {
        let tasks: Vec<TaskSpec> = (0..6)
            .map(|i| TaskSpec {
                rail_index: i % 2,
                endpoint: Arc::from(format!("10.0.0.{i}:50053")),
                stripes: vec![i as u32],
                bytes: 8,
            })
            .collect();
        let limits = RailLimits {
            max_connections_per_rail: 2,
            max_connections_total: 4,
            ..RailLimits::default()
        };
        let waves = plan_waves(tasks, &limits);
        assert!(waves.len() >= 2);
        for wave in &waves {
            assert!(wave.len() <= limits.max_connections_total);
            let mut per_rail: HashMap<usize, usize> = HashMap::new();
            for task in wave {
                *per_rail.entry(task.rail_index).or_insert(0) += 1;
            }
            for count in per_rail.values() {
                assert!(*count <= limits.max_connections_per_rail);
            }
        }
    }

    #[test]
    fn gid_ipv4_mapping() {
        let mut raw = [0u8; 16];
        raw[10] = 0xff;
        raw[11] = 0xff;
        raw[12..16].copy_from_slice(&[10, 0, 0, 7]);
        assert_eq!(
            gid_to_v4(raw).map(|ip| ip.to_string()),
            Some("10.0.0.7".into())
        );
        assert!(gid_to_v4([0u8; 16]).is_none());
    }

    #[test]
    fn endpoint_host_extraction() {
        assert_eq!(
            endpoint_v4("10.1.2.3:50053").map(|ip| ip.to_string()),
            Some("10.1.2.3".into())
        );
        assert_eq!(endpoint_v4("not-a-host:50053"), None);
    }

    #[test]
    fn cancel_token_is_shared_and_maps_to_cancelled() {
        let token = CancelToken::new();
        let observer = token.clone();
        assert!(!token.is_cancelled());
        // 取消在克隆之间共享同一标志; wave 边界/背压等待/应答收集点
        // 经 check() 协作退出, 映射为类型化 Cancelled 错误。
        observer.cancel();
        assert!(token.is_cancelled());
        assert!(matches!(token.check(), Err(MultiRailError::Cancelled)));
    }

    #[test]
    fn checksum_verification_matches_server_encoding() {
        let mut buffer = vec![0u8; 16];
        buffer[..8].copy_from_slice(b"stripe0!");
        buffer[8..].copy_from_slice(b"stripe1!");
        let expected0 = format!("{:016x}", twox_hash::xxh3::hash64(&buffer[..8]));
        assert!(verify_stripe_checksum(&buffer, 0, 8, &expected0).is_ok());
        let wrong = format!("{:016x}", twox_hash::xxh3::hash64(&buffer[8..]));
        assert!(matches!(
            verify_stripe_checksum(&buffer, 0, 8, &wrong),
            Err(MultiRailError::ChecksumMismatch { .. })
        ));
        // Out-of-bounds stripe.
        assert!(matches!(
            verify_stripe_checksum(&buffer, 2, 8, &expected0),
            Err(MultiRailError::OutOfBoundsStripe { .. })
        ));
    }

    #[test]
    fn topology_parses_sysfs_shape() {
        let dir = tempfile::tempdir().unwrap();
        let dev = dir.path().join("rxe0/device");
        std::fs::create_dir_all(&dev).unwrap();
        std::fs::write(dev.join("numa_node"), "1\n").unwrap();
        let topology = topology_from(dir.path(), "rxe0");
        assert_eq!(topology.numa_node, 1);
        assert!(!topology.pci_slot.is_empty());
    }

    #[test]
    fn cooldown_expires_and_rail_recovers() {
        let rails = test_rails(1);
        rails[0].mark_unhealthy(Duration::from_millis(20));
        assert!(!rails[0].selectable());
        std::thread::sleep(Duration::from_millis(30));
        assert!(rails[0].selectable());
    }

    #[test]
    fn policy_parsing_round_trip() {
        assert_eq!(
            RailSelectPolicy::parse("endpoint-affinity"),
            Some(RailSelectPolicy::EndpointAffinity)
        );
        assert_eq!(
            RailSelectPolicy::parse("least-loaded"),
            Some(RailSelectPolicy::LeastLoaded)
        );
        assert_eq!(RailSelectPolicy::parse("bogus"), None);
    }

    #[test]
    fn expected_task_outcome_matches_layout() {
        let desc = descriptor(20, 3, 8);
        let (bytes, chunks) = expected_task_outcome(&desc, &[0, 2]);
        assert_eq!((bytes, chunks), (12, 2));
        let (bytes, chunks) = expected_task_outcome(&desc, &[]);
        assert_eq!((bytes, chunks), (20, 1));
    }
}
