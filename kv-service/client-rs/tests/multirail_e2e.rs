//! Hardware-gated end-to-end coverage for the multi-rail RDMA read path.
//!
//! Run explicitly on a host with an RDMA-enabled ContextStore server that
//! exposes at least two RDMA listeners (e.g. Soft-RoCE):
//! ```text
//! CS_MR_COORDINATOR=http://127.0.0.1:50051 \
//! CS_MR_RAILS=rxe0,rxe1 \
//! cargo test --manifest-path kv-service/client-rs/Cargo.toml --features rdma \
//!   --test multirail_e2e -- --ignored --nocapture
//! ```

#![cfg(feature = "rdma")]

use contextstore_client_rs::multirail::{MultiRailClient, MultiRailError, RailConfig, RailLimits};
use contextstore_client_rs::{KvClient, ObjectLookup};
use prost::bytes::Bytes;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn rails_from_env() -> Vec<RailConfig> {
    let pins = pin_map_from_env();
    env_or("CS_MR_RAILS", "rxe0,rxe1")
        .split(',')
        .filter_map(RailConfig::parse)
        .map(|mut rail| {
            if let Some(endpoints) = pins.get(&rail.device) {
                rail.endpoints = endpoints.clone();
            }
            if let Ok(mtu) = std::env::var("CS_MR_RAIL_MTU") {
                if let Ok(mtu) = mtu.trim().parse::<u16>() {
                    rail.mtu = mtu;
                }
            }
            rail
        })
        .collect()
}

/// Rails with endpoint whitelists cleared (single-rail reference reads and
/// failure injection must be able to reach any endpoint).
fn unpinned(rails: &[RailConfig]) -> Vec<RailConfig> {
    rails
        .iter()
        .cloned()
        .map(|mut rail| {
            rail.endpoints.clear();
            rail
        })
        .collect()
}

/// `CS_MR_PIN="rxe0=host[:port][;host...],rxe1=..."` → device → endpoints.
fn pin_map_from_env() -> std::collections::HashMap<String, Vec<String>> {
    let mut map = std::collections::HashMap::new();
    for spec in env_or("CS_MR_PIN", "").split(',') {
        if let Some((device, targets)) = spec.split_once('=') {
            let endpoints: Vec<String> = targets
                .split(';')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect();
            if !endpoints.is_empty() {
                map.insert(device.trim().to_string(), endpoints);
            }
        }
    }
    map
}

/// Rewrite the placement's per-stripe RDMA endpoints round-robin across
/// `CS_MR_ALTERNATE_ENDPOINTS` (comma-separated `host:port` list). Used on
/// single-host testbeds where one node owns several NIC listeners.
fn remap_lookup_endpoints(lookup: &ObjectLookup) -> ObjectLookup {
    let list = env_or("CS_MR_ALTERNATE_ENDPOINTS", "");
    if list.is_empty() {
        return lookup.clone();
    }
    let endpoints: Vec<String> = list.split(',').map(|s| s.trim().to_string()).collect();
    let mut remapped = lookup.clone();
    if let Some(placement) = remapped.placement.as_mut() {
        for (index, chunk) in placement.chunks.iter_mut().enumerate() {
            chunk.rdma_endpoint = endpoints[index % endpoints.len()].clone();
        }
    }
    remapped
}

fn unique_key(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_nanos();
    format!("{prefix}-{nanos}")
}

fn pattern_word(word_index: u64) -> u64 {
    word_index
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(23)
        ^ word_index
}

fn fill_pattern(buffer: &mut [u8]) {
    for (index, chunk) in buffer.chunks_exact_mut(8).enumerate() {
        chunk.copy_from_slice(&pattern_word(index as u64).to_le_bytes());
    }
}

fn verify_pattern(buffer: &[u8]) -> bool {
    buffer
        .chunks_exact(8)
        .enumerate()
        .all(|(index, chunk)| chunk == pattern_word(index as u64).to_le_bytes())
}

struct Fixture {
    runtime: tokio::runtime::Runtime,
    namespace: String,
    key: String,
    lookup: ObjectLookup,
    size: usize,
}

/// Seed a striped object over gRPC and look it up. The ContextStore server
/// stripes large objects across nodes/NICs per its placement policy; a
/// single-node two-NIC server still produces a striped layout with both
/// endpoints when the object exceeds the stripe threshold.
fn seed(namespace: &str, key: &str, size_mb: usize) -> Fixture {
    let runtime = tokio::runtime::Runtime::new().expect("create tokio runtime");
    let size = size_mb * 1024 * 1024;
    let mut payload = vec![0u8; size];
    fill_pattern(&mut payload);
    let coordinator = env_or("CS_MR_COORDINATOR", "http://127.0.0.1:50051");
    let lookup = runtime.block_on(async {
        let mut client = KvClient::connect(coordinator)
            .await
            .expect("connect gRPC coordinator");
        let big = Bytes::from(payload);
        let chunk = 4 * 1024 * 1024;
        let mut segments = Vec::new();
        for offset in (0..size).step_by(chunk) {
            segments.push(big.slice(offset..(offset + chunk).min(size)));
        }
        client
            .put_stream_chunks(namespace, key, segments)
            .await
            .expect("seed object over gRPC");
        client
            .lookup_object(namespace, key)
            .await
            .expect("lookup seeded object")
            .expect("seeded object present")
    });
    Fixture {
        runtime,
        namespace: namespace.to_string(),
        key: key.to_string(),
        size: usize::try_from(lookup.descriptor.size).expect("size fits usize"),
        lookup,
    }
}

fn limits() -> RailLimits {
    // CS_MR_TASK_MAX_STRIPES > 0 exercises the intra-rail task split (several
    // connections per rail) in addition to the plain per-endpoint layout.
    let task_max_stripes = std::env::var("CS_MR_TASK_MAX_STRIPES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    RailLimits {
        io_timeout: Duration::from_secs(20),
        rail_cooldown: Duration::from_millis(500),
        task_max_stripes,
        ..RailLimits::default()
    }
}

#[test]
#[ignore = "requires an RDMA-enabled ContextStore server with 2+ listeners"]
fn multirail_read_returns_identical_bytes_across_rail_counts() {
    let fixture = seed("multirail-e2e", &unique_key("dual"), 128);
    assert!(
        fixture.lookup.placement.is_some(),
        "lookup returned no placement"
    );
    let rails = rails_from_env();
    assert!(rails.len() >= 2, "CS_MR_RAILS must list at least two rails");
    let lookup = remap_lookup_endpoints(&fixture.lookup);

    // Dual-rail read.
    let dual = MultiRailClient::new(rails.clone())
        .expect("create dual-rail client")
        .with_limits(limits());
    let mut buffer = vec![0xA5u8; fixture.size];
    let bytes = dual
        .read_lookup_into(&lookup, &mut buffer)
        .expect("dual-rail read");
    assert_eq!(bytes, fixture.size);
    assert!(verify_pattern(&buffer), "dual-rail content mismatch");

    // Multi-rail metrics must show traffic on more than one rail.
    let snapshots = dual.rails_snapshot();
    let active = snapshots.iter().filter(|s| s.bytes_read > 0).count();
    assert!(
        active >= 2,
        "expected bytes on >=2 rails, got {active}: {snapshots:?}"
    );

    // Single-rail read of the same object must match byte-for-byte
    // (compatibility: the safe-API buffer is reused across clients). On
    // cross-wired testbeds a lone rail may only reach its own-side listener,
    // so the reference reads the original (un-remapped) placement.
    let single = MultiRailClient::new(unpinned(&rails[..1]))
        .expect("create single-rail client")
        .with_limits(limits());
    let mut single_buffer = vec![0x5Au8; fixture.size];
    let single_bytes = single
        .read_lookup_into(&fixture.lookup, &mut single_buffer)
        .expect("single-rail read");
    assert_eq!(single_bytes, fixture.size);
    assert_eq!(single_buffer, buffer, "rail counts changed content");
}

#[test]
#[ignore = "requires an RDMA-enabled ContextStore server with 2+ listeners"]
fn rail_failure_fails_safely_and_next_read_recovers() {
    let fixture = seed("multirail-e2e", &unique_key("fail"), 128);
    let rails = rails_from_env();
    let dead_endpoint = env_or("CS_MR_DEAD_ENDPOINT", "127.0.0.1:59999");

    // Sabotage the placement: point every stripe at a dead endpoint. The
    // read must fail (safe failure — no transparent retry) and return
    // without leaving in-flight work behind. Rails are unpinned here so the
    // failure surfaces as a transport error, not a planning error.
    let mut dead_chunks = fixture
        .lookup
        .placement
        .as_ref()
        .expect("placement")
        .chunks
        .clone();
    for chunk in &mut dead_chunks {
        chunk.rdma_endpoint = dead_endpoint.clone();
    }
    let client = MultiRailClient::new(unpinned(&rails))
        .expect("create client")
        .with_limits(limits());
    let mut buffer = vec![0xA5u8; fixture.size];
    match client.read_object_into(&fixture.lookup.descriptor, &dead_chunks, &mut buffer) {
        Err(error @ (MultiRailError::TaskFailed { .. } | MultiRailError::Timeout { .. })) => {
            let text = error.to_string();
            assert!(
                text.contains(dead_endpoint.as_str()),
                "error should mention the dead endpoint: {text}"
            );
        }
        other => panic!("expected safe failure, got {other:?}"),
    }

    // Path recovery: the very same buffer, still poisoned, is served fine by
    // a client whose rails point at the live endpoints. A late RDMA WRITE
    // from the failed attempt would corrupt this verification.
    let recovered = MultiRailClient::new(rails)
        .expect("create recovery client")
        .with_limits(limits());
    let bytes = recovered
        .read_lookup_into(&remap_lookup_endpoints(&fixture.lookup), &mut buffer)
        .expect("recovery read after failure");
    assert_eq!(bytes, fixture.size);
    assert!(verify_pattern(&buffer), "recovered content mismatch");
}

#[test]
#[ignore = "requires an RDMA-enabled ContextStore server with 2+ listeners"]
fn stale_descriptor_is_reported_as_relookup() {
    let fixture = seed("multirail-e2e", &unique_key("stale"), 128);
    let coordinator = env_or("CS_MR_COORDINATOR", "http://127.0.0.1:50051");

    // Rewrite the object (new generation) through gRPC after the lookup.
    let mut payload = vec![0u8; fixture.size];
    fill_pattern(&mut payload);
    payload[0] ^= 0xFF; // different content → different etag/generation
    fixture.runtime.block_on(async {
        let mut client = KvClient::connect(coordinator)
            .await
            .expect("connect coordinator");
        client
            .delete(&fixture.namespace, &fixture.key)
            .await
            .expect("delete old version");
        client
            .put(&fixture.namespace, &fixture.key, payload)
            .await
            .expect("rewrite object");
    });

    let client = MultiRailClient::new(rails_from_env())
        .expect("create client")
        .with_limits(limits());
    let mut buffer = vec![0u8; fixture.size];
    match client.read_lookup_into(&remap_lookup_endpoints(&fixture.lookup), &mut buffer) {
        Err(MultiRailError::StaleDescriptor { .. }) => {}
        other => panic!("expected StaleDescriptor, got {other:?}"),
    }
}
