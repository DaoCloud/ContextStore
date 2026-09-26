//! Hardware-free tests for the multi-rail read path (run with default features).

use contextstore_client_rs::mock_rail::{MockFault, MockRailClient, MockStore};
use contextstore_client_rs::multi_rail::{
    plan_stripes, stripe_checksum, MultiRailReader, RailReader, ReadOptions,
};
use contextstore_client_rs::pb;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn make_descriptor(stripe_count: u32, chunk: u64, size: u64) -> pb::ObjectDescriptor {
    pb::ObjectDescriptor {
        key: Some(pb::ObjectKey {
            namespace: "ns".into(),
            object_key: "obj".into(),
        }),
        object_handle: "h".into(),
        object_generation: 1,
        content_etag: "e".into(),
        layout_version: 1,
        size,
        is_striped: true,
        stripe_count,
        chunk_size: chunk,
    }
}

#[test]
fn multi_rail_read_aggregates_stripes() {
    let store = Arc::new(Mutex::new(MockStore::default()));
    let stripe_count = 4u32;
    let chunk = 1024u64;
    let mut data = vec![0u8; (stripe_count as usize) * chunk as usize];
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    store
        .lock()
        .unwrap()
        .objects
        .insert("2:nsobj".to_string(), Arc::new(data.clone()));

    let rails: Vec<Box<dyn RailReader>> = vec![
        Box::new(MockRailClient::new("rail0", 1e9, store.clone())),
        Box::new(MockRailClient::new("rail1", 1e9, store.clone())),
    ];
    let mut reader = MultiRailReader::new(rails);
    let desc = make_descriptor(stripe_count, chunk, stripe_count as u64 * chunk);
    let mut buf = vec![0u8; (stripe_count as usize) * chunk as usize];
    let checksums: Vec<Option<String>> = (0..stripe_count as usize)
        .map(|i| Some(stripe_checksum(&data[i * chunk as usize..(i + 1) * chunk as usize])))
        .collect();

    let stats = reader.read(&desc, &checksums, &mut buf).unwrap();
    assert_eq!(buf, data, "multi-rail read must reconstruct the object exactly");
    assert_eq!(stats.rail_count, 2);
    assert!(stats.verify_ok);
}

#[test]
fn single_rail_is_pass_through() {
    let store = Arc::new(Mutex::new(MockStore::default()));
    let stripe_count = 3u32;
    let chunk = 512u64;
    let data = vec![7u8; (stripe_count as usize) * chunk as usize];
    store
        .lock()
        .unwrap()
        .objects
        .insert("2:nsobj".to_string(), Arc::new(data.clone()));

    let rails: Vec<Box<dyn RailReader>> =
        vec![Box::new(MockRailClient::new("rail0", 1e9, store.clone()))];
    let mut reader = MultiRailReader::new(rails);
    let desc = make_descriptor(stripe_count, chunk, stripe_count as u64 * chunk);
    let mut buf = vec![0u8; data.len()];
    let checksums = vec![None; stripe_count as usize];
    let stats = reader.read(&desc, &checksums, &mut buf).unwrap();
    assert_eq!(buf, data);
    assert_eq!(stats.rail_count, 1);
}

/// A/B of the use-after-free guard. Without a cancel (live == epoch) the late
/// write lands and corrupts the victim region; with a cancel (live bumped) the
/// epoch guard blocks it. This is the hazard the brief calls out explicitly.
#[test]
fn epoch_guard_blocks_late_write_after_cancel() {
    let store = Arc::new(Mutex::new(MockStore::default()));
    let stripe_count = 4u32;
    let chunk = 1024u64;
    let data = vec![1u8; (stripe_count as usize) * chunk as usize];
    store
        .lock()
        .unwrap()
        .objects
        .insert("2:nsobj".to_string(), Arc::new(data.clone()));

    let epoch = Arc::new(AtomicU64::new(7));
    let live = Arc::new(AtomicU64::new(7));
    let victim = (0usize, chunk as usize);

    // Case A: no cancel -> late write lands (demonstrates the hazard).
    {
        let fault = MockFault {
            late_write_after_cancel: true,
            epoch: Some(epoch.clone()),
            live: Some(live.clone()),
            victim_region: Some(victim),
            garbage: 0xAB,
            ..Default::default()
        };
        let rail: Box<dyn RailReader> =
            Box::new(MockRailClient::new("rail0", 1e9, store.clone()).with_fault(fault));
        let mut reader = MultiRailReader::new(vec![rail]);
        let desc = make_descriptor(stripe_count, chunk, stripe_count as u64 * chunk);
        let mut buf = vec![0u8; data.len()];
        reader.read(&desc, &vec![None; stripe_count as usize], &mut buf).unwrap();
        std::thread::sleep(Duration::from_millis(80));
        assert_eq!(
            buf[0], 0xAB,
            "without cancel the late write must corrupt the buffer (hazard reproduced)"
        );
    }

    // Case B: cancel (bump live) -> epoch guard blocks the late write.
    {
        live.store(99, Ordering::SeqCst); // simulate cancel / buffer free
        let fault = MockFault {
            late_write_after_cancel: true,
            epoch: Some(epoch.clone()),
            live: Some(live.clone()),
            victim_region: Some(victim),
            garbage: 0xAB,
            ..Default::default()
        };
        let rail: Box<dyn RailReader> =
            Box::new(MockRailClient::new("rail0", 1e9, store.clone()).with_fault(fault));
        let mut reader = MultiRailReader::new(vec![rail]);
        let desc = make_descriptor(stripe_count, chunk, stripe_count as u64 * chunk);
        let mut buf = vec![0u8; data.len()];
        reader.read(&desc, &vec![None; stripe_count as usize], &mut buf).unwrap();
        std::thread::sleep(Duration::from_millis(80));
        assert_ne!(
            buf[0], 0xAB,
            "after cancel the epoch guard must prevent the late write (no corruption)"
        );
        assert_eq!(buf[0], 1, "victim region keeps the legitimate read data");
    }
}

#[test]
fn plan_stripes_round_robin() {
    let plans = plan_stripes(4096, 1024, 4, 2, &[None; 4]);
    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].stripes, vec![0, 2]);
    assert_eq!(plans[1].stripes, vec![1, 3]);
}

/// Stripe checksum failure: inject a corrupt checksum -> assert the whole read
/// errors out and no dirty data is returned.
/// Maps to the official test matrix entry "stripe checksum failure".
#[test]
fn test_stripe_checksum_failure_detected() {
    let store = Arc::new(Mutex::new(MockStore::default()));
    let stripe_count = 4u32;
    let chunk = 1024u64;
    let data = vec![3u8; (stripe_count as usize) * chunk as usize];
    store
        .lock()
        .unwrap()
        .objects
        .insert("2:nsobj".to_string(), Arc::new(data.clone()));

    // Corrupt stripe 1 in transit: the client must detect the mismatch and
    // refuse to return the (now dirty) object to the caller.
    let fault = MockFault {
        corrupt_stripe: Some(1),
        ..Default::default()
    };
    let rail: Box<dyn RailReader> =
        Box::new(MockRailClient::new("rail0", 1e9, store.clone()).with_fault(fault));
    let mut reader = MultiRailReader::new(vec![rail]);
    let desc = make_descriptor(stripe_count, chunk, stripe_count as u64 * chunk);
    let mut buf = vec![0u8; data.len()];
    // Supply the *correct* per-stripe checksums so the consistency layer can
    // detect the in-transit corruption injected above.
    let checksums: Vec<Option<String>> = (0..stripe_count as usize)
        .map(|i| Some(stripe_checksum(&data[i * chunk as usize..(i + 1) * chunk as usize])))
        .collect();
    let res = reader.read(&desc, &checksums, &mut buf);
    assert!(res.is_err(), "checksum mismatch must surface as an error");
    assert!(
        res.unwrap_err().to_string().contains("checksum"),
        "error must name the checksum failure"
    );
}

/// Single-path disconnect: inject a disconnect -> assert safe failure semantics
/// (no partial data returned).
/// Maps to the official test matrix entry "single-path timeout/disconnect".
#[test]
fn test_single_rail_disconnect_safe_fail() {
    let store = Arc::new(Mutex::new(MockStore::default()));
    let stripe_count = 2u32;
    let chunk = 512u64;
    let data = vec![9u8; (stripe_count as usize) * chunk as usize];
    store
        .lock()
        .unwrap()
        .objects
        .insert("2:nsobj".to_string(), Arc::new(data.clone()));

    let fault = MockFault {
        fail: true,
        ..Default::default()
    };
    let rail: Box<dyn RailReader> =
        Box::new(MockRailClient::new("rail0", 1e9, store.clone()).with_fault(fault));
    let mut reader = MultiRailReader::new(vec![rail]);
    let desc = make_descriptor(stripe_count, chunk, stripe_count as u64 * chunk);
    let mut buf = vec![0u8; data.len()];
    let res = reader.read(&desc, &vec![None; stripe_count as usize], &mut buf);
    assert!(res.is_err(), "disconnect must fail the whole read");
    assert!(
        res.unwrap_err().to_string().contains("rails failed"),
        "disconnect must fail the whole read safely (never partial data)"
    );
}

/// Partial completion: one rail under-delivers (claims success but moved only
/// part of the bytes) -> caught by the consistency layer; assert whole read fails.
/// Maps to the official test matrix entry "partial completion".
#[test]
fn test_partial_completion_under_delivery_detected() {
    let store = Arc::new(Mutex::new(MockStore::default()));
    let stripe_count = 4u32;
    let chunk = 1024u64;
    let data = vec![5u8; (stripe_count as usize) * chunk as usize];
    store
        .lock()
        .unwrap()
        .objects
        .insert("2:nsobj".to_string(), Arc::new(data.clone()));

    // Rail under-delivers stripe 0 (only 16 of 1024 bytes), claiming success.
    let fault = MockFault {
        short_read: Some(16),
        ..Default::default()
    };
    let rail: Box<dyn RailReader> =
        Box::new(MockRailClient::new("rail0", 1e9, store.clone()).with_fault(fault));
    let mut reader = MultiRailReader::new(vec![rail]);
    let desc = make_descriptor(stripe_count, chunk, stripe_count as u64 * chunk);
    let mut buf = vec![0u8; data.len()];
    // Supply the *correct* per-stripe checksums so the consistency layer can
    // detect the missing bytes left behind by the under-delivering rail.
    let checksums: Vec<Option<String>> = (0..stripe_count as usize)
        .map(|i| Some(stripe_checksum(&data[i * chunk as usize..(i + 1) * chunk as usize])))
        .collect();
    let res = reader.read(&desc, &checksums, &mut buf);
    assert!(
        res.is_err(),
        "under-delivery must be caught by the consistency layer"
    );
}

/// Single-path timeout: inject a stall (5s) but grant only a 200ms deadline ->
/// assert a safe timeout failure that returns quickly (no hang) and leaves the
/// caller's buffer untouched. Maps to "single-path timeout/disconnect".
#[test]
fn test_single_rail_timeout_safe_fail() {
    let store = Arc::new(Mutex::new(MockStore::default()));
    let stripe_count = 2u32;
    let chunk = 512u64;
    let data = vec![1u8; (stripe_count as usize) * chunk as usize];
    store
        .lock()
        .unwrap()
        .objects
        .insert("2:nsobj".to_string(), Arc::new(data.clone()));

    // Rail stalls 5s; we only give 200ms. The read must abort safely and fast.
    let fault = MockFault {
        stall_secs: 5.0,
        ..Default::default()
    };
    let rail: Box<dyn RailReader> =
        Box::new(MockRailClient::new("rail0", 1e9, store.clone()).with_fault(fault));
    let mut reader = MultiRailReader::new(vec![rail]);
    let desc = make_descriptor(stripe_count, chunk, stripe_count as u64 * chunk);
    let mut buf = vec![0u8; data.len()];
    let start = std::time::Instant::now();
    let res = reader.read_ex(
        &desc,
        &vec![None; stripe_count as usize],
        &mut buf,
        ReadOptions {
            timeout: Some(Duration::from_millis(200)),
            ..Default::default()
        },
    );
    let elapsed = start.elapsed();
    assert!(res.is_err(), "stalled rail must trigger a timeout error");
    assert!(
        elapsed < Duration::from_secs(2),
        "read must not hang for the full stall (elapsed {:?})",
        elapsed
    );
    assert!(
        res.unwrap_err().to_string().contains("timed out"),
        "error must name the timeout"
    );
    // The caller's buffer must be untouched on timeout.
    assert_eq!(buf, vec![0u8; data.len()], "buffer must be untouched on timeout");
}

/// Resource budget: object exceeds the inflight budget -> backpressure rejection;
/// within budget -> success. Maps to the official test matrix entry "resource limit".
#[test]
fn test_resource_limit_rejects_oversized() {
    let store = Arc::new(Mutex::new(MockStore::default()));
    let stripe_count = 4u32;
    let chunk = 1024u64;
    let data = vec![2u8; (stripe_count as usize) * chunk as usize];
    store
        .lock()
        .unwrap()
        .objects
        .insert("2:nsobj".to_string(), Arc::new(data.clone()));
    let desc = make_descriptor(stripe_count, chunk, stripe_count as u64 * chunk); // size = 4096
    let mut buf = vec![0u8; data.len()];

    // Budget of 2048 < 4096 -> must be rejected (backpressure).
    let rails: Vec<Box<dyn RailReader>> = vec![
        Box::new(MockRailClient::new("rail0", 1e9, store.clone())),
        Box::new(MockRailClient::new("rail1", 1e9, store.clone())),
    ];
    let mut reader = MultiRailReader::new(rails);
    let res = reader.read_ex(
        &desc,
        &vec![None; stripe_count as usize],
        &mut buf,
        ReadOptions {
            max_inflight_bytes: Some(2048),
            ..Default::default()
        },
    );
    assert!(
        res.is_err(),
        "oversized object must be rejected by the budget guard"
    );
    assert!(
        res.unwrap_err().to_string().contains("budget"),
        "error must name the backpressure budget"
    );

    // Within budget -> succeeds.
    let rails2: Vec<Box<dyn RailReader>> = vec![
        Box::new(MockRailClient::new("rail0", 1e9, store.clone())),
        Box::new(MockRailClient::new("rail1", 1e9, store.clone())),
    ];
    let mut reader2 = MultiRailReader::new(rails2);
    let ok = reader2.read_ex(
        &desc,
        &vec![None; stripe_count as usize],
        &mut buf,
        ReadOptions {
            max_inflight_bytes: Some(8192),
            ..Default::default()
        },
    );
    assert!(ok.is_ok(), "within-budget read must succeed");
}

/// Multi-rail layer version change: descriptor generation disagrees with the
/// generation observed at lookup -> reject (guard against stale reads); agree ->
/// success. Maps to "object version change".
#[test]
fn test_version_change_rejected() {
    let store = Arc::new(Mutex::new(MockStore::default()));
    let stripe_count = 2u32;
    let chunk = 512u64;
    let data = vec![4u8; (stripe_count as usize) * chunk as usize];
    store
        .lock()
        .unwrap()
        .objects
        .insert("2:nsobj".to_string(), Arc::new(data.clone()));
    // descriptor generation is 1 (make_descriptor sets object_generation = 1).
    let desc = make_descriptor(stripe_count, chunk, stripe_count as u64 * chunk);
    let mut buf = vec![0u8; data.len()];

    // Expecting a different generation (stale placement) -> must be rejected.
    let rails: Vec<Box<dyn RailReader>> = vec![
        Box::new(MockRailClient::new("rail0", 1e9, store.clone())),
        Box::new(MockRailClient::new("rail1", 1e9, store.clone())),
    ];
    let mut reader = MultiRailReader::new(rails);
    let res = reader.read_ex(
        &desc,
        &vec![None; stripe_count as usize],
        &mut buf,
        ReadOptions {
            expected_generation: Some(99),
            ..Default::default()
        },
    );
    assert!(
        res.is_err(),
        "generation mismatch must be rejected as a stale read"
    );
    assert!(
        res.unwrap_err().to_string().contains("generation"),
        "error must name the generation mismatch"
    );

    // Matching generation -> succeeds.
    let rails2: Vec<Box<dyn RailReader>> = vec![
        Box::new(MockRailClient::new("rail0", 1e9, store.clone())),
        Box::new(MockRailClient::new("rail1", 1e9, store.clone())),
    ];
    let mut reader2 = MultiRailReader::new(rails2);
    let ok = reader2.read_ex(
        &desc,
        &vec![None; stripe_count as usize],
        &mut buf,
        ReadOptions {
            expected_generation: Some(1),
            ..Default::default()
        },
    );
    assert!(ok.is_ok(), "matching generation must succeed");
}

/// Sanity: a timeout is set but the rail finishes fast -> success and the object
/// is reconstructed correctly (verifies the timeout path does not break the
/// normal-completion semantics).
#[test]
fn test_read_with_timeout_succeeds_when_fast() {
    let store = Arc::new(Mutex::new(MockStore::default()));
    let stripe_count = 4u32;
    let chunk = 1024u64;
    let data = vec![6u8; (stripe_count as usize) * chunk as usize];
    store
        .lock()
        .unwrap()
        .objects
        .insert("2:nsobj".to_string(), Arc::new(data.clone()));
    let rails: Vec<Box<dyn RailReader>> = vec![
        Box::new(MockRailClient::new("rail0", 1e9, store.clone())),
        Box::new(MockRailClient::new("rail1", 1e9, store.clone())),
    ];
    let mut reader = MultiRailReader::new(rails);
    let desc = make_descriptor(stripe_count, chunk, stripe_count as u64 * chunk);
    let mut buf = vec![0u8; data.len()];
    let res = reader.read_ex(
        &desc,
        &vec![None; stripe_count as usize],
        &mut buf,
        ReadOptions {
            timeout: Some(Duration::from_secs(5)),
            ..Default::default()
        },
    );
    assert!(res.is_ok(), "fast read within timeout must succeed");
    assert_eq!(
        buf, data,
        "timeout path must reconstruct the object exactly"
    );
}
