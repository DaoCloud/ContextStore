//! Hardware-free tests for the multi-rail read path (run with default features).

use contextstore_client_rs::mock_rail::{MockFault, MockRailClient, MockStore};
use contextstore_client_rs::multi_rail::{plan_stripes, stripe_checksum, MultiRailReader, RailReader};
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
        .insert("2:nsobj".to_string(), data.clone());

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
        .insert("2:nsobj".to_string(), data.clone());

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
        .insert("2:nsobj".to_string(), data.clone());

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
