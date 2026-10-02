//! Hardware-gated checks for one Worker reading one object over two RDMA rails.
//!
//! Configure a striped KVService with two independent listeners, then set
//! `CS_RAIL_COORDINATOR`, `CS_RAIL_LISTENER0/1`, and `CS_RAIL_DEVICE0/1`.

#![cfg(feature = "rdma")]

use contextstore_client_rs::rail_read::{RailCancel, RailLimits, RailReader, RailRoute};
use contextstore_client_rs::rdma::{RdmaClient, RdmaClientConfig};
use contextstore_client_rs::KvClient;
use prost::bytes::Bytes;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn setting(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn route(advertised: &str, index: usize, listener: &str) -> RailRoute {
    let device = setting(&format!("CS_RAIL_DEVICE{index}"), &format!("rxe{index}"));
    let gid = setting(&format!("CS_RAIL_GID{index}"), "3")
        .parse::<u8>()
        .expect("GID index");
    RailRoute::new(
        format!("rail{index}"),
        advertised,
        RdmaClientConfig::new(listener, device).with_gid_index(gid),
    )
}

async fn seeded_object() -> (KvClient, String, Vec<u8>, String) {
    let coordinator = setting("CS_RAIL_COORDINATOR", "http://127.0.0.1:50051");
    let mut client = KvClient::connect(coordinator).await.expect("connect gRPC");
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("wall clock")
        .as_nanos();
    let key = format!("rail-e2e-{nanos}");
    let mib = setting("CS_RAIL_OBJECT_MIB", "64")
        .parse::<usize>()
        .expect("object size in MiB");
    let payload: Vec<u8> = (0..mib * 1024 * 1024)
        .map(|offset| (offset % 251) as u8)
        .collect();
    let bytes = Bytes::from(payload.clone());
    let segment_size = 4 * 1024 * 1024;
    let segments = (0..bytes.len())
        .step_by(segment_size)
        .map(|offset| bytes.slice(offset..(offset + segment_size).min(bytes.len())))
        .collect();
    client
        .put_stream_chunks("rail-e2e", &key, segments)
        .await
        .expect("seed object");
    let lookup = client
        .lookup_object("rail-e2e", &key)
        .await
        .expect("lookup")
        .expect("seeded object exists");
    assert!(
        lookup.descriptor.is_striped && lookup.descriptor.stripe_count >= 2,
        "configure striped storage and seed an object above its threshold"
    );
    let advertised = lookup
        .placement
        .as_ref()
        .expect("placement")
        .chunks
        .first()
        .expect("stripe")
        .rdma_endpoint
        .clone();
    assert!(
        lookup
            .placement
            .as_ref()
            .expect("placement")
            .chunks
            .iter()
            .all(|chunk| chunk.rdma_endpoint == advertised),
        "this test requires one storage node with two RDMA listeners"
    );
    (client, key, payload, advertised)
}

#[tokio::test]
#[ignore = "requires one reachable HCA or RXE listener and a striped KVService"]
async fn single_real_rail_restores_verified_object() {
    let (mut client, key, payload, advertised) = seeded_object().await;
    let listener = setting("CS_RAIL_LISTENER0", &advertised);
    let reader = Arc::new(
        RailReader::new(
            vec![route(&advertised, 0, &listener)],
            RailLimits::default(),
        )
        .expect("single rail"),
    );
    let mut destination = vec![0xA5; payload.len()];
    assert_eq!(
        client
            .read_multi_rail_into(
                Arc::clone(&reader),
                "rail-e2e",
                &key,
                &mut destination,
                None
            )
            .await
            .expect("real rail read"),
        Some(payload.len())
    );
    assert_eq!(destination, payload);
    assert_eq!(reader.snapshots()[0].bytes, payload.len() as u64);
}

#[tokio::test]
#[ignore = "requires a striped KVService and an injected dead RDMA listener"]
async fn dead_single_real_rail_leaves_destination_unchanged() {
    let (mut client, key, payload, advertised) = seeded_object().await;
    let dead = setting("CS_RAIL_DEAD_LISTENER", "127.0.0.1:59999");
    let reader = Arc::new(
        RailReader::new(
            vec![route(&advertised, 0, &dead)],
            RailLimits {
                io_timeout: Duration::from_secs(3),
                ..RailLimits::default()
            },
        )
        .expect("dead rail"),
    );
    let mut destination = vec![0xA5; payload.len()];
    assert!(client
        .read_multi_rail_into(reader, "rail-e2e", &key, &mut destination, None)
        .await
        .is_err());
    assert!(destination.iter().all(|byte| *byte == 0xA5));
}

#[tokio::test]
#[ignore = "requires one reachable HCA or RXE listener and a large striped object"]
async fn cancellation_after_real_transfer_starts_cannot_write_reused_buffer() {
    let (mut client, key, payload, advertised) = seeded_object().await;
    let listener = setting("CS_RAIL_LISTENER0", &advertised);
    let reader = Arc::new(
        RailReader::new(
            vec![route(&advertised, 0, &listener)],
            RailLimits::default(),
        )
        .expect("single rail"),
    );
    let cancel = RailCancel::default();
    let task_reader = Arc::clone(&reader);
    let task_cancel = cancel.clone();
    let read_task = tokio::spawn(async move {
        let mut destination = vec![0xA5; payload.len()];
        let result = client
            .read_multi_rail_into(
                task_reader,
                "rail-e2e",
                &key,
                &mut destination,
                Some(task_cancel),
            )
            .await;
        (result, destination)
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    while reader.snapshots()[0].inflight_requests == 0 {
        assert!(
            Instant::now() < deadline,
            "real RDMA transfer never started"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    cancel.cancel();
    let (result, mut destination) = tokio::time::timeout(Duration::from_secs(40), read_task)
        .await
        .expect("cancelled transfer did not quiesce")
        .expect("read task joined");
    assert!(result
        .expect_err("cancelled transfer must fail")
        .to_string()
        .contains("cancelled"));
    assert!(destination.iter().all(|byte| *byte == 0xA5));
    destination.fill(0x33);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(destination.iter().all(|byte| *byte == 0x33));
    assert_eq!(reader.snapshots()[0].inflight_requests, 0);
    assert_eq!(reader.snapshots()[0].registered_bytes, 0);
}

#[tokio::test]
#[ignore = "requires two real rails and a large striped object"]
async fn cancellation_during_two_rail_transfer_preserves_reused_buffer() {
    let (mut client, key, payload, advertised) = seeded_object().await;
    let listener0 = setting("CS_RAIL_LISTENER0", &advertised);
    let listener1 = setting("CS_RAIL_LISTENER1", "127.0.0.1:50054");
    let reader = Arc::new(
        RailReader::new(
            vec![
                route(&advertised, 0, &listener0),
                route(&advertised, 1, &listener1),
            ],
            RailLimits::default(),
        )
        .expect("dual reader"),
    );
    let cancel = RailCancel::default();
    let task_reader = Arc::clone(&reader);
    let task_cancel = cancel.clone();
    let read_task = tokio::spawn(async move {
        let mut destination = vec![0xA5; payload.len()];
        let result = client
            .read_multi_rail_into(
                task_reader,
                "rail-e2e",
                &key,
                &mut destination,
                Some(task_cancel),
            )
            .await;
        (result, destination)
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    while !reader
        .snapshots()
        .iter()
        .all(|snapshot| snapshot.inflight_requests > 0)
    {
        assert!(
            Instant::now() < deadline,
            "both RDMA rails never entered flight"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    cancel.cancel();
    let (result, mut destination) = tokio::time::timeout(Duration::from_secs(40), read_task)
        .await
        .expect("cancelled dual transfer did not quiesce")
        .expect("read task joined");
    assert!(result
        .expect_err("cancelled dual transfer must fail")
        .to_string()
        .contains("cancelled"));
    assert!(destination.iter().all(|byte| *byte == 0xA5));
    destination.fill(0x33);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(destination.iter().all(|byte| *byte == 0x33));
    assert!(reader
        .snapshots()
        .iter()
        .all(|snapshot| snapshot.inflight_requests == 0 && snapshot.registered_bytes == 0));
}

#[tokio::test]
#[ignore = "requires an existing striped object with one deliberately corrupted physical stripe"]
async fn corrupted_real_stripe_does_not_publish_bytes() {
    let coordinator = setting("CS_RAIL_COORDINATOR", "http://127.0.0.1:50051");
    let namespace = setting("CS_RAIL_EXISTING_NAMESPACE", "rust-bench");
    let key = setting("CS_RAIL_EXISTING_KEY", "railtest0/__combined__");
    let mut client = KvClient::connect(coordinator).await.expect("connect gRPC");
    let lookup = client
        .lookup_object(&namespace, &key)
        .await
        .expect("lookup")
        .expect("object exists");
    let placement = lookup.placement.expect("placement");
    assert!(
        placement
            .chunks
            .iter()
            .all(|chunk| !chunk.checksum.is_empty()),
        "checksum injection requires a newly written, checksummed object"
    );
    let advertised = placement.chunks[0].rdma_endpoint.clone();
    let listener = setting("CS_RAIL_LISTENER0", &advertised);
    let reader = Arc::new(
        RailReader::new(
            vec![route(&advertised, 0, &listener)],
            RailLimits::default(),
        )
        .expect("single rail"),
    );
    let mut destination = vec![0xA5; lookup.descriptor.size as usize];
    assert!(client
        .read_multi_rail_into(reader, &namespace, &key, &mut destination, None)
        .await
        .is_err());
    assert!(destination.iter().all(|byte| *byte == 0xA5));
}

#[tokio::test]
#[ignore = "requires two real rails and a deliberately corrupted stripe 1 on an isolated object"]
async fn corrupt_stripe_on_second_rail_cannot_publish_partial_object() {
    let coordinator = setting("CS_RAIL_COORDINATOR", "http://127.0.0.1:50051");
    let namespace = setting("CS_RAIL_EXISTING_NAMESPACE", "rust-bench");
    let key = setting("CS_RAIL_EXISTING_KEY", "rail-corrupt0/__combined__");
    let mut client = KvClient::connect(coordinator).await.expect("connect gRPC");
    let lookup = client
        .lookup_object(&namespace, &key)
        .await
        .expect("lookup")
        .expect("object exists");
    let placement = lookup.placement.expect("placement");
    assert!(
        placement
            .chunks
            .iter()
            .all(|chunk| !chunk.checksum.is_empty()),
        "a newly written checksummed object is required"
    );
    let advertised = placement.chunks[0].rdma_endpoint.clone();
    let listener0 = setting("CS_RAIL_LISTENER0", &advertised);
    let listener1 = setting("CS_RAIL_LISTENER1", "127.0.0.1:50054");
    let reader = Arc::new(
        RailReader::new(
            vec![
                route(&advertised, 0, &listener0),
                route(&advertised, 1, &listener1),
            ],
            RailLimits::default(),
        )
        .expect("dual reader"),
    );
    let mut destination = vec![0xA5; lookup.descriptor.size as usize];
    assert!(client
        .read_multi_rail_into(
            Arc::clone(&reader),
            &namespace,
            &key,
            &mut destination,
            None
        )
        .await
        .is_err());
    assert!(destination.iter().all(|byte| *byte == 0xA5));
    let snapshots = reader.snapshots();
    assert!(
        snapshots[0].bytes > 0,
        "healthy first rail should have finished"
    );
    assert!(
        snapshots[1].reads_err > 0 || snapshots[1].bytes > 0,
        "second rail must have been involved"
    );
}

#[tokio::test]
#[ignore = "requires CS_RDMA_TEST_PRE_WRITE_DELAY_MS=3000 on an isolated real RDMA server"]
async fn late_server_write_after_timeout_cannot_corrupt_reused_buffer() {
    let (mut client, key, payload, advertised) = seeded_object().await;
    let listener = setting("CS_RAIL_LISTENER0", &advertised);
    let reader = Arc::new(
        RailReader::new(
            vec![route(&advertised, 0, &listener)],
            RailLimits {
                io_timeout: Duration::from_secs(1),
                ..RailLimits::default()
            },
        )
        .expect("single rail"),
    );
    let mut destination = vec![0xA5; payload.len()];
    assert!(client
        .read_multi_rail_into(reader, "rail-e2e", &key, &mut destination, None)
        .await
        .is_err());
    assert!(destination.iter().all(|byte| *byte == 0xA5));
    destination.fill(0x33);
    tokio::time::sleep(Duration::from_secs(4)).await;
    assert!(destination.iter().all(|byte| *byte == 0x33));
}

#[tokio::test]
#[ignore = "requires two real rails and a three-second pre-WRITE delay on server nic_idx=1"]
async fn late_second_rail_after_first_completion_cannot_publish_or_corrupt() {
    let (mut client, key, payload, advertised) = seeded_object().await;
    let listener0 = setting("CS_RAIL_LISTENER0", &advertised);
    let listener1 = setting("CS_RAIL_LISTENER1", "127.0.0.1:50054");
    let reader = Arc::new(
        RailReader::new(
            vec![
                route(&advertised, 0, &listener0),
                route(&advertised, 1, &listener1),
            ],
            RailLimits {
                io_timeout: Duration::from_secs(1),
                ..RailLimits::default()
            },
        )
        .expect("dual reader"),
    );
    let mut destination = vec![0xA5; payload.len()];
    assert!(client
        .read_multi_rail_into(
            Arc::clone(&reader),
            "rail-e2e",
            &key,
            &mut destination,
            None,
        )
        .await
        .is_err());
    let snapshots = reader.snapshots();
    assert_eq!(snapshots[0].reads_ok, 1, "first rail must have completed");
    assert_eq!(snapshots[1].reads_err, 1, "second rail must have failed");
    assert!(destination.iter().all(|byte| *byte == 0xA5));
    destination.fill(0x33);
    tokio::time::sleep(Duration::from_secs(8)).await;
    assert!(destination.iter().all(|byte| *byte == 0x33));
    assert!(reader
        .snapshots()
        .iter()
        .all(|snapshot| snapshot.inflight_requests == 0 && snapshot.registered_bytes == 0));
}

#[tokio::test]
#[ignore = "requires a three-second delay on server nic_idx=1 and a two-second CQ deadline"]
async fn uncertain_completion_retires_old_server_connection() {
    let (mut client, key, _payload, _advertised) = seeded_object().await;
    let descriptor = client
        .lookup_object("rail-e2e", &key)
        .await
        .expect("lookup")
        .expect("seeded object")
        .descriptor;
    let device = setting("CS_RAIL_DEVICE1", "rxe_c1");
    let listener = setting("CS_RAIL_LISTENER1", "127.0.0.1:50054");
    let gid = setting("CS_RAIL_GID1", "1")
        .parse::<u8>()
        .expect("GID index");
    let mut rdma = RdmaClient::connect(
        RdmaClientConfig::new(listener, device)
            .with_gid_index(gid)
            .with_io_timeout(Duration::from_secs(1)),
    )
    .expect("second rail connects");
    let mut target = vec![0u8; descriptor.size as usize];
    let registered = rdma.register_buffer(&mut target).expect("register buffer");
    let view = registered.view();
    let segments = [(view.addr(), view.rkey(), descriptor.size)];
    assert!(rdma
        .get_descriptor_stripes_sge_detailed(&descriptor, &[1], &segments)
        .is_err());
    tokio::time::sleep(Duration::from_secs(6)).await;
    assert!(
        rdma.get_descriptor_stripes_sge_detailed(&descriptor, &[1], &segments)
            .is_err(),
        "server must close the old QP/CQ rather than return a stale response"
    );
    drop(rdma);
    drop(registered);
}

#[tokio::test]
#[ignore = "requires two reachable RDMA listeners on the same storage node"]
async fn same_object_matches_single_and_dual_rail() {
    let (mut client, key, payload, advertised) = seeded_object().await;
    let listener0 = setting("CS_RAIL_LISTENER0", &advertised);
    let listener1 = setting("CS_RAIL_LISTENER1", "127.0.0.1:50054");
    let limits = RailLimits {
        io_timeout: Duration::from_secs(10),
        ..RailLimits::default()
    };
    let dual = Arc::new(
        RailReader::new(
            vec![
                route(&advertised, 0, &listener0),
                route(&advertised, 1, &listener1),
            ],
            limits.clone(),
        )
        .expect("dual reader"),
    );
    let mut dual_bytes = vec![0xA5; payload.len()];
    assert_eq!(
        client
            .read_multi_rail_into(Arc::clone(&dual), "rail-e2e", &key, &mut dual_bytes, None)
            .await
            .expect("dual read"),
        Some(payload.len())
    );
    assert_eq!(dual_bytes, payload);
    assert!(dual.snapshots().iter().all(|snapshot| snapshot.bytes > 0));

    let single = Arc::new(
        RailReader::new(vec![route(&advertised, 0, &listener0)], limits).expect("single reader"),
    );
    let mut single_bytes = vec![0x5A; payload.len()];
    assert_eq!(
        client
            .read_multi_rail_into(single, "rail-e2e", &key, &mut single_bytes, None)
            .await
            .expect("single read"),
        Some(payload.len())
    );
    assert_eq!(single_bytes, dual_bytes);
}

#[tokio::test]
#[ignore = "requires one reachable RDMA listener and one injected dead listener"]
async fn failed_second_rail_does_not_publish_partial_bytes() {
    let (mut client, key, payload, advertised) = seeded_object().await;
    let listener0 = setting("CS_RAIL_LISTENER0", &advertised);
    let dead = setting("CS_RAIL_DEAD_LISTENER", "127.0.0.1:59999");
    let limits = RailLimits {
        io_timeout: Duration::from_secs(3),
        ..RailLimits::default()
    };
    let reader = Arc::new(
        RailReader::new(
            vec![
                route(&advertised, 0, &listener0),
                route(&advertised, 1, &dead),
            ],
            limits,
        )
        .expect("reader"),
    );
    let mut destination = vec![0xA5; payload.len()];
    assert!(client
        .read_multi_rail_into(reader, "rail-e2e", &key, &mut destination, None)
        .await
        .is_err());
    assert!(destination.iter().all(|byte| *byte == 0xA5));
}

#[tokio::test]
#[ignore = "requires two reachable RDMA listeners and a striped KVService"]
async fn old_generation_is_rejected_without_publishing_bytes() {
    let (mut client, key, mut payload, advertised) = seeded_object().await;
    let old = client
        .lookup_object("rail-e2e", &key)
        .await
        .expect("lookup")
        .expect("seeded object");
    assert!(client.delete("rail-e2e", &key).await.expect("delete"));
    payload[0] ^= 0xFF;
    let bytes = Bytes::from(payload.clone());
    let segment_size = 4 * 1024 * 1024;
    let segments = (0..bytes.len())
        .step_by(segment_size)
        .map(|offset| bytes.slice(offset..(offset + segment_size).min(bytes.len())))
        .collect();
    client
        .put_stream_chunks("rail-e2e", &key, segments)
        .await
        .expect("rewrite");
    let reader = RailReader::new(
        vec![
            route(&advertised, 0, &setting("CS_RAIL_LISTENER0", &advertised)),
            route(
                &advertised,
                1,
                &setting("CS_RAIL_LISTENER1", "127.0.0.1:50054"),
            ),
        ],
        RailLimits::default(),
    )
    .expect("reader");
    let mut destination = vec![0xA5; payload.len()];
    assert!(reader
        .read_into(
            &old.descriptor,
            old.placement.as_ref().expect("old placement"),
            &mut destination,
            None,
        )
        .is_err());
    assert!(destination.iter().all(|byte| *byte == 0xA5));
}
