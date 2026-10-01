//! Hardware-gated checks for one Worker reading one object over two RDMA rails.
//!
//! Configure a striped KVService with two independent listeners, then set
//! `CS_RAIL_COORDINATOR`, `CS_RAIL_LISTENER0/1`, and `CS_RAIL_DEVICE0/1`.

#![cfg(feature = "rdma")]

use contextstore_client_rs::rail_read::{RailLimits, RailReader, RailRoute};
use contextstore_client_rs::rdma::RdmaClientConfig;
use contextstore_client_rs::KvClient;
use prost::bytes::Bytes;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
