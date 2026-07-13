//! Hardware-gated end-to-end coverage for the native RDMA client.
//!
//! Run explicitly on a host with an RDMA-enabled ContextStore server:
//! `cargo test --manifest-path kv-service/client-rs/Cargo.toml --features rdma
//! --test rdma_e2e -- --ignored`.

#![cfg(feature = "rdma")]

use contextstore_client_rs::rdma::{RdmaClient, RdmaClientConfig};
use contextstore_client_rs::KvClient;
use std::time::{SystemTime, UNIX_EPOCH};

fn env_u8(name: &str, default: u8) -> u8 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn unique_key(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_nanos();
    format!("{prefix}-{nanos}")
}

#[test]
#[ignore = "requires an RDMA-enabled ContextStore server and an HCA"]
fn rdma_put_and_get_round_trip() {
    let grpc_endpoint =
        std::env::var("CS_GRPC_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:50051".to_string());
    let rdma_endpoint =
        std::env::var("CS_RDMA_ENDPOINT").unwrap_or_else(|_| "127.0.0.1:50053".to_string());
    let device = std::env::var("CS_RDMA_DEVICE").unwrap_or_else(|_| "mlx5_0".to_string());
    let config = RdmaClientConfig::new(rdma_endpoint, device)
        .with_port(env_u8("CS_RDMA_PORT", 1))
        .with_gid_index(env_u8("CS_RDMA_GID_INDEX", 3));
    let payload: Vec<u8> = (0..8192u32).map(|value| (value % 251) as u8).collect();
    let namespace = "rdma-sdk-e2e";
    let read_key = unique_key("read");
    let write_key = unique_key("write");

    let runtime = tokio::runtime::Runtime::new().expect("create tokio runtime");
    let mut grpc = runtime
        .block_on(KvClient::connect(grpc_endpoint))
        .expect("connect gRPC client");
    assert!(runtime
        .block_on(grpc.put(namespace, &read_key, payload.clone()))
        .expect("seed object through gRPC"));

    let mut rdma = RdmaClient::connect(config).expect("connect RDMA client");
    let mut read_buffer = vec![0u8; payload.len()];
    let registered_read = rdma
        .register_buffer(&mut read_buffer)
        .expect("register RDMA read buffer");
    assert_eq!(
        rdma.get_into(namespace, &read_key, &registered_read, 0)
            .expect("read through RDMA"),
        Some(payload.len())
    );
    assert_eq!(registered_read.as_slice(), payload.as_slice());
    drop(registered_read);

    let mut write_buffer = payload.clone();
    let registered_write = rdma
        .register_buffer(&mut write_buffer)
        .expect("register RDMA write buffer");
    assert!(rdma
        .put_if_absent_from(namespace, &write_key, &registered_write, 0, payload.len())
        .expect("write through RDMA"));
    drop(registered_write);

    assert_eq!(
        runtime
            .block_on(grpc.get(namespace, &write_key))
            .expect("verify RDMA write through gRPC"),
        Some(payload)
    );

    let mut second_write = vec![42u8; 8192];
    let second_write_len = second_write.len();
    let registered_second = rdma
        .register_buffer(&mut second_write)
        .expect("register second RDMA write buffer");
    assert!(!rdma
        .put_if_absent_from(
            namespace,
            &write_key,
            &registered_second,
            0,
            second_write_len
        )
        .expect("repeat immutable RDMA write"));
    drop(registered_second);
}
