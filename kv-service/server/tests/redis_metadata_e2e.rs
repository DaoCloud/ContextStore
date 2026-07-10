//! Redis metadata end-to-end tests.
//!
//! These tests require a single Redis instance. They are skipped unless
//! `CS_REDIS_URL` is set, for example:
//! `CS_REDIS_URL=redis://127.0.0.1:6379/ cargo test --test redis_metadata_e2e -- --nocapture`.

use contextstore_server::config::Config;
use contextstore_server::metadata::BlockMeta;
use contextstore_server::router::ObjectKey;
use contextstore_server::KVServiceContext;
use prost::bytes::Bytes;
use redis::Commands;
use std::time::{SystemTime, UNIX_EPOCH};
use tempfile::TempDir;

fn redis_url() -> Option<String> {
    std::env::var("CS_REDIS_URL")
        .ok()
        .filter(|url| !url.is_empty())
}

fn require_redis() -> Option<String> {
    let Some(url) = redis_url() else {
        eprintln!("SKIP: CS_REDIS_URL is not set");
        return None;
    };
    let client = redis::Client::open(url.as_str()).expect("open Redis client");
    let mut connection = client.get_connection().expect("connect Redis");
    redis::cmd("PING")
        .query::<String>(&mut connection)
        .expect("ping Redis");
    Some(url)
}

fn unique_prefix(test_name: &str) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "contextstore:e2e:{}:{}:{}:",
        test_name,
        std::process::id(),
        now
    )
}

fn config(redis_url: &str, redis_key_prefix: &str, dir: &TempDir) -> Config {
    let mut cfg = Config::default();
    cfg.metadata.redis_url = redis_url.to_string();
    cfg.metadata.redis_key_prefix = redis_key_prefix.to_string();
    cfg.storage.devices = vec![dir.path().join("nvme0"), dir.path().join("nvme1")];
    cfg.storage.striping_threshold = 0;
    cfg
}

fn meta() -> BlockMeta {
    BlockMeta {
        device_id: 0,
        file_path: String::new(),
        size: 0,
        object_handle: String::new(),
        object_generation: 1,
        content_etag: String::new(),
        layout_version: 1,
        created_at: 0,
        last_accessed_at: 0,
        ttl_seconds: 0,
        num_tokens: 16,
        num_layers: 1,
        dtype: "bfloat16".to_string(),
        compressed: false,
        striping: None,
    }
}

fn delete_metadata_keys(redis_url: &str, prefix: &str, key: &ObjectKey) {
    let Ok(client) = redis::Client::open(redis_url) else {
        return;
    };
    let Ok(mut connection) = client.get_connection() else {
        return;
    };
    let logical_key = key.to_string_key();
    let _: redis::RedisResult<()> = connection.del(format!("{}block_meta:{}", prefix, logical_key));
    let _: redis::RedisResult<()> = connection.del(format!("{}generation:{}", prefix, logical_key));
}

#[test]
fn shared_redis_metadata_is_visible_across_contexts() {
    let Some(redis_url) = require_redis() else {
        return;
    };
    let prefix = unique_prefix("shared");
    let tmp = TempDir::new().unwrap();
    let key = ObjectKey {
        namespace: "redis-e2e".to_string(),
        object_key: "shared-metadata".to_string(),
    };

    let ctx_a = KVServiceContext::new(config(&redis_url, &prefix, &tmp)).unwrap();
    let ctx_b = KVServiceContext::new(config(&redis_url, &prefix, &tmp)).unwrap();

    ctx_a
        .storage
        .put(&key, Bytes::from_static(b"first"), meta())
        .unwrap();

    let (data, first_meta) = ctx_b.storage.get(&key).unwrap().unwrap();
    assert_eq!(data.as_ref(), b"first");
    assert_eq!(first_meta.object_generation, 1);

    ctx_b
        .storage
        .put(&key, Bytes::from_static(b"second"), meta())
        .unwrap();
    let (data, second_meta) = ctx_a.storage.get(&key).unwrap().unwrap();
    assert_eq!(data.as_ref(), b"second");
    assert_eq!(second_meta.object_generation, 2);

    delete_metadata_keys(&redis_url, &prefix, &key);
}

#[test]
fn redis_if_absent_commits_only_once() {
    let Some(redis_url) = require_redis() else {
        return;
    };
    let prefix = unique_prefix("if-absent");
    let tmp = TempDir::new().unwrap();
    let key = ObjectKey {
        namespace: "redis-e2e".to_string(),
        object_key: "if-absent".to_string(),
    };

    let ctx_a = KVServiceContext::new(config(&redis_url, &prefix, &tmp)).unwrap();
    let ctx_b = KVServiceContext::new(config(&redis_url, &prefix, &tmp)).unwrap();

    assert!(ctx_a
        .storage
        .put_if_absent(&key, Bytes::from_static(b"winner"), meta())
        .unwrap());
    assert!(!ctx_b
        .storage
        .put_if_absent(&key, Bytes::from_static(b"loser"), meta())
        .unwrap());

    let (data, committed) = ctx_b.storage.get(&key).unwrap().unwrap();
    assert_eq!(data.as_ref(), b"winner");
    assert_eq!(committed.object_generation, 1);

    delete_metadata_keys(&redis_url, &prefix, &key);
}
