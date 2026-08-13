//! Hardware-independent E2E coverage for an isolated two-node local cluster.

use contextstore_client_rs::KvClient;
use std::collections::HashSet;
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

const TEST_NAMESPACE: &str = "e2e";
const TEST_OBJECT_KEY: &str = "striped-object";
const STREAM_CHUNK_BYTES: usize = 128 * 1024;
const STRIPED_OBJECT_BYTES: usize = 4 * 1024 * 1024;

fn ports() -> [u16; 3] {
    let listeners: Vec<TcpListener> = (0..3)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    [
        listeners[0].local_addr().unwrap().port(),
        listeners[1].local_addr().unwrap().port(),
        listeners[2].local_addr().unwrap().port(),
    ]
}

fn wait_for_redis(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) {
            stream.write_all(b"*1\r\n$4\r\nPING\r\n").unwrap();
            let mut response = [0; 16];
            if stream.read(&mut response).is_ok() && response.starts_with(b"+PONG") {
                return;
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    panic!("temporary Redis did not become ready");
}

fn stop(child: &mut Child) {
    if child.try_wait().unwrap().is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

struct Cluster {
    scenario: &'static str,
    root: TempDir,
    server: PathBuf,
    redis_port: u16,
    a_port: u16,
    b_port: u16,
    redis: Child,
    a: Option<Child>,
    b: Option<Child>,
}

impl Cluster {
    async fn start(scenario: &'static str) -> Self {
        let server = PathBuf::from(env::var("CS_E2E_SERVER_BIN").expect("CS_E2E_SERVER_BIN"));
        assert!(
            server.is_file(),
            "missing server binary: {}",
            server.display()
        );
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("redis")).unwrap();
        let [redis_port, a_port, b_port] = ports();
        let log = File::create(root.path().join("redis.log")).unwrap();
        let redis = Command::new("redis-server")
            .args([
                "--bind",
                "127.0.0.1",
                "--port",
                &redis_port.to_string(),
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
            ])
            .arg(root.path().join("redis"))
            .stdout(Stdio::from(log.try_clone().unwrap()))
            .stderr(Stdio::from(log))
            .spawn()
            .unwrap();
        wait_for_redis(redis_port);
        let mut cluster = Self {
            scenario,
            root,
            server,
            redis_port,
            a_port,
            b_port,
            redis,
            a: None,
            b: None,
        };
        cluster.a = Some(cluster.start_node("node-a").await);
        cluster.b = Some(cluster.start_node("node-b").await);
        cluster.log("cluster_ready");
        cluster
    }

    fn log(&self, event: &str) {
        eprintln!(
            "e2e scenario={} event={} redis=127.0.0.1:{} node_a={} node_b={} root={}",
            self.scenario,
            event,
            self.redis_port,
            self.endpoint("node-a"),
            self.endpoint("node-b"),
            self.root.path().display(),
        );
    }

    fn endpoint(&self, node: &str) -> String {
        format!(
            "127.0.0.1:{}",
            if node == "node-a" {
                self.a_port
            } else {
                self.b_port
            }
        )
    }

    fn config(&self, node: &str) -> PathBuf {
        let port = if node == "node-a" {
            self.a_port
        } else {
            self.b_port
        };
        let dir = self.root.path().join(node);
        let nvme0 = dir.join("nvme0");
        let nvme1 = dir.join("nvme1");
        fs::create_dir_all(&nvme0).unwrap();
        fs::create_dir_all(&nvme1).unwrap();
        let path = dir.join("server.toml");
        fs::write(
            &path,
            format!(
                r#"[api]
listen = "127.0.0.1:{port}"
max_connections = 100
[storage]
devices = ["{}", "{}"]
data_subdir = "data"
striping_threshold = 1048576
striping_chunk_size = 1048576
[memory_tier]
capacity_mb = 0
slab_size_mb = 1
use_pinned_memory = false
[io_executor]
kind = "tier_a"
thread_pool_size = 4
io_uring_depth = 16
[router]
strategy = "object_hash"
[metadata]
redis_url = "redis://127.0.0.1:{}/0"
redis_key_prefix = "contextstore:e2e:"
redis_connect_timeout_ms = 1000
redis_command_timeout_ms = 1000
[metrics]
enabled = false
listen = "127.0.0.1:0"
[cluster]
node_id = "{node}"
grpc_advertise = "127.0.0.1:{port}"
rdma_advertise = ""
[[cluster.data_nodes]]
node_id = "node-a"
grpc_endpoint = "127.0.0.1:{}"
rdma_endpoint = ""
[[cluster.data_nodes]]
node_id = "node-b"
grpc_endpoint = "127.0.0.1:{}"
rdma_endpoint = ""
"#,
                nvme0.display(),
                nvme1.display(),
                self.redis_port,
                self.a_port,
                self.b_port
            ),
        )
        .unwrap();
        path
    }

    async fn start_node(&self, node: &str) -> Child {
        let config = self.config(node);
        let log_path = self.root.path().join(format!("{node}.log"));
        let log = File::create(&log_path).unwrap();
        let mut child = Command::new(&self.server)
            .args(["--config", config.to_str().unwrap(), "--log-level", "warn"])
            .env("CS_RDMA_DISABLED", "1")
            .stdout(Stdio::from(log.try_clone().unwrap()))
            .stderr(Stdio::from(log))
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            assert!(
                child.try_wait().unwrap().is_none(),
                "{} exited: {}",
                node,
                fs::read_to_string(&log_path).unwrap_or_default()
            );
            if let Ok(mut client) =
                KvClient::connect(format!("http://{}", self.endpoint(node))).await
            {
                if matches!(client.health().await, Ok(true)) {
                    return child;
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        panic!(
            "{} did not become healthy: {}",
            node,
            fs::read_to_string(log_path).unwrap_or_default()
        );
    }

    async fn restart_b(&mut self) {
        self.log("node_b_restart_begin");
        stop(self.b.as_mut().unwrap());
        self.b = Some(self.start_node("node-b").await);
        self.log("node_b_restart_complete");
    }

    fn file_count(&self, node: &str, device: usize) -> usize {
        fn count(path: &Path) -> usize {
            fs::read_dir(path)
                .map(|entries| {
                    entries
                        .filter_map(Result::ok)
                        .map(|entry| {
                            let path = entry.path();
                            if path.is_dir() {
                                count(&path)
                            } else {
                                usize::from(path.extension().is_some_and(|ext| ext == "bin"))
                            }
                        })
                        .sum()
                })
                .unwrap_or(0)
        }
        count(
            &self
                .root
                .path()
                .join(node)
                .join(format!("nvme{device}"))
                .join("data"),
        )
    }

    fn diagnostics(&self) -> String {
        let read_log = |name: &str| {
            fs::read_to_string(self.root.path().join(name))
                .unwrap_or_else(|err| format!("<failed to read {name}: {err}>"))
        };
        format!(
            "e2e scenario={} root={}\n--- redis.log ---\n{}\n--- node-a.log ---\n{}\n--- node-b.log ---\n{}",
            self.scenario,
            self.root.path().display(),
            read_log("redis.log"),
            read_log("node-a.log"),
            read_log("node-b.log"),
        )
    }
}

impl Drop for Cluster {
    fn drop(&mut self) {
        if let Some(child) = self.a.as_mut() {
            stop(child);
        }
        if let Some(child) = self.b.as_mut() {
            stop(child);
        }
        stop(&mut self.redis);
        if std::thread::panicking() {
            eprintln!("{}", self.diagnostics());
        }
    }
}

fn striped_payload() -> Vec<u8> {
    (0..STRIPED_OBJECT_BYTES)
        .map(|index| (index % 251) as u8)
        .collect()
}

async fn connect(cluster: &Cluster, node: &str) -> KvClient {
    cluster.log(&format!("{node}_client_connect"));
    KvClient::connect(format!("http://{}", cluster.endpoint(node)))
        .await
        .unwrap_or_else(|err| panic!("connect {node}: {err}"))
}

async fn write_striped_object(cluster: &Cluster, client: &mut KvClient, payload: &[u8]) {
    cluster.log("streaming_put_begin");
    let inserted = client
        .put_stream(
            TEST_NAMESPACE,
            TEST_OBJECT_KEY,
            payload.to_vec(),
            STREAM_CHUNK_BYTES,
        )
        .await
        .expect("streaming put");
    assert!(inserted, "streaming put must insert a new object");
    cluster.log("streaming_put_complete");
}

async fn assert_four_way_placement(cluster: &Cluster, client: &mut KvClient) {
    cluster.log("placement_lookup_begin");
    let lookup = client
        .lookup_object(TEST_NAMESPACE, TEST_OBJECT_KEY)
        .await
        .expect("lookup object")
        .expect("object metadata must exist after put");
    assert!(lookup.descriptor.is_striped, "object must use striped placement");
    assert_eq!(lookup.descriptor.stripe_count, 4, "expected four stripes");
    let locations: HashSet<(String, u32)> = lookup
        .placement
        .expect("placement must be materialized")
        .chunks
        .into_iter()
        .map(|chunk| (chunk.node_id, chunk.device_id))
        .collect();
    assert_eq!(
        locations,
        HashSet::from([
            ("node-a".into(), 0),
            ("node-a".into(), 1),
            ("node-b".into(), 0),
            ("node-b".into(), 1),
        ]),
        "stripes must cover both devices on both nodes",
    );
    for node in ["node-a", "node-b"] {
        for device in 0..2 {
            assert_eq!(
                cluster.file_count(node, device),
                1,
                "expected one stripe on {node}/nvme{device}",
            );
        }
    }
    cluster.log("placement_verified");
}

#[tokio::test]
async fn two_node_streaming_put_places_stripes_on_all_devices() {
    let cluster = Cluster::start("four_way_placement").await;
    let payload = striped_payload();
    let mut client = connect(&cluster, "node-a").await;

    write_striped_object(&cluster, &mut client, &payload).await;
    assert_four_way_placement(&cluster, &mut client).await;
}

#[tokio::test]
async fn two_node_cross_node_get_returns_original_payload() {
    let cluster = Cluster::start("cross_node_get").await;
    let payload = striped_payload();
    let mut writer = connect(&cluster, "node-a").await;
    write_striped_object(&cluster, &mut writer, &payload).await;

    let mut reader = connect(&cluster, "node-b").await;
    cluster.log("cross_node_get_begin");
    let read = reader
        .get_stream(TEST_NAMESPACE, TEST_OBJECT_KEY)
        .await
        .expect("cross-node get");
    assert_eq!(read, Some(payload));
    cluster.log("cross_node_get_verified");
}

#[tokio::test]
async fn two_node_restart_recovers_shared_metadata_and_stripes() {
    let mut cluster = Cluster::start("node_restart_recovery").await;
    let payload = striped_payload();
    let mut client = connect(&cluster, "node-a").await;

    write_striped_object(&cluster, &mut client, &payload).await;
    assert_four_way_placement(&cluster, &mut client).await;
    cluster.restart_b().await;
    cluster.log("post_restart_get_begin");
    assert_eq!(
        client
            .get_stream(TEST_NAMESPACE, TEST_OBJECT_KEY)
            .await
            .expect("get after node-b restart"),
        Some(payload),
        "shared metadata and remote stripes must survive node restart",
    );
    cluster.log("post_restart_get_verified");
}

#[tokio::test]
async fn two_node_distributed_delete_removes_all_stripe_files() {
    let cluster = Cluster::start("distributed_delete").await;
    let payload = striped_payload();
    let mut client = connect(&cluster, "node-a").await;

    write_striped_object(&cluster, &mut client, &payload).await;
    assert_four_way_placement(&cluster, &mut client).await;
    cluster.log("distributed_delete_begin");
    assert!(
        client
            .delete(TEST_NAMESPACE, TEST_OBJECT_KEY)
            .await
            .expect("distributed delete"),
        "delete must remove the existing object",
    );
    for node in ["node-a", "node-b"] {
        for device in 0..2 {
            assert_eq!(
                cluster.file_count(node, device),
                0,
                "distributed delete must remove stripe from {node}/nvme{device}",
            );
        }
    }
    cluster.log("distributed_delete_verified");
}
