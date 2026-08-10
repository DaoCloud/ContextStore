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
    async fn start() -> Self {
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
        cluster
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
        stop(self.b.as_mut().unwrap());
        self.b = Some(self.start_node("node-b").await);
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
    }
}

#[tokio::test]
async fn two_node_streaming_placement_restart_and_delete() {
    let mut cluster = Cluster::start().await;
    let payload: Vec<u8> = (0..4 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    let mut a = KvClient::connect(format!("http://{}", cluster.endpoint("node-a")))
        .await
        .unwrap();
    assert!(a
        .put_stream("e2e", "striped-object", payload.clone(), 128 * 1024)
        .await
        .unwrap());
    let lookup = a
        .lookup_object("e2e", "striped-object")
        .await
        .unwrap()
        .unwrap();
    assert!(lookup.descriptor.is_striped);
    assert_eq!(lookup.descriptor.stripe_count, 4);
    let locations: HashSet<(String, u32)> = lookup
        .placement
        .unwrap()
        .chunks
        .into_iter()
        .map(|c| (c.node_id, c.device_id))
        .collect();
    assert_eq!(
        locations,
        HashSet::from([
            ("node-a".into(), 0),
            ("node-a".into(), 1),
            ("node-b".into(), 0),
            ("node-b".into(), 1)
        ])
    );
    for node in ["node-a", "node-b"] {
        for device in 0..2 {
            assert_eq!(cluster.file_count(node, device), 1);
        }
    }
    let mut b = KvClient::connect(format!("http://{}", cluster.endpoint("node-b")))
        .await
        .unwrap();
    assert_eq!(
        b.get_stream("e2e", "striped-object").await.unwrap(),
        Some(payload.clone())
    );
    cluster.restart_b().await;
    assert_eq!(
        a.get_stream("e2e", "striped-object").await.unwrap(),
        Some(payload)
    );
    assert!(a.delete("e2e", "striped-object").await.unwrap());
    for node in ["node-a", "node-b"] {
        for device in 0..2 {
            assert_eq!(cluster.file_count(node, device), 0);
        }
    }
}
