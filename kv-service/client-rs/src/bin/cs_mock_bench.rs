//! `cs-mock-bench` — 多轨并行读取的「带宽容量模型」基准（**无需 RDMA 硬件**）。
//!
//! 用 `MockRailClient`（每条 rail 可配置带宽）喂给 `MultiRailReader`，
//! 扫描 rail 数 1..=N，输出「N 轨聚合带宽 vs 单轨」曲线与瓶颈归因。
//!
//! 这正是答辩最硬的素材：它用**可配置每轨带宽的容量模型**证明
//! "多轨把单 NIC 带宽上限叠加上去"，而题面承认 Soft-RoCE/Mock 只能证
//! 功能与失败语义、不能证硬件聚合带宽——所以这里诚实地把它标为 model。
//!
//! 运行（任意装了 Rust 的机器，含 Windows 原生）：
//! ```text
//! cargo run --bin cs-mock-bench
//! OBJ_SIZE=$((128*1024*1024)) cargo run --bin cs-mock-bench   # 128 MiB 对象
//! ```
//! 默认 feature，不依赖 libibverbs / GPU。

use contextstore_client_rs::mock_rail::{MockRailClient, MockStore};
use contextstore_client_rs::multi_rail::{RailManager, RailReader, RailReadStats};
use contextstore_client_rs::pb;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const NS: &str = "bench";
const KEY: &str = "multi-rail-object";

/// 必须与 `mock_rail::canonical_key` 保持一致的对象键格式。
fn canon(ns: &str, key: &str) -> String {
    format!("{}:{}{}", ns.len(), ns, key)
}

fn main() {
    // ---- 可调参数（环境变量覆盖）----
    let object_size: u64 = std::env::var("OBJ_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64 * 1024 * 1024); // 64 MiB
    let chunk_size: u64 = 4 * 1024 * 1024; // 4 MiB / stripe
    let max_rails: usize = std::env::var("MAX_RAILS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    // 每轨带宽：模拟一块 Soft-RoCE 软网卡的吞吐上限（这里取 1 Gbps ≈ 125 MB/s）。
    // 把它调小/调大，曲线斜率会随之变化——这就是"容量模型"的旋钮。
    let per_rail_bps: f64 = std::env::var("PER_RAIL_MBPS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|mbps| mbps * 1024.0 * 1024.0)
        .unwrap_or(125.0 * 1024.0 * 1024.0);

    let stripe_count = ((object_size + chunk_size - 1) / chunk_size) as u32;

    // ---- 构造 Mock 对象（共享 store，各 rail 读同一份）----
    let data = vec![0xABu8; object_size as usize];
    let store = Arc::new(Mutex::new(MockStore::default()));
    store
        .lock()
        .unwrap()
        .objects
        .insert(canon(NS, KEY), data.clone());

    let descriptor = pb::ObjectDescriptor {
        key: Some(pb::ObjectKey {
            namespace: NS.to_string(),
            object_key: KEY.to_string(),
        }),
        object_handle: String::new(),
        object_generation: 1,
        content_etag: String::new(),
        layout_version: 1,
        size: object_size,
        is_striped: true,
        stripe_count,
        chunk_size,
    };

    // 一致性层：预先算好每 stripe 的校验和，喂给 read()，聚合后逐 stripe 比对。
    let mut checksums: Vec<Option<String>> = Vec::with_capacity(stripe_count as usize);
    for i in 0..stripe_count as usize {
        let off = (i as u64) * chunk_size;
        let l = chunk_size.min(object_size - off) as usize;
        checksums.push(Some(contextstore_client_rs::multi_rail::stripe_checksum(
            &data[off as usize..off as usize + l],
        )));
    }

    println!("=== ContextStore 多轨并行读取 · Mock 带宽容量模型 ===");
    println!(
        "object={} MiB  chunk={} MiB  stripes={}  per_rail={:.0} MB/s",
        object_size / 1024 / 1024,
        chunk_size / 1024 / 1024,
        stripe_count,
        per_rail_bps / 1024.0 / 1024.0
    );
    println!(
        "{:<6} {:<12} {:<12} {:<12} {:<10}  {}",
        "rails", "agg_MB/s", "obj_ms", "theo_MB/s", "speedup", "bottleneck"
    );

    for r in 1..=max_rails {
        // 建 r 条 rail（每条 Mock 带宽相同），交给 RailManager。
        let mut rails: Vec<Box<dyn RailReader>> = Vec::with_capacity(r);
        for i in 0..r {
            rails.push(
                Box::new(MockRailClient::new(
                    format!("mock-{i}"),
                    per_rail_bps,
                    store.clone(),
                )) as Box<dyn RailReader>,
            );
        }
        let mut manager = RailManager::new(rails);
        let mut reader = manager.reader();
        let mut buf = vec![0u8; object_size as usize];

        let t0 = Instant::now();
        let stats: RailReadStats = reader
            .read(&descriptor, &checksums, &mut buf)
            .unwrap_or_else(|e| panic!("rail_count={r} read failed: {e}"));
        let elapsed = t0.elapsed().as_secs_f64();
        manager.reclaim(reader);

        let agg_mbps = (stats.total_bytes as f64) / elapsed / (1024.0 * 1024.0);
        let theo_mbps = per_rail_bps * (r as f64) / (1024.0 * 1024.0);
        let speedup = agg_mbps / (per_rail_bps / (1024.0 * 1024.0));
        println!(
            "{:<6} {:<12.1} {:<12.2} {:<12.1} {:<10.2}x  {}",
            r,
            agg_mbps,
            elapsed * 1000.0,
            theo_mbps,
            speedup,
            stats.bottleneck()
        );
    }

    println!(
        "\n观察：rails 从 1→{max_rails}，聚合带宽应近似线性增长（speedup→{max_rails}x），\n\
         直到撞上磁盘/JBOF 聚合带宽或 PCIe——`bottleneck()` 会告诉你瓶颈在哪。\n\
         注意：这是容量模型，非硬件实测；真实多轨聚合带宽需实体 RDMA 网卡（赛题已知限制）。"
    );
}
