//! `softroce_dual_rail` — 在 WSL2 的两条 Soft-RoCE 路径上接入赛题多轨读取。
//!
//! 编译/运行（需先按 `wsl2-softroce-setup.md` 在 WSL2 拉起 rxe0/rxe1，并部署
//! 一个在两条 rxe 上各监听一个 RDMA 端点的 ContextStore server）：
//! ```text
//! cargo run --features rdma --example softroce_dual_rail
//! ```
//!
//! 本例演示**接线**：把两条独立 `RdmaClient`（各绑 rxe0 / rxe1，各连一个
//! server 端点）塞进 `RailManager`，即构成"多轨并行读取"的 rail 集合。
//! 真实读路径复用 `rdma.rs` 已有的 `get_descriptor_stripes_sge`（wire tag 15，
//! RDMA WRITE 直写 caller 分散缓冲），read 完成信号走 TCP `GET_RESP`。
//!
//! 前提（赛题后续项）：server 端需支持"多 RDMA 监听"——即同一对象的不同
//! stripe 子集分布在两个以不同 rxe 为下层的 QP 上。当前仓库 server 默认单
//! 监听；本例的 `endpoints` 因此指向两个端口，待 server 改造后即可端到端跑通。

use contextstore_client_rs::multi_rail::{RailManager, RailReader};
use contextstore_client_rs::pb;
use contextstore_client_rs::rdma::RdmaClient;

fn main() {
    // 两条独立 Soft-RoCE 路径：rxe0 挂在 veth0、rxe1 挂在 veth1（见 setup-wsl2-rxe.sh）。
    // 每个 RdmaClient 连接一个 ContextStore 数据节点端点；该端点在其对应 rxe 上建 QP。
    let endpoints: [&str; 2] = ["127.0.0.1:50051", "127.0.0.1:50052"];
    let devices: [&str; 2] = ["rxe0", "rxe1"];

    // 把每条真实 RDMA 路径包成一个 `Box<dyn RailReader>`，交给 RailManager。
    let mut rails: Vec<Box<dyn RailReader>> = Vec::with_capacity(2);
    for i in 0..2 {
        rails.push(
            Box::new(RdmaClient::new(endpoints[i], devices[i])) as Box<dyn RailReader>,
        );
    }

    let manager = RailManager::new(rails);
    println!(
        "[dual-rail] 已用 {} 条独立 RDMA 路径（rxe0/rxe1）组装多轨读取器",
        manager.rail_count()
    );
    if manager.is_single() {
        println!("[dual-rail] 警告：只有 1 条路径，未构成多轨。请检查 rxe 设备是否拉起。");
    }

    // ---- 真实读模板（取消注释并填入 descriptor / buffer / checksums 即可跑）----
    //
    // let lookup = client.lookup_object(NS, KEY).await?;  // 从 gRPC 控制面取 descriptor+placement
    // let descriptor = lookup.descriptor;
    // let mut buf = vec![0u8; descriptor.size as usize];
    // let checksums: Vec<Option<String>> = descriptor_derived_checksums(&lookup.placement);
    // let mut reader = manager.reader();
    // let stats = reader.read(&descriptor, &checksums, &mut buf)?;
    // manager.reclaim(reader);
    // println!("[dual-rail] 读完成: {} bytes, 瓶颈={}", stats.total_bytes, stats.bottleneck());

    println!(
        "[dual-rail] 说明：端到端读需 server 支持多 RDMA 监听（后续项）。\n\
         \t当前可先在 WSL2 用 `ib_send_bw -d rxe0 ... & ib_send_bw -d rxe1 ... &`\n\
         \t并行测两条 rxe 的带宽，叠加即多轨聚合的真 verbs 证据（见 setup 手册第 5 步）。"
    );
}

// 让 pb 被引用，避免未使用告警（真实读时取消注释本函数与调用）。
#[allow(dead_code)]
fn _unused(_d: pb::ObjectDescriptor) {}
