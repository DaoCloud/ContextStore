//! `softroce_dual_rail` — wires the contest multi-rail read onto two Soft-RoCE
//! paths inside WSL2.
//!
//! Build/run (first bring up rxe0/rxe1 in WSL2 per `wsl2-softroce-setup.md` and
//! deploy a ContextStore server that listens on one RDMA endpoint per rxe):
//! ```text
//! cargo run --features rdma --example softroce_dual_rail
//! ```
//!
//! This example shows the **wiring**: two independent `RdmaClient`s (one bound
//! to rxe0, one to rxe1, each connected to a server endpoint) are dropped into
//! `RailManager`, forming the rail set for "multi-rail parallel read". The real
//! read path reuses `rdma.rs`'s existing `get_descriptor_stripes_sge` (wire tag
//! 15, RDMA WRITE straight into the caller's scattered buffer); read completion
//! is signaled over TCP `GET_RESP`.
//!
//! Prerequisite (server-side follow-up): the server must support "multi-RDMA
//! listen" — i.e. different stripe subsets of the same object land on QPs that
//! sit on different rxe devices. The current in-repo server listens on a single
//! device by default; this example's `endpoints` therefore point at two ports
//! and will run end-to-end once the server is enhanced.

use contextstore_client_rs::multi_rail::{RailManager, RailReader};
use contextstore_client_rs::pb;
use contextstore_client_rs::rdma::RdmaClient;

fn main() {
    // Two independent Soft-RoCE paths: rxe0 on veth0, rxe1 on veth1 (see setup-wsl2-rxe.sh).
    // Each RdmaClient connects to one ContextStore data-node endpoint; that endpoint
    // builds its QP on the corresponding rxe.
    let endpoints: [&str; 2] = ["127.0.0.1:50051", "127.0.0.1:50052"];
    let devices: [&str; 2] = ["rxe0", "rxe1"];

    // Wrap each real RDMA path as a `Box<dyn RailReader>` and hand it to RailManager.
    let mut rails: Vec<Box<dyn RailReader>> = Vec::with_capacity(2);
    for i in 0..2 {
        rails.push(Box::new(RdmaClient::new(endpoints[i], devices[i])) as Box<dyn RailReader>);
    }

    let manager = RailManager::new(rails);
    println!(
        "[dual-rail] assembled multi-rail reader from {} independent RDMA paths (rxe0/rxe1)",
        manager.rail_count()
    );
    if manager.is_single() {
        println!(
            "[dual-rail] warning: only 1 path present, multi-rail not formed. Check that rxe devices are up."
        );
    }

    // ---- Real read template (uncomment and fill descriptor / buffer / checksums to run) ----
    //
    // let lookup = client.lookup_object(NS, KEY).await?;  // fetch descriptor+placement from gRPC control plane
    // let descriptor = lookup.descriptor;
    // let mut buf = vec![0u8; descriptor.size as usize];
    // let checksums: Vec<Option<String>> = descriptor_derived_checksums(&lookup.placement);
    // let mut reader = manager.reader();
    // let stats = reader.read(&descriptor, &checksums, &mut buf)?;
    // manager.reclaim(reader);
    // println!("[dual-rail] read done: {} bytes, bottleneck={}", stats.total_bytes, stats.bottleneck());

    println!(
        "[dual-rail] note: end-to-end read needs the server to support multi-RDMA listen (server-side follow-up).\n\
         \tFor now, on WSL2 run `ib_send_bw -d rxe0 ... & ib_send_bw -d rxe1 ... &`\n\
         \tto measure both rxe bandwidths in parallel; summing them is the real-verbs proof of multi-rail aggregation (see setup manual step 5)."
    );
}

// Keep `pb` referenced to avoid an unused-import warning (used in the real read template above).
#[allow(dead_code)]
fn _unused(_d: pb::ObjectDescriptor) {}
