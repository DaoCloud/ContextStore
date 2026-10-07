# Multi-Rail Parallel Read — Design Document

> Status: Draft (tracking issue: DaoCloud/ContextStore#33, draft PR: #34)
> Scope: `kv-service/client-rs` multi-rail read layer. Server changes: none
> required for v1 (the server already multi-listens via `CS_RDMA_DEVICES`).

## 1. Background and problem statement

KVService stripes large objects across multiple NVMe devices / nodes, so
aggregate disk bandwidth scales with stripe count. A single client worker,
however, reads through ONE RDMA path (one NIC / QP). Once
`stripe_count × disk_bw > single-NIC bandwidth`, the network becomes the
bottleneck while the disks still have headroom — adding disks or concurrency
on the storage side no longer shortens read time.

**Multi-rail** lets one client worker transfer disjoint stripe subsets of the
same object over several independent RDMA paths concurrently, aggregating NIC
bandwidth.

## 2. Goals and non-goals

### Goals (v1)

- Transport-agnostic rail abstraction (`RailReader`) with two
  implementations: real verbs (`RdmaClient`) and a fault-injecting mock.
- Stripe→rail scheduling, cross-rail completion aggregation, per-stripe
  integrity verification, bottleneck attribution.
- Memory-consistency guards against the late-RDMA-WRITE hazard.
- Failure semantics, resource governance, and observability as first-class
  design elements (not afterthoughts).
- Full functional validation without RDMA hardware; Soft-RoCE end-to-end
  validation on commodity Ethernet (WSL2).

### Non-goals (v1)

- No change to on-disk stripe layout or placement policy.
- No change to the upper read interface (`LookupObject` / `ReadByDescriptor`).
- No transparent in-request retry (a failed request fails safely as a whole).
- No GDS / GPU path changes.
- No claim of physical multi-NIC aggregate bandwidth from Mock/Soft-RoCE data.

## 3. Why this is a generalization, not a rewrite

The codebase already provides the hard parts:

- `RdmaClient.get_descriptor_stripes_sge` (wire tag 15) lets the server
  RDMA-WRITE stripe subsets straight into scattered caller buffers, and is
  already used for *per-node* concurrency.
- Read completion is signalled over the TCP control channel (`GET_RESP`), so
  N rails = N tasks + join; no server-side polling changes.
- Consistency fields are on the wire: `object_generation` in
  `ObjectDescriptor`, `chunk_checksums` (xxh3-64) in `PlacementChunk` /
  `StripingInfo`.
- The server already listens on multiple RDMA devices
  (`CS_RDMA_DEVICES=dev0:host:port[:gid],dev1:...`), one QP group per NIC.

Multi-rail = generalize "concurrent per node" to "concurrent per rail
(independent NIC/QP)", plus path lifecycle management, completion
aggregation, and memory-consistency guards.

## 4. Architecture

```
Worker (vLLM / Dynamo / NIXL)
   │  upper interface unchanged: LookupObject / ReadByDescriptor
   ▼
RailManager  ── N independent rails (NIC + QP + CQ); single NIC = 1 rail
   ▼
Stripe→Rail scheduler ── locality affinity + round-robin (least-loaded later)
   ▼
per rail, concurrently: get_descriptor_stripes_sge(descriptor, stripe subset, SGE list)
   │  each rail writes disjoint regions of the same destination buffer
   ▼
Completion aggregation + integrity layer
   (object_generation check · per-stripe xxh3-64 · epoch buffer guard)
   ▼
Bottleneck attribution / observability (per-rail timing, classify
   NIC / PCIe / memory / disk / software)
```

Modules (all in `kv-service/client-rs`):

| File | Responsibility |
|---|---|
| `src/multi_rail.rs` | `RailReader` trait, `RailManager`, `MultiRailReader`, `plan_stripes`, `ReadOptions`, `RailReadStats` + bottleneck classifier, `stripe_checksum` (xxh3-64) |
| `src/mock_rail.rs` | Mock transport: configurable per-rail bandwidth; fault injection (disconnect, stall, under-delivery, corrupt stripe, late-write-after-cancel) |
| `src/rdma.rs` | `RailReader for RdmaClient`: cached-MR registration, `get_descriptor_stripes_sge` reads |
| `src/bin/cs_mock_bench.rs` | Rail-count sweep capacity model with bottleneck attribution |
| `tests/multi_rail_mock.rs` | 11 hardware-free functional/robustness tests |
| `examples/softroce_dual_rail.rs` | Soft-RoCE end-to-end: PUT → lookup → single- vs dual-rail read → verification + speedup report |
| `kv-service/configs/server-wsl2-softroce.toml` | Single-machine demo config (4 MiB striping threshold so demo objects actually stripe) |

### 4.1 Path model

A **rail** is the tuple
`(rail_id, local RDMA device + port + GID, remote endpoint, QP/CQ, PD + MRs)`:

- The *remote endpoint* is one of the server's per-NIC RDMA control listeners
  (`CS_RDMA_DEVICES=dev:host:port[:gid],...`), so each rail terminates on a
  distinct server-side NIC/QP group.
- The *local device* is the client NIC whose fabric path reaches that
  endpoint. In the single-machine Soft-RoCE demo both sides share the rxe
  pair (client rxe0 ↔ server rxe0, rxe1 ↔ rxe1 over two veth subnets); in a
  multi-NIC deployment the operator pairs each server listener with the
  client NIC on the same subnet.
- **Discovery v1 is static config**: rails are assembled from explicit
  endpoint+device pairs (`CS_RAIL_ENDPOINTS` / `CS_RAIL_DEVICES` in the
  example). Rails are individually identifiable (`rail_id`), individually
  configurable, and individually fault-injectable (mock `MockFault` knobs).
  Independent runtime start/stop and a `down`/`draining` state machine are
  planned work (§13); the `RailReader` contract already isolates per-rail
  state so this is additive.

### 4.2 Connection and memory lifecycle

- **QP/CQ**: each rail's QP is established against its endpoint on first use
  and reused for the rail's lifetime. Completion is observed through the
  existing `GET_RESP` TCP control signal — no new polling loops, no CQ
  threading changes.
- **Memory regions**: the destination buffer is registered per rail through
  the cached-MR path at read start, pinned until join, and deregistered only
  after every rail's completion has been accounted for.
- **WR/CQE matching**: one striped GET (wire tag 15) per rail per object;
  completions are matched to their request, and a completion arriving after
  cancel is dropped by the epoch guard (§6).
- **Buffer**: the caller's buffer is never treated as read output until
  aggregation and verification succeed; on timeout it is provably untouched
  (test 8 in §11).

## 5. Stripe→rail scheduling

`plan_stripes(total_size, chunk_size, stripe_count, rail_count, locality)`:

- `locality[i]` pins stripe `i` to a preferred rail (e.g. the rail whose NIC
  reaches the node owning the stripe); `None` falls back to round-robin.
- Round-robin is deterministic and stripe-quantization-aware: with `S`
  stripes and `R` rails the slowest rail carries `ceil(S/R)` stripes, which
  the capacity model reports separately (`bal` vs `theo` columns).
- A dynamic least-loaded scheduler can be layered on later without changing
  the `RailReader` contract (it only changes `locality`).

## 6. Memory safety: the late-RDMA-WRITE hazard

If a cancelled request's buffer is freed and reused, a late-arriving RDMA
WRITE can corrupt the new owner's data. Three defences, all covered by tests:

1. **pin-until-completion** — the read holds the destination buffer and every
   rail's registration until join; cancel flips a logical flag only, MRs are
   never deregistered early.
2. **epoch buffer guard** — late writes land only while `live == epoch`;
   bumping `live` on cancel makes the guard skip them. The A/B test
   demonstrates corruption without the guard and suppression with it.
3. **two-phase commit + per-stripe checksum** — stripes are verified against
   expected xxh3-64 digests after aggregation; a cancelled request never
   reaches commit, and the caller's buffer is only written after a clean,
   verified success (timeout path leaves it untouched — asserted by test).

## 7. Failure semantics (v1)

Single, simple rule: **if any required rail fails, the whole request fails
safely** — no partial data returned, no transparent in-request retry.

| Scenario | Behavior | Test |
|---|---|---|
| Stripe checksum mismatch after aggregation | whole read errors | `test_stripe_checksum_failure_detected` |
| Single rail disconnects mid-read | rail error surfaces; whole read fails; no partial data | `test_single_rail_disconnect_safe_fail` |
| Rail under-delivers (partial completion) | missing bytes detected via checksum layer; read fails | `test_partial_completion_under_delivery_detected` |
| Read exceeds `ReadOptions.timeout` | error naming the timeout; caller's buffer untouched | `test_single_rail_timeout_safe_fail` / `test_read_with_timeout_succeeds_when_fast` |
| Object version changes mid-read (`object_generation`) | read rejected instead of returning torn data | `test_version_change_rejected` |
| Late write after cancel | epoch guard drops it | `epoch_guard_blocks_late_write_after_cancel` |

Later versions may add per-stripe retry on surviving rails; the wire format
and invariants above do not preclude it.

## 8. Resource governance and backpressure

- **In-flight budget**: `ReadOptions.max_inflight_bytes` rejects oversized
  reads up front (backpressure) instead of queueing unbounded work —
  `test_resource_limit_rejects_oversized`.
- **Registration reuse**: `RdmaClient` registrations go through the cached-MR
  path (`register_raw_buffer_cached`), bounding MR churn per rail.
- **Thread model**: one scoped thread per rail per read; threads join before
  the read returns, so concurrency is bounded by rail count.
- Server-side knobs (`CS_RDMA_SLAB_MB`, per-device listeners) are shared
  across rails by design: each PD does its own `reg_mr` against the same
  slab.

## 9. Compatibility strategy

Hard invariants:

1. On-disk stripe layout (`StripingInfo`) untouched.
2. Upper read interface (`LookupObject` / `ReadByDescriptor`) untouched.
3. **Single-NIC deployments are a pass-through**: one rail → the scheduler
   degenerates to the existing sequential path; covered by
   `single_rail_is_pass_through` (byte-exact).
4. Default features build and test without RDMA hardware (`rdma` feature
   gates the verbs path only).
5. Rail set is assembled from explicit endpoint+device pairs
   (`CS_RAIL_ENDPOINTS` / `CS_RAIL_DEVICES` in the example); a single entry
   reproduces today's behavior.

## 10. Observability and bottleneck attribution

`RailReadStats` per object: `rail_count`, `total_bytes`, transfer-phase wall
time (`object_ms`), scheduling time, per-rail times, verification time,
aggregate goodput (`transfer_mbps`), `verify_ok`.

`bottleneck()` classifies each read into one of: client scheduling, slowest
rail (network / single-NIC cap), post-read verification (client CPU), or
storage behind the rails. This makes the multi-rail transition measurable:
once the NIC cap is removed, the *next* ceiling becomes visible in the same
units.

## 11. Test plan and current status

`cargo test -p contextstore-client-rs --test multi_rail_mock` — 11/11 pass
(default features, no RDMA hardware):

1. `multi_rail_read_aggregates_stripes` — byte-exact reconstruction
2. `single_rail_is_pass_through` — compatibility
3. `plan_stripes_round_robin` — scheduling
4. `epoch_guard_blocks_late_write_after_cancel` — memory safety (A/B)
5. `test_stripe_checksum_failure_detected` — integrity
6. `test_single_rail_disconnect_safe_fail` — fault
7. `test_partial_completion_under_delivery_detected` — partial completion
8. `test_single_rail_timeout_safe_fail` — timeout, buffer untouched
9. `test_read_with_timeout_succeeds_when_fast` — timeout non-trigger
10. `test_resource_limit_rejects_oversized` — backpressure
11. `test_version_change_rejected` — consistency

Upstream e2e (`make e2e`) remains green — no upstream interface was modified.

## 12. Performance evidence (capacity model, labelled)

`cs-mock-bench`, measured on WSL2: 64 MiB object, 4 MiB chunks, 16 stripes,
125 MB/s per rail:

```
rails  agg_MB/s  bal_MB/s  theo_MB/s  speedup  bottleneck
1      123.5     125.0     125.0      0.99x    network: slowest rail caps the object
2      244.9     250.0     250.0      1.96x    software: post-read stripe verification (client CPU)
3      327.4     333.3     375.0      2.62x    software: post-read stripe verification
4      484.1     500.0     500.0      3.87x    software: post-read stripe verification
```

Reading: near-linear aggregate scaling (3.87x at 4 rails); the 3-rail dip is
stripe quantization (`ceil(16/3)=6` stripes on the slowest rail, see `bal`);
once the NIC cap is lifted, the classifier already flags the next ceiling —
client-side checksum CPU — which motivates the xxh3 alignment (done) and any
future SIMD/parallel verify.

**Environment honesty.** Functional validation uses the Mock transport and
Soft-RoCE (RXE) on WSL2. These prove functionality, failure semantics,
scheduling and resource governance. They do NOT prove hardware aggregate
bandwidth, HCA offload, PCIe/NUMA effects, or zero-copy behavior on physical
NICs. Every number is labelled with its environment. No GPU is involved.

## 13. Known limitations and risks

| Limitation | Impact | Mitigation / plan |
|---|---|---|
| No physical multi-NIC testbed | Hardware aggregate bandwidth unproven | Labelled Soft-RoCE evidence; design keeps NIC count a config knob |
| v1 has no in-request retry | A rail failure fails the whole read | Documented semantics; retry is additive later |
| Round-robin ignores transient rail load | Skewed rails can straggle | `locality` hook + least-loaded scheduler planned |
| Checksum verification is single-threaded client CPU | Becomes the next bottleneck at ≥2 rails (measured) | xxh3 aligned with server; parallel/SIMD verify is future work |
| RXE GID index varies by setup | Wrong index = connect failure | `CS_RAIL_GID` knob; `show_gids` documented |
| Rails are configured, not yet independently start/stop-able at runtime | Rail lifecycle ops limited | Rail state machine (available/down/draining) planned |

## 14. Reproduction

```bash
# functional suite (any machine, no RDMA hardware)
cargo test -p contextstore-client-rs --test multi_rail_mock

# capacity model
cargo run -p contextstore-client-rs --bin cs-mock-bench

# Soft-RoCE end-to-end (WSL2, see wsl2-softroce-setup.md):
#   rxe0/rxe1 up, server with
#   CS_RDMA_DEVICES=rxe0:0.0.0.0:50053:1,rxe1:0.0.0.0:50054:1 \
#     contextstore-server --config configs/server-wsl2-softroce.toml
cargo run -p contextstore-client-rs --features rdma --example softroce_dual_rail
```
