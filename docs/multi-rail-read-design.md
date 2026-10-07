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
- **Registration reuse**: the default read path registers the destination
  buffer afresh per read and keeps the MR alive only across the
  `register` → `read_stripes` pair (held in `RdmaClient::in_flight_mr`),
  then deregisters it. An opt-in pooled path
  (`register_raw_buffer_pooled` + `invalidate_mr_cache`) is available for
  long-lived buffer pools — see §8.1 for why caching by virtual address is
  not safe as a default.
- **Thread model**: one scoped thread per rail per read; threads join before
  the read returns, so concurrency is bounded by rail count.
- Server-side knobs (`CS_RDMA_SLAB_MB`, per-device listeners) are shared
  across rails by design: each PD does its own `reg_mr` against the same
  slab.

### 8.1 Case study: MR caching vs. buffer lifetime

An early version of the read path registered destination buffers through a
cache keyed by `(base_ptr, length)` (`register_raw_buffer_cached`). It worked
for the end-to-end demo — every read there used a fresh buffer and a fresh
client — but failed once a single client performed **two consecutive reads
into a per-iteration `Vec`**: the second read returned an all-zero buffer
while reporting success.

Root cause. After the first `Vec` is dropped, the allocator frequently hands
the *same virtual address* to the next same-sized `Vec`. The cache hits on
`(addr, len)` and reuses the MR registered for the first `Vec`. That MR still
pins the **old physical pages**, but the CPU now reads through the **new**
virtual→physical mapping at the same address. The NIC's RDMA WRITE goes to
the old pages, so from the caller's view the buffer was never written. The
`# Safety` note asked callers to keep memory alive forever, but a `Vec` per
iteration silently violates that — and no runtime check can catch it, because
a raw pointer carries no lifetime.

Fix. The default path no longer caches: `RailReader::register` calls
`register_raw_buffer` and stores the owning `RegisteredBuffer` in
`in_flight_mr`, so the MR lives exactly from `register` to the next
`register` (i.e. across `read_stripes`) and is then deregistered. The pooled
path survives as `register_raw_buffer_pooled`, explicitly documented as
requiring buffer-pool semantics, with `invalidate_mr_cache` for callers that
recycle buffers at a reused address.

Evidence. `tests/multi_rail_mr_lifetime.rs` reproduces the failure shape
without RDMA hardware (per-iteration buffers, repeated reads) and asserts the
read path never observes a stale registration; the Soft-RoCE concurrency
sweep (`softroce_concurrency`, `CS_ITERS>1`) exercises the real verbs path.
The cost of always re-registering is ~1.5 ms per 56 MB — under 2 % of a
32 MiB read (~75 ms) — so correctness is bought cheaply.

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

### 9.1 Compatibility matrix

Cell values: **OK** = behaves exactly as before; **extended** = only adds a
new capability, existing behavior unchanged; **n/a** = not applicable.

| Client / deployment | Wire protocol | On-disk layout | Upper read API | Config surface | Multi-rail read |
|---|---|---|---|---|---|
| Existing gRPC-only client (no `rdma` feature) | OK (unchanged) | OK | OK | null | n/a (no RDMA path compiled) |
| Existing RDMA client, single rail / single NIC | OK (tag 15 reused as-is) | OK | OK | extended (endpoint list of 1) | pass-through (byte-exact) |
| Multi-rail client, single NIC configured | OK | OK | OK | extended | pass-through (1 rail) |
| Multi-rail client, N rails / N NICs | OK (no new wire tags) | OK | OK | extended (`CS_RAIL_*`) | new capability |
| Old server + new multi-rail client | OK (client requests the same tag-15 GET) | OK | OK | extended | degraded to 1 rail if the server advertises one endpoint |
| New server + old client | OK (server multi-listens; old client uses one listener) | OK | OK | OK | n/a (client has no multi-rail layer) |

**Upgrade path**: no data migration and no protocol version bump. Multi-rail
is a *client-side* capability layered on the existing tag-15 stripe-subset GET;
the server only needs to listen on more than one RDMA device
(`CS_RDMA_DEVICES=dev0:...,dev1:...`), which it already supports. Rolling
back = configure a single rail (or drop the `rdma` feature) — the read path is
then byte-identical to today's.

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

## 12. Performance evidence (labelled)

Three layers of evidence, each labelled with its environment. Together they
cover contest deliverable e (throughput, read time, extension trend, CPU /
memory / registered memory / inflight bytes) and f (bottleneck attribution).

### 12.1 Capacity model (`cs-mock-bench`, Mock, WSL2)

64 MiB object, 4 MiB chunks, 16 stripes, 125 MB/s per rail:

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

### 12.2 End-to-end single- vs dual-rail (Soft-RoCE, real verbs, WSL2)

`examples/softroce_dual_rail`, real xxh3-64 per-stripe verification plus byte
equality on every run:

```
object   single-rail      dual-rail        speedup  verify
64 MiB   73.8 MB/s        217.6 MB/s       2.95x    ok
128 MiB  93.3 MB/s        384.1 MB/s       4.11x    ok
```

Single runs; run-to-run variance on WSL2 is noticeable (single-rail 64 MiB
measured 73.8–110.1 MB/s across runs), so treat the speedup as indicative.

### 12.3 Concurrency and resource overhead sweep (`softroce_concurrency`)

Same object, same stripe layout, same server and client process across rows;
the sweep varies concurrency (independent workers, each with its own rails)
and reports aggregate goodput, read latency percentiles, and process resources.

Method note: `CS_ITERS=2` timed iterations per (size, mode, concurrency);
latency percentiles are computed over all worker iterations. The first read
of a fresh rail pays RDMA-CM connect + first-WR cost, which at small object
sizes dominates the p50 (the single-rail 32 MiB row is the clearest case:
296 ms p50 against 118 ms once concurrency hides it). Absolute throughput on
Soft-RoCE is CPU-bound and run-to-run noisy; the trends and the resource
accounting are the stable signals.

**Sweep 1 — object size × concurrency (32–256 MiB), all points `verify_ok`**

```
size   mode    conc  agg_MB/s  p50_ms   p95_ms   RSS_MiB  reg_MiB  infl_MiB  verify
32MiB  single  1      128.4     296.27   296.27      70       32       32     true
32MiB  dual    1      197.1     216.86   216.86      70       64       32     true
32MiB  single  2      489.8     118.06   135.88      70       64       64     true
32MiB  dual    2      490.0     118.21   134.82      70      128       64     true
64MiB  single  1      121.8     657.17   657.17     134       64       64     true
64MiB  dual    1      339.9     213.39   213.39     134      128       64     true
64MiB  single  2      498.6     235.07   325.97     134      128      128     true
64MiB  dual    2      492.9     236.35   267.82     134      256      128     true
128MiB single  1      111.1    1365.93  1365.93     262      128      128     true
128MiB dual    1      411.3     305.64   305.64     262      256      128     true
128MiB single  2      495.9     473.96   650.21     262      256      256     true
128MiB dual    2      495.2     471.93   534.02     262      512      256     true
256MiB single  1      116.0    2485.63  2485.63     518      256      256     true
256MiB dual    1      171.3    1617.36  1617.36     518      512      256     true
256MiB single  2      133.3    3723.68  3905.96     518      512      512     true
256MiB dual    2      138.1    3889.50  3940.31     518     1024      512     true
```

**Sweep 2 — large objects (512 MiB / 1024 MiB), all points `verify_ok`**

```
size    mode    conc  agg_MB/s  p50_ms     RSS_MiB  reg_MiB  infl_MiB  verify
512MiB  single  1      101.3     5550.93    1030      512      512     true
512MiB  dual    1      112.1     4558.36    1030     1024      512     true
1024MiB single  1       92.6    12551.71    2054     1024     1024     true
1024MiB dual    1      112.5     9004.30    2054     2048     1024     true
```

**Dual-rail speedup vs object size (conc = 1)**

```
size_MiB  single_MB/s  dual_MB/s  speedup
      32       128.4       197.1    1.54x
      64       121.8       339.9    2.79x
     128       111.1       411.3    3.70x   <- peak
     256       116.0       171.3    1.48x
     512       101.3       112.1    1.11x
    1024        92.6       112.5    1.22x
```

**Concurrency scaling (aggregate goodput, 1 -> 2 workers)**

```
size   mode    1->2 workers
32MiB  single  3.82x      dual  2.49x
64MiB  single  4.09x      dual  1.45x
128MiB single  4.46x      dual  1.20x
256MiB single  1.15x      dual  0.81x   <- regression
```

Reading the sweep (the honest version):

1. **Multi-rail pays off in the 64–128 MiB band** (2.79x–3.70x). Below it the
   per-read connect/first-WR cost dominates (32 MiB: 1.54x); above it the
   single-rail path is no longer the binding constraint.
2. **Speedup decays hard past 256 MiB and absolute goodput falls to ~112 MB/s
   at 512–1024 MiB.** On Soft-RoCE every rail is emulated in the host CPU, so a
   large object means long chains of 4 MiB WRITEs serialized through one QP per
   rail; the CPU, not the NIC, is the ceiling, and a second rail competes for
   the same CPU (1.11x–1.22x). This is exactly the "single QP / shared CPU"
   behavior the design documents as the expected boundary; it is NOT a defect
   in the scheduling logic.
3. **Concurrency helps single-rail more than dual-rail** in the sweet spot
   (single 3.82x–4.46x vs dual 1.20x–2.49x): with two rails already using both
   RXE devices and the host CPU, extra workers add contention. At 256 MiB,
   dual concurrency 1->2 reverses to 0.81x — the point where more parallelism
   buys nothing and costs scheduling overhead. This is the kind of ceiling the
   contest asks to surface, and it is why the read path exposes an inflight
   budget (`ReadOptions::max_inflight_bytes`) rather than accepting unbounded
   concurrency.
4. **Resources are predictable and bounded.** RSS tracks the live destination
   buffers (e.g. 262 MiB at 128 MiB, 2054 MiB at 1024 MiB); registered memory
   is `object_size × workers × rails` by construction; inflight bytes is
   `object_size × workers`. None of these grow without bound — the backpressure
   guard is what keeps them honest.

**Variance caveat.** Soft-RoCE shares the host CPU across all rails and
workers, so the sweep characterizes scheduling, concurrency behavior and
software overhead — resource bounds and backpressure included — rather than
physical NIC aggregate bandwidth. Absolute numbers will differ on real
hardware; the *relative* trends and the resource accounting are what the
design controls.

**Environment honesty.** Functional validation uses the Mock transport and
Soft-RoCE (RXE) on WSL2. These prove functionality, failure semantics,
scheduling and resource governance. They do NOT prove hardware aggregate
bandwidth, HCA offload, PCIe/NUMA effects, or zero-copy behavior on physical
NICs. Every number is labelled with its environment. No GPU is involved.

## 13. Known limitations and risks

| Limitation | Impact | Mitigation / plan |
|---|---|---|
| No physical multi-NIC testbed | Hardware aggregate bandwidth unproven | Labelled Soft-RoCE evidence; design keeps NIC count a config knob |
| **Topology analysis (NUMA / PCIe) not measurable here** | Contest deliverable f asks for NUMA/PCIe affinity; Soft-RoCE on WSL2 has no PCIe path and presents a single NUMA node, so those axes cannot be measured meaningfully | Documented explicitly: NUMA/PCIe attribution is deferred to a physical-NIC environment. The client already exposes per-rail timing, which is the input such an analysis needs; only the environment is missing. |
| v1 has no in-request retry | A rail failure fails the whole read | Documented semantics; retry is additive later |
| Round-robin ignores transient rail load | Skewed rails can straggle | `locality` hook + least-loaded scheduler planned |
| Address-keyed MR caching is unsound as a default | Reusing a freed-then-reallocated buffer silently reads zeros (§8.1) | Default path registers per read; pooled path is explicit + `invalidate_mr_cache`; guarded by `multi_rail_mr_lifetime` |
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
