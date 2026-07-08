# ClaudeCode Handoff: Redhare KVService Integration

本文档给 ClaudeCode 接手 redhare 侧对接工作使用。核心点：ContextStore 侧只提供 KVService
contract/client/server 能力；redhare 侧才实现真正的 G4/KVService tier backend。

## 当前目录与分支

ContextStore contract 改动已经迁移到：

```bash
/Users/mmzhou/program/ContextStore-new
branch: codex/redhare-kvservice-contract
```

redhare 仓库当前在：

```bash
/Users/mmzhou/program/daocloud/redhare
```

redhare 侧实现建议新开独立 worktree，不要直接在 main 工作区改：

```bash
git -C /Users/mmzhou/program/daocloud/redhare worktree add \
  -b codex/kvservice-g4-backend \
  /Users/mmzhou/program/daocloud/redhare-kvservice-g4 \
  main
```

## 必读设计文档

先读 ContextStore 中的 contract 文档：

```bash
/Users/mmzhou/program/ContextStore-new/docs/redhare-kvservice-contract.md
```

该文档定义了职责边界：

- ContextStore 负责 KVService server、proto contract、Rust client surface、descriptor read、immutable write。
- redhare 负责 `TierBackend`、placement policy、config、cleaner/admission、Redis metadata 表达。
- 不要把 redhare tier 代码放进 ContextStore。
- 不要把 KVService server、RocksDB、storage tier 或 RDMA server 复制进 redhare。

## ContextStore 侧 contract

ContextStore 侧本分支提供：

- `kv-service/proto/kv_service.proto`
  - `PutChunk` 增加 `PutOptions options = 7`。
  - stream PUT 首 chunk 可以携带 `if_not_exists`。
- `kv-service/server/src/api/service.rs`
  - `Put` / `PutBatch` / `PutStream` 支持 `if_not_exists`。
  - 已存在对象返回 `PutResponse { success: false, message: "already exists" }`。
  - distributed placement 在单 KVService coordinator 内按 object key 串行化 check-and-commit。
- `kv-service/server/src/storage_tier.rs`
  - `put_if_absent`。
  - `put_chunks_if_absent`。
- `kv-service/server/src/memory_tier.rs`
  - `put_if_absent`。
  - `put_chunks_if_absent`。
- `kv-service/client-rs/src/lib.rs`
  - `KvClient::put_if_absent`。
  - `KvClient::put_stream_chunks_if_absent`。
  - `KvClient::lookup_object`。
  - `KvClient::read_by_descriptor_stream_chunks`。
  - `KvClient::delete`。

## redhare 侧接入目标

在 redhare 仓库新增一个 G4/KVService cold tier backend。首版目标是语义跑通，不做 RDMA/GDS。

推荐第一阶段能力：

- redhare local DRAM 满或策略决定 demote 时，可以写入 KVService。
- redhare Redis metadata 中能表达对象位于 KVService/G4 tier。
- `ObjectLocation.storage_ref` 保存 KVService descriptor hint。
- redhare get 看到该对象在 KVService/G4 时，使用 descriptor read 从 KVService 读回 payload。
- 若 KVService 返回 not found 或 stale descriptor，应按 miss/fallback 处理，不能返回错误 payload。

## redhare 现有可复用结构

redhare 已有统一 tier abstraction：

```rust
crates/redhare/src/tier/mod.rs
```

关键接口：

- `TierBackend`
- `Slot`
- `ReserveContext`
- `TierCapabilities`
- `TierClass`

redhare metadata model 在：

```rust
crates/redhare/src/model.rs
```

关键字段：

- `Tier`
- `ObjectLocation`
- `ObjectLocation.storage_ref: Option<String>`
- `ObjectMetadata`

首版可以评估两种路径：

1. 复用 `Tier::RemoteObject` 表达 KVService/G4。
2. 新增更明确的 `Tier::KvService` 或 `Tier::G4`。

如果新增 variant，需要同时检查 cleaner、metadata compatibility、tests 和所有 match 分支。为了降低第一版风险，建议优先评估复用 `RemoteObject`，但若语义上容易混淆，可以新增明确 variant。

## 建议实现模块

建议新增：

```text
crates/redhare/src/tier/kvservice.rs
```

并在现有 tier/backend 组织方式内接入。

建议新增结构：

```rust
pub struct KvServiceBackend {
    client: ...,
    namespace: String,
    endpoint: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvServiceStorageRef {
    pub v: u32,
    pub namespace: String,
    pub object_key: String,
    pub generation: u64,
    pub etag: String,
    pub layout_version: u64,
    pub size: u64,
}
```

`storage_ref` 建议 JSON 序列化，避免把 KVService proto 类型直接塞进 redhare Redis metadata。

## Cargo dependency 建议

首版可以在 redhare worktree 中用 path dependency 指向 ContextStore 的 Rust client：

```toml
contextstore-client-rs = {
  path = "/Users/mmzhou/program/ContextStore-new/kv-service/client-rs"
}
```

如果 crate 名称与实际 `Cargo.toml` 不一致，以 `kv-service/client-rs/Cargo.toml` 为准。

不要把 ContextStore KVService server 作为 redhare crate dependency。

## 写入路径建议

redhare 对象本身是 immutable。KVService 写入应使用：

```rust
KvClient::put_stream_chunks_if_absent(...)
```

语义：

- `Ok(true)`：KVService 写入成功，redhare 可把 metadata 发布为 KVService/G4 location。
- `Ok(false)`：对象已存在，不覆盖。redhare 可以 `lookup_object` 获取 descriptor，并按幂等成功或冲突处理。
- `Err(_)`：不要发布 KVService metadata，按 redhare 现有 failure/fallback 处理。

object key 建议：

```text
namespace = "redhare:<cluster>"
object_key = redhare ObjectKey
```

如果需要模型隔离，可扩展：

```text
namespace = "redhare:<cluster>:<model>"
```

## 读取路径建议

redhare get 遇到 KVService/G4 location 时：

1. 从 `storage_ref` 解析 `KvServiceStorageRef`。
2. 构造或 lookup KVService descriptor。
3. 调用 `KvClient::read_by_descriptor_stream_chunks(...)`。
4. 拼接 segments，或后续优化成直接 scatter 到 caller buffers。
5. 如果返回 stale descriptor / not found，执行 `lookup_object` 刷新 descriptor；刷新失败则按 miss。

首版可以先拼接 `Vec<u8>`，目的是跑通语义。性能优化阶段再做 registered buffer / RDMA / GDS。

## Cleaner 与删除语义

首版建议谨慎处理：

- redhare remove 可以调用 `KvClient::delete(namespace, object_key)`，但失败不能破坏 Redis metadata 的一致性。
- redhare cleaner 不应直接理解 KVService RocksDB 或 storage handle。
- 如果 KVService/G4 metadata stale，redhare cleaner 只能清理 redhare Redis metadata；KVService 侧对象 GC 后续单独设计。

## 测试计划

### redhare standalone 已验证

测试环境：

```text
ssh root@wf20562321o.vicp.fun -p 41897
```

已部署 redhare standalone：

```text
/root/redhare-test/redhare-src-20260706-codex
cluster: codex-redhare-test
Redis: 127.0.0.1:6379/9
client-a UDS: /tmp/redhare-codex-20260706.sock
client-b UDS: /tmp/redhare-codex-20260706-b.sock
```

已验证：

- client-a put
- client-a local get
- client-b remote get
- client-b metrics `redhare_get_hit_remote_total = 1`

### ContextStore KVService smoke

下一步需要把本分支部署到测试环境，跑 KVService smoke：

- `put_stream_chunks_if_absent` first write returns true
- duplicate write returns false
- `lookup_object` returns descriptor + placement
- `read_by_descriptor_stream_chunks` returns original bytes
- stale descriptor returns failed precondition / miss-like behavior

### redhare + KVService integration

redhare side 实现后，端到端测试：

1. 启动 KVService。
2. 启动 Redis。
3. 启动 redhare client with KVService/G4 config。
4. put object，使其落到 KVService/G4 tier。
5. 检查 Redis metadata：
   - tier is KVService/G4 or RemoteObject
   - `storage_ref` contains KVService descriptor JSON
6. 另一个 redhare client get 同 key。
7. 验证 bytes 正确。
8. 验证 KVService `lookup_object` 能找到对象。
9. 重复 put 同 key 不覆盖 KVService object。

## 不要做的事

- 不要把所有对接代码放到 ContextStore。
- 不要复制 KVService server 到 redhare。
- 不要绕过 redhare `TierBackend` / metadata model 直接在 client get/put 里硬编码 KVService。
- 不要在 scheduler/index lookup 这类轻量路径里引入网络 I/O。
- 不要第一版就做 RDMA/GDS，先用 gRPC stream 跑通语义。

## 交接结论

ContextStore 侧设计和 contract 改动在 `/Users/mmzhou/program/ContextStore-new` 的
`codex/redhare-kvservice-contract` 分支。ClaudeCode 应新开 redhare worktree，实现 redhare
仓库内的 KVService/G4 backend，并严格遵循 `docs/redhare-kvservice-contract.md` 中的职责边界。
