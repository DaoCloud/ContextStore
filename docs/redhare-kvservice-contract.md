# Redhare KVService Contract

本文档定义 ContextStore 为 redhare 接入 G4 cold tier 提供的最小契约。redhare 侧的
`TierBackend`、placement policy、vLLM connector 配置和 cleaner 逻辑仍应在 redhare 仓库实现；
ContextStore 只提供 KVService 协议、轻量 Rust client 和必要的服务端语义。

## 目标边界

### ContextStore 负责

- 保持 KVService server 独立部署和独立构建。
- 暴露 redhare 可依赖的轻量 Rust client crate。
- 提供对象级 descriptor / placement lookup。
- 提供 descriptor 校验读，支持 stale descriptor 快速失败。
- 提供 immutable 写入语义：`if_not_exists=true` 时对象已存在则不覆盖。
- 保留 gRPC stream 可用路径，并为 RDMA/GDS zero-copy 路径预留同一对象身份。

### redhare 负责

- 在 redhare 仓库新增 G4/KVService tier backend。
- 将 redhare key 映射为 KVService `(namespace, object_key)`。
- 将 KVService descriptor hint 存入 redhare `storage_ref`。
- 在 redhare Redis metadata 中表达对象当前位于 G4/KVService tier。
- 根据 redhare admission / eviction / cleaner 语义决定何时写入、删除或复查 G4 对象。
- 后续使用 KVService RDMA/GDS fast path 直接写入 redhare/vLLM 注册 buffer。

## 元数据分工

redhare 和 KVService 使用两套元数据，各自职责不同：

```text
redhare Redis metadata
  key -> ObjectLocation {
    tier: KvService/G4,
    storage_ref: KvServiceObjectRef
  }

KVService RocksDB metadata
  namespace/object_key -> ObjectDescriptor + PlacementDescriptor
```

Redis 是 redhare 的运行时索引，丢失后可退化为 cache miss。RocksDB 是 KVService 的持久对象
metadata，记录 generation、etag、layout、placement 和 storage handle。两者不互相替代。

建议 redhare 的 `storage_ref` 保存轻量引用：

```json
{
  "v": 1,
  "namespace": "redhare:<cluster>:<model>",
  "object_key": "<redhare-object-key>",
  "generation": 1,
  "etag": "<content-etag>",
  "layout_version": 1,
  "size": 67108864
}
```

redhare 后续可先用 `ReadByDescriptorStream` 校验读；若 KVService 返回
`FAILED_PRECONDITION`，则重新 `LookupObject` 或把该 G4 hit 视为 miss。

## 写入语义

redhare 对象是 immutable，KVService 必须支持同一语义：

- `if_not_exists=false`：保持现有覆盖写行为，生成新的 generation。
- `if_not_exists=true`：对象已存在时不覆盖，返回 `PutResponse { success: false,
  message: "already exists" }`。
- 对 stream PUT，`PutOptions` 只放在第一块 `PutChunk`。

这让 redhare 可以安全地把多个 worker 的同 key 并发写映射为幂等结果，而不是让 KVService
生成新的 generation 覆盖原对象。

第一阶段的原子性边界是单个 KVService coordinator：同一 coordinator 内的普通 L2 写和
distributed placement 写都会按 object key 串行化 check-and-commit。若 redhare 后续让同一
object key 同时写入多个独立 coordinator，需要在 redhare routing 层固定 owner，或在 KVService
metadata 层补充跨 coordinator CAS/lease。

## 读取语义

redhare G4 backend 首版建议使用：

1. `LookupObject(namespace, object_key)` 获取 descriptor 和 placement。
2. `ReadByDescriptorStream(descriptor, placement)` 读取数据并校验对象身份。
3. 返回前更新 redhare 本地 descriptor hint。

高性能路径演进：

```text
ReadByDescriptorStream
  -> RDMA get_into(descriptor, registered_region, offset)
  -> GDS/GDR get_to_gpu(descriptor, gpu_buffer)
```

descriptor/generation/etag/layout_version 必须贯穿所有数据面，避免 stale Redis metadata 或
slot 复用导致错误 KV 被加载。

## Rust Client Surface

`kv-service/client-rs` 是 redhare 后续应依赖的最小 crate。当前提供：

- `KvClient::connect(endpoint)`
- `KvClient::put_with_options(...)`
- `KvClient::put_if_absent(...)`
- `KvClient::put_stream_with_options(...)`
- `KvClient::put_stream_if_absent(...)`
- `KvClient::put_stream_chunks_with_options(...)`
- `KvClient::put_stream_chunks_if_absent(...)`
- `KvClient::lookup_object(...)`
- `KvClient::read_by_descriptor_stream_chunks(...)`
- `KvClient::delete(...)`
- `KvClient::exists(...)`

后续 redhare 侧可以先用 path dependency：

```toml
contextstore-client-rs = {
  path = "/Users/mmzhou/program/ContextStore-new/kv-service/client-rs"
}
```

接口稳定后再切到 git/tag dependency。KVService server、RocksDB、storage tier 和 RDMA server
实现不应复制进 redhare。

## 性能路线

### 阶段 1：语义跑通

- redhare G4 backend 使用 `put_stream_chunks_if_absent` 和
  `read_by_descriptor_stream_chunks`。
- 数据仍经过 gRPC stream，有协议开销，但可验证对象身份、删除和 miss/fallback 语义。

### 阶段 2：RDMA Host Zero-copy

- redhare 注册 pinned host buffer。
- KVService RDMA server 将对象写入已注册 region。
- redhare 避免 Python bytes、`ctypes.string_at` 和大 `Vec<u8>` 拼接。

### 阶段 3：GDS/GDR

- GDS：KVService 直接读写同机 GPU buffer。
- GDR：跨机 RDMA 直接写 GPU HBM。
- descriptor 身份校验保持不变，只替换数据面。

## Worktree 工作流

当前 ContextStore 支撑侧开发在独立目录中进行：

```bash
/Users/mmzhou/program/ContextStore-new
branch: codex/redhare-kvservice-contract
```

常用查看：

```bash
git -C /Users/mmzhou/program/ContextStore-new status --short --branch
git -C /Users/mmzhou/program/ContextStore-new diff --stat
```

同步上游：

```bash
git -C /Users/mmzhou/program/ContextStore-new fetch origin
git -C /Users/mmzhou/program/ContextStore-new rebase origin/main
```

合并回主线有两种方式：

1. 推分支开 PR：推荐用于两个仓库协同。
2. 本地 merge：在目标主目录切到目标分支后执行
   `git merge codex/redhare-kvservice-contract`。

不要在已有脏工作区合并；先用 `git status --short` 确认目标目录干净。
