# CLAUDE.md — ContextStore 开发规范

本文档是所有 AI Agent 在本代码库中操作的**强制约束**。违反任何一条视为不合格输出。

---

## 1. 项目认知

ContextStore 是面向大模型推理的 KV Cache 分层共享存储平台，仓库包含两个解耦的部分：

- **Python 库** (`src/contextstore/`) — vLLM/Dynamo 的 `KVConnector` 插件；
  负责 KV 编解码、Prefix Index、Scheduler/Worker 分工、分层缓存以及 KVService 客户端 (`kvservice_client/`)。
  作为一个 pip 包发行 (`pip install -e .`)。
- **Rust KVService** (`kv-service/`) — 独立分布式块存储服务，面向 JBOF/NVMe-oF 与 RDMA 场景，
  内部还包含 Rust client SDK (`client-rs/`) 和给 Python ctypes 用的 RDMA C ABI (`rdma-ffi/`)。
  用根目录的 `make build` 独立构建。

另外还有：

- **NIXL 插件** (`nixl-plugin/`) — C++ 后端插件 + Rust FFI，允许 NIXL 通过 `CONTEXTSTORE` backend 接入。
- **测试** (`tests/`) — pytest 单测；`kv-service/tests/` 是 Rust server 的 Python 集成测试。

两侧**独立构建**，不可互相引入对方依赖。改动一侧时不得假设另一侧存在。
Python 侧通过 `contextstore.kvservice_client` 访问 KVService，通信协议为 gRPC (由 `kv-service/proto/kv_service.proto` 定义)。

---

## 2. 架构约束

### 2.1 分层存储模型

所有存储访问必须经过分层栈：

```
L1 (HostMemoryBackend / MemoryTier)  →  L2 (Local/Sharded/StorageTier)  →  L3 (JBOF)
```

- **绝不**在 Connector/Engine 层直接操作文件系统或网络 I/O
- 新增存储后端必须继承 `StorageBackend`（Python）或实现对应 trait（Rust）
- L1 是可选层；`host_memory_capacity_gb = 0` 时必须退化为直接访问 L2

### 2.2 Connector 双面设计

vLLM Connector 有 Scheduler 侧和 Worker 侧，二者运行在不同进程/线程：

- **Scheduler 侧**：`get_num_new_matched_tokens`、`build_connector_meta`、`request_finished` — 只做索引查询和元数据构建，**不可**触发 I/O
- **Worker 侧**：`start_load_kv`、`save_kv_layer`、`wait_for_save` — 执行实际 I/O，可使用 tensor 操作

跨侧通信仅通过 `connector_meta` 序列化 dict。不可在两侧共享可变状态。

### 2.3 Combined Tensor 优化

所有 layer 的 KV 合并为单一 pinned tensor 后做单次 DMA H2D 传输。新增的存储方法必须兼容这个模式：
- `put_tensor` / `get_tensor` 接受完整 combined tensor
- 不要引入逐 layer 落盘的热路径

### 2.4 KVService（Rust 侧）

- 共享上下文通过 `Arc<T>` 传递（`KVServiceContext`），不使用全局 static
- 可选能力用 feature gate：`io-uring`、`gds`、`metrics`、`rdma`
- 新增 feature 不得影响默认编译路径（`#[cfg(feature = "...")]` 完全隔离）
- 错误统一用 `anyhow::Result`，公共 API 用自定义 `Error` enum

---

## 3. Python 代码规范

### 3.1 类型与风格

- **必须** `from __future__ import annotations` 在每个文件首行
- 类型标注用 PEP 604 新语法：`bytes | None`、`list[str]`、`dict[str, Any]`
- **禁止** `Optional`、`List`、`Dict`、`Tuple`、`Union` 等旧式泛型
- 数据结构用 `@dataclass`，不使用 TypedDict 或 NamedTuple（除非与 vLLM 接口对齐）
- 类属性命名：私有用 `_` 前缀，暴露通过 `@property`
- 格式化：`black`；静态检查：`ruff`

### 3.2 抽象与继承

- 抽象基类用 `abc.ABC` + `@abstractmethod`，方法体用 `...`（不写 `pass`）
- 基类**应提供**可覆写的默认实现（如 `get_parallel` 默认串行循环）
- 子类覆写时保持签名完全一致，不要添加额外必填参数

### 3.3 导入规则

- 重量级依赖（`redis`、`torch`）在模块顶部导入
- 可选依赖（如 `contextstore.index.prefix_index_redis`）使用**延迟导入**（在函数/方法内 `from ... import ...`）
- `contextstore.kvservice_client` 也使用延迟导入，避免不启用 KVService 的部署强依赖 grpcio
- 导入顺序：stdlib → 第三方 → 本项目，各组之间空行分隔

### 3.4 配置传递

- 所有配置通过 `ContextStoreConfig` dataclass 集中管理
- 新增配置项**必须**有合理默认值
- 工厂方法用 `@classmethod`（如 `from_extra_config`）
- 派生值用 `@property`（如 `max_capacity_bytes`）
- 不要在代码中硬编码魔法数字，放入 config 或定义为模块级常量

### 3.5 测试

- 测试框架：`pytest`，放在 `tests/unit/`（Python 侧单测）或 `kv-service/tests/`（Rust server 集成测试）
- 测试类命名 `TestXxx`，方法命名 `test_xxx`
- 使用 `setup_method` 初始化 fixture
- 存储测试使用 `MemoryStorageBackend`，Redis 测试使用 `fakeredis`
- **不依赖** GPU 或外部 Redis 服务
- 运行方式：`pytest tests/ -v`

---

## 4. Rust 代码规范

- 模块用 `//!` 文档注释说明职责
- 公共结构体/函数用 `///` 文档注释
- 构造模式：`pub fn new(config: &Config) -> anyhow::Result<Self>`
- 格式化：`cargo fmt`；检查：`cargo clippy -- -D warnings`
- 配置加载用 TOML（`configs/*.toml`）
- Protobuf 定义在 `proto/` 目录，server 通过 `build.rs` 生成，Python client 通过 `make proto` 生成到 `src/contextstore/kvservice_client/_pb/`

---

## 5. 文档与注释

- 代码注释、docstring、日志/错误消息一律用**英文**（面向开源发布，方便外部贡献者阅读）
- 标识符、commit message 用**英文**
- 仓库只保留顶层 `README.md`（英文）和 `CLAUDE.md`（本文，中文，仅面向内部 Agent），另外允许子目录 README（`kv-service/deploy/README.md`、`nixl-plugin/contextstore/README.md`）作为局部说明
- 不要新建 `docs/` 目录或散落 `.md` 文件；架构讨论写在代码注释或 PR 描述里

---

## 6. Git 与构建

- Python 包构建：`pip install -e .`（从项目根目录，包含 Connector + KVService client）
- KVService 构建：`make build`（server + client-rs + rdma-ffi）
- 生成 protobuf：`make proto`（Rust 由 `build.rs` 触发；Python 输出到 `src/contextstore/kvservice_client/_pb/`）
- 两侧测试独立：`pytest tests/` 与 `make -C kv-service test`
- 依赖只在 `pyproject.toml` / `Cargo.toml` 声明；不要引入 `requirements/*.txt`
- commit message 格式：简短英文动词开头（`add`/`fix`/`refactor`/`test`/`docs`）
- 不要修改 `pyproject.toml` 或 `Cargo.toml` 的依赖除非明确要求

---

## 7. 性能敏感区域

以下路径是热路径，改动时必须考虑性能影响：

| 路径 | 预期延迟 | 注意事项 |
|------|----------|----------|
| `HostMemoryBackend.get_tensor` | < 0.1ms | pinned memory 零拷贝，不可引入额外内存分配 |
| `KVCodec.encode/decode` | < 0.2ms/block | INT8 量化，使用 torch 向量化操作 |
| `PrefixIndex.lookup_prefix` | < 0.01ms | 纯内存 trie 查询，不可有 I/O |
| `ShardedStorageBackend.get_parallel` | 线性缩放 | 多线程并发，不可持有 GIL |
| Rust `io_executor` | 取决于 tier | Tier B (io_uring) 是零拷贝异步路径 |
| Rust `rdma::server` | ~10 GB/s | slab 预注册 + 零 memcpy，单 lkey 复用 |

**规则**：
- 热路径中不可引入 `logging.debug` 等有格式化开销的调用
- 不可在热路径中创建临时 Python 对象（列表推导等需评估）
- tensor 操作优先使用 in-place（`tensor.copy_()` 而非 `tensor = ...`）

---

## 8. 安全与边界

- 存储后端必须处理 key 不存在的情况（返回 `None`，不抛异常）
- 容量超限时使用 LRU 驱逐，不可静默丢弃数据
- Redis 连接失败时必须优雅降级（退回进程内索引），不可崩溃
- Rust 侧：`unwrap()` 仅允许在 `main.rs` 启动阶段或测试代码中使用

---

## 9. 变更检查清单

每次修改代码前，agent 必须确认：

1. [ ] 是否阅读了相关模块的现有实现（至少读了要修改的文件）
2. [ ] 新代码是否遵循上述类型标注/命名/导入规范
3. [ ] 是否有对应的单元测试（新功能必须有，bugfix 应有回归测试）
4. [ ] 是否影响热路径（如果是，说明性能影响）
5. [ ] 是否需要更新 `ContextStoreConfig`（新增配置项需有默认值）
6. [ ] `pytest tests/ -v` 是否通过（改动 Python 侧后必须验证）
7. [ ] `make build && make test-server` 是否通过（改动 Rust 侧后必须验证）
