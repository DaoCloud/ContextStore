# WSL2 Soft-RoCE（双 rxe 软 RDMA）环境搭建手册

> 用途：为赛题「多轨并行读取」构造两条独立 RDMA 路径（rxe0/rxe1），在无实体 RDMA 网卡上做真 verbs 端到端验证。
> 配套脚本：仓库根目录 `setup-wsl2-rxe.sh`（本手册第 4 步调用）。
> 原则：所有数据只声称 Soft-RoCE 环境结论，不声称实体网卡带宽。

## 第 0 步 · 门禁检查（30 秒）

WSL2 终端：

```bash
uname -r
sudo modprobe rdma_rxe && lsmod | grep rxe
```

- **有输出（rdma_rxe 已加载）** → 内核已支持，直接跳到第 4 步。
- **报错 `modprobe: FATAL: Module rdma_rxe not found`** → 继续第 1~3 步编译自定义内核（一次性，约 30~90 分钟，主要为编译等待）。

## 第 1 步 · 安装编译依赖

```bash
sudo apt update
sudo apt install -y build-essential flex bison libssl-dev libelf-dev bc dwarves git
```

## 第 2 步 · 编译带 RXE 的 WSL2 内核

```bash
# 拉取与当前 WSL2 版本完全匹配的内核源码（tag 形如 linux-msft-wsl-5.15.153.1）
cd ~
KVER=$(uname -r | sed 's/-microsoft.*//')
echo "kernel version: $KVER"
git clone --depth 1 --branch "linux-msft-wsl-${KVER}" https://github.com/microsoft/WSL2-Linux-Kernel.git
cd WSL2-Linux-Kernel

# 以微软官方 WSL2 配置为底
cp Microsoft/config-wsl .config

# 开启 Soft-RoCE 所需选项（olddefconfig 会自动补齐依赖）
./scripts/config --enable CONFIG_INFINIBAND \
                 --enable CONFIG_INFINIBAND_USER_ACCESS \
                 --enable CONFIG_RDMA_RXE \
                 --enable CONFIG_INFINIBAND_VIRT_DMA
make olddefconfig

# 编译（32 线程机器约 20~40 分钟；可去干别的，完成后会回到提示符）
make -j$(nproc) 2>&1 | tee build.log
sudo make modules_install
```

验证产物存在：

```bash
ls -lh arch/x86/boot/bzImage
```

## 第 3 步 · 切换到自定义内核

```bash
# WSL 里：把内核镜像拷到 Windows 侧
mkdir -p /mnt/c/Users/SHJ/wsl-kernels
cp arch/x86/boot/bzImage /mnt/c/Users/SHJ/wsl-kernels/bzImage-rxe
```

**Windows 侧**：新建/编辑 `C:\Users\SHJ\.wslconfig`（纯文本，注意是用户目录下的隐藏文件）：

```ini
[wsl2]
kernel=C:\\Users\\SHJ\\wsl-kernels\\bzImage-rxe
```

**PowerShell** 中重启 WSL：

```powershell
wsl --shutdown
```

重新打开 WSL 终端，验证：

```bash
sudo modprobe rdma_rxe && lsmod | grep rxe   # 这次必须有输出
```

> 回滚方法：删掉 `.wslconfig` 里的 `kernel=` 行再 `wsl --shutdown` 即恢复官方内核。

## 第 4 步 · 拉起双 rxe 路径

```bash
sudo apt install -y rdma-core infiniband-diags ibverbs-utils perftest
cd ~/python_project/Industrial_LLM/ContextStore   # WSL 克隆
sudo bash setup-wsl2-rxe.sh
```

脚本会：加载 rdma_rxe → 建 veth0/veth1（192.168.96.110/111）→ 各绑一个 rxe → 自检。
看到 `OK：rxe0->veth0 与 rxe1->veth1 两条独立 Soft-RoCE 路径已就绪` 即成功。

## 第 5 步 · 真 verbs 双轨带宽证据（演示视频素材）

```bash
# 两个后台流并行：rxe0 与 rxe1 同时打满，证明两条路径独立聚合
ib_send_bw -d rxe0 192.168.96.111 --report_gbits &
ib_send_bw -d rxe1 192.168.96.110 --report_gbits &
wait
```

把输出中两个带宽值相加 = 多轨聚合带宽的真 verbs 证据。**此步骤建议录屏。**

## 第 6 步 · ContextStore 双轨端到端

服务端已支持多 NIC 监听（`server/src/main.rs` 的 `CS_RDMA_DEVICES`），示例配置已备好
（`kv-service/configs/server-wsl2-softroce.toml`：4 MiB 条带阈值，64 MiB 演示对象 = 16 条带）：

```bash
# 1) 编译带 rdma feature 的 server（首次较久，~10-20 分钟）
cd ~/python_project/Industrial_LLM/ContextStore
cargo build -p contextstore-server --features rdma --release

# 2) 起本地 Redis（server 元数据依赖，配置指向 127.0.0.1:6379）
redis-server --daemonize yes --save ''

# 3) 准备演示数据目录（配置里的两个"设备"）
mkdir -p /tmp/cs-data/dev0 /tmp/cs-data/dev1

# 4) 双 rxe 监听启动 server（注意 gid_index=1：RXE 的 RoCEv2 GID；
#    若连接失败用 `show_gids` 核对 rxe0/rxe1 的实际索引）
cd kv-service
CS_RDMA_DEVICES=rxe0:0.0.0.0:50053:1,rxe1:0.0.0.0:50054:1 \
  ../target/release/contextstore-server --config configs/server-wsl2-softroce.toml
# 另开一个 WSL 终端做下面的客户端步骤（server 前台运行）
```

客户端双轨端到端读（PUT → lookup → 单轨读 → 双轨读 → 校验 + 加速比报告）：

```bash
cd ~/python_project/Industrial_LLM/ContextStore
cargo run -p contextstore-client-rs --features rdma --example softroce_dual_rail
```

期望输出：`[setup] PUT ...`、`[single-rail] ... transfer=XX MB/s`、
`[dual-rail] ...`、`[result] dual-rail vs single-rail transfer-phase speedup: ~1.5-2x`。
**全程录屏**——这是演示视频的核心素材；速度比和 bottleneck 归因截图进赛事报告。

> 端口约定：gRPC 控制面 50051；RDMA 控制通道 50053(rxe0)/50054(rxe1)，避开 gRPC 端口。
> 可用环境变量覆盖：`CS_GRPC` / `CS_RAIL_ENDPOINTS` / `CS_RAIL_DEVICES` / `CS_RAIL_GID` / `CS_DEMO_MIB`。

## 常见问题

| 症状 | 处理 |
|---|---|
| `modprobe rdma_rxe` 报 not found | 内核没换成功：检查 `.wslconfig` 路径、是否 `wsl --shutdown` 后重开 |
| `rdma link add` 报 `Operation not supported` | rxe 模块没加载，先 `sudo modprobe rdma_rxe` |
| 跨 veth ping 失败 | 检查 `sysctl net.ipv4.conf.veth*.accept_local=1`（脚本已做） |
| 换内核后 WSL 起不来 | 删 `.wslconfig` 的 `kernel=` 行回滚，检查 bzImage 是否完整拷贝 |
| `ib_send_bw` 卡住 | 确认两 rxe 都 up：`rdma link`；确认 perftest 版本一致 |
