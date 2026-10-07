@echo off
REM ============================================================================
REM  build-and-test.cmd — 一键构建 + 测试 ContextStore 多轨代码
REM
REM  用途：在 Windows 上验证 multi-rail 代码可编译、测试全绿、基准可跑。
REM        不依赖 RDMA 硬件（默认 feature），不依赖 GPU。
REM
REM  前置：已安装 Rust（rustup）。本脚本会自动探测 toolchain 路径。
REM  用法：双击运行，或在终端执行  build-and-test.cmd
REM ============================================================================
setlocal enabledelayedexpansion

REM ---- 1. 定位 Rust toolchain（优先 rustup 默认位置）----
if exist "%USERPROFILE%\.cargo\bin\cargo.exe" (
  set "CARGO_BIN=%USERPROFILE%\.cargo\bin"
  goto :found
)
if exist "%LOCALAPPDATA%\Programs\Rust" (
  set "CARGO_BIN=%LOCALAPPDATA%\Programs\Rust\bin"
  goto :found
)
echo [错误] 未找到 cargo.exe。请先安装 Rust: https://rustup.rs
exit /b 1

:found
echo [信息] 使用 toolchain: %CARGO_BIN%
set "PATH=%CARGO_BIN%;%PATH%"

REM 重要：RUSTC 必须是 Windows 反斜杠绝对路径。
REM 若用正斜杠或依赖 PATH 中的 0 字节 shim，indexmap 等 crate 的
REM autocfg 构建探测会失败，导致 tower 报 E0107。
set "RUSTC=%CARGO_BIN%\rustc.exe"

REM ---- 2. 构建目录（避开可能的文件锁；可改回项目内 target）----
if "%CARGO_TARGET_DIR%"=="" set "CARGO_TARGET_DIR=%TEMP%\contextstore-target"
set "CARGO_INCREMENTAL=0"
echo [信息] CARGO_TARGET_DIR=%CARGO_TARGET_DIR%

cd /d "%~dp0kv-service\client-rs" || (echo [错误] 未找到 kv-service\client-rs & exit /b 1)

echo.
echo ============ 1/3  编译（default feature，无 RDMA 依赖）============
cargo build || goto :fail

echo.
echo ============ 2/3  多轨功能测试 ============
cargo test --test multi_rail_mock || goto :fail

echo.
echo ============ 3/3  多轨带宽容量模型（8 轨）============
set "MAX_RAILS=8"
cargo run --release --bin cs-mock-bench || goto :fail

echo.
echo [成功] 编译通过、测试全绿、基准可跑。
echo        下一步：开 Issue / 设计提案与社区对齐（见 docs/multi-rail-read-design.md）。
exit /b 0

:fail
echo.
echo [失败] 请把上面的报错贴回给助手。
exit /b 1