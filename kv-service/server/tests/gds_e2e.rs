//! GDS end-to-end integration tests (require GPU + libcufile + nvidia-fs + a GDS-capable filesystem)
//!
//! How to run:
//!   GDS_TEST_DIR=/data/gds cargo test --features "io-uring,gds" --test gds_e2e -- --nocapture --test-threads=1
//!
//! Tests come in two categories:
//! 1. "platform-only" — paths that don't depend on nvidia-fs filesystem support and should pass on any GPU host:
//!    cudaMalloc + cuFileBufRegister + cuIpcGetMemHandle
//! 2. "fs-required" — depend on an FS taken over by nvidia-fs (typically WekaFS, BeeGFS, GPFS, or ext4/xfs
//!    with the nvme.ko shim). In a plain container/VM where nvidia-fs is not registered against the local NVMe,
//!    calls fail with 5018 (unsupported file type) and the tests auto-skip instead of failing.
//!
//! overlayfs / tmpfs never support GDS; you need an ext4/xfs directly mounted on NVMe and nvidia-fs must have
//! patched the nvme block driver. Check `cat /proc/driver/nvidia-fs/modules` to see which FS nvidia-fs has taken over.

#![cfg(feature = "gds")]

use contextstore_server::gds::{GdsDriver, GpuBuffer, GpuFileHandle};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

fn gds_test_dir() -> Option<PathBuf> {
    std::env::var_os("GDS_TEST_DIR")
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

/// Probe the GDS driver. Does not require FS support; returns true as long as libcufile + cuFileDriverOpen succeed.
fn require_gds_driver() -> bool {
    match GdsDriver::init() {
        Ok(true) => true,
        Ok(false) => {
            eprintln!("SKIP: GDS driver unavailable (libcufile missing or driver_open failed)");
            false
        }
        Err(e) => {
            eprintln!("SKIP: GDS init error: {}", e);
            false
        }
    }
}

fn make_test_file(dir: &std::path::Path, name: &str, payload: &[u8]) -> PathBuf {
    let p = dir.join(name);
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(payload).unwrap();
    f.sync_all().unwrap();
    p
}

/// (Platform-only) cudaMalloc + cuFileBufRegister + Drop must not error.
/// Does not depend on nvidia-fs FS support; should pass on any GPU host with GDS installed.
#[test]
fn gpu_buffer_alloc_and_register() {
    if !require_gds_driver() {
        return;
    }
    let buf = GpuBuffer::alloc(4 * 1024 * 1024).expect("cudaMalloc + cuFileBufRegister");
    println!(
        "OK: alloc + register 4MB GPU buf @ {:?}",
        buf.as_ptr()
    );
    drop(buf); // implicit cuFileBufDeregister + cudaFree
}

/// (Platform-only) H2D + D2H via cudaMemcpy. Verifies the buffer.copy_*_host interface.
#[test]
fn gpu_buffer_host_roundtrip() {
    if !require_gds_driver() {
        return;
    }
    let payload: Vec<u8> = (0..64 * 1024).map(|i| (i % 251) as u8).collect();
    let mut buf = GpuBuffer::alloc(payload.len()).unwrap();
    buf.copy_from_host(&payload).unwrap();
    let mut back = vec![0u8; payload.len()];
    buf.copy_to_host(&mut back).unwrap();
    assert_eq!(back, payload, "H2D -> D2H data mismatch");
    println!("OK: 64KB H2D->D2H roundtrip");
}

/// (Platform-only) cuIpcGetMemHandle produces a 64B handle.
#[test]
fn ipc_get_handle_produces_64_bytes() {
    if !require_gds_driver() {
        return;
    }
    let d = GdsDriver::get().unwrap();
    let owner = GpuBuffer::alloc(64 * 1024).unwrap();
    let mut handle = contextstore_server::gds::ffi::CUipcMemHandle { reserved: [0u8; 64] };
    let cu = unsafe { (d.cu_ipc_get_mem_handle)(&mut handle, owner.as_ptr() as u64) };
    assert_eq!(cu, 0, "cuIpcGetMemHandle failed: {}", cu);
    assert!(
        handle.reserved.iter().any(|&b| b != 0),
        "handle is all zero, likely not actually written"
    );
    println!("OK: cuIpcGetMemHandle produced 64B handle (cross-process open covered by the Python e2e tests)");
}

/// (FS-required) Real NVMe -> GPU direct DMA. Auto-skips when nvidia-fs has not taken over the FS.
#[test]
fn gds_nvme_to_gpu_direct_dma() {
    if !require_gds_driver() {
        return;
    }
    let dir = match gds_test_dir() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: GDS_TEST_DIR is not set (e.g. /data/gds)");
            return;
        }
    };

    let payload: Vec<u8> = (0..4 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    let path = make_test_file(&dir, "gds_read_test.bin", &payload);

    let mut gpu = GpuBuffer::alloc(payload.len()).expect("cudaMalloc + buf register");

    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(&path)
        .unwrap();

    let fh = match GpuFileHandle::register(file) {
        Ok(h) => h,
        Err(e) => {
            eprintln!(
                "SKIP: cuFileHandleRegister failed ({}). Usually means nvidia-fs has not taken over {}\n\
                 check: cat /proc/driver/nvidia-fs/modules",
                e,
                dir.display()
            );
            std::fs::remove_file(&path).ok();
            return;
        }
    };
    let n = fh.pread(&mut gpu, 0, payload.len()).expect("cuFileRead");
    assert_eq!(n, payload.len(), "cuFileRead byte count mismatch");

    let mut host = vec![0u8; payload.len()];
    gpu.copy_to_host(&mut host).expect("cudaMemcpy D2H");
    assert_eq!(host, payload, "GDS read content does not match original payload");

    std::fs::remove_file(&path).ok();
    println!("OK: 4MB NVMe -> GPU direct DMA + D2H validation passed");
}
