//! Control-plane wire protocol — simple binary framing over TCP
//!
//! ## Message flow
//!
//! 1. client connect TCP → server accept
//! 2. ClientHello { qp_info } → ServerHello { qp_info }
//!    (after both sides have the peer's QP info, each transitions its local QP RESET→INIT→RTR→RTS)
//! 3. repeatedly:
//!    - GetRequest { key, dst_addr, dst_rkey, max_size }
//!    - server posts WRITE on the RDMA data channel
//!    - on completion → GetResponse { bytes_written, found }
//! 4. close

use anyhow::{anyhow, Result};
use rdma_sys::ibv_gid;
use std::io::{Read, Write};
use std::net::TcpStream;

use crate::rdma::qp::QpInfo;

const MSG_HELLO: u8 = 1;
pub const MSG_GET_REQ: u8 = 2;
pub const MSG_GET_RESP: u8 = 3;
// ===== PUT data plane (client → server, RDMA WRITE pushes data) =====
// Flow:
//   1. client → server: PUT_REQ {key, size}
//   2. server: slab.alloc(size), replies PUT_READY {ok, dst_addr, dst_rkey}
//   3. client RDMA WRITE_WITH_IMM → server slab
//   4. client → server: PUT_COMMIT {} (short TCP message signalling WRITE completion)
//   5. server pwrites O_DIRECT from the slab straight to NVMe (zero memcpy)
//   6. server → client: PUT_RESP {ok}
// Using a TCP COMMIT instead of RDMA WRITE_WITH_IMM + post_recv keeps the change small:
// the server's existing CQ only polls send completions and does not open a recv path.
// The microsecond-scale TCP overhead is negligible here.
pub const MSG_PUT_REQ: u8 = 4;
pub const MSG_PUT_READY: u8 = 5;
pub const MSG_PUT_COMMIT: u8 = 6;
pub const MSG_PUT_RESP: u8 = 7;
/// Descriptor GET data plane: client carries an ObjectDescriptor identity; server rebuilds the
/// physical layout from the descriptor to read, and validates the version against current metadata.
pub const MSG_GET_DESCRIPTOR_REQ: u8 = 8;
const MSG_BYE: u8 = 99;

/// Synchronous read of exactly N bytes (TCP control plane messages are small; blocking read is fine).
pub fn read_exact(stream: &mut TcpStream, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    stream
        .read_exact(&mut buf)
        .map_err(|e| anyhow!("tcp read {} bytes failed: {}", len, e))?;
    Ok(buf)
}

/// Send Hello: 24 bytes of QP info
pub fn send_hello(stream: &mut TcpStream, qp_info: &QpInfo) -> Result<()> {
    let bytes = qp_info.to_bytes();
    let mut frame = Vec::with_capacity(1 + 24);
    frame.push(MSG_HELLO);
    frame.extend_from_slice(&bytes);
    stream
        .write_all(&frame)
        .map_err(|e| anyhow!("tcp write hello failed: {}", e))?;
    stream.flush().ok();
    Ok(())
}

/// Receive Hello, returning the remote QP info
pub fn recv_hello(stream: &mut TcpStream) -> Result<QpInfo> {
    let tag = read_exact(stream, 1)?[0];
    if tag != MSG_HELLO {
        return Err(anyhow!("expected MSG_HELLO ({}), got {}", MSG_HELLO, tag));
    }
    let body = read_exact(stream, 24)?;
    let arr: [u8; 24] = body.try_into().map_err(|_| anyhow!("hello body size"))?;
    Ok(QpInfo::from_bytes(&arr))
}

/// Get request message
/// wire: tag(1) + key_len(2) + key + dst_addr(8) + dst_rkey(4) + max_size(8)
pub struct GetReqMsg {
    pub key: String,
    pub dst_addr: u64,
    pub dst_rkey: u32,
    pub max_size: u64,
}

pub fn send_get_req(stream: &mut TcpStream, msg: &GetReqMsg) -> Result<()> {
    let key_bytes = msg.key.as_bytes();
    if key_bytes.len() > 65535 {
        return Err(anyhow!("key too long: {}", key_bytes.len()));
    }
    let mut frame = Vec::with_capacity(1 + 2 + key_bytes.len() + 8 + 4 + 8);
    frame.push(MSG_GET_REQ);
    frame.extend_from_slice(&(key_bytes.len() as u16).to_le_bytes());
    frame.extend_from_slice(key_bytes);
    frame.extend_from_slice(&msg.dst_addr.to_le_bytes());
    frame.extend_from_slice(&msg.dst_rkey.to_le_bytes());
    frame.extend_from_slice(&msg.max_size.to_le_bytes());
    stream
        .write_all(&frame)
        .map_err(|e| anyhow!("tcp write get_req failed: {}", e))?;
    stream.flush().ok();
    Ok(())
}

pub fn recv_get_req(stream: &mut TcpStream) -> Result<GetReqMsg> {
    let tag = read_exact(stream, 1)?[0];
    if tag != MSG_GET_REQ {
        return Err(anyhow!("expected MSG_GET_REQ, got {}", tag));
    }
    let key_len_bytes = read_exact(stream, 2)?;
    let key_len = u16::from_le_bytes([key_len_bytes[0], key_len_bytes[1]]) as usize;
    let key_bytes = read_exact(stream, key_len)?;
    let key = String::from_utf8(key_bytes).map_err(|e| anyhow!("key utf8: {}", e))?;
    let dst_addr_b = read_exact(stream, 8)?;
    let dst_addr = u64::from_le_bytes(dst_addr_b.try_into().unwrap());
    let dst_rkey_b = read_exact(stream, 4)?;
    let dst_rkey = u32::from_le_bytes(dst_rkey_b.try_into().unwrap());
    let max_size_b = read_exact(stream, 8)?;
    let max_size = u64::from_le_bytes(max_size_b.try_into().unwrap());
    Ok(GetReqMsg {
        key,
        dst_addr,
        dst_rkey,
        max_size,
    })
}

fn read_string_u16(stream: &mut TcpStream, field: &str) -> Result<String> {
    let len_bytes = read_exact(stream, 2)?;
    let len = u16::from_le_bytes([len_bytes[0], len_bytes[1]]) as usize;
    let bytes = read_exact(stream, len)?;
    String::from_utf8(bytes).map_err(|e| anyhow!("{} utf8: {}", field, e))
}

fn write_string_u16(frame: &mut Vec<u8>, value: &str, field: &str) -> Result<()> {
    let bytes = value.as_bytes();
    if bytes.len() > 65535 {
        return Err(anyhow!("{} too long: {}", field, bytes.len()));
    }
    frame.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
    frame.extend_from_slice(bytes);
    Ok(())
}

/// Descriptor GET request message.
///
/// wire:
/// tag(1)
/// + key_len(2) + key
/// + handle_len(2) + object_handle
/// + object_generation(8)
/// + etag_len(2) + content_etag
/// + layout_version(8) + size(8)
/// + is_striped(1) + stripe_count(4) + chunk_size(8)
/// + dst_addr(8) + dst_rkey(4) + max_size(8)
pub struct DescriptorGetReqMsg {
    pub key: String,
    pub object_handle: String,
    pub object_generation: u64,
    pub content_etag: String,
    pub layout_version: u64,
    pub size: u64,
    pub is_striped: bool,
    pub stripe_count: u32,
    pub chunk_size: u64,
    pub dst_addr: u64,
    pub dst_rkey: u32,
    pub max_size: u64,
}

pub fn send_descriptor_get_req(stream: &mut TcpStream, msg: &DescriptorGetReqMsg) -> Result<()> {
    let mut frame = Vec::with_capacity(
        1 + 2
            + msg.key.len()
            + 2
            + msg.object_handle.len()
            + 8
            + 2
            + msg.content_etag.len()
            + 8
            + 8
            + 1
            + 4
            + 8
            + 8
            + 4
            + 8,
    );
    frame.push(MSG_GET_DESCRIPTOR_REQ);
    write_string_u16(&mut frame, &msg.key, "key")?;
    write_string_u16(&mut frame, &msg.object_handle, "object_handle")?;
    frame.extend_from_slice(&msg.object_generation.to_le_bytes());
    write_string_u16(&mut frame, &msg.content_etag, "content_etag")?;
    frame.extend_from_slice(&msg.layout_version.to_le_bytes());
    frame.extend_from_slice(&msg.size.to_le_bytes());
    frame.push(if msg.is_striped { 1 } else { 0 });
    frame.extend_from_slice(&msg.stripe_count.to_le_bytes());
    frame.extend_from_slice(&msg.chunk_size.to_le_bytes());
    frame.extend_from_slice(&msg.dst_addr.to_le_bytes());
    frame.extend_from_slice(&msg.dst_rkey.to_le_bytes());
    frame.extend_from_slice(&msg.max_size.to_le_bytes());
    stream
        .write_all(&frame)
        .map_err(|e| anyhow!("tcp write descriptor_get_req failed: {}", e))?;
    stream.flush().ok();
    Ok(())
}

/// Receive DescriptorGetReq body (the tag has already been consumed by the caller).
pub fn recv_descriptor_get_req_body(stream: &mut TcpStream) -> Result<DescriptorGetReqMsg> {
    let key = read_string_u16(stream, "key")?;
    let object_handle = read_string_u16(stream, "object_handle")?;
    let generation_b = read_exact(stream, 8)?;
    let object_generation = u64::from_le_bytes(generation_b.try_into().unwrap());
    let content_etag = read_string_u16(stream, "content_etag")?;
    let layout_b = read_exact(stream, 8)?;
    let layout_version = u64::from_le_bytes(layout_b.try_into().unwrap());
    let size_b = read_exact(stream, 8)?;
    let size = u64::from_le_bytes(size_b.try_into().unwrap());
    let is_striped = read_exact(stream, 1)?[0] != 0;
    let stripe_count_b = read_exact(stream, 4)?;
    let stripe_count = u32::from_le_bytes(stripe_count_b.try_into().unwrap());
    let chunk_size_b = read_exact(stream, 8)?;
    let chunk_size = u64::from_le_bytes(chunk_size_b.try_into().unwrap());
    let dst_addr_b = read_exact(stream, 8)?;
    let dst_addr = u64::from_le_bytes(dst_addr_b.try_into().unwrap());
    let dst_rkey_b = read_exact(stream, 4)?;
    let dst_rkey = u32::from_le_bytes(dst_rkey_b.try_into().unwrap());
    let max_size_b = read_exact(stream, 8)?;
    let max_size = u64::from_le_bytes(max_size_b.try_into().unwrap());
    Ok(DescriptorGetReqMsg {
        key,
        object_handle,
        object_generation,
        content_etag,
        layout_version,
        size,
        is_striped,
        stripe_count,
        chunk_size,
        dst_addr,
        dst_rkey,
        max_size,
    })
}

/// Get response
/// wire: tag(1) + found(1) + bytes_written(8) + num_chunks(4)
pub struct GetRespMsg {
    pub found: bool,
    pub bytes_written: u64,
    pub num_chunks: u32,
}

pub fn send_get_resp(stream: &mut TcpStream, msg: &GetRespMsg) -> Result<()> {
    let mut frame = [0u8; 1 + 1 + 8 + 4];
    frame[0] = MSG_GET_RESP;
    frame[1] = if msg.found { 1 } else { 0 };
    frame[2..10].copy_from_slice(&msg.bytes_written.to_le_bytes());
    frame[10..14].copy_from_slice(&msg.num_chunks.to_le_bytes());
    stream
        .write_all(&frame)
        .map_err(|e| anyhow!("tcp write get_resp failed: {}", e))?;
    stream.flush().ok();
    Ok(())
}

pub fn recv_get_resp(stream: &mut TcpStream) -> Result<GetRespMsg> {
    let tag = read_exact(stream, 1)?[0];
    if tag != MSG_GET_RESP {
        return Err(anyhow!("expected MSG_GET_RESP, got {}", tag));
    }
    let body = read_exact(stream, 1 + 8 + 4)?;
    let found = body[0] != 0;
    let bytes_written = u64::from_le_bytes(body[1..9].try_into().unwrap());
    let num_chunks = u32::from_le_bytes(body[9..13].try_into().unwrap());
    Ok(GetRespMsg {
        found,
        bytes_written,
        num_chunks,
    })
}

pub fn send_bye(stream: &mut TcpStream) -> Result<()> {
    let _ = stream.write_all(&[MSG_BYE]);
    let _ = stream.flush();
    Ok(())
}

// ===================== PUT data plane =====================

/// Put request message (client → server)
/// wire: tag(1) + key_len(2) + key + size(8)
pub struct PutReqMsg {
    pub key: String,
    pub size: u64,
}

pub fn send_put_req(stream: &mut TcpStream, msg: &PutReqMsg) -> Result<()> {
    let key_bytes = msg.key.as_bytes();
    if key_bytes.len() > 65535 {
        return Err(anyhow!("key too long: {}", key_bytes.len()));
    }
    let mut frame = Vec::with_capacity(1 + 2 + key_bytes.len() + 8);
    frame.push(MSG_PUT_REQ);
    frame.extend_from_slice(&(key_bytes.len() as u16).to_le_bytes());
    frame.extend_from_slice(key_bytes);
    frame.extend_from_slice(&msg.size.to_le_bytes());
    stream
        .write_all(&frame)
        .map_err(|e| anyhow!("tcp write put_req failed: {}", e))?;
    stream.flush().ok();
    Ok(())
}

/// Receive PutReq body (the tag has already been consumed by the caller)
pub fn recv_put_req_body(stream: &mut TcpStream) -> Result<PutReqMsg> {
    let key_len_bytes = read_exact(stream, 2)?;
    let key_len = u16::from_le_bytes([key_len_bytes[0], key_len_bytes[1]]) as usize;
    let key_bytes = read_exact(stream, key_len)?;
    let key = String::from_utf8(key_bytes).map_err(|e| anyhow!("key utf8: {}", e))?;
    let size_b = read_exact(stream, 8)?;
    let size = u64::from_le_bytes(size_b.try_into().unwrap());
    Ok(PutReqMsg { key, size })
}

/// Put Ready response (server → client): tells the client where to WRITE the data
/// wire: tag(1) + ok(1) + dst_addr(8) + dst_rkey(4)
/// ok=0 means the server rejected the request (slab full / internal error); the client sends
/// neither a WRITE nor a COMMIT and falls back to a gRPC PUT.
pub struct PutReadyMsg {
    pub ok: bool,
    pub dst_addr: u64,
    pub dst_rkey: u32,
}

pub fn send_put_ready(stream: &mut TcpStream, msg: &PutReadyMsg) -> Result<()> {
    let mut frame = [0u8; 1 + 1 + 8 + 4];
    frame[0] = MSG_PUT_READY;
    frame[1] = if msg.ok { 1 } else { 0 };
    frame[2..10].copy_from_slice(&msg.dst_addr.to_le_bytes());
    frame[10..14].copy_from_slice(&msg.dst_rkey.to_le_bytes());
    stream
        .write_all(&frame)
        .map_err(|e| anyhow!("tcp write put_ready failed: {}", e))?;
    stream.flush().ok();
    Ok(())
}

pub fn recv_put_ready(stream: &mut TcpStream) -> Result<PutReadyMsg> {
    let tag = read_exact(stream, 1)?[0];
    if tag != MSG_PUT_READY {
        return Err(anyhow!("expected MSG_PUT_READY, got {}", tag));
    }
    let body = read_exact(stream, 1 + 8 + 4)?;
    let ok = body[0] != 0;
    let dst_addr = u64::from_le_bytes(body[1..9].try_into().unwrap());
    let dst_rkey = u32::from_le_bytes(body[9..13].try_into().unwrap());
    Ok(PutReadyMsg {
        ok,
        dst_addr,
        dst_rkey,
    })
}

/// Put Commit (client → server): the client's RDMA WRITE has been poll-completed; ask server to flush to disk
/// wire: tag(1)  (no body; server already remembers dst_addr/size)
pub fn send_put_commit(stream: &mut TcpStream) -> Result<()> {
    stream
        .write_all(&[MSG_PUT_COMMIT])
        .map_err(|e| anyhow!("tcp write put_commit failed: {}", e))?;
    stream.flush().ok();
    Ok(())
}

/// Final Put response (server → client): write to disk complete
/// wire: tag(1) + ok(1)
pub struct PutRespMsg {
    pub ok: bool,
}

pub fn send_put_resp(stream: &mut TcpStream, msg: &PutRespMsg) -> Result<()> {
    let mut frame = [0u8; 2];
    frame[0] = MSG_PUT_RESP;
    frame[1] = if msg.ok { 1 } else { 0 };
    stream
        .write_all(&frame)
        .map_err(|e| anyhow!("tcp write put_resp failed: {}", e))?;
    stream.flush().ok();
    Ok(())
}

pub fn recv_put_resp(stream: &mut TcpStream) -> Result<PutRespMsg> {
    let tag = read_exact(stream, 1)?[0];
    if tag != MSG_PUT_RESP {
        return Err(anyhow!("expected MSG_PUT_RESP, got {}", tag));
    }
    let body = read_exact(stream, 1)?;
    Ok(PutRespMsg { ok: body[0] != 0 })
}

// Make the Rust compiler accept GID appearing in the public API
#[allow(dead_code)]
fn _assert_gid_used(_g: ibv_gid) {}
