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
/// Immutable RDMA PUT. It uses the same body as `MSG_PUT_REQ`, but the server
/// commits metadata with `SET NX` and reports an existing object distinctly.
pub const MSG_PUT_IF_ABSENT_REQ: u8 = 9;
/// PUT with options (currently: TTL). Body = the classic PutReq body plus a
/// trailing options block; see `PutReqMsg::ttl_seconds`. A separate tag keeps
/// old servers rejecting it loudly (unknown tag) instead of misparsing, and
/// old clients keep sending tag 4/9 which new servers still accept.
pub const MSG_PUT_WITH_OPTIONS_REQ: u8 = 10;
/// Immutable PUT with options. `MSG_PUT_WITH_OPTIONS_REQ`'s SET-NX variant,
/// mirroring the tag-4 / tag-9 pairing.
pub const MSG_PUT_IF_ABSENT_WITH_OPTIONS_REQ: u8 = 11;
/// Descriptor GET restricted to a caller-selected stripe subset. Body = the
/// classic descriptor-GET body + stripe_count(u16) + stripe_index(u32)*.
/// Each served stripe is RDMA-written at `dst_addr + index * chunk_size`,
/// so N nodes can fill disjoint regions of one destination buffer in
/// parallel — the multi-endpoint direct-read building block.
pub const MSG_GET_DESCRIPTOR_STRIPES_REQ: u8 = 12;
/// Stripe-subset descriptor GET with a scatter destination list (SGE): the
/// tag-12 body followed by seg_count(u16) + [seg_addr(8) seg_rkey(4)
/// seg_len(8)]*. Segments describe the object's byte range in order and must
/// cover at least the served stripes; the server maps each stripe's object
/// offset onto the segment list and issues one RDMA WRITE per (stripe-chunk x
/// segment) overlap. Eliminates the client-side staging buffer + scatter
/// memcpy for readers whose destination is not contiguous (vLLM KV blocks).
pub const MSG_GET_DESCRIPTOR_STRIPES_SGE_REQ: u8 = 15;
/// Stripe-subset PUT: push a set of stripes to the node that owns them.
/// Flow mirrors the whole-object PUT (req -> ready -> RDMA WRITE -> commit ->
/// result), but the slab region carries the listed stripes back-to-back and
/// the server pwrites each as a placement chunk, answering with per-stripe
/// storage handles for the coordinator commit.
pub const MSG_PUT_STRIPES_REQ: u8 = 13;
/// Server -> client: per-stripe locations after a stripe-subset PUT.
pub const MSG_PUT_STRIPES_RESP: u8 = 14;
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
    /// Stripe indices to serve. Empty = whole object (legacy tag 8).
    /// Non-empty = tag 12; the server reads only these stripes and writes
    /// each at `dst_addr + index * chunk_size`.
    pub stripes: Vec<u32>,
    /// Non-empty = tag 15; scatter destination segments (addr, rkey, len)
    /// covering the object byte range in order. Overrides dst_addr/dst_rkey.
    pub dst_segments: Vec<(u64, u32, u64)>,
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
    frame.push(if !msg.dst_segments.is_empty() {
        MSG_GET_DESCRIPTOR_STRIPES_SGE_REQ
    } else if msg.stripes.is_empty() {
        MSG_GET_DESCRIPTOR_REQ
    } else {
        MSG_GET_DESCRIPTOR_STRIPES_REQ
    });
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
    if !msg.stripes.is_empty() || !msg.dst_segments.is_empty() {
        // tag-15 always carries a stripe list (possibly empty = all stripes
        // is NOT supported; callers must enumerate) followed by the segment
        // list; tag-12 carries the stripe list only.
        let count = u16::try_from(msg.stripes.len())
            .map_err(|_| anyhow!("too many stripes: {}", msg.stripes.len()))?;
        frame.extend_from_slice(&count.to_le_bytes());
        for idx in &msg.stripes {
            frame.extend_from_slice(&idx.to_le_bytes());
        }
    }
    if !msg.dst_segments.is_empty() {
        let count = u16::try_from(msg.dst_segments.len())
            .map_err(|_| anyhow!("too many dst segments: {}", msg.dst_segments.len()))?;
        frame.extend_from_slice(&count.to_le_bytes());
        for (addr, rkey, len) in &msg.dst_segments {
            frame.extend_from_slice(&addr.to_le_bytes());
            frame.extend_from_slice(&rkey.to_le_bytes());
            frame.extend_from_slice(&len.to_le_bytes());
        }
    }
    stream
        .write_all(&frame)
        .map_err(|e| anyhow!("tcp write descriptor_get_req failed: {}", e))?;
    stream.flush().ok();
    Ok(())
}

/// Receive DescriptorGetReq body (the tag has already been consumed by the
/// caller). `with_stripes` selects the tag-12 body carrying a stripe list.
pub fn recv_descriptor_get_req_body(
    stream: &mut TcpStream,
    with_stripes: bool,
) -> Result<DescriptorGetReqMsg> {
    recv_descriptor_get_req_body_ext(stream, with_stripes, false)
}

/// `with_segments` selects the tag-15 body: the tag-12 body followed by a
/// scatter destination segment list.
pub fn recv_descriptor_get_req_body_ext(
    stream: &mut TcpStream,
    with_stripes: bool,
    with_segments: bool,
) -> Result<DescriptorGetReqMsg> {
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
    let stripes = if with_stripes {
        let count_b = read_exact(stream, 2)?;
        let count = u16::from_le_bytes([count_b[0], count_b[1]]) as usize;
        let mut stripes = Vec::with_capacity(count);
        for _ in 0..count {
            let idx_b = read_exact(stream, 4)?;
            stripes.push(u32::from_le_bytes(idx_b.try_into().unwrap()));
        }
        stripes
    } else {
        Vec::new()
    };
    let dst_segments = if with_segments {
        let count_b = read_exact(stream, 2)?;
        let count = u16::from_le_bytes([count_b[0], count_b[1]]) as usize;
        if count == 0 {
            return Err(anyhow!("tag-15 GET carries an empty segment list"));
        }
        let mut segs = Vec::with_capacity(count);
        for _ in 0..count {
            let addr_b = read_exact(stream, 8)?;
            let rkey_b = read_exact(stream, 4)?;
            let len_b = read_exact(stream, 8)?;
            segs.push((
                u64::from_le_bytes(addr_b.try_into().unwrap()),
                u32::from_le_bytes(rkey_b.try_into().unwrap()),
                u64::from_le_bytes(len_b.try_into().unwrap()),
            ));
        }
        segs
    } else {
        Vec::new()
    };
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
        stripes,
        dst_segments,
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
/// wire (tag 4/9):   tag(1) + key_len(2) + key + size(8)
/// wire (tag 10/11): tag(1) + key_len(2) + key + size(8) + ttl_seconds(8)
pub struct PutReqMsg {
    pub key: String,
    pub size: u64,
    /// Object lifetime in seconds; 0 = no expiry. Only carried by the
    /// with-options tags — the legacy tags imply 0.
    pub ttl_seconds: i64,
}

pub fn send_put_req(stream: &mut TcpStream, msg: &PutReqMsg) -> Result<()> {
    let key_bytes = msg.key.as_bytes();
    if key_bytes.len() > 65535 {
        return Err(anyhow!("key too long: {}", key_bytes.len()));
    }
    let with_options = msg.ttl_seconds != 0;
    let mut frame = Vec::with_capacity(1 + 2 + key_bytes.len() + 8 + 8);
    frame.push(if with_options {
        MSG_PUT_WITH_OPTIONS_REQ
    } else {
        MSG_PUT_REQ
    });
    frame.extend_from_slice(&(key_bytes.len() as u16).to_le_bytes());
    frame.extend_from_slice(key_bytes);
    frame.extend_from_slice(&msg.size.to_le_bytes());
    if with_options {
        frame.extend_from_slice(&msg.ttl_seconds.to_le_bytes());
    }
    stream
        .write_all(&frame)
        .map_err(|e| anyhow!("tcp write put_req failed: {}", e))?;
    stream.flush().ok();
    Ok(())
}

/// Receive PutReq body (the tag has already been consumed by the caller).
/// `with_options` selects the extended body carrying ttl_seconds.
pub fn recv_put_req_body(stream: &mut TcpStream, with_options: bool) -> Result<PutReqMsg> {
    let key_len_bytes = read_exact(stream, 2)?;
    let key_len = u16::from_le_bytes([key_len_bytes[0], key_len_bytes[1]]) as usize;
    let key_bytes = read_exact(stream, key_len)?;
    let key = String::from_utf8(key_bytes).map_err(|e| anyhow!("key utf8: {}", e))?;
    let size_b = read_exact(stream, 8)?;
    let size = u64::from_le_bytes(size_b.try_into().unwrap());
    let ttl_seconds = if with_options {
        let ttl_b = read_exact(stream, 8)?;
        i64::from_le_bytes(ttl_b.try_into().unwrap())
    } else {
        0
    };
    Ok(PutReqMsg {
        key,
        size,
        ttl_seconds,
    })
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

/// `PUT_IF_ABSENT` completion codes. Legacy PUT only uses `STORED` and
/// `FAILED`, so existing clients remain wire-compatible.
pub const PUT_RESULT_FAILED: u8 = 0;
pub const PUT_RESULT_STORED: u8 = 1;
pub const PUT_RESULT_EXISTS: u8 = 2;

pub fn send_put_resp(stream: &mut TcpStream, msg: &PutRespMsg) -> Result<()> {
    let mut frame = [0u8; 2];
    frame[0] = MSG_PUT_RESP;
    frame[1] = if msg.ok {
        PUT_RESULT_STORED
    } else {
        PUT_RESULT_FAILED
    };
    stream
        .write_all(&frame)
        .map_err(|e| anyhow!("tcp write put_resp failed: {}", e))?;
    stream.flush().ok();
    Ok(())
}

/// Send an extended immutable PUT result. Only clients that initiated
/// `MSG_PUT_IF_ABSENT_REQ` interpret `PUT_RESULT_EXISTS`.
pub fn send_put_result(stream: &mut TcpStream, result: u8) -> Result<()> {
    stream
        .write_all(&[MSG_PUT_RESP, result])
        .map_err(|e| anyhow!("tcp write put result failed: {}", e))?;
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

// ===================== Stripe-subset PUT (multi-endpoint writes) =====================

/// Stripe-subset PUT request (client -> server).
/// wire: tag(1) + key_len(2) + key + handle_len(2) + object_handle +
///       generation(8) + layout_version(8) + chunk_size(8) + total_size(8) +
///       stripe_count(2) + [stripe_index(4) + stripe_len(8)]*
/// The client then RDMA-WRITEs the stripes back-to-back (request order) into
/// the slab region granted by PutReady, and sends MSG_PUT_COMMIT.
pub struct PutStripesReqMsg {
    pub key: String,
    pub object_handle: String,
    pub object_generation: u64,
    pub layout_version: u64,
    pub chunk_size: u64,
    pub total_size: u64,
    pub stripes: Vec<(u32, u64)>,
}

pub fn send_put_stripes_req(stream: &mut TcpStream, msg: &PutStripesReqMsg) -> Result<()> {
    let mut frame = Vec::with_capacity(64 + msg.key.len() + msg.object_handle.len() + msg.stripes.len() * 12);
    frame.push(MSG_PUT_STRIPES_REQ);
    write_string_u16(&mut frame, &msg.key, "key")?;
    write_string_u16(&mut frame, &msg.object_handle, "object_handle")?;
    frame.extend_from_slice(&msg.object_generation.to_le_bytes());
    frame.extend_from_slice(&msg.layout_version.to_le_bytes());
    frame.extend_from_slice(&msg.chunk_size.to_le_bytes());
    frame.extend_from_slice(&msg.total_size.to_le_bytes());
    let count = u16::try_from(msg.stripes.len())
        .map_err(|_| anyhow!("too many stripes: {}", msg.stripes.len()))?;
    frame.extend_from_slice(&count.to_le_bytes());
    for (idx, len) in &msg.stripes {
        frame.extend_from_slice(&idx.to_le_bytes());
        frame.extend_from_slice(&len.to_le_bytes());
    }
    stream
        .write_all(&frame)
        .map_err(|e| anyhow!("tcp write put_stripes_req failed: {}", e))?;
    stream.flush().ok();
    Ok(())
}

/// Receive PutStripesReq body (tag already consumed).
pub fn recv_put_stripes_req_body(stream: &mut TcpStream) -> Result<PutStripesReqMsg> {
    let key = read_string_u16(stream, "key")?;
    let object_handle = read_string_u16(stream, "object_handle")?;
    let gen_b = read_exact(stream, 8)?;
    let object_generation = u64::from_le_bytes(gen_b.try_into().unwrap());
    let layout_b = read_exact(stream, 8)?;
    let layout_version = u64::from_le_bytes(layout_b.try_into().unwrap());
    let cs_b = read_exact(stream, 8)?;
    let chunk_size = u64::from_le_bytes(cs_b.try_into().unwrap());
    let ts_b = read_exact(stream, 8)?;
    let total_size = u64::from_le_bytes(ts_b.try_into().unwrap());
    let count_b = read_exact(stream, 2)?;
    let count = u16::from_le_bytes([count_b[0], count_b[1]]) as usize;
    let mut stripes = Vec::with_capacity(count);
    for _ in 0..count {
        let idx_b = read_exact(stream, 4)?;
        let idx = u32::from_le_bytes(idx_b.try_into().unwrap());
        let len_b = read_exact(stream, 8)?;
        let len = u64::from_le_bytes(len_b.try_into().unwrap());
        stripes.push((idx, len));
    }
    Ok(PutStripesReqMsg {
        key,
        object_handle,
        object_generation,
        layout_version,
        chunk_size,
        total_size,
        stripes,
    })
}

/// Per-stripe write result (server -> client) after a stripe-subset PUT.
/// wire: tag(1) + ok(1) + stripe_count(2) +
///       [stripe_index(4) + device_id(4) + handle_len(2) + storage_handle + checksum_len(2) + checksum]*
pub struct PutStripesRespMsg {
    pub ok: bool,
    pub stripes: Vec<PutStripeLocation>,
}

pub struct PutStripeLocation {
    pub stripe_index: u32,
    pub device_id: u32,
    pub storage_handle: String,
    pub checksum: String,
}

pub fn send_put_stripes_resp(stream: &mut TcpStream, msg: &PutStripesRespMsg) -> Result<()> {
    let mut frame = Vec::with_capacity(8 + msg.stripes.len() * 64);
    frame.push(MSG_PUT_STRIPES_RESP);
    frame.push(u8::from(msg.ok));
    let count = u16::try_from(msg.stripes.len())
        .map_err(|_| anyhow!("too many stripe results: {}", msg.stripes.len()))?;
    frame.extend_from_slice(&count.to_le_bytes());
    for loc in &msg.stripes {
        frame.extend_from_slice(&loc.stripe_index.to_le_bytes());
        frame.extend_from_slice(&loc.device_id.to_le_bytes());
        write_string_u16(&mut frame, &loc.storage_handle, "storage_handle")?;
        write_string_u16(&mut frame, &loc.checksum, "checksum")?;
    }
    stream
        .write_all(&frame)
        .map_err(|e| anyhow!("tcp write put_stripes_resp failed: {}", e))?;
    stream.flush().ok();
    Ok(())
}

pub fn recv_put_stripes_resp(stream: &mut TcpStream) -> Result<PutStripesRespMsg> {
    let tag = read_exact(stream, 1)?[0];
    if tag != MSG_PUT_STRIPES_RESP {
        return Err(anyhow!("expected MSG_PUT_STRIPES_RESP, got {}", tag));
    }
    let ok = read_exact(stream, 1)?[0] != 0;
    let count_b = read_exact(stream, 2)?;
    let count = u16::from_le_bytes([count_b[0], count_b[1]]) as usize;
    let mut stripes = Vec::with_capacity(count);
    for _ in 0..count {
        let idx_b = read_exact(stream, 4)?;
        let stripe_index = u32::from_le_bytes(idx_b.try_into().unwrap());
        let dev_b = read_exact(stream, 4)?;
        let device_id = u32::from_le_bytes(dev_b.try_into().unwrap());
        let storage_handle = read_string_u16(stream, "storage_handle")?;
        let checksum = read_string_u16(stream, "checksum")?;
        stripes.push(PutStripeLocation {
            stripe_index,
            device_id,
            storage_handle,
            checksum,
        });
    }
    Ok(PutStripesRespMsg { ok, stripes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use std::net::{TcpListener, TcpStream};

    /// Round-trip a PutReq frame through a loopback socket with the given
    /// send-side message and receive-side options flag.
    fn roundtrip(msg: &PutReqMsg, with_options: bool) -> PutReqMsg {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        send_put_req(&mut client, msg).unwrap();
        client.flush().unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let tag = read_exact(&mut server, 1).unwrap()[0];
        let expected_tag = if with_options {
            MSG_PUT_WITH_OPTIONS_REQ
        } else {
            MSG_PUT_REQ
        };
        assert_eq!(tag, expected_tag, "tag selects the legacy/options body");
        recv_put_req_body(&mut server, with_options).unwrap()
    }

    #[test]
    fn put_req_without_ttl_uses_legacy_tag_and_body() {
        let msg = PutReqMsg {
            key: "ns/obj".into(),
            size: 4096,
            ttl_seconds: 0,
        };
        let parsed = roundtrip(&msg, false);
        assert_eq!(parsed.key, "ns/obj");
        assert_eq!(parsed.size, 4096);
        assert_eq!(parsed.ttl_seconds, 0);
    }

    #[test]
    fn descriptor_get_req_stripe_subset_round_trips() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let msg = DescriptorGetReqMsg {
            key: "ns/obj".into(),
            object_handle: "h1".into(),
            object_generation: 7,
            content_etag: "etag".into(),
            layout_version: 1,
            size: 1 << 28,
            is_striped: true,
            stripe_count: 8,
            chunk_size: 1 << 25,
            dst_addr: 0xdead_beef,
            dst_rkey: 42,
            max_size: 1 << 28,
            stripes: vec![1, 3, 5],
            dst_segments: Vec::new(),
        };
        send_descriptor_get_req(&mut client, &msg).unwrap();
        client.flush().unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let tag = read_exact(&mut server, 1).unwrap()[0];
        assert_eq!(tag, MSG_GET_DESCRIPTOR_STRIPES_REQ);
        let parsed = recv_descriptor_get_req_body(&mut server, true).unwrap();
        assert_eq!(parsed.stripes, vec![1, 3, 5]);
        assert_eq!(parsed.chunk_size, 1 << 25);
        assert_eq!(parsed.dst_addr, 0xdead_beef);
    }

    #[test]
    fn descriptor_get_req_empty_stripes_keeps_legacy_tag() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let msg = DescriptorGetReqMsg {
            key: "ns/obj".into(),
            object_handle: String::new(),
            object_generation: 1,
            content_etag: String::new(),
            layout_version: 1,
            size: 64,
            is_striped: false,
            stripe_count: 0,
            chunk_size: 0,
            dst_addr: 1,
            dst_rkey: 2,
            max_size: 64,
            stripes: Vec::new(),
            dst_segments: Vec::new(),
        };
        send_descriptor_get_req(&mut client, &msg).unwrap();
        client.flush().unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let tag = read_exact(&mut server, 1).unwrap()[0];
        assert_eq!(tag, MSG_GET_DESCRIPTOR_REQ, "no stripes = legacy wire form");
        let parsed = recv_descriptor_get_req_body(&mut server, false).unwrap();
        assert!(parsed.stripes.is_empty());
    }

    #[test]
    fn put_stripes_req_round_trips() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let msg = PutStripesReqMsg {
            key: "ns/obj".into(),
            object_handle: "ctxobj-v1-x".into(),
            object_generation: 5,
            layout_version: 1,
            chunk_size: 1 << 26,
            total_size: 1 << 28,
            stripes: vec![(0, 1 << 26), (2, 1 << 26), (4, 12345)],
        };
        send_put_stripes_req(&mut client, &msg).unwrap();
        client.flush().unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let tag = read_exact(&mut server, 1).unwrap()[0];
        assert_eq!(tag, MSG_PUT_STRIPES_REQ);
        let parsed = recv_put_stripes_req_body(&mut server).unwrap();
        assert_eq!(parsed.stripes, vec![(0, 1 << 26), (2, 1 << 26), (4, 12345)]);
        assert_eq!(parsed.object_generation, 5);
        assert_eq!(parsed.object_handle, "ctxobj-v1-x");
    }

    #[test]
    fn put_stripes_resp_round_trips() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let msg = PutStripesRespMsg {
            ok: true,
            stripes: vec![
                PutStripeLocation {
                    stripe_index: 0,
                    device_id: 1,
                    storage_handle: "/dev0/a.chunk0.bin".into(),
                    checksum: "abcd".into(),
                },
                PutStripeLocation {
                    stripe_index: 2,
                    device_id: 0,
                    storage_handle: "/dev1/a.chunk2.bin".into(),
                    checksum: String::new(),
                },
            ],
        };
        send_put_stripes_resp(&mut client, &msg).unwrap();
        client.flush().unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let parsed = recv_put_stripes_resp(&mut server).unwrap();
        assert!(parsed.ok);
        assert_eq!(parsed.stripes.len(), 2);
        assert_eq!(parsed.stripes[0].storage_handle, "/dev0/a.chunk0.bin");
        assert_eq!(parsed.stripes[1].stripe_index, 2);
        assert_eq!(parsed.stripes[1].checksum, "");
    }

    #[test]
    fn put_req_with_ttl_round_trips_through_options_body() {
        let msg = PutReqMsg {
            key: "ns/obj".into(),
            size: 1 << 30,
            ttl_seconds: 90_000,
        };
        let parsed = roundtrip(&msg, true);
        assert_eq!(parsed.key, "ns/obj");
        assert_eq!(parsed.size, 1 << 30);
        assert_eq!(parsed.ttl_seconds, 90_000);
    }
}
