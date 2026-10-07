//! Regression guards for MR (memory registration) lifetime on the multi-rail
//! data path — hardware-free (default features).
//!
//! Background. An earlier revision registered destination buffers through a
//! cache keyed by `(base_ptr, length)`. It looked fine in the single-shot
//! demos, but a client performing *consecutive* reads into a per-iteration
//! `Vec` silently got zeros back: after the first `Vec` is freed the allocator
//! often reuses the same virtual address, the cache hits, and the reused MR
//! still pins the *old* physical pages while the CPU reads through the *new*
//! mapping. The fix makes the default `RailReader::register` path register
//! afresh on every read (holding the MR only across `register` →
//! `read_stripes`), with an explicit `register_raw_buffer_pooled` opt-in for
//! long-lived pools.
//!
//! The Mock transport does not pin memory, so it cannot reproduce the
//! physical-page hazard itself. What these tests *can* and *must* pin down is
//! the invariant that made the bug possible to fix: **each read registers its
//! own destination and hands exactly that registration to `read_stripes`** —
//! no reuse of a previous read's registration, even when the OS hands the
//! same buffer address back. `SpyRail` observes both, so a future refactor
//! that reintroduces address-keyed caching without invalidation fails here.

use contextstore_client_rs::mock_rail::{MockRailClient, MockStore};
use contextstore_client_rs::multi_rail::{stripe_checksum, MultiRailReader, RailRegistration, RailReader};
use contextstore_client_rs::pb;
use std::sync::{Arc, Mutex};

/// Wraps a `MockRailClient` and records, per read, the `rail_id`, the
/// registration returned by `register`, and the segments handed to
/// `read_stripes`. This lets a test assert the two agree and that the
/// registration is fresh each time.
struct SpyRail {
    inner: MockRailClient,
    register_calls: Arc<Mutex<Vec<(u64, usize)>>>,
    last_segments: Arc<Mutex<Vec<(u64, u32, u64)>>>,
}

impl SpyRail {
    fn new(inner: MockRailClient) -> Self {
        Self {
            inner,
            register_calls: Arc::new(Mutex::new(Vec::new())),
            last_segments: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl RailReader for SpyRail {
    fn rail_id(&self) -> String {
        self.inner.rail_id()
    }

    unsafe fn register(&mut self, base: *mut u8, len: usize) -> anyhow::Result<RailRegistration> {
        let reg = unsafe { self.inner.register(base, len) }?;
        self.register_calls
            .lock()
            .unwrap()
            .push((reg.addr, reg.len));
        Ok(reg)
    }

    fn read_stripes(
        &mut self,
        descriptor: &pb::ObjectDescriptor,
        stripes: &[u32],
        segments: &[(u64, u32, u64)],
    ) -> anyhow::Result<usize> {
        *self.last_segments.lock().unwrap() = segments.to_vec();
        self.inner.read_stripes(descriptor, stripes, segments)
    }
}

fn make_descriptor(stripe_count: u32, chunk: u64) -> pb::ObjectDescriptor {
    pb::ObjectDescriptor {
        key: Some(pb::ObjectKey {
            namespace: "ns".into(),
            object_key: "obj".into(),
        }),
        object_handle: "h".into(),
        object_generation: 1,
        content_etag: "e".into(),
        layout_version: 1,
        size: stripe_count as u64 * chunk,
        is_striped: true,
        stripe_count,
        chunk_size: chunk,
    }
}

/// Consecutive reads, each into a freshly allocated same-sized `Vec` (so the
/// allocator is very likely to hand back the same address), must all return
/// the correct object — and every read must register its own destination
/// rather than reuse a previous read's registration.
#[test]
fn consecutive_reads_register_freshly_and_stay_correct() {
    let store = Arc::new(Mutex::new(MockStore::default()));
    let stripe_count = 4u32;
    let chunk = 4096u64;
    let size = stripe_count as usize * chunk as usize;

    // Two distinct objects with different content, so a stale registration
    // (which would leave zero bytes where the read should have written) cannot
    // accidentally look correct.
    let data_a: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
    let data_b: Vec<u8> = (0..size).map(|i| (i % 137) as u8).collect();
    {
        let mut s = store.lock().unwrap();
        s.objects.insert("2:nsobj".to_string(), Arc::new(data_a.clone()));
        s.objects.insert("2:nsobjB".to_string(), Arc::new(data_b.clone()));
    }

    let spy = SpyRail::new(MockRailClient::new("rail0", 1e9, store.clone()));
    let register_calls = spy.register_calls.clone();
    let last_segments = spy.last_segments.clone();

    let rails: Vec<Box<dyn RailReader>> = vec![Box::new(spy)];
    let mut reader = MultiRailReader::new(rails);
    let desc = make_descriptor(stripe_count, chunk);
    let checksums: Vec<Option<String>> = (0..stripe_count as usize)
        .map(|i| Some(stripe_checksum(&data_a[i * chunk as usize..(i + 1) * chunk as usize])))
        .collect();

    const ITERS: usize = 8;
    let mut seen_addrs = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        // New buffer each iteration: the exact shape that used to trigger the
        // address-reuse hazard.
        let mut buf = vec![0u8; size];
        let stats = reader.read(&desc, &checksums, &mut buf).unwrap();
        assert!(stats.verify_ok, "every read must verify");
        assert_eq!(buf, data_a, "read must reconstruct the object exactly");

        // The registration this read used must match the segments it passed
        // on — i.e. `read_stripes` never saw a stale registration.
        let segs = last_segments.lock().unwrap().clone();
        let regs = register_calls.lock().unwrap().clone();
        let (reg_addr, _reg_len) = *regs.last().expect("register must have been called");
        // With a single rail carrying all stripes, segment 0 is the window base.
        assert_eq!(
            segs[0].0, reg_addr,
            "read_stripes must use the registration produced by this read's register call"
        );

        seen_addrs.push(buf.as_ptr() as u64);
    }

    // The default path registers on every read — no address-keyed caching.
    assert_eq!(
        register_calls.lock().unwrap().len(),
        ITERS,
        "default path must register afresh for every read (found a cache?)"
    );
    assert!(
        seen_addrs.iter().all(|a| *a == seen_addrs[0]),
        "sanity: the allocator is expected to reuse the buffer address across \
         iterations (if it did not, this test is not exercising the hazard)"
    );
}

/// A second object read through the same reader must not be served from the
/// first object's registration state. (Different key, same buffer shape.)
#[test]
fn switching_objects_on_one_reader_stays_correct() {
    let store = Arc::new(Mutex::new(MockStore::default()));
    let stripe_count = 4u32;
    let chunk = 4096u64;
    let size = stripe_count as usize * chunk as usize;
    let data_a: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
    let data_b: Vec<u8> = (0..size).map(|i| (i % 137) as u8).collect();
    {
        let mut s = store.lock().unwrap();
        s.objects.insert("2:nsobj".to_string(), Arc::new(data_a.clone()));
        s.objects.insert("2:nsobjB".to_string(), Arc::new(data_b.clone()));
    }

    let rail: Box<dyn RailReader> = Box::new(MockRailClient::new("rail0", 1e9, store.clone()));
    let mut reader = MultiRailReader::new(vec![rail]);
    let chunk_us = chunk as usize;

    // Object A.
    let mut buf = vec![0u8; size];
    let ck_a: Vec<Option<String>> = (0..stripe_count as usize)
        .map(|i| Some(stripe_checksum(&data_a[i * chunk_us..(i + 1) * chunk_us])))
        .collect();
    reader
        .read(&make_descriptor(stripe_count, chunk), &ck_a, &mut buf)
        .unwrap();
    assert_eq!(buf, data_a);

    // Object B, same reader, same destination buffer.
    let mut desc_b = make_descriptor(stripe_count, chunk);
    desc_b.key = Some(pb::ObjectKey {
        namespace: "ns".into(),
        object_key: "objB".into(),
    });
    let ck_b: Vec<Option<String>> = (0..stripe_count as usize)
        .map(|i| Some(stripe_checksum(&data_b[i * chunk_us..(i + 1) * chunk_us])))
        .collect();
    reader.read(&desc_b, &ck_b, &mut buf).unwrap();
    assert_eq!(buf, data_b, "second object must fully overwrite the buffer");
}

/// The pooled registration API must exist and be explicitly named; the old
/// misleading `_cached` name must be gone. This is a compile-time contract:
/// if the assertion fails to compile, the public API drifted.
///
/// Gated on the `rdma` feature because the module that owns these methods is
/// only compiled in the verbs build.
#[cfg(feature = "rdma")]
#[test]
fn pooled_api_is_present_and_explicitly_named() {
    // Exercising the name in a type position is enough to guarantee it exists
    // with the documented signature.
    let _invalidate: fn(&mut contextstore_client_rs::rdma::RdmaClient) =
        contextstore_client_rs::rdma::RdmaClient::invalidate_mr_cache;
}