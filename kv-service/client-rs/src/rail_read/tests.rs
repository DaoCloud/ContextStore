use super::*;
use crate::pb;
use std::sync::Mutex;

struct MockTransport {
    object: Vec<u8>,
    calls: Mutex<Vec<String>>,
    failing_rail: Option<String>,
}

impl MockTransport {
    fn new(object: Vec<u8>) -> Self {
        Self {
            object,
            calls: Mutex::new(Vec::new()),
            failing_rail: None,
        }
    }
}

impl RailTransport for MockTransport {
    fn fetch(
        &self,
        route: &RailRoute,
        task: &RailTask,
        _descriptor: &pb::ObjectDescriptor,
        _timeout: std::time::Duration,
    ) -> Result<Vec<u8>, RailReadError> {
        self.calls
            .lock()
            .unwrap()
            .push(route.connection.endpoint.clone());
        let mut packed = vec![0u8; task.packed_len];
        for stripe in &task.stripes {
            packed[stripe.packed_offset..stripe.packed_offset + stripe.length].copy_from_slice(
                &self.object[stripe.object_offset..stripe.object_offset + stripe.length],
            );
        }
        if self.failing_rail.as_deref() == Some(&route.id) {
            return Err(RailReadError::Transport("injected disconnect".into()));
        }
        Ok(packed)
    }
}

fn fixture(size: usize, chunk_size: usize) -> (pb::ObjectDescriptor, pb::PlacementDescriptor) {
    let count = size.div_ceil(chunk_size);
    let descriptor = pb::ObjectDescriptor {
        key: Some(pb::ObjectKey {
            namespace: "test".into(),
            object_key: "obj".into(),
        }),
        object_handle: "handle".into(),
        object_generation: 1,
        content_etag: "etag".into(),
        layout_version: 1,
        size: size as u64,
        is_striped: true,
        stripe_count: count as u32,
        chunk_size: chunk_size as u64,
    };
    let placement = pb::PlacementDescriptor {
        key: descriptor.key.clone(),
        chunks: (0..count)
            .map(|index| pb::PlacementChunk {
                stripe_index: index as u32,
                node_id: "node-a".into(),
                grpc_endpoint: "10.0.0.1:50051".into(),
                rdma_endpoint: "10.0.0.1:50053".into(),
                device_id: 0,
                storage_handle: format!("stripe-{index}"),
                offset: (index * chunk_size) as u64,
                length: (size - index * chunk_size).min(chunk_size) as u64,
                checksum: String::new(),
            })
            .collect(),
        ..Default::default()
    };
    (descriptor, placement)
}

fn routes() -> Vec<RailRoute> {
    vec![
        RailRoute::new(
            "rail0",
            "10.0.0.1:50053",
            crate::rdma::RdmaClientConfig::new("10.0.0.1:50053", "mock0"),
        ),
        RailRoute::new(
            "rail1",
            "10.0.0.1:50053",
            crate::rdma::RdmaClientConfig::new("10.0.1.1:50054", "mock1"),
        ),
    ]
}

#[test]
fn two_independent_listeners_restore_one_unmodified_placement() {
    let bytes: Vec<u8> = (0..64).map(|index| index as u8).collect();
    let mock = MockTransport::new(bytes.clone());
    let reader = RailReader::new(routes(), RailLimits::default()).unwrap();
    let (descriptor, placement) = fixture(64, 8);
    let mut destination = vec![0xA5; 64];
    assert_eq!(
        reader
            .read_into_with(&descriptor, &placement, &mut destination, &mock, None)
            .unwrap(),
        64
    );
    assert_eq!(destination, bytes);
    let calls = mock.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert!(calls.contains(&"10.0.0.1:50053".to_string()));
    assert!(calls.contains(&"10.0.1.1:50054".to_string()));
    let snapshots = reader.snapshots();
    assert_eq!(snapshots[0].bytes, 32);
    assert_eq!(snapshots[1].bytes, 32);
    assert!(snapshots.iter().all(|rail| rail.inflight_requests == 0));
    assert!(snapshots.iter().all(|rail| rail.registered_bytes == 0));
    assert_eq!(snapshots[0].peak_inflight_bytes, 32);
    assert!(snapshots.iter().all(|rail| rail.peak_registered_bytes > 0));
}

#[test]
fn partial_rail_failure_leaves_caller_buffer_unchanged() {
    let mut mock = MockTransport::new(vec![0x42; 64]);
    mock.failing_rail = Some("rail1".into());
    let reader = RailReader::new(routes(), RailLimits::default()).unwrap();
    let (descriptor, placement) = fixture(64, 8);
    let mut destination = vec![0xA5; 64];
    assert!(reader
        .read_into_with(&descriptor, &placement, &mut destination, &mock, None)
        .is_err());
    assert_eq!(destination, vec![0xA5; 64]);
    assert_eq!(mock.calls.lock().unwrap().len(), 2);
}

#[test]
fn duplicate_stripe_is_rejected_before_transport() {
    let mock = MockTransport::new(vec![0x42; 64]);
    let reader = RailReader::new(routes(), RailLimits::default()).unwrap();
    let (descriptor, mut placement) = fixture(64, 8);
    placement.chunks[1].stripe_index = 0;
    let mut destination = vec![0xA5; 64];
    assert!(reader
        .read_into_with(&descriptor, &placement, &mut destination, &mock, None)
        .is_err());
    assert!(mock.calls.lock().unwrap().is_empty());
    assert_eq!(destination, vec![0xA5; 64]);
}

#[test]
fn registration_budget_rejects_before_transport() {
    let mock = MockTransport::new(vec![0x42; 64]);
    let limits = RailLimits {
        max_registered_bytes: 16,
        ..RailLimits::default()
    };
    let reader = RailReader::new(routes(), limits).unwrap();
    let (descriptor, placement) = fixture(64, 8);
    let mut destination = vec![0xA5; 64];
    assert!(reader
        .read_into_with(&descriptor, &placement, &mut destination, &mock, None)
        .is_err());
    assert!(mock.calls.lock().unwrap().is_empty());
}

#[test]
fn process_memlock_headroom_caps_registered_bytes_before_dispatch() {
    let budget = effective_registered_budget(4 * 1024 * 1024 * 1024, Some(500_012 * 1024));
    assert_eq!(budget, 500_012 * 1024 - (500_012 * 1024) / 5);
    assert!(budget < 4 * 128 * 1024 * 1024);
    assert_eq!(effective_registered_budget(1024, None), 1024);
}

#[test]
fn per_rail_inflight_limit_counts_concurrent_requests() {
    let (descriptor, placement) = fixture(64, 8);
    let configured = vec![routes()[0].clone()];
    let plan = RailPlan::build(&descriptor, &placement, &configured).unwrap();
    let limits = RailLimits {
        max_active_reads: 2,
        max_inflight_bytes_per_rail: 96,
        ..RailLimits::default()
    };
    let reader = RailReader::new(configured, limits).unwrap();
    let first = reader.reserve(&plan).unwrap();
    assert!(matches!(
        reader.reserve(&plan),
        Err(RailReadError::ResourceExhausted(_))
    ));
    drop(first);
    assert!(reader.reserve(&plan).is_ok());
}

#[test]
fn per_rail_task_limit_counts_concurrent_requests() {
    let (descriptor, placement) = fixture(64, 8);
    let configured = vec![routes()[0].clone()];
    let plan = RailPlan::build(&descriptor, &placement, &configured).unwrap();
    let limits = RailLimits {
        max_active_reads: 2,
        max_active_reads_per_rail: 1,
        ..RailLimits::default()
    };
    let reader = RailReader::new(configured, limits).unwrap();
    let first = reader.reserve(&plan).unwrap();
    assert!(matches!(
        reader.reserve(&plan),
        Err(RailReadError::ResourceExhausted(_))
    ));
    drop(first);
    assert!(reader.reserve(&plan).is_ok());
}

#[test]
fn completed_payload_keeps_staging_and_active_read_reserved_until_drop() {
    let mock = MockTransport::new(vec![0x42; 64]);
    let limits = RailLimits {
        max_active_reads: 1,
        ..RailLimits::default()
    };
    let reader = RailReader::new(routes(), limits).unwrap();
    let (descriptor, placement) = fixture(64, 8);
    let staged = reader
        .read_staged_with(&descriptor, &placement, &mock, None)
        .unwrap();
    assert_eq!(reader.budget.lock().unwrap().active_reads, 1);
    assert!(reader.budget.lock().unwrap().staging_bytes >= 64);
    assert!(reader
        .budget
        .lock()
        .unwrap()
        .rail_active_reads
        .iter()
        .all(|count| *count == 0));
    assert!(reader
        .budget
        .lock()
        .unwrap()
        .rail_inflight_bytes
        .iter()
        .all(|bytes| *bytes == 0));
    assert!(matches!(
        reader.read_staged_with(&descriptor, &placement, &mock, None),
        Err(RailReadError::ResourceExhausted(_))
    ));
    drop(staged);
    assert_eq!(reader.budget.lock().unwrap().active_reads, 0);
}

#[test]
fn cancellation_before_publish_preserves_caller_buffer() {
    let (descriptor, placement) = fixture(64, 8);
    let lookup = crate::ObjectLookup {
        descriptor,
        placement: Some(placement),
    };
    let cancel = RailCancel::default();
    cancel.cancel();
    let mut destination = vec![0xA5; 64];
    assert!(matches!(
        commit_if_unchanged(
            &lookup,
            &lookup,
            &[0x42; 64],
            &mut destination,
            Some(&cancel)
        ),
        Err(RailReadError::Cancelled)
    ));
    assert_eq!(destination, vec![0xA5; 64]);
}

#[test]
fn cancellation_and_publish_have_one_order() {
    use std::sync::mpsc;
    let token = RailCancel::default();
    let (publishing_tx, publishing_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let committing = token.clone();
    let publish = std::thread::spawn(move || {
        committing.publish_if_live(|| {
            publishing_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
    });
    publishing_rx.recv().unwrap();
    let cancelling = token.clone();
    let cancel = std::thread::spawn(move || cancelling.cancel());
    assert!(!token.is_cancelled());
    release_tx.send(()).unwrap();
    publish.join().unwrap().unwrap();
    cancel.join().unwrap();
    assert!(token.is_cancelled());
}

#[test]
fn descriptor_identity_rejects_generation_and_layout_changes() {
    let (original, _) = fixture(64, 8);
    let mut changed = original.clone();
    assert!(same_descriptor_identity(&original, &changed));
    changed.object_generation += 1;
    assert!(!same_descriptor_identity(&original, &changed));
    changed = original.clone();
    changed.layout_version += 1;
    assert!(!same_descriptor_identity(&original, &changed));
}

#[test]
fn two_rails_for_one_node_require_distinct_local_ports() {
    let mut rails = routes();
    rails[1].connection.device = rails[0].connection.device.clone();
    assert!(RailReader::new(rails, RailLimits::default()).is_err());
}

#[test]
fn weighted_scheduler_assigns_more_stripes_to_faster_rail() {
    let mut configured = routes();
    configured[1] = configured[1].clone().with_weight(3);
    let (descriptor, placement) = fixture(80, 8);
    let plan = RailPlan::build(&descriptor, &placement, &configured).unwrap();
    assert_eq!(plan.tasks.len(), 2);
    assert!(plan.tasks[1].packed_len > plan.tasks[0].packed_len);
    assert_eq!(
        plan.tasks.iter().map(|task| task.packed_len).sum::<usize>(),
        80
    );
}

#[cfg(unix)]
#[test]
fn topology_reads_numa_and_pci_address_from_sysfs_shape() {
    let temp = tempfile::tempdir().unwrap();
    let device = temp.path().join("mlx5_0");
    let pci = temp.path().join("0000:03:00.0");
    std::fs::create_dir_all(&device).unwrap();
    std::fs::create_dir_all(&pci).unwrap();
    std::fs::write(pci.join("numa_node"), "1\n").unwrap();
    std::os::unix::fs::symlink(&pci, device.join("device")).unwrap();
    let topology = read_topology_from(temp.path(), "mlx5_0");
    assert_eq!(topology.numa_node, Some(1));
    assert_eq!(topology.pci_bdf.as_deref(), Some("0000:03:00.0"));
}

#[cfg(unix)]
#[test]
fn virtual_soft_roce_device_is_not_reported_as_pci_hardware() {
    let temp = tempfile::tempdir().unwrap();
    let device = temp.path().join("rxe0");
    let virtual_device = temp.path().join("virtual-rxe0");
    std::fs::create_dir_all(&device).unwrap();
    std::fs::create_dir_all(&virtual_device).unwrap();
    std::os::unix::fs::symlink(&virtual_device, device.join("device")).unwrap();
    let topology = read_topology_from(temp.path(), "rxe0");
    assert_eq!(topology.pci_bdf, None);
}

#[test]
fn one_rail_restores_the_same_object_layout() {
    let bytes: Vec<u8> = (0..64).map(|index| index as u8).collect();
    let mock = MockTransport::new(bytes.clone());
    let reader = RailReader::new(vec![routes()[0].clone()], RailLimits::default()).unwrap();
    let (descriptor, placement) = fixture(64, 8);
    let mut destination = vec![0xA5; 64];
    assert_eq!(
        reader
            .read_into_with(&descriptor, &placement, &mut destination, &mock, None)
            .unwrap(),
        64
    );
    assert_eq!(destination, bytes);
    assert_eq!(mock.calls.lock().unwrap().len(), 1);
}

#[test]
fn disabled_second_rail_falls_back_to_first_listener() {
    let bytes = vec![0x42; 64];
    let mock = MockTransport::new(bytes.clone());
    let mut configured = routes();
    configured[1] = configured[1].clone().disabled();
    let reader = RailReader::new(configured, RailLimits::default()).unwrap();
    let (descriptor, placement) = fixture(64, 8);
    let mut destination = vec![0xA5; 64];
    reader
        .read_into_with(&descriptor, &placement, &mut destination, &mock, None)
        .unwrap();
    assert_eq!(destination, bytes);
    assert_eq!(mock.calls.lock().unwrap().as_slice(), ["10.0.0.1:50053"]);
}

#[test]
fn runtime_disable_and_enable_replans_without_changing_placement() {
    let reader = RailReader::new(routes(), RailLimits::default()).unwrap();
    assert!(reader.set_enabled("rail1", false));
    let mock = MockTransport::new(vec![0x42; 64]);
    let (descriptor, placement) = fixture(64, 8);
    let mut destination = vec![0xA5; 64];
    reader
        .read_into_with(&descriptor, &placement, &mut destination, &mock, None)
        .unwrap();
    assert_eq!(mock.calls.lock().unwrap().as_slice(), ["10.0.0.1:50053"]);
    assert!(reader.set_enabled("rail1", true));
    reader
        .read_into_with(&descriptor, &placement, &mut destination, &mock, None)
        .unwrap();
    assert_eq!(mock.calls.lock().unwrap().len(), 3);
}

#[test]
fn failed_rail_cools_down_then_recovers_on_next_request() {
    let limits = RailLimits {
        rail_cooldown: Duration::from_millis(30),
        ..RailLimits::default()
    };
    let reader = RailReader::new(routes(), limits).unwrap();
    let mut failed = MockTransport::new(vec![0x42; 64]);
    failed.failing_rail = Some("rail1".into());
    let (descriptor, placement) = fixture(64, 8);
    let mut destination = vec![0xA5; 64];
    assert!(reader
        .read_into_with(&descriptor, &placement, &mut destination, &failed, None)
        .is_err());
    assert!(!reader.snapshots()[1].healthy);

    let recovered = MockTransport::new(vec![0x42; 64]);
    reader
        .read_into_with(&descriptor, &placement, &mut destination, &recovered, None)
        .unwrap();
    assert_eq!(recovered.calls.lock().unwrap().len(), 1);
    std::thread::sleep(Duration::from_millis(40));
    reader
        .read_into_with(&descriptor, &placement, &mut destination, &recovered, None)
        .unwrap();
    assert_eq!(recovered.calls.lock().unwrap().len(), 3);
}

#[test]
fn stale_descriptor_does_not_mark_healthy_transport_as_down() {
    struct StaleMock(MockTransport);
    impl RailTransport for StaleMock {
        fn fetch(
            &self,
            route: &RailRoute,
            task: &RailTask,
            descriptor: &pb::ObjectDescriptor,
            timeout: Duration,
        ) -> Result<Vec<u8>, RailReadError> {
            if route.id == "rail1" {
                Err(RailReadError::StaleDescriptor)
            } else {
                self.0.fetch(route, task, descriptor, timeout)
            }
        }
    }
    let reader = RailReader::new(routes(), RailLimits::default()).unwrap();
    let (descriptor, placement) = fixture(64, 8);
    let mut destination = vec![0xA5; 64];
    let transport = StaleMock(MockTransport::new(vec![0x42; 64]));
    assert!(matches!(
        reader.read_into_with(&descriptor, &placement, &mut destination, &transport, None),
        Err(RailReadError::StaleDescriptor)
    ));
    assert!(reader.snapshots().iter().all(|rail| rail.healthy));
}

#[test]
fn checksum_failure_keeps_destination_unchanged() {
    let mock = MockTransport::new(vec![0x42; 64]);
    let reader = RailReader::new(routes(), RailLimits::default()).unwrap();
    let (descriptor, mut placement) = fixture(64, 8);
    for chunk in &mut placement.chunks {
        chunk.checksum = "0000000000000000".into();
    }
    let mut destination = vec![0xA5; 64];
    assert!(matches!(
        reader.read_into_with(&descriptor, &placement, &mut destination, &mock, None),
        Err(RailReadError::Checksum { stripe: 0 })
    ));
    assert_eq!(destination, vec![0xA5; 64]);
}

#[test]
fn matching_stripe_checksums_allow_complete_read() {
    let payload: Vec<u8> = (0..64).map(|index| index as u8).collect();
    let mock = MockTransport::new(payload.clone());
    let reader = RailReader::new(routes(), RailLimits::default()).unwrap();
    let (descriptor, mut placement) = fixture(64, 8);
    for chunk in &mut placement.chunks {
        let start = chunk.offset as usize;
        let end = start + chunk.length as usize;
        chunk.checksum = format!("{:016x}", twox_hash::xxh3::hash64(&payload[start..end]));
    }
    let mut destination = vec![0xA5; 64];
    assert_eq!(
        reader
            .read_into_with(&descriptor, &placement, &mut destination, &mock, None)
            .unwrap(),
        64
    );
    assert_eq!(destination, payload);
}

#[test]
fn missing_or_out_of_bounds_stripe_is_rejected_before_dispatch() {
    let mock = MockTransport::new(vec![0x42; 64]);
    let reader = RailReader::new(routes(), RailLimits::default()).unwrap();
    let (descriptor, mut placement) = fixture(64, 8);
    placement.chunks.pop();
    let mut destination = vec![0xA5; 64];
    assert!(reader
        .read_into_with(&descriptor, &placement, &mut destination, &mock, None)
        .is_err());
    let (_, mut placement) = fixture(64, 8);
    placement.chunks[1].offset = 99;
    assert!(reader
        .read_into_with(&descriptor, &placement, &mut destination, &mock, None)
        .is_err());
    assert!(mock.calls.lock().unwrap().is_empty());
}

#[test]
fn placement_for_another_object_is_rejected_before_dispatch() {
    let mock = MockTransport::new(vec![0x42; 64]);
    let reader = RailReader::new(routes(), RailLimits::default()).unwrap();
    let (descriptor, mut placement) = fixture(64, 8);
    placement.key.as_mut().unwrap().object_key = "different".into();
    let mut destination = vec![0xA5; 64];
    assert!(matches!(
        reader.read_into_with(&descriptor, &placement, &mut destination, &mock, None),
        Err(RailReadError::InvalidPlacement(_))
    ));
    assert!(mock.calls.lock().unwrap().is_empty());
}

#[test]
fn partially_populated_checksums_are_rejected_before_dispatch() {
    let mock = MockTransport::new(vec![0x42; 64]);
    let reader = RailReader::new(routes(), RailLimits::default()).unwrap();
    let (descriptor, mut placement) = fixture(64, 8);
    placement.chunks[0].checksum = "0000000000000000".into();
    let mut destination = vec![0xA5; 64];
    assert!(matches!(
        reader.read_into_with(&descriptor, &placement, &mut destination, &mock, None),
        Err(RailReadError::InvalidPlacement(_))
    ));
    assert!(mock.calls.lock().unwrap().is_empty());
}

#[test]
fn cancellation_waits_for_started_mock_transfer_and_preserves_buffer() {
    use std::sync::mpsc;
    struct SlowMock {
        base: MockTransport,
        started: Mutex<Option<mpsc::Sender<()>>>,
    }
    impl RailTransport for SlowMock {
        fn fetch(
            &self,
            route: &RailRoute,
            task: &RailTask,
            descriptor: &pb::ObjectDescriptor,
            timeout: Duration,
        ) -> Result<Vec<u8>, RailReadError> {
            if let Some(sender) = self.started.lock().unwrap().take() {
                let _ = sender.send(());
            }
            std::thread::sleep(Duration::from_millis(60));
            self.base.fetch(route, task, descriptor, timeout)
        }
    }
    let (started_tx, started_rx) = mpsc::channel();
    let mock = SlowMock {
        base: MockTransport::new(vec![0x42; 64]),
        started: Mutex::new(Some(started_tx)),
    };
    let token = RailCancel::default();
    let cancelling = token.clone();
    let canceller = std::thread::spawn(move || {
        started_rx.recv().unwrap();
        cancelling.cancel();
    });
    let reader = RailReader::new(routes(), RailLimits::default()).unwrap();
    let (descriptor, placement) = fixture(64, 8);
    let mut destination = vec![0xA5; 64];
    let started = Instant::now();
    assert!(matches!(
        reader.read_into_with(
            &descriptor,
            &placement,
            &mut destination,
            &mock,
            Some(&token)
        ),
        Err(RailReadError::Cancelled)
    ));
    canceller.join().unwrap();
    assert!(started.elapsed() >= Duration::from_millis(60));
    assert_eq!(destination, vec![0xA5; 64]);
    destination.fill(0x33);
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(destination, vec![0x33; 64]);
}

#[test]
fn sge_map_keeps_every_stripe_inside_its_registered_region() {
    let (descriptor, placement) = fixture(64, 8);
    let plan = RailPlan::build(&descriptor, &placement, &routes()).unwrap();
    let task = &plan.tasks[0];
    let base = 0x1000u64;
    let segments = build_sge_segments(task, &descriptor, base, 7).unwrap();
    assert_eq!(segments.len(), 8);
    assert_eq!(segments.iter().map(|(_, _, len)| len).sum::<u64>(), 64);
    for (address, rkey, length) in &segments {
        assert_eq!(*rkey, 7);
        assert!(*address >= base);
        assert!(*address + *length <= base + (task.packed_len + task.dummy_len) as u64);
    }
    assert_eq!(segments[0].0, base);
    assert_eq!(segments[1].0, base + task.packed_len as u64);
    assert_eq!(segments[2].0, base + 8);
}

#[test]
fn changed_generation_cannot_publish_completed_rail_bytes() {
    let (descriptor, placement) = fixture(64, 8);
    let initial = crate::ObjectLookup {
        descriptor: descriptor.clone(),
        placement: Some(placement.clone()),
    };
    let mut current = initial.clone();
    current.descriptor.object_generation += 1;
    let mut destination = vec![0xA5; 64];
    assert!(matches!(
        commit_if_unchanged(&initial, &current, &[0x42; 64], &mut destination, None),
        Err(RailReadError::VersionChanged)
    ));
    assert_eq!(destination, vec![0xA5; 64]);
    current = initial.clone();
    current.placement.as_mut().unwrap().layout_hash = "moved".into();
    assert!(matches!(
        commit_if_unchanged(&initial, &current, &[0x42; 64], &mut destination, None),
        Err(RailReadError::VersionChanged)
    ));
    assert_eq!(destination, vec![0xA5; 64]);
    current = initial.clone();
    assert_eq!(
        commit_if_unchanged(&initial, &current, &[0x42; 64], &mut destination, None).unwrap(),
        64
    );
    assert_eq!(destination, vec![0x42; 64]);
}

#[test]
fn panicked_worker_releases_inflight_metrics_and_preserves_buffer() {
    struct PanicTransport;
    impl RailTransport for PanicTransport {
        fn fetch(
            &self,
            _route: &RailRoute,
            _task: &RailTask,
            _descriptor: &pb::ObjectDescriptor,
            _timeout: Duration,
        ) -> Result<Vec<u8>, RailReadError> {
            panic!("injected worker panic")
        }
    }
    let reader = RailReader::new(vec![routes()[0].clone()], RailLimits::default()).unwrap();
    let (descriptor, placement) = fixture(64, 8);
    let mut destination = vec![0xA5; 64];
    assert!(matches!(
        reader.read_into_with(
            &descriptor,
            &placement,
            &mut destination,
            &PanicTransport,
            None
        ),
        Err(RailReadError::WorkerPanic)
    ));
    assert_eq!(reader.snapshots()[0].inflight_bytes, 0);
    assert_eq!(destination, vec![0xA5; 64]);
}

#[test]
fn active_read_limit_rejects_concurrent_second_request() {
    use std::sync::mpsc;
    struct HeldTransport {
        object: MockTransport,
        started: Mutex<Option<mpsc::Sender<()>>>,
        release: Mutex<mpsc::Receiver<()>>,
    }
    impl RailTransport for HeldTransport {
        fn fetch(
            &self,
            route: &RailRoute,
            task: &RailTask,
            descriptor: &pb::ObjectDescriptor,
            timeout: Duration,
        ) -> Result<Vec<u8>, RailReadError> {
            if let Some(sender) = self.started.lock().unwrap().take() {
                let _ = sender.send(());
            }
            self.release.lock().unwrap().recv().unwrap();
            self.object.fetch(route, task, descriptor, timeout)
        }
    }
    let limits = RailLimits {
        max_active_reads: 1,
        ..RailLimits::default()
    };
    let reader = Arc::new(RailReader::new(routes(), limits).unwrap());
    let (descriptor, placement) = fixture(64, 8);
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let held = HeldTransport {
        object: MockTransport::new(vec![0x42; 64]),
        started: Mutex::new(Some(started_tx)),
        release: Mutex::new(release_rx),
    };
    let first_reader = Arc::clone(&reader);
    let first_desc = descriptor.clone();
    let first_placement = placement.clone();
    let first = std::thread::spawn(move || {
        let mut destination = vec![0xA5; 64];
        first_reader
            .read_into_with(&first_desc, &first_placement, &mut destination, &held, None)
            .map(|_| destination)
    });
    started_rx.recv().unwrap();
    let second_mock = MockTransport::new(vec![0x42; 64]);
    let mut second_destination = vec![0xA5; 64];
    assert!(matches!(
        reader.read_into_with(
            &descriptor,
            &placement,
            &mut second_destination,
            &second_mock,
            None
        ),
        Err(RailReadError::ResourceExhausted(_))
    ));
    assert!(second_mock.calls.lock().unwrap().is_empty());
    release_tx.send(()).unwrap();
    release_tx.send(()).unwrap();
    assert_eq!(first.join().unwrap().unwrap(), vec![0x42; 64]);
    assert_eq!(second_destination, vec![0xA5; 64]);
}

#[test]
#[ignore = "software-only Mock measurement; this is not RDMA bandwidth"]
fn software_only_mock_benchmark() {
    fn process_usage() -> (u64, u64, u64) {
        let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
        assert_eq!(unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) }, 0);
        let micros = |time: libc::timeval| time.tv_sec as u64 * 1_000_000 + time.tv_usec as u64;
        (
            micros(usage.ru_utime),
            micros(usage.ru_stime),
            usage.ru_maxrss as u64,
        )
    }
    let env_number = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let mib = env_number("CS_RAIL_MOCK_MIB", 64);
    let iterations = env_number("CS_RAIL_MOCK_ITERS", 5);
    let rail_count = env_number("CS_RAIL_MOCK_RAILS", 2);
    assert!(mib > 0 && iterations > 0 && (1..=2).contains(&rail_count));
    let size = mib * 1024 * 1024;
    let object = vec![0x42; size];
    let mock = MockTransport::new(object);
    let reader = RailReader::new(routes()[..rail_count].to_vec(), RailLimits::default()).unwrap();
    let (descriptor, placement) = fixture(size, 4 * 1024 * 1024);
    let mut destination = vec![0xA5; size];
    reader
        .read_into_with(&descriptor, &placement, &mut destination, &mock, None)
        .unwrap();
    let warmup_bytes: Vec<_> = reader
        .snapshots()
        .iter()
        .map(|snapshot| snapshot.bytes)
        .collect();
    let mut durations = Vec::with_capacity(iterations);
    let mut cpu_user_us = 0u64;
    let mut cpu_system_us = 0u64;
    for _ in 0..iterations {
        destination.fill(0xA5);
        let before_cpu = process_usage();
        let start = Instant::now();
        reader
            .read_into_with(&descriptor, &placement, &mut destination, &mock, None)
            .unwrap();
        durations.push(start.elapsed().as_micros() as u64);
        let after_cpu = process_usage();
        cpu_user_us += after_cpu.0 - before_cpu.0;
        cpu_system_us += after_cpu.1 - before_cpu.1;
        assert!(destination.iter().all(|byte| *byte == 0x42));
    }
    let average = durations.iter().sum::<u64>() / iterations as u64;
    println!("mock_samples_us={durations:?}");
    let gib_per_s = size as f64 * 1_000_000.0 / average as f64 / 1024f64.powi(3);
    let bytes: Vec<_> = reader
        .snapshots()
        .iter()
        .zip(warmup_bytes)
        .map(|(snapshot, warmup)| snapshot.bytes - warmup)
        .collect();
    println!(
            "mock_only,rails={rail_count},size_bytes={size},iters={iterations},avg_us={average},gib_per_s={gib_per_s:.3},cpu_user_us={cpu_user_us},cpu_system_us={cpu_system_us},peak_rss_kb={},rail_bytes={bytes:?}",
            process_usage().2
        );
}
