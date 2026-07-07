//! Request Router — shard routing
//!
//! The KVService currently treats keys as opaque object identities:
//! - namespace: isolates connector / model / tenant / environment
//! - object_key: defined by the upper-layer connector; KVService does not
//!   interpret prefix/block/layer semantics
//!
//! By default routing hashes the full object identity to a device, so the
//! underlying storage is not locked into layer/prefix semantics.

use crate::config::Config;
use crate::error::Result;
use md5::{Digest, Md5};
use std::path::PathBuf;
use twox_hash::xxh3::hash64;

/// Key struct (matches protobuf ObjectKey)
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ObjectKey {
    pub namespace: String,
    pub object_key: String,
}

impl ObjectKey {
    /// Canonical string key used by RocksDB metadata and RDMA wire.
    ///
    /// The format is `<namespace_byte_len>:<namespace><object_key>`, which
    /// avoids ambiguity when business keys contain `:` or `/` characters.
    pub fn to_string_key(&self) -> String {
        format!(
            "{}:{}{}",
            self.namespace.len(),
            self.namespace,
            self.object_key
        )
    }

    pub fn object_digest(&self) -> String {
        let mut hasher = Md5::new();
        hasher.update(self.to_string_key().as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn from_string_key(value: &str) -> Result<Self> {
        let Some((len_part, rest)) = value.split_once(':') else {
            return Err(crate::error::KVError::InvalidArgument(format!(
                "invalid object key encoding: {}",
                value
            )));
        };
        let ns_len: usize = len_part.parse().map_err(|_| {
            crate::error::KVError::InvalidArgument(format!(
                "invalid namespace length in object key: {}",
                value
            ))
        })?;
        if rest.len() < ns_len || !rest.is_char_boundary(ns_len) {
            return Err(crate::error::KVError::InvalidArgument(format!(
                "invalid namespace boundary in object key: {}",
                value
            )));
        }
        let namespace = rest[..ns_len].to_string();
        let object_key = rest[ns_len..].to_string();
        Ok(Self {
            namespace,
            object_key,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    ObjectHash,
}

impl Strategy {
    pub fn from_config(s: &str) -> Self {
        match s {
            "object_hash" => Self::ObjectHash,
            _ => Self::ObjectHash,
        }
    }
}

pub struct ShardRouter {
    devices: Vec<PathBuf>,
    data_subdir: String,
    strategy: Strategy,
}

impl ShardRouter {
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            devices: config.storage.devices.clone(),
            data_subdir: config.storage.data_subdir.clone(),
            strategy: Strategy::from_config(&config.router.strategy),
        })
    }

    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    pub fn devices(&self) -> &[PathBuf] {
        &self.devices
    }

    /// Route a key to a device index
    pub fn route(&self, key: &ObjectKey) -> usize {
        match self.strategy {
            Strategy::ObjectHash => self.object_hash(key),
        }
    }

    fn object_hash(&self, key: &ObjectKey) -> usize {
        let h = hash64(key.to_string_key().as_bytes());
        (h as usize) % self.devices.len()
    }

    /// Compute the local file path for a key
    pub fn key_to_path(&self, key: &ObjectKey) -> PathBuf {
        let device_id = self.route(key);
        self.key_to_path_on_device(key, device_id)
    }

    /// Compute the versioned physical path for a key.
    ///
    /// Plain `key_to_path` returns the compatibility path for the logical
    /// key; new writes encode generation/layout into the filename so that
    /// overwrites write a new file and then commit metadata, never
    /// clobbering a physical file still referenced by old metadata.
    pub fn key_to_versioned_path(
        &self,
        key: &ObjectKey,
        device_id: usize,
        generation: u64,
        layout_version: u64,
    ) -> PathBuf {
        let device_root = &self.devices[device_id];
        let digest = key.object_digest();
        let (hi, rest) = digest.split_at(2);
        let (lo, file) = rest.split_at(2);
        device_root
            .join(&self.data_subdir)
            .join("data")
            .join(Self::path_component(&key.namespace))
            .join(hi)
            .join(lo)
            .join(format!("{}.g{}.l{}.bin", file, generation, layout_version))
    }

    /// Compute the path for a key on a specific device (used for striped chunks)
    pub fn key_to_path_on_device(&self, key: &ObjectKey, device_id: usize) -> PathBuf {
        let device_root = &self.devices[device_id];
        let digest = key.object_digest();
        let (hi, rest) = digest.split_at(2);
        let (lo, file) = rest.split_at(2);
        device_root
            .join(&self.data_subdir)
            .join("data")
            .join(Self::path_component(&key.namespace))
            .join(hi)
            .join(lo)
            .join(format!("{}.bin", file))
    }

    /// Compute the path for a striped chunk (chunk_i lives on device_id, filename carries the chunk suffix)
    pub fn chunk_path(&self, key: &ObjectKey, chunk_idx: usize, device_id: usize) -> PathBuf {
        let device_root = &self.devices[device_id];
        let digest = key.object_digest();
        let (hi, rest) = digest.split_at(2);
        let (lo, file) = rest.split_at(2);
        device_root
            .join(&self.data_subdir)
            .join("data")
            .join(Self::path_component(&key.namespace))
            .join(hi)
            .join(lo)
            .join(format!("{}.chunk{}.bin", file, chunk_idx))
    }

    /// Compute the versioned physical path for a striped chunk.
    pub fn chunk_versioned_path(
        &self,
        key: &ObjectKey,
        chunk_idx: usize,
        device_id: usize,
        generation: u64,
        layout_version: u64,
    ) -> PathBuf {
        let device_root = &self.devices[device_id];
        let digest = key.object_digest();
        let (hi, rest) = digest.split_at(2);
        let (lo, file) = rest.split_at(2);
        device_root
            .join(&self.data_subdir)
            .join("data")
            .join(Self::path_component(&key.namespace))
            .join(hi)
            .join(lo)
            .join(format!(
                "{}.g{}.l{}.chunk{}.bin",
                file, generation, layout_version, chunk_idx
            ))
    }

    fn path_component(value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        for ch in value.chars() {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                out.push(ch);
            } else {
                out.push('_');
            }
        }
        if out.is_empty() || out == "." || out == ".." {
            "_".to_string()
        } else {
            out
        }
    }

    /// Pick the device that chunk_idx should land on (simple round-robin, starting from the object identity)
    pub fn chunk_device(&self, key: &ObjectKey, chunk_idx: usize) -> usize {
        let base = self.object_hash(key);
        (base + chunk_idx) % self.devices.len()
    }

    /// Batch grouping: group keys by device (for parallel I/O)
    pub fn group_by_device<'a>(&self, keys: &'a [ObjectKey]) -> Vec<(usize, Vec<&'a ObjectKey>)> {
        let mut groups: Vec<Vec<&'a ObjectKey>> = vec![Vec::new(); self.devices.len()];
        for k in keys {
            groups[self.route(k)].push(k);
        }
        groups
            .into_iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(n_dev: usize, strategy: &str) -> Config {
        let mut cfg = Config::default();
        cfg.storage.devices = (0..n_dev)
            .map(|i| PathBuf::from(format!("/mnt/nvme{}", i)))
            .collect();
        cfg.router.strategy = strategy.to_string();
        cfg
    }

    #[test]
    fn canonical_key_round_trips_with_delimiters() {
        let key = ObjectKey {
            namespace: "tenant:a".into(),
            object_key: "/prefix:64/layer:0".into(),
        };

        let decoded = ObjectKey::from_string_key(&key.to_string_key()).unwrap();

        assert_eq!(decoded, key);
    }

    #[test]
    fn object_hash_is_stable_for_same_object() {
        let cfg = make_config(4, "object_hash");
        let r = ShardRouter::new(&cfg).unwrap();
        let key = ObjectKey {
            namespace: "ns".into(),
            object_key: "object/a".into(),
        };

        assert_eq!(r.route(&key), r.route(&key));
    }

    #[test]
    fn chunk_device_stripes_from_object_home() {
        let cfg = make_config(4, "object_hash");
        let r = ShardRouter::new(&cfg).unwrap();
        let key = ObjectKey {
            namespace: "ns".into(),
            object_key: "object/a".into(),
        };
        let base = r.route(&key);

        assert_eq!(r.chunk_device(&key, 0), base);
        assert_eq!(r.chunk_device(&key, 1), (base + 1) % 4);
        assert_eq!(r.chunk_device(&key, 4), base);
    }

    #[test]
    fn group_by_device_works() {
        let cfg = make_config(2, "object_hash");
        let r = ShardRouter::new(&cfg).unwrap();
        let keys: Vec<ObjectKey> = (0..6)
            .map(|i| ObjectKey {
                namespace: "ns".into(),
                object_key: format!("object_{}", i),
            })
            .collect();
        let groups = r.group_by_device(&keys);
        assert!(groups.len() <= 2);
        assert_eq!(
            groups.iter().map(|(_, v)| v.len()).sum::<usize>(),
            keys.len()
        );
    }

    #[test]
    fn unsafe_namespace_stays_under_device_root() {
        let cfg = make_config(2, "object_hash");
        let r = ShardRouter::new(&cfg).unwrap();
        let key = ObjectKey {
            namespace: "/data/models/Qwen2.5-32B-Instruct".into(),
            object_key: "prefix:abcdef/blocks_1000_tp0".into(),
        };

        let path = r.chunk_path(&key, 0, 1);

        assert!(path.starts_with("/mnt/nvme1"));
        assert!(path
            .to_string_lossy()
            .contains("_data_models_Qwen2.5-32B-Instruct"));
    }
}
