//! Redis metadata inspection CLI for ContextStore objects.
//!
//! The tool accepts human-readable namespace/object-key pairs and only exposes
//! the canonical key format when reporting the underlying storage identity.

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use clap::{Args, Parser, Subcommand};
use contextstore_server::config::Config;
use contextstore_server::metadata::{BlockMeta, StripingInfo};
use contextstore_server::router::ObjectKey;
use redis::Connection;
use serde::Serialize;
use std::collections::BTreeMap;
use std::cmp::Ordering;
use std::path::PathBuf;

const BLOCK_META_SEGMENT: &str = "block_meta";
const NAMESPACE_WIDTH: usize = 20;
const OBJECT_KEY_WIDTH: usize = 56;
const SIZE_WIDTH: usize = 12;
const STRIPES_WIDTH: usize = 7;
const CHECKSUMS_WIDTH: usize = 18;
const GENERATION_WIDTH: usize = 10;
const LAYOUT_WIDTH: usize = 7;

#[derive(Parser)]
#[command(name = "cs-meta", about = "Inspect ContextStore Redis object metadata")]
struct Cli {
    /// KVService TOML config used to locate Redis and its metadata key prefix.
    #[arg(long, default_value = "configs/server.toml")]
    config: PathBuf,
    /// Emit complete metadata as JSON instead of a human-readable table.
    #[arg(long)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect one object by a readable namespace/object-key pair or canonical key.
    Get(GetArgs),
    /// List namespaces that currently have object metadata in Redis.
    Namespaces,
    /// Search current object metadata by readable namespace and object-key prefix.
    List(ListArgs),
}

#[derive(Args)]
struct GetArgs {
    /// Internal canonical key (<namespace-byte-length>:<namespace><object-key>).
    #[arg(long, conflicts_with_all = ["namespace", "object_key"])]
    key: Option<String>,
    /// Logical namespace, such as "redhare" or "rust-bench".
    #[arg(long, requires = "object_key")]
    namespace: Option<String>,
    /// Logical object key within the namespace.
    #[arg(long, requires = "namespace")]
    object_key: Option<String>,
}

#[derive(Args)]
struct ListArgs {
    /// Logical namespace to search. Requiring a namespace prevents accidental full scans.
    #[arg(long)]
    namespace: String,
    /// Optional prefix of the logical object key.
    #[arg(long, default_value = "")]
    object_prefix: String,
    /// Optionally limit matching entries after sorting. By default every current key is shown.
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Serialize)]
struct MetadataRecord<'a> {
    redis_key: &'a str,
    canonical_key: &'a str,
    namespace: &'a str,
    object_key: &'a str,
    metadata: &'a BlockMeta,
}

struct Entry {
    redis_key: String,
    canonical_key: String,
    object_key: ObjectKey,
    metadata: BlockMeta,
}

#[derive(Serialize)]
struct NamespaceSummary {
    namespace: String,
    object_count: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::from_file(&cli.config)
        .with_context(|| format!("load config {}", cli.config.display()))?;
    let client = redis::Client::open(config.metadata.redis_url.as_str())
        .context("open Redis metadata client")?;
    let mut connection = client
        .get_connection()
        .context("connect to Redis metadata store")?;

    match cli.command {
        Command::Get(args) => {
            let object_key = resolve_get_key(args)?;
            let canonical_key = object_key.to_string_key();
            let redis_key = block_redis_key(&config, &canonical_key);
            let Some(metadata) = get_metadata(&mut connection, &redis_key)? else {
                bail!(
                    "object not found: namespace={:?}, object_key={:?}",
                    object_key.namespace,
                    object_key.object_key
                );
            };
            let entry = Entry {
                redis_key,
                canonical_key,
                object_key,
                metadata,
            };
            print_entry(&entry, cli.json)?;
        }
        Command::Namespaces => {
            let namespaces = list_namespaces(&mut connection, &config)?;
            print_namespaces(&namespaces, cli.json)?;
        }
        Command::List(args) => {
            let entries = list_entries(&mut connection, &config, &args)?;
            print_entries(&entries, cli.json)?;
        }
    }

    Ok(())
}

fn list_namespaces(connection: &mut Connection, config: &Config) -> Result<Vec<NamespaceSummary>> {
    let redis_prefix = block_redis_prefix(config);
    let pattern = format!("{redis_prefix}*");
    let mut cursor = 0_u64;
    let mut object_counts = BTreeMap::<String, usize>::new();

    loop {
        let (next_cursor, redis_keys): (u64, Vec<String>) = redis::cmd("SCAN")
            .arg(cursor)
            .arg("MATCH")
            .arg(&pattern)
            .arg("COUNT")
            .arg(256)
            .query(connection)
            .context("scan Redis metadata keys")?;

        for redis_key in redis_keys {
            let Some(canonical_key) = redis_key.strip_prefix(&redis_prefix) else {
                continue;
            };
            let object_key = ObjectKey::from_string_key(canonical_key)
                .with_context(|| format!("parse Redis metadata key {redis_key}"))?;
            *object_counts.entry(object_key.namespace).or_default() += 1;
        }

        if next_cursor == 0 {
            break;
        }
        cursor = next_cursor;
    }

    Ok(object_counts
        .into_iter()
        .map(|(namespace, object_count)| NamespaceSummary {
            namespace,
            object_count,
        })
        .collect())
}

fn resolve_get_key(args: GetArgs) -> Result<ObjectKey> {
    match (args.key, args.namespace, args.object_key) {
        (Some(key), None, None) => ObjectKey::from_string_key(&key)
            .with_context(|| format!("parse canonical key {key:?}")),
        (None, Some(namespace), Some(object_key)) => Ok(ObjectKey {
            namespace,
            object_key,
        }),
        _ => bail!(
            "provide either --key <canonical-key> or both --namespace <namespace> and --object-key <object-key>"
        ),
    }
}

fn block_redis_key(config: &Config, canonical_key: &str) -> String {
    format!(
        "{}{}:{}",
        config.metadata.redis_key_prefix, BLOCK_META_SEGMENT, canonical_key
    )
}

fn block_redis_prefix(config: &Config) -> String {
    format!("{}{}:", config.metadata.redis_key_prefix, BLOCK_META_SEGMENT)
}

fn get_metadata(connection: &mut Connection, redis_key: &str) -> Result<Option<BlockMeta>> {
    let raw: Option<Vec<u8>> = redis::cmd("GET")
        .arg(redis_key)
        .query(connection)
        .with_context(|| format!("read Redis key {redis_key}"))?;
    raw.map(|bytes| {
        serde_json::from_slice(&bytes).with_context(|| format!("decode Redis key {redis_key}"))
    })
    .transpose()
}

fn list_entries(connection: &mut Connection, config: &Config, args: &ListArgs) -> Result<Vec<Entry>> {
    let redis_prefix = block_redis_prefix(config);
    let pattern = format!("{redis_prefix}*");
    let mut cursor = 0_u64;
    let mut entries = Vec::new();

    loop {
        let (next_cursor, redis_keys): (u64, Vec<String>) = redis::cmd("SCAN")
            .arg(cursor)
            .arg("MATCH")
            .arg(&pattern)
            .arg("COUNT")
            .arg(256)
            .query(connection)
            .context("scan Redis metadata keys")?;

        for redis_key in redis_keys {
            let Some(canonical_key) = redis_key.strip_prefix(&redis_prefix) else {
                continue;
            };
            let canonical_key = canonical_key.to_string();
            let object_key = ObjectKey::from_string_key(&canonical_key)
                .with_context(|| format!("parse Redis metadata key {redis_key}"))?;
            if object_key.namespace != args.namespace
                || !object_key.object_key.starts_with(&args.object_prefix)
            {
                continue;
            }
            let Some(metadata) = get_metadata(connection, &redis_key)? else {
                continue;
            };
            entries.push(Entry {
                redis_key,
                canonical_key,
                object_key,
                metadata,
            });
        }

        if next_cursor == 0 {
            break;
        }
        cursor = next_cursor;
    }

    entries.sort_by(|left, right| match left.object_key.namespace.cmp(&right.object_key.namespace) {
        Ordering::Equal => left.object_key.object_key.cmp(&right.object_key.object_key),
        other => other,
    });
    if let Some(limit) = args.limit {
        entries.truncate(limit);
    }
    Ok(entries)
}

fn print_entry(entry: &Entry, json: bool) -> Result<()> {
    if json {
        let record = MetadataRecord {
            redis_key: &entry.redis_key,
            canonical_key: &entry.canonical_key,
            namespace: &entry.object_key.namespace,
            object_key: &entry.object_key.object_key,
            metadata: &entry.metadata,
        };
        println!("{}", serde_json::to_string_pretty(&record)?);
        return Ok(());
    }

    println!("namespace:       {}", entry.object_key.namespace);
    println!("object key:      {}", entry.object_key.object_key);
    println!("canonical key:   {}", entry.canonical_key);
    println!("Redis key:       {}", entry.redis_key);
    println!("size:            {}", format_bytes(entry.metadata.size));
    println!("generation:      {}", entry.metadata.object_generation);
    println!("layout version:  {}", entry.metadata.layout_version);
    println!("content etag:    {}", empty_as_unset(&entry.metadata.content_etag));
    println!("created at:      {}", format_timestamp(entry.metadata.created_at));
    println!(
        "last accessed:   {}",
        format_timestamp(entry.metadata.last_accessed_at)
    );
    println!("TTL seconds:     {}", entry.metadata.ttl_seconds);
    print_striping(entry.metadata.striping.as_ref());
    Ok(())
}

fn print_namespaces(namespaces: &[NamespaceSummary], json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(namespaces)?);
        return Ok(());
    }
    if namespaces.is_empty() {
        println!("No current object metadata.");
        return Ok(());
    }
    println!("{:<NAMESPACE_WIDTH$}  {:>7}", "NAMESPACE", "OBJECTS");
    for namespace in namespaces {
        println!(
            "{:<NAMESPACE_WIDTH$}  {:>7}",
            truncate_for_table(&namespace.namespace, NAMESPACE_WIDTH),
            namespace.object_count,
        );
    }
    Ok(())
}

fn print_entries(entries: &[Entry], json: bool) -> Result<()> {
    if json {
        let records: Vec<MetadataRecord<'_>> = entries
            .iter()
            .map(|entry| MetadataRecord {
                redis_key: &entry.redis_key,
                canonical_key: &entry.canonical_key,
                namespace: &entry.object_key.namespace,
                object_key: &entry.object_key.object_key,
                metadata: &entry.metadata,
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&records)?);
        return Ok(());
    }

    if entries.is_empty() {
        println!("No matching current object metadata.");
        return Ok(());
    }
    println!(
        "{:<NAMESPACE_WIDTH$}  {:<OBJECT_KEY_WIDTH$}  {:>SIZE_WIDTH$}  {:>STRIPES_WIDTH$}  {:<CHECKSUMS_WIDTH$}  {:>GENERATION_WIDTH$}  {:>LAYOUT_WIDTH$}",
        "NAMESPACE",
        "OBJECT KEY",
        "SIZE",
        "STRIPES",
        "CHECKSUMS",
        "GENERATION",
        "LAYOUT",
    );
    for entry in entries {
        let stripe_count = entry
            .metadata
            .striping
            .as_ref()
            .map(|striping| striping.chunk_paths.len())
            .unwrap_or(0);
        println!(
            "{:<NAMESPACE_WIDTH$}  {:<OBJECT_KEY_WIDTH$}  {:>SIZE_WIDTH$}  {:>STRIPES_WIDTH$}  {:<CHECKSUMS_WIDTH$}  {:>GENERATION_WIDTH$}  {:>LAYOUT_WIDTH$}",
            truncate_for_table(&entry.object_key.namespace, NAMESPACE_WIDTH),
            truncate_for_table(&entry.object_key.object_key, OBJECT_KEY_WIDTH),
            format_bytes(entry.metadata.size),
            stripe_count,
            checksum_status(entry.metadata.striping.as_ref()),
            entry.metadata.object_generation,
            entry.metadata.layout_version,
        );
    }
    Ok(())
}

fn print_striping(striping: Option<&StripingInfo>) {
    let Some(striping) = striping else {
        println!("striping:        no");
        return;
    };

    println!(
        "striping:        {} stripes, chunk size {}, checksum status {}",
        striping.chunk_paths.len(),
        format_bytes(striping.chunk_size),
        checksum_status(Some(striping)),
    );
    println!("{:>6}  {:<20}  {:<18}  PATH", "STRIPE", "DEVICE", "CHECKSUM");
    for (index, path) in striping.chunk_paths.iter().enumerate() {
        let device = stripe_device(striping, index);
        let checksum = striping
            .chunk_checksums
            .get(index)
            .map(String::as_str)
            .filter(|checksum| !checksum.is_empty())
            .unwrap_or("<missing>");
        println!("{:>6}  {:<20}  {:<18}  {}", index, device, checksum, path);
    }
}

fn stripe_device(striping: &StripingInfo, index: usize) -> String {
    striping
        .chunk_locations
        .get(index)
        .map(|location| format!("{}:{}", location.node_id, location.device_id))
        .or_else(|| {
            striping
                .chunk_devices
                .get(index)
                .map(|device| format!("local:{device}"))
        })
        .unwrap_or_else(|| "<unknown>".to_string())
}

fn checksum_status(striping: Option<&StripingInfo>) -> String {
    let Some(striping) = striping else {
        return "not-striped".to_string();
    };
    let stripe_count = striping.chunk_paths.len();
    if stripe_count == 0 {
        return "empty-layout".to_string();
    }
    let checksum_count = striping
        .chunk_checksums
        .iter()
        .filter(|checksum| !checksum.is_empty())
        .count();
    if checksum_count == stripe_count {
        format!("complete ({checksum_count}/{stripe_count})")
    } else if checksum_count == 0 {
        format!("missing (0/{stripe_count})")
    } else {
        format!("partial ({checksum_count}/{stripe_count})")
    }
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes as f64 >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes as f64 >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB)
    } else {
        format!("{bytes} B")
    }
}

fn truncate_for_table(value: &str, width: usize) -> String {
    let character_count = value.chars().count();
    if character_count <= width {
        return value.to_string();
    }
    if width <= 3 {
        return ".".repeat(width);
    }
    let prefix: String = value.chars().take(width - 3).collect();
    format!("{prefix}...")
}

fn empty_as_unset(value: &str) -> &str {
    if value.is_empty() {
        "<unset>"
    } else {
        value
    }
}

fn format_timestamp(timestamp: i64) -> String {
    DateTime::<Utc>::from_timestamp(timestamp, 0)
        .map(|time| time.format("%Y-%m-%dT%H:%M:%SZ").to_string())
        .unwrap_or_else(|| format!("<invalid: {timestamp}>"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readable_selector_encodes_canonical_key() {
        let key = resolve_get_key(GetArgs {
            key: None,
            namespace: Some("redhare".to_string()),
            object_key: Some("request/prefix".to_string()),
        })
        .unwrap();
        assert_eq!(key.to_string_key(), "7:redharerequest/prefix");
    }

    #[test]
    fn checksum_status_distinguishes_legacy_metadata() {
        let mut striping = StripingInfo {
            chunk_paths: vec!["a".to_string(), "b".to_string()],
            ..Default::default()
        };
        assert_eq!(checksum_status(Some(&striping)), "missing (0/2)");
        striping.chunk_checksums = vec!["abcd".to_string(), "efgh".to_string()];
        assert_eq!(checksum_status(Some(&striping)), "complete (2/2)");
    }

    #[test]
    fn table_truncation_preserves_character_boundaries() {
        assert_eq!(truncate_for_table("short", 8), "short");
        assert_eq!(truncate_for_table("abcdef", 5), "ab...");
        assert_eq!(truncate_for_table("metadata-key", 3), "...");
        assert_eq!(truncate_for_table("rust-bench/combined", 12), "rust-benc...");
    }
}
