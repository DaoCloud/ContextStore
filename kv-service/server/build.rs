fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Decode every `bytes` proto field into prost::bytes::Bytes (an Arc-counted view)
    // instead of the default Vec<u8>. This means large fields like PutRequest.data
    // no longer trigger a ~480MB memcpy on RPC decode — the gRPC buffer is handed
    // over by refcount, and slicing it into 8 stripes inside storage_tier is just
    // a refcount bump on the same underlying buffer.
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .bytes(["."])
        .compile(&["../proto/kv_service.proto"], &["../proto"])?;
    println!("cargo:rerun-if-changed=../proto/kv_service.proto");
    Ok(())
}
