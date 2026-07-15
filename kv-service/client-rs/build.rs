//! Generate gRPC client stubs (server-side build.rs uses build_client(false); we need the client here).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("PROTOC").is_none() {
        let protoc = protoc_bin_vendored::protoc_bin_path()?;
        std::env::set_var("PROTOC", protoc);
    }

    // Enable bytes(["."]) so prost decodes every `bytes` field into prost::bytes::Bytes
    // (an Arc-refcounted view). This lets DataChunk.data / GetResponse.data reference
    // tonic's inbound buffer directly, so get_stream_chunks returning Vec<Bytes> is fully
    // zero-copy (avoids the 480MB page-fault first-touch when concatenating on the client).
    tonic_build::configure()
        .build_server(false)
        .build_client(true)
        .bytes(["."])
        .compile(&["../proto/kv_service.proto"], &["../proto"])?;
    println!("cargo:rerun-if-changed=../proto/kv_service.proto");
    Ok(())
}
