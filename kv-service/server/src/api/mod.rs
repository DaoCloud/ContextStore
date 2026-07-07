//! gRPC API layer
//!
//! Generated protobuf code lives under `generated::contextstore::kv::v1`

pub mod generated {
    pub mod contextstore {
        pub mod kv {
            pub mod v1 {
                tonic::include_proto!("contextstore.kv.v1");
            }
        }
    }
}

mod service;

pub use service::KVServiceImpl;
