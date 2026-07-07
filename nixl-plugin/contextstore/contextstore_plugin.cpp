#include "contextstore_backend.h"

#include <exception>
#include <iostream>

#define NIXL_PLUGIN_API_VERSION 1
#define NIXL_PLUGIN_EXPORT __attribute__((visibility("default")))

class nixlBackendPlugin {
public:
    int api_version;
    nixlBackendEngine *(*create_engine)(const nixlBackendInitParams *init_params);
    void (*destroy_engine)(nixlBackendEngine *engine);
    const char *(*get_plugin_name)();
    const char *(*get_plugin_version)();
    nixl_b_params_t (*get_backend_options)();
    nixl_mem_list_t (*get_backend_mems)();
};

namespace {

nixlBackendEngine *
createEngine(const nixlBackendInitParams *init_params) {
    try {
        return new ContextStoreNixlEngine(init_params);
    } catch (const std::exception &e) {
        std::cerr << "Failed to create CONTEXTSTORE NIXL backend: " << e.what() << "\n";
        return nullptr;
    }
}

void
destroyEngine(nixlBackendEngine *engine) {
    delete engine;
}

const char *
getPluginName() {
    return "CONTEXTSTORE";
}

const char *
getPluginVersion() {
    return "0.1.0";
}

nixl_b_params_t
getBackendOptions() {
    return {
        {"client_library", "Path to libcontextstore_nixl_client.so"},
        {"endpoint", "ContextStore KVService endpoint, for example 127.0.0.1:50051"},
        {"namespace", "KVService namespace for NIXL/KVBM object keys"},
        {"model_id", "Legacy namespace fallback; prefer namespace"},
        {"rdma_enabled", "true or false"},
        {"rdma_server_addr", "ContextStore RDMA server address"},
        {"file_root", "Smoke-test only local file object root"},
    };
}

nixl_mem_list_t
getBackendMems() {
    return {DRAM_SEG, OBJ_SEG};
}

} // namespace

extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    static nixlBackendPlugin plugin{
        NIXL_PLUGIN_API_VERSION,
        createEngine,
        destroyEngine,
        getPluginName,
        getPluginVersion,
        getBackendOptions,
        getBackendMems,
    };
    return &plugin;
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
