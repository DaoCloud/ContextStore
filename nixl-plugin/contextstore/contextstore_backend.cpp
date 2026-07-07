#include "contextstore_backend.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "contextstore_nixl_client.h"

namespace {

std::string
getParam(const nixl_b_params_t *params, const std::string &key, const std::string &fallback = "") {
    if (!params) {
        return fallback;
    }
    auto it = params->find(key);
    if (it == params->end()) {
        return fallback;
    }
    return it->second;
}

bool
getBoolParam(const nixl_b_params_t *params, const std::string &key, bool fallback = false) {
    std::string value = getParam(params, key);
    if (value.empty()) {
        return fallback;
    }
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::string
hexKey(const std::string &key) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char c : key) {
        out << std::setw(2) << static_cast<unsigned int>(c);
    }
    return out.str();
}

class FileContextStoreClient final : public ContextStoreClient {
public:
    explicit FileContextStoreClient(std::filesystem::path root) : root_(std::move(root)) {
        std::filesystem::create_directories(root_);
    }

    nixl_status_t put(const std::string &key, const void *data, size_t len, uint64_t offset) override {
        auto path = pathForKey(key);
        std::filesystem::create_directories(path.parent_path());
        std::fstream stream(path, std::ios::in | std::ios::out | std::ios::binary);
        if (!stream.good()) {
            stream.open(path, std::ios::out | std::ios::binary);
            stream.close();
            stream.open(path, std::ios::in | std::ios::out | std::ios::binary);
        }
        if (!stream.good()) {
            std::cerr << "CONTEXTSTORE NIXL file client failed to open for write: " << path << "\n";
            return NIXL_ERR_BACKEND;
        }
        stream.seekp(static_cast<std::streamoff>(offset));
        stream.write(static_cast<const char *>(data), static_cast<std::streamsize>(len));
        return stream.good() ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
    }

    nixl_status_t get(const std::string &key, void *data, size_t len, uint64_t offset) override {
        auto path = pathForKey(key);
        std::ifstream stream(path, std::ios::binary);
        if (!stream.good()) {
            return NIXL_ERR_NOT_FOUND;
        }
        stream.seekg(static_cast<std::streamoff>(offset));
        stream.read(static_cast<char *>(data), static_cast<std::streamsize>(len));
        return stream.gcount() == static_cast<std::streamsize>(len) ? NIXL_SUCCESS : NIXL_ERR_NOT_FOUND;
    }

    nixl_status_t exists(const std::string &key, uint64_t &size, bool &found) override {
        auto path = pathForKey(key);
        found = std::filesystem::exists(path);
        size = found ? static_cast<uint64_t>(std::filesystem::file_size(path)) : 0;
        return NIXL_SUCCESS;
    }

private:
    std::filesystem::path pathForKey(const std::string &key) const {
        return root_ / hexKey(key);
    }

    std::filesystem::path root_;
};

class DynamicContextStoreClient final : public ContextStoreClient {
public:
    explicit DynamicContextStoreClient(const nixl_b_params_t *params) {
        const auto library = getParam(params, "client_library");
        if (library.empty()) {
            throw std::runtime_error("client_library backend parameter is required");
        }
        handle_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle_) {
            throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
        }

        open_ = loadSymbol<OpenFn>("cs_nixl_client_open");
        close_ = loadSymbol<CloseFn>("cs_nixl_client_close");
        put_ = loadSymbol<PutFn>("cs_nixl_client_put");
        get_ = loadSymbol<GetFn>("cs_nixl_client_get");
        exists_ = loadSymbol<ExistsFn>("cs_nixl_client_exists");

        const std::string endpoint = getParam(params, "endpoint");
        const std::string model_id = getParam(params, "model_id");
        const std::string namespace_name = getParam(params, "namespace");
        const std::string rdma_server_addr = getParam(params, "rdma_server_addr");

        cs_nixl_client_config config{
            endpoint.c_str(),
            model_id.c_str(),
            namespace_name.c_str(),
            rdma_server_addr.c_str(),
            getBoolParam(params, "rdma_enabled", false) ? 1 : 0,
        };

        void *client = nullptr;
        int rc = open_(&config, &client);
        if (rc != 0 || client == nullptr) {
            throw std::runtime_error("cs_nixl_client_open failed");
        }
        client_ = client;
    }

    ~DynamicContextStoreClient() override {
        if (client_ && close_) {
            close_(client_);
        }
        if (handle_) {
            dlclose(handle_);
        }
    }

    nixl_status_t put(const std::string &key, const void *data, size_t len, uint64_t offset) override {
        return put_(client_, key.c_str(), data, len, offset) == 0 ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
    }

    nixl_status_t get(const std::string &key, void *data, size_t len, uint64_t offset) override {
        return get_(client_, key.c_str(), data, len, offset) == 0 ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
    }

    nixl_status_t exists(const std::string &key, uint64_t &size, bool &found) override {
        int found_int = 0;
        int rc = exists_(client_, key.c_str(), &size, &found_int);
        found = found_int != 0;
        return rc == 0 ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
    }

private:
    using OpenFn = int (*)(const cs_nixl_client_config *, void **);
    using CloseFn = void (*)(void *);
    using PutFn = int (*)(void *, const char *, const void *, size_t, uint64_t);
    using GetFn = int (*)(void *, const char *, void *, size_t, uint64_t);
    using ExistsFn = int (*)(void *, const char *, uint64_t *, int *);

    template <typename T>
    T loadSymbol(const char *name) {
        dlerror();
        auto *symbol = dlsym(handle_, name);
        const char *error = dlerror();
        if (error || !symbol) {
            throw std::runtime_error(std::string("missing symbol ") + name);
        }
        return reinterpret_cast<T>(symbol);
    }

    void *handle_ = nullptr;
    void *client_ = nullptr;
    OpenFn open_ = nullptr;
    CloseFn close_ = nullptr;
    PutFn put_ = nullptr;
    GetFn get_ = nullptr;
    ExistsFn exists_ = nullptr;
};

std::unique_ptr<ContextStoreClient>
makeClient(const nixl_b_params_t *params) {
    const auto file_root = getParam(params, "file_root");
    if (!file_root.empty()) {
        std::cerr << "CONTEXTSTORE NIXL plugin using file_root smoke-test client: " << file_root << "\n";
        return std::make_unique<FileContextStoreClient>(file_root);
    }
    return std::make_unique<DynamicContextStoreClient>(params);
}

} // namespace

ContextStoreMetadata::ContextStoreMetadata(nixl_mem_t mem_type_, uint64_t dev_id_, std::string key_)
    : nixlBackendMD(true),
      mem_type(mem_type_),
      dev_id(dev_id_),
      key(std::move(key_)) {}

ContextStoreReqHandle::ContextStoreReqHandle(nixl_status_t status_) : status(status_) {}

ContextStoreNixlEngine::ContextStoreNixlEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {
    try {
        client_ = makeClient(init_params->customParams);
        std::cerr << "CONTEXTSTORE NIXL backend initialized\n";
    } catch (const std::exception &e) {
        initErr = true;
        std::cerr << "Failed to initialize CONTEXTSTORE NIXL backend: " << e.what() << "\n";
    }
}

ContextStoreNixlEngine::~ContextStoreNixlEngine() = default;

bool ContextStoreNixlEngine::supportsRemote() const {
    return false;
}

bool ContextStoreNixlEngine::supportsLocal() const {
    return true;
}

bool ContextStoreNixlEngine::supportsNotif() const {
    return false;
}

nixl_mem_list_t ContextStoreNixlEngine::getSupportedMems() const {
    return {DRAM_SEG, OBJ_SEG};
}

nixl_status_t
ContextStoreNixlEngine::registerMem(
    const nixlBlobDesc &mem,
    const nixl_mem_t &nixl_mem,
    nixlBackendMD *&out) {
    if (nixl_mem == DRAM_SEG) {
        out = nullptr;
        return NIXL_SUCCESS;
    }
    if (nixl_mem != OBJ_SEG) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    const std::string key = mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo;
    dev_id_to_key_[mem.devId] = key;
    out = new ContextStoreMetadata(nixl_mem, mem.devId, key);
    return NIXL_SUCCESS;
}

nixl_status_t ContextStoreNixlEngine::deregisterMem(nixlBackendMD *meta) {
    auto *ctx_meta = static_cast<ContextStoreMetadata *>(meta);
    if (ctx_meta != nullptr) {
        dev_id_to_key_.erase(ctx_meta->dev_id);
        delete ctx_meta;
    }
    return NIXL_SUCCESS;
}

nixl_status_t ContextStoreNixlEngine::connect(const std::string &) {
    return NIXL_SUCCESS;
}

nixl_status_t ContextStoreNixlEngine::disconnect(const std::string &) {
    return NIXL_SUCCESS;
}

nixl_status_t ContextStoreNixlEngine::unloadMD(nixlBackendMD *) {
    return NIXL_SUCCESS;
}

nixl_status_t ContextStoreNixlEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    output = input;
    return NIXL_SUCCESS;
}

nixl_status_t
ContextStoreNixlEngine::queryMem(
    const nixl_reg_dlist_t &descs,
    std::vector<nixl_query_resp_t> &resp) const {
    if (!client_) {
        return NIXL_ERR_BACKEND;
    }
    resp.assign(descs.descCount(), std::nullopt);
    for (int i = 0; i < descs.descCount(); ++i) {
        const auto &desc = descs[i];
        const std::string key = desc.metaInfo.empty() ? std::to_string(desc.devId) : desc.metaInfo;
        uint64_t size = 0;
        bool found = false;
        nixl_status_t status = client_->exists(key, size, found);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        if (found) {
            nixl_b_params_t attrs;
            if (size > 0) {
                attrs["size"] = std::to_string(size);
            }
            resp[i] = std::move(attrs);
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
ContextStoreNixlEngine::prepXfer(
    const nixl_xfer_op_t &operation,
    const nixl_meta_dlist_t &local,
    const nixl_meta_dlist_t &remote,
    const std::string &remote_agent,
    nixlBackendReqH *&handle,
    const nixl_opt_b_args_t *) const {
    const nixl_status_t status = validateXfer(operation, local, remote, remote_agent);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    auto *req = new ContextStoreReqHandle(NIXL_ERR_NOT_POSTED);
    req->ops.reserve(local.descCount());
    for (int i = 0; i < local.descCount(); ++i) {
        const auto &local_desc = local[i];
        const auto &remote_desc = remote[i];
        const std::string key = keyForRemoteDesc(remote_desc);
        if (key.empty()) {
            std::cerr << "CONTEXTSTORE NIXL missing OBJ key for devId=" << remote_desc.devId
                      << " metadataP=" << remote_desc.metadataP << "\n";
            delete req;
            return NIXL_ERR_INVALID_PARAM;
        }

        ContextStoreReqHandle::TransferOp op;
        op.key = key;
        op.local_addr = static_cast<std::uintptr_t>(local_desc.addr);
        op.len = local_desc.len;
        op.remote_offset = static_cast<uint64_t>(remote_desc.addr);
        req->ops.push_back(std::move(op));
    }

    handle = req;
    return NIXL_SUCCESS;
}

nixl_status_t
ContextStoreNixlEngine::postXfer(
    const nixl_xfer_op_t &operation,
    const nixl_meta_dlist_t &local,
    const nixl_meta_dlist_t &remote,
    const std::string &remote_agent,
    nixlBackendReqH *&handle,
    const nixl_opt_b_args_t *) const {
    if (!handle || !client_) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *req = static_cast<ContextStoreReqHandle *>(handle);
    const nixl_status_t valid = validateXfer(operation, local, remote, remote_agent);
    if (valid != NIXL_SUCCESS) {
        req->status = valid;
        return valid;
    }

    for (const auto &op : req->ops) {
        nixl_status_t status = NIXL_ERR_BACKEND;
        if (operation == NIXL_WRITE) {
            status = client_->put(
                op.key,
                reinterpret_cast<const void *>(op.local_addr),
                op.len,
                op.remote_offset);
        } else {
            status = client_->get(
                op.key,
                reinterpret_cast<void *>(op.local_addr),
                op.len,
                op.remote_offset);
        }
        if (status != NIXL_SUCCESS) {
            req->status = status;
            return status;
        }
    }

    req->status = NIXL_SUCCESS;
    return NIXL_SUCCESS;
}

nixl_status_t ContextStoreNixlEngine::checkXfer(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return static_cast<ContextStoreReqHandle *>(handle)->status;
}

nixl_status_t ContextStoreNixlEngine::releaseReqH(nixlBackendReqH *handle) const {
    delete static_cast<ContextStoreReqHandle *>(handle);
    return NIXL_SUCCESS;
}

std::string ContextStoreNixlEngine::keyForRemoteDesc(const nixlMetaDesc &desc) const {
    auto it = dev_id_to_key_.find(desc.devId);
    if (it == dev_id_to_key_.end()) {
        return "";
    }
    return it->second;
}

nixl_status_t
ContextStoreNixlEngine::validateXfer(
    const nixl_xfer_op_t &operation,
    const nixl_meta_dlist_t &local,
    const nixl_meta_dlist_t &remote,
    const std::string &remote_agent) const {
    if (operation != NIXL_READ && operation != NIXL_WRITE) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote_agent != localAgent) {
        std::cerr << "CONTEXTSTORE backend only supports local object transfers; remote_agent="
                  << remote_agent << " local_agent=" << localAgent << "\n";
    }
    if (local.getType() != DRAM_SEG || remote.getType() != OBJ_SEG) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (local.descCount() != remote.descCount()) {
        return NIXL_ERR_MISMATCH;
    }
    return NIXL_SUCCESS;
}
