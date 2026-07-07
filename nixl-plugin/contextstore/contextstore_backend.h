#ifndef CONTEXTSTORE_NIXL_BACKEND_H
#define CONTEXTSTORE_NIXL_BACKEND_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/backend_engine.h"

class ContextStoreClient {
public:
    virtual ~ContextStoreClient() = default;
    virtual nixl_status_t put(const std::string &key, const void *data, size_t len, uint64_t offset) = 0;
    virtual nixl_status_t get(const std::string &key, void *data, size_t len, uint64_t offset) = 0;
    virtual nixl_status_t exists(const std::string &key, uint64_t &size, bool &found) = 0;
};

class ContextStoreMetadata : public nixlBackendMD {
public:
    ContextStoreMetadata(nixl_mem_t mem_type, uint64_t dev_id, std::string key);

    nixl_mem_t mem_type;
    uint64_t dev_id;
    std::string key;
};

class ContextStoreReqHandle : public nixlBackendReqH {
public:
    struct TransferOp {
        std::string key;
        std::uintptr_t local_addr = 0;
        size_t len = 0;
        uint64_t remote_offset = 0;
    };

    explicit ContextStoreReqHandle(nixl_status_t status);

    nixl_status_t status;
    std::vector<TransferOp> ops;
};

class ContextStoreNixlEngine : public nixlBackendEngine {
public:
    explicit ContextStoreNixlEngine(const nixlBackendInitParams *init_params);
    ~ContextStoreNixlEngine() override;

    bool supportsRemote() const override;
    bool supportsLocal() const override;
    bool supportsNotif() const override;
    nixl_mem_list_t getSupportedMems() const override;

    nixl_status_t registerMem(
        const nixlBlobDesc &mem,
        const nixl_mem_t &nixl_mem,
        nixlBackendMD *&out) override;
    nixl_status_t deregisterMem(nixlBackendMD *meta) override;
    nixl_status_t connect(const std::string &remote_agent) override;
    nixl_status_t disconnect(const std::string &remote_agent) override;
    nixl_status_t unloadMD(nixlBackendMD *input) override;
    nixl_status_t loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override;

    nixl_status_t queryMem(
        const nixl_reg_dlist_t &descs,
        std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t prepXfer(
        const nixl_xfer_op_t &operation,
        const nixl_meta_dlist_t &local,
        const nixl_meta_dlist_t &remote,
        const std::string &remote_agent,
        nixlBackendReqH *&handle,
        const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t postXfer(
        const nixl_xfer_op_t &operation,
        const nixl_meta_dlist_t &local,
        const nixl_meta_dlist_t &remote,
        const std::string &remote_agent,
        nixlBackendReqH *&handle,
        const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t checkXfer(nixlBackendReqH *handle) const override;
    nixl_status_t releaseReqH(nixlBackendReqH *handle) const override;

private:
    [[nodiscard]] std::string keyForRemoteDesc(const nixlMetaDesc &desc) const;
    [[nodiscard]] nixl_status_t validateXfer(
        const nixl_xfer_op_t &operation,
        const nixl_meta_dlist_t &local,
        const nixl_meta_dlist_t &remote,
        const std::string &remote_agent) const;

    std::unique_ptr<ContextStoreClient> client_;
    mutable std::unordered_map<uint64_t, std::string> dev_id_to_key_;
};

#endif
