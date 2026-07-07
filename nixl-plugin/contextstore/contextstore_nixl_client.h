#ifndef CONTEXTSTORE_NIXL_CLIENT_H
#define CONTEXTSTORE_NIXL_CLIENT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct cs_nixl_client_config {
    const char *endpoint;
    /* Legacy compatibility field. New KVService uses namespace + object_key. */
    const char *model_id;
    const char *namespace_name;
    const char *rdma_server_addr;
    int rdma_enabled;
};

int cs_nixl_client_open(const struct cs_nixl_client_config *config, void **out_client);
void cs_nixl_client_close(void *client);
int cs_nixl_client_put(void *client, const char *key, const void *data, size_t len, uint64_t offset);
int cs_nixl_client_get(void *client, const char *key, void *data, size_t len, uint64_t offset);
int cs_nixl_client_exists(void *client, const char *key, uint64_t *size, int *found);

#ifdef __cplusplus
}
#endif

#endif
