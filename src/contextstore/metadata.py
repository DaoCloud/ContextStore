from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class RequestMeta:
    request_id: str
    token_ids: list[int]
    block_ids: list[int]
    slot_mapping: torch.Tensor
    is_store: bool
    matched_prefix_len: int = 0
    block_keys: list[str] = field(default_factory=list)

    @staticmethod
    def make_slot_mapping(block_ids: list[int], block_size: int, num_tokens: int) -> torch.Tensor:
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.long)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape(1, block_size)
            + block_ids_tensor.reshape(num_blocks, 1) * block_size
        )
        return slot_mapping.flatten()[:num_tokens]


class ContextStoreMetadata:
    """Metadata passed from scheduler-side connector to worker-side connector."""

    def __init__(self, block_size: int):
        self.requests: list[RequestMeta] = []
        self.block_size = block_size

    def add_load_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        block_keys: list[str],
        matched_prefix_len: int,
    ) -> None:
        slot_mapping = RequestMeta.make_slot_mapping(block_ids, block_size, matched_prefix_len)
        self.requests.append(RequestMeta(
            request_id=request_id,
            token_ids=token_ids,
            block_ids=block_ids,
            slot_mapping=slot_mapping,
            is_store=False,
            matched_prefix_len=matched_prefix_len,
            block_keys=block_keys,
        ))

    def add_store_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        block_keys: list[str],
    ) -> None:
        num_tokens = (len(block_ids)) * block_size
        num_tokens = min(num_tokens, len(token_ids))
        slot_mapping = RequestMeta.make_slot_mapping(block_ids, block_size, num_tokens)
        self.requests.append(RequestMeta(
            request_id=request_id,
            token_ids=token_ids,
            block_ids=block_ids,
            slot_mapping=slot_mapping,
            is_store=True,
            block_keys=block_keys,
        ))
