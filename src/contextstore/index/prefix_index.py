from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class _TrieNode:
    children: dict[tuple[int, ...], "_TrieNode"] = field(default_factory=dict)
    block_keys: set[str] = field(default_factory=set)
    prefix_count: int = 0

    @property
    def registered(self) -> bool:
        return bool(self.block_keys) or self.prefix_count > 0


class PrefixIndex:
    def __init__(self, model_id: str, block_size: int):
        self._model_id = model_id
        self._block_size = block_size
        self._registered: set[str] = set()
        self._prefix_lengths: dict[str, int] = {}
        self._root = _TrieNode()
        self._block_key_paths: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._lock = threading.Lock()

    def compute_block_key(self, token_ids: Sequence[int], block_idx: int) -> str:
        end = (block_idx + 1) * self._block_size
        prefix_tokens = token_ids[:end]
        payload = f"{self._model_id}:{','.join(str(t) for t in prefix_tokens)}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def compute_prefix_key(self, token_ids: Sequence[int], num_tokens: int) -> str:
        payload = f"{self._model_id}:{','.join(str(t) for t in token_ids[:num_tokens])}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def lookup_prefix(self, token_ids: Sequence[int]) -> int:
        num_full_blocks = len(token_ids) // self._block_size
        with self._lock:
            if num_full_blocks == 0 or not self._root.children:
                return 0
            node = self._root
            matched_blocks = 0
            for block_idx in range(num_full_blocks):
                block = self._block_tuple(token_ids, block_idx)
                child = node.children.get(block)
                if child is None or not child.registered:
                    break
                matched_blocks = block_idx + 1
                node = child
            return matched_blocks * self._block_size

    def register_blocks(self, token_ids: Sequence[int], num_tokens: int) -> list[str]:
        num_blocks = num_tokens // self._block_size
        keys = []
        with self._lock:
            path: list[tuple[int, ...]] = []
            node = self._root
            for block_idx, key in enumerate(self._block_keys(token_ids, num_blocks)):
                block = self._block_tuple(token_ids, block_idx)
                path.append(block)
                node = node.children.setdefault(block, _TrieNode())
                self._registered.add(key)
                node.block_keys.add(key)
                self._block_key_paths[key] = tuple(path)
                keys.append(key)
        return keys

    def register_prefix(self, token_ids: Sequence[int], num_tokens: int) -> str:
        prefix_key = self.compute_prefix_key(token_ids, num_tokens)
        num_blocks = num_tokens // self._block_size
        with self._lock:
            self._prefix_lengths[prefix_key] = num_tokens
            path: list[tuple[int, ...]] = []
            node = self._root
            for block_idx, key in enumerate(self._block_keys(token_ids, num_blocks)):
                block = self._block_tuple(token_ids, block_idx)
                path.append(block)
                node = node.children.setdefault(block, _TrieNode())
                self._registered.add(key)
                node.block_keys.add(key)
                self._block_key_paths[key] = tuple(path)
                node.prefix_count += 1
        return prefix_key

    def remove_blocks(self, block_keys: list[str]) -> None:
        with self._lock:
            for key in block_keys:
                self._registered.discard(key)
                path = self._block_key_paths.pop(key, None)
                if path is None:
                    continue
                node = self._root
                for block in path:
                    child = node.children.get(block)
                    if child is None:
                        break
                    node = child
                node.block_keys.discard(key)

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._registered or key in self._prefix_lengths

    @property
    def num_registered_blocks(self) -> int:
        with self._lock:
            return len(self._registered)

    def _block_tuple(self, token_ids: Sequence[int], block_idx: int) -> tuple[int, ...]:
        start = block_idx * self._block_size
        end = start + self._block_size
        return tuple(token_ids[start:end])

    def _block_keys(self, token_ids: Sequence[int], num_blocks: int) -> list[str]:
        """Incrementally compute the history-compatible SHA key for each block, avoiding repeated joins over the entire prefix."""
        hasher = hashlib.sha256()
        hasher.update(f"{self._model_id}:".encode())
        keys: list[str] = []
        token_limit = num_blocks * self._block_size
        for token_idx, token in enumerate(token_ids[:token_limit]):
            if token_idx > 0:
                hasher.update(b",")
            hasher.update(str(token).encode())
            if (token_idx + 1) % self._block_size == 0:
                keys.append(hasher.copy().hexdigest()[:32])
        return keys
