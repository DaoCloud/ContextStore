from __future__ import annotations

import hashlib
import logging
from typing import Sequence

logger = logging.getLogger(__name__)


class RedisPrefixIndex:
    def __init__(self, model_id: str, block_size: int, redis_url: str):
        self._model_id = model_id
        self._block_size = block_size
        self._redis_key = f"contextstore:{model_id}:blocks"
        try:
            import redis
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed ({redis_url}): {e}. Will retry on each operation.")
            import redis
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def compute_block_key(self, token_ids: Sequence[int], block_idx: int) -> str:
        end = (block_idx + 1) * self._block_size
        prefix_tokens = token_ids[:end]
        payload = f"{self._model_id}:{','.join(str(t) for t in prefix_tokens)}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def lookup_prefix(self, token_ids: Sequence[int]) -> int:
        num_full_blocks = len(token_ids) // self._block_size
        if num_full_blocks == 0:
            return 0
        try:
            keys = [
                self.compute_block_key(token_ids, i) for i in range(num_full_blocks)
            ]
            pipe = self._client.pipeline(transaction=False)
            for key in keys:
                pipe.sismember(self._redis_key, key)
            results = pipe.execute()
            matched_tokens = 0
            for i, exists in enumerate(results):
                if not exists:
                    break
                matched_tokens = (i + 1) * self._block_size
            return matched_tokens
        except Exception as e:
            logger.warning(f"Redis lookup failed: {e}")
            return 0

    def register_blocks(self, token_ids: Sequence[int], num_tokens: int) -> list[str]:
        num_blocks = num_tokens // self._block_size
        keys = [self.compute_block_key(token_ids, i) for i in range(num_blocks)]
        if keys:
            try:
                self._client.sadd(self._redis_key, *keys)
            except Exception as e:
                logger.warning(f"Redis register failed: {e}")
        return keys

    def remove_blocks(self, block_keys: list[str]) -> None:
        if block_keys:
            try:
                self._client.srem(self._redis_key, *block_keys)
            except Exception as e:
                logger.warning(f"Redis remove failed: {e}")

    def contains(self, key: str) -> bool:
        try:
            return bool(self._client.sismember(self._redis_key, key))
        except Exception:
            return False

    @property
    def num_registered_blocks(self) -> int:
        try:
            return self._client.scard(self._redis_key) or 0
        except Exception:
            return 0
