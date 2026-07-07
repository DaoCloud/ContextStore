from __future__ import annotations

"""Prefix index implementations: in-process and Redis-backed."""

from contextstore.index.prefix_index import PrefixIndex

try:
    from contextstore.index.prefix_index_redis import RedisPrefixIndex
    __all__ = ["PrefixIndex", "RedisPrefixIndex"]
except ImportError:
    # Redis is optional
    __all__ = ["PrefixIndex"]
