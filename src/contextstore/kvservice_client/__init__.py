from __future__ import annotations

"""ContextStore KV Service - Python Client SDK."""

from .client import KVClient
from .types import (
    DataReadResult,
    KVMetadata,
    ObjectDescriptor,
    ObjectKey,
    ObjectLookupResult,
    PlacementChunk,
    PlacementDescriptor,
    PutOptions,
)

__version__ = "0.1.0"
__all__ = [
    "KVClient",
    "ObjectKey",
    "KVMetadata",
    "PutOptions",
    "ObjectDescriptor",
    "PlacementChunk",
    "PlacementDescriptor",
    "ObjectLookupResult",
    "DataReadResult",
]
