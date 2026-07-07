from __future__ import annotations

"""ContextStore — KV Cache tiered storage platform for LLM inference."""

from contextstore.core.config import ContextStoreConfig
from contextstore.core.engine import ContextStoreEngine

__version__ = "0.1.0"

__all__ = ["ContextStoreConfig", "ContextStoreEngine"]

# ContextStoreConnector requires vLLM, import lazily
def __getattr__(name):
    if name == "ContextStoreConnector":
        from contextstore.connector import ContextStoreConnector
        return ContextStoreConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
