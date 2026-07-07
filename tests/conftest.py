from __future__ import annotations

"""Pytest configuration and fixtures for ContextStore tests."""
import sys
from pathlib import Path

# Add src to path so tests can import contextstore
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
