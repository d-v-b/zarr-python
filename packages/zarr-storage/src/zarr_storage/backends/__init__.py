"""Backends: the objects a `Store` reads and writes through.

`ObjectStoreBackend` needs the `obstore` extra and is imported on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_storage.backends._local import LocalBackend
from zarr_storage.backends._memory import MemoryBackend

if TYPE_CHECKING:
    from zarr_storage.backends._obstore import ObjectStoreBackend

__all__ = ["LocalBackend", "MemoryBackend", "ObjectStoreBackend"]


def __getattr__(name: str) -> object:
    if name == "ObjectStoreBackend":
        from zarr_storage.backends._obstore import ObjectStoreBackend

        return ObjectStoreBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
