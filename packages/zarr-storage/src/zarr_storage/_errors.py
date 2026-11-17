"""The errors a store raises. A missing key is not one of them: reads return `None`."""

from __future__ import annotations

__all__ = [
    "InvalidByteRangeError",
    "InvalidKeyError",
    "KeyConflictError",
    "ReadOnlyError",
    "StorageError",
    "StoreClosedError",
    "UnsupportedOperationError",
]


class StorageError(Exception):
    """Base class for every error this package raises."""


class InvalidKeyError(StorageError, ValueError):
    """A key or directory does not follow the key grammar, or names something the backend reserves."""


class InvalidByteRangeError(StorageError, ValueError):
    """A byte range starts past the end of the value it reads."""


class KeyConflictError(StorageError):
    """A key collides with a directory of the same name on a backend that can't hold both.

    Object stores and in-memory stores hold `a` and `a/b` side by side; a filesystem can't.
    """


class ReadOnlyError(StorageError):
    """A write was attempted through a read-only store."""


class StoreClosedError(StorageError):
    """An operation was attempted after the store's backend was closed."""


class UnsupportedOperationError(StorageError, NotImplementedError):
    """The store's backend does not implement the requested operation."""
