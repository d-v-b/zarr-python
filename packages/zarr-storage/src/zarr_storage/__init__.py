"""A Zarr-agnostic key-value storage layer.

A `Store` is a view of a backend at a root directory. Backends implement capability protocols
(`Readable`, `Writable`, `Listable`, and async twins) over backend paths; `Store` validates keys,
applies its root, enforces read-only access, and is the only object most callers need.
"""

from zarr_storage._errors import (
    InvalidByteRangeError,
    InvalidKeyError,
    KeyConflictError,
    ReadOnlyError,
    StorageError,
    StoreClosedError,
    UnsupportedOperationError,
)
from zarr_storage._protocols import (
    Backend,
    Listable,
    ListableAsync,
    Readable,
    ReadableAsync,
    ReadableInto,
    ReadableIntoAsync,
    Writable,
    WritableAsync,
)
from zarr_storage._store import Store
from zarr_storage._types import (
    ByteRange,
    DeleteResult,
    Entry,
    Generation,
    Key,
    ListItem,
    ObjectMetadata,
    OffsetRequest,
    Prefix,
    PutResult,
    RangeRequest,
    RangesResult,
    ReadResult,
    SuffixRequest,
)

__all__ = [
    "Backend",
    "ByteRange",
    "DeleteResult",
    "Entry",
    "Generation",
    "InvalidByteRangeError",
    "InvalidKeyError",
    "Key",
    "KeyConflictError",
    "ListItem",
    "Listable",
    "ListableAsync",
    "ObjectMetadata",
    "OffsetRequest",
    "Prefix",
    "PutResult",
    "RangeRequest",
    "RangesResult",
    "ReadOnlyError",
    "ReadResult",
    "Readable",
    "ReadableAsync",
    "ReadableInto",
    "ReadableIntoAsync",
    "StorageError",
    "Store",
    "StoreClosedError",
    "SuffixRequest",
    "UnsupportedOperationError",
    "Writable",
    "WritableAsync",
]
