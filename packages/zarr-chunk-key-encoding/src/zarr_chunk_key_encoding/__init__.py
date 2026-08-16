"""Chunk key encodings for Zarr version 3 arrays.

A chunk key encoding maps the grid index of a chunk — a tuple of
non-negative integers — to the string key under which that chunk is stored,
and (where well-defined) back again. This package provides:

- `ChunkKeyEncoding` — the abstract base class
- `DefaultChunkKeyEncoding`, `V2ChunkKeyEncoding` — the two encodings
  defined by the Zarr v3 core spec
- `BoundedChunkKeyEncoding` — an encoding bound to a known chunk grid
  (via `ChunkKeyEncoding.bind`), whose finite key set supports membership
  testing, iteration, and `len`
- a name-keyed registry (`register_chunk_key_encoding` and friends) with
  entry-point discovery, modeled on the plugin registry of the `zarrs`
  Rust implementation
- `chunk_key_encoding_from_json` / `parse_chunk_key_encoding` — construct
  encodings from JSON metadata or looser user input

JSON shapes are typed by `zarr-metadata`; this package supplies the runtime
behavior for those types.

>>> from zarr_chunk_key_encoding import chunk_key_encoding_from_json
>>> encoding = chunk_key_encoding_from_json(
...     {"name": "default", "configuration": {"separator": "/"}}
... )
>>> encoding.encode((1, 23))
'c/1/23'
>>> encoding.decode("c/1/23")
(1, 23)
"""

from importlib.metadata import version

from zarr_chunk_key_encoding.abc import ChunkKey, ChunkKeyEncoding, ChunkKeyEncodingJSON
from zarr_chunk_key_encoding.bounded import BoundedChunkKeyEncoding
from zarr_chunk_key_encoding.default import DefaultChunkKeyEncoding
from zarr_chunk_key_encoding.errors import (
    ChunkCoordsOutOfBoundsError,
    ChunkKeyConfigurationError,
    ChunkKeyDecodeError,
    ChunkKeyEncodingError,
    ChunkKeyOutOfBoundsError,
    ChunkKeyPluginWarning,
    ChunkKeyRegistryError,
    InvalidChunkCoordsError,
    UnknownChunkKeyEncodingError,
)
from zarr_chunk_key_encoding.registry import (
    ENTRY_POINT_GROUP,
    ChunkKeyEncodingLike,
    ChunkKeyEncodingParams,
    chunk_key_encoding_from_json,
    get_chunk_key_encoding_class,
    get_chunk_key_encoding_support,
    parse_chunk_key_encoding,
    register_chunk_key_encoding,
    registered_chunk_key_encodings,
    unregister_chunk_key_encoding,
)
from zarr_chunk_key_encoding.separator import SEPARATORS, Separator, parse_separator
from zarr_chunk_key_encoding.support import (
    CORE_CHUNK_KEY_ENCODING_NAMES,
    ChunkKeyEncodingSupport,
)
from zarr_chunk_key_encoding.v2 import V2ChunkKeyEncoding

__version__ = version("zarr-chunk-key-encoding")

__all__ = [
    "CORE_CHUNK_KEY_ENCODING_NAMES",
    "ENTRY_POINT_GROUP",
    "SEPARATORS",
    "BoundedChunkKeyEncoding",
    "ChunkCoordsOutOfBoundsError",
    "ChunkKey",
    "ChunkKeyConfigurationError",
    "ChunkKeyDecodeError",
    "ChunkKeyEncoding",
    "ChunkKeyEncodingError",
    "ChunkKeyEncodingJSON",
    "ChunkKeyEncodingLike",
    "ChunkKeyEncodingParams",
    "ChunkKeyEncodingSupport",
    "ChunkKeyOutOfBoundsError",
    "ChunkKeyPluginWarning",
    "ChunkKeyRegistryError",
    "DefaultChunkKeyEncoding",
    "InvalidChunkCoordsError",
    "Separator",
    "UnknownChunkKeyEncodingError",
    "V2ChunkKeyEncoding",
    "__version__",
    "chunk_key_encoding_from_json",
    "get_chunk_key_encoding_class",
    "get_chunk_key_encoding_support",
    "parse_chunk_key_encoding",
    "parse_separator",
    "register_chunk_key_encoding",
    "registered_chunk_key_encodings",
    "unregister_chunk_key_encoding",
]

register_chunk_key_encoding(DefaultChunkKeyEncoding)
register_chunk_key_encoding(V2ChunkKeyEncoding)
