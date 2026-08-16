"""
Support levels: where a chunk key encoding comes from.

The Zarr v3 core spec defines chunk key encoding as an *extension point*, so
the set of encodings is open-ended. Not every encoding carries the same
weight, though, and a consumer usually wants to say which tiers it is willing
to honour -- reading data written with a locally-invented encoding is a very
different proposition from reading data written with ``default``.

Three tiers, narrowest first:

- `ChunkKeyEncodingSupport.CORE` -- defined in the Zarr v3 core spec itself.
  A closed set: ``default`` and ``v2``. Every v3 implementation is expected
  to understand these.
- `ChunkKeyEncodingSupport.EXTENSION` -- registered in the
  `zarr-extensions <https://github.com/zarr-developers/zarr-extensions>`_
  repository, which the Zarr Format Working Group reviews. Named, reviewed,
  and interoperable in principle, but not something every implementation
  has.
- `ChunkKeyEncodingSupport.CUSTOM` -- anything else: third-party or local
  encodings that were never registered. Interoperable only with software
  that has the same code.

This distinction has teeth for chunk key encodings specifically. The spec's
usual escape hatch for unrecognized extensions -- marking them
``must_understand: false`` so a reader may ignore them -- is *not available*
here: "must_understand=False is not supported for the following extension
points: data type, chunk grid, and chunk key encoding." An implementation
that meets a chunk key encoding it does not understand has to fail, so
deciding up front which tiers are acceptable is the only graceful option.

Gate on it by filtering the registry::

    from zarr_chunk_key_encoding import (
        ChunkKeyEncodingSupport,
        registered_chunk_key_encodings,
    )

    allowed = registered_chunk_key_encodings(support=ChunkKeyEncodingSupport.CORE)
    if encoding_name not in allowed:
        raise SomeApplicationError(...)

or by asking an encoding directly, since the level is a class attribute::

    if encoding.support is not ChunkKeyEncodingSupport.CORE and not allow_extensions:
        raise SomeApplicationError(...)
"""

from enum import StrEnum
from typing import Final

__all__ = [
    "CORE_CHUNK_KEY_ENCODING_NAMES",
    "ChunkKeyEncodingSupport",
]


class ChunkKeyEncodingSupport(StrEnum):
    """How well-established a chunk key encoding is.

    A `enum.StrEnum`, so members compare equal to their string values and a
    level read from a configuration file (``"core"``) can be used directly.
    """

    CORE = "core"
    """Defined in the Zarr v3 core spec: ``default`` and ``v2``."""

    EXTENSION = "extension"
    """Registered in the `zarr-extensions` repository."""

    CUSTOM = "custom"
    """Neither core nor registered: third-party or local."""


CORE_CHUNK_KEY_ENCODING_NAMES: Final = frozenset({"default", "v2"})
"""The names the Zarr v3 core spec defines, a closed set.

Registration checks `ChunkKeyEncodingSupport.CORE` claims against this, so
the tier means "the spec says so" rather than "the author said so".
"""
