"""
Codec kind classification.

The Zarr v3 spec sorts codecs into three kinds by their position in a
pipeline: zero or more `array -> array` codecs, followed by exactly one
`array -> bytes` codec, followed by zero or more `bytes -> bytes` codecs.
This module provides one branded union per kind over the concrete codec
types this package defines, plus `TypeIs` guards that classify a codec
entry by its `name`.

Two classification surfaces with different contracts:

- The `TypeIs` guards (`is_array_array_codec`, ...) are *spelling-strict*:
  they answer `True` only for JSON shapes the codec's canonical type
  permits, because `TypeIs` narrowing must be sound — a bare `"transpose"`
  is not a `TransposeCodecObject`, so narrowing it to one would lie to the
  type checker. Use the guards to narrow a value to a concrete codec type.
- `codec_kind_of_name` classifies by *name alone*, ignoring spelling. Use
  it for pipeline-ordering semantics, where a known codec in an invalid
  spelling must still rank as its kind: two spellings of the same pipeline
  must never get different ordering verdicts, and a misspelled known codec
  must not be mistaken for an unknown extension (which would suppress the
  exactly-one-`array->bytes` count).

Neither surface validates. The guards do not re-check `configuration`
contents, and `codec_kind_of_name` does not check spelling at all;
spelling validity for known names is the job of the semantic rule layer
(`zarr_metadata.builder`). Codecs this package has no types for
(extension codecs from outside `zarr-extensions`) answer `False` to every
guard and `None` from `codec_kind_of_name`: an unknown codec has unknown
kind. Callers enforcing pipeline structure should treat "unclassifiable"
as its own case, not as any particular kind.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/index.html
"""

from typing import Final, Literal

from typing_extensions import TypeIs

from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
from zarr_metadata.v3.codec.blosc import BLOSC_CODEC_NAME, BloscCodecMetadata
from zarr_metadata.v3.codec.bytes import BYTES_CODEC_NAME, BytesCodecMetadata
from zarr_metadata.v3.codec.cast_value import CAST_VALUE_CODEC_NAME, CastValueCodecMetadata
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC_NAME, Crc32cCodecMetadata
from zarr_metadata.v3.codec.gzip import GZIP_CODEC_NAME, GzipCodecMetadata
from zarr_metadata.v3.codec.scale_offset import SCALE_OFFSET_CODEC_NAME, ScaleOffsetCodecMetadata
from zarr_metadata.v3.codec.sharding_indexed import (
    SHARDING_INDEXED_CODEC_NAME,
    ShardingIndexedCodecMetadata,
)
from zarr_metadata.v3.codec.transpose import TRANSPOSE_CODEC_NAME, TransposeCodecMetadata
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC_NAME, ZstdCodecMetadata

ArrayArrayCodecMetadata = TransposeCodecMetadata | CastValueCodecMetadata | ScaleOffsetCodecMetadata
"""Permitted JSON shapes of the `array -> array` codecs this package defines."""

ARRAY_ARRAY_CODEC_NAMES: Final = (
    TRANSPOSE_CODEC_NAME,
    CAST_VALUE_CODEC_NAME,
    SCALE_OFFSET_CODEC_NAME,
)
"""Tuple of the `name` field values of the known `array -> array` codecs."""

ArrayBytesCodecMetadata = BytesCodecMetadata | ShardingIndexedCodecMetadata
"""Permitted JSON shapes of the `array -> bytes` codecs this package defines."""

ARRAY_BYTES_CODEC_NAMES: Final = (BYTES_CODEC_NAME, SHARDING_INDEXED_CODEC_NAME)
"""Tuple of the `name` field values of the known `array -> bytes` codecs."""

BytesBytesCodecMetadata = (
    BloscCodecMetadata | Crc32cCodecMetadata | GzipCodecMetadata | ZstdCodecMetadata
)
"""Permitted JSON shapes of the `bytes -> bytes` codecs this package defines."""

BYTES_BYTES_CODEC_NAMES: Final = (
    BLOSC_CODEC_NAME,
    CRC32C_CODEC_NAME,
    GZIP_CODEC_NAME,
    ZSTD_CODEC_NAME,
)
"""Tuple of the `name` field values of the known `bytes -> bytes` codecs."""

KnownCodecMetadata = ArrayArrayCodecMetadata | ArrayBytesCodecMetadata | BytesBytesCodecMetadata
"""Permitted JSON shapes of every codec this package defines."""

_ARRAY_ARRAY_BARE_NAMES: Final = frozenset({SCALE_OFFSET_CODEC_NAME})
_ARRAY_BYTES_BARE_NAMES: Final = frozenset({BYTES_CODEC_NAME})
_BYTES_BYTES_BARE_NAMES: Final = frozenset({CRC32C_CODEC_NAME})
# The bare short-hand form is only part of a codec's canonical type when its
# spec permits it (no required configuration keys). A bare name outside these
# sets — e.g. the string "transpose" — is not valid metadata for that codec,
# so the guards answer False for it rather than narrowing to an object type
# the value does not have.


def _classify(
    codec: ZarrV3MetadataFieldJSON,
    object_names: tuple[str, ...],
    bare_names: frozenset[str],
) -> bool:
    if isinstance(codec, str):
        return codec in bare_names
    return codec["name"] in object_names


def is_array_array_codec(codec: ZarrV3MetadataFieldJSON) -> TypeIs[ArrayArrayCodecMetadata]:
    """Whether `codec` is a known `array -> array` codec (classified by name)."""
    return _classify(codec, ARRAY_ARRAY_CODEC_NAMES, _ARRAY_ARRAY_BARE_NAMES)


def is_array_bytes_codec(codec: ZarrV3MetadataFieldJSON) -> TypeIs[ArrayBytesCodecMetadata]:
    """Whether `codec` is a known `array -> bytes` codec (classified by name)."""
    return _classify(codec, ARRAY_BYTES_CODEC_NAMES, _ARRAY_BYTES_BARE_NAMES)


def is_bytes_bytes_codec(codec: ZarrV3MetadataFieldJSON) -> TypeIs[BytesBytesCodecMetadata]:
    """Whether `codec` is a known `bytes -> bytes` codec (classified by name)."""
    return _classify(codec, BYTES_BYTES_CODEC_NAMES, _BYTES_BYTES_BARE_NAMES)


def is_known_codec(codec: ZarrV3MetadataFieldJSON) -> TypeIs[KnownCodecMetadata]:
    """Whether `codec` is any codec this package defines (classified by name)."""
    return is_array_array_codec(codec) or is_array_bytes_codec(codec) or is_bytes_bytes_codec(codec)


CodecKind = Literal["array_array", "array_bytes", "bytes_bytes"]
"""The three pipeline positions the v3 spec sorts codecs into."""


def codec_kind_of_name(name: str) -> CodecKind | None:
    """The pipeline kind of the codec named `name`, or None if unknown.

    Classifies by name alone, with no spelling judgment: `"transpose"`
    answers `"array_array"` here even though the bare-string spelling is
    not valid transpose metadata (the `TypeIs` guards answer False for
    it). See the module docstring for when to use which surface.
    """
    if name in ARRAY_ARRAY_CODEC_NAMES:
        return "array_array"
    if name in ARRAY_BYTES_CODEC_NAMES:
        return "array_bytes"
    if name in BYTES_BYTES_CODEC_NAMES:
        return "bytes_bytes"
    return None


__all__ = [
    "ARRAY_ARRAY_CODEC_NAMES",
    "ARRAY_BYTES_CODEC_NAMES",
    "BYTES_BYTES_CODEC_NAMES",
    "ArrayArrayCodecMetadata",
    "ArrayBytesCodecMetadata",
    "BytesBytesCodecMetadata",
    "CodecKind",
    "KnownCodecMetadata",
    "codec_kind_of_name",
    "is_array_array_codec",
    "is_array_bytes_codec",
    "is_bytes_bytes_codec",
    "is_known_codec",
]
