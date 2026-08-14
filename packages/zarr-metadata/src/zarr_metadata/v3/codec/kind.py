"""Classify Zarr v3 codecs by pipeline kind.

The `TypeIs` guards are shape-exact against canonical codec TypedDicts;
normalize JSON arrays to tuples before using them. `codec_kind_of_name`
classifies a known name without validating its object shape, which is
useful for pipeline ordering. Unknown names return no kind.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/index.html
"""

from typing import Final, Literal

from typing_extensions import TypeIs

from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
from zarr_metadata.v3._shape import is_valid_known_codec_name
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


def is_array_array_codec(codec: ZarrV3MetadataFieldJSON) -> TypeIs[ArrayArrayCodecMetadata]:
    """Whether `codec` is an instance of a known `array -> array` codec type."""
    return is_valid_known_codec_name(codec) in ARRAY_ARRAY_CODEC_NAMES


def is_array_bytes_codec(codec: ZarrV3MetadataFieldJSON) -> TypeIs[ArrayBytesCodecMetadata]:
    """Whether `codec` is an instance of a known `array -> bytes` codec type."""
    return is_valid_known_codec_name(codec) in ARRAY_BYTES_CODEC_NAMES


def is_bytes_bytes_codec(codec: ZarrV3MetadataFieldJSON) -> TypeIs[BytesBytesCodecMetadata]:
    """Whether `codec` is an instance of a known `bytes -> bytes` codec type."""
    return is_valid_known_codec_name(codec) in BYTES_BYTES_CODEC_NAMES


def is_known_codec(codec: ZarrV3MetadataFieldJSON) -> TypeIs[KnownCodecMetadata]:
    """Whether `codec` is an instance of any codec type this package defines."""
    return is_valid_known_codec_name(codec) is not None


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
