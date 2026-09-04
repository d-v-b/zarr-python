"""
Zarr v3 codec spec types.

Each codec defined by the spec or by zarr-extensions has its own submodule
(`blosc`, `bytes`, `cast_value`, `crc32c`, `gzip`, `scale_offset`,
`sharding_indexed`, `transpose`, `zstd`).

The `<X>CodecMetadata` aliases re-exported here are the canonical type for
each codec's permitted JSON shapes (object form plus, where the spec allows,
a bare-string short-hand form). For the underlying `<X>CodecObject`,
`<X>CodecConfiguration`, etc., import directly from the leaf submodule.

For the field-level "any codec entry" alias (used in array metadata's
`codecs` list and in sharding's inner pipelines), import `ZarrV3MetadataFieldJSON`
from `zarr_metadata.v3`.

The `kind` submodule sorts the known codecs into the spec's three pipeline
kinds (`array -> array`, `array -> bytes`, `bytes -> bytes`); its kind
unions and `TypeIs` classification guards are re-exported here.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/index.html
"""

from zarr_metadata.v3.codec.blosc import BloscCodecMetadata
from zarr_metadata.v3.codec.bytes import BytesCodecMetadata
from zarr_metadata.v3.codec.cast_value import CastValueCodecMetadata
from zarr_metadata.v3.codec.crc32c import Crc32cCodecMetadata
from zarr_metadata.v3.codec.gzip import GzipCodecMetadata
from zarr_metadata.v3.codec.kind import (
    ARRAY_ARRAY_CODEC_NAMES,
    ARRAY_BYTES_CODEC_NAMES,
    BYTES_BYTES_CODEC_NAMES,
    ArrayArrayCodecMetadata,
    ArrayBytesCodecMetadata,
    BytesBytesCodecMetadata,
    KnownCodecMetadata,
    is_array_array_codec,
    is_array_bytes_codec,
    is_bytes_bytes_codec,
    is_known_codec,
)
from zarr_metadata.v3.codec.scale_offset import ScaleOffsetCodecMetadata
from zarr_metadata.v3.codec.sharding_indexed import ShardingIndexedCodecMetadata
from zarr_metadata.v3.codec.transpose import TransposeCodecMetadata
from zarr_metadata.v3.codec.zstd import ZstdCodecMetadata

__all__ = [
    "ARRAY_ARRAY_CODEC_NAMES",
    "ARRAY_BYTES_CODEC_NAMES",
    "BYTES_BYTES_CODEC_NAMES",
    "ArrayArrayCodecMetadata",
    "ArrayBytesCodecMetadata",
    "BloscCodecMetadata",
    "BytesBytesCodecMetadata",
    "BytesCodecMetadata",
    "CastValueCodecMetadata",
    "Crc32cCodecMetadata",
    "GzipCodecMetadata",
    "KnownCodecMetadata",
    "ScaleOffsetCodecMetadata",
    "ShardingIndexedCodecMetadata",
    "TransposeCodecMetadata",
    "ZstdCodecMetadata",
    "is_array_array_codec",
    "is_array_bytes_codec",
    "is_bytes_bytes_codec",
    "is_known_codec",
]
