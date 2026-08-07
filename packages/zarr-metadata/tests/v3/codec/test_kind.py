"""Tests for codec kind classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr_metadata.v3.codec.kind import (
    codec_kind_of_name,
    is_array_array_codec,
    is_array_bytes_codec,
    is_bytes_bytes_codec,
    is_known_codec,
)

if TYPE_CHECKING:
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON

# (codec entry, expected kind) — kind is one of "aa", "ab", "bb", or None
# for entries no guard should claim. Object forms use minimal spec-shaped
# configurations; bare strings appear only for codecs whose canonical type
# permits the short-hand form.
CASES: dict[str, tuple[ZarrV3MetadataFieldJSON, str | None]] = {
    "transpose": ({"name": "transpose", "configuration": {"order": (1, 0)}}, "aa"),
    "cast_value": (
        {"name": "cast_value", "configuration": {"data_type": "uint8"}},
        "aa",
    ),
    "scale_offset": (
        {"name": "scale_offset", "configuration": {"scale": 2, "offset": 1}},
        "aa",
    ),
    "scale_offset-bare": ("scale_offset", "aa"),
    "bytes": ({"name": "bytes", "configuration": {"endian": "little"}}, "ab"),
    "bytes-bare": ("bytes", "ab"),
    "sharding_indexed": (
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": (2, 2),
                "codecs": ("bytes",),
                "index_codecs": ("bytes", "crc32c"),
            },
        },
        "ab",
    ),
    "blosc": (
        {
            "name": "blosc",
            "configuration": {
                "cname": "zstd",
                "clevel": 5,
                "shuffle": "shuffle",
                "blocksize": 0,
            },
        },
        "bb",
    ),
    "crc32c": ({"name": "crc32c"}, "bb"),
    "crc32c-bare": ("crc32c", "bb"),
    "gzip": ({"name": "gzip", "configuration": {"level": 5}}, "bb"),
    "zstd": ({"name": "zstd", "configuration": {"level": 3, "checksum": False}}, "bb"),
    # Unknown codecs have unknown kind.
    "unknown-object": ({"name": "lightspeed"}, None),
    "unknown-bare": ("lightspeed", None),
    # A bare name is only classified when the codec's spec permits the
    # short-hand form; "transpose" as a bare string is not valid transpose
    # metadata, so no guard claims it.
    "transpose-bare-invalid": ("transpose", None),
    "blosc-bare-invalid": ("blosc", None),
}


@pytest.mark.parametrize(("codec", "kind"), CASES.values(), ids=list(CASES))
def test_classification(codec: ZarrV3MetadataFieldJSON, kind: str | None) -> None:
    assert is_array_array_codec(codec) is (kind == "aa")
    assert is_array_bytes_codec(codec) is (kind == "ab")
    assert is_bytes_bytes_codec(codec) is (kind == "bb")
    assert is_known_codec(codec) is (kind is not None)


# (codec name, expected kind) — `codec_kind_of_name` classifies by name
# alone, so a name answers its kind even where the bare spelling is not
# valid metadata for that codec (unlike the spelling-strict guards above).
NAME_CASES: dict[str, str | None] = {
    "transpose": "array_array",
    "cast_value": "array_array",
    "scale_offset": "array_array",
    "bytes": "array_bytes",
    "sharding_indexed": "array_bytes",
    "blosc": "bytes_bytes",
    "crc32c": "bytes_bytes",
    "gzip": "bytes_bytes",
    "zstd": "bytes_bytes",
    "lightspeed": None,
}


@pytest.mark.parametrize(("name", "kind"), NAME_CASES.items(), ids=list(NAME_CASES))
def test_kind_of_name(name: str, kind: str | None) -> None:
    assert codec_kind_of_name(name) == kind
