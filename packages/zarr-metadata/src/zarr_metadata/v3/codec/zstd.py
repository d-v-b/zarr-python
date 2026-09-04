"""
Zstandard codec types.

See https://github.com/zarr-developers/zarr-specs/pull/256 (unmerged at
time of writing; the configuration shape below reflects the proposed
specification).
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

ZSTD_CODEC_NAME: Final = "zstd"
"""The `name` field value of the `zstd` codec."""

ZstdCodecName = Literal["zstd"]
"""Literal type of the `name` field of the `zstd` codec."""


class ZstdCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `zstd` codec.

    Both fields are required per the proposed specification.
    """

    level: int
    checksum: bool


class ZstdCodecObject(TypedDict, closed=True):
    """`zstd` codec metadata in object form."""

    name: ZstdCodecName
    configuration: ZstdCodecConfiguration
    must_understand: NotRequired[bool]


ZstdCodecMetadata = ZstdCodecObject
"""Permitted JSON shape for `zstd` codec metadata.

Both `level` and `checksum` are required, so only the object form is
valid; the short-hand-name form is not permitted by the spec for this codec.
"""

__all__ = [
    "ZSTD_CODEC_NAME",
    "ZstdCodecConfiguration",
    "ZstdCodecMetadata",
    "ZstdCodecName",
    "ZstdCodecObject",
]
