"""Static regression checks for successful and unsuccessful codec guards."""

from typing import Literal, assert_type

from zarr_metadata.v3.codec.kind import (
    ArrayArrayCodecMetadata,
    ArrayBytesCodecMetadata,
    BytesBytesCodecMetadata,
    KnownCodecMetadata,
    is_array_array_codec,
    is_array_bytes_codec,
    is_bytes_bytes_codec,
    is_known_codec,
)


def check_array_array(codec: ArrayArrayCodecMetadata | Literal["unknown"]) -> None:
    if is_array_array_codec(codec):
        assert_type(codec, ArrayArrayCodecMetadata)
    else:
        # Runtime validation can reject typed values: e.g. bool is an int
        # subtype, but is not a JSON integer in a transpose order.
        assert_type(codec, ArrayArrayCodecMetadata | Literal["unknown"])


def check_array_bytes(codec: ArrayBytesCodecMetadata | Literal["unknown"]) -> None:
    if is_array_bytes_codec(codec):
        assert_type(codec, ArrayBytesCodecMetadata)
    else:
        assert_type(codec, ArrayBytesCodecMetadata | Literal["unknown"])


def check_bytes_bytes(codec: BytesBytesCodecMetadata | Literal["unknown"]) -> None:
    if is_bytes_bytes_codec(codec):
        assert_type(codec, BytesBytesCodecMetadata)
    else:
        assert_type(codec, BytesBytesCodecMetadata | Literal["unknown"])


def check_known(codec: KnownCodecMetadata | Literal["unknown"]) -> None:
    if is_known_codec(codec):
        assert_type(codec, KnownCodecMetadata)
    else:
        assert_type(codec, KnownCodecMetadata | Literal["unknown"])
