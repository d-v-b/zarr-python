from __future__ import annotations

import numpy as np
import pytest

from zarr.abc.codec import SupportsSyncCodec
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.dtype import get_data_type_from_native_dtype


def test_crc32c_codec_supports_sync() -> None:
    assert isinstance(Crc32cCodec(), SupportsSyncCodec)


def test_crc32c_codec_sync_roundtrip() -> None:
    codec = Crc32cCodec()
    arr = np.arange(100, dtype="float64")
    zdtype = get_data_type_from_native_dtype(arr.dtype)
    spec = ArraySpec(
        shape=arr.shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    buf = default_buffer_prototype().buffer.from_array_like(arr.view("B"))

    encoded = codec._encode_sync(buf, spec)
    assert encoded is not None
    decoded = codec._decode_sync(encoded, spec)
    result = np.frombuffer(decoded.as_numpy_array(), dtype="float64")
    np.testing.assert_array_equal(arr, result)


def _bytes_spec(nbytes: int) -> ArraySpec:
    zdtype = get_data_type_from_native_dtype(np.dtype("uint8"))
    return ArraySpec(
        shape=(nbytes,),
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )


@pytest.mark.parametrize(
    ("data", "checksum"),
    [
        # check value of CRC-32C (RFC 3720)
        (b"123456789", bytes.fromhex("839206e3")),
        (b"", bytes.fromhex("00000000")),
        (b"\x00" * 32, bytes.fromhex("aa36918a")),
    ],
)
def test_crc32c_encoded_bytes(data: bytes, checksum: bytes) -> None:
    """
    The crc32c codec appends the checksum as a little-endian uint32, whatever the host
    byte order, and decoding returns the original bytes.
    """
    codec = Crc32cCodec()
    spec = _bytes_spec(len(data))
    encoded = codec._encode_sync(default_buffer_prototype().buffer.from_bytes(data), spec)
    assert encoded is not None
    assert encoded.to_bytes() == data + checksum
    assert codec._decode_sync(encoded, spec).to_bytes() == data


def test_crc32c_decode_rejects_mismatched_checksum() -> None:
    """
    Decoding raises when the stored checksum does not match the data, including when
    the right checksum is stored in big-endian byte order.
    """
    codec = Crc32cCodec()
    data = b"123456789"
    spec = _bytes_spec(len(data))
    stored = data + bytes.fromhex("e3069283")
    with pytest.raises(ValueError, match="checksum do not match"):
        codec._decode_sync(default_buffer_prototype().buffer.from_bytes(stored), spec)
