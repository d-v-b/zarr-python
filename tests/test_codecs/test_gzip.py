import numpy as np
import pytest

import zarr
from tests.conftest import LOCAL_MEMORY_STORES, gzip_streams_equal_except_mtime
from zarr.abc.store import Store
from zarr.codecs import GzipCodec
from zarr.storage import StorePath


@pytest.mark.parametrize("store", LOCAL_MEMORY_STORES, indirect=True)
def test_gzip(store: Store) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = zarr.create_array(
        StorePath(store),
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        compressors=GzipCodec(),
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


def test_gzip_streams_equal_except_mtime() -> None:
    prefix = b"\x1f\x8b\x08\x00"
    mtime = b"\x01\x02\x03\x04"
    suffix = b"\x00\xff\x10\x20"

    # Identical streams are equal.
    assert gzip_streams_equal_except_mtime(
        prefix + mtime + suffix,
        prefix + mtime + suffix,
    )

    # Streams with different MTIME values are still equal.
    assert gzip_streams_equal_except_mtime(
        prefix + b"\x01\x02\x03\x04" + suffix,
        prefix + b"\x05\x06\x07\x08" + suffix,
    )

    # Differences after MTIME are detected.
    assert not gzip_streams_equal_except_mtime(
        prefix + mtime + suffix,
        prefix + mtime + b"\x00\xff\x10\x21",
    )

    # Differences before MTIME are detected.
    assert not gzip_streams_equal_except_mtime(
        prefix + mtime + suffix,
        b"\x1f\x8b\x09\x00" + mtime + suffix,
    )

    # Different lengths are detected.
    assert not gzip_streams_equal_except_mtime(
        prefix + mtime + suffix,
        prefix + mtime + suffix + b"\x00",
    )
