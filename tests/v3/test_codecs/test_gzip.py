import numpy as np
import pytest

from zarr import Array
from zarr.abc.store import Store
from zarr.codecs import BytesCodec, GzipCodec
from zarr.store.common import StorePath


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
def test_gzip(store: Store) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        StorePath(store),
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[BytesCodec(), GzipCodec()],
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])
