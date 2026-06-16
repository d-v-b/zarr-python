import numpy as np
import pytest

import zarr
from tests.conftest import LOCAL_MEMORY_STORES
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
