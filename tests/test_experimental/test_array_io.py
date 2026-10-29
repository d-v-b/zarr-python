"""Tests for the explicit-IO array layer in `zarr.experimental.array_io`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from zarr.api.asynchronous import open_array
from zarr.errors import ArrayNotFoundError
from zarr.experimental import ArrayIO, AsyncArrayIO
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat

METADATA_DOCUMENTS = ("zarr.json", ".zarray", ".zattrs", ".zgroup")


async def _store_keys(store: MemoryStore) -> set[str]:
    return {key async for key in store.list()}


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_chunks_before_metadata(zarr_format: ZarrFormat) -> None:
    """Declaring an array stores nothing; chunks, then metadata, can be stored
    in that order, and the result is an ordinary readable array."""
    store = MemoryStore()
    data = np.arange(16, dtype="uint16").reshape(4, 4)
    aio = await AsyncArrayIO.build(
        store,
        name="foo",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(2, 2),
        zarr_format=zarr_format,
        attributes={"answer": 42},
    )
    assert await _store_keys(store) == set()

    await aio.setitem(Ellipsis, data)
    keys = await _store_keys(store)
    assert len(keys) == 4
    assert not any(key.split("/")[-1] in METADATA_DOCUMENTS for key in keys)
    np.testing.assert_array_equal(await aio.getitem(Ellipsis), data)

    await aio.store_metadata()
    opened = await AsyncArrayIO.open(store, name="foo", zarr_format=zarr_format)
    assert opened.metadata.attributes == {"answer": 42}
    np.testing.assert_array_equal(await opened.getitem(Ellipsis), data)

    arr = await open_array(store=store, path="foo", mode="r", zarr_format=zarr_format)
    np.testing.assert_array_equal(await arr.getitem(Ellipsis), data)


def test_chunks_before_metadata_sync() -> None:
    """The synchronous `ArrayIO` supports the same chunks-first flow."""
    store = MemoryStore()
    data = np.arange(16, dtype="uint16").reshape(4, 4)
    aio = ArrayIO.build(store, name="foo", shape=data.shape, dtype=data.dtype, chunks=(2, 2))
    aio.setitem(Ellipsis, data)
    aio.store_metadata()
    opened = ArrayIO.open(store, name="foo")
    np.testing.assert_array_equal(opened.getitem(Ellipsis), data)
    np.testing.assert_array_equal(opened.to_array()[:], data)


def test_build_v2_shards_raises() -> None:
    with pytest.raises(ValueError, match="shard_shape"):
        ArrayIO.build(
            MemoryStore(),
            name="foo",
            shape=(4,),
            dtype="uint8",
            chunks=(2,),
            shards=(4,),
            zarr_format=2,
        )


async def test_open_missing_raises() -> None:
    with pytest.raises(ArrayNotFoundError):
        await AsyncArrayIO.open(MemoryStore(), name="foo")
