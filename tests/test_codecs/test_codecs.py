from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
import zarr.api
import zarr.api.asynchronous
from tests.conftest import LOCAL_MEMORY_STORES
from zarr import Array, AsyncArray, config
from zarr.abc.codec import SupportsSyncCodec
from zarr.codecs import (
    BloscCodec,
    BytesCodec,
    GzipCodec,
    ShardingCodec,
    TransposeCodec,
    ZstdCodec,
)
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.codecs.vlen_utf8 import VLenBytesCodec, VLenUTF8Codec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.core.indexing import BasicSelection, decode_morton, morton_order_iter
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.dtype import UInt8
from zarr.errors import ZarrUserWarning
from zarr.storage import StorePath

if TYPE_CHECKING:
    from zarr.abc.codec import Codec
    from zarr.abc.store import Store
    from zarr.core.buffer.core import Buffer, NDArrayLikeOrScalar
    from zarr.core.common import MemoryOrder
    from zarr.types import AnyAsyncArray


@dataclass(frozen=True)
class _AsyncArrayProxy:
    array: AnyAsyncArray

    def __getitem__(self, selection: BasicSelection) -> _AsyncArraySelectionProxy:
        return _AsyncArraySelectionProxy(self.array, selection)


@dataclass(frozen=True)
class _AsyncArraySelectionProxy:
    array: AnyAsyncArray
    selection: BasicSelection

    async def get(self) -> NDArrayLikeOrScalar:
        return await self.array.getitem(self.selection)

    async def set(self, value: np.ndarray[Any, Any]) -> None:
        return await self.array.setitem(self.selection, value)


def order_from_dim(order: MemoryOrder, ndim: int) -> tuple[int, ...]:
    if order == "F":
        return tuple(ndim - x - 1 for x in range(ndim))
    else:
        return tuple(range(ndim))


def test_sharding_pickle() -> None:
    """
    Test that sharding codecs can be pickled
    """


@pytest.mark.parametrize("store", LOCAL_MEMORY_STORES, indirect=True)
@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("store_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["F", "C"])
@pytest.mark.parametrize("runtime_read_order", ["F", "C"])
@pytest.mark.parametrize("with_sharding", [True, False])
async def test_order(
    store: Store,
    input_order: MemoryOrder,
    store_order: MemoryOrder,
    runtime_write_order: MemoryOrder,
    runtime_read_order: MemoryOrder,
    with_sharding: bool,
) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((32, 8), order=input_order)
    path = "order"
    spath = StorePath(store, path=path)

    a = await zarr.api.asynchronous.create_array(
        spath,
        shape=data.shape,
        chunks=(16, 8) if with_sharding else (32, 8),
        shards=(32, 8) if with_sharding else None,
        dtype=data.dtype,
        fill_value=0,
        chunk_key_encoding={"name": "v2", "separator": "."},
        filters=[TransposeCodec(order=order_from_dim(store_order, data.ndim))],
        config={"order": runtime_write_order},
    )

    await _AsyncArrayProxy(a)[:, :].set(data)
    read_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, read_data)

    with config.set({"array.order": runtime_read_order}):
        a = await AsyncArray.open(
            spath,
        )
    read_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, read_data)

    assert isinstance(read_data, np.ndarray)
    if runtime_read_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("store", LOCAL_MEMORY_STORES, indirect=True)
@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["F", "C"])
@pytest.mark.parametrize("runtime_read_order", ["F", "C"])
@pytest.mark.parametrize("with_sharding", [True, False])
def test_order_implicit(
    store: Store,
    input_order: MemoryOrder,
    runtime_write_order: MemoryOrder,
    runtime_read_order: MemoryOrder,
    with_sharding: bool,
) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16), order=input_order)
    path = "order_implicit"
    spath = StorePath(store, path)

    with config.set({"array.order": runtime_write_order}):
        a = zarr.create_array(
            spath,
            shape=data.shape,
            chunks=(8, 8) if with_sharding else (16, 16),
            shards=(16, 16) if with_sharding else None,
            dtype=data.dtype,
            fill_value=0,
        )

    a[:, :] = data

    with config.set({"array.order": runtime_read_order}):
        a = Array.open(spath)
    read_data = a[:, :]
    assert np.array_equal(data, read_data)

    assert isinstance(read_data, np.ndarray)
    if runtime_read_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("store", LOCAL_MEMORY_STORES, indirect=True)
def test_open(store: Store) -> None:
    spath = StorePath(store)
    a = zarr.create_array(
        spath,
        shape=(16, 16),
        chunks=(16, 16),
        dtype="int32",
        fill_value=0,
    )
    b = Array.open(spath)
    assert a.metadata == b.metadata


def test_morton_exact_order() -> None:
    """Test exact morton ordering for power-of-2 shapes."""
    assert list(morton_order_iter((2, 2))) == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert list(morton_order_iter((2, 2, 2))) == [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
    assert list(morton_order_iter((2, 2, 2, 2))) == [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (1, 1, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (1, 1, 1, 0),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 1),
        (1, 1, 0, 1),
        (0, 0, 1, 1),
        (1, 0, 1, 1),
        (0, 1, 1, 1),
        (1, 1, 1, 1),
    ]


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2, 2),
        (5, 2),
        (2, 5),
        (2, 9, 2),
        (3, 2, 12),
        (2, 5, 1),
        (4, 3, 6, 2, 7),
        (3, 2, 1, 6, 4, 5, 2),
        (1,),
        (1, 1),
        (5, 1, 3),
        (1, 4, 1, 2),
        (5, 5, 5),  # triggers argsort strategy (n_z/n_total > 4)
    ],
)
def test_morton_is_permutation(shape: tuple[int, ...]) -> None:
    """Test that morton_order_iter produces every valid coordinate exactly once."""
    import itertools

    from zarr.core.common import product

    order = list(morton_order_iter(shape))
    expected_len = product(shape)
    # completeness: every valid coordinate is present
    assert len(order) == expected_len
    # no duplicates
    assert len(set(order)) == expected_len
    # all coordinates are within bounds
    assert all(all(c < s for c, s in zip(coord, shape, strict=True)) for coord in order)
    # the set of coordinates equals the full cartesian product
    assert set(order) == set(itertools.product(*(range(s) for s in shape)))


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (4, 4),
        (2, 2, 2),
        (4, 4, 4),
        (2, 2, 2, 2),
    ],
)
def test_morton_ordering(shape: tuple[int, ...]) -> None:
    """Test that the iteration order matches consecutive decode_morton outputs.

    For power-of-2 shapes, every decode_morton output is in-bounds,
    so the ordering should be exactly decode_morton(0), decode_morton(1), ...
    """

    order = list(morton_order_iter(shape))
    for i, coord in enumerate(order):
        assert coord == decode_morton(i, shape)


@pytest.mark.parametrize("store", LOCAL_MEMORY_STORES, indirect=True)
def test_write_partial_chunks(store: Store) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))
    spath = StorePath(store)
    a = zarr.create_array(
        spath,
        shape=data.shape,
        chunks=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)


@pytest.mark.parametrize("store", LOCAL_MEMORY_STORES, indirect=True)
async def test_delete_empty_chunks(store: Store) -> None:
    data = np.ones((16, 16))
    path = "delete_empty_chunks"
    spath = StorePath(store, path)
    a = await zarr.api.asynchronous.create_array(
        spath,
        shape=data.shape,
        chunks=(32, 32),
        dtype=data.dtype,
        fill_value=1,
    )
    await _AsyncArrayProxy(a)[:16, :16].set(np.zeros((16, 16)))
    await _AsyncArrayProxy(a)[:16, :16].set(data)
    assert np.array_equal(await _AsyncArrayProxy(a)[:16, :16].get(), data)
    assert await store.get(f"{path}/c0/0", prototype=default_buffer_prototype()) is None


@pytest.mark.parametrize("store", LOCAL_MEMORY_STORES, indirect=True)
async def test_dimension_names(store: Store) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))
    path = "dimension_names"
    spath = StorePath(store, path)
    await zarr.api.asynchronous.create_array(
        spath,
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        dimension_names=("x", "y"),
    )

    assert isinstance(
        meta := (await zarr.api.asynchronous.open_array(store=spath)).metadata, ArrayV3Metadata
    )
    assert meta.dimension_names == (
        "x",
        "y",
    )
    path2 = "dimension_names2"
    spath2 = StorePath(store, path2)
    await zarr.api.asynchronous.create_array(
        spath2,
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    assert isinstance(meta := (await AsyncArray.open(spath2)).metadata, ArrayV3Metadata)
    assert meta.dimension_names is None
    zarr_json_buffer = await store.get(f"{path2}/zarr.json", prototype=default_buffer_prototype())
    assert zarr_json_buffer is not None
    assert "dimension_names" not in json.loads(zarr_json_buffer.to_bytes())


@pytest.mark.parametrize(
    "codecs",
    [
        (BytesCodec(), TransposeCodec(order=order_from_dim("F", 2))),
        (TransposeCodec(order=order_from_dim("F", 2)),),
    ],
)
def test_invalid_metadata(codecs: tuple[Codec, ...]) -> None:
    shape = (16,)
    chunks = (16,)
    data_type = UInt8()
    with pytest.raises(ValueError, match="The `order` tuple must have as many entries"):
        ArrayV3Metadata(
            shape=shape,
            chunk_grid={"name": "regular", "configuration": {"chunk_shape": chunks}},
            chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
            fill_value=0,
            data_type=data_type,
            codecs=codecs,
            attributes={},
            dimension_names=None,
        )


def test_invalid_metadata_create_array() -> None:
    with pytest.warns(
        ZarrUserWarning,
        match="codec disables partial reads and writes, which may lead to inefficient performance",
    ):
        zarr.create_array(
            {},
            shape=(16, 16),
            chunks=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            serializer=ShardingCodec(chunk_shape=(8, 8)),
            compressors=[
                GzipCodec(),
            ],
        )


@pytest.mark.parametrize("store", LOCAL_MEMORY_STORES, indirect=True)
async def test_resize(store: Store) -> None:
    data = np.zeros((16, 18), dtype="uint16")
    path = "resize"
    spath = StorePath(store, path)
    a = await zarr.api.asynchronous.create_array(
        spath,
        shape=data.shape,
        chunks=(10, 10),
        dtype=data.dtype,
        chunk_key_encoding={"name": "v2", "separator": "."},
        fill_value=1,
    )

    await _AsyncArrayProxy(a)[:16, :18].set(data)
    assert await store.get(f"{path}/1.1", prototype=default_buffer_prototype()) is not None
    assert await store.get(f"{path}/0.0", prototype=default_buffer_prototype()) is not None
    assert await store.get(f"{path}/0.1", prototype=default_buffer_prototype()) is not None
    assert await store.get(f"{path}/1.0", prototype=default_buffer_prototype()) is not None

    await a.resize((10, 12))
    assert a.metadata.shape == (10, 12)
    assert a.shape == (10, 12)
    assert await store.get(f"{path}/0.0", prototype=default_buffer_prototype()) is not None
    assert await store.get(f"{path}/0.1", prototype=default_buffer_prototype()) is not None
    assert await store.get(f"{path}/1.0", prototype=default_buffer_prototype()) is None
    assert await store.get(f"{path}/1.1", prototype=default_buffer_prototype()) is None


# Consolidated codec sync-support tests. These replace the per-codec
# ``test_*_codec_supports_sync`` and ``test_*_codec_sync_roundtrip`` duplicates that were
# byte-identical (apart from the codec instance) across the individual codec test modules.
@pytest.mark.parametrize(
    "codec",
    [
        ZstdCodec(),
        Crc32cCodec(),
        GzipCodec(),
        BytesCodec(),
        BloscCodec(),
        TransposeCodec(order=(0, 1)),
        VLenUTF8Codec(),
        VLenBytesCodec(),
    ],
    ids=lambda codec: type(codec).__name__,
)
def test_codec_supports_sync(codec: Codec) -> None:
    assert isinstance(codec, SupportsSyncCodec)


@pytest.mark.parametrize(
    "codec",
    [
        ZstdCodec(level=1),
        Crc32cCodec(),
        GzipCodec(level=1),
        BloscCodec(typesize=8),
    ],
    ids=lambda codec: type(codec).__name__,
)
def test_bytes_to_bytes_codec_sync_roundtrip(codec: SupportsSyncCodec[Buffer, Buffer]) -> None:
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
