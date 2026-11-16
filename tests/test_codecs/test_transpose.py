import numpy as np
import pytest

import zarr
from zarr import AsyncArray, config
from zarr.abc.codec import SupportsSyncCodec
from zarr.abc.store import Store
from zarr.codecs import ShardingCodec, TransposeCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import NDBuffer, default_buffer_prototype
from zarr.core.common import MemoryOrder
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.storage import StorePath

from .test_codecs import _AsyncArrayProxy


@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["F", "C"])
@pytest.mark.parametrize("runtime_read_order", ["F", "C"])
@pytest.mark.parametrize("with_sharding", [True, False])
@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
async def test_transpose(
    store: Store,
    input_order: MemoryOrder,
    runtime_write_order: MemoryOrder,
    runtime_read_order: MemoryOrder,
    with_sharding: bool,
) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((1, 32, 8), order=input_order)
    spath = StorePath(store, path="transpose")
    with config.set({"array.order": runtime_write_order}):
        a = await zarr.api.asynchronous.create_array(
            spath,
            shape=data.shape,
            chunks=(1, 16, 8) if with_sharding else (1, 32, 8),
            shards=(1, 32, 8) if with_sharding else None,
            dtype=data.dtype,
            fill_value=0,
            chunk_key_encoding={"name": "v2", "separator": "."},
            filters=[TransposeCodec(order=(2, 1, 0))],
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


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("order", [[1, 2, 0], [1, 2, 3, 0], [3, 2, 4, 0, 1]])
def test_transpose_non_self_inverse(store: Store, order: list[int]) -> None:
    shape = [i + 3 for i in range(len(order))]
    data = np.arange(0, np.prod(shape), dtype="uint16").reshape(shape)
    spath = StorePath(store, "transpose_non_self_inverse")
    a = zarr.create_array(
        spath,
        shape=data.shape,
        chunks=data.shape,
        dtype=data.dtype,
        fill_value=0,
        filters=[TransposeCodec(order=order)],
    )
    a[:, :] = data
    read_data = a[:, :]
    assert np.array_equal(data, read_data)


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_transpose_invalid(
    store: Store,
) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((1, 32, 8))
    spath = StorePath(store, "transpose_invalid")
    for order in [(1, 0), (3, 2, 1), (3, 3, 1), "F", "C"]:
        with pytest.raises((ValueError, TypeError)):
            zarr.create_array(
                spath,
                shape=data.shape,
                chunks=(1, 32, 8),
                dtype=data.dtype,
                fill_value=0,
                chunk_key_encoding={"name": "v2", "separator": "."},
                filters=[TransposeCodec(order=order)],  # type: ignore[arg-type]
            )


# Each case pairs a chunk grid with array->array filters and a sharding serializer whose
# inner chunk shape divides the grid only as the filters leave it, not as stored.
# The rectilinear cases use a square array, whose shape a transpose leaves unchanged,
# so only the permuted edge lists tell the grids apart.
TRANSPOSED_SHARDING_CASES = [
    pytest.param((12, 40), (6, 20), [TransposeCodec(order=(1, 0))], (10, 3), id="regular"),
    pytest.param(
        (12, 40),
        (6, 20),
        [TransposeCodec(order=(1, 0)), TransposeCodec(order=(1, 0))],
        (3, 10),
        id="regular-transposed-twice",
    ),
    pytest.param(
        (4, 12, 40), (2, 6, 20), [TransposeCodec(order=(1, 2, 0))], (3, 5, 2), id="regular-3d"
    ),
    pytest.param(
        (12, 12), [[4, 8], [6, 6]], [TransposeCodec(order=(1, 0))], (6, 4), id="rectilinear"
    ),
    pytest.param(
        (12, 40), [6, [10, 10, 20]], [TransposeCodec(order=(1, 0))], (10, 3), id="rectilinear-int"
    ),
]


# A transpose before the sharding codec disables partial reads and writes, which warns.
@pytest.mark.filterwarnings("ignore:Combining a `sharding_indexed` codec")
@pytest.mark.parametrize(("shape", "chunks", "filters", "inner_chunks"), TRANSPOSED_SHARDING_CASES)
def test_sharding_after_transpose(
    shape: tuple[int, ...],
    chunks: tuple[int, ...] | list[int | list[int]],
    filters: list[TransposeCodec],
    inner_chunks: tuple[int, ...],
) -> None:
    """
    A serializer after a transpose filter is validated against the transposed chunk
    grid, and the array round-trips.
    """
    data = np.arange(np.prod(shape), dtype="uint16").reshape(shape)
    with config.set({"array.rectilinear_chunks": True}):
        a = zarr.create_array(
            zarr.storage.MemoryStore(),
            shape=shape,
            chunks=chunks,
            dtype=data.dtype,
            filters=filters,
            serializer=ShardingCodec(chunk_shape=inner_chunks),
        )
        a[...] = data
        assert np.array_equal(a[...], data)


@pytest.mark.parametrize(
    ("shape", "chunks", "filters", "inner_chunks"),
    [
        pytest.param((12, 40), (6, 20), [TransposeCodec(order=(1, 0))], (3, 10), id="regular"),
        pytest.param(
            (12, 12), [[4, 8], [6, 6]], [TransposeCodec(order=(1, 0))], (4, 6), id="rectilinear"
        ),
    ],
)
def test_sharding_after_transpose_indivisible(
    shape: tuple[int, ...],
    chunks: tuple[int, ...] | list[list[int]],
    filters: list[TransposeCodec],
    inner_chunks: tuple[int, ...],
) -> None:
    """
    An inner chunk shape that divides the stored chunk grid but not the transposed one
    is rejected.
    """
    with (
        config.set({"array.rectilinear_chunks": True}),
        pytest.raises(ValueError, match="is not divisible by the shard's inner chunk size"),
    ):
        zarr.create_array(
            zarr.storage.MemoryStore(),
            shape=shape,
            chunks=chunks,
            dtype="uint16",
            filters=filters,
            serializer=ShardingCodec(chunk_shape=inner_chunks),
        )


def test_transpose_codec_supports_sync() -> None:
    assert isinstance(TransposeCodec(order=(0, 1)), SupportsSyncCodec)


def test_transpose_codec_sync_roundtrip() -> None:
    codec = TransposeCodec(order=(1, 0))
    arr = np.arange(12, dtype="float64").reshape(3, 4)
    zdtype = get_data_type_from_native_dtype(arr.dtype)
    spec = ArraySpec(
        shape=arr.shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    nd_buf: NDBuffer = default_buffer_prototype().nd_buffer.from_numpy_array(arr)

    encoded = codec._encode_sync(nd_buf, spec)
    assert encoded is not None
    resolved_spec = codec.resolve_metadata(spec)
    decoded = codec._decode_sync(encoded, resolved_spec)
    np.testing.assert_array_equal(arr, decoded.as_numpy_array())
