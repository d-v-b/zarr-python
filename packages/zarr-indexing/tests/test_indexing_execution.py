"""Exercise the optional indexing prototype through real Zarr codec pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from zarr.core.buffer.core import default_buffer_prototype

from zarr_indexing import _execution as execution

if TYPE_CHECKING:
    from zarr.core.indexing import Indexer

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("pipeline", ["BatchedCodecPipeline", "FusedCodecPipeline"])
@pytest.mark.parametrize("layout", ["v2", "v3", "sharded"])
@pytest.mark.parametrize(
    "case", ["basic", "integer", "reverse", "sorted", "components", "orthogonal"]
)
async def test_execution_codec_read_write(pipeline: str, layout: str, case: str) -> None:
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    selection: Any
    if case == "sorted":
        shape, chunks = (1003,), (100,)
        selection, mode = (np.arange(1, 1003),), "vectorized"
    else:
        shape, chunks = (7, 9, 5), (3, 4, 2)
        selection, mode = {
            "basic": ((slice(1, 7, 2), slice(None), slice(1, 5)), "basic"),
            "integer": ((2, slice(1, 8, 2), slice(None)), "basic"),
            "reverse": ((slice(None, None, -1), slice(None), slice(None)), "basic"),
            "orthogonal": ((3, np.array([1, 2]), slice(None)), "orthogonal"),
            "components": (
                (
                    np.array([6, 0])[:, None],
                    np.array([8, 2])[:, None],
                    np.array([4, 0, 2])[None, :],
                ),
                "vectorized",
            ),
        }[case]
    source = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    kwargs: dict[str, Any] = {"zarr_format": 2 if layout == "v2" else 3}
    if layout == "sharded":
        kwargs["shards"] = tuple(c * 2 for c in chunks)
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline." + pipeline}):
        array = zarr.create_array(
            store=zarr.storage.MemoryStore(), shape=shape, chunks=chunks, dtype="int64", **kwargs
        )
        array[:] = source
        async_array = array._async_array
        # The pipeline processes shard-sized buffers for sharded arrays.
        grids = async_array._chunk_grid._dimensions
        plan = execution.execute_selection(selection, shape, grids, mode=mode)
        if layout == "sharded":
            plan = plan.lower("shard")
        indexer = cast("Indexer", plan)
        prototype = default_buffer_prototype()
        result = await async_array._get_selection(indexer, prototype=prototype)
        np.testing.assert_array_equal(result, source[selection])
        replacement = np.arange(np.prod(plan.shape)).reshape(plan.shape) + 10000
        write_plan = execution.execute_selection(selection, shape, grids, mode=mode, access="write")
        if layout == "sharded":
            write_plan = write_plan.lower("shard")
        await async_array._set_selection(
            cast("Indexer", write_plan), replacement, prototype=prototype
        )
        expected = source.copy()
        expected[selection] = replacement
        np.testing.assert_array_equal(await async_array.getitem(Ellipsis), expected)


@pytest.mark.parametrize("pipeline", ["BatchedCodecPipeline", "FusedCodecPipeline"])
@pytest.mark.parametrize("step", [1, 2])
async def test_boundary_complete_write_skips_read(pipeline: str, step: int) -> None:
    store = zarr.storage.LoggingStore(zarr.storage.MemoryStore())
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline." + pipeline}):
        array = zarr.create_array(store=store, shape=(7,), chunks=(3,), dtype="int64")
        array[:] = np.arange(7)
        plan = execution.execute_selection(
            slice(6, 7, step), (7,), array._async_array._chunk_grid._dimensions, access="write"
        )
        store.counter.clear()
        await array._async_array._set_selection(
            cast("Indexer", plan), np.array([99]), prototype=default_buffer_prototype()
        )
        assert store.counter["get"] == 0
        assert store.counter["get_sync"] == 0
        np.testing.assert_array_equal(
            await array._async_array.getitem(Ellipsis), [0, 1, 2, 3, 4, 5, 99]
        )
