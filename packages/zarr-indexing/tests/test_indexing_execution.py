"""Drive real Zarr codec pipelines with partition rows in place of Zarr's indexers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from zarr_indexing._indexer import PartitionIndexer

zarr = pytest.importorskip("zarr")
sync = pytest.importorskip("zarr.core.sync").sync

if TYPE_CHECKING:
    from zarr.core.indexing import Indexer


@pytest.mark.parametrize("pipeline", ["BatchedCodecPipeline", "FusedCodecPipeline"])
@pytest.mark.parametrize("layout", ["v2", "v3", "sharded"])
@pytest.mark.parametrize(
    "case", ["basic", "integer", "reverse", "sorted", "components", "orthogonal"]
)
def test_execution_codec_read_write(pipeline: str, layout: str, case: str) -> None:
    from zarr.core.buffer.core import default_buffer_prototype

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
        indexer = cast(
            "Indexer", PartitionIndexer.from_selection(selection, shape, grids, mode=mode)
        )
        prototype = default_buffer_prototype()
        result = sync(async_array._get_selection(indexer, prototype=prototype))
        np.testing.assert_array_equal(result, source[selection])
        replacement = np.arange(np.prod(indexer.shape)).reshape(indexer.shape) + 10000
        sync(async_array._set_selection(indexer, replacement, prototype=prototype))
        expected = source.copy()
        expected[selection] = replacement
        np.testing.assert_array_equal(sync(async_array.getitem(Ellipsis)), expected)


@pytest.mark.parametrize("pipeline", ["BatchedCodecPipeline", "FusedCodecPipeline"])
@pytest.mark.parametrize(
    ("selection", "value", "expected"),
    [
        (slice(6, 7), [99], [0, 1, 2, 3, 4, 5, 99]),
        (slice(6, 7, 2), [99], [0, 1, 2, 3, 4, 5, 99]),
        (slice(None, None, -1), list(range(100, 107)), [106, 105, 104, 103, 102, 101, 100]),
    ],
)
def test_complete_chunk_writes_skip_the_read(
    pipeline: str, selection: slice, value: list[int], expected: list[int]
) -> None:
    """Every touched chunk is covered exactly once, so no chunk is read before writing."""
    from zarr.core.buffer.core import default_buffer_prototype

    store = zarr.storage.LoggingStore(zarr.storage.MemoryStore())
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline." + pipeline}):
        array = zarr.create_array(store=store, shape=(7,), chunks=(3,), dtype="int64")
        array[:] = np.arange(7)
        plan = PartitionIndexer.from_selection(
            selection, (7,), array._async_array._chunk_grid._dimensions
        )
        store.counter.clear()
        sync(
            array._async_array._set_selection(
                cast("Indexer", plan), np.array(value), prototype=default_buffer_prototype()
            )
        )
        assert store.counter["get"] == 0
        assert store.counter["get_sync"] == 0
        np.testing.assert_array_equal(sync(array._async_array.getitem(Ellipsis)), expected)
