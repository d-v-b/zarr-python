"""Measure actual in-memory Zarr codec reads/writes, including plan construction.

Storage, source data, grids, and replacement data are prepared before timing.
Both paths use the same array and codec pipeline. These are local MemoryStore
measurements, not estimates of cloud or filesystem throughput.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import tracemalloc
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import zarr.api.asynchronous as za
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.indexing import BasicIndexer, CoordinateIndexer
from zarr.storage import MemoryStore

from zarr_indexing._execution import execute_selection

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from zarr.core.indexing import Indexer


async def measure(operation: Callable[[], Awaitable[Any]]) -> dict[str, float]:
    await operation()
    samples = []
    for _ in range(9):
        start = time.perf_counter()
        await operation()
        samples.append((time.perf_counter() - start) * 1000)
    tracemalloc.start()
    result = await operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return {"ms": statistics.median(samples), "peak_mib": peak / 2**20}


async def run_case(case: str, sharded: bool) -> dict[str, Any]:
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    selection: Any
    if case == "basic":
        shape, chunks = (128, 128), (16, 16)
        selection, mode = (slice(1, 127), slice(1, 127)), "basic"
    elif case == "sorted":
        shape, chunks = (100000,), (10000,)
        selection, mode = (np.arange(1, 99999),), "vectorized"
    else:
        shape, chunks = (64, 64, 64), (16, 16, 16)
        i = np.arange(64)
        selection, mode = (i[:, None], i[:, None], i[None, :]), "vectorized"
    kwargs: dict[str, Any] = {}
    if sharded:
        kwargs["shards"] = tuple(c * 2 for c in chunks)
    array = await za.create_array(
        store=MemoryStore(), shape=shape, chunks=chunks, dtype="int64", **kwargs
    )
    source = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    await array.setitem(Ellipsis, source)
    expected = source[selection]
    replacement = expected + 1
    grids = array._chunk_grid._dimensions
    prototype = default_buffer_prototype()
    cls = BasicIndexer if mode == "basic" else CoordinateIndexer

    async def old_read() -> Any:
        result = await array._get_selection(
            cls(selection, shape, array._chunk_grid), prototype=prototype
        )
        # Zarr's public coordinate API restores sel_shape outside the codec
        # entry point; include that view operation in the baseline.
        return np.asarray(result).reshape(expected.shape)

    async def new_read() -> Any:
        plan = execute_selection(selection, shape, grids, mode=mode, ownership="borrow")
        return await array._get_selection(
            cast("Indexer", plan.lower("shard" if sharded else "numpy")), prototype=prototype
        )

    async def old_write() -> None:
        await array._set_selection(
            cls(selection, shape, array._chunk_grid),
            replacement if mode == "basic" else replacement.reshape(-1),
            prototype=prototype,
        )

    async def new_write() -> None:
        plan = execute_selection(
            selection, shape, grids, mode=mode, ownership="borrow", access="write"
        )
        await array._set_selection(
            cast("Indexer", plan.lower("shard" if sharded else "numpy")),
            replacement,
            prototype=prototype,
        )

    np.testing.assert_array_equal(await old_read(), expected)
    np.testing.assert_array_equal(await new_read(), expected)
    results = {
        "zarr_read": await measure(old_read),
        "new_read": await measure(new_read),
        "zarr_write": await measure(old_write),
        "new_write": await measure(new_write),
    }
    expected_full = source.copy()
    expected_full[selection] = replacement
    np.testing.assert_array_equal(await array.getitem(Ellipsis), expected_full)
    return results


async def main() -> None:
    results = {}
    for case in ("basic", "sorted", "components"):
        for sharded in (False, True):
            results[case + ("_sharded" if sharded else "")] = await run_case(case, sharded)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
