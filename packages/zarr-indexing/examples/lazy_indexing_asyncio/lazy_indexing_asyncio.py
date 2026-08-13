# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "zarr @ git+https://github.com/zarr-developers/zarr-python.git@main",
#   "zarr-indexing>=0.1",
#   "numpy==2.4.3",
#   "pytest==9.0.2"
# ]
# ///
#

"""
Demonstrate driving zarr_indexing.LazyArray's partition plan with asyncio
"""

import asyncio
import sys
from typing import Any, Protocol

import numpy as np
import pytest
import zarr
import zarr.api.asynchronous

from zarr_indexing import LazyArray, NoBasicSelectionError, Partition


class AsyncSource(Protocol):
    """The surface the async loop needs: one awaitable basic-selection read.

    `zarr.AsyncArray` satisfies it; so does anything else that can serve a
    tuple of integers and ascending slices — an HTTP tile endpoint, an fsspec
    wrapper, a database. The planner never sees this object.
    """

    async def getitem(self, selection: Any) -> Any: ...


@pytest.fixture
def store() -> dict[str, Any]:
    """A fresh in-memory store, shared by the sync and async handles."""
    return {}


@pytest.fixture
def source(store: dict[str, Any]) -> zarr.Array:
    """A chunked Zarr array, created synchronously and shared with the async side."""
    array = zarr.create_array(store=store, shape=(40, 30), chunks=(10, 10), dtype="i4")
    array[:] = np.arange(40 * 30).reshape(40, 30)
    return array


async def read_through(view: LazyArray, source: AsyncSource) -> np.ndarray:
    """Materialize `view` by fetching every partition concurrently.

    The wrapper plans; the async source fetches. Each box partition lowers to
    the basic selection `part.source_selection`, is fetched through the
    caller's own I/O layer, and lands at `part.out_selection` — no reader, no
    thread pool, and no scheduler inside zarr-indexing.
    """
    parts = tuple(view.parts())
    blocks = await asyncio.gather(*(source.getitem(part.source_selection) for part in parts))
    out = np.empty(view.shape, dtype=view.dtype)
    for part, block in zip(parts, blocks, strict=True):
        out[part.out_selection] = block
    return out


def test_parts_with_asyncio_gather(store: dict[str, Any], source: zarr.Array) -> None:
    """Fetch a view's partitions concurrently through zarr.AsyncArray."""
    view = LazyArray(source).lazy[5:35, 3:27]

    async def scenario() -> np.ndarray:
        async_source = await zarr.api.asynchronous.open_array(store=store)
        return await read_through(view, async_source)

    result = asyncio.run(scenario())
    assert result.shape == (30, 24)
    assert np.array_equal(result, source[5:35, 3:27])

    # Strided and integer selections lower the same way; a scalar axis lowers
    # to an integer and drops, exactly as it does in the output placement.
    decimated = LazyArray(source).lazy[::4, 7]

    async def decimated_scenario() -> np.ndarray:
        async_source = await zarr.api.asynchronous.open_array(store=store)
        return await read_through(decimated, async_source)

    assert np.array_equal(asyncio.run(decimated_scenario()), source[::4, 7])


def test_decoded_chunk_cache(store: dict[str, Any], source: zarr.Array) -> None:
    """Cache decoded cells; place each view's share with `chunk_local_selection`.

    A tile server reads many overlapping views of the same array. Fetching
    whole chunks once and slicing every view out of the cached cells turns
    N overlapping requests into one fetch per chunk. `projection.chunk_domain`
    is the cell to fetch, and `chunk_local_selection` is the view's read
    relative to the cell's origin.

    The cache is keyed on that domain's origin rather than on `base_coords`,
    which counts cells of whatever base its view partitions: two views sharing
    a source but not a grid — or a part re-partitioned into smaller boxes —
    number their cells differently, while the origin names one region of the
    source however it was reached.
    """
    cache: dict[tuple[int, ...], np.ndarray] = {}

    def cell_key(part: Partition) -> tuple[int, ...]:
        return part.projection.chunk_domain.inclusive_min

    def cell_selection(part: Partition) -> tuple[slice, ...]:
        domain = part.projection.chunk_domain
        return tuple(
            slice(lo, hi) for lo, hi in zip(domain.inclusive_min, domain.exclusive_max, strict=True)
        )

    async def fetch_missing(parts: tuple[Partition, ...], async_source: AsyncSource) -> None:
        missing = {
            cell_key(part): cell_selection(part) for part in parts if cell_key(part) not in cache
        }
        cells = await asyncio.gather(
            *(async_source.getitem(selection) for selection in missing.values())
        )
        cache.update(zip(missing.keys(), cells, strict=True))

    async def read_cached(view: LazyArray, async_source: AsyncSource) -> np.ndarray:
        parts = tuple(view.parts())
        await fetch_missing(parts, async_source)
        out = np.empty(view.shape, dtype=view.dtype)
        for part in parts:
            out[part.out_selection] = cache[cell_key(part)][part.chunk_local_selection]
        return out

    async def scenario() -> tuple[np.ndarray, np.ndarray]:
        async_source = await zarr.api.asynchronous.open_array(store=store)
        first = await read_cached(LazyArray(source).lazy[5:25, 3:27], async_source)
        # The second view overlaps the first, so most cells are already cached.
        second = await read_cached(LazyArray(source).lazy[15:35, ::2], async_source)
        return first, second

    first, second = asyncio.run(scenario())
    assert np.array_equal(first, source[5:25, 3:27])
    assert np.array_equal(second, source[15:35, ::2])
    # Every cached cell was fetched at most once: the first view touches 9
    # chunks, and of the 9 the second touches only 3 are new.
    assert len(cache) == 12


def test_query_parts_fall_back(store: dict[str, Any], source: zarr.Array) -> None:
    """A query part refuses to lower; the wrapper's own reader is the fallback.

    A gather (`oindex`, `vindex`, a mask) has no single-slab spelling, so
    `source_selection` raises `ValueError` instead of guessing one. A consumer
    mixing selection kinds catches that and resolves the part through
    `part.view.result()`, which reads through the wrapped array's reader.
    Catching the dedicated subclass — not bare `ValueError` — keeps a genuine
    defect in the lowering loud instead of silently degrading every part to
    the fallback path.
    """
    view = LazyArray(source).lazy.oindex[[30, 2, 2], 4:10]

    async def scenario() -> np.ndarray:
        async_source = await zarr.api.asynchronous.open_array(store=store)
        parts = tuple(view.parts())
        out = np.empty(view.shape, dtype=view.dtype)

        async def resolve(part: Partition) -> tuple[Partition, np.ndarray]:
            try:
                selection = part.source_selection
            except NoBasicSelectionError:
                # The gathered axis needs a lookup, not a slab: read this part
                # through the wrapper (synchronously here; a real consumer
                # might push it to a thread, or fetch the part's bounding box
                # and gather in memory).
                return part, np.asarray(part.view.result())
            return part, np.asarray(await async_source.getitem(selection))

        for part, block in await asyncio.gather(*(resolve(part) for part in parts)):
            out[part.out_selection] = block
        return out

    assert np.array_equal(asyncio.run(scenario()), source.oindex[[30, 2, 2], 4:10])


if __name__ == "__main__":
    # Run the example with printed output, and a dummy pytest configuration file specified.
    # Without the dummy configuration file, at test time pytest will attempt to use the
    # configuration file in the project root, which will error because Zarr is using some
    # plugins that are not installed in this example.
    sys.exit(
        pytest.main(
            [
                "-s",
                __file__,
                f"-c {__file__}",
                # Suppress: "PytestAssertRewriteWarning: Module already imported so
                # cannot be rewritten; zarr"
                "-W",
                "ignore::pytest.PytestAssertRewriteWarning",
            ]
        )
    )
