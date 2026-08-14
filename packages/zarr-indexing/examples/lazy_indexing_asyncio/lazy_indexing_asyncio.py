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
from typing import Any, cast

import numpy as np
import pytest
import zarr
import zarr.api.asynchronous
from zarr.core.indexing import BasicSelection as ZarrBasicSelection

from zarr_indexing import (
    BasicSelection,
    LazyArray,
    NoBasicSelectionError,
    Partition,
    ReadContext,
    numpy_reader,
)


class _NoSynchronousRead:
    """A test reader proving the async adapter performs every source read."""

    def read_into(
        self,
        _source: Any,
        _context: ReadContext,
        _out: np.ndarray[Any, Any],
        /,
    ) -> None:
        raise AssertionError("the AsyncArray integration performed a synchronous read")


def _as_zarr_selection(selection: BasicSelection) -> ZarrBasicSelection | None:
    """Narrow a NumPy basic selection to the dialect AsyncArray accepts.

    Zarr accepts integers and positive-step slices, but not NumPy's ``None``
    newaxis or negative-step slices. Returning ``None`` selects the cover and
    residual path below; no backend exception is used for feature detection.
    """
    if any(
        item is None or (isinstance(item, slice) and item.step is not None and item.step < 1)
        for item in selection
    ):
        return None
    return cast("ZarrBasicSelection", selection)


async def _read_part(part: Partition, source: zarr.AsyncArray[Any]) -> np.ndarray[Any, Any]:
    """Fetch one part through AsyncArray, normalizing its narrower dialect."""
    try:
        direct = _as_zarr_selection(part.source_selection)
    except NoBasicSelectionError:
        direct = None
    if direct is not None:
        return np.asanyarray(await source.getitem(direct))

    # A cover contains one ascending slice per source axis, which is always a
    # Zarr basic selection. The residual restores a newaxis or reversal and
    # performs any query gather against the fetched NumPy block.
    cover, residual = part.view.transform.decompose()
    block = np.asanyarray(await source.getitem(cover))
    out = np.empty(part.view.shape, dtype=part.view.dtype)
    numpy_reader.read_into(block, ReadContext(residual), out)
    return out


@pytest.fixture
def store() -> dict[str, Any]:
    """A fresh in-memory store, shared by the sync and async handles."""
    return {}


@pytest.fixture
def source(store: dict[str, Any]) -> zarr.Array[Any]:
    """A chunked Zarr array, created synchronously and shared with the async side."""
    array = zarr.create_array(store=store, shape=(40, 30), chunks=(10, 10), dtype="i4")
    array[:] = np.arange(40 * 30).reshape(40, 30)
    return array


async def read_through(view: LazyArray, source: zarr.AsyncArray[Any]) -> np.ndarray[Any, Any]:
    """Materialize `view` by fetching every partition concurrently.

    The wrapper plans; the async source fetches. Compatible box partitions use
    `part.source_selection` directly. Other partitions fetch an ascending
    cover and apply the residual NumPy selection in memory. Every block lands
    at `part.out_selection` — no thread pool or scheduler inside zarr-indexing.
    """
    parts = tuple(view.parts())
    blocks = await asyncio.gather(*(_read_part(part, source) for part in parts))
    out = np.empty(view.shape, dtype=view.dtype)
    for part, block in zip(parts, blocks, strict=True):
        out[part.out_selection] = block
    return out


def test_parts_with_asyncio_gather(store: dict[str, Any], source: zarr.Array[Any]) -> None:
    """Fetch a view's partitions concurrently through zarr.AsyncArray."""
    view = LazyArray(cast(Any, source)).lazy[5:35, 3:27]

    async def scenario() -> np.ndarray:
        async_source = await zarr.api.asynchronous.open_array(store=store)
        return await read_through(view, async_source)

    result = asyncio.run(scenario())
    assert result.shape == (30, 24)
    assert np.array_equal(result, source[5:35, 3:27])

    # Strided and integer selections lower the same way; a scalar axis lowers
    # to an integer and drops, exactly as it does in the output placement.
    decimated = LazyArray(cast(Any, source)).lazy[::4, 7]

    async def decimated_scenario() -> np.ndarray:
        async_source = await zarr.api.asynchronous.open_array(store=store)
        return await read_through(decimated, async_source)

    assert np.array_equal(asyncio.run(decimated_scenario()), source[::4, 7])


def test_asyncarray_dialect_is_normalized(store: dict[str, Any], source: zarr.Array[Any]) -> None:
    """New axes and reversals take the cover-and-residual path through Zarr."""
    data = np.asarray(source[:])
    planner = LazyArray(cast(Any, source)).with_reader(_NoSynchronousRead())
    views_and_expected = (
        (planner.lazy[None, 5:35, 3:27], data[None, 5:35, 3:27]),
        (planner.lazy[35:4:-3, ::-2], data[35:4:-3, ::-2]),
        (planner.lazy[5:10, :, None], data[5:10, :, None]),
    )

    async def scenario() -> tuple[np.ndarray[Any, Any], ...]:
        async_source = await zarr.api.asynchronous.open_array(store=store)
        return tuple(
            await asyncio.gather(
                *(read_through(view, async_source) for view, _expected in views_and_expected)
            )
        )

    for result, (_view, expected) in zip(asyncio.run(scenario()), views_and_expected, strict=True):
        np.testing.assert_array_equal(result, expected)


def test_decoded_chunk_cache(store: dict[str, Any], source: zarr.Array[Any]) -> None:
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

    async def fetch_missing(
        parts: tuple[Partition, ...], async_source: zarr.AsyncArray[Any]
    ) -> None:
        missing = {
            cell_key(part): cell_selection(part) for part in parts if cell_key(part) not in cache
        }
        cells = await asyncio.gather(
            *(async_source.getitem(selection) for selection in missing.values())
        )
        cache.update(
            (key, np.asanyarray(cell)) for key, cell in zip(missing.keys(), cells, strict=True)
        )

    async def read_cached(
        view: LazyArray, async_source: zarr.AsyncArray[Any]
    ) -> np.ndarray[Any, Any]:
        parts = tuple(view.parts())
        await fetch_missing(parts, async_source)
        out = np.empty(view.shape, dtype=view.dtype)
        for part in parts:
            out[part.out_selection] = cache[cell_key(part)][part.chunk_local_selection]
        return out

    async def scenario() -> tuple[np.ndarray, np.ndarray]:
        async_source = await zarr.api.asynchronous.open_array(store=store)
        first = await read_cached(LazyArray(cast(Any, source)).lazy[5:25, 3:27], async_source)
        # The second view overlaps the first, so most cells are already cached.
        second = await read_cached(LazyArray(cast(Any, source)).lazy[15:35, ::2], async_source)
        return first, second

    first, second = asyncio.run(scenario())
    assert np.array_equal(first, source[5:25, 3:27])
    assert np.array_equal(second, source[15:35, ::2])
    # Every cached cell was fetched at most once: the first view touches 9
    # chunks, and of the 9 the second touches only 3 are new.
    assert len(cache) == 12


def test_query_parts_fetch_cover_asynchronously(
    store: dict[str, Any], source: zarr.Array[Any]
) -> None:
    """A query part fetches its cover asynchronously and gathers in memory.

    A gather (`oindex`, `vindex`, a mask) has no single-slab spelling, so
    `source_selection` raises `NoBasicSelectionError` instead of guessing one.
    The adapter catches that dedicated error, fetches the part's ascending
    cover through AsyncArray, and applies its residual transform in memory.
    """
    view = (
        LazyArray(cast(Any, source)).with_reader(_NoSynchronousRead()).lazy.oindex[[30, 2, 2], 4:10]
    )

    async def scenario() -> np.ndarray:
        async_source = await zarr.api.asynchronous.open_array(store=store)
        return await read_through(view, async_source)

    expected = np.asarray(source[:])[[30, 2, 2]][:, 4:10]
    assert np.array_equal(asyncio.run(scenario()), expected)


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
