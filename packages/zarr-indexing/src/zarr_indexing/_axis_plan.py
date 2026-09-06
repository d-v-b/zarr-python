"""Shared implicit affine-axis planning for tables and immediate selectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from zarr_indexing._affine import checked_affine
from zarr_indexing.chunk_resolution import _data_size  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_indexing.grid import DimensionGridLike


class AxisRun(NamedTuple):
    chunk: int
    chunk_start: int
    data_extent: int
    local_start: int
    nitems: int
    position: int


def axis_runs(start: int, stride: int, nitems: int, grid: DimensionGridLike) -> Iterator[AxisRun]:
    """Intersect an affine request with chunks, in request traversal order.

    Source endpoints are checked before the first run. Request positions and
    counts retain Python integer precision; only storage coordinates must fit
    intp. Stride zero keeps repetitions symbolic in a single run.
    """
    if nitems == 0:
        return
    grid.index_to_chunk(checked_affine(start, stride, 0))
    grid.index_to_chunk(checked_affine(start, stride, nitems - 1))
    position = 0
    while position < nitems:
        coordinate = start + position * stride
        chunk = grid.index_to_chunk(coordinate)
        offset = grid.chunk_offset(chunk)
        local = coordinate - offset
        extent = _data_size(grid, chunk)
        if stride == 0:
            count = nitems
        else:
            count = min(
                nitems - position,
                (extent - 1 - local) // stride + 1 if stride > 0 else local // -stride + 1,
            )
        yield AxisRun(chunk, offset, extent, local, count, position)
        position += count
