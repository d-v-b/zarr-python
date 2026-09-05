"""Internal selector-execution prototype; not a public API.

Selection semantics are the same literal-coordinate semantics as IndexTransform.
An immediate sorted-coordinate plan borrows its input array: callers must keep
it unchanged for every use of the plan and its iterators. Declarative transforms retain their owned,
immutable index arrays. Neither path changes Zarr's default indexers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np

from zarr_indexing.chunk_resolution import (
    _data_size,  # pyright: ignore[reportPrivateUsage]
    plan_chunks,
)
from zarr_indexing.errors import BoundsCheckError
from zarr_indexing.grid import DimensionGridLike, FixedDimension
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_indexing.transform import (
    IndexTransform,
    _normalize_basic_selection,  # pyright: ignore[reportPrivateUsage]
    _positional_slice,  # pyright: ignore[reportPrivateUsage]
    _resolve_slice_ts,  # pyright: ignore[reportPrivateUsage]
)

type Selector = int | slice | np.ndarray[Any, Any]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


class ExecutionChunk(NamedTuple):
    """One codec-compatible gather/scatter operation."""

    chunk_coords: tuple[int, ...]
    chunk_selection: tuple[Selector, ...]
    out_selection: tuple[Selector, ...]
    is_complete_chunk: bool


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Reusable execution iterator with a zero-origin output buffer shape."""

    shape: tuple[int, ...]
    _rows: Callable[[], Iterator[ExecutionChunk]]
    drop_axes: tuple[int, ...] = ()

    def __iter__(self) -> Iterator[ExecutionChunk]:
        return self._rows()


class _BasicAxis(NamedTuple):
    start: int
    step: int
    nitems: int
    grid: DimensionGridLike
    scalar: bool = False


def _axis_rows(axis: _BasicAxis) -> Iterator[tuple[int, int | slice, slice | None, bool]]:
    start, step, count, grid, scalar = axis
    position = 0
    while position < count:
        coordinate = start + position * step
        chunk = grid.index_to_chunk(coordinate)
        offset = grid.chunk_offset(chunk)
        local = coordinate - offset
        extent = _data_size(grid, chunk)
        n = min(
            count - position,
            (extent - 1 - local) // step + 1 if step > 0 else local // -step + 1,
        )
        yield (
            chunk,
            local if scalar else _positional_slice(local, n, step),
            None if scalar else slice(position, position + n),
            step == 1 and local == 0 and n == grid.chunk_size(chunk),
        )
        position += n


def _basic_rows(axes: tuple[_BasicAxis, ...]) -> Iterator[ExecutionChunk]:
    # Like Zarr's BasicIndexer, product retains per-axis pieces, not the full
    # Cartesian product. A fully implicit multidimensional walk is future work.
    for pieces in itertools.product(*(_axis_rows(axis) for axis in axes)):
        yield ExecutionChunk(
            tuple(piece[0] for piece in pieces),
            tuple(piece[1] for piece in pieces),
            tuple(piece[2] for piece in pieces if piece[2] is not None),
            all(piece[3] for piece in pieces),
        )


def _basic_plan(shape: tuple[int, ...], axes: tuple[_BasicAxis, ...]) -> ExecutionPlan:
    # Validate all axes before handing any work to a writer. A late failure
    # must not leave earlier chunks of an invalid selection modified.
    if all(shape):
        for axis in axes:
            if axis.nitems:
                axis.grid.index_to_chunk(axis.start)
                axis.grid.index_to_chunk(axis.start + (axis.nitems - 1) * axis.step)
    return ExecutionPlan(shape, partial(_basic_rows, axes))


def _sorted_plan(
    coordinates: np.ndarray[Any, Any], grids: tuple[DimensionGridLike, ...]
) -> ExecutionPlan | None:
    if (
        len(grids) != 1
        or not isinstance(grids[0], FixedDimension)
        or coordinates.dtype != np.dtype(np.intp)
        or coordinates.ndim != 1
        or coordinates.size == 0
    ):
        return None
    grid = grids[0]
    if coordinates[0] < 0 or coordinates[-1] >= grid.extent or grid.size == 0:
        return None
    first = int(coordinates[0]) // grid.size
    last = int(coordinates[-1]) // grid.size
    # Sparse or unordered selections use the shared component/table planner.
    if (last - first + 1) * coordinates.size.bit_length() >= coordinates.size:
        return None
    if not bool((coordinates[:-1] <= coordinates[1:]).all()):
        return None
    # Only internal boundaries: the last chunk's end may exceed intp.
    edges = np.arange(first + 1, last + 1, dtype=np.intp) * grid.size
    cuts = np.searchsorted(coordinates, edges)
    return ExecutionPlan(coordinates.shape, partial(_sorted_rows, coordinates, grid, first, cuts))


def _sorted_rows(
    coordinates: np.ndarray[Any, Any],
    grid: FixedDimension,
    first: int,
    cuts: np.ndarray[Any, Any],
) -> Iterator[ExecutionChunk]:
    start = 0
    for relative in range(cuts.size + 1):
        stop = int(cuts[relative]) if relative < cuts.size else coordinates.size
        if stop > start:
            chunk = first + relative
            yield ExecutionChunk(
                (chunk,),
                (coordinates[start:stop] - chunk * grid.size,),
                (slice(start, stop),),
                False,
            )
        start = stop


def execute_selection(
    selection: Any,
    shape: tuple[int, ...],
    dimension_grids: Sequence[DimensionGridLike],
    *,
    mode: str = "basic",
) -> ExecutionPlan:
    """Compile literal-coordinate selections directly to execution selectors.

    The optimized sorted-coordinate path borrows the supplied ndarray, which
    must remain unchanged for every use of the returned plan and its iterators. Other
    fancy selections fall back to an owned IndexTransform. This internal
    prototype is deliberately not a NumPy/Zarr selection-normalization API.
    """
    grids = tuple(dimension_grids)
    if len(grids) != len(shape):
        raise ValueError("dimension_grids must have one entry per storage dimension")
    if any(size < 0 for size in shape):
        raise ValueError("shape dimensions must be nonnegative")
    if mode == "basic":
        normalized = _normalize_basic_selection(selection, len(shape))
        if all(sel is not None for sel in normalized):
            axes: list[_BasicAxis] = []
            out_shape: list[int] = []
            for dim, (sel, size, grid) in enumerate(zip(normalized, shape, grids, strict=True)):
                if isinstance(sel, int):
                    if not 0 <= sel < size:
                        raise BoundsCheckError(f"index {sel} is out of bounds for dimension {dim}")
                    axes.append(_BasicAxis(sel, 1, 1, grid, True))
                else:
                    assert isinstance(sel, slice)
                    start, step, _origin, count = _resolve_slice_ts(sel, dim, 0, size)
                    axes.append(_BasicAxis(start, step, count, grid))
                    out_shape.append(count)
            return _basic_plan(tuple(out_shape), tuple(axes))
    elif mode in ("orthogonal", "vectorized"):
        items: tuple[Any, ...] = selection if isinstance(selection, tuple) else (selection,)
        if (
            len(shape) == len(items) == 1
            and isinstance(items[0], np.ndarray)
            and items[0].ndim == 1
            and items[0].size > 0
            and 0 <= items[0][0] <= items[0][-1] < shape[0]
        ):
            sorted_plan = _sorted_plan(items[0], grids)
            if sorted_plan is not None:
                return sorted_plan
    else:
        raise ValueError(f"unknown indexing mode: {mode}")
    base = IndexTransform.from_shape(shape)
    transform = (
        base[selection]
        if mode == "basic"
        else base.oindex[selection]
        if mode == "orthogonal"
        else base.vindex[selection]
    )
    return execute_transform(transform, grids)


def execute_transform(
    transform: IndexTransform,
    dimension_grids: Sequence[DimensionGridLike],
) -> ExecutionPlan:
    """Lower an existing immutable transform through the same execution paths."""
    grids = tuple(dimension_grids)
    if len(grids) != transform.output_rank:
        raise ValueError("dimension_grids must have one entry per storage dimension")
    domain = transform.domain
    referenced_axes: set[int] = set()
    for m in transform.output:
        if isinstance(m, DimensionMap):
            referenced_axes.add(m.input_dimension)
        elif isinstance(m, ArrayMap):
            referenced_axes.update(m.dependency_axes)
    if any(size > 1 and axis not in referenced_axes for axis, size in enumerate(domain.shape)):
        raise NotImplementedError(
            "execution of repeated unread input axes needs an explicit write-order policy"
        )
    axes: list[_BasicAxis] = []
    input_axes: list[int] = []
    for m, grid in zip(transform.output, grids, strict=True):
        if isinstance(m, ConstantMap):
            axes.append(_BasicAxis(m.offset, 1, 1, grid, True))
        elif isinstance(m, DimensionMap) and m.stride != 0:
            axis = m.input_dimension
            input_axes.append(axis)
            axes.append(
                _BasicAxis(
                    m.offset + m.stride * domain.inclusive_min[axis],
                    m.stride,
                    domain.shape[axis],
                    grid,
                )
            )
        else:
            break
    else:
        if input_axes == list(range(domain.ndim)):
            return _basic_plan(domain.shape, tuple(axes))
    if transform.input_rank == transform.output_rank == 1:
        (m,) = transform.output
        if isinstance(m, ArrayMap) and m.offset == 0 and m.stride == 1:
            sorted_plan = _sorted_plan(m.index_array, grids)
            if sorted_plan is not None and sorted_plan.shape == domain.shape:
                return sorted_plan
    return ExecutionPlan(domain.shape, partial(_general_rows, transform, grids))


def _coordinates(transform: IndexTransform, origins: tuple[int, ...]) -> tuple[Selector, ...]:
    """Broadcast coordinate selectors in synthetic-axis order without expansion."""
    domain = transform.domain
    selectors: list[Selector] = []
    for m, origin in zip(transform.output, origins, strict=True):
        if isinstance(m, ConstantMap):
            values = np.full((1,) * domain.ndim, m.offset - origin, dtype=np.intp)
        elif isinstance(m, DimensionMap):
            axis = m.input_dimension
            shape = tuple(domain.shape[k] if k == axis else 1 for k in range(domain.ndim))
            values = np.arange(
                domain.inclusive_min[axis], domain.exclusive_max[axis], dtype=np.intp
            ).reshape(shape)
            values = m.offset - origin + m.stride * values
        else:
            values = m.offset - origin + m.stride * m.index_array
        selectors.append(values)
    return tuple(selectors)


def _general_rows(
    transform: IndexTransform,
    grids: tuple[DimensionGridLike, ...],
) -> Iterator[ExecutionChunk]:
    for projection in plan_chunks(transform, grids):
        yield ExecutionChunk(
            projection.chunk_coords,
            _coordinates(projection.chunk_transform, (0,) * transform.output_rank),
            _coordinates(projection.cell_transform, transform.domain.inclusive_min),
            # Arrays may repeat coordinates; only the direct basic path proves
            # coverage of the entire codec buffer and permits skipping reads.
            False,
        )


def for_shard_indexer(
    plan: ExecutionPlan, dimension_grids: Sequence[DimensionGridLike]
) -> ExecutionPlan:
    """Adapt selectors to the current sharding codec's flat coordinate output.

    Positive basic slices pass through unchanged. Broadcast coordinate arrays
    and negative slices become flat, paired coordinate selectors per shard.
    This materialization is a limitation of the current codec consumer, not a
    requirement of the planner, and can allocate the selected points in a shard.
    """
    return ExecutionPlan(plan.shape, partial(_shard_rows, plan, tuple(dimension_grids)))


def _flat_selectors(
    selection: tuple[Selector, ...], shape: tuple[int, ...]
) -> tuple[Selector, ...]:
    if not selection:
        return ()
    if all(isinstance(sel, np.ndarray) for sel in selection):
        arrays = cast("tuple[np.ndarray[Any, Any], ...]", selection)
    else:
        rank = sum(not isinstance(sel, int) for sel in selection)
        basic_arrays: list[np.ndarray[Any, Any]] = []
        axis = 0
        for sel, extent in zip(selection, shape, strict=True):
            if isinstance(sel, int):
                values = np.asarray(sel, dtype=np.intp).reshape((1,) * rank)
            else:
                values = (
                    np.arange(*sel.indices(extent), dtype=np.intp)
                    if isinstance(sel, slice)
                    else sel
                )
                values = values.reshape((1,) * axis + (values.size,) + (1,) * (rank - axis - 1))
                axis += 1
            basic_arrays.append(values)
        arrays = tuple(basic_arrays)
    return tuple(array.reshape(-1) for array in np.broadcast_arrays(*arrays))


def _shard_rows(
    plan: ExecutionPlan, grids: tuple[DimensionGridLike, ...]
) -> Iterator[ExecutionChunk]:
    for row in plan:
        if any(
            isinstance(sel, np.ndarray)
            or (isinstance(sel, slice) and sel.step is not None and sel.step < 0)
            for sel in row.chunk_selection
        ):
            shape = tuple(
                grid.chunk_size(c) for grid, c in zip(grids, row.chunk_coords, strict=True)
            )
            yield ExecutionChunk(
                row.chunk_coords,
                _flat_selectors(row.chunk_selection, shape),
                _flat_selectors(row.out_selection, plan.shape),
                False,
            )
        else:
            yield row
