"""Internal selector-execution prototype; not a public API.

Prepared work shares literal-coordinate semantics with IndexTransform. Inputs
are snapshotted by default; borrowing, access intent, and duplicate-write policy
are explicit. NumPy and shard consumers lower the same work to their required
selector layout. Neither path changes Zarr's default indexers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np

from zarr_indexing._affine import checked_affine
from zarr_indexing._axis_plan import axis_runs
from zarr_indexing.chunk_resolution import (
    ChunkPlan,
    IndexedSet,
    _shared_input_axis,  # pyright: ignore[reportPrivateUsage]
    plan_chunks,
)
from zarr_indexing.errors import BoundsCheckError
from zarr_indexing.grid import DimensionGridLike, RegularDimensionGridLike
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_indexing.transform import (
    IndexTransform,
    _normalize_basic_selection,  # pyright: ignore[reportPrivateUsage]
    _positional_slice,  # pyright: ignore[reportPrivateUsage]
    _resolve_slice_ts,  # pyright: ignore[reportPrivateUsage]
)

type Selector = int | slice | np.ndarray[Any, Any]

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class ExecutionChunk(NamedTuple):
    """Legacy four-field codec row produced by a named consumer's lowering."""

    chunk_coords: tuple[int, ...]
    chunk_selection: tuple[Selector, ...]
    out_selection: tuple[Selector, ...]
    is_complete_chunk: bool


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Prepared semantic work, independent of a consumer's selector layout.

    Snapshot ownership is the default. Borrowing is an explicit caller promise
    to leave arrays unchanged throughout every use of the plan and its iterators.
    Writers must prepare with access='write'; duplicate coordinates are rejected
    unless conflicts='last' explicitly requests request-order last-write-wins.
    """

    shape: tuple[int, ...]
    work: _BasicWork | _SortedWork | _ComponentWork | ChunkPlan
    access: Literal["read", "write"] = "read"
    ownership: Literal["snapshot", "borrow"] = "snapshot"
    conflicts: Literal["error", "last"] = "error"
    drop_axes: tuple[int, ...] = ()

    def __iter__(self) -> Iterator[ExecutionChunk]:
        return iter(self.lower())

    def lower(self, consumer: Literal["numpy", "shard"] = "numpy") -> LoweredPlan:
        """Choose a consumer; shard lowering may materialize paired coordinates."""
        if consumer not in ("numpy", "shard"):
            raise ValueError(f"unknown execution consumer: {consumer}")
        return LoweredPlan(self, consumer)


@dataclass(frozen=True, slots=True)
class LoweredOperation:
    """Selectors and their selected-value shape for one named consumer.

    Coordinate selectors are paired and broadcast to value_shape. Basic
    selectors use NumPy slice/integer semantics. row is the legacy four-field
    tuple consumed by Zarr's current codec pipeline.
    """

    row: ExecutionChunk
    value_shape: tuple[int, ...]
    selector_kind: Literal["basic", "paired"]


@dataclass(frozen=True, slots=True)
class LoweredPlan:
    plan: ExecutionPlan
    consumer: Literal["numpy", "shard"]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.plan.shape

    @property
    def drop_axes(self) -> tuple[int, ...]:
        return self.plan.drop_axes

    def operations(self) -> Iterator[LoweredOperation]:
        return _lower(self.plan, self.consumer)

    def __iter__(self) -> Iterator[ExecutionChunk]:
        return _consumer_rows(self.plan, self.consumer)


class _BasicAxis(NamedTuple):
    start: int
    step: int
    nitems: int
    grid: DimensionGridLike
    scalar: bool = False


@dataclass(frozen=True, slots=True)
class _BasicWork:
    axes: tuple[_BasicAxis, ...]


@dataclass(frozen=True, slots=True)
class _SortedWork:
    coordinates: np.ndarray[Any, Any]
    chunk_size: int
    first: int
    cuts: np.ndarray[Any, Any]


@dataclass(frozen=True, slots=True)
class _ComponentWork:
    plan: ChunkPlan
    local: tuple[np.ndarray[Any, Any], ...]


def _axis_rows(axis: _BasicAxis) -> Iterator[tuple[int, int | slice, slice | None, bool]]:
    start, step, count, grid, scalar = axis
    for run in axis_runs(start, step, count, grid):
        yield (
            run.chunk,
            run.local_start if scalar else _positional_slice(run.local_start, run.nitems, step),
            None if scalar else slice(run.position, run.position + run.nitems),
            step == 1 and run.local_start == 0 and run.nitems == grid.chunk_size(run.chunk),
        )


def _basic_rows(axes: tuple[_BasicAxis, ...]) -> Iterator[ExecutionChunk]:
    if any(axis.nitems == 0 for axis in axes):
        return
    # Cache only small axes. Large axes stay implicit, including before the
    # first result; a long dimension must not be drained by itertools.product.
    pools: list[tuple[tuple[int, int | slice, slice | None, bool], ...] | None] = []
    for axis in axes:
        span = (
            abs(
                axis.grid.index_to_chunk(axis.start + (axis.nitems - 1) * axis.step)
                - axis.grid.index_to_chunk(axis.start)
            )
            + 1
        )
        pools.append(tuple(_axis_rows(axis)) if min(span, axis.nitems) <= 128 else None)
    combinations = (
        itertools.product(*cast("list[tuple[Any, ...]]", pools))
        if all(pool is not None for pool in pools)
        else _implicit_product(axes, pools)
    )
    for pieces in combinations:
        yield ExecutionChunk(
            tuple(piece[0] for piece in pieces),
            tuple(piece[1] for piece in pieces),
            tuple(piece[2] for piece in pieces if piece[2] is not None),
            all(piece[3] for piece in pieces),
        )


def _implicit_product(
    axes: tuple[_BasicAxis, ...],
    pools: list[tuple[Any, ...] | None],
    dimension: int = 0,
    prefix: tuple[Any, ...] = (),
) -> Iterator[tuple[Any, ...]]:
    if dimension == len(axes):
        yield prefix
    else:
        pool = pools[dimension]
        for piece in pool if pool is not None else _axis_rows(axes[dimension]):
            yield from _implicit_product(axes, pools, dimension + 1, (*prefix, piece))


def _basic_plan(shape: tuple[int, ...], axes: tuple[_BasicAxis, ...]) -> ExecutionPlan:
    # Validate all axes before handing any work to a writer. A late failure
    # must not leave earlier chunks of an invalid selection modified.
    if all(shape):
        for axis in axes:
            if axis.nitems:
                axis.grid.index_to_chunk(axis.start)
                axis.grid.index_to_chunk(axis.start + (axis.nitems - 1) * axis.step)
    return ExecutionPlan(shape, _BasicWork(axes))


def _sorted_plan(
    coordinates: np.ndarray[Any, Any], grids: tuple[DimensionGridLike, ...]
) -> ExecutionPlan | None:
    if (
        len(grids) != 1
        or not isinstance(grids[0], RegularDimensionGridLike)
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
    cuts.setflags(write=False)
    return ExecutionPlan(coordinates.shape, _SortedWork(coordinates, grid.size, first, cuts))


def _sorted_rows(
    coordinates: np.ndarray[Any, Any],
    chunk_size: int,
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
                (coordinates[start:stop] - chunk * chunk_size,),
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
    ownership: Literal["snapshot", "borrow"] = "snapshot",
    access: Literal["read", "write"] = "read",
    conflicts: Literal["error", "last"] = "error",
) -> ExecutionPlan:
    """Compile literal-coordinate selections directly to execution selectors.

    Snapshot ownership is the default, independent of optimizer dispatch.
    ownership='borrow' permits borrowing; callers must leave inputs unchanged
    for every use of the plan. This is a literal-coordinate frontend, not a
    NumPy/Zarr selection-normalization API.
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
            return _with_policy(
                _basic_plan(tuple(out_shape), tuple(axes)), access, ownership, conflicts
            )
    elif mode in ("orthogonal", "vectorized"):
        items: tuple[Any, ...] = selection if isinstance(selection, tuple) else (selection,)
        if (
            len(shape) == len(items) == 1
            and isinstance(items[0], np.ndarray)
            and items[0].ndim == 1
            and items[0].size > 0
            and 0 <= items[0][0] <= items[0][-1] < shape[0]
        ):
            coordinates = items[0]
            if ownership == "snapshot":
                coordinates = coordinates.copy()
                coordinates.setflags(write=False)
            sorted_plan = _sorted_plan(coordinates, grids)
            if sorted_plan is not None:
                return _with_policy(sorted_plan, access, ownership, conflicts)
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
    return _with_policy(execute_transform(transform, grids), access, ownership, conflicts)


def execute_transform(
    transform: IndexTransform,
    dimension_grids: Sequence[DimensionGridLike],
    *,
    access: Literal["read", "write"] = "read",
    conflicts: Literal["error", "last"] = "error",
) -> ExecutionPlan:
    """Lower an existing immutable transform through the same execution paths."""
    grids = tuple(dimension_grids)
    if len(grids) != transform.output_rank:
        raise ValueError("dimension_grids must have one entry per storage dimension")
    domain = transform.domain
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
            return _with_policy(
                _basic_plan(domain.shape, tuple(axes)), access, "snapshot", conflicts
            )
    if transform.input_rank == transform.output_rank == 1:
        (m,) = transform.output
        if isinstance(m, ArrayMap) and m.offset == 0 and m.stride == 1:
            sorted_plan = _sorted_plan(m.index_array, grids)
            if sorted_plan is not None and sorted_plan.shape == domain.shape:
                return _with_policy(sorted_plan, access, "snapshot", conflicts)
    _validate_storage_bounds(transform, grids)
    plan = plan_chunks(transform, grids)
    # Prepare factored array grouping once. Affine diagonals intentionally use
    # ChunkPlan's shared projection path rather than per-output-axis tables.
    if (
        any(isinstance(m, ArrayMap) for m in transform.output)
        and _shared_input_axis(transform) is None
    ):
        partition = plan.partition()
        if not partition.sets and all(
            bool((joint.chunk_start >= 0).all()) for joint in partition.joint_sets
        ):
            # Column arithmetic is checked once by JointSet.local; nonnegative
            # chunk origins make its final local subtraction safe in intp.
            work = _ComponentWork(plan, tuple(joint.local for joint in partition.joint_sets))
            return _with_policy(ExecutionPlan(domain.shape, work), access, "snapshot", conflicts)
    return _with_policy(ExecutionPlan(domain.shape, plan), access, "snapshot", conflicts)


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
            positions = np.arange(domain.shape[axis], dtype=np.intp).reshape(shape)
            values = checked_affine(
                m.offset + m.stride * domain.inclusive_min[axis] - origin, m.stride, positions
            )
        else:
            values = checked_affine(m.offset - origin, m.stride, m.index_array)
        selectors.append(np.broadcast_to(values, domain.shape))
    return tuple(selectors)


def _general_rows(plan: ChunkPlan) -> Iterator[ExecutionChunk]:
    transform = plan.transform
    for projection in plan:
        yield ExecutionChunk(
            projection.chunk_coords,
            _coordinates(projection.chunk_transform, (0,) * transform.output_rank),
            _coordinates(projection.cell_transform, transform.domain.inclusive_min),
            # Arrays may repeat coordinates; only the direct basic path proves
            # coverage of the entire codec buffer and permits skipping reads.
            False,
        )


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


def _shard_rows(plan: ExecutionPlan, rows: Iterator[ExecutionChunk]) -> Iterator[ExecutionChunk]:
    for row in rows:
        if not row.out_selection:
            # The shard coordinate indexer returns a length-one vector for a
            # 0-D array selector. Integer selectors preserve a scalar result.
            yield ExecutionChunk(
                row.chunk_coords,
                tuple(
                    int(sel.item()) if isinstance(sel, np.ndarray) and sel.ndim == 0 else sel
                    for sel in row.chunk_selection
                ),
                (),
                row.is_complete_chunk,
            )
            continue
        if all(isinstance(sel, np.ndarray) and sel.ndim <= 1 for sel in row.chunk_selection) and (
            all(isinstance(sel, np.ndarray) and sel.ndim <= 1 for sel in row.out_selection)
            or (len(row.out_selection) == 1 and isinstance(row.out_selection[0], slice))
        ):
            # The existing shard indexer already produces this flat value
            # shape. In particular, retain sorted runs' output slices.
            yield row
            continue
        if any(
            isinstance(sel, np.ndarray)
            or (isinstance(sel, slice) and sel.step is not None and sel.step < 0)
            for sel in row.chunk_selection
        ):
            shape = _chunk_shape(plan, row.chunk_coords)
            yield ExecutionChunk(
                row.chunk_coords,
                _flat_selectors(row.chunk_selection, shape),
                _flat_selectors(row.out_selection, plan.shape),
                False,
            )
        else:
            yield row


def _validate_storage_bounds(
    transform: IndexTransform, grids: tuple[DimensionGridLike, ...]
) -> None:
    if 0 in transform.domain.shape:
        return
    for m, grid in zip(transform.output, grids, strict=True):
        if isinstance(m, ConstantMap):
            bounds = (m.offset, m.offset)
        elif isinstance(m, DimensionMap):
            axis = m.input_dimension
            bounds = (
                checked_affine(m.offset, m.stride, transform.domain.inclusive_min[axis]),
                checked_affine(m.offset, m.stride, transform.domain.exclusive_max[axis] - 1),
            )
        else:
            mapped = checked_affine(m.offset, m.stride, m.index_array)
            bounds = (int(mapped.min()), int(mapped.max()))
        grid.index_to_chunk(min(bounds))
        grid.index_to_chunk(max(bounds))


def _with_policy(
    plan: ExecutionPlan,
    access: Literal["read", "write"],
    ownership: Literal["snapshot", "borrow"],
    conflicts: Literal["error", "last"],
) -> ExecutionPlan:
    if access not in ("read", "write"):
        raise ValueError(f"unknown access intent: {access}")
    if ownership not in ("snapshot", "borrow"):
        raise ValueError(f"unknown ownership policy: {ownership}")
    if conflicts not in ("error", "last"):
        raise ValueError(f"unknown conflict policy: {conflicts}")
    result = (
        plan
        if (plan.access, plan.ownership, plan.conflicts) == (access, ownership, conflicts)
        else replace(plan, access=access, ownership=ownership, conflicts=conflicts)
    )
    if access == "write" and conflicts == "error":
        _validate_unique_writes(result)
    return result


def _validate_unique_writes(plan: ExecutionPlan) -> None:
    if 0 in plan.shape or isinstance(plan.work, _BasicWork):
        return
    work = plan.work
    if isinstance(work, _SortedWork):
        unique = not bool((work.coordinates[1:] == work.coordinates[:-1]).any())
    else:
        source_plan = work.plan if isinstance(work, _ComponentWork) else work
        transform = source_plan.transform
        referenced: set[int] = set()
        for m in transform.output:
            if isinstance(m, DimensionMap) and m.stride != 0:
                referenced.add(m.input_dimension)
            elif isinstance(m, ArrayMap) and m.stride != 0:
                referenced.update(m.dependency_axes)
        unique = all(size <= 1 or axis in referenced for axis, size in enumerate(plan.shape))
        if unique and transform.index_array_structure != "general":
            for axis, size in enumerate(plan.shape):
                if size <= 1 or any(
                    isinstance(m, DimensionMap) and m.input_dimension == axis and m.stride != 0
                    for m in transform.output
                ):
                    continue
                columns = [
                    m.index_array.reshape(-1)
                    for m in transform.output
                    if isinstance(m, ArrayMap) and m.dependent_axis == axis and m.stride != 0
                ]
                unique &= (
                    bool(columns) and np.unique(np.stack(columns, axis=1), axis=0).shape[0] == size
                )
        elif unique and any(isinstance(m, ArrayMap) for m in transform.output):
            partition = source_plan.partition()
            for axis in partition.sets:
                if isinstance(axis, IndexedSet):
                    unique &= axis.stride != 0 and np.unique(axis.index).size == axis.index.size
            for joint in partition.joint_sets:
                columns = [i for i, stride in enumerate(joint.strides) if stride != 0]
                values = joint.index[:, columns]
                unique &= np.unique(values, axis=0).shape[0] == values.shape[0]
    if not unique:
        raise ValueError("duplicate writes require conflicts='last'")


def _raw_rows(plan: ExecutionPlan) -> Iterator[ExecutionChunk]:
    work = plan.work
    if isinstance(work, _BasicWork):
        return _basic_rows(work.axes)
    if isinstance(work, _SortedWork):
        return _sorted_rows(work.coordinates, work.chunk_size, work.first, work.cuts)
    if isinstance(work, _ComponentWork):
        return _component_rows(work)
    return _general_rows(work)


def _chunk_shape(plan: ExecutionPlan, coords: tuple[int, ...]) -> tuple[int, ...]:
    work = plan.work
    if isinstance(work, _SortedWork):
        return (work.chunk_size,)
    grids = (
        tuple(axis.grid for axis in work.axes)
        if isinstance(work, _BasicWork)
        else work.plan.dimension_grids
        if isinstance(work, _ComponentWork)
        else work.dimension_grids
    )
    return tuple(grid.chunk_size(c) for grid, c in zip(grids, coords, strict=True))


def _ordered_write_rows(
    plan: ExecutionPlan, rows: Iterator[ExecutionChunk]
) -> Iterator[ExecutionChunk]:
    for row in rows:
        chunk = _flat_selectors(row.chunk_selection, _chunk_shape(plan, row.chunk_coords))
        out = _flat_selectors(row.out_selection, plan.shape)
        if not out:
            yield row
            continue
        positions = cast("tuple[np.ndarray[Any, Any], ...]", out)
        order = np.lexsort(positions[::-1])
        if not chunk:
            # A scalar target repeated over a request has one final value.
            yield ExecutionChunk(
                row.chunk_coords, (), tuple(int(p[order[-1]]) for p in positions), False
            )
        else:
            # Eliminate duplicate destinations ourselves: a backend's repeated
            # advanced-assignment order must not define our conflict policy.
            destinations = np.stack([cast("np.ndarray[Any, Any]", c)[order] for c in chunk], axis=1)
            _, reversed_positions = np.unique(destinations[::-1], axis=0, return_index=True)
            order = order[np.sort(order.size - 1 - reversed_positions)]
            yield ExecutionChunk(
                row.chunk_coords,
                tuple(cast("np.ndarray[Any, Any]", c)[order] for c in chunk),
                tuple(p[order] for p in positions),
                False,
            )


def _consumer_rows(
    plan: ExecutionPlan, consumer: Literal["numpy", "shard"]
) -> Iterator[ExecutionChunk]:
    rows = _raw_rows(plan)
    if (
        plan.access == "write"
        and plan.conflicts == "last"
        and not isinstance(plan.work, _BasicWork)
    ):
        rows = _ordered_write_rows(plan, rows)
    return _shard_rows(plan, rows) if consumer == "shard" else rows


def _lower(plan: ExecutionPlan, consumer: Literal["numpy", "shard"]) -> Iterator[LoweredOperation]:
    for row in _consumer_rows(plan, consumer):
        if all(isinstance(sel, np.ndarray) for sel in row.out_selection) and row.out_selection:
            shape = np.broadcast_shapes(
                *(cast("np.ndarray[Any, Any]", sel).shape for sel in row.out_selection)
            )
        else:
            shape = tuple(
                len(range(*sel.indices(size)))
                for sel, size in zip(row.out_selection, plan.shape, strict=True)
                if isinstance(sel, slice)
            )
        kind: Literal["basic", "paired"] = (
            "paired" if any(isinstance(sel, np.ndarray) for sel in row.chunk_selection) else "basic"
        )
        yield LoweredOperation(row, shape, kind)


def _component_rows(work: _ComponentWork) -> Iterator[ExecutionChunk]:
    """Lower factored coordinate columns without constructing transform pairs."""
    partition = work.plan.partition()
    joints = partition.joint_sets
    domain = work.plan.transform.domain
    if 0 in domain.shape:
        return
    referenced = {axis for joint in joints for axis in joint.broadcast_axes}
    unread = tuple(axis for axis in range(domain.ndim) if axis not in referenced)
    slots: list[int | None] = []
    lead = 0
    for joint in joints:
        slots.append(lead if joint.broadcast_axes else None)
        lead += bool(joint.broadcast_axes)
    rank = lead + len(unread)
    for rows in itertools.product(*(range(len(joint)) for joint in joints)):
        chunk: list[Selector] = [0] * work.plan.transform.output_rank
        out: list[Selector] = [0] * domain.ndim
        coords = [0] * len(chunk)
        shape = [1] * rank
        for joint, local, slot, row in zip(joints, work.local, slots, rows, strict=True):
            run = joint.run(row)
            component_shape = [1] * rank
            if slot is not None:
                shape[slot] = component_shape[slot] = run.stop - run.start
            for column, dimension in enumerate(joint.output_dimensions):
                coords[dimension] = int(joint.chunk[row, column])
                chunk[dimension] = local[run, column].reshape(component_shape)
            for column, axis in enumerate(joint.broadcast_axes):
                out[axis] = joint.block_coordinates[run, column].reshape(component_shape)
        for i, axis in enumerate(unread):
            component_shape = [1] * rank
            shape[lead + i] = component_shape[lead + i] = domain.shape[axis]
            out[axis] = np.arange(domain.shape[axis], dtype=np.intp).reshape(component_shape)
        if any(domain.shape[axis] > 1 for axis in unread):
            chunk = [
                np.broadcast_to(cast("np.ndarray[Any, Any]", sel), tuple(shape)) for sel in chunk
            ]
        yield ExecutionChunk(tuple(coords), tuple(chunk), tuple(out), False)
