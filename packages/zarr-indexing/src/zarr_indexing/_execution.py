"""Partition rows in the shape zarr's codec pipeline consumes; not a public API.

`execute_transform` factors a transform with `plan_chunks(...).partition()` and
wraps the `GridPartition` as an object with zarr's `Indexer` surface: `shape`,
`drop_axes`, and iteration yielding one
`(chunk_coords, chunk_selection, out_selection, is_complete_chunk)` row per
touched chunk. Rows are read straight off the partition's `StridedSet`,
`IndexedSet` and `JointSet` columns; no `ChunkProjection` is built.
`execute_selection` is the NumPy-dialect front door: `LazyArray`'s selection
handling (scalars first, positional coordinates) followed by `execute_transform`.

Both sides of the pipeline's assignment, `out[out_selection]` and
`chunk[chunk_selection]`, must produce values of one shape. Two layouts
guarantee that without relying on NumPy's advanced-index placement rules:

- **basic**: every table is a `StridedSet` and the request axes appear in
  storage order. Chunk selectors are ints and ascending slices (a reversed
  request puts the reversal in the out slice), out selectors are slices, and a
  singleton request axis no output reads selects position ``0``. A row is
  complete when every table row covers its chunk's data extent exactly once,
  so a whole-chunk write can skip its read.
- **coordinates**: everything else. Every selector is an integer array shaped
  along one *slot* of a common layout: one slot per request axis, with the
  broadcast axes of a connected index-array component collapsed into the slot
  of its first. All selectors are advanced, so both sides take the broadcast
  shape in slot order. Rows are never complete.

A mixed layout, one index array with the rest slices, is used when it is safe
(exactly one `IndexedSet`, no constants, no unread request axes, storage order).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from zarr_indexing.boundary import normalize_positional_selection, split_scalar_axes
from zarr_indexing.chunk_resolution import GridPartition, IndexedSet, StridedSet, plan_chunks
from zarr_indexing.transform import IndexTransform

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from zarr_indexing.boundary import SelectionMode
    from zarr_indexing.grid import DimensionGridLike

type Selector = int | slice | np.ndarray[Any, np.dtype[np.intp]]
type _Array = np.ndarray[Any, np.dtype[np.intp]]


class ExecutionChunk(NamedTuple):
    """One chunk's share of a request, in the four fields zarr's codec pipeline reads."""

    chunk_coords: tuple[int, ...]
    chunk_selection: tuple[Selector, ...]
    out_selection: tuple[Selector, ...]
    is_complete_chunk: bool


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """A partition walked as zarr `Indexer` rows.

    `shape` is the request shape and `drop_axes` is empty: integer axes are
    dropped by the selectors themselves. Iteration yields the rows in the
    partition's row-major order, the same rows on every walk.
    """

    partition: GridPartition
    drop_axes: tuple[int, ...] = ()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.partition.transform.domain.shape

    def __len__(self) -> int:
        return len(self.partition)

    def __iter__(self) -> Iterator[ExecutionChunk]:
        return rows(self.partition)


def execute_transform(
    transform: IndexTransform, dimension_grids: Sequence[DimensionGridLike]
) -> ExecutionPlan:
    """Factor `transform` over `dimension_grids` and wrap the partition as an indexer."""
    return ExecutionPlan(plan_chunks(transform, dimension_grids).partition())


def execute_selection(
    selection: Any,
    shape: tuple[int, ...],
    dimension_grids: Sequence[DimensionGridLike],
    *,
    mode: SelectionMode = "basic",
) -> ExecutionPlan:
    """Plan a NumPy-dialect selection of an array of `shape`.

    The selection is read as `LazyArray` reads it: positional coordinates with
    negatives counted from the end, and, in the orthogonal and vectorized
    modes, scalar integers applied first so their axes drop.
    """
    if mode not in ("basic", "orthogonal", "vectorized"):
        raise ValueError(f"unknown indexing mode: {mode!r}")
    transform = IndexTransform.from_shape(shape)
    if mode != "basic":
        scalars, selection = split_scalar_axes(selection, transform.domain, mode)
        if scalars is not None:
            transform = transform.select(scalars, "basic")
    literal = normalize_positional_selection(selection, transform.domain, mode)
    transform = transform[literal] if mode == "basic" else transform.select(literal, mode)
    return execute_transform(transform, dimension_grids)


def rows(partition: GridPartition) -> Iterator[ExecutionChunk]:
    """Lower every row of `partition` to a codec row, in row-major order."""
    if len(partition) == 0:
        return iter(())
    domain = partition.transform.domain
    sets, joints = partition.sets, partition.joint_sets
    read = [axis.input_dimension for axis in sets if axis.input_dimension is not None]
    bound = set(read).union(axis for joint in joints for axis in joint.broadcast_axes)
    unread = tuple(k for k in range(domain.ndim) if k not in bound)
    arrays = sum(isinstance(axis, IndexedSet) for axis in sets)
    basic = (
        not joints
        and read == sorted(read)
        and all(domain.shape[k] == 1 for k in unread)
        and (arrays == 0 or (arrays == 1 and len(read) == len(sets) and not unread))
    )
    return _basic_rows(partition, unread) if basic else _coordinate_rows(partition, unread)


# -- basic layout -----------------------------------------------------------

type _Piece = tuple[int, Selector, Selector | None, bool]
"""A table row: chunk index, chunk selector, out selector (None for a constant), whole-chunk cover."""


def _strided_piece(axis: StridedSet, i: int) -> _Piece:
    chunk, local, full = int(axis.chunk[i]), int(axis.local_start[i]), bool(axis.full[i])
    if axis.input_dimension is None:
        return chunk, local, None, full
    n, origin, stride = int(axis.extent[i]), int(axis.origin[i]), axis.stride
    if stride > 0:
        return (
            chunk,
            slice(local, local + stride * (n - 1) + 1, stride),
            slice(origin, origin + n),
            full,
        )
    # Storage is read ascending; the reversal lives in the out slice.
    last = local + stride * (n - 1)
    out = slice(origin + n - 1, origin - 1 if origin else None, -1)
    return chunk, slice(last, local + 1, -stride), out, full


def _indexed_piece(axis: IndexedSet, i: int) -> _Piece:
    run = axis.run(i)
    return int(axis.chunk[i]), axis.local[run], axis.positions[run], False


def _basic_rows(partition: GridPartition, unread: tuple[int, ...]) -> Iterator[ExecutionChunk]:
    sets = partition.sets
    ndim = partition.transform.domain.ndim
    rank = partition.transform.output_rank
    pieces = [
        [
            _strided_piece(axis, i) if isinstance(axis, StridedSet) else _indexed_piece(axis, i)
            for i in range(len(axis))
        ]
        for axis in sets
    ]
    for combo in itertools.product(*pieces):
        coords = [0] * rank
        chunk: list[Selector] = [0] * rank
        out: list[Selector | None] = [None] * ndim
        for k in unread:
            out[k] = 0
        complete = True
        for axis, (c, chunk_sel, out_sel, full) in zip(sets, combo, strict=True):
            coords[axis.output_dimension] = c
            chunk[axis.output_dimension] = chunk_sel
            if axis.input_dimension is not None and out_sel is not None:
                out[axis.input_dimension] = out_sel
            complete = complete and full
        yield ExecutionChunk(
            tuple(coords), tuple(chunk), tuple(o for o in out if o is not None), complete
        )


# -- coordinate layout ------------------------------------------------------


def _coordinate_rows(partition: GridPartition, unread: tuple[int, ...]) -> Iterator[ExecutionChunk]:
    transform = partition.transform
    domain = transform.domain
    sets, joints = partition.sets, partition.joint_sets
    # One slot per request axis; a joint's broadcast axes share the slot of its first.
    lead = {joint.broadcast_axes[0]: n for n, joint in enumerate(joints) if joint.broadcast_axes}
    skip = {axis for joint in joints for axis in joint.broadcast_axes[1:]}
    slot: dict[int, int] = {}
    joint_slot: dict[int, int] = {}
    for k in range(domain.ndim):
        if k in skip:
            continue
        if k in lead:
            joint_slot[lead[k]] = len(slot) + len(joint_slot)
        else:
            slot[k] = len(slot) + len(joint_slot)
    rank = len(slot) + len(joint_slot)
    scalar_shape = (1,) * rank

    def along(values: _Array, s: int) -> _Array:
        return values.reshape((1,) * s + (-1,) + (1,) * (rank - s - 1))

    per_set: list[list[tuple[int, _Array, _Array | None]]] = []
    for axis in sets:
        table: list[tuple[int, _Array, _Array | None]] = []
        for i in range(len(axis)):
            if isinstance(axis, IndexedSet):
                run, s = axis.run(i), slot[axis.input_dimension]
                table.append(
                    (int(axis.chunk[i]), along(axis.local[run], s), along(axis.positions[run], s))
                )
            elif axis.input_dimension is None:
                table.append(
                    (
                        int(axis.chunk[i]),
                        np.full(scalar_shape, int(axis.local_start[i]), dtype=np.intp),
                        None,
                    )
                )
            else:
                s = slot[axis.input_dimension]
                steps = np.arange(int(axis.extent[i]), dtype=np.intp)
                local = int(axis.local_start[i]) + axis.stride * steps
                table.append(
                    (int(axis.chunk[i]), along(local, s), along(int(axis.origin[i]) + steps, s))
                )
        per_set.append(table)
    per_joint: list[list[tuple[tuple[int, ...], tuple[_Array, ...], tuple[_Array, ...]]]] = []
    for n, joint in enumerate(joints):
        local, block = joint.local, joint.block_coordinates
        columns, axes = range(len(joint.output_dimensions)), range(len(joint.broadcast_axes))
        table_j: list[tuple[tuple[int, ...], tuple[_Array, ...], tuple[_Array, ...]]] = []
        for i in range(len(joint)):
            run = joint.run(i)
            coords_j = tuple(int(c) for c in joint.chunk[i])
            if n in joint_slot:
                s = joint_slot[n]
                chunk_cols = tuple(along(local[run, c], s) for c in columns)
                out_cols = tuple(along(block[run, a], s) for a in axes)
            else:
                # A component with no broadcast axis is a single point.
                chunk_cols = tuple(local[run, c].reshape(scalar_shape) for c in columns)
                out_cols = ()
            table_j.append((coords_j, chunk_cols, out_cols))
        per_joint.append(table_j)
    unread_out = {k: along(np.arange(domain.shape[k], dtype=np.intp), slot[k]) for k in unread}
    # A chunk value is read through an unread axis by broadcasting, but cannot
    # be written through one: expand the chunk selectors so the shapes match.
    expand = any(domain.shape[k] > 1 for k in unread)
    out_rank = transform.output_rank
    for set_combo in itertools.product(*per_set):
        for joint_combo in itertools.product(*per_joint):
            coords = [0] * out_rank
            chunk: list[_Array] = [np.empty(0, dtype=np.intp)] * out_rank
            out: list[_Array | None] = [None] * domain.ndim
            for k, values in unread_out.items():
                out[k] = values
            for axis, (c, chunk_sel, out_sel) in zip(sets, set_combo, strict=True):
                coords[axis.output_dimension] = c
                chunk[axis.output_dimension] = chunk_sel
                if axis.input_dimension is not None and out_sel is not None:
                    out[axis.input_dimension] = out_sel
            for joint, (coords_j, chunk_cols, out_cols) in zip(joints, joint_combo, strict=True):
                for d, c, sel in zip(joint.output_dimensions, coords_j, chunk_cols, strict=True):
                    coords[d] = c
                    chunk[d] = sel
                for a, sel in zip(joint.broadcast_axes, out_cols, strict=True):
                    out[a] = sel
            out_sels = tuple(o for o in out if o is not None)
            chunk_sels: tuple[Selector, ...]
            if expand:
                shape = np.broadcast_shapes(*(a.shape for a in (*chunk, *out_sels)))
                chunk_sels = tuple(np.broadcast_to(a, shape) for a in chunk)
            elif rank == 0:
                chunk_sels = tuple(int(a) for a in chunk)
            else:
                chunk_sels = tuple(chunk)
            yield ExecutionChunk(tuple(coords), chunk_sels, out_sels, False)
