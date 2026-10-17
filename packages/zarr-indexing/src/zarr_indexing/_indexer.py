"""A `GridPartition` speaking zarr's `Indexer` protocol; not a public API.

`PartitionIndexer` walks a partition as the rows zarr's codec pipeline reads:
`shape`, `drop_axes`, and one `(chunk_coords, chunk_selection, out_selection,
is_complete_chunk)` per touched chunk, read straight off the `StridedSet`,
`IndexedSet` and `JointSet` columns. Both sides of the pipeline's assignment,
`out[out_selection]` and `chunk[chunk_selection]`, must take one shape. Two row
layouts guarantee that without leaning on NumPy's placement rules for mixed
basic and advanced indices:

- **slices**: every table is a `StridedSet` with a nonzero stride and the
  request axes appear in storage order. Chunk selectors are ints and ascending
  slices, out selectors are slices; a reversed request keeps the reversal in
  the out slice. A row is complete only when its chunk selectors walk the
  chunk's data extent in order, because zarr's whole-chunk write shortcut takes
  `value[out_selection]` as the chunk without applying `chunk_selection`.
- **coordinates**: everything else. Every selector is an integer array shaped
  along one slot of the partition's synthetic layout, so all selectors are
  advanced and both sides take the broadcast shape. Never complete.

The guide page "Partition rows for a codec pipeline" states the contract.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np

from zarr_indexing.boundary import select_positional
from zarr_indexing.chunk_resolution import GridPartition, StridedSet, plan_chunks
from zarr_indexing.transform import IndexTransform

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from zarr_indexing.boundary import SelectionMode
    from zarr_indexing.grid import DimensionGridLike

type Selector = int | slice | np.ndarray[Any, np.dtype[np.intp]]
type _Array = np.ndarray[Any, np.dtype[np.intp]]


class IndexerRow(NamedTuple):
    """One chunk's share of a request, in the four fields zarr's codec pipeline reads."""

    chunk_coords: tuple[int, ...]
    chunk_selection: tuple[Selector, ...]
    out_selection: tuple[Selector, ...]
    is_complete_chunk: bool


@dataclass(frozen=True, slots=True)
class PartitionIndexer:
    """A partition walked as zarr `Indexer` rows.

    `shape` is the request shape and `drop_axes` is empty: integer axes are
    dropped by the selectors themselves. Iteration yields the rows in the
    partition's row-major order, the same rows on every walk.
    """

    partition: GridPartition

    @classmethod
    def from_transform(
        cls, transform: IndexTransform, dimension_grids: Sequence[DimensionGridLike]
    ) -> PartitionIndexer:
        """Factor `transform` over `dimension_grids`."""
        return cls(plan_chunks(transform, dimension_grids).partition())

    @classmethod
    def from_selection(
        cls,
        selection: Any,
        shape: tuple[int, ...],
        dimension_grids: Sequence[DimensionGridLike],
        *,
        mode: SelectionMode = "basic",
    ) -> PartitionIndexer:
        """Plan a NumPy-dialect selection of an array of `shape`, read as `LazyArray` reads it."""
        transform = select_positional(IndexTransform.from_shape(shape), selection, mode)
        return cls.from_transform(transform, dimension_grids)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.partition.transform.domain.shape

    @property
    def drop_axes(self) -> tuple[int, ...]:
        return ()

    def __iter__(self) -> Iterator[IndexerRow]:
        partition = self.partition
        if len(partition) == 0:
            return iter(())
        domain = partition.transform.domain
        sets = partition.sets
        read = [axis.input_dimension for axis in sets if axis.input_dimension is not None]
        component_slots, slot_of = partition._slots()  # pyright: ignore[reportPrivateUsage]
        unread = tuple(k for k in slot_of if k not in read)
        for k in unread:
            if domain.shape[k] > 1:
                raise ValueError(
                    f"request axis {k} is read by no storage dimension and has extent "
                    f"{domain.shape[k]}: a row cannot scatter one chunk cell to several positions"
                )
        # A slice cannot repeat a coordinate, and both sides can only use basic
        # selectors when the request axes come out in storage order.
        slices = (
            not partition.joint_sets
            and read == sorted(read)
            and all(
                isinstance(axis, StridedSet) and (axis.stride != 0 or axis.input_dimension is None)
                for axis in sets
            )
        )
        if slices:
            return _slice_rows(partition, unread)
        return _coordinate_rows(partition, unread, component_slots, slot_of)


# -- slices layout ----------------------------------------------------------

type _Selectors = tuple[int, Selector, slice | None, bool]
"""A table row: chunk index, chunk selector, out slice (None for a constant), whole-chunk cover."""


def _strided_selectors(axis: StridedSet, i: int) -> _Selectors:
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


def _slice_rows(partition: GridPartition, unread: tuple[int, ...]) -> Iterator[IndexerRow]:
    sets = cast("tuple[StridedSet, ...]", partition.sets)
    ndim = partition.transform.domain.ndim
    rank = partition.transform.output_rank
    tables = [[_strided_selectors(axis, i) for i in range(len(axis))] for axis in sets]
    for combo in itertools.product(*tables):
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
        yield IndexerRow(
            tuple(coords), tuple(chunk), tuple(o for o in out if o is not None), complete
        )


# -- coordinates layout -----------------------------------------------------


def _coordinate_rows(
    partition: GridPartition,
    unread: tuple[int, ...],
    component_slots: tuple[int | None, ...],
    slot_of: dict[int, int],
) -> Iterator[IndexerRow]:
    transform = partition.transform
    domain = transform.domain
    sets, joints = partition.sets, partition.joint_sets
    rank = len(slot_of) + sum(s is not None for s in component_slots)
    scalar_shape = (1,) * rank

    def along(values: _Array, s: int) -> _Array:
        return values.reshape((1,) * s + (-1,) + (1,) * (rank - s - 1))

    per_set: list[list[tuple[int, _Array, _Array | None]]] = []
    for axis in sets:
        table: list[tuple[int, _Array, _Array | None]] = []
        for i in range(len(axis)):
            if not isinstance(axis, StridedSet):
                run, s = axis.run(i), slot_of[axis.input_dimension]
                table.append(
                    (int(axis.chunk[i]), along(axis.local[run], s), along(axis.positions[run], s))
                )
            elif axis.input_dimension is None:
                constant = np.full(scalar_shape, int(axis.local_start[i]), dtype=np.intp)
                table.append((int(axis.chunk[i]), constant, None))
            else:
                s = slot_of[axis.input_dimension]
                steps = np.arange(int(axis.extent[i]), dtype=np.intp)
                local = int(axis.local_start[i]) + axis.stride * steps
                table.append(
                    (int(axis.chunk[i]), along(local, s), along(int(axis.origin[i]) + steps, s))
                )
        per_set.append(table)
    per_joint: list[list[tuple[tuple[int, ...], tuple[_Array, ...], tuple[_Array, ...]]]] = []
    for joint, s in zip(joints, component_slots, strict=True):
        local, block = joint.local, joint.block_coordinates
        columns, axes = range(len(joint.output_dimensions)), range(len(joint.broadcast_axes))
        table_j: list[tuple[tuple[int, ...], tuple[_Array, ...], tuple[_Array, ...]]] = []
        for i in range(len(joint)):
            run = joint.run(i)
            coords_j = tuple(int(c) for c in joint.chunk[i])
            if s is not None:
                chunk_cols = tuple(along(local[run, c], s) for c in columns)
                out_cols = tuple(along(block[run, a], s) for a in axes)
            else:
                # A component with no broadcast axis is a single point.
                chunk_cols = tuple(local[run, c].reshape(scalar_shape) for c in columns)
                out_cols = ()
            table_j.append((coords_j, chunk_cols, out_cols))
        per_joint.append(table_j)
    unread_out = {k: np.zeros(scalar_shape, dtype=np.intp) for k in unread}
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
            chunk_sels: tuple[Selector, ...] = (
                tuple(int(a) for a in chunk) if rank == 0 else tuple(chunk)
            )
            yield IndexerRow(tuple(coords), chunk_sels, out_sels, False)
