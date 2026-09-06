"""Plan conflict-free batches of writes without executing or buffering data.

Destination grids describe independent read-modify-write units, which may be
shards rather than codec chunks. Finish all destination I/O in one batch before
starting destination I/O in the next. Source prefetch is independent only when
source and destination do not alias. Unrelated writers are not coordinated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from zarr_indexing._affine import checked_affine
from zarr_indexing.chunk_resolution import (
    _shared_input_axis,  # pyright: ignore[reportPrivateUsage]
    plan_chunks,
)
from zarr_indexing.domain import IndexDomain
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_indexing.transform import IndexTransform

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from zarr_indexing.grid import DimensionGridLike


@dataclass(frozen=True, slots=True)
class WriteSchedule:
    """Immutable task IDs grouped into barrier-separated parallel batches.

    Attributes
    ----------
    batches
        Task positions in input order within each batch. Every task occurs once,
        including empty writes. Empty input produces no batches.
    order
        ``preserve`` orders tasks sharing any write unit by input position;
        unrelated tasks can move earlier. ``reorder`` uses first-fit coloring.
    n_write_units
        Number of distinct destination units touched by the whole schedule.
    n_memberships
        Sum of distinct destination units per task; repeated indices within a
        task count once. This measures planning work, not bytes or memory limits.
    """

    batches: tuple[tuple[int, ...], ...]
    order: Literal["preserve", "reorder"]
    n_write_units: int
    n_memberships: int


@dataclass(frozen=True, slots=True)
class RechunkPiece:
    """One unchanged source chunk's contribution to a rechunk request.

    Attributes
    ----------
    source_chunk
        Coordinate identifying the source chunk to read.
    source_selection
        Basic slices relative to that decoded source chunk, clipped to the
        requested domain and valid source data.
    destination
        Identity mapping over this piece's global destination coordinates.
        Its domain bounds give the destination slice, with shape matching the
        values selected by ``source_selection``.
    """

    source_chunk: tuple[int, ...]
    source_selection: tuple[slice, ...]
    destination: IndexTransform


@dataclass(frozen=True, slots=True)
class RechunkPlan:
    """Source pieces and the schedule indexing them; no array data is retained.

    Attributes
    ----------
    pieces
        One piece per source chunk intersecting the requested domain.
    schedule
        A reorder schedule, safe because the pieces have disjoint logical
        destination elements. Storage write units may still overlap.
    """

    pieces: tuple[RechunkPiece, ...]
    schedule: WriteSchedule


def _validate_bounds(transform: IndexTransform, grids: tuple[DimensionGridLike, ...]) -> None:
    if len(grids) != transform.output_rank:
        raise ValueError("destination grid rank must match transform output rank")
    if 0 in transform.domain.shape:
        return
    for mapping, grid in zip(transform.output, grids, strict=True):
        if isinstance(mapping, ConstantMap):
            lo = hi = checked_affine(mapping.offset, 0, 0)
        else:
            if isinstance(mapping, DimensionMap):
                axis = mapping.input_dimension
                first = transform.domain.origin[axis]
                last = transform.domain.exclusive_max[axis] - 1
            else:
                assert isinstance(mapping, ArrayMap)
                first, last = int(mapping.index_array.min()), int(mapping.index_array.max())
            lo = checked_affine(mapping.offset, mapping.stride, first)
            hi = checked_affine(mapping.offset, mapping.stride, last)
        grid.index_to_chunk(min(lo, hi))
        grid.index_to_chunk(max(lo, hi))


def _write_units(
    transform: IndexTransform, grids: tuple[DimensionGridLike, ...]
) -> set[tuple[int, ...]]:
    _validate_bounds(transform, grids)
    if 0 in transform.domain.shape:
        return set()
    plan = plan_chunks(transform, grids)
    if _shared_input_axis(transform) is not None:
        return {projection.chunk_coords for projection in plan}
    return {
        tuple(int(c) for c in row)
        for batch in plan.partition().chunk_coord_batches()
        for row in batch
    }


def _schedule(
    footprints: Iterable[set[tuple[int, ...]]], order: Literal["preserve", "reorder"]
) -> WriteSchedule:
    batches: list[list[int]] = []
    last_batch: dict[tuple[int, ...], int] = {}
    colors: dict[tuple[int, ...], set[int]] = {}
    first_free: dict[tuple[int, ...], int] = {}
    memberships = 0
    for task, units in enumerate(footprints):
        memberships += len(units)
        if order == "preserve":
            batch = max((last_batch.get(unit, -1) + 1 for unit in units), default=0)
            for unit in units:
                last_batch[unit] = batch
        else:
            # No batch below this bound can be free for every unit. Keeping
            # each unit's first free color avoids rescanning a hot unit's
            # entire history when every task conflicts with all earlier ones.
            batch = max((first_free.get(unit, 0) for unit in units), default=0)
            while any(batch in colors.get(unit, ()) for unit in units):
                batch += 1
            for unit in units:
                used = colors.setdefault(unit, set())
                used.add(batch)
                free = first_free.get(unit, 0)
                while free in used:
                    free += 1
                first_free[unit] = free
        while len(batches) <= batch:
            batches.append([])
        batches[batch].append(task)
    return WriteSchedule(
        tuple(tuple(batch) for batch in batches),
        order,
        len(last_batch) if order == "preserve" else len(colors),
        memberships,
    )


def plan_write_batches(
    writes: Iterable[IndexTransform],
    destination_grid: Sequence[DimensionGridLike],
    *,
    order: Literal["preserve", "reorder"] = "preserve",
) -> WriteSchedule:
    """Group write tasks so each destination unit has one writer per batch.

    Parameters
    ----------
    writes
        Finite iterable of request-to-destination transforms. Input positions
        identify tasks. A transform describes the entire write of one task.
        Repeated destinations within that task remain its own responsibility.
    destination_grid
        One dimension grid per destination axis, describing independent write
        units. Use shard grids when writes replace shard objects. Grids must
        remain unchanged throughout planning and execution.
    order
        ``preserve`` (default) keeps input order between conflicting tasks.
        ``reorder`` permits reordering and chooses the first available batch;
        use only when task order does not determine the required final values.

    Returns
    -------
    WriteSchedule
        Reusable metadata-only schedule. Complete every batch before starting
        destination reads or writes in the next. This is not an executor, lock,
        byte-memory budget, or minimum-batch optimizer. Planning is eager and
        retains task IDs and per-unit scheduling state, not a pairwise graph.

    Raises
    ------
    ValueError
        If the order or grid rank is invalid, or a transform requires a
        partition representation that ``plan_chunks`` does not support.
    IndexError
        If a nonempty write reaches outside a destination grid.
    NotImplementedError
        For transform combinations unsupported by ``plan_chunks``, including
        index-array and affine maps sharing an input axis. Pure affine
        diagonals retain the projection-iteration compatibility path.

    Examples
    --------
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> base = IndexTransform.from_shape((12,))
    >>> writes = [base[i:i + 3] for i in range(0, 12, 3)]
    >>> grid = dimension_grids_from_chunks((4,), (12,))
    >>> plan_write_batches(writes, grid, order="reorder").batches
    ((0, 2), (1, 3))
    """
    if order not in ("preserve", "reorder"):
        raise ValueError("order must be 'preserve' or 'reorder'")
    grids = tuple(destination_grid)
    return _schedule((_write_units(transform, grids) for transform in writes), order)


def plan_rechunk(
    domain: IndexDomain,
    source_grid: Sequence[DimensionGridLike],
    destination_grid: Sequence[DimensionGridLike],
) -> RechunkPlan:
    """Schedule source chunks copied to another grid over the same coordinates.

    Parameters
    ----------
    domain
        Finite global-coordinate region to copy. Nonzero origins and clipped
        source chunks are supported; no coordinate translation is implied.
    source_grid
        Source chunk dimensions. Each intersecting source chunk remains one
        task even when its values cross several destination write units.
    destination_grid
        Destination independent write-unit dimensions, e.g. shards. Source and
        destination storage must not alias, and unrelated writers must be
        excluded. Both grids must remain unchanged during planning and copying.

    Returns
    -------
    RechunkPlan
        Source pieces and a conflict-free reorder schedule. No temporary store,
        task coalescing, byte-memory bound, or data movement is provided.

    Examples
    --------
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> domain = IndexDomain.from_shape((12,))
    >>> source = dimension_grids_from_chunks((3,), domain.shape)
    >>> target = dimension_grids_from_chunks((4,), domain.shape)
    >>> plan = plan_rechunk(domain, source, target)
    >>> len(plan.pieces), plan.schedule.batches
    (4, ((0, 2), (1, 3)))
    """
    source, target = tuple(source_grid), tuple(destination_grid)
    if len(source) != domain.ndim or len(target) != domain.ndim:
        raise ValueError("source and destination grid rank must match domain rank")
    identity = IndexTransform.identity(domain)
    _validate_bounds(identity, source)
    _validate_bounds(identity, target)

    def pieces() -> Iterator[RechunkPiece]:
        for projection in plan_chunks(identity, source):
            lo = tuple(
                max(a, b)
                for a, b in zip(domain.origin, projection.chunk_domain.origin, strict=True)
            )
            hi = tuple(
                min(a, b)
                for a, b in zip(
                    domain.exclusive_max, projection.chunk_domain.exclusive_max, strict=True
                )
            )
            selection = tuple(
                slice(a - grid.chunk_offset(c), b - grid.chunk_offset(c))
                for a, b, grid, c in zip(lo, hi, source, projection.chunk_coords, strict=True)
            )
            yield RechunkPiece(
                projection.chunk_coords, selection, IndexTransform.identity(IndexDomain(lo, hi))
            )

    result = tuple(pieces())
    schedule = plan_write_batches((piece.destination for piece in result), target, order="reorder")
    return RechunkPlan(result, schedule)
