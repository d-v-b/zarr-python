"""Chunk resolution — mapping transforms to chunk-level I/O.

Given an `IndexTransform` (which coordinates a request reads) and one grid per
storage dimension (how storage is divided into chunks), chunk resolution
answers:

    For each chunk, which storage coordinates does this transform touch,
    and where do those values land in the request?

The public result is a lazy, reusable `ChunkPlan` whose rows are
`ChunkProjection`s. Each identifies a chunk and pairs a chunk-local transform
with a transform back to the request's cells, over one shared zero-origin
cell domain, without assuming NumPy selectors, a codec pipeline, or a
scheduler.

The plan is computed in factored form, the `GridPartition`. Restricting a
transform to a chunk box distributes over output dimensions whenever each
output map reads its own input axis — every basic and orthogonal selection —
so each axis is resolved once against its grid into a table:

- `StridedSet` — a `ConstantMap` or `DimensionMap` axis: one row per touched
  chunk, holding the chunk-local start, the extent, the request origin of the
  first cell, and whether the row covers its chunk exactly once.
- `IndexedSet` — an orthogonal `ArrayMap` axis: its coordinates grouped by
  chunk in CSR form, with the request positions they fill.
- `JointSet` — the correlated (`vindex`) index arrays, which read the same
  input axes and so do not distribute: their points are sorted into chunks
  together, once.

A projection is one row of each table combined. Building the tables costs the
sum of the touched chunks per axis rather than their product, rows are
materialized only on request, and a consumer may read the tables directly
instead. The one transform with no factored form is a hand-built diagonal,
two output maps reading one input axis; `ChunkPlan` walks those by
intersecting the whole transform with each candidate chunk
(`IndexTransform.intersect`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from zarr_indexing._affine import checked_affine
from zarr_indexing.domain import IndexDomain
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr_indexing.transform import (
    IndexTransform,
    _intersect_dimension_map,  # pyright: ignore[reportPrivateUsage]
    _intersect_general,  # pyright: ignore[reportPrivateUsage]
    _prepare_correlated,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from zarr_indexing.grid import DimensionGridLike

_OutIndices = (
    dict[int, np.ndarray[Any, np.dtype[np.intp]]] | np.ndarray[Any, np.dtype[np.intp]] | None
)

_ChunkTransformResult = tuple[
    tuple[int, ...],
    IndexTransform,
    _OutIndices,
]

type ChunkCoverage = Literal["full", "partial", "unknown"]


def _data_size(dim_grid: DimensionGridLike, chunk_ix: int) -> int:
    """Return a chunk's data extent, falling back for narrow-protocol grids."""
    data_size = getattr(dim_grid, "data_size", None)
    if data_size is None:
        return dim_grid.chunk_size(chunk_ix)
    return int(data_size(chunk_ix))


@dataclass(frozen=True, slots=True)
class ChunkProjection:
    """One source-independent projection of a request through a chunk.

    Both transforms share a synthetic input domain. ``chunk_transform`` maps
    that domain to chunk-local storage coordinates; ``cell_transform`` maps it
    to the original request domain.

    Attributes
    ----------
    chunk_coords
        Coordinates of the selected cell in the caller's grid.
    chunk_domain
        Bounds of that grid cell in global storage coordinates.
    chunk_transform
        Mapping from the shared synthetic domain to chunk-local storage.
    cell_transform
        Mapping from the shared synthetic domain to request coordinates.
    coverage
        Whether the request is proven to cover the whole grid cell exactly
        once. Fancy selections are conservatively ``"unknown"``.

    Examples
    --------
    Row 1 of a `(3, 4)` array with `(2, 2)` chunks touches only part of the
    first chunk, whose domain spans rows `[0, 2)` and columns `[0, 2)`:

    >>> from zarr_indexing import IndexTransform
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((2, 2), shape=(3, 4))
    >>> plan = plan_chunks(IndexTransform.from_shape((3, 4))[1, :], grids)
    >>> first = next(iter(plan))
    >>> first.chunk_coords
    (0, 0)
    >>> first.chunk_domain.shape
    (2, 2)
    >>> first.coverage
    'partial'
    """

    chunk_coords: tuple[int, ...]
    chunk_domain: IndexDomain
    chunk_transform: IndexTransform
    cell_transform: IndexTransform
    coverage: ChunkCoverage

    def __post_init__(self) -> None:
        if self.chunk_transform.domain != self.cell_transform.domain:
            raise ValueError(
                "chunk_transform and cell_transform must share an input domain; "
                f"got {self.chunk_transform.domain!r} and {self.cell_transform.domain!r}"
            )


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """A reusable, lazy partition of an index transform over a chunk grid.

    Construct plans with `plan_chunks`; iterating either the plan or
    `projections()` performs a fresh chunk walk. `partition()` exposes the
    factored form the walk is derived from (see `GridPartition`).

    Examples
    --------
    Row 1 of a `(3, 4)` array with `(2, 2)` chunks crosses two chunks, and
    the plan can be walked again after it is exhausted:

    >>> from zarr_indexing import IndexTransform
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((2, 2), shape=(3, 4))
    >>> plan = plan_chunks(IndexTransform.from_shape((3, 4))[1, :], grids)
    >>> [p.chunk_coords for p in plan]
    [(0, 0), (0, 1)]
    >>> [p.chunk_coords for p in plan.projections()]
    [(0, 0), (0, 1)]
    """

    transform: IndexTransform
    """The composed request this plan partitions."""

    dimension_grids: tuple[DimensionGridLike, ...]
    """One grid per storage dimension, defining the chunk layout the plan walks."""

    # Memoized factored form; derived state, excluded from identity (see
    # `IndexDomain._shape` for the same pattern).
    _partition: GridPartition | None = field(default=None, init=False, repr=False, compare=False)

    def partition(self) -> GridPartition:
        """The plan in factored, columnar form: one table per axis plus a joint table.

        Built once per plan and memoized. Raises `ValueError` for the one
        structure that has no factored form, a transform whose output maps
        share an input axis (a diagonal); `projections()` still walks those.
        """
        cached = self._partition
        if cached is None:
            cached = partition_transform(self.transform, self.dimension_grids)
            object.__setattr__(self, "_partition", cached)
        return cached

    def projections(self) -> Iterator[ChunkProjection]:
        """Return a fresh iterator over the chunks touched by this plan."""
        if _partitionable(self.transform):
            return iter(self.partition())
        return _iter_general_projections(self.transform, self.dimension_grids)

    def __iter__(self) -> Iterator[ChunkProjection]:
        """Equivalent to `projections()`: each iteration performs a fresh chunk walk."""
        return self.projections()


def plan_chunks(
    transform: IndexTransform,
    dimension_grids: Sequence[DimensionGridLike],
) -> ChunkPlan:
    """Plan a transform against a caller-selected chunk grid.

    Parameters
    ----------
    transform
        Mapping from the request domain to storage coordinates.
    dimension_grids
        One storage grid per transform output dimension.

    Returns
    -------
    ChunkPlan
        A reusable plan whose projections are computed lazily; its
        `partition()` is the factored form they are derived from.

    Examples
    --------
    Row 1 of a `(3, 4)` array with `(2, 2)` chunks touches the two chunks in
    the top grid row, each contributing a `(2, 2)` chunk domain:

    >>> from zarr_indexing import IndexTransform
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((2, 2), shape=(3, 4))
    >>> plan = plan_chunks(IndexTransform.from_shape((3, 4))[1, :], grids)
    >>> [p.chunk_coords for p in plan]
    [(0, 0), (0, 1)]
    >>> [p.chunk_domain.shape for p in plan]
    [(2, 2), (2, 2)]
    """
    grids = tuple(dimension_grids)
    if len(grids) != transform.output_rank:
        raise ValueError(
            "dimension_grids must have one entry per transform output dimension; "
            f"got {len(grids)} grids for output rank {transform.output_rank}"
        )
    return ChunkPlan(transform=transform, dimension_grids=grids)


def _one_dimensional_array_map(
    transform: IndexTransform,
) -> tuple[ArrayMap, np.ndarray[Any, np.dtype[np.intp]]] | None:
    """Return a nonempty 1-D single-ArrayMap transform's map and storage coords.

    A one-dimensional array selection has no cross-dimensional correlation to
    preserve — the orthogonal and vectorized flavors coincide there — so the
    sorted fast path applies to either spelling. The computed storage
    coordinates are also reused by general resolution when they are unsorted.
    """
    if transform.input_rank != 1 or transform.output_rank != 1:
        return None

    m = transform.output[0]
    if not isinstance(m, ArrayMap) or m.index_array.ndim != 1 or m.index_array.size == 0:
        return None

    return m, checked_affine(m.offset, m.stride, m.index_array)


def _iter_sorted_1d_array_map(
    m: ArrayMap,
    storage: np.ndarray[Any, np.dtype[np.intp]],
    dim_grid: DimensionGridLike,
) -> Iterator[_ChunkTransformResult]:
    """Resolve a sorted 1-D ArrayMap one touched chunk at a time."""
    start = 0
    while start < storage.size:
        chunk = dim_grid.index_to_chunk(int(storage[start]))
        chunk_start = dim_grid.chunk_offset(chunk)
        chunk_stop = chunk_start + _data_size(dim_grid, chunk)
        stop = int(np.searchsorted(storage, chunk_stop, side="left"))

        restricted = IndexTransform(
            domain=IndexDomain(inclusive_min=(0,), exclusive_max=(stop - start,)),
            output=(
                ArrayMap(
                    index_array=m.index_array[start:stop],
                    offset=m.offset,
                    stride=m.stride,
                ),
            ),
        )
        local = restricted.translate((-chunk_start,))
        surviving = np.arange(start, stop, dtype=np.intp)

        yield (chunk,), local, surviving
        start = stop


def _group_points_by_chunk(
    chunk_ids: Sequence[np.ndarray[Any, np.dtype[np.intp]]],
) -> list[tuple[tuple[int, ...], np.ndarray[Any, np.dtype[np.intp]]]]:
    """Partition the points of a correlated block by the chunk each lands in.

    ``chunk_ids`` holds, per correlated output dimension, the chunk index of
    every block point. Returns one ``(chunk_coords, positions)`` pair per
    touched chunk, in lexicographic chunk order, with ``positions`` (flat
    indices into the block) ascending — the same order `np.nonzero` would
    give. Costs ``O(points log points)`` regardless of grid size, so a
    selection scattered over many chunks is not rescanned once per chunk.
    """
    n = int(chunk_ids[0].size)
    if n == 0:
        return []
    keys: np.ndarray[Any, np.dtype[np.intp]] | None
    if len(chunk_ids) == 1:
        keys = np.asarray(chunk_ids[0], dtype=np.intp)
    else:
        # Mixed-radix key with the first dimension most significant, so sorting
        # the keys sorts the chunk coordinates lexicographically.
        keys = np.zeros(n, dtype=np.intp)
        multiplier = 1
        for ids in reversed(chunk_ids):
            radix = int(ids.max()) + 1
            if multiplier * radix >= 2**62:
                keys = None
                break
            keys += np.asarray(ids, dtype=np.intp) * multiplier
            multiplier *= radix
        if keys is None:
            stacked = np.stack(
                [np.asarray(ids, dtype=np.intp).ravel() for ids in chunk_ids], axis=1
            )
            _, inverse = np.unique(stacked, axis=0, return_inverse=True)
            keys = np.asarray(inverse, dtype=np.intp).reshape(-1)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    starts = [0, *boundaries.tolist()]
    ends = [*starts[1:], n]
    groups: list[tuple[tuple[int, ...], np.ndarray[Any, np.dtype[np.intp]]]] = []
    for start, stop in zip(starts, ends, strict=True):
        first = order[start]
        coords = tuple(int(ids[first]) for ids in chunk_ids)
        groups.append((coords, order[start:stop]))
    return groups


def _iter_chunk_transform_results(
    transform: IndexTransform,
    dim_grids: Sequence[DimensionGridLike],
) -> Iterator[_ChunkTransformResult]:
    """Resolve a transform into private intersection bookkeeping.

    The survivor arrays are an implementation detail immediately converted to
    a public `cell_transform` by `_iter_chunk_projections`.
    """

    if any(size == 0 for size in transform.domain.shape):
        # An empty view touches no chunk. Checked on the domain rather than on
        # the index arrays: an axis of genuine extent 1 is stored as a broadcast
        # singleton, so a slice that empties the domain does not shrink the
        # array, and the emptiness shows only here.
        return

    array_map_1d = _one_dimensional_array_map(transform)
    if array_map_1d is not None:
        sorted_map, storage = array_map_1d
        if storage[0] <= storage[-1] and bool(np.all(storage[1:] >= storage[:-1])):
            dim_grid = dim_grids[0]
            first_chunk = dim_grid.index_to_chunk(int(storage[0]))
            if dim_grid.chunk_size(first_chunk) > 0:
                yield from _iter_sorted_1d_array_map(sorted_map, storage, dim_grid)
                return

    # Enumerate candidate chunks via the cartesian product of per-slot candidate
    # chunk ids, then for each candidate intersect the transform with the chunk
    # domain (`transform.intersect` handles orthogonal and vectorized cases
    # alike, filtering out combinations it does not actually touch).
    #
    # A slot covers one or more output dimensions and contributes exactly the
    # chunk-coordinate tuples those dimensions can touch:
    #
    # - `ConstantMap`/`DimensionMap` dims each form their own slot with a
    #   contiguous range — a single chunk for a constant, and the span between
    #   the first and last chunk for a slice. These are already tight (or
    #   nearly so).
    # - Orthogonal `ArrayMap` (fancy) dims each form their own slot with only
    #   the *distinct* chunk ids the index array actually lands in
    #   (`np.unique`), never the dense `range(min_chunk, max_chunk + 1)`
    #   between them. A sparse fancy selection (e.g. two far-apart coordinates)
    #   would otherwise enumerate every chunk in the bounding box, making
    #   resolution scale with grid size instead of with the number of selected
    #   coordinates.
    # - Correlated (vindex) `ArrayMap` dims share one *joint* slot holding the
    #   distinct chunk-coordinate tuples the points actually land in, found by
    #   sorting the points once (`_group_points_by_chunk`). The cartesian
    #   product of their per-dimension distinct sets would include
    #   combinations no point touches — quadratic in the number of selected
    #   points for a diagonal selection — while the joint distinct set is
    #   bounded by the point count (see zarr-python gh-4174). The same sort
    #   hands each chunk its surviving points, so the per-chunk intersection
    #   never rescans the whole selection.
    structure = transform.index_array_structure
    block = _prepare_correlated(transform) if structure == "general" else None
    correlated_dims: list[int] = []
    slot_dims: list[tuple[int, ...]] = []
    slot_candidates: list[Sequence[tuple[int, ...]]] = []
    for out_dim, m in enumerate(transform.output):
        dg = dim_grids[out_dim]
        if isinstance(m, ConstantMap):
            # Single chunk
            coordinate = checked_affine(m.offset, 0, 0)
            c = dg.index_to_chunk(coordinate)
            slot_dims.append((out_dim,))
            slot_candidates.append(((c,),))
        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            dim_lo = transform.domain.inclusive_min[d]
            dim_hi = transform.domain.exclusive_max[d]
            if dim_lo >= dim_hi:
                return  # empty domain
            slot_dims.append((out_dim,))
            slot_candidates.append(_dimension_map_candidates(m, dim_lo, dim_hi, dg))
        elif block is None:
            # m: ArrayMap with orthogonal structure.
            # Storage coordinates were already computed for a correlated 1-D map.
            storage = (
                array_map_1d[1]
                if array_map_1d is not None
                else checked_affine(m.offset, m.stride, m.index_array)
            )
            if storage.size == 0:
                # Empty fancy selection: no coordinates, so no chunks are touched.
                return
            chunk_ids = dg.indices_to_chunks(storage)
            slot_dims.append((out_dim,))
            slot_candidates.append([(int(c),) for c in np.unique(chunk_ids)])
        else:
            correlated_dims.append(out_dim)

    joint_slot: int | None = None
    joint_positions: dict[tuple[int, ...], np.ndarray[Any, np.dtype[np.intp]]] = {}
    if correlated_dims:
        assert block is not None
        chunk_ids_per_dim = [
            dim_grids[out_dim].indices_to_chunks(block.flat_storage[out_dim])
            for out_dim in correlated_dims
        ]
        groups = _group_points_by_chunk(chunk_ids_per_dim)
        if not groups:
            return
        joint_slot = len(slot_dims)
        slot_dims.append(tuple(correlated_dims))
        slot_candidates.append([coords for coords, _ in groups])
        joint_positions = dict(groups)

    output_rank = len(transform.output)
    for combo in itertools.product(*slot_candidates):
        chunk_coords_list = [0] * output_rank
        for dims, part in zip(slot_dims, combo, strict=True):
            for d, c in zip(dims, part, strict=True):
                chunk_coords_list[d] = c
        chunk_coords = tuple(chunk_coords_list)

        # Build the chunk domain in storage space
        chunk_min: list[int] = []
        chunk_max: list[int] = []
        chunk_shift: list[int] = []
        for out_dim, c in enumerate(chunk_coords):
            dg = dim_grids[out_dim]
            c_start = dg.chunk_offset(c)
            c_size = _data_size(dg, c)
            chunk_min.append(c_start)
            chunk_max.append(c_start + c_size)
            chunk_shift.append(-c_start)

        chunk_domain = IndexDomain(
            inclusive_min=tuple(chunk_min),
            exclusive_max=tuple(chunk_max),
        )

        # Intersect transform with chunk domain
        result: tuple[IndexTransform, _OutIndices] | None
        if block is not None and joint_slot is not None:
            result = _intersect_general(
                transform, chunk_domain, block=block, positions=joint_positions[combo[joint_slot]]
            )
        else:
            result = transform.intersect(chunk_domain)
        if result is None:
            continue

        restricted, surviving = result

        # Translate to chunk-local coordinates
        local = restricted.translate(tuple(chunk_shift))

        yield (chunk_coords, local, surviving)


def _dimension_map_candidates(
    m: DimensionMap, dim_lo: int, dim_hi: int, dg: DimensionGridLike
) -> Sequence[tuple[int, ...]]:
    """The chunks a nonempty `DimensionMap` over input `[dim_lo, dim_hi)` can touch, ascending."""
    first_storage = checked_affine(m.offset, m.stride, dim_lo)
    if m.stride > 0:
        s_min = first_storage
        s_max = checked_affine(m.offset, m.stride, dim_hi - 1)
    elif m.stride < 0:
        s_min = checked_affine(m.offset, m.stride, dim_hi - 1)
        s_max = first_storage
    else:
        s_min = s_max = first_storage
    first = dg.index_to_chunk(s_min)
    last = dg.index_to_chunk(s_max)
    point_count = dim_hi - dim_lo
    chunk_count = last - first + 1
    if point_count < chunk_count:
        steps = np.arange(point_count, dtype=np.intp)
        storage = checked_affine(first_storage, m.stride, steps)
        chunk_ids = dg.indices_to_chunks(storage)
        return [(int(c),) for c in np.unique(chunk_ids)]
    return [(c,) for c in range(first, last + 1)]


def _covers_whole_chunk(transform: IndexTransform, chunk_shape: tuple[int, ...]) -> bool:
    """Whether an affine chunk-local transform bijects onto every chunk cell."""
    domain = transform.domain
    used_nontrivial_inputs: set[int] = set()
    for out_dim, m in enumerate(transform.output):
        extent = chunk_shape[out_dim]
        if isinstance(m, ConstantMap):
            if extent != 1 or m.offset != 0:
                return False
        elif isinstance(m, DimensionMap):
            if abs(m.stride) != 1:
                return False
            lo = domain.inclusive_min[m.input_dimension]
            hi = domain.exclusive_max[m.input_dimension]
            if hi <= lo:
                if extent != 0:
                    return False
                continue
            first = m.offset + m.stride * lo
            last = m.offset + m.stride * (hi - 1)
            if min(first, last) != 0 or max(first, last) != extent - 1:
                return False
            if extent > 1:
                if m.input_dimension in used_nontrivial_inputs:
                    return False
                used_nontrivial_inputs.add(m.input_dimension)
        else:
            return False
    nontrivial_inputs = {dimension for dimension, extent in enumerate(domain.shape) if extent > 1}
    return used_nontrivial_inputs == nontrivial_inputs


def _orthogonal_cell_transform(
    original: IndexTransform,
    restricted: IndexTransform,
    survivors: dict[int, np.ndarray[Any, np.dtype[np.intp]]] | np.ndarray[Any, np.dtype[np.intp]],
) -> IndexTransform:
    """Map a compacted orthogonal intersection back to request coordinates."""
    by_input_dimension: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {}
    if isinstance(survivors, dict):
        survivor_items = survivors.items()
    else:
        array_output_dimensions = [
            output_dimension
            for output_dimension, output_map in enumerate(original.output)
            if isinstance(output_map, ArrayMap)
        ]
        if len(array_output_dimensions) != 1:
            raise ValueError(
                "one survivor array requires exactly one orthogonal ArrayMap; "
                f"found output dimensions {array_output_dimensions}"
            )
        survivor_items = ((array_output_dimensions[0], survivors),)

    for output_dimension, positions in survivor_items:
        output_map = original.output[output_dimension]
        if not isinstance(output_map, ArrayMap):
            raise TypeError(
                f"survivors for output dimension {output_dimension} do not describe an ArrayMap"
            )
        input_dimension = output_map.dependent_axis
        if input_dimension is None:
            raise ValueError(
                f"output dimension {output_dimension} has no orthogonal input dimension"
            )
        by_input_dimension[input_dimension] = np.asarray(positions, dtype=np.intp)

    output: list[ConstantMap | DimensionMap | ArrayMap] = []
    rank = original.input_rank
    for input_dimension in range(rank):
        positions = by_input_dimension.get(input_dimension)
        if positions is None:
            output.append(DimensionMap(input_dimension=input_dimension))
            continue
        shape = (1,) * input_dimension + (positions.size,) + (1,) * (rank - input_dimension - 1)
        output.append(
            ArrayMap(
                index_array=positions.reshape(shape),
                offset=original.domain.inclusive_min[input_dimension],
            )
        )
    return IndexTransform(domain=restricted.domain, output=tuple(output))


def _correlated_cell_transform(
    original: IndexTransform,
    restricted: IndexTransform,
    survivors: np.ndarray[Any, np.dtype[np.intp]],
) -> IndexTransform:
    """Map compacted correlated points back through the request's row-major domain."""
    positions = np.asarray(survivors, dtype=np.intp)
    # Correlated broadcast axes already contribute positional survivor offsets;
    # residual affine axes still contribute literal coordinates. Remove only
    # the latter origins before unraveling the fully positional flat offsets.
    literal_axes = {
        output_map.input_dimension
        for output_map in original.output
        if isinstance(output_map, DimensionMap)
    }
    origin_offset = 0
    flat_stride = 1
    for input_dimension in range(original.input_rank - 1, -1, -1):
        if input_dimension in literal_axes:
            origin_offset += original.domain.inclusive_min[input_dimension] * flat_stride
        extent = original.domain.shape[input_dimension]
        flat_stride *= extent
    coordinates = np.unravel_index(
        checked_affine(-origin_offset, 1, positions), original.domain.shape
    )
    output = tuple(
        ArrayMap(
            index_array=np.asarray(coordinate, dtype=np.intp),
            offset=origin,
        )
        for coordinate, origin in zip(coordinates, original.domain.inclusive_min, strict=True)
    )
    return IndexTransform(domain=restricted.domain, output=output)


def _cell_transform(
    original: IndexTransform,
    restricted: IndexTransform,
    survivors: _OutIndices,
) -> IndexTransform:
    """Convert private survivor bookkeeping into a direction-neutral transform."""
    if survivors is None:
        return IndexTransform.identity(restricted.domain)
    if original.index_array_structure == "general":
        if isinstance(survivors, dict):
            raise ValueError("general intersections require one shared survivor array")
        return _correlated_cell_transform(original, restricted, survivors)
    return _orthogonal_cell_transform(original, restricted, survivors)


def _iter_general_projections(
    transform: IndexTransform,
    dim_grids: Sequence[DimensionGridLike],
) -> Iterator[ChunkProjection]:
    """Convert private intersection results into public paired projections.

    The general walk: intersect the whole transform with every candidate
    chunk. `GridPartition` covers every transform a selection can produce;
    this remains for hand-built diagonals, which have no factored form.
    """
    if any(size == 0 for size in transform.domain.shape):
        return
    for chunk_coords, chunk_transform, survivors in _iter_chunk_transform_results(
        transform, dim_grids
    ):
        chunk_min = tuple(
            grid.chunk_offset(coord) for grid, coord in zip(dim_grids, chunk_coords, strict=True)
        )
        chunk_shape = tuple(
            _data_size(grid, coord) for grid, coord in zip(dim_grids, chunk_coords, strict=True)
        )
        chunk_domain = IndexDomain(
            inclusive_min=chunk_min,
            exclusive_max=tuple(
                origin + extent for origin, extent in zip(chunk_min, chunk_shape, strict=True)
            ),
        )
        cell_transform = _cell_transform(transform, chunk_transform, survivors)
        synthetic_origin = (0,) * chunk_transform.input_rank
        chunk_transform = chunk_transform.translate_domain_to(synthetic_origin)
        cell_transform = cell_transform.translate_domain_to(synthetic_origin)
        if survivors is not None or any(isinstance(m, ArrayMap) for m in chunk_transform.output):
            coverage: ChunkCoverage = "unknown"
        elif _covers_whole_chunk(chunk_transform, chunk_shape):
            coverage = "full"
        else:
            coverage = "partial"
        yield ChunkProjection(
            chunk_coords=chunk_coords,
            chunk_domain=chunk_domain,
            chunk_transform=chunk_transform,
            cell_transform=cell_transform,
            coverage=coverage,
        )


# --------------------------------------------------------------------------- #
# Grid partition: the factored form of a plan
# --------------------------------------------------------------------------- #
#
# Restricting a transform to a chunk box distributes over output dimensions
# whenever each output map reads its own input axis: the domain is a product
# of intervals, each map depends on one of them, and the box is a product. The
# chunks such a transform touches are then the cartesian product of the chunks
# each axis touches, and the restriction to any one of them is the product of
# one-dimensional restrictions. `GridPartition` stores those one-dimensional
# restrictions as tables (`StridedSet`, `IndexedSet`); a `ChunkProjection` is
# one row of each table, combined. Correlated index arrays (`vindex`) read the
# same input axes, so they do not distribute; they are sorted into chunks once
# and kept in a single `JointSet`. This is the structure TensorStore's
# `IndexTransformGridPartition` uses (strided sets and index-array sets).


def _factorizable(transform: IndexTransform) -> bool:
    """True when every output map binds its own input axis and none is correlated."""
    if transform.index_array_structure == "general":
        return False
    seen: set[int] = set()
    for m in transform.output:
        if isinstance(m, DimensionMap):
            axis = m.input_dimension
        elif isinstance(m, ArrayMap):
            dependent = m.dependent_axis
            if dependent is None:
                return False
            axis = dependent
        else:
            continue
        if axis in seen:
            return False
        seen.add(axis)
    return True


def _correlated_partitionable(transform: IndexTransform) -> bool:
    """True when a general transform's index arrays vary only over its broadcast axes."""
    if transform.index_array_structure != "general":
        return False
    bound = {m.input_dimension for m in transform.output if isinstance(m, DimensionMap)}
    return all(
        not any(axis in bound for axis in m.dependency_axes)
        for m in transform.output
        if isinstance(m, ArrayMap)
    )


def _partitionable(transform: IndexTransform) -> bool:
    return _factorizable(transform) or _correlated_partitionable(transform)


def _int_column(values: Sequence[int]) -> np.ndarray[Any, np.dtype[np.intp]]:
    """An ``intp`` column, or an object column of Python ints when a value does not fit.

    Request coordinates are unbounded Python ints in the transform algebra; a
    domain near the ``intp`` limit (see `IndexDomain`) still partitions, its
    table just carries exact ints.
    """
    try:
        return np.array(values, dtype=np.intp)
    except OverflowError:
        return cast("np.ndarray[Any, np.dtype[np.intp]]", np.array(values, dtype=object))


def _chunk_bounds(
    dg: DimensionGridLike, chunks: np.ndarray[Any, np.dtype[np.intp]]
) -> tuple[np.ndarray[Any, np.dtype[np.intp]], np.ndarray[Any, np.dtype[np.intp]]]:
    """Storage origin and data extent of each chunk, one grid call per touched chunk."""
    n = int(chunks.size)
    starts = np.fromiter((dg.chunk_offset(int(c)) for c in chunks), dtype=np.intp, count=n)
    extents = np.fromiter((_data_size(dg, int(c)) for c in chunks), dtype=np.intp, count=n)
    return starts, extents


@dataclass(frozen=True, slots=True)
class StridedSet:
    """One output dimension read through a `ConstantMap` or `DimensionMap`, one row per chunk.

    Row ``i`` is the map restricted to chunk ``chunk[i]`` and re-based to
    chunk-local, zero-origin coordinates: the chunk-local map is
    `DimensionMap(input_dimension, offset=local_start[i], stride=stride)` over
    ``[0, extent[i])`` (a `ConstantMap(local_start[i])` for a constant), and
    its cells are request coordinates ``[origin[i], origin[i] + extent[i])``
    along the input axis.

    Examples
    --------
    >>> from zarr_indexing import IndexTransform, plan_chunks
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((4,), shape=(10,))
    >>> (axis,) = plan_chunks(IndexTransform.from_shape((10,))[1:9:2], grids).partition().sets
    >>> axis.chunk.tolist(), axis.local_start.tolist(), axis.extent.tolist(), axis.origin.tolist()
    ([0, 1], [1, 1], [2, 2], [0, 2])
    """

    output_dimension: int
    """The storage axis this table describes."""

    input_dimension: int | None
    """The request axis the map reads, or `None` for a constant."""

    stride: int
    """Storage step per request cell; ``0`` for a constant."""

    chunk: np.ndarray[Any, np.dtype[np.intp]]
    """Chunk index along the axis, one per row, ascending."""

    chunk_start: np.ndarray[Any, np.dtype[np.intp]]
    """Storage origin of each chunk."""

    chunk_extent: np.ndarray[Any, np.dtype[np.intp]]
    """Data extent of each chunk (clipped at the array boundary)."""

    local_start: np.ndarray[Any, np.dtype[np.intp]]
    """Chunk-local storage coordinate of the row's first cell."""

    extent: np.ndarray[Any, np.dtype[np.intp]]
    """Cells the row selects along the request axis (``1`` for a constant)."""

    origin: np.ndarray[Any, np.dtype[np.intp]]
    """Request coordinate of the row's first cell (``0`` for a constant)."""

    full: np.ndarray[Any, np.dtype[np.bool_]]
    """Whether the row covers its chunk's data extent exactly once, in order."""

    def __len__(self) -> int:
        return int(self.chunk.size)

    def chunk_map(self, row: int, input_dimension: int | None = None) -> OutputIndexMap:
        """The chunk-local output map of a row, optionally with a renumbered input axis."""
        offset = int(self.local_start[row])
        if self.input_dimension is None:
            return ConstantMap(offset=offset)
        axis = self.input_dimension if input_dimension is None else input_dimension
        return DimensionMap(input_dimension=axis, offset=offset, stride=self.stride)

    def cell_map(self, row: int) -> DimensionMap | None:
        """The map from the row's zero-origin cells back to request coordinates."""
        if self.input_dimension is None:
            return None
        return DimensionMap(input_dimension=self.input_dimension, offset=int(self.origin[row]))


@dataclass(frozen=True, slots=True)
class IndexedSet:
    """One output dimension read through an orthogonal `ArrayMap`, one row per chunk.

    The map's coordinates are grouped by chunk in CSR form: row ``i`` owns
    ``index[pointer[i]:pointer[i + 1]]`` (the index-array values, in request
    order) and ``positions[pointer[i]:pointer[i + 1]]`` (their positions along
    the request axis, ascending). `local` gives the same values as chunk-local
    storage coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> from zarr_indexing import IndexTransform, plan_chunks
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((4,), shape=(10,))
    >>> transform = IndexTransform.from_shape((10,)).oindex[np.array([9, 1, 2, 8])]
    >>> (axis,) = plan_chunks(transform, grids).partition().sets
    >>> axis.chunk.tolist(), axis.pointer.tolist()
    ([0, 2], [0, 2, 4])
    >>> axis.local.tolist(), axis.positions.tolist()
    ([1, 2, 1, 0], [1, 2, 0, 3])
    """

    output_dimension: int
    """The storage axis this table describes."""

    input_dimension: int
    """The request axis the index array varies over."""

    offset: int
    """The map's affine offset: storage is ``offset + stride * index``."""

    stride: int
    """The map's affine stride."""

    chunk: np.ndarray[Any, np.dtype[np.intp]]
    """Chunk index along the axis, one per row, ascending."""

    chunk_start: np.ndarray[Any, np.dtype[np.intp]]
    """Storage origin of each chunk."""

    chunk_extent: np.ndarray[Any, np.dtype[np.intp]]
    """Data extent of each chunk."""

    pointer: np.ndarray[Any, np.dtype[np.intp]]
    """CSR row pointer: row ``i`` owns entries ``pointer[i]`` to ``pointer[i + 1]``."""

    index: np.ndarray[Any, np.dtype[np.intp]]
    """Index-array values grouped by chunk."""

    positions: np.ndarray[Any, np.dtype[np.intp]]
    """Positions along the request axis, grouped by chunk, ascending within a row."""

    def __len__(self) -> int:
        return int(self.chunk.size)

    @property
    def counts(self) -> np.ndarray[Any, np.dtype[np.intp]]:
        """Entries per row."""
        return np.diff(self.pointer)

    @property
    def local(self) -> np.ndarray[Any, np.dtype[np.intp]]:
        """Chunk-local storage coordinate of every entry, grouped like `index`."""
        storage = checked_affine(self.offset, self.stride, self.index)
        return storage - np.repeat(self.chunk_start, self.counts)

    def run(self, row: int) -> slice:
        """The slice of `index` / `positions` a row owns."""
        return slice(int(self.pointer[row]), int(self.pointer[row + 1]))


@dataclass(frozen=True, slots=True)
class JointSet:
    """The correlated index arrays of a transform, grouped by the chunk each point lands in.

    Correlated (`vindex`) arrays read the same input axes, so a chunk
    constrains all of them at once; they are sorted into chunks together.
    Row ``i`` is one touched chunk, `chunk[i]` its coordinates on the
    `output_dimensions`, and CSR range ``pointer[i]:pointer[i + 1]`` its
    points: `index` holds their index-array values per output dimension,
    `positions` their flat positions in the request's broadcast block, and
    `block_coordinates` those positions unravelled over the block.

    Examples
    --------
    >>> import numpy as np
    >>> from zarr_indexing import IndexTransform, plan_chunks
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((3, 4), shape=(7, 9))
    >>> transform = IndexTransform.from_shape((7, 9)).vindex[
    ...     np.array([0, 6, 6, 1]), np.array([8, 0, 1, 8])
    ... ]
    >>> joint = plan_chunks(transform, grids).partition().joint
    >>> joint.chunk.tolist(), joint.pointer.tolist(), joint.positions.tolist()
    ([[0, 2], [2, 0]], [0, 2, 4], [0, 3, 1, 2])
    """

    output_dimensions: tuple[int, ...]
    """The storage axes read by correlated index arrays."""

    offsets: tuple[int, ...]
    """Affine offset of each array's map, aligned with `output_dimensions`."""

    strides: tuple[int, ...]
    """Affine stride of each array's map."""

    broadcast_axes: tuple[int, ...]
    """The request axes the arrays broadcast over."""

    broadcast_shape: tuple[int, ...]
    """The extent of those axes."""

    chunk: np.ndarray[Any, np.dtype[np.intp]]
    """Chunk coordinates on `output_dimensions`, shape ``(rows, k)``, lexicographic."""

    chunk_start: np.ndarray[Any, np.dtype[np.intp]]
    """Storage origin of each chunk on `output_dimensions`, shape ``(rows, k)``."""

    chunk_extent: np.ndarray[Any, np.dtype[np.intp]]
    """Data extent of each chunk on `output_dimensions`, shape ``(rows, k)``."""

    pointer: np.ndarray[Any, np.dtype[np.intp]]
    """CSR row pointer into `index`, `positions` and `block_coordinates`."""

    index: np.ndarray[Any, np.dtype[np.intp]]
    """Index-array values per point and output dimension, shape ``(points, k)``."""

    positions: np.ndarray[Any, np.dtype[np.intp]]
    """Flat block position of each point, ascending within a row."""

    block_coordinates: np.ndarray[Any, np.dtype[np.intp]]
    """`positions` unravelled over `broadcast_shape`, shape ``(points, len(broadcast_axes))``."""

    def __len__(self) -> int:
        return int(self.chunk.shape[0])

    @property
    def counts(self) -> np.ndarray[Any, np.dtype[np.intp]]:
        """Points per row."""
        return np.diff(self.pointer)

    @property
    def local(self) -> np.ndarray[Any, np.dtype[np.intp]]:
        """Chunk-local storage coordinates of every point, shape ``(points, k)``."""
        storage = np.stack(
            [
                checked_affine(offset, stride, self.index[:, column])
                for column, (offset, stride) in enumerate(
                    zip(self.offsets, self.strides, strict=True)
                )
            ],
            axis=1,
        )
        return storage - np.repeat(self.chunk_start, self.counts, axis=0)

    def run(self, row: int) -> slice:
        """The slice of the point arrays a row owns."""
        return slice(int(self.pointer[row]), int(self.pointer[row + 1]))


def _strided_set(
    transform: IndexTransform, out_dim: int, m: ConstantMap | DimensionMap, dg: DimensionGridLike
) -> StridedSet:
    domain = transform.domain
    if isinstance(m, ConstantMap):
        c = dg.index_to_chunk(checked_affine(m.offset, 0, 0))
        chunks = np.array([c], dtype=np.intp)
        starts, extents = _chunk_bounds(dg, chunks)
        local = m.offset - starts
        return StridedSet(
            output_dimension=out_dim,
            input_dimension=None,
            stride=0,
            chunk=chunks,
            chunk_start=starts,
            chunk_extent=extents,
            local_start=local,
            extent=np.ones(1, dtype=np.intp),
            origin=np.zeros(1, dtype=np.intp),
            full=(extents == 1) & (local == 0),
        )
    k = m.input_dimension
    lo = domain.inclusive_min[k]
    hi = domain.exclusive_max[k]
    stride = m.stride
    unit = abs(stride) == 1
    rows: list[tuple[int, int, int, int, int, int, bool]] = []
    # Exact Python-int arithmetic per touched chunk: the values can exceed
    # np.intp before cancellation (a large-origin domain), and the number of
    # touched chunks along one axis is a sum, not a product.
    for (c,) in _dimension_map_candidates(m, lo, hi, dg):
        c_start = dg.chunk_offset(c)
        c_extent = _data_size(dg, c)
        narrowed = _intersect_dimension_map(m, lo, hi, c_start, c_start + c_extent)
        if narrowed is None:
            continue
        nlo, nhi = narrowed
        extent = nhi - nlo
        # Chunk-local, then re-based to a zero-origin input axis.
        local_start = m.offset - c_start + stride * nlo
        if unit:
            last = local_start + stride * (extent - 1)
            full = min(local_start, last) == 0 and max(local_start, last) == c_extent - 1
        else:
            full = False
        rows.append((c, c_start, c_extent, local_start, extent, nlo, full))
    columns = list(zip(*rows, strict=True)) if rows else [()] * 7
    return StridedSet(
        output_dimension=out_dim,
        input_dimension=k,
        stride=stride,
        chunk=_int_column(columns[0]),
        chunk_start=_int_column(columns[1]),
        chunk_extent=_int_column(columns[2]),
        local_start=_int_column(columns[3]),
        extent=_int_column(columns[4]),
        origin=_int_column(columns[5]),
        full=np.array(columns[6], dtype=np.bool_),
    )


def _indexed_set(out_dim: int, m: ArrayMap, dg: DimensionGridLike) -> IndexedSet:
    dependent = m.dependent_axis
    assert dependent is not None
    flat = m.index_array.reshape(-1)
    n = int(flat.size)
    storage = checked_affine(m.offset, m.stride, flat)
    # Probe the extreme coordinates with the scalar lookup first: a grid's
    # scalar error names the offending coordinate, where the vectorized one
    # reports a range.
    dg.index_to_chunk(int(storage.min()))
    dg.index_to_chunk(int(storage.max()))
    chunk_ids = dg.indices_to_chunks(storage)
    # Already-sorted coordinates (the common case) need no sort.
    if bool((chunk_ids[1:] >= chunk_ids[:-1]).all()):
        positions = np.arange(n, dtype=np.intp)
        sorted_ids = chunk_ids
        index = flat
    else:
        positions = np.argsort(chunk_ids, kind="stable")
        sorted_ids = chunk_ids[positions]
        index = flat[positions]
    boundaries = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
    pointer = np.concatenate([[0], boundaries, [n]]).astype(np.intp)
    chunks = np.asarray(sorted_ids[pointer[:-1]], dtype=np.intp)
    starts, extents = _chunk_bounds(dg, chunks)
    return IndexedSet(
        output_dimension=out_dim,
        input_dimension=dependent,
        offset=m.offset,
        stride=m.stride,
        chunk=chunks,
        chunk_start=starts,
        chunk_extent=extents,
        pointer=pointer,
        index=np.asarray(index, dtype=np.intp),
        positions=np.asarray(positions, dtype=np.intp),
    )


def _chunk_keys(
    chunk_ids: Sequence[np.ndarray[Any, np.dtype[np.intp]]],
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """One sortable key per point whose order is the lexicographic chunk order."""
    n = int(chunk_ids[0].size)
    if len(chunk_ids) == 1:
        return np.asarray(chunk_ids[0], dtype=np.intp)
    # Mixed-radix key with the first dimension most significant.
    keys = np.zeros(n, dtype=np.intp)
    multiplier = 1
    for ids in reversed(chunk_ids):
        radix = int(ids.max()) + 1
        if multiplier * radix >= 2**62:
            stacked = np.stack([np.asarray(i, dtype=np.intp).ravel() for i in chunk_ids], axis=1)
            _, inverse = np.unique(stacked, axis=0, return_inverse=True)
            return np.asarray(inverse, dtype=np.intp).reshape(-1)
        keys += np.asarray(ids, dtype=np.intp) * multiplier
        multiplier *= radix
    return keys


def _joint_set(transform: IndexTransform, dim_grids: Sequence[DimensionGridLike]) -> JointSet:
    block = _prepare_correlated(transform)
    dims = block.correlated_dims
    chunk_ids = [dim_grids[d].indices_to_chunks(block.flat_storage[d]) for d in dims]
    n_points = int(chunk_ids[0].size)
    keys = _chunk_keys(chunk_ids)
    positions = np.argsort(keys, kind="stable")
    sorted_keys = keys[positions]
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    pointer = np.concatenate([[0], boundaries, [n_points]]).astype(np.intp)
    first = positions[pointer[:-1]]
    chunk = np.stack([np.asarray(ids, dtype=np.intp)[first] for ids in chunk_ids], axis=1)
    index = np.stack([block.flat_index[d][positions] for d in dims], axis=1)
    if len(block.broadcast_shape) > 0:
        block_coordinates = np.stack(
            np.unravel_index(positions, block.broadcast_shape), axis=1
        ).astype(np.intp)
    else:
        block_coordinates = np.empty((n_points, 0), dtype=np.intp)
    starts = np.empty_like(chunk)
    extents = np.empty_like(chunk)
    for column, d in enumerate(dims):
        starts[:, column], extents[:, column] = _chunk_bounds(dim_grids[d], chunk[:, column])
    maps = [cast("ArrayMap", transform.output[d]) for d in dims]
    return JointSet(
        output_dimensions=dims,
        offsets=tuple(m.offset for m in maps),
        strides=tuple(m.stride for m in maps),
        broadcast_axes=block.broadcast_axes,
        broadcast_shape=block.broadcast_shape,
        chunk=chunk,
        chunk_start=starts,
        chunk_extent=extents,
        pointer=pointer,
        index=index,
        positions=np.asarray(positions, dtype=np.intp),
        block_coordinates=block_coordinates,
    )


# One table row's contribution to a projection: (chunk index, chunk start,
# chunk data extent, bound input axis or None, restricted extent of that axis,
# the rebased chunk-local map, the cell map for the axis, whole-chunk cover).
_AxisPiece = tuple[int, int, int, int | None, int, OutputIndexMap, OutputIndexMap | None, bool]


def _slot(slot_of: dict[int, int], axis: StridedSet | IndexedSet) -> int | None:
    """The restricted-domain axis a residual table's input axis maps to, if it has one."""
    if axis.input_dimension is None:
        return None
    return slot_of.get(axis.input_dimension)


def _strided_piece(axis: StridedSet, row: int, input_dimension: int | None = None) -> _AxisPiece:
    return (
        int(axis.chunk[row]),
        int(axis.chunk_start[row]),
        int(axis.chunk_extent[row]),
        axis.input_dimension,
        int(axis.extent[row]),
        axis.chunk_map(row, input_dimension),
        axis.cell_map(row),
        bool(axis.full[row]),
    )


def _indexed_piece(axis: IndexedSet, row: int, rank: int, origin: int) -> _AxisPiece:
    run = axis.run(row)
    k = axis.input_dimension
    count = run.stop - run.start
    shape = (1,) * k + (count,) + (1,) * (rank - k - 1)
    c_start = int(axis.chunk_start[row])
    return (
        int(axis.chunk[row]),
        c_start,
        int(axis.chunk_extent[row]),
        k,
        count,
        ArrayMap(
            index_array=axis.index[run].reshape(shape),
            offset=axis.offset - c_start,
            stride=axis.stride,
        ),
        ArrayMap(index_array=axis.positions[run].reshape(shape), offset=origin),
        False,
    )


@dataclass(frozen=True, slots=True)
class GridPartition:
    """A plan in factored form: per-axis tables whose product is the chunk walk.

    `sets` holds one `StridedSet` or `IndexedSet` per output dimension the
    transform reads independently, in output-dimension order; `joint` holds
    the correlated index arrays, if any. A projection is one row of each
    table, so the partition has ``prod(row_shape)`` rows, walked in row-major
    order over `row_shape` (the joint table last). Rows are materialized into
    `ChunkProjection` objects only on request; a vectorized consumer can read
    the tables directly.

    Build one with `partition_transform`, or take it from `ChunkPlan.partition`.

    Examples
    --------
    `arr[1:6:2, 5:]` on a `(7, 9)` array with `(3, 4)` chunks touches two
    chunks along each axis, so the partition has four rows:

    >>> from zarr_indexing import IndexTransform, plan_chunks
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((3, 4), shape=(7, 9))
    >>> partition = plan_chunks(IndexTransform.from_shape((7, 9))[1:6:2, 5:], grids).partition()
    >>> partition.row_shape, len(partition)
    ((2, 2), 4)
    >>> partition.chunk_coords().tolist()
    [[0, 1], [0, 2], [1, 1], [1, 2]]
    >>> partition[3].chunk_transform.selection_repr
    '{ [0, 4) step 2, [0, 1) }'
    """

    transform: IndexTransform
    """The transform this partition factors."""

    dimension_grids: tuple[DimensionGridLike, ...]
    """One grid per storage dimension."""

    sets: tuple[StridedSet | IndexedSet, ...]
    """Independent per-axis tables, in output-dimension order."""

    joint: JointSet | None
    """The correlated index arrays' table, or `None` when there are none."""

    row_shape: tuple[int, ...]
    """Rows per table, `joint` last; the partition is walked in row-major order over it."""

    def __len__(self) -> int:
        return int(np.prod(self.row_shape, dtype=np.intp)) if self.row_shape else 1

    def chunk_coords(self) -> np.ndarray[Any, np.dtype[np.intp]]:
        """Chunk coordinates of every row, shape ``(len(self), output rank)``, without materializing rows."""
        n_rows = len(self)
        out = np.empty((n_rows, self.transform.output_rank), dtype=np.intp)
        if n_rows == 0 or not self.row_shape:
            return out
        indices = np.unravel_index(np.arange(n_rows, dtype=np.intp), self.row_shape)
        for axis, table_rows in zip(self.sets, indices, strict=False):
            out[:, axis.output_dimension] = axis.chunk[table_rows]
        if self.joint is not None:
            joint_rows = indices[len(self.sets)]
            out[:, list(self.joint.output_dimensions)] = self.joint.chunk[joint_rows]
        return out

    def __getitem__(self, row: int) -> ChunkProjection:
        """Materialize one row (a negative index counts from the end)."""
        n_rows = len(self)
        if row < 0:
            row += n_rows
        if not 0 <= row < n_rows:
            raise IndexError(f"row {row} is out of range for a partition of {n_rows} rows")
        indices = tuple(int(i) for i in np.unravel_index(row, self.row_shape))
        return self._materialize(indices)

    def __iter__(self) -> Iterator[ChunkProjection]:
        """Materialize every row in order."""
        if len(self) == 0:
            return
        if self.joint is None:
            yield from self._iter_factorized()
        else:
            yield from self._iter_correlated()

    # -- factorized assembly ------------------------------------------------

    def _pieces(self, axis: StridedSet | IndexedSet, row: int) -> _AxisPiece:
        if isinstance(axis, StridedSet):
            return _strided_piece(axis, row)
        domain = self.transform.domain
        return _indexed_piece(axis, row, domain.ndim, domain.inclusive_min[axis.input_dimension])

    def _materialize(self, indices: tuple[int, ...]) -> ChunkProjection:
        if self.joint is None:
            pieces = [self._pieces(axis, row) for axis, row in zip(self.sets, indices, strict=True)]
            return self._factorized_projection(pieces, *self._unbound())
        n_lead, slot_of = self._slots()
        residual = [
            _strided_piece(cast("StridedSet", axis), row, _slot(slot_of, axis))
            for axis, row in zip(self.sets, indices[:-1], strict=True)
        ]
        return self._correlated_projection(residual, indices[-1], n_lead, slot_of)

    def _unbound(self) -> tuple[list[OutputIndexMap | None], bool]:
        """Cell maps for request axes no output reads, and whether they permit full coverage.

        A whole-chunk cover must biject onto the chunk, so an unread axis can
        only be a singleton.
        """
        domain = self.transform.domain
        bound = {axis.input_dimension for axis in self.sets if axis.input_dimension is not None}
        cell_maps: list[OutputIndexMap | None] = [None] * domain.ndim
        unbound_ok = True
        for axis in range(domain.ndim):
            if axis not in bound:
                cell_maps[axis] = DimensionMap(
                    input_dimension=axis, offset=domain.inclusive_min[axis]
                )
                unbound_ok = unbound_ok and domain.shape[axis] <= 1
        return cell_maps, unbound_ok

    def _iter_factorized(self) -> Iterator[ChunkProjection]:
        pieces_per_set = [
            [self._pieces(axis, row) for row in range(len(axis))] for axis in self.sets
        ]
        base_cell_maps, unbound_ok = self._unbound()
        for combo in itertools.product(*pieces_per_set):
            yield self._factorized_projection(list(combo), base_cell_maps, unbound_ok)

    def _factorized_projection(
        self,
        pieces: list[_AxisPiece],
        base_cell_maps: list[OutputIndexMap | None],
        unbound_ok: bool,
    ) -> ChunkProjection:
        domain = self.transform.domain
        shape = list(domain.shape)
        cell_maps = list(base_cell_maps)
        chunk_maps: list[OutputIndexMap] = []
        chunk_coords: list[int] = []
        chunk_min: list[int] = []
        chunk_max: list[int] = []
        has_array = False
        full = unbound_ok
        for c, c_start, c_extent, k, extent, chunk_map, cell_map, piece_full in pieces:
            chunk_coords.append(c)
            chunk_min.append(c_start)
            chunk_max.append(c_start + c_extent)
            chunk_maps.append(chunk_map)
            if isinstance(chunk_map, ArrayMap):
                has_array = True
            if k is not None:
                shape[k] = extent
                cell_maps[k] = cell_map
            full = full and piece_full
        synthetic = IndexDomain._unchecked((0,) * domain.ndim, tuple(shape))  # pyright: ignore[reportPrivateUsage]
        if has_array:
            coverage: ChunkCoverage = "unknown"
        elif full:
            coverage = "full"
        else:
            coverage = "partial"
        return ChunkProjection(
            chunk_coords=tuple(chunk_coords),
            chunk_domain=IndexDomain._unchecked(tuple(chunk_min), tuple(chunk_max)),  # pyright: ignore[reportPrivateUsage]
            chunk_transform=IndexTransform._unchecked(synthetic, tuple(chunk_maps)),  # pyright: ignore[reportPrivateUsage]
            cell_transform=IndexTransform._unchecked(  # pyright: ignore[reportPrivateUsage]
                synthetic, tuple(cast("list[OutputIndexMap]", cell_maps))
            ),
            coverage=coverage,
        )

    # -- correlated assembly ------------------------------------------------

    def _slots(self) -> tuple[int, dict[int, int]]:
        """Where each residual slice axis sits in the restricted domain.

        The restricted domain is the collapsed points axis (if the broadcast
        block has any axis) followed by the residual slice axes in
        input-dimension order.
        """
        joint = self.joint
        assert joint is not None
        n_lead = 1 if len(joint.broadcast_shape) > 0 else 0
        slice_axes = sorted(
            m.input_dimension for m in self.transform.output if isinstance(m, DimensionMap)
        )
        return n_lead, {axis: n_lead + slot for slot, axis in enumerate(slice_axes)}

    def _iter_correlated(self) -> Iterator[ChunkProjection]:
        joint = self.joint
        assert joint is not None
        n_lead, slot_of = self._slots()
        pieces_per_set = [
            [
                _strided_piece(cast("StridedSet", axis), row, _slot(slot_of, axis))
                for row in range(len(axis))
            ]
            for axis in self.sets
        ]
        for residual in itertools.product(*pieces_per_set):
            for row in range(len(joint)):
                yield self._correlated_projection(list(residual), row, n_lead, slot_of)

    def _correlated_projection(
        self,
        residual: list[_AxisPiece],
        row: int,
        n_lead: int,
        slot_of: dict[int, int],
    ) -> ChunkProjection:
        joint = self.joint
        assert joint is not None
        transform = self.transform
        domain = transform.domain
        rank = domain.ndim
        lo_all = domain.inclusive_min
        output_rank = transform.output_rank
        n_slice = len(slot_of)
        run = joint.run(row)
        n_points = run.stop - run.start
        points_shape = (n_points,) if n_lead else ()
        corr_shape = points_shape + (1,) * n_slice

        chunk_coords = [0] * output_rank
        chunk_min = [0] * output_rank
        chunk_max = [0] * output_rank
        chunk_maps: list[OutputIndexMap | None] = [None] * output_rank
        extents = [0] * n_slice
        slice_origin = [0] * n_slice
        for out_dim, (c, c_start, c_extent, k, extent, chunk_map, cell_map, _full) in zip(
            (axis.output_dimension for axis in self.sets), residual, strict=True
        ):
            chunk_coords[out_dim] = c
            chunk_min[out_dim] = c_start
            chunk_max[out_dim] = c_start + c_extent
            chunk_maps[out_dim] = chunk_map
            if k is not None:
                slot = slot_of[k] - n_lead
                extents[slot] = extent
                slice_origin[slot] = cast("DimensionMap", cell_map).offset
        for column, out_dim in enumerate(joint.output_dimensions):
            c_start = int(joint.chunk_start[row, column])
            chunk_coords[out_dim] = int(joint.chunk[row, column])
            chunk_min[out_dim] = c_start
            chunk_max[out_dim] = c_start + int(joint.chunk_extent[row, column])
            chunk_maps[out_dim] = ArrayMap(
                index_array=joint.index[run, column].reshape(corr_shape),
                offset=joint.offsets[column] - c_start,
                stride=joint.strides[column],
            )
        shape = points_shape + tuple(extents)
        synthetic = IndexDomain._unchecked((0,) * (n_lead + n_slice), shape)  # pyright: ignore[reportPrivateUsage]

        # One cell map per request axis, materialized over the whole restricted
        # block exactly as unravelling the flat scatter offsets would give.
        cell_maps: list[OutputIndexMap] = []
        for axis in range(rank):
            slot_index = slot_of.get(axis)
            if slot_index is None:
                column = joint.broadcast_axes.index(axis)
                values = joint.block_coordinates[run, column].reshape(corr_shape)
            else:
                slot = slot_index - n_lead
                extent = extents[slot]
                values = (
                    np.arange(extent, dtype=np.intp) + (slice_origin[slot] - lo_all[axis])
                ).reshape((1,) * (n_lead + slot) + (extent,) + (1,) * (n_slice - slot - 1))
            cell_maps.append(
                ArrayMap(index_array=np.broadcast_to(values, shape), offset=lo_all[axis])
            )
        return ChunkProjection(
            chunk_coords=tuple(chunk_coords),
            chunk_domain=IndexDomain._unchecked(tuple(chunk_min), tuple(chunk_max)),  # pyright: ignore[reportPrivateUsage]
            chunk_transform=IndexTransform._unchecked(  # pyright: ignore[reportPrivateUsage]
                synthetic, tuple(cast("list[OutputIndexMap]", chunk_maps))
            ),
            cell_transform=IndexTransform._unchecked(synthetic, tuple(cell_maps)),  # pyright: ignore[reportPrivateUsage]
            coverage="unknown",
        )


def partition_transform(
    transform: IndexTransform,
    dimension_grids: Sequence[DimensionGridLike],
) -> GridPartition:
    """Factor a transform over a chunk grid into per-axis tables.

    Parameters
    ----------
    transform
        Mapping from the request domain to storage coordinates.
    dimension_grids
        One storage grid per transform output dimension.

    Returns
    -------
    GridPartition
        The factored plan; iterate it for `ChunkProjection` rows.

    Raises
    ------
    ValueError
        If two output maps read the same input axis (a diagonal), which has
        no factored form; `plan_chunks` still walks such transforms.

    Examples
    --------
    >>> from zarr_indexing import IndexTransform
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((2, 2), shape=(3, 4))
    >>> partition = partition_transform(IndexTransform.from_shape((3, 4))[1, :], grids)
    >>> len(partition), [p.chunk_coords for p in partition]
    (2, [(0, 0), (0, 1)])
    """
    grids = tuple(dimension_grids)
    if len(grids) != transform.output_rank:
        raise ValueError(
            "dimension_grids must have one entry per transform output dimension; "
            f"got {len(grids)} grids for output rank {transform.output_rank}"
        )
    if not _partitionable(transform):
        raise ValueError(
            "the transform has no factored form: two of its output maps read the same "
            "input axis; iterate plan_chunks(...) instead"
        )
    if any(size == 0 for size in transform.domain.shape):
        return GridPartition(
            transform=transform, dimension_grids=grids, sets=(), joint=None, row_shape=(0,)
        )
    correlated = transform.index_array_structure == "general"
    sets: list[StridedSet | IndexedSet] = []
    for out_dim, m in enumerate(transform.output):
        if isinstance(m, ArrayMap):
            if correlated:
                continue
            sets.append(_indexed_set(out_dim, m, grids[out_dim]))
        else:
            sets.append(_strided_set(transform, out_dim, m, grids[out_dim]))
    joint = _joint_set(transform, grids) if correlated else None
    row_shape = tuple(len(axis) for axis in sets) + ((len(joint),) if joint is not None else ())
    if any(rows == 0 for rows in row_shape):
        row_shape = (0,)
    return GridPartition(
        transform=transform,
        dimension_grids=grids,
        sets=tuple(sets),
        joint=joint,
        row_shape=row_shape,
    )
