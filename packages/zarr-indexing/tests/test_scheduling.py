from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from zarr_indexing import DimensionMap, IndexDomain, IndexTransform
from zarr_indexing.grid import dimension_grids_from_chunks
from zarr_indexing.scheduling import plan_rechunk, plan_write_batches


def footprint(transform: IndexTransform, grids: Any) -> set[tuple[int, ...]]:
    return {
        tuple(g.index_to_chunk(c) for g, c in zip(grids, transform.apply(point), strict=True))
        for position in np.ndindex(transform.domain.shape)
        for point in [tuple(x + o for x, o in zip(position, transform.domain.origin, strict=True))]
    }


@pytest.mark.parametrize("order", ["preserve", "reorder"])
@pytest.mark.parametrize(
    "indices", [[], [[]], [[0], [1], [2]], [[0, 0], [0], []], [[0], [0, 1], [1, 2], [2]]]
)
def test_write_batches(indices: list[list[int]], order: Any) -> None:
    transforms = [
        IndexTransform.from_shape((3,)).vindex[np.array(x, dtype=np.intp)] for x in indices
    ]
    grids = dimension_grids_from_chunks((1,), (3,))
    schedule = plan_write_batches(transforms, grids, order=order)
    batches = schedule.batches
    assert sorted(i for batch in batches for i in batch) == list(range(len(indices)))
    membership = {task: b for b, batch in enumerate(batches) for task in batch}
    for batch in batches:
        occupied: set[tuple[int, ...]] = set()
        for task in batch:
            units = footprint(transforms[task], grids)
            assert occupied.isdisjoint(units)
            occupied |= units
    if order == "preserve":
        for i, t in enumerate(transforms):
            for j in range(i + 1, len(transforms)):
                if footprint(t, grids) & footprint(transforms[j], grids):
                    assert membership[i] < membership[j]
    assert schedule == plan_write_batches(transforms, grids, order=order)


def test_rechunk_example() -> None:
    domain = IndexDomain.from_shape((12,))
    plan = plan_rechunk(
        domain, dimension_grids_from_chunks((3,), (12,)), dimension_grids_from_chunks((4,), (12,))
    )
    assert plan.schedule.batches == ((0, 2), (1, 3))
    assert [p.source_chunk for p in plan.pieces] == [(0,), (1,), (2,), (3,)]
    assert all(p.source_selection == (slice(0, 3),) for p in plan.pieces)


@given(
    shape=st.lists(st.integers(1, 9), min_size=1, max_size=3),
    source_size=st.integers(1, 7),
    target_size=st.integers(1, 7),
)
def test_rechunk_survives_simultaneous_read_modify_write(
    shape: list[int], source_size: int, target_size: int
) -> None:
    shape_tuple = tuple(shape)
    source_grids = dimension_grids_from_chunks((source_size,) * len(shape), shape_tuple)
    target_grids = dimension_grids_from_chunks((target_size,) * len(shape), shape_tuple)
    plan = plan_rechunk(IndexDomain.from_shape(shape_tuple), source_grids, target_grids)
    source = np.arange(np.prod(shape_tuple)).reshape(shape_tuple)
    result = np.full(shape_tuple, -1)
    covered = np.zeros(shape_tuple, dtype=int)
    assert sorted(i for batch in plan.schedule.batches for i in batch) == list(
        range(len(plan.pieces))
    )
    for batch in plan.schedule.batches:
        pending = []
        occupied: set[tuple[int, ...]] = set()
        # All tasks take their RMW snapshots before any task commits.
        for task in batch:
            piece = plan.pieces[task]
            units = footprint(piece.destination, target_grids)
            assert occupied.isdisjoint(units)
            occupied |= units
            bounds = tuple(
                slice(lo, hi)
                for lo, hi in zip(
                    piece.destination.domain.origin,
                    piece.destination.domain.exclusive_max,
                    strict=True,
                )
            )
            covered[bounds] += 1
            snapshot = result.copy()
            snapshot[bounds] = source[bounds]
            for unit in units:
                chunk_bounds = tuple(
                    slice(g.chunk_offset(c), g.chunk_offset(c) + g.data_size(c))
                    for g, c in zip(target_grids, unit, strict=True)
                )
                pending.append((chunk_bounds, snapshot[chunk_bounds].copy()))
        for bounds, data in reversed(pending):
            result[bounds] = data
    np.testing.assert_array_equal(result, source)
    np.testing.assert_array_equal(covered, np.ones(shape_tuple))


@pytest.mark.parametrize("shape", [(), (0,), (3, 0)])
def test_empty_and_scalar_rechunk(shape: tuple[int, ...]) -> None:
    grids = dimension_grids_from_chunks((2,) * len(shape), shape)
    plan = plan_rechunk(IndexDomain.from_shape(shape), grids, grids)
    assert plan.schedule.batches == (((0,),) if shape == () else ())


def test_irregular_subdomain() -> None:
    source = dimension_grids_from_chunks(((2, 5, 3),), (10,))
    target = dimension_grids_from_chunks(((4, 1, 5),), (10,))
    plan = plan_rechunk(IndexDomain((1,), (9,)), source, target)
    assert [p.source_selection for p in plan.pieces] == [
        (slice(1, 2),),
        (slice(0, 5),),
        (slice(0, 2),),
    ]
    assert plan.schedule.batches == ((0, 2), (1,))


def test_order_preserves_last_write() -> None:
    base = IndexTransform.from_shape((3,))
    writes = [base[0:1], base[0:2], base[1:3], base[2:3]]
    schedule = plan_write_batches(writes, dimension_grids_from_chunks((1,), (3,)))
    assert schedule.batches == ((0,), (1,), (2,), (3,))


def test_unsupported_affine_diagonal() -> None:
    domain = IndexDomain.from_shape((3,))
    diagonal = IndexTransform(domain, (DimensionMap(0), DimensionMap(0)))
    point = IndexTransform.from_shape((3, 3))[1, 1]
    grids = dimension_grids_from_chunks((1, 1), (3, 3))
    with pytest.raises(ValueError, match="input axis"):
        plan_write_batches([diagonal, point], grids)


def test_invalid_order() -> None:
    with pytest.raises(ValueError, match="order"):
        plan_write_batches([], (), order="invalid")


def test_rank_mismatch() -> None:
    with pytest.raises(ValueError, match="rank"):
        plan_write_batches([IndexTransform.from_shape((2,))], ())


def test_out_of_bounds() -> None:
    with pytest.raises(IndexError):
        plan_write_batches(
            [IndexTransform.from_shape((3,))], dimension_grids_from_chunks((2,), (2,))
        )


def test_rechunk_source_rank_mismatch() -> None:
    with pytest.raises(ValueError, match="rank"):
        plan_rechunk(IndexDomain.from_shape((2,)), (), dimension_grids_from_chunks((2,), (2,)))


def test_rechunk_destination_rank_mismatch_even_when_empty() -> None:
    with pytest.raises(ValueError, match="rank"):
        plan_rechunk(IndexDomain.from_shape((0,)), dimension_grids_from_chunks((2,), (0,)), ())


@given(st.lists(st.lists(st.integers(0, 7), max_size=12), max_size=25))
def test_reorder_matches_reference_first_fit(indices: list[list[int]]) -> None:
    writes = [IndexTransform.from_shape((8,)).vindex[np.array(x, dtype=np.intp)] for x in indices]
    expected: list[list[int]] = []
    occupied: list[set[int]] = []
    for task, values in enumerate(indices):
        units = set(values)
        batch = next(
            (i for i, used in enumerate(occupied) if units.isdisjoint(used)), len(occupied)
        )
        if batch == len(occupied):
            expected.append([])
            occupied.append(set())
        expected[batch].append(task)
        occupied[batch].update(units)
    schedule = plan_write_batches(writes, dimension_grids_from_chunks((1,), (8,)), order="reorder")
    assert schedule.batches == tuple(tuple(batch) for batch in expected)
    assert schedule.n_memberships == sum(len(set(x)) for x in indices)
    assert schedule.n_write_units == len({x for row in indices for x in row})


def test_unsupported_mixed_transform() -> None:
    from zarr_indexing import ArrayMap

    transform = IndexTransform(
        IndexDomain.from_shape((2,)), (DimensionMap(0), ArrayMap(np.array([1, 0])))
    )
    with pytest.raises(ValueError, match="input axis"):
        plan_write_batches([transform], dimension_grids_from_chunks((1, 1), (2, 2)))


@given(
    requests=st.lists(
        st.lists(st.tuples(st.integers(0, 4), st.integers(0, 6)), max_size=15),
        max_size=15,
    ),
    sizes=st.tuples(st.integers(1, 5), st.integers(1, 7)),
)
def test_coordinate_schedules_match_enumerated_footprints(
    requests: list[list[tuple[int, int]]], sizes: tuple[int, int]
) -> None:
    """Paired multidimensional gathers schedule exactly their enumerated units."""
    grids = dimension_grids_from_chunks(sizes, (5, 7))
    base = IndexTransform.from_shape((5, 7))
    writes = []
    footprints = []
    for points in requests:
        coordinates = np.array(points, dtype=np.intp).reshape(-1, 2)
        writes.append(base.vindex[coordinates[:, 0], coordinates[:, 1]])
        footprints.append({(x // sizes[0], y // sizes[1]) for x, y in points})
    for order in ("preserve", "reorder"):
        schedule = plan_write_batches(writes, grids, order=order)
        assignments = {task: b for b, batch in enumerate(schedule.batches) for task in batch}
        assert sorted(assignments) == list(range(len(requests)))
        for i, units in enumerate(footprints):
            for j in range(i):
                if units & footprints[j]:
                    assert assignments[i] != assignments[j]
                    if order == "preserve":
                        assert assignments[j] < assignments[i]
        assert schedule.n_memberships == sum(map(len, footprints))
        assert schedule.n_write_units == len(set().union(*footprints))
