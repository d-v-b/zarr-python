from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from zarr_indexing import DimensionMap, IndexDomain, IndexTransform
from zarr_indexing._execution import execute_selection, execute_transform
from zarr_indexing.grid import dimension_grids_from_chunks


@pytest.mark.parametrize(
    ("shape", "chunks", "selection", "mode"),
    [
        ((7, 9), (3, 4), (slice(None), slice(None)), "basic"),
        ((7, 9), (3, 4), (slice(1, 7, 2), 3), "basic"),
        ((7, 9), (3, 4), (slice(None, None, -1), slice(1, 8, 2)), "basic"),
        ((7, 9), ((2, 5), (4, 5)), (slice(2, 7), slice(None)), "basic"),
        ((7, 9), (3, 4), (slice(2, 2), slice(None)), "basic"),
        ((7, 9), (3, 4), (None, Ellipsis), "basic"),
        ((), (), (), "basic"),
        ((1000,), (100,), (np.repeat(np.arange(1000), 2),), "vectorized"),
        ((7,), (3,), (np.array([6, 0, 6, 2]),), "vectorized"),
        ((7,), (3,), (np.array([], dtype=np.intp),), "vectorized"),
        ((1000,), (100,), (np.r_[np.arange(100), np.arange(900, 1000)],), "vectorized"),
        (
            (7, 9, 5),
            (3, 4, 2),
            (np.array([6, 0])[:, None], np.array([8, 2])[:, None], np.array([4, 0, 2])[None, :]),
            "vectorized",
        ),
    ],
)
def test_execution_reads_and_writes(
    shape: tuple[int, ...], chunks: tuple[Any, ...], selection: Any, mode: str
) -> None:
    source = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    grids = dimension_grids_from_chunks(chunks, shape)
    base = IndexTransform.from_shape(shape)
    transform = base[selection] if mode == "basic" else base.vindex[selection]
    expected = source[selection]
    for execution in (
        execute_selection(selection, shape, grids, mode=mode),
        execute_transform(transform, grids),
    ):
        result = np.empty(execution.shape, dtype=source.dtype)
        written = source.copy()
        replacements = np.arange(expected.size).reshape(expected.shape) + 10000
        for row in execution:
            bounds = tuple(
                slice(g.chunk_offset(c), g.chunk_offset(c) + g.data_size(c))
                for g, c in zip(grids, row.chunk_coords, strict=True)
            )
            result[row.out_selection] = source[bounds][row.chunk_selection]
            target = written[bounds] if bounds else written
            target[row.chunk_selection] = replacements[row.out_selection]
        np.testing.assert_array_equal(result, expected)
        expected_write = source.copy()
        expected_write[selection] = replacements
        np.testing.assert_array_equal(written, expected_write)
        assert [r.chunk_coords for r in execution] == [r.chunk_coords for r in execution]


def test_boundary_chunk_has_complete_data_extent() -> None:
    grids = dimension_grids_from_chunks((3,), (7,))
    rows = list(execute_selection(slice(None), (7,), grids))
    assert [row.is_complete_chunk for row in rows] == [True, True, True]


def test_execution_rejects_grid_rank_mismatch() -> None:
    with pytest.raises(ValueError, match="one entry"):
        execute_selection(slice(None), (7,), ())


def test_execution_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown indexing mode"):
        execute_selection(
            slice(None), (7,), dimension_grids_from_chunks((3,), (7,)), mode="invalid"
        )


def test_execution_rejects_negative_shape() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        execute_selection(slice(None), (-1,), dimension_grids_from_chunks((3,), (7,)))


def test_execution_rejects_out_of_bounds_integer() -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        execute_selection(7, (7,), dimension_grids_from_chunks((3,), (7,)))


def test_execution_sorted_coordinates_check_source_bounds() -> None:
    with pytest.raises(IndexError):
        execute_selection(
            np.arange(1000), (100,), dimension_grids_from_chunks((100,), (1000,)), mode="vectorized"
        )


def test_execution_rejects_zero_slice_step() -> None:
    with pytest.raises(IndexError, match="step must not be zero"):
        execute_selection(slice(None, None, 0), (7,), dimension_grids_from_chunks((3,), (7,)))


def test_execution_validates_grid_bounds_before_returning_work() -> None:
    with pytest.raises(IndexError):
        execute_selection(slice(None), (100,), dimension_grids_from_chunks((10,), (50,)))


def test_execution_rejects_repeated_unread_input_axes() -> None:
    transform = IndexTransform(IndexDomain.from_shape((2, 3)), (DimensionMap(0),))
    with pytest.raises(ValueError, match="duplicate writes"):
        execute_transform(transform, dimension_grids_from_chunks((2,), (2,)), access="write")


def test_declarative_execution_retains_snapshot() -> None:
    coordinates = np.arange(1000)
    transform = IndexTransform.from_shape((1000,)).vindex[coordinates]
    plan = execute_transform(transform, dimension_grids_from_chunks((100,), (1000,)))
    coordinates[:] = 0
    first = next(iter(plan))
    np.testing.assert_array_equal(first.chunk_selection[0], np.arange(100))


def test_immediate_execution_accepts_readonly_sorted_coordinates() -> None:
    coordinates = np.arange(1000)
    coordinates.setflags(write=False)
    plan = execute_selection(
        coordinates, (1000,), dimension_grids_from_chunks((100,), (1000,)), mode="vectorized"
    )
    assert len(list(plan)) == 10
    np.testing.assert_array_equal(coordinates, np.arange(1000))


@pytest.mark.parametrize(
    ("offset", "stride", "values", "extent"),
    [
        (2**63, -1, [2**63 - 1, 2**63 - 2], 3),
        (-(2**63) + 1, 2, [2**62, 2**62 + 2], 6),
    ],
)
def test_lowering_preserves_exact_affine_cancellation(
    offset: int, stride: int, values: list[int], extent: int
) -> None:
    from zarr_indexing import ArrayMap

    transform = IndexTransform(
        IndexDomain.from_shape((2,)),
        (ArrayMap(np.array(values, dtype=np.intp), offset=offset, stride=stride),),
    )
    plan = execute_transform(
        transform, dimension_grids_from_chunks((3,), (extent,)), access="write"
    )
    result = np.zeros(extent, dtype=np.int64)
    for row in plan:
        result[row.chunk_coords[0] * 3 : row.chunk_coords[0] * 3 + 3][row.chunk_selection] = (
            np.array([10, 20])[row.out_selection]
        )
    assert result[transform.apply((0,))[0]] == 10
    assert result[transform.apply((1,))[0]] == 20


@pytest.mark.parametrize("ownership", ["snapshot", "borrow"])
def test_explicit_ownership(ownership: Any) -> None:
    coordinates = np.arange(1000)
    plan = execute_selection(
        coordinates,
        (1000,),
        dimension_grids_from_chunks((100,), (1000,)),
        mode="vectorized",
        ownership=ownership,
    )
    assert plan.ownership == ownership
    if ownership == "snapshot":
        coordinates[:] = 0
    np.testing.assert_array_equal(next(iter(plan)).chunk_selection[0], np.arange(100))


@pytest.mark.parametrize(("access", "conflicts"), [("read", "error"), ("write", "last")])
def test_repeated_unread_axes_have_explicit_access_policy(access: Any, conflicts: Any) -> None:
    transform = IndexTransform(IndexDomain.from_shape((2, 3)), (DimensionMap(0),))
    plan = execute_transform(
        transform, dimension_grids_from_chunks((2,), (2,)), access=access, conflicts=conflicts
    )
    source = np.array([10, 20])
    for consumer in ("numpy", "shard"):
        result = np.empty(plan.shape, dtype=np.int64)
        written = source.copy()
        for op in plan.lower(consumer).operations():
            row = op.row
            result[row.out_selection] = source[row.chunk_selection]
            if access == "write":
                written[row.chunk_selection] = np.arange(6).reshape(2, 3)[row.out_selection]
        if access == "read":
            np.testing.assert_array_equal(result, [[10, 10, 10], [20, 20, 20]])
        else:
            np.testing.assert_array_equal(written, [2, 5])


def test_write_rejects_duplicate_sorted_coordinates() -> None:
    with pytest.raises(ValueError, match="duplicate writes"):
        execute_selection(
            np.repeat(np.arange(1000), 2),
            (1000,),
            dimension_grids_from_chunks((100,), (1000,)),
            mode="vectorized",
            access="write",
        )


def test_first_basic_result_keeps_large_axes_implicit() -> None:
    import tracemalloc

    plan = execute_selection(Ellipsis, (10**9, 2), dimension_grids_from_chunks((1, 1), (10**9, 2)))
    tracemalloc.start()
    first = next(iter(plan))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert first.chunk_coords == (0, 0)
    assert peak < 100_000


@pytest.mark.parametrize("consumer", ["numpy", "shard"])
def test_scalar_coordinate_consumer_preserves_value_shape(consumer: Any) -> None:
    from zarr_indexing import ArrayMap

    transform = IndexTransform(IndexDomain.from_shape(()), (ArrayMap(np.array(4)),))
    plan = execute_transform(transform, dimension_grids_from_chunks((1,), (8,)))
    (operation,) = plan.lower(consumer).operations()
    result = np.empty((), dtype=np.int64)
    result[operation.row.out_selection] = np.array([4])[operation.row.chunk_selection]
    assert result == 4
    assert operation.value_shape == ()


@pytest.mark.parametrize("access", ["read", "write"])
def test_diagonal_with_independent_gather_is_preparable(access: Any) -> None:
    from zarr_indexing import ArrayMap

    transform = IndexTransform(
        IndexDomain.from_shape((3, 2)),
        (DimensionMap(0), DimensionMap(0), ArrayMap(np.array([[1, 0]]))),
    )
    plan = execute_transform(
        transform, dimension_grids_from_chunks((2, 2, 2), (3, 3, 2)), access=access
    )
    assert len(list(plan)) == 2


def test_last_write_lowering_removes_duplicate_destinations() -> None:
    values = np.repeat(np.arange(1000), 2)
    plan = execute_selection(
        values,
        (1000,),
        dimension_grids_from_chunks((100,), (1000,)),
        mode="vectorized",
        access="write",
        conflicts="last",
    )
    for row in plan:
        destinations = row.chunk_selection[0]
        assert np.unique(destinations).size == np.size(destinations)
        np.testing.assert_array_equal(np.asarray(row.out_selection[0]) % 2, 1)


def test_empty_coordinate_plan_emits_no_io() -> None:
    plan = execute_selection(
        np.array([], dtype=np.intp),
        (5,),
        dimension_grids_from_chunks((2,), (5,)),
        mode="vectorized",
        access="write",
    )
    assert list(plan) == []
    assert list(plan.lower("shard")) == []


def test_execution_rejects_unknown_access() -> None:
    with pytest.raises(ValueError, match="access intent"):
        execute_selection(Ellipsis, (1,), dimension_grids_from_chunks((1,), (1,)), access="bad")  # type: ignore[arg-type]


def test_execution_rejects_unknown_ownership() -> None:
    with pytest.raises(ValueError, match="ownership policy"):
        execute_selection(Ellipsis, (1,), dimension_grids_from_chunks((1,), (1,)), ownership="bad")  # type: ignore[arg-type]


def test_execution_rejects_unknown_conflicts() -> None:
    with pytest.raises(ValueError, match="conflict policy"):
        execute_selection(Ellipsis, (1,), dimension_grids_from_chunks((1,), (1,)), conflicts="bad")  # type: ignore[arg-type]


def test_execution_rejects_unknown_consumer() -> None:
    plan = execute_selection(Ellipsis, (1,), dimension_grids_from_chunks((1,), (1,)))
    with pytest.raises(ValueError, match="consumer"):
        plan.lower("bad")  # type: ignore[arg-type]


def test_orthogonal_scalar_rejects_negative_literal_coordinate() -> None:
    with pytest.raises(IndexError, match="negative scalar"):
        execute_selection(
            (-1, [1, 2]), (4, 4), dimension_grids_from_chunks((2, 2), (4, 4)), mode="orthogonal"
        )
