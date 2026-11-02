"""Contract tests for the private partition indexer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from zarr_indexing import DimensionMap, IndexDomain, IndexTransform, plan_chunks
from zarr_indexing._indexer import PartitionIndexer
from zarr_indexing.grid import dimension_grids_from_chunks
from zarr_indexing.testing import apply_selection


def _bounds(grids: tuple[Any, ...], coords: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(
        slice(g.chunk_offset(c), g.chunk_offset(c) + g.data_size(c))
        for g, c in zip(grids, coords, strict=True)
    )


def _assemble(
    indexer: PartitionIndexer, source: np.ndarray[Any, Any], grids: tuple[Any, ...], values: Any
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Read through the rows into a fresh result and scatter `values` into a copy of `source`."""
    result = np.full(indexer.shape, -1, dtype=source.dtype)
    written = source.copy()
    for row in indexer:
        bounds = _bounds(grids, row.chunk_coords)
        chunk = source[bounds] if bounds else source
        target = written[bounds] if bounds else written
        result[row.out_selection] = chunk[row.chunk_selection]
        target[row.chunk_selection] = values[row.out_selection]
    return result, written


CASES = [
    ((7, 9), (3, 4), (slice(None), slice(None)), "basic"),
    ((7, 9), (3, 4), (slice(1, 7, 2), 3), "basic"),
    ((7, 9), (3, 4), (slice(None, None, -1), slice(1, 8, 2)), "basic"),
    ((7, 9), (3, 4), (-1, slice(-5, None)), "basic"),
    ((7, 9), ((2, 5), (4, 5)), (slice(2, 7), slice(None)), "basic"),
    ((7, 9), (3, 4), (slice(2, 2), slice(None)), "basic"),
    ((7, 9), (3, 4), (None, Ellipsis), "basic"),
    ((7, 9), (3, 4), (slice(None), None, 2), "basic"),
    ((), (), (), "basic"),
    ((7, 9), (3, 4), (np.array([6, 0, 6]), slice(1, 8, 2)), "orthogonal"),
    ((7, 9), (3, 4), (np.array([6, 0]), np.array([8, 2, 2])), "orthogonal"),
    ((7, 9), (3, 4), (2, np.array([8, 2])), "orthogonal"),
    ((7, 9, 5), (3, 4, 2), (np.array([1, 5]), slice(None), np.array([4, 0])), "orthogonal"),
    ((7, 9, 5), (3, 4, 2), (slice(None, None, -1), np.array([1, 5]), 2), "orthogonal"),
    ((1000,), (100,), (np.repeat(np.arange(1000), 2),), "vectorized"),
    ((7,), (3,), (np.array([6, 0, 6, 2]),), "vectorized"),
    ((7,), (3,), (np.array([], dtype=np.intp),), "vectorized"),
    ((7, 9), (3, 4), (np.array([6, 0, 6]), np.array([8, 2, 8])), "vectorized"),
    ((7, 9), (3, 4), (2, 3), "vectorized"),
    (
        (7, 9, 5),
        (3, 4, 2),
        (np.array([6, 0])[:, None], np.array([8, 2])[:, None], np.array([4, 0, 2])[None, :]),
        "vectorized",
    ),
    ((7, 9, 5), (3, 4, 2), (np.array([6, 0]), 3, np.array([4, 0])), "vectorized"),
]


@pytest.mark.parametrize(("shape", "chunks", "selection", "mode"), CASES)
def test_rows_assemble_reads_and_scatter_writes(
    shape: tuple[int, ...], chunks: tuple[Any, ...], selection: Any, mode: Any
) -> None:
    """Assembling the rows reproduces NumPy, from both constructors, on every walk."""
    source = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    grids = dimension_grids_from_chunks(chunks, shape)
    expected = np.asarray(apply_selection(source, selection, mode))
    values = np.arange(expected.size, dtype=np.int64).reshape(expected.shape) + 10000
    indexer = PartitionIndexer.from_selection(selection, shape, grids, mode=mode)
    assert indexer.shape == expected.shape
    assert indexer.drop_axes == ()
    for candidate in (indexer, PartitionIndexer.from_transform(indexer.partition.transform, grids)):
        result, written = _assemble(candidate, source, grids, values)
        np.testing.assert_array_equal(result, expected)
        if mode != "orthogonal":
            expected_write = source.copy()
            expected_write[selection] = values
            np.testing.assert_array_equal(written, expected_write)
        assert [r.chunk_coords for r in candidate] == [r.chunk_coords for r in candidate]


def test_rows_follow_partition_order() -> None:
    grids = dimension_grids_from_chunks((3, 4), (7, 9))
    selection = (np.array([6, 0]), slice(None))
    indexer = PartitionIndexer.from_selection(selection, (7, 9), grids, mode="orthogonal")
    assert [list(r.chunk_coords) for r in indexer] == indexer.partition.chunk_coords().tolist()


def test_full_chunks_are_complete_including_reversed_ones() -> None:
    grids = dimension_grids_from_chunks((3,), (7,))
    forward = list(PartitionIndexer.from_selection(slice(None), (7,), grids))
    assert [row.is_complete_chunk for row in forward] == [True, True, True]
    reverse = list(PartitionIndexer.from_selection(slice(None, None, -1), (7,), grids))
    assert [row.is_complete_chunk for row in reverse] == [True, True, True]
    # Storage is read ascending; the reversal lives in the out selection.
    assert [row.chunk_selection for row in reverse] == [
        (slice(0, 3, 1),),
        (slice(0, 3, 1),),
        (slice(0, 1, 1),),
    ]
    assert [row.out_selection for row in reverse] == [
        (slice(6, 3, -1),),
        (slice(3, 0, -1),),
        (slice(0, None, -1),),
    ]


def test_partial_and_strided_rows_are_not_complete() -> None:
    grids = dimension_grids_from_chunks((3,), (7,))
    partial = PartitionIndexer.from_selection(slice(1, 7), (7,), grids)
    assert [r.is_complete_chunk for r in partial] == [False, True, True]
    # A strided row is partial, unless the chunk holds one data cell selected once.
    strided = PartitionIndexer.from_selection(slice(None, None, 2), (7,), grids)
    assert [r.is_complete_chunk for r in strided] == [False, False, True]
    gather = PartitionIndexer.from_selection(np.arange(7), (7,), grids, mode="vectorized")
    assert not any(r.is_complete_chunk for r in gather)


def test_transposed_transform_assembles_through_coordinates() -> None:
    """A permuted affine view cannot use slices on both sides; it still assembles."""
    source = np.arange(12, dtype=np.int64).reshape(4, 3)
    grids = dimension_grids_from_chunks((2, 2), (4, 3))
    transform = IndexTransform(IndexDomain.from_shape((3, 4)), (DimensionMap(1), DimensionMap(0)))
    indexer = PartitionIndexer.from_transform(transform, grids)
    values = np.arange(12, dtype=np.int64).reshape(3, 4) + 100
    result, written = _assemble(indexer, source, grids, values)
    np.testing.assert_array_equal(result, source.T)
    np.testing.assert_array_equal(written, values.T)
    assert not any(row.is_complete_chunk for row in indexer)


def test_stride_zero_map_replicates_one_coordinate() -> None:
    grids = dimension_grids_from_chunks((2,), (4,))
    transform = IndexTransform(IndexDomain.from_shape((3,)), (DimensionMap(0, offset=1, stride=0),))
    indexer = PartitionIndexer.from_transform(transform, grids)
    source = np.array([10, 20, 30, 40])
    values = np.array([7, 8, 9])
    result, written = _assemble(indexer, source, grids, values)
    np.testing.assert_array_equal(result, [20, 20, 20])
    np.testing.assert_array_equal(written[[0, 2, 3]], [10, 30, 40])
    assert written[1] in values


def test_indexer_does_not_follow_later_mutation_of_its_input() -> None:
    coordinates = np.arange(1000)
    grids = dimension_grids_from_chunks((100,), (1000,))
    indexer = PartitionIndexer.from_selection(coordinates, (1000,), grids, mode="vectorized")
    coordinates[:] = 0
    np.testing.assert_array_equal(next(iter(indexer)).chunk_selection[0], np.arange(100))


def test_scalar_vectorized_selection_drops_its_axes() -> None:
    grids = dimension_grids_from_chunks((3, 4), (7, 9))
    indexer = PartitionIndexer.from_selection((2, 3), (7, 9), grids, mode="vectorized")
    assert indexer.shape == ()
    assert list(indexer) == [((0, 0), (2, 3), (), False)]


def test_partition_from_plan_chunks_is_the_same_walk() -> None:
    grids = dimension_grids_from_chunks((3, 4), (7, 9))
    transform = IndexTransform.from_shape((7, 9))[1:6:2, 5:]
    indexer = PartitionIndexer(plan_chunks(transform, grids).partition())
    assert [r.chunk_coords for r in indexer] == [(0, 1), (0, 2), (1, 1), (1, 2)]


def test_indexer_rejects_grid_rank_mismatch() -> None:
    with pytest.raises(ValueError, match="one entry per transform output dimension"):
        PartitionIndexer.from_selection(slice(None), (7,), ())


def test_indexer_rejects_unknown_mode() -> None:
    grids = dimension_grids_from_chunks((3,), (7,))
    with pytest.raises(ValueError, match="unknown indexing mode"):
        PartitionIndexer.from_selection(slice(None), (7,), grids, mode="invalid")  # type: ignore[arg-type]


def test_indexer_rejects_out_of_bounds_integer() -> None:
    with pytest.raises(IndexError):
        PartitionIndexer.from_selection(7, (7,), dimension_grids_from_chunks((3,), (7,)))


def test_indexer_rejects_zero_slice_step() -> None:
    grids = dimension_grids_from_chunks((3,), (7,))
    with pytest.raises((IndexError, ValueError), match="step"):
        PartitionIndexer.from_selection(slice(None, None, 0), (7,), grids)


def test_indexer_rejects_a_grid_smaller_than_the_selection() -> None:
    with pytest.raises(IndexError):
        PartitionIndexer.from_selection(
            slice(None), (100,), dimension_grids_from_chunks((10,), (50,))
        )


def test_indexer_rejects_an_unread_request_axis_wider_than_one() -> None:
    """One chunk cell cannot be scattered to several positions; only a singleton axis may be unread."""
    grids = dimension_grids_from_chunks((2,), (2,))
    transform = IndexTransform(IndexDomain.from_shape((2, 3)), (DimensionMap(0),))
    with pytest.raises(ValueError, match="read by no storage dimension"):
        list(PartitionIndexer.from_transform(transform, grids))
