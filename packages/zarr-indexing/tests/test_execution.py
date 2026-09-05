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


def test_boundary_chunk_is_not_complete_codec_buffer() -> None:
    grids = dimension_grids_from_chunks((3,), (7,))
    rows = list(execute_selection(slice(None), (7,), grids))
    assert [row.is_complete_chunk for row in rows] == [True, True, False]


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
    with pytest.raises(NotImplementedError, match="unread input axes"):
        execute_transform(transform, dimension_grids_from_chunks((2,), (2,)))


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
