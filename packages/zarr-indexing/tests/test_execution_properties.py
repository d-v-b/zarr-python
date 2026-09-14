"""Prepared execution against a chunked NumPy reference.

For any array shape, chunk grid (fixed, rectilinear, clipped to the extent, or
the narrow edge-grid protocol without `data_size`) and basic, orthogonal or
vectorized selection, assembling the rows of an execution plan out of the
chunks of a reference array must reproduce NumPy's answer; a plan must yield
the same rows every time it is walked; `is_complete_chunk` must be a proof
that the row covers its chunk's data exactly once; and a write plan must
scatter values exactly as NumPy assignment does, or refuse a selection that
repeats a destination unless told to keep the last value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from zarr_indexing import (
    EdgeDimensionGrid,
    FixedDimension,
    IndexDomain,
    LazyArray,
    VaryingDimension,
)
from zarr_indexing._execution import (
    ExecutionChunk,
    ExecutionPlan,
    execute_selection,
    execute_transform,
)
from zarr_indexing.boundary import normalize_positional_selection
from zarr_indexing.testing import (
    apply_selection,
    basic_selections,
    orthogonal_selections,
    vectorized_selections,
)

GRID_KINDS = ("fixed", "rectilinear", "clipped", "edges")
SELECTIONS = {
    "basic": basic_selections,
    "orthogonal": orthogonal_selections,
    "vectorized": vectorized_selections,
}


@st.composite
def _axis_grid(draw: st.DrawFn, size: int, kind: str) -> Any:
    if kind == "fixed":
        return FixedDimension(size=draw(st.integers(1, size + 1)), extent=size)
    flags = draw(st.lists(st.booleans(), min_size=size - 1, max_size=size - 1))
    cuts = [position + 1 for position, flag in enumerate(flags) if flag]
    edges = [b - a for a, b in zip([0, *cuts], [*cuts, size], strict=True)]
    if kind == "rectilinear":
        return VaryingDimension(edges=edges, extent=size)
    if kind == "edges":
        return EdgeDimensionGrid(edges)
    # Declared edges past the extent: the last data chunk is shorter than it
    # declares, and there may be declared chunks holding no data at all.
    edges[-1] += draw(st.integers(1, 3))
    edges.extend(draw(st.lists(st.integers(1, 3), max_size=2)))
    return VaryingDimension(edges=edges, extent=size)


@st.composite
def _cases(draw: st.DrawFn) -> tuple[tuple[int, ...], str, tuple[Any, ...], str, Any]:
    if draw(st.integers(0, 4)) == 0:
        # A dense sorted coordinate gather over few chunks: the only shape the
        # sorted fast path accepts, which the small shapes below never produce.
        size = draw(st.integers(8, 64))
        grid = FixedDimension(size=draw(st.integers(1, size)), extent=size)
        points = draw(st.lists(st.integers(0, size - 1), min_size=1, max_size=2 * size))
        return (size,), "fixed", (grid,), "vectorized", (np.array(sorted(points), dtype=np.intp),)
    ndim = draw(st.integers(0, 3))
    shape = tuple(draw(st.lists(st.integers(1, 6), min_size=ndim, max_size=ndim)))
    kind = draw(st.sampled_from(GRID_KINDS))
    grids = tuple(draw(_axis_grid(size, kind)) for size in shape)
    modes = ["basic", "orthogonal", "vectorized"] if ndim else ["basic", "orthogonal"]
    mode = draw(st.sampled_from(modes))
    return shape, kind, grids, mode, draw(SELECTIONS[mode](shape))


def _data_size(grid: Any, chunk: int) -> int:
    data_size = getattr(grid, "data_size", None)
    return int(grid.chunk_size(chunk) if data_size is None else data_size(chunk))


def _bounds(grids: tuple[Any, ...], coords: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(
        slice(g.chunk_offset(c), g.chunk_offset(c) + _data_size(g, c))
        for g, c in zip(grids, coords, strict=True)
    )


def _gather(
    rows: list[ExecutionChunk], reference: np.ndarray[Any, Any], grids: tuple[Any, ...], shape: Any
) -> np.ndarray[Any, Any]:
    """Assemble a read from chunks, checking every cell lands exactly once."""
    result = np.full(shape, -1, dtype=reference.dtype)
    touched = np.zeros(shape, dtype=np.intp)
    for row in rows:
        chunk = reference[_bounds(grids, row.chunk_coords)]
        result[row.out_selection] = chunk[row.chunk_selection]
        np.add.at(touched, row.out_selection, 1)
        if row.is_complete_chunk:
            cover = np.zeros(chunk.shape, dtype=np.intp)
            np.add.at(cover, row.chunk_selection, 1)
            assert (cover == 1).all(), (row, cover)
    assert (touched == 1).all(), touched
    return result


def _scatter(
    rows: list[ExecutionChunk], target: np.ndarray[Any, Any], grids: tuple[Any, ...], values: Any
) -> None:
    for row in rows:
        bounds = _bounds(grids, row.chunk_coords)
        chunk = target[bounds] if bounds else target
        chunk[row.chunk_selection] = values[row.out_selection]


def _assign(reference: np.ndarray[Any, Any], selection: Any, mode: str, values: Any) -> None:
    """NumPy's assignment for a selection in `mode`: the oracle for a write plan."""
    if mode != "orthogonal":
        reference[selection] = values
        return
    scalars = tuple(
        sel if isinstance(sel, (int, np.integer)) and not isinstance(sel, bool) else slice(None)
        for sel in selection
    )
    axes = [
        np.arange(size)[sel]
        for size, sel in zip(
            reference[scalars].shape,
            [s for s in selection if not isinstance(s, (int, np.integer))],
            strict=True,
        )
    ]
    if axes:
        reference[scalars][np.ix_(*axes)] = values
    else:
        reference[scalars] = values


def _same_rows(first: list[ExecutionChunk], second: list[ExecutionChunk]) -> bool:
    if len(first) != len(second):
        return False
    for a, b in zip(first, second, strict=True):
        if a.chunk_coords != b.chunk_coords or a.is_complete_chunk != b.is_complete_chunk:
            return False
        for x, y in zip(
            a.chunk_selection + a.out_selection, b.chunk_selection + b.out_selection, strict=True
        ):
            if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
                    return False
                if x.shape != y.shape or not np.array_equal(x, y):
                    return False
            elif x != y:
                return False
    return True


@settings(max_examples=300)
@given(case=_cases())
def test_execution_reproduces_numpy(
    case: tuple[tuple[int, ...], str, tuple[Any, ...], str, Any],
) -> None:
    shape, kind, grids, mode, selection = case
    event(f"grid={kind}")
    event(f"mode={mode}")
    event(f"rank={len(shape)}")
    reference = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    expected = np.asarray(apply_selection(reference, selection, mode))
    # Distinct cell ids tell a selection that repeats a storage cell from one that does not.
    unique = np.unique(expected).size == expected.size
    event("unique" if unique else "duplicates")
    values = np.arange(expected.size, dtype=np.int64).reshape(expected.shape) + 1000

    view = LazyArray.from_numpy(reference)
    lazy = view.lazy
    indexed = (
        lazy[selection]
        if mode == "basic"
        else lazy.oindex[selection]
        if mode == "orthogonal"
        else lazy.vindex[selection]
    )
    literal = normalize_positional_selection(selection, IndexDomain.from_shape(shape), mode)
    plans: list[tuple[str, ExecutionPlan]] = [
        ("transform", execute_transform(indexed.transform, grids)),
        ("selection", execute_selection(literal, shape, grids, mode=mode)),
    ]
    for entry, plan in plans:
        event(f"{entry}:{type(plan.work).__name__}")
        assert plan.shape == expected.shape, (entry, plan.shape, expected.shape)
        for consumer in ("numpy", "shard"):
            rows = list(plan.lower(consumer))
            assert _same_rows(rows, list(plan.lower(consumer))), "a plan must walk the same twice"
            np.testing.assert_array_equal(_gather(rows, reference, grids, expected.shape), expected)

    expected_write = reference.copy()
    _assign(expected_write, selection, mode, values)
    if unique:
        write_plan = execute_transform(indexed.transform, grids, access="write")
    else:
        with pytest.raises(ValueError, match="duplicate writes"):
            execute_transform(indexed.transform, grids, access="write")
        write_plan = execute_transform(indexed.transform, grids, access="write", conflicts="last")
    for consumer in ("numpy", "shard"):
        written = reference.copy()
        _scatter(list(write_plan.lower(consumer)), written, grids, values)
        np.testing.assert_array_equal(written, expected_write)
