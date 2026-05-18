"""Tests for `zarr.core._lazy._LazyArray` and helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from zarr.core._lazy import _LazyArray, transform_to_selection
from zarr.core._transforms import IndexTransform
from zarr.core._transforms.domain import IndexDomain
from zarr.core._transforms.output_map import ArrayMap, ConstantMap, DimensionMap


@pytest.fixture
def sample_array() -> Any:
    """A small in-memory zarr array for use in tests."""
    arr = zarr.create_array(
        store=zarr.storage.MemoryStore(),
        shape=(10, 20),
        dtype="int32",
        chunks=(5, 10),
    )
    arr[:] = np.arange(200, dtype="int32").reshape(10, 20)
    return arr


def test_lazy_array_shape_dtype_ndim_from_identity_transform(sample_array: Any) -> None:
    """A _LazyArray built with an identity transform reports the underlying
    array's shape, dtype, and ndim."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    assert lazy.shape == (10, 20)
    assert lazy.ndim == 2
    assert lazy.dtype == np.dtype("int32")


def test_lazy_array_repr_contains_shape_and_dtype(sample_array: Any) -> None:
    """__repr__ surfaces the shape, dtype, and a domain string for debugging."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    r = repr(lazy)
    assert "(10, 20)" in r
    assert "int32" in r
    assert "[0, 10)" in r  # selection_repr formatting


def test_transform_to_selection_basic_slices() -> None:
    """An identity 2D transform produces two full-range slices in basic mode."""
    t = IndexTransform.from_shape((10, 20))
    sel, mode = transform_to_selection(t)
    assert mode == "basic"
    assert sel == (slice(0, 10), slice(0, 20))


def test_transform_to_selection_basic_int_then_slice() -> None:
    """A ConstantMap-then-DimensionMap transform reduces to (int, slice)."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((20,)),
        output=(ConstantMap(offset=3), DimensionMap(input_dimension=0)),
    )
    sel, mode = transform_to_selection(t)
    assert mode == "basic"
    assert sel == (3, slice(0, 20))


def test_transform_to_selection_basic_strided_slice() -> None:
    """A DimensionMap with stride > 1 produces a strided slice."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((5,)),
        output=(DimensionMap(input_dimension=0, offset=2, stride=3),),
    )
    sel, mode = transform_to_selection(t)
    assert mode == "basic"
    assert sel == (slice(2, 17, 3),)


def test_transform_to_selection_orthogonal_single_arraymap() -> None:
    """One ArrayMap output → orthogonal mode."""
    arr = np.array([1, 4, 7], dtype=np.intp)
    t = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(ArrayMap(index_array=arr, input_dimensions=(0,)),),
    )
    sel, mode = transform_to_selection(t)
    assert mode == "orthogonal"
    np.testing.assert_array_equal(sel[0], arr)


def test_transform_to_selection_orthogonal_two_arraymaps_disjoint() -> None:
    """Two ArrayMaps on disjoint input dims → orthogonal mode."""
    a0 = np.array([0, 2], dtype=np.intp)
    a1 = np.array([1, 3, 5], dtype=np.intp)
    t = IndexTransform(
        domain=IndexDomain.from_shape((2, 3)),
        output=(
            ArrayMap(index_array=a0, input_dimensions=(0,)),
            ArrayMap(index_array=a1, input_dimensions=(1,)),
        ),
    )
    _sel, mode = transform_to_selection(t)
    assert mode == "orthogonal"


def test_transform_to_selection_vectorized_two_arraymaps_correlated() -> None:
    """Two ArrayMaps sharing an input dim → vectorized mode."""
    a0 = np.array([0, 1, 2], dtype=np.intp)
    a1 = np.array([10, 11, 12], dtype=np.intp)
    t = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(
            ArrayMap(index_array=a0, input_dimensions=(0,)),
            ArrayMap(index_array=a1, input_dimensions=(0,)),
        ),
    )
    sel, mode = transform_to_selection(t)
    assert mode == "vectorized"
    np.testing.assert_array_equal(sel[0], a0)
    np.testing.assert_array_equal(sel[1], a1)


def test_transform_to_selection_mixed_constant_and_arraymap() -> None:
    """A ConstantMap mixed with an ArrayMap → orthogonal mode; ConstantMap is
    ignored by mode classification (no input_dimensions), and contributes an
    integer index to the selection."""
    arr = np.array([1, 4, 7], dtype=np.intp)
    t = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(
            ConstantMap(offset=5),
            ArrayMap(index_array=arr, input_dimensions=(0,)),
        ),
    )
    sel, mode = transform_to_selection(t)
    assert mode == "orthogonal"
    assert sel[0] == 5
    np.testing.assert_array_equal(sel[1], arr)


def test_transform_to_selection_dimensionmap_over_nonzero_origin_domain() -> None:
    """A DimensionMap over an IndexDomain with non-zero origin produces a slice
    whose start/stop are in absolute coordinates (offset + stride * inclusive_min
    .. offset + stride * exclusive_max)."""
    t = IndexTransform(
        domain=IndexDomain(inclusive_min=(5,), exclusive_max=(10,)),
        output=(DimensionMap(input_dimension=0, offset=0, stride=1),),
    )
    sel, mode = transform_to_selection(t)
    assert mode == "basic"
    assert sel == (slice(5, 10),)


def test_transform_to_selection_orthogonal_multi_dim_arraymap_raises() -> None:
    """A multi-dimensional ArrayMap is rejected in orthogonal mode; zarr's
    oindex requires 1-D per-axis selectors."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((3, 4)),
        output=(
            ArrayMap(
                index_array=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.intp),
                input_dimensions=(0, 1),
            ),
        ),
    )
    with pytest.raises(NotImplementedError, match="multi-dimensional ArrayMap"):
        transform_to_selection(t)


# ---------------------------------------------------------------------------
# _LazyArray.__getitem__ (basic) + .result()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param(slice(None), id="full-slice"),
        pytest.param(slice(2, 8), id="narrowing-slice"),
        pytest.param(slice(None, None, 2), id="strided-slice"),
        pytest.param(3, id="int-drops-leading-dim"),
        pytest.param((slice(2, 8), slice(5, 15)), id="2d-narrowing-slices"),
        pytest.param((3, slice(None)), id="2d-int-and-slice"),
        pytest.param((slice(None), 5), id="2d-slice-and-int"),
        pytest.param(..., id="ellipsis"),
        pytest.param((slice(2, 8), ...), id="leading-slice-then-ellipsis"),
    ],
)
def test_lazy_basic_indexing_matches_eager(sample_array: Any, selection: Any) -> None:
    """For each basic selection, arr.lazy[sel].result() == arr[sel]."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    actual = lazy[selection].result()
    expected = sample_array[selection]
    np.testing.assert_array_equal(actual, expected)


def test_lazy_basic_indexing_composes(sample_array: Any) -> None:
    """Composed slices on a _LazyArray equal the same composed slices on the array."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    actual = lazy[2:8][1:4].result()
    expected = sample_array[2:8][1:4]
    np.testing.assert_array_equal(actual, expected)


def test_lazy_getitem_on_nonzero_origin_domain() -> None:
    """Basic indexing into a _LazyArray whose transform has a non-zero-origin
    domain materializes correctly. Not a common case for the public API but
    exercises the bridge between transform domains and eager selections."""
    src = zarr.create_array(
        store=zarr.storage.MemoryStore(), shape=(10,), dtype="int32", chunks=(10,)
    )
    src[:] = np.arange(10, dtype="int32")
    t = IndexTransform(
        domain=IndexDomain(inclusive_min=(5,), exclusive_max=(10,)),
        output=(DimensionMap(input_dimension=0, offset=0, stride=1),),
    )
    lazy = _LazyArray(_array=src, _transform=t)
    result = lazy.result()
    np.testing.assert_array_equal(result, np.arange(5, 10, dtype="int32"))


# ---------------------------------------------------------------------------
# _LazyOIndex and _LazyVIndex helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param(np.array([1, 3, 5], dtype=np.intp), id="1d-int-array"),
        pytest.param(
            (np.array([1, 3], dtype=np.intp), np.array([2, 4, 6], dtype=np.intp)),
            id="2d-two-arrays",
        ),
        pytest.param(
            (np.array([1, 3, 5], dtype=np.intp), slice(None)),
            id="2d-array-and-slice",
        ),
        pytest.param(
            np.array([True, False, True, False, True, False, True, False, True, False]),
            id="1d-bool-mask",
        ),
    ],
)
def test_lazy_oindex_matches_eager(sample_array: Any, selection: Any) -> None:
    """arr.lazy.oindex[sel].result() == arr.oindex[sel]."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    actual = lazy.oindex[selection].result()
    expected = sample_array.oindex[selection]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param(
            (np.array([1, 3, 5], dtype=np.intp), np.array([2, 4, 6], dtype=np.intp)),
            id="2d-two-correlated-arrays",
        ),
        pytest.param(
            (
                np.array([[1, 2], [3, 4]], dtype=np.intp),
                np.array([[5, 6], [7, 8]], dtype=np.intp),
            ),
            id="2d-broadcasted-arrays",
        ),
    ],
)
def test_lazy_vindex_matches_eager(sample_array: Any, selection: Any) -> None:
    """arr.lazy.vindex[sel].result() == arr.vindex[sel]."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    actual = lazy.vindex[selection].result()
    expected = sample_array.vindex[selection]
    np.testing.assert_array_equal(actual, expected)


def test_lazy_oindex_returns_lazy_array(sample_array: Any) -> None:
    """lazy.oindex[sel] returns a _LazyArray (composition friendly) rather
    than materializing immediately. Calling .result() then materializes."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    result = lazy.oindex[np.array([1, 3, 5], dtype=np.intp)]
    assert isinstance(result, _LazyArray)


def test_lazy_vindex_returns_lazy_array(sample_array: Any) -> None:
    """lazy.vindex[sel] returns a _LazyArray (composition friendly) rather
    than materializing immediately."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    result = lazy.vindex[(np.array([1, 3], dtype=np.intp), np.array([2, 4], dtype=np.intp))]
    assert isinstance(result, _LazyArray)


# ---------------------------------------------------------------------------
# __array__ numpy interop
# ---------------------------------------------------------------------------


def test_np_asarray_materializes_lazy(sample_array: Any) -> None:
    """np.asarray(lazy[sel]) materializes the lazy view and returns the
    eager-indexing result."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    actual = np.asarray(lazy[2:8, 5:15])
    expected = np.asarray(sample_array[2:8, 5:15])
    np.testing.assert_array_equal(actual, expected)


def test_np_asarray_with_dtype_cast(sample_array: Any) -> None:
    """np.asarray(lazy, dtype=...) materializes and then casts to the requested
    dtype, preserving the cast value."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    actual = np.asarray(lazy[:2, :2], dtype="float64")
    assert actual.dtype == np.dtype("float64")
    np.testing.assert_array_equal(actual, np.asarray(sample_array[:2, :2]).astype("float64"))


def test_np_array_constructor_materializes_lazy(sample_array: Any) -> None:
    """np.array(lazy) also materializes via __array__ (equivalent path to np.asarray)."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    actual = np.array(lazy[1:5])
    expected = np.array(sample_array[1:5])
    np.testing.assert_array_equal(actual, expected)


def test_np_array_copy_false_raises(sample_array: Any) -> None:
    """NumPy 2.0's __array__ protocol requires raising ValueError when copy=False
    cannot be honored. _LazyArray.__array__ must materialize (allocate), so
    copy=False can never be satisfied."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    with pytest.raises(ValueError, match="copy=False"):
        np.array(lazy[1:5], copy=False)


def test_np_array_copy_true_materializes(sample_array: Any) -> None:
    """Explicit copy=True behaves the same as copy=None (default): materialize."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    actual = np.array(lazy[1:5], copy=True)
    expected = np.asarray(sample_array[1:5])
    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# __setitem__ materializing write-through
# ---------------------------------------------------------------------------


def test_lazy_setitem_writes_through_basic(sample_array: Any) -> None:
    """lazy[sel] = value writes through the eager set path; a subsequent read
    of the same coordinate sees the new value."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    lazy[5, 10] = -1
    assert sample_array[5, 10] == -1


def test_lazy_setitem_writes_through_basic_slice(sample_array: Any) -> None:
    """lazy[slice] = value writes a slab through the eager set path."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    lazy[0, :] = -2
    np.testing.assert_array_equal(sample_array[0, :], np.full(20, -2, dtype="int32"))


def test_lazy_oindex_setitem_writes_through(sample_array: Any) -> None:
    """lazy.oindex[sel] = value writes through array.oindex[sel] = value;
    multiple rows selected, each gets the broadcast value."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    lazy.oindex[np.array([1, 3], dtype=np.intp), :] = -99
    np.testing.assert_array_equal(sample_array[1, :], np.full(20, -99, dtype="int32"))
    np.testing.assert_array_equal(sample_array[3, :], np.full(20, -99, dtype="int32"))


def test_lazy_vindex_setitem_writes_through(sample_array: Any) -> None:
    """lazy.vindex[sel] = value writes through array.vindex[sel] = value;
    correlated point selection assigns the value at each (row, col) pair."""
    t = IndexTransform.from_shape(sample_array.shape)
    lazy = _LazyArray(_array=sample_array, _transform=t)
    lazy.vindex[(np.array([1, 3, 5], dtype=np.intp), np.array([2, 4, 6], dtype=np.intp))] = -7
    assert sample_array[1, 2] == -7
    assert sample_array[3, 4] == -7
    assert sample_array[5, 6] == -7


# ---------------------------------------------------------------------------
# Array.lazy property — the public user-facing entry point.
# ---------------------------------------------------------------------------


def test_array_lazy_property_returns_lazy_array(sample_array: Any) -> None:
    """arr.lazy returns a _LazyArray over the full array (identity transform)."""
    lazy = sample_array.lazy
    assert isinstance(lazy, _LazyArray)
    assert lazy.shape == sample_array.shape
    assert lazy.dtype == sample_array.dtype


def test_array_lazy_end_to_end_basic(sample_array: Any) -> None:
    """arr.lazy[a][b].result() composes through both selections and yields
    the same data as the equivalent eager indexing."""
    actual = sample_array.lazy[2:8, 5:15][1:3, 2:5].result()
    expected = sample_array[2:8, 5:15][1:3, 2:5]
    np.testing.assert_array_equal(actual, expected)


def test_array_lazy_end_to_end_oindex(sample_array: Any) -> None:
    """arr.lazy.oindex[...] matches arr.oindex[...] from the public entry point."""
    sel = (np.array([1, 3, 5], dtype=np.intp), slice(2, 10))
    actual = sample_array.lazy.oindex[sel].result()
    expected = sample_array.oindex[sel]
    np.testing.assert_array_equal(actual, expected)


def test_array_lazy_end_to_end_vindex(sample_array: Any) -> None:
    """arr.lazy.vindex[...] matches arr.vindex[...] from the public entry point."""
    sel = (np.array([1, 3, 5], dtype=np.intp), np.array([2, 4, 6], dtype=np.intp))
    actual = sample_array.lazy.vindex[sel].result()
    expected = sample_array.vindex[sel]
    np.testing.assert_array_equal(actual, expected)


def test_array_lazy_np_asarray_interop(sample_array: Any) -> None:
    """np.asarray(arr.lazy[sel]) returns the materialized eager-indexed array."""
    actual = np.asarray(sample_array.lazy[3:7, 2:8])
    expected = np.asarray(sample_array[3:7, 2:8])
    np.testing.assert_array_equal(actual, expected)
