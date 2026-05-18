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
