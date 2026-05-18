"""Tests for `zarr.core._lazy._LazyArray` and helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from zarr.core._lazy import _LazyArray
from zarr.core._transforms import IndexTransform


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
