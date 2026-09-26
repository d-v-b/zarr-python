"""Tests for the state cell shared by `Array` and `AsyncArray`."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest

import zarr
from zarr.core.array import (
    Array,
    _plan_append,
    _plan_resize,
    _plan_update_attributes,
)
from zarr.core.sync import sync

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("via", ["sync", "async"])
@pytest.mark.parametrize("op", ["resize", "append", "update_attributes"])
def test_views_share_state(
    zarr_format: ZarrFormat,
    via: Literal["sync", "async"],
    op: Literal["resize", "append", "update_attributes"],
) -> None:
    """
    A change made through an `Array` or through its `async_array` is seen by both, and by
    the store.
    """
    arr = zarr.create_array({}, shape=(10,), chunks=(2,), dtype="i4", zarr_format=zarr_format)
    arr[:] = np.arange(10)
    aa = arr.async_array
    expected_attrs: dict[str, Any]

    if op == "resize":
        expected_shape, expected_attrs = (6,), {}
        expected_data = np.arange(6)
        if via == "sync":
            arr.resize((6,))
        else:
            sync(aa.resize((6,)))
    elif op == "append":
        expected_shape, expected_attrs = (12,), {}
        expected_data = np.concatenate([np.arange(10), [7, 7]])
        if via == "sync":
            arr.append(np.array([7, 7]))
        else:
            sync(aa.append(np.array([7, 7])))
    else:
        expected_shape, expected_attrs = (10,), {"a": 1}
        expected_data = np.arange(10)
        if via == "sync":
            arr.update_attributes({"a": 1})
        else:
            sync(aa.update_attributes({"a": 1}))

    reopened = zarr.open_array(arr.store_path.store, zarr_format=zarr_format)
    for view in (arr, aa, reopened):
        assert view.shape == expected_shape
        assert dict(view.attrs) == expected_attrs
    np.testing.assert_array_equal(arr[:], expected_data)
    np.testing.assert_array_equal(reopened[:], expected_data)
    assert aa._cell is arr._cell


@pytest.mark.parametrize(
    ("new_shape", "delete_outside_chunks", "expected_keys"),
    [
        ((12,), True, set()),
        ((5,), False, set()),
        ((5,), True, {"c/3", "c/4"}),
        ((0,), True, {"c/0", "c/1", "c/2", "c/3", "c/4"}),
    ],
)
def test_plan_resize(
    new_shape: tuple[int, ...], delete_outside_chunks: bool, expected_keys: set[str]
) -> None:
    """
    Planning a resize computes the new state and the keys to delete, and changes nothing.
    """
    arr = zarr.create_array({}, shape=(10,), chunks=(2,), dtype="i4")
    state = arr._cell.state

    new_state, keys = _plan_resize(state, new_shape, delete_outside_chunks=delete_outside_chunks)

    assert new_state.metadata.shape == new_shape
    assert new_state.chunk_grid.grid_shape == (-(-new_shape[0] // 2),)
    assert new_state.codec_pipeline is state.codec_pipeline
    assert set(keys) == expected_keys
    assert arr._cell.state is state
    assert arr.shape == (10,)


def test_plan_resize_wrong_ndim() -> None:
    """
    Planning a resize to a shape with a different number of dimensions raises.
    """
    arr = zarr.create_array({}, shape=(10,), chunks=(2,), dtype="i4")
    with pytest.raises(ValueError, match="same number of dimensions"):
        _plan_resize(arr._cell.state, (10, 10), delete_outside_chunks=True)


@pytest.mark.parametrize(
    ("shape", "data_shape", "axis", "expected_shape", "expected_selection"),
    [
        ((10,), (3,), 0, (13,), (slice(10, 13),)),
        ((4, 5), (2, 5), 0, (6, 5), (slice(4, 6), slice(None))),
        ((4, 5), (4, 1), 1, (4, 6), (slice(None), slice(5, 6))),
    ],
)
def test_plan_append(
    shape: tuple[int, ...],
    data_shape: tuple[int, ...],
    axis: int,
    expected_shape: tuple[int, ...],
    expected_selection: tuple[slice, ...],
) -> None:
    """
    Planning an append computes the new shape and the region the data is written to.
    """
    assert _plan_append(shape, data_shape, axis) == (expected_shape, expected_selection)


def test_plan_append_incompatible_shape() -> None:
    """
    Planning an append of data whose other dimensions do not match the array raises.
    """
    with pytest.raises(ValueError, match="not compatible"):
        _plan_append((4, 5), (2, 3), 0)


def test_plan_update_attributes() -> None:
    """
    Planning an attribute update merges into a copy; the current state is not modified.
    """
    arr = zarr.create_array({}, shape=(10,), chunks=(2,), dtype="i4", attributes={"a": 1})
    state = arr._cell.state

    new_state = _plan_update_attributes(state, {"b": 2})

    assert new_state.metadata.attributes == {"a": 1, "b": 2}
    assert state.metadata.attributes == {"a": 1}
    assert new_state.codec_pipeline is state.codec_pipeline


def test_pickle_keeps_views_coherent() -> None:
    """
    After a pickle round trip, an `Array` and its `async_array` still share one state cell.
    """
    arr = zarr.create_array({}, shape=(10,), chunks=(2,), dtype="i4")
    restored: Array[Any] = pickle.loads(pickle.dumps(arr))

    restored.resize((4,))

    assert restored.async_array.shape == (4,)
    assert restored.async_array._cell is restored._cell
