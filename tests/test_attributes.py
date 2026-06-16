from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr.core
import zarr.core.attributes
import zarr.storage
from tests.conftest import deep_nan_equal

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat
    from zarr.types import AnyArray


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "data", [{"inf": np.inf, "-inf": -np.inf, "nan": np.nan}, {"a": 3, "c": 4}]
)
def test_put(data: dict[str, Any], zarr_format: ZarrFormat) -> None:
    store = zarr.storage.MemoryStore()
    attrs = zarr.core.attributes.Attributes(zarr.Group.from_store(store, zarr_format=zarr_format))
    attrs.put(data)
    expected = json.loads(json.dumps(data, allow_nan=True))
    assert deep_nan_equal(dict(attrs), expected)


def test_asdict() -> None:
    store = zarr.storage.MemoryStore()
    attrs = zarr.core.attributes.Attributes(
        zarr.Group.from_store(store, attributes={"a": 1, "b": 2})
    )
    result = attrs.asdict()
    assert result == {"a": 1, "b": 2}


def test_update_attributes_preserves_existing() -> None:
    """
    Test that `update_attributes` only updates the specified attributes
    and preserves existing ones.
    """
    store = zarr.storage.MemoryStore()
    z = zarr.create(10, store=store, overwrite=True)
    z.attrs["a"] = []
    z.attrs["b"] = 3
    assert dict(z.attrs) == {"a": [], "b": 3}

    z.update_attributes({"a": [3, 4], "c": 4})
    assert dict(z.attrs) == {"a": [3, 4], "b": 3, "c": 4}


def test_update_empty_attributes() -> None:
    """
    Ensure updating when initial attributes are empty works.
    """
    store = zarr.storage.MemoryStore()
    z = zarr.create(10, store=store, overwrite=True)
    assert dict(z.attrs) == {}
    z.update_attributes({"a": [3, 4], "c": 4})
    assert dict(z.attrs) == {"a": [3, 4], "c": 4}


def test_update_no_changes() -> None:
    """
    Ensure updating when no new or modified attributes does not alter existing ones.
    """
    store = zarr.storage.MemoryStore()
    z = zarr.create(10, store=store, overwrite=True)
    z.attrs["a"] = []
    z.attrs["b"] = 3

    z.update_attributes({})
    assert dict(z.attrs) == {"a": [], "b": 3}


def _make_node(
    store: zarr.storage.MemoryStore,
    *,
    group: bool,
    attributes: dict[str, Any] | None = None,
) -> zarr.Group | AnyArray:
    """Create a fresh sync group or array with the given attributes."""
    if group:
        return zarr.create_group(store, attributes=attributes)
    return zarr.create_array(
        store=store, shape=10, dtype=int, attributes=attributes, overwrite=True
    )


@pytest.mark.parametrize("group", [True, False])
def test_update_attributes_merges(group: bool) -> None:
    """`update_attributes` merges into existing attributes; pre-existing keys survive."""
    store = zarr.storage.MemoryStore()
    z = _make_node(store, group=group, attributes={"a": 1})
    z.update_attributes({"b": 2})
    assert dict(z.attrs) == {"a": 1, "b": 2}


@pytest.mark.parametrize("group", [True, False])
def test_update_attributes_no_mutation(group: bool) -> None:
    """`update_attributes` must not mutate the original frozen metadata object."""
    store = zarr.storage.MemoryStore()
    z = _make_node(store, group=group, attributes={"a": 1})
    old_metadata = z.metadata
    snapshot = dict(old_metadata.attributes)
    z.update_attributes({"b": 2})
    assert dict(old_metadata.attributes) == snapshot


@pytest.mark.parametrize("group", [True, False])
def test_replace_attributes_replaces_sync(group: bool) -> None:
    """`replace_attributes` drops keys absent from the new dict (sync)."""
    store = zarr.storage.MemoryStore()
    z = _make_node(store, group=group, attributes={"a": 1, "b": 2})
    z.replace_attributes({"a": 3, "c": 4})
    assert dict(z.attrs) == {"a": 3, "c": 4}


@pytest.mark.parametrize("group", [True, False])
async def test_replace_attributes_replaces_async(group: bool) -> None:
    """`replace_attributes` drops keys absent from the new dict (async)."""
    store = zarr.storage.MemoryStore()
    z = _make_node(store, group=group, attributes={"a": 1, "b": 2})
    async_obj = z._async_group if isinstance(z, zarr.Group) else z.async_array
    await async_obj.replace_attributes({"a": 3, "c": 4})
    assert dict(async_obj.metadata.attributes) == {"a": 3, "c": 4}


@pytest.mark.parametrize("group", [True, False])
def test_replace_attributes_no_mutation(group: bool) -> None:
    """`replace_attributes` must not mutate the original frozen metadata object."""
    store = zarr.storage.MemoryStore()
    z = _make_node(store, group=group, attributes={"a": 1, "b": 2})
    old_metadata = z.metadata
    snapshot = dict(old_metadata.attributes)
    z.replace_attributes({"a": 3, "c": 4})
    assert dict(old_metadata.attributes) == snapshot


@pytest.mark.parametrize("group", [True, False])
def test_put_replaces(group: bool) -> None:
    """`attrs.put` replaces all attributes, dropping absent keys."""
    store = zarr.storage.MemoryStore()
    z = _make_node(store, group=group, attributes={"a": 1, "b": 2})
    z.attrs.put({"a": 3, "c": 4})
    assert dict(z.attrs) == {"a": 3, "c": 4}


@pytest.mark.parametrize("group", [True, False])
def test_replace_attributes_persists(group: bool) -> None:
    """After replace, reopening from the store reflects the dropped keys."""
    store = zarr.storage.MemoryStore()
    z = _make_node(store, group=group, attributes={"a": 1, "b": 2})
    z.replace_attributes({"a": 3, "c": 4})

    z2: zarr.Group | AnyArray
    if group:
        z2 = zarr.open_group(store)
    else:
        z2 = zarr.open_array(store)
    assert dict(z2.attrs) == {"a": 3, "c": 4}


@pytest.mark.parametrize("group", [True, False])
def test_del_works(group: bool) -> None:
    store = zarr.storage.MemoryStore()
    z: zarr.Group | AnyArray
    if group:
        z = zarr.create_group(store)
    else:
        z = zarr.create_array(store=store, shape=10, dtype=int)
    assert dict(z.attrs) == {}
    z.update_attributes({"a": [3, 4], "c": 4})
    del z.attrs["a"]
    assert dict(z.attrs) == {"c": 4}

    z2: zarr.Group | AnyArray
    if group:
        z2 = zarr.open_group(store)
    else:
        z2 = zarr.open_array(store)
    assert dict(z2.attrs) == {"c": 4}
