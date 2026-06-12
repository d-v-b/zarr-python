import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr.core
import zarr.core.attributes
import zarr.storage
from tests.conftest import deep_nan_equal
from zarr.core.common import ZarrFormat

if TYPE_CHECKING:
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


# ---------------------------------------------------------------------------
# Regression tests for frozen-metadata mutation and put semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group", [True, False])
def test_update_attributes_does_not_mutate_original_metadata(group: bool) -> None:
    """update_attributes must not mutate the pre-existing metadata object.

    Previously, both AsyncArray._update_attributes and AsyncGroup.update_attributes
    called ``metadata.attributes.update(...)`` in place, silently mutating the dict
    inside a frozen dataclass.  After the fix the old metadata object must be
    identical to what it was before the call.
    """
    store = zarr.storage.MemoryStore()
    z: zarr.Group | AnyArray
    if group:
        z = zarr.create_group(store, attributes={"a": 1})
    else:
        z = zarr.create_array(store=store, shape=10, dtype=int, attributes={"a": 1})

    # Capture a reference to the *current* metadata object (not a copy).
    original_metadata = z.metadata
    original_attrs_snapshot: dict[str, Any] = dict(original_metadata.attributes)

    z = z.update_attributes({"b": 2})

    # The in-memory view should have the merged result.
    assert dict(z.attrs) == {"a": 1, "b": 2}

    # The old metadata object must be unchanged.
    assert dict(original_metadata.attributes) == original_attrs_snapshot


@pytest.mark.parametrize("group", [True, False])
def test_put_replaces_not_merges(group: bool) -> None:
    """Attributes.put must fully replace attributes, not merge.

    Keys present before put() but absent from the new dict must be removed,
    both in memory and after re-opening from the store.
    """
    store = zarr.storage.MemoryStore()
    z: zarr.Group | AnyArray
    if group:
        z = zarr.create_group(store, attributes={"keep": 1, "remove": 99})
    else:
        z = zarr.create_array(
            store=store, shape=10, dtype=int, attributes={"keep": 1, "remove": 99}
        )

    z.attrs.put({"keep": 2, "new": 3})

    # In-memory view must match exactly.
    assert dict(z.attrs) == {"keep": 2, "new": 3}

    # Persisted view must also match.
    z2: zarr.Group | AnyArray
    if group:
        z2 = zarr.open_group(store)
    else:
        z2 = zarr.open_array(store)
    assert dict(z2.attrs) == {"keep": 2, "new": 3}


@pytest.mark.parametrize("group", [True, False])
def test_put_does_not_mutate_metadata_before_write(group: bool) -> None:
    """put() must not clear in-memory state before the store write completes.

    The old code called ``metadata.attributes.clear()`` before ``update_attributes``,
    which emptied the live dict before persisting, leaving inconsistent state on
    write failure and mutating a frozen dataclass.  After the fix the original
    metadata object must still hold its old attributes.
    """
    store = zarr.storage.MemoryStore()
    z: zarr.Group | AnyArray
    if group:
        z = zarr.create_group(store, attributes={"old": 1})
    else:
        z = zarr.create_array(store=store, shape=10, dtype=int, attributes={"old": 1})

    old_metadata = z.metadata
    old_attrs_snapshot: dict[str, Any] = dict(old_metadata.attributes)

    z.attrs.put({"new": 2})

    # After put, the original metadata object should be unchanged (no mutation).
    assert dict(old_metadata.attributes) == old_attrs_snapshot
