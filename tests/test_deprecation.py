"""Tests for the deprecation of ``zarr.create`` and the advance-notice warning
emitted by the array-creation convenience functions when given legacy kwargs.
"""

from __future__ import annotations

import warnings

import pytest

import zarr
import zarr.api.asynchronous
from zarr.errors import ZarrDeprecationWarning


def test_sync_create_deprecated() -> None:
    """zarr.create emits exactly one ZarrDeprecationWarning pointing at create_array."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        zarr.create(shape=(4,), store={})
    deprecations = [w for w in record if issubclass(w.category, ZarrDeprecationWarning)]
    assert len(deprecations) == 1
    assert "create_array" in str(deprecations[0].message)


async def test_async_create_deprecated() -> None:
    """zarr.api.asynchronous.create emits exactly one ZarrDeprecationWarning."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        await zarr.api.asynchronous.create(shape=(4,), store={})
    deprecations = [w for w in record if issubclass(w.category, ZarrDeprecationWarning)]
    assert len(deprecations) == 1
    assert "create_array" in str(deprecations[0].message)


@pytest.mark.parametrize("name", ["empty", "zeros", "ones", "full", "open_array"])
def test_convenience_fns_no_create_deprecation_on_clean_kwargs(name: str) -> None:
    """Convenience functions must not emit ZarrDeprecationWarning for clean kwargs."""
    fn = getattr(zarr.api.synchronous, name)
    common: dict[str, object] = {"store": {}, "shape": (4,), "chunks": (2,)}
    if name == "full":
        common["fill_value"] = 0
    if name == "open_array":
        # open_array creates on miss; supply a mode that allows creation
        common["mode"] = "w"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrDeprecationWarning)
        fn(**common)  # must not raise


@pytest.mark.parametrize("name", ["empty", "zeros", "ones", "full"])
def test_convenience_fns_warn_on_legacy_kwarg(name: str) -> None:
    """Convenience functions warn when passed a legacy-only kwarg (compressor).

    zarr_format=2 is required here because compressor=None is valid for v2 arrays
    but raises a ValueError for v3 arrays (where bytes-to-bytes codecs replace it).
    The intent is preserved: passing a legacy kwarg triggers ZarrDeprecationWarning.
    """
    fn = getattr(zarr.api.synchronous, name)
    common: dict[str, object] = {
        "store": {},
        "shape": (4,),
        "chunks": (2,),
        "compressor": None,
        "zarr_format": 2,
    }
    if name == "full":
        common["fill_value"] = 0
    with pytest.warns(ZarrDeprecationWarning, match="create_array"):
        fn(**common)


def test_zeros_like_warns_on_legacy_kwarg() -> None:
    """A *_like variant warns transitively (zeros_like -> zeros).

    zarr_format=2 is required both on the base array and the call because
    compressor=None is only valid for v2 arrays.
    """
    base = zarr.create_array(store={}, shape=(4,), chunks=(2,), dtype="i4", zarr_format=2)
    with pytest.warns(ZarrDeprecationWarning, match="create_array"):
        zarr.zeros_like(base, store={}, compressor=None, zarr_format=2)
