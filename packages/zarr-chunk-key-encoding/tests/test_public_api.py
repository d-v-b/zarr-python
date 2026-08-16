"""Tests for the shape of the public API."""

from __future__ import annotations

import zarr_chunk_key_encoding


def test_all_names_resolve() -> None:
    """Every name in `__all__` is an attribute of the package.

    Ordering of `__all__` is owned by ruff (RUF022), so it is not asserted
    here.
    """
    for name in zarr_chunk_key_encoding.__all__:
        assert hasattr(zarr_chunk_key_encoding, name)


def test_no_plugin_surface() -> None:
    """The package exposes no registration or discovery API.

    Chunk key encoding is an extension point, but that machinery is common
    to every v3 extension point and belongs in a shared package. Adding it
    here later is compatible; removing it after release would not be, so
    this pins the absence.
    """
    for name in zarr_chunk_key_encoding.__all__:
        assert "register" not in name
        assert "ENTRY_POINT" not in name
        assert "Support" not in name
