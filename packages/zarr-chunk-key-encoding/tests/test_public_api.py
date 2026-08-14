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


def test_builtin_encodings_registered_on_import() -> None:
    """Importing the package registers the two spec-defined encodings."""
    registered = zarr_chunk_key_encoding.registered_chunk_key_encodings()
    assert "default" in registered
    assert "v2" in registered
