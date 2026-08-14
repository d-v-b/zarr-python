"""Behavioral parity with `zarr.core.chunk_key_encodings`.

These tests only run when `zarr` is importable (e.g. via
`just test-parity`, which layers this package over the repo-root
environment); the package itself does not depend on `zarr`.
"""

from __future__ import annotations

import pytest

from zarr_chunk_key_encoding import (
    DefaultChunkKeyEncoding,
    Separator,
    V2ChunkKeyEncoding,
)

zarr_cke = pytest.importorskip("zarr.core.chunk_key_encodings")

COORDS: tuple[tuple[int, ...], ...] = ((), (0,), (1, 23), (0, 0, 0), (7, 8, 9, 10))


@pytest.mark.parametrize("separator", [".", "/"])
@pytest.mark.parametrize("chunk_coords", COORDS)
def test_default_encode_parity(separator: Separator, chunk_coords: tuple[int, ...]) -> None:
    """`default` produces byte-identical keys to zarr-python."""
    theirs = zarr_cke.DefaultChunkKeyEncoding(separator=separator)
    ours = DefaultChunkKeyEncoding(separator=separator)
    assert ours.encode(chunk_coords) == theirs.encode_chunk_key(chunk_coords)


@pytest.mark.parametrize("separator", [".", "/"])
@pytest.mark.parametrize("chunk_coords", COORDS)
def test_v2_encode_parity(separator: Separator, chunk_coords: tuple[int, ...]) -> None:
    """`v2` produces byte-identical keys to zarr-python."""
    theirs = zarr_cke.V2ChunkKeyEncoding(separator=separator)
    ours = V2ChunkKeyEncoding(separator=separator)
    assert ours.encode(chunk_coords) == theirs.encode_chunk_key(chunk_coords)


@pytest.mark.parametrize("separator", [".", "/"])
@pytest.mark.parametrize("chunk_coords", COORDS)
def test_json_parity(separator: Separator, chunk_coords: tuple[int, ...]) -> None:
    """`to_json` matches zarr-python's `to_dict` for both encodings."""
    assert (
        DefaultChunkKeyEncoding(separator=separator).to_json()
        == zarr_cke.DefaultChunkKeyEncoding(separator=separator).to_dict()
    )
    assert (
        V2ChunkKeyEncoding(separator=separator).to_json()
        == zarr_cke.V2ChunkKeyEncoding(separator=separator).to_dict()
    )
