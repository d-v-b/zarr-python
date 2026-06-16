"""Regression tests for chunk key encoding round-trips.

Fixes: DefaultChunkKeyEncoding.decode_chunk_key was broken for non-empty
coordinates because ``chunk_key[1:].split(separator)`` left a leading empty
token (e.g. "/0/1".split("/") == ["", "0", "1"]) causing int("") to raise
ValueError.
"""

from __future__ import annotations

import pytest

from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding, V2ChunkKeyEncoding

# ---------------------------------------------------------------------------
# DefaultChunkKeyEncoding round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("separator", ["/", "."])
@pytest.mark.parametrize(
    "coords",
    [
        (),
        (0,),
        (1,),
        (0, 1),
        (3, 7),
        (0, 0, 0),
        (10, 20, 30),
    ],
)
def test_default_chunk_key_encoding_round_trip(coords: tuple[int, ...], separator: str) -> None:
    """encode_chunk_key followed by decode_chunk_key must return the original coords."""
    enc = DefaultChunkKeyEncoding(separator=separator)  # type: ignore[arg-type]
    encoded = enc.encode_chunk_key(coords)
    decoded = enc.decode_chunk_key(encoded)
    assert decoded == coords, (
        f"Round-trip failed for coords={coords!r}, separator={separator!r}: "
        f"encode produced {encoded!r}, decode produced {decoded!r}"
    )


@pytest.mark.parametrize("separator", ["/", "."])
def test_default_chunk_key_encoding_scalar_is_c(separator: str) -> None:
    """encode_chunk_key(()) must produce the literal string 'c'."""
    enc = DefaultChunkKeyEncoding(separator=separator)  # type: ignore[arg-type]
    assert enc.encode_chunk_key(()) == "c"


def test_default_chunk_key_encoding_decode_slash() -> None:
    """Regression: decode_chunk_key('c/0/1') must return (0, 1), not raise ValueError."""
    enc = DefaultChunkKeyEncoding(separator="/")
    # This was the specific failure case: chunk_key[1:] == "/0/1", and
    # "/0/1".split("/") == ["", "0", "1"] so int("") raised ValueError.
    assert enc.decode_chunk_key("c/0/1") == (0, 1)


def test_default_chunk_key_encoding_decode_dot() -> None:
    """Regression: decode_chunk_key('c.0.1') must return (0, 1) with dot separator."""
    enc = DefaultChunkKeyEncoding(separator=".")
    assert enc.decode_chunk_key("c.0.1") == (0, 1)


# ---------------------------------------------------------------------------
# V2ChunkKeyEncoding round-trip tests (smoke / sanity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("separator", ["/", "."])
@pytest.mark.parametrize(
    "coords",
    [
        (0,),
        (1,),
        (0, 1),
        (3, 7),
        (0, 0, 0),
    ],
)
def test_v2_chunk_key_encoding_round_trip(coords: tuple[int, ...], separator: str) -> None:
    """V2ChunkKeyEncoding encode/decode round-trip for non-empty coords."""
    enc = V2ChunkKeyEncoding(separator=separator)  # type: ignore[arg-type]
    encoded = enc.encode_chunk_key(coords)
    decoded = enc.decode_chunk_key(encoded)
    assert decoded == coords
