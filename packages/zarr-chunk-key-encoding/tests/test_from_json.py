"""Tests for closed-set name dispatch and the JSON/lenient parsing entry points."""

from __future__ import annotations

import operator

import pytest

from zarr_chunk_key_encoding import (
    CHUNK_KEY_ENCODINGS,
    ChunkKeyConfigurationError,
    ChunkKeyEncoding,
    DefaultChunkKeyEncoding,
    UnknownChunkKeyEncodingError,
    V2ChunkKeyEncoding,
    chunk_key_encoding_from_json,
    get_chunk_key_encoding_class,
    parse_chunk_key_encoding,
)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ("default", DefaultChunkKeyEncoding()),
        ("v2", V2ChunkKeyEncoding()),
        ({"name": "default"}, DefaultChunkKeyEncoding()),
        (
            {"name": "default", "configuration": {"separator": "."}},
            DefaultChunkKeyEncoding(separator="."),
        ),
        (
            {"name": "v2", "configuration": {"separator": "/"}},
            V2ChunkKeyEncoding(separator="/"),
        ),
    ],
)
def test_from_json_dispatch(data: object, expected: ChunkKeyEncoding) -> None:
    """`chunk_key_encoding_from_json` resolves the name against the closed set
    and delegates to the encoding's own `from_json`."""
    assert chunk_key_encoding_from_json(data) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (DefaultChunkKeyEncoding(separator="."), DefaultChunkKeyEncoding(separator=".")),
        ("v2", V2ChunkKeyEncoding()),
        ({"name": "default", "separator": "."}, DefaultChunkKeyEncoding(separator=".")),
        ({"name": "v2", "separator": "/"}, V2ChunkKeyEncoding(separator="/")),
        ({"name": "v2"}, V2ChunkKeyEncoding()),
        (
            {"name": "default", "configuration": {"separator": "/"}},
            DefaultChunkKeyEncoding(separator="/"),
        ),
    ],
)
def test_parse_chunk_key_encoding(data: object, expected: ChunkKeyEncoding) -> None:
    """`parse_chunk_key_encoding` additionally accepts instances and the flat
    params form."""
    assert parse_chunk_key_encoding(data) == expected  # type: ignore[arg-type]


def test_known_set_is_the_spec_encodings() -> None:
    """The dispatch table is exactly the two encodings the v3 core spec
    defines, keyed by their metadata names."""
    assert dict(CHUNK_KEY_ENCODINGS) == {
        "default": DefaultChunkKeyEncoding,
        "v2": V2ChunkKeyEncoding,
    }
    assert get_chunk_key_encoding_class("default") is DefaultChunkKeyEncoding
    assert get_chunk_key_encoding_class("v2") is V2ChunkKeyEncoding


def test_known_set_is_immutable() -> None:
    """The closed dispatch table cannot become a process-wide registry at runtime."""
    with pytest.raises(TypeError):
        operator.setitem(CHUNK_KEY_ENCODINGS, "custom", DefaultChunkKeyEncoding)


def test_get_unknown_name() -> None:
    with pytest.raises(UnknownChunkKeyEncodingError, match="not_an_encoding"):
        get_chunk_key_encoding_class("not_an_encoding")


def test_from_json_unknown_name() -> None:
    with pytest.raises(UnknownChunkKeyEncodingError, match="not_an_encoding"):
        chunk_key_encoding_from_json({"name": "not_an_encoding"})


def test_from_json_missing_name() -> None:
    with pytest.raises(ChunkKeyConfigurationError, match="'name'"):
        chunk_key_encoding_from_json({"configuration": {}})


def test_from_json_invalid_type() -> None:
    with pytest.raises(ChunkKeyConfigurationError, match="Invalid chunk key encoding metadata"):
        chunk_key_encoding_from_json(3)  # type: ignore[arg-type]


def test_parse_invalid_type() -> None:
    with pytest.raises(ChunkKeyConfigurationError, match="Invalid chunk key encoding metadata"):
        parse_chunk_key_encoding(3)  # type: ignore[arg-type]
