"""Tests for the registry and the JSON/lenient parsing entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Self

    from zarr_metadata import ZarrV3NamedConfigJSON

from zarr_chunk_key_encoding import (
    ChunkKeyConfigurationError,
    ChunkKeyEncoding,
    ChunkKeyEncodingJSON,
    ChunkKeyRegistryError,
    DefaultChunkKeyEncoding,
    UnknownChunkKeyEncodingError,
    V2ChunkKeyEncoding,
    chunk_key_encoding_from_json,
    get_chunk_key_encoding_class,
    parse_chunk_key_encoding,
    register_chunk_key_encoding,
    registered_chunk_key_encodings,
    unregister_chunk_key_encoding,
)


@dataclass(frozen=True)
class SuffixEncoding(ChunkKeyEncoding):
    """A minimal third-party-style encoding used to exercise the registry."""

    name: ClassVar[str] = "test_suffix"
    suffix: str = ".bin"

    @classmethod
    def from_json(cls, data: ChunkKeyEncodingJSON) -> Self:
        assert isinstance(data, dict)
        configuration = data.get("configuration", {})
        assert isinstance(configuration, dict)
        return cls(**configuration)

    def to_json(self) -> ZarrV3NamedConfigJSON:
        return {"name": self.name, "configuration": {"suffix": self.suffix}}

    def encode(self, chunk_coords: Sequence[int]) -> str:
        return "/".join(map(str, chunk_coords)) + self.suffix


@pytest.fixture
def suffix_encoding_registered() -> Iterator[None]:
    register_chunk_key_encoding(SuffixEncoding)
    yield
    unregister_chunk_key_encoding(SuffixEncoding.name)


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
    """`chunk_key_encoding_from_json` resolves the name via the registry and
    delegates to the encoding's own `from_json`."""
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


def test_register_get_unregister(suffix_encoding_registered: None) -> None:
    """A registered encoding is retrievable by name, listed, constructible
    from JSON, and re-registering the same class is a no-op."""
    assert get_chunk_key_encoding_class("test_suffix") is SuffixEncoding
    assert "test_suffix" in registered_chunk_key_encodings()
    constructed = chunk_key_encoding_from_json(
        {"name": "test_suffix", "configuration": {"suffix": ".raw"}}
    )
    assert constructed == SuffixEncoding(suffix=".raw")
    register_chunk_key_encoding(SuffixEncoding)  # idempotent


def test_register_conflict_requires_overwrite(suffix_encoding_registered: None) -> None:
    @dataclass(frozen=True)
    class Impostor(SuffixEncoding):
        name: ClassVar[str] = "test_suffix"

    with pytest.raises(ChunkKeyRegistryError, match="already registered"):
        register_chunk_key_encoding(Impostor)
    register_chunk_key_encoding(Impostor, overwrite=True)
    assert get_chunk_key_encoding_class("test_suffix") is Impostor
    register_chunk_key_encoding(SuffixEncoding, overwrite=True)


def test_get_unknown_name() -> None:
    with pytest.raises(UnknownChunkKeyEncodingError, match="not_an_encoding"):
        get_chunk_key_encoding_class("not_an_encoding")


def test_unregister_unknown_name() -> None:
    with pytest.raises(UnknownChunkKeyEncodingError, match="not_an_encoding"):
        unregister_chunk_key_encoding("not_an_encoding")


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
