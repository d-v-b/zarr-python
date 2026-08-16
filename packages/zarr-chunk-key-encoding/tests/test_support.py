"""Tests for chunk key encoding support levels."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

import pytest

from zarr_chunk_key_encoding import (
    CORE_CHUNK_KEY_ENCODING_NAMES,
    ChunkKey,
    ChunkKeyEncoding,
    ChunkKeyEncodingSupport,
    ChunkKeyRegistryError,
    DefaultChunkKeyEncoding,
    V2ChunkKeyEncoding,
    get_chunk_key_encoding_support,
    register_chunk_key_encoding,
    registered_chunk_key_encodings,
    unregister_chunk_key_encoding,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from zarr_metadata import JSONValue

    from zarr_chunk_key_encoding import ChunkKeyEncodingJSON


class _Stub(ChunkKeyEncoding):
    """A minimal encoding used to exercise registration."""

    name: ClassVar[str] = "test.stub"

    @classmethod
    def from_json(cls, data: ChunkKeyEncodingJSON) -> Self:
        """Construct without inspecting the metadata."""
        return cls()

    def to_json(self) -> Mapping[str, JSONValue]:
        """Return the name-only object form."""
        return {"name": self.name}

    def encode(self, chunk_coords: Sequence[int]) -> ChunkKey:
        """Join coordinates with `/`."""
        return ChunkKey("/".join(str(c) for c in chunk_coords))


@pytest.fixture
def stub_cls() -> type[_Stub]:
    """A fresh subclass per test, so registrations never leak between tests."""
    return type(_Stub)("_StubCopy", (_Stub,), {})


def test_builtins_are_core() -> None:
    """The two spec-defined encodings declare and report CORE, and the core
    name set matches what they register under."""
    assert DefaultChunkKeyEncoding.support is ChunkKeyEncodingSupport.CORE
    assert V2ChunkKeyEncoding.support is ChunkKeyEncodingSupport.CORE
    assert get_chunk_key_encoding_support("default") is ChunkKeyEncodingSupport.CORE
    assert get_chunk_key_encoding_support("v2") is ChunkKeyEncodingSupport.CORE
    assert set(CORE_CHUNK_KEY_ENCODING_NAMES) == {"default", "v2"}


def test_default_support_is_custom(stub_cls: type[_Stub]) -> None:
    """An encoding that declares nothing is CUSTOM, the safe assumption."""
    assert stub_cls.support is ChunkKeyEncodingSupport.CUSTOM


@pytest.mark.parametrize(
    "support", [ChunkKeyEncodingSupport.EXTENSION, ChunkKeyEncodingSupport.CUSTOM]
)
def test_registration_round_trip(stub_cls: type[_Stub], support: ChunkKeyEncodingSupport) -> None:
    """A non-core encoding registers at its declared level, is reported back,
    and appears in exactly that level's filtered listing."""
    stub_cls.support = support
    register_chunk_key_encoding(stub_cls)
    try:
        assert get_chunk_key_encoding_support(stub_cls.name) is support
        assert stub_cls.name in registered_chunk_key_encodings(support=support)
        assert stub_cls.name not in registered_chunk_key_encodings(
            support=ChunkKeyEncodingSupport.CORE
        )
        assert stub_cls.name in registered_chunk_key_encodings()
    finally:
        unregister_chunk_key_encoding(stub_cls.name)


def test_core_filter_is_the_spec_set() -> None:
    """Filtering on CORE yields exactly the spec-defined names, which is what
    makes it usable as a gating allowlist."""
    assert set(registered_chunk_key_encodings(support=ChunkKeyEncodingSupport.CORE)) == (
        CORE_CHUNK_KEY_ENCODING_NAMES
    )


def test_support_compares_as_string() -> None:
    """Levels are a StrEnum, so a value read from configuration works directly."""
    assert ChunkKeyEncodingSupport.CORE == "core"
    assert ChunkKeyEncodingSupport("extension") is ChunkKeyEncodingSupport.EXTENSION


def test_falsely_claiming_core_is_rejected(stub_cls: type[_Stub]) -> None:
    """A non-spec name may not register as CORE, so the tier cannot be
    self-asserted by the code a consumer is trying to gate."""
    stub_cls.support = ChunkKeyEncodingSupport.CORE
    with pytest.raises(ChunkKeyRegistryError, match="core spec defines only"):
        register_chunk_key_encoding(stub_cls)
    assert stub_cls.name not in registered_chunk_key_encodings()
