"""Tests for entry-point discovery of third-party chunk key encodings."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

import pytest

import zarr_chunk_key_encoding.registry as registry_module
from zarr_chunk_key_encoding import (
    ChunkKey,
    ChunkKeyEncoding,
    ChunkKeyEncodingSupport,
    ChunkKeyPluginWarning,
    get_chunk_key_encoding_class,
    registered_chunk_key_encodings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from zarr_metadata import JSONValue

    from zarr_chunk_key_encoding import ChunkKeyEncodingJSON


class _Plugin(ChunkKeyEncoding):
    """A stand-in for an encoding supplied by a third-party package."""

    name: ClassVar[str] = "test.plugin"

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


class _FakeEntryPoint:
    """Minimal stand-in for `importlib.metadata.EntryPoint`."""

    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self._value = value

    def load(self) -> object:
        """Return the object, or raise it if it is an exception."""
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, *entries: _FakeEntryPoint) -> None:
    """Make discovery see exactly *entries*.

    Overrides the registry isolation in `conftest`, which otherwise leaves
    discovery finding nothing.
    """
    monkeypatch.setattr(registry_module, "entry_points", lambda group: entries if group else ())


def test_discovery_registers_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed entry point is registered on the first lookup that misses,
    keeps its declared support level, and is not loaded before it is needed."""
    _patch_entry_points(monkeypatch, _FakeEntryPoint("plugin", _Plugin))
    assert _Plugin.name not in registered_chunk_key_encodings()
    assert get_chunk_key_encoding_class(_Plugin.name) is _Plugin
    assert _Plugin.name in registered_chunk_key_encodings()
    assert _Plugin.name in registered_chunk_key_encodings(support=ChunkKeyEncodingSupport.CUSTOM)


def test_entry_point_does_not_displace_explicit_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery never overwrites a name that is already registered."""
    impostor = type(_Plugin)("_Impostor", (_Plugin,), {"name": "default"})
    _patch_entry_points(monkeypatch, _FakeEntryPoint("impostor", impostor))
    # Force discovery via a lookup that misses, then check `default` survived.
    with pytest.raises(Exception, match="Unknown chunk key encoding"):
        get_chunk_key_encoding_class("nonexistent")
    assert get_chunk_key_encoding_class("default").__name__ == "DefaultChunkKeyEncoding"


def test_one_broken_plugin_does_not_strand_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each failure mode is warned and skipped, and a good entry point
    enumerated after a broken one still registers -- a broken third-party
    package cannot break discovery for everyone."""
    not_a_class = _FakeEntryPoint("not_a_class", object())
    raises_on_load = _FakeEntryPoint("raises", ImportError("no such module"))
    claims_core = type(_Plugin)(
        "_ClaimsCore",
        (_Plugin,),
        {"name": "test.claims_core", "support": ChunkKeyEncodingSupport.CORE},
    )
    _patch_entry_points(
        monkeypatch,
        not_a_class,
        raises_on_load,
        _FakeEntryPoint("claims_core", claims_core),
        _FakeEntryPoint("good", _Plugin),
    )
    with pytest.warns(ChunkKeyPluginWarning) as warned:
        assert get_chunk_key_encoding_class(_Plugin.name) is _Plugin
    messages = [str(w.message) for w in warned]
    assert any("not a ChunkKeyEncoding subclass" in m for m in messages)
    assert any("could not be loaded" in m for m in messages)
    assert any("could not be registered" in m for m in messages)
    # The plugin that lied about being core was skipped, not registered.
    assert "test.claims_core" not in registered_chunk_key_encodings()


def test_entry_point_cannot_claim_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery routes through `register_chunk_key_encoding`, so an entry
    point cannot bypass the check that CORE is spec-defined."""
    claims_core = type(_Plugin)(
        "_ClaimsCore",
        (_Plugin,),
        {"name": "test.claims_core", "support": ChunkKeyEncodingSupport.CORE},
    )
    _patch_entry_points(monkeypatch, _FakeEntryPoint("claims_core", claims_core))
    with (
        pytest.warns(ChunkKeyPluginWarning, match="core spec defines only"),
        pytest.raises(Exception, match="Unknown chunk key encoding"),
    ):
        get_chunk_key_encoding_class("test.claims_core")
    assert set(registered_chunk_key_encodings(support=ChunkKeyEncodingSupport.CORE)) == {
        "default",
        "v2",
    }


def test_discovery_runs_at_most_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second miss does not re-scan, so a warning is not repeated forever."""
    calls = 0

    def counting_entry_points(group: str) -> tuple[_FakeEntryPoint, ...]:
        nonlocal calls
        calls += 1
        return ()

    monkeypatch.setattr(registry_module, "entry_points", counting_entry_points)
    for _ in range(3):
        with pytest.raises(Exception, match="Unknown chunk key encoding"):
            get_chunk_key_encoding_class("nonexistent")
    assert calls == 1
