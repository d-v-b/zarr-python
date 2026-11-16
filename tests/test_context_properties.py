"""Properties every route that resolves array metadata under a `Context` must share.

A document's names resolve under a context along several routes: read as a document,
opened from a store, bound from a mapping, entered as metadata already resolved
(the constructor, `with_context`), and declared by a builder. Two properties hold for
any document and any context:

- route parity: every route reaches the same outcome, the same classes or the same
  kind of error;
- read-back: what a builder declares under a context, stored and opened again under
  that context, holds the same classes and the same data, and metadata a context
  resolved can be declared again under it.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import example, given

from zarr import ArrayIO
from zarr.codecs import BytesCodec, GzipCodec, ShardingCodec, TransposeCodec, ZstdCodec
from zarr.codecs.numcodecs import Shuffle
from zarr.core.buffer import default_buffer_prototype
from zarr.core.dtype import UInt8, UInt16, data_type_registry, parse_dtype
from zarr.core.dtype.registry import DataTypeRegistry
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.errors import DataTypeValidationError, UnknownCodecError
from zarr.experimental import Context
from zarr.storage import MemoryStore, StorePath

if TYPE_CHECKING:
    from zarr.abc.codec import Codec
    from zarr.core.common import JSON, ZarrFormat
    from zarr.core.dtype import ZDType
    from zarr.core.dtype.common import DTypeJSON


class _Renames:
    """Writes its codec under a name of its own (`x.` and the name it reads), and reads
    either name."""

    def to_dict(self) -> dict[str, JSON]:
        written: dict[str, JSON] = super().to_dict()  # type: ignore[misc]
        return {**written, "name": f"x.{written['name']}"}

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Codec:
        name = cast("str", data["name"]).removeprefix("x.")
        return super().from_dict({**data, "name": name})  # type: ignore[misc,no-any-return]


class _RenamesUnreadably(_Renames):
    """Writes its codec under a name of its own that its `from_dict` refuses."""

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Codec:
        if cast("str", data["name"]).startswith("x."):
            raise ValueError(f"{cls.__name__} reads no name {data['name']!r}")
        return super().from_dict(data)


class _BuildsBase:
    """A `from_dict` that builds the base class it derives from, not its own."""

    _base: type[Codec]

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Codec:
        return cls._base.from_dict(data)


class _Refuses:
    """A `from_dict` that refuses every configuration."""

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Codec:
        raise TypeError(f"{cls.__name__} refuses {data!r}")


_DEFAULT_CODECS: dict[str, type[Codec]] = {
    "bytes": BytesCodec,
    "zstd": ZstdCodec,
    "gzip": GzipCodec,
    "transpose": TransposeCodec,
    "numcodecs.shuffle": Shuffle,
    "sharding_indexed": ShardingCodec,
}


def _variants(name: str, default: type[Codec]) -> dict[str, dict[str, type[Codec]]]:
    """The ways a codec scope can hold `name`: the scope entries for each."""
    stem = default.__name__
    renaming: type[Codec] = type(f"Renaming{stem}", (_Renames, default), {})
    unreadable: type[Codec] = type(f"UnreadablyRenaming{stem}", (_RenamesUnreadably, default), {})
    return {
        "default": {name: default},
        "subclass": {name: type(f"Sub{stem}", (default,), {})},
        "renaming": {name: renaming},
        "renaming listed under both names": {name: renaming, f"x.{name}": renaming},
        "renaming unreadably, listed under both names": {name: unreadable, f"x.{name}": unreadable},
        "base-building": {
            name: type(f"BaseBuilding{stem}", (_BuildsBase, default), {"_base": default})
        },
        "refusing": {name: type(f"Refusing{stem}", (_Refuses, default), {})},
        "absent": {},
    }


_CODEC_VARIANTS: dict[str, dict[str, dict[str, type[Codec]]]] = {
    name: _variants(name, default) for name, default in _DEFAULT_CODECS.items()
}


class _WritesXName:
    """A data type that reads its name or `x.` and its name, and writes the latter."""

    _zarr_v3_name: ClassVar[str]

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> Self:
        if data in (cls._zarr_v3_name, f"x.{cls._zarr_v3_name}"):
            return cls()
        raise DataTypeValidationError(f"{cls.__name__} reads no {data!r}")

    def to_json(self, zarr_format: ZarrFormat) -> DTypeJSON:
        if zarr_format == 3:
            return f"x.{self._zarr_v3_name}"
        return super().to_json(zarr_format)  # type: ignore[misc,no-any-return]


class _ReadsOnlyXName:
    """A data type that reads only `x.` and its name."""

    _zarr_v3_name: ClassVar[str]

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> Self:
        if data == f"x.{cls._zarr_v3_name}":
            return cls()
        raise DataTypeValidationError(f"{cls.__name__} reads no {data!r}")


_DATA_TYPES: dict[str, type[ZDType[Any, Any]]] = {"uint8": UInt8, "uint16": UInt16}
_SUBCLASSED_DATA_TYPES: dict[str, type[ZDType[Any, Any]]] = {
    name: type(f"Sub{cls.__name__}", (cls,), {}) for name, cls in _DATA_TYPES.items()
}
_X_WRITING_DATA_TYPES: dict[str, type[ZDType[Any, Any]]] = {
    name: type(f"XWriting{cls.__name__}", (_WritesXName, cls), {})
    for name, cls in _DATA_TYPES.items()
}
_X_READING_DATA_TYPES: dict[str, type[ZDType[Any, Any]]] = {
    name: type(f"XReading{cls.__name__}", (_ReadsOnlyXName, cls), {})
    for name, cls in _DATA_TYPES.items()
}

_CHAINS: tuple[tuple[str, ...], ...] = (
    ("bytes",),
    ("bytes", "zstd"),
    ("transpose", "bytes", "gzip"),
    ("bytes", "numcodecs.shuffle"),
    ("sharding_indexed",),
)


def _codec_json(name: str) -> dict[str, JSON]:
    configurations: dict[str, dict[str, JSON]] = {
        "bytes": {"endian": "little"},
        "zstd": {"level": 0, "checksum": False},
        "gzip": {"level": 1},
        "transpose": {"order": [0]},
        "numcodecs.shuffle": {},
        "sharding_indexed": {
            "chunk_shape": [2],
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "index_codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "crc32c"},
            ],
            "index_location": "end",
        },
    }
    return {"name": name, "configuration": configurations[name]}


def _document(data_type: str, chain: tuple[str, ...]) -> dict[str, JSON]:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": data_type,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [_codec_json(name) for name in chain],
    }


def _data_type_registry(data_type: str, variant: str) -> DataTypeRegistry:
    data_type_registry._lazy_load()
    registry = DataTypeRegistry()
    for name, cls in data_type_registry.contents.items():
        if name != data_type:
            registry.register(name, cls)
    if variant == "default":
        registry.register(data_type, _DATA_TYPES[data_type])
    elif variant == "subclass":
        registry.register(data_type, _SUBCLASSED_DATA_TYPES[data_type])
    elif variant == "writing a name another class reads":
        # What it writes, another class in the scope reads: the scope cannot read back.
        registry.register(f"x.{data_type}", _X_READING_DATA_TYPES[data_type])
        registry.register(data_type, _X_WRITING_DATA_TYPES[data_type])
    return registry


@st.composite
def _cases(draw: st.DrawFn) -> tuple[dict[str, JSON], Context]:
    """A document and a context to resolve it in."""
    data_type = draw(st.sampled_from(sorted(_DATA_TYPES)))
    chain = draw(st.sampled_from(_CHAINS))
    codecs: dict[str, type[Codec]] | None = None
    if draw(st.booleans()):
        codecs = {}
        for name in chain:
            variant = draw(st.sampled_from(sorted(_CODEC_VARIANTS[name])), label=f"codec {name}")
            codecs.update(_CODEC_VARIANTS[name][variant])
    data_types = None
    data_type_variant = draw(
        st.sampled_from(
            ("none", "default", "subclass", "absent", "writing a name another class reads")
        )
    )
    if data_type_variant != "none":
        data_types = _data_type_registry(data_type, data_type_variant)
    return _document(data_type, chain), Context(data_types=data_types, codecs=codecs)


_RESOLVING_ERRORS = (ValueError, TypeError)
"""What resolving metadata under a context raises when it refuses the metadata (zarr's
own errors are `ValueError`s)."""


def _outcome(read: Any) -> tuple[object, ...]:
    """The classes metadata resolves to, or the kind of error resolving it raises."""
    try:
        metadata = read()
    except _RESOLVING_ERRORS as e:
        return ("error", type(e).__name__)
    data_type = metadata.data_type
    return (
        "ok",
        type(data_type).__name__,
        getattr(data_type, "endianness", None),
        tuple(type(codec).__name__ for codec in metadata.codecs),
    )


def _stored(document: dict[str, JSON]) -> MemoryStore:
    store = MemoryStore()
    buffer = default_buffer_prototype().buffer.from_bytes(json.dumps(document).encode())
    sync(store.set("a/zarr.json", buffer))
    return store


def _build(
    document: dict[str, JSON], context: Context, store: MemoryStore | None = None
) -> ArrayIO:
    return ArrayIO.build_v3(
        MemoryStore() if store is None else store,
        name="a",
        shape=(4,),
        data_type=document["data_type"],  # type: ignore[arg-type]
        chunk_grid=(4,),
        codecs=document["codecs"],  # type: ignore[arg-type]
        context=context,
    )


@given(_cases())
def test_every_route_resolves_a_document_alike(case: tuple[dict[str, JSON], Context]) -> None:
    document, context = case
    legacy = ArrayV3Metadata.from_dict(document)
    location = StorePath(MemoryStore(), "a")
    routes = {
        "document": lambda: ArrayV3Metadata.from_dict(document, context=context),
        "open": lambda: ArrayIO.open(_stored(document), name="a", context=context).metadata_legacy,
        "mapping": lambda: ArrayIO(document, location, context=context).metadata_legacy,  # type: ignore[arg-type]
        "entering": lambda: ArrayIO(legacy, location, context=context).metadata_legacy,
        "with_context": lambda: ArrayIO(legacy, location).with_context(context).metadata_legacy,
        "builder": lambda: _build(document, context).metadata_legacy,
    }
    outcomes = {route: _outcome(read) for route, read in routes.items()}
    assert len(set(outcomes.values())) == 1, outcomes


@given(_cases())
def test_what_a_context_declares_reads_back_under_it(
    case: tuple[dict[str, JSON], Context],
) -> None:
    document, context = case
    try:
        declared = _build(document, context)
        resolved = cast("ArrayV3Metadata", declared.metadata_legacy)
    except _RESOLVING_ERRORS:
        # A context that refuses the document: route parity covers the refusal.
        return
    expected = _outcome(lambda: resolved)
    # Declared again from the classes it resolved to, under the same context.
    again = ArrayIO.build_v3(
        MemoryStore(),
        name="a",
        shape=(4,),
        data_type=resolved.data_type,
        chunk_grid=(4,),
        codecs=list(resolved.codecs),
        context=context,
    )
    assert _outcome(lambda: again.metadata_legacy) == expected
    # Stored, then opened again under the same context: the same classes and data.
    store = MemoryStore()
    stored = _build(document, context, store)
    stored.store_metadata()
    data = np.arange(4, dtype=resolved.data_type.to_native_dtype())
    stored.to_array()[:] = data
    reopened = ArrayIO.open(store, name="a", context=context)
    assert _outcome(lambda: reopened.metadata_legacy) == expected
    np.testing.assert_array_equal(reopened.to_array()[:], data)


# Parked for the public release (see the findings list on this branch): each test below
# fails today and names the finding it pins. `strict` makes a fix announce itself.


class _MarksItsJson(BytesCodec):
    """Reads its input as itself, but the JSON it writes as `_ReadsMarked`."""

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "x.marks", "configuration": {"endian": self.endian, "marked": True}}

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        configuration = dict(cast("dict[str, Any]", data.get("configuration", {})))
        if configuration.pop("marked", False):
            return cast("Self", _ReadsMarked(**configuration))
        return cls(**configuration)


class _ReadsMarked(BytesCodec):
    def to_dict(self) -> dict[str, JSON]:
        return {"name": "x.marks", "configuration": {"endian": self.endian}}


class _WritesAnExtraKey(ZstdCodec):
    """A zstd class that writes a configuration key the default class refuses."""

    def to_dict(self) -> dict[str, JSON]:
        written = super().to_dict()
        configuration = cast("dict[str, JSON]", written["configuration"])
        return {**written, "configuration": {**configuration, "extra": 1}}

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        configuration = cast("dict[str, JSON]", data["configuration"])
        kept = {key: value for key, value in configuration.items() if key != "extra"}
        return super().from_dict({**data, "configuration": kept})


def _instance(name: str, context: Context) -> Codec:
    """An instance of the class `context` maps `name` to, made without that class's
    `from_dict` (as a user holding such an instance has it)."""
    default = _DEFAULT_CODECS[name]
    instance = default.from_dict(_codec_json(name))
    scoped = None if context.codecs is None else context.codecs.get(name)
    if scoped is None or scoped is default:
        return instance
    reclassed = copy.copy(instance)
    object.__setattr__(reclassed, "__class__", scoped)
    return reclassed


def _legacy_over(document: dict[str, JSON], codecs: list[Codec]) -> ArrayV3Metadata:
    """Metadata built with no context over `codecs` as given."""
    return ArrayV3Metadata(
        shape=(4,),
        data_type=_DATA_TYPES[cast("str", document["data_type"])](),
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": [4]}},
        chunk_key_encoding={"name": "default"},
        fill_value=0,
        codecs=codecs,
        attributes={},
        dimension_names=None,
    )


@pytest.mark.xfail(
    strict=True,
    reason="Parked M1: a codec instance already of the scope's class skips the read-back "
    "check, so instance routes admit what document routes refuse.",
)
@example((_document("uint16", ("bytes",)), Context(codecs=_CODEC_VARIANTS["bytes"]["refusing"])))
@given(_cases())
def test_instance_routes_resolve_alike(case: tuple[dict[str, JSON], Context]) -> None:
    """A builder or metadata entering a scope, given instances of the scope's own
    classes, reaches the outcome the document reaches."""
    document, context = case
    expected = _outcome(lambda: ArrayV3Metadata.from_dict(document, context=context))
    chain = cast("list[dict[str, str]]", document["codecs"])
    codecs = [_instance(codec["name"], context) for codec in chain]
    location = StorePath(MemoryStore(), "a")
    routes = {
        "builder": lambda: (
            ArrayIO.build_v3(
                MemoryStore(),
                name="a",
                shape=(4,),
                data_type=document["data_type"],  # type: ignore[arg-type]
                chunk_grid=(4,),
                codecs=codecs,
                context=context,
            ).metadata_legacy
        ),
        "entering": lambda: (
            ArrayIO(_legacy_over(document, codecs), location, context=context).metadata_legacy
        ),
        "with_context": lambda: (
            ArrayIO(_legacy_over(document, codecs), location).with_context(context).metadata_legacy
        ),
    }
    outcomes = {route: _outcome(read) for route, read in routes.items()}
    assert set(outcomes.values()) == {expected}, (expected, outcomes)


@pytest.mark.xfail(
    strict=True,
    reason="Parked L2: leaving a scope raises a bare TypeError where opening the stored "
    "document with no context raises MetadataValidationError.",
)
@example(
    (
        _document("uint16", ("bytes", "zstd")),
        Context(codecs={"bytes": BytesCodec, "zstd": _WritesAnExtraKey}),
    )
)
@given(_cases())
def test_leaving_a_scope_reads_as_opening_with_no_context(
    case: tuple[dict[str, JSON], Context],
) -> None:
    """What a context declared, rebound with no context, reaches the outcome its
    stored document reaches when opened with no context."""
    document, context = case
    store = MemoryStore()
    try:
        declared = _build(document, context, store)
        declared.metadata_legacy  # noqa: B018
    except _RESOLVING_ERRORS:
        return
    declared.store_metadata()
    left = _outcome(lambda: declared.with_context(None).metadata_legacy)
    assert left == _outcome(lambda: ArrayIO.open(store, name="a").metadata_legacy)


@pytest.mark.xfail(
    strict=True,
    reason="Parked L1: the codec read-back checks only that from_dict does not raise, "
    "not which class it builds.",
)
def test_a_codec_whose_json_reads_back_as_another_class_is_refused() -> None:
    document = _document("uint16", ("bytes",))
    document["codecs"] = [{"name": "x.marks", "configuration": {"endian": "little"}}]
    with pytest.raises(UnknownCodecError):
        ArrayV3Metadata.from_dict(document, context=Context(codecs={"x.marks": _MarksItsJson}))


@pytest.mark.xfail(
    strict=True,
    reason="Parked L3: parse_dtype's JSON branch swallows the data type read-back error "
    "and reports a misleading native-dtype match instead.",
)
def test_parse_dtype_reports_a_data_type_scope_that_cannot_read_back() -> None:
    registry = _data_type_registry("uint8", "writing a name another class reads")
    with pytest.raises(ValueError, match="could not read back"):
        parse_dtype("uint8", zarr_format=3, context=Context(data_types=registry))
