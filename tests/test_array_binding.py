"""Tests for how `AsyncArray` delegates to its private binding: copies, pickles and
subclasses keep the behavior they had when an array's fields were its only state."""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING, Any, SupportsIndex, cast
from unittest import mock

import numpy as np
import pytest

import zarr
import zarr.core.metadata.stored as stored_module
from zarr import AsyncArray
from zarr.api.asynchronous import open_array
from zarr.core.array import create_codec_pipeline
from zarr.core.array_spec import ArrayConfig
from zarr.core.buffer import default_buffer_prototype
from zarr.core.codec_pipeline import BatchedCodecPipeline, FusedCodecPipeline
from zarr.core.group import GroupMetadata
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.io import save_metadata
from zarr.core.metadata.stored import stored_documents
from zarr.core.sync import sync
from zarr.dtype import Int16
from zarr.errors import ZarrUserWarning
from zarr.storage import MemoryStore, StorePath

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr.core.buffer import Buffer, BufferPrototype


class _TaggedAsyncArray(AsyncArray[Any]):
    """An `AsyncArray` subclass with an instance attribute of its own."""

    tag: str

    def __init__(self, metadata: Any, store_path: StorePath, tag: str = "default") -> None:
        super().__init__(metadata, store_path)
        object.__setattr__(self, "tag", tag)


def test_async_array_subclass_state_survives_pickle_and_copy() -> None:
    """A subclass's own instance attributes pickle and copy with it, as they did
    when the dataclass fields were an array's only state."""
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    tagged = _TaggedAsyncArray(created.metadata, created.store_path, tag="mine")
    for restored in (pickle.loads(pickle.dumps(tagged)), copy.copy(tagged), copy.deepcopy(tagged)):
        assert type(restored) is _TaggedAsyncArray
        assert restored.tag == "mine"
        assert restored == tagged
        assert set(vars(restored)) == {"_zarr_binding", "tag"}


async def test_copying_an_array_shares_its_metadata() -> None:
    """A shallow copy shares the metadata object, also for an array opened from a
    document that needs a repair; rebinding one (resize) leaves the other."""
    store = MemoryStore()
    doc: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "int32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [0]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    prototype = default_buffer_prototype()
    await store.set("r/zarr.json", prototype.buffer.from_bytes(json.dumps(doc).encode()))
    with pytest.warns(ZarrUserWarning):
        opened = await open_array(store=store, path="r")
    copied = copy.copy(opened)
    assert copied.metadata is opened.metadata
    assert "_stored_document" not in vars(copied.metadata)
    await opened.resize((8,))
    assert copied.shape == (4,)
    assert opened.shape == (8,)


async def test_writing_valid_metadata_parses_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Storing metadata that cannot move chunks, at a path the store's cache knows
    nothing about, records nothing and parses no document."""
    monkeypatch.setattr(
        stored_module,
        "buffer_to_json_object",
        mock.Mock(side_effect=AssertionError("a written document was parsed")),
    )
    store = MemoryStore()
    arr = zarr.create_array(store, name="a", shape=(4,), dtype="uint8", attributes={"x": 1})
    arr.attrs["y"] = 2
    arr.resize((8,))
    v2 = zarr.create_array(store, name="b", shape=(4,), dtype="uint8", zarr_format=2)
    v2.attrs["y"] = 2
    assert not stored_documents(store).knows("a", 3)


_V2_CHUNK_ZERO: dict[str, Any] = {
    "zarr_format": 2,
    "shape": [0],
    "chunks": [0],
    "dtype": "<i4",
    "fill_value": 0,
    "order": "C",
    "filters": None,
    "compressor": None,
}


@dataclass(frozen=True)
class _Typed(AsyncArray[Any]):
    """A `@dataclass` subclass that annotates a field again and adds one with no default."""

    metadata: ArrayV3Metadata
    label: str

    def __init__(
        self, metadata: Any, store_path: StorePath, config: Any = None, label: str = "x"
    ) -> None:
        super().__init__(metadata, store_path, config)
        object.__setattr__(self, "label", label)


def test_a_dataclass_subclass_may_annotate_fields_again() -> None:
    """A `@dataclass` subclass that annotates a field again, or adds one without a
    default, defines and works, as when the fields had no class attribute."""
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    typed = _Typed(created.metadata, created.store_path)
    assert typed.shape == (4,)
    assert dataclasses.replace(typed, label="y").label == "y"


@dataclass(frozen=True)
class _Tagged(AsyncArray[Any]):
    """A `@dataclass` subclass with a field of its own and the `__init__` it generates."""

    tag: str = "untagged"


def test_a_dataclass_subclass_binds_through_its_generated_init() -> None:
    """The `__init__` a `@dataclass` subclass generates assigns each field in turn,
    without calling `AsyncArray.__init__`; the array binds once its metadata, store path
    and configuration are assigned, and reads, writes, copies and pickles."""
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    tagged = _Tagged(created.metadata, created.store_path, created.config, tag="t")

    zarr.Array(tagged)[...] = np.arange(4)

    for kept in (tagged, copy.copy(tagged), pickle.loads(pickle.dumps(tagged))):
        assert kept.tag == "t"
        np.testing.assert_array_equal(zarr.Array(kept)[...], np.arange(4))
    assert dataclasses.replace(tagged, tag="u").tag == "u"


class _IncrementingAsyncArray(AsyncArray[Any]):
    """A subclass that stores one more than each write gives it, by overriding
    `_set_selection`."""

    async def _set_selection(
        self, indexer: Any, value: Any, *, prototype: Any, fields: Any = None
    ) -> None:
        await super()._set_selection(
            indexer, np.asarray(value) + 1, prototype=prototype, fields=fields
        )


def _write_with_setitem(array: AsyncArray[Any], data: np.ndarray[Any, Any]) -> None:
    sync(array.setitem(slice(None), data))


def _write_with_array_setitem(array: AsyncArray[Any], data: np.ndarray[Any, Any]) -> None:
    zarr.Array(array)[...] = data


def _write_with_orthogonal_selection(array: AsyncArray[Any], data: np.ndarray[Any, Any]) -> None:
    zarr.Array(array).set_orthogonal_selection((slice(None),), data)


@pytest.mark.parametrize(
    "write", [_write_with_setitem, _write_with_array_setitem, _write_with_orthogonal_selection]
)
def test_every_write_goes_through_a_subclass_set_selection(
    write: Callable[[AsyncArray[Any], np.ndarray[Any, Any]], None],
) -> None:
    """Every write of an array goes through `AsyncArray._set_selection`, so a
    subclass that overrides it changes what each of them stores."""
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    array = _IncrementingAsyncArray(created.metadata, created.store_path)

    write(array, np.arange(4, dtype="uint8"))

    np.testing.assert_array_equal(created[...], np.arange(4) + 1)


class _SlottedAsyncArray(AsyncArray[Any]):
    """An `AsyncArray` subclass with a slot of its own."""

    __slots__ = ("__private", "extra")
    extra: int


class _ReducingAsyncArray(AsyncArray[Any]):
    """An `AsyncArray` subclass that decides how it is copied."""

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        return (_rebuilt_marked, (self.metadata, self.store_path))


def _rebuilt_marked(metadata: Any, store_path: StorePath) -> _ReducingAsyncArray:
    rebuilt = _ReducingAsyncArray(metadata, store_path)
    object.__setattr__(rebuilt, "marked", True)
    return rebuilt


def test_subclass_slots_and_reductions_are_honored() -> None:
    """A subclass's slots pickle and copy with it, and a subclass that defines its
    own reduction is copied its way, as when the fields were an array's only state."""
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    slotted = _SlottedAsyncArray(created.metadata, created.store_path)
    object.__setattr__(slotted, "extra", 5)
    object.__setattr__(slotted, "_SlottedAsyncArray__private", 6)
    for copied in (pickle.loads(pickle.dumps(slotted)), copy.copy(slotted), copy.deepcopy(slotted)):
        assert copied.extra == 5
        assert vars(type(copied))["_SlottedAsyncArray__private"].__get__(copied) == 6
    reducing = _ReducingAsyncArray(created.metadata, created.store_path)
    assert vars(copy.copy(reducing))["marked"] is True


def test_fields_read_as_missing_on_the_class() -> None:
    """The fields have no class value, as fields without defaults have none."""
    assert all(getattr(AsyncArray, name) is MISSING for name in ("metadata", "config"))


class _StampingAsyncArray(AsyncArray[Any]):
    """A subclass whose `__init__` changes the metadata it is given."""

    def __init__(self, metadata: Any, store_path: StorePath, config: Any = None) -> None:
        stamped = dataclasses.replace(
            metadata, attributes={**metadata.attributes, "n": metadata.attributes.get("n", 0) + 1}
        )
        super().__init__(stamped, store_path, config)


class _NamedAsyncArray(AsyncArray[Any]):
    """A subclass with attributes named as a binding's own state might be."""


def test_subclass_attributes_keep_names_the_binding_uses() -> None:
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    named = _NamedAsyncArray(created.metadata, created.store_path)
    names = ("_context", "_pipeline_class", "_binding_class", "_model_documents", "_stored_entry")
    for name in names:
        object.__setattr__(named, name, f"mine {name}")
    for restored in (pickle.loads(pickle.dumps(named)), copy.deepcopy(named)):
        assert all(vars(restored)[name] == f"mine {name}" for name in names)


class _OwnReductionAsyncArray(AsyncArray[Any]):
    """A subclass whose reduction carries a state of its own, with no `__setstate__`."""

    def __reduce__(self) -> Any:
        return (_own_reduction_rebuilt, (self.metadata, self.store_path), {"note": "hi"})


def _own_reduction_rebuilt(metadata: Any, store_path: StorePath) -> _OwnReductionAsyncArray:
    return _OwnReductionAsyncArray(metadata, store_path)


def test_a_subclass_reduction_with_its_own_state_copies_and_pickles() -> None:
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    array = _OwnReductionAsyncArray(created.metadata, created.store_path)
    for copied in (copy.copy(array), copy.deepcopy(array), pickle.loads(pickle.dumps(array))):
        assert vars(copied)["note"] == "hi"
        assert copied.shape == (4,)


class _SlotStateReductionAsyncArray(AsyncArray[Any]):
    """A subclass whose reduction carries a state in the default protocol's
    `(None, slots)` form."""

    def __reduce__(self) -> Any:
        return (_slot_state_rebuilt, (self.metadata, self.store_path), (None, {"note": "hi"}))


def _slot_state_rebuilt(metadata: Any, store_path: StorePath) -> _SlotStateReductionAsyncArray:
    return _SlotStateReductionAsyncArray(metadata, store_path)


class _MetadataStateReductionAsyncArray(AsyncArray[Any]):
    """A subclass whose own state holds some of an array's fields, which is not an
    array's field state: it does not hold them all."""

    def __reduce__(self) -> Any:
        return (
            _metadata_state_rebuilt,
            (self.metadata, self.store_path),
            {"metadata": self.metadata, "codec_pipeline": self.codec_pipeline},
        )


def _metadata_state_rebuilt(
    metadata: Any, store_path: StorePath
) -> _MetadataStateReductionAsyncArray:
    return _MetadataStateReductionAsyncArray(metadata, store_path)


def test_a_subclass_reduction_state_is_read_in_every_form() -> None:
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    slotted = _SlotStateReductionAsyncArray(created.metadata, created.store_path)
    for copied in (copy.copy(slotted), copy.deepcopy(slotted), pickle.loads(pickle.dumps(slotted))):
        assert vars(copied)["note"] == "hi"
    holding = _MetadataStateReductionAsyncArray(created.metadata, created.store_path)
    for copied in (copy.copy(holding), copy.deepcopy(holding), pickle.loads(pickle.dumps(holding))):
        assert copied.shape == (4,)


class _AssignedPipeline(BatchedCodecPipeline):
    """A pipeline class a subclass `__init__` assigns."""


class _PipelineAssigningAsyncArray(AsyncArray[Any]):
    """A subclass whose `__init__` assigns a pipeline of its own."""

    def __init__(self, metadata: Any, store_path: StorePath, config: Any = None) -> None:
        super().__init__(metadata, store_path, config)
        pipeline = create_codec_pipeline(self.metadata, pipeline_class=_AssignedPipeline)
        object.__setattr__(self, "codec_pipeline", pipeline)


def test_an_array_follows_the_global_config_when_made() -> None:
    """The pipeline implementation is captured when an array is made, and holds after
    the global configuration changes."""
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"}):
        array = zarr.create_array(MemoryStore(), shape=(2,), dtype="uint8", chunks=(2,))
    assert isinstance(array.async_array.codec_pipeline, FusedCodecPipeline)
    assert isinstance(
        array.with_config({"order": "F"}).async_array.codec_pipeline, BatchedCodecPipeline
    )


def test_zero_chunk_v2_metadata_binds_repaired() -> None:
    """`ArrayV2Metadata` with a chunk size of 0 is read as repaired when an array is
    made over it, as `parse_array_metadata` reads it."""
    metadata = ArrayV2Metadata(shape=(3,), chunks=(0,), dtype=Int16(), fill_value=0, order="C")
    assert AsyncArray(metadata, StorePath(MemoryStore())).metadata.chunks == (1,)


def test_a_0d_array_reads_empty_dimension_names_as_none() -> None:
    created = zarr.create_array(MemoryStore(), shape=(), dtype="int8", dimension_names=[])
    assert created.metadata.dimension_names is None  # type: ignore[union-attr]


async def test_an_array_copied_by_its_dict_keeps_its_own_state() -> None:
    """An array made by copying another's `__dict__` (as subclasses do) is changed
    apart from it: a resize, an attribute update or a field set through either
    leaves the other, as when its fields were its only state."""
    original = zarr.create_array(MemoryStore(), shape=(4,), dtype="int32", chunks=(2,))._async_array
    clone = object.__new__(AsyncArray)
    clone.__dict__.update(original.__dict__)
    await clone.resize((6,))
    assert (original.shape, clone.shape) == ((4,), (6,))
    await clone.update_attributes({"x": 1})
    assert original.attrs == {}
    object.__setattr__(clone, "metadata", clone.metadata.update_shape((3,)))
    assert original.shape == (4,)


def test_a_tuple_valued_v2_chunk_size_zero_binds_as_repaired() -> None:
    """A metadata mapping written in Python (tuples, not lists) is read with a chunk
    size of 0 as repaired."""
    document = {**_V2_CHUNK_ZERO, "shape": (0,), "chunks": (0,)}
    array = AsyncArray(cast("Any", document), StorePath(MemoryStore(), "a"))
    assert array.metadata.chunks == (1,)


async def test_attribute_updates_show_through_an_array_copied_by_its_dict() -> None:
    """An attribute update through an array changes its metadata in place, so it shows
    through an array copied by its `__dict__`, as it did when the fields were an
    array's only state."""
    array = zarr.create_array(
        MemoryStore(), shape=(4,), dtype="uint8", chunks=(4,), attributes={"v": 0}
    )._async_array
    clone = object.__new__(AsyncArray)
    clone.__dict__.update(array.__dict__)
    await array.update_attributes({"v": 1})
    assert clone.attrs == {"v": 1}


def test_with_config_keeps_what_a_subclass_init_built() -> None:
    """`with_config` rebuilds through the constructor and keeps the metadata a
    subclass `__init__` built, as it always has."""
    store = MemoryStore()
    zarr.create_array(store, name="a", shape=(4,), dtype="uint8", chunks=(4,))
    array = _StampingAsyncArray(zarr.open_array(store, path="a").metadata, StorePath(store, "a"))
    before = cast("int", array.attrs.get("n", 0))
    assert array.with_config({"order": "F"}).attrs["n"] == before + 1


def test_with_config_keeps_a_pipeline_a_subclass_init_assigned() -> None:
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    assigned = _PipelineAssigningAsyncArray(created.metadata, created.store_path)
    assert isinstance(assigned.with_config({"order": "F"}).codec_pipeline, _AssignedPipeline)


class _EmptyChunkWritingAsyncArray(AsyncArray[Any]):
    """A subclass whose `config` field writes empty chunks, whatever it was given."""

    @property
    def config(self) -> ArrayConfig:
        return dataclasses.replace(self._zarr_binding.config, write_empty_chunks=True)


def test_reads_and_writes_go_through_a_subclass_field() -> None:
    """An array reads and writes through its fields, so a subclass that overrides one
    (here `config`) is read through it, as it always was."""
    store = MemoryStore()
    created = zarr.create_array(store, shape=(4,), dtype="uint8", chunks=(2,), fill_value=0)
    array = zarr.Array(_EmptyChunkWritingAsyncArray(created.metadata, created.store_path))
    array[:] = 0
    assert sorted(sync(_store_keys(store))) == ["c/0", "c/1", "zarr.json"]


class _ReadsBackBeforeBinding(AsyncArray[Any]):
    """A subclass `__init__` that sets the fields itself and reads one back before the
    others are set."""

    def __init__(self, metadata: Any, store_path: StorePath) -> None:
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "shape_seen", self.metadata.shape)
        object.__setattr__(self, "store_path", store_path)
        object.__setattr__(self, "config", ArrayConfig.from_dict({}))


def test_a_field_reads_back_before_the_array_is_bound() -> None:
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    array = _ReadsBackBeforeBinding(created.metadata, created.store_path)
    assert vars(array)["shape_seen"] == (4,)
    assert array.shape == (4,)


class _OwnsAnIoAttribute(AsyncArray[Any]):
    """A subclass with an attribute of its own named `_io`."""

    def __init__(self, metadata: Any, store_path: StorePath) -> None:
        super().__init__(metadata, store_path)
        object.__setattr__(self, "_io", "the subclass's own")


def test_a_subclass_attribute_named_io_leaves_the_array_working() -> None:
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    array = _OwnsAnIoAttribute(created.metadata, created.store_path)
    assert array.shape == (4,)
    assert vars(array)["_io"] == "the subclass's own"


class _OldSignatureGroupMetadata(GroupMetadata):
    """A group metadata subclass overriding `to_buffer_dict` with its old signature."""

    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:  # type: ignore[override]
        return super().to_buffer_dict(prototype)


def test_a_group_metadata_subclass_with_the_old_signature_stores() -> None:
    store = MemoryStore()
    group = zarr.create_group(store)
    sync(save_metadata(group.store_path, _OldSignatureGroupMetadata(attributes={"a": 1})))
    assert zarr.open_group(store).attrs == {"a": 1}


async def _store_keys(store: MemoryStore) -> list[str]:
    return [key async for key in store.list()]
