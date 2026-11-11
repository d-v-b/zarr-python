"""Tests for the explicit-IO array layer, `zarr.ArrayIO` and `zarr.AsyncArrayIO`."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import pickle
import threading
import warnings
from dataclasses import MISSING, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, SupportsIndex, cast
from unittest import mock

import numpy as np
import pytest
from zarr_metadata import UNSET, MetadataValidationError, ZarrV3ArrayMetadata
from zarr_metadata import ZarrV2ArrayMetadata as ModelV2

import zarr
import zarr.core.context as context_module
import zarr.core.metadata.stored as stored_module
from zarr import ArrayIO, AsyncArray, AsyncArrayIO
from zarr.api.asynchronous import open_array
from zarr.codecs import BloscCodec, BytesCodec, GzipCodec, ShardingCodec, ZstdCodec
from zarr.codecs.numcodecs import Shuffle
from zarr.core.array import build_array_metadata_v3, create_codec_pipeline, parse_array_metadata
from zarr.core.array_spec import parse_array_config
from zarr.core.buffer import default_buffer_prototype
from zarr.core.codec_pipeline import BatchedCodecPipeline, FusedCodecPipeline
from zarr.core.dtype import (
    Float32,
    Int8,
    NullTerminatedBytes,
    Struct,
    UInt16,
    VariableLengthUTF8,
    data_type_registry,
    parse_dtype,
)
from zarr.core.dtype.registry import DataTypeRegistry
from zarr.core.metadata import ArrayMetadata, ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.stored import stored_documents
from zarr.core.sync import sync
from zarr.dtype import Int16
from zarr.errors import (
    ArrayNotFoundError,
    UnknownCodecError,
    UnstableSpecificationWarning,
    ZarrUserWarning,
)
from zarr.errors import MetadataValidationError as ZarrMetadataValidationError
from zarr.experimental import Context
from zarr.storage import LocalStore, MemoryStore, StorePath

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path
    from typing import Self

    from zarr.core.array_spec import ArrayConfigLike
    from zarr.core.common import JSON, ZarrFormat

METADATA_DOCUMENTS = ("zarr.json", ".zarray", ".zattrs", ".zgroup")


async def _store_keys(store: MemoryStore) -> set[str]:
    return {key async for key in store.list()}


def _declare_answer_array(
    zarr_format: ZarrFormat, store: MemoryStore, data: np.ndarray[Any, np.dtype[Any]]
) -> AsyncArrayIO:
    """Declare the combo test's array with the builder of `zarr_format`."""
    if zarr_format == 2:
        return AsyncArrayIO.build_v2(
            store,
            name="foo",
            shape=data.shape,
            dtype=data.dtype,
            chunks=(2, 2),
            attributes={"answer": 42},
        )
    return AsyncArrayIO.build_v3(
        store,
        name="foo",
        shape=data.shape,
        data_type=data.dtype,
        chunk_grid=(2, 2),
        attributes={"answer": 42},
    )


async def _store_foreign_codec_array(store: MemoryStore) -> None:
    """Store a valid v3 array document whose codec this environment lacks."""
    document = (
        b'{"zarr_format": 3, "node_type": "array", "shape": [4], "data_type": "uint8",'
        b' "fill_value": 0, "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},'
        b' "chunk_key_encoding": {"name": "default"},'
        b' "codecs": [{"name": "zarr-python-does-not-know-this-codec"}]}'
    )
    prototype = default_buffer_prototype()
    await store.set("foo/zarr.json", prototype.buffer.from_bytes(document))
    await store.set("foo/c/0", prototype.buffer.from_bytes(b"xx"))
    await store.set("foo/c/1", prototype.buffer.from_bytes(b"yy"))


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_chunks_before_metadata(zarr_format: ZarrFormat) -> None:
    """Declaring an array stores nothing; chunks, then metadata, can be stored
    in that order, and the result is an ordinary readable array."""
    store = MemoryStore()
    data = np.arange(16, dtype="uint16").reshape(4, 4)
    aio = _declare_answer_array(zarr_format, store, data)
    assert await _store_keys(store) == set()

    await aio.setitem(Ellipsis, data)
    keys = await _store_keys(store)
    assert len(keys) == 4
    assert not any(key.split("/")[-1] in METADATA_DOCUMENTS for key in keys)
    np.testing.assert_array_equal(await aio.getitem(Ellipsis), data)

    await aio.store_metadata()
    # the default zarr_format=None auto-detects the format, like zarrs
    opened = await AsyncArrayIO.open(store, name="foo")
    # the binding is metadata; the stored and declared metadata agree
    assert opened.metadata == aio.metadata
    assert opened.metadata.shape == data.shape
    assert opened.metadata_legacy.zarr_format == zarr_format
    assert dict(opened.metadata.attributes) == {"answer": 42}
    np.testing.assert_array_equal(await opened.getitem(Ellipsis), data)

    arr = await open_array(store=store, path="foo", mode="r", zarr_format=zarr_format)
    np.testing.assert_array_equal(await arr.getitem(Ellipsis), data)
    # every AsyncArray is built on an AsyncArrayIO, reachable as .io
    assert arr.io.metadata_legacy is arr.metadata
    np.testing.assert_array_equal(await arr.io.getitem(Ellipsis), data)

    # a writer that declared the same array independently can verify its assumption
    worker = _declare_answer_array(zarr_format, store, data)
    await worker.verify()

    # every write has an erasing inverse: one chunk, all chunks, then metadata
    await opened.erase_chunk((0, 0))
    partially_erased = np.asarray(await opened.getitem(Ellipsis))
    np.testing.assert_array_equal(partially_erased[:2, :2], np.zeros((2, 2), dtype="uint16"))
    np.testing.assert_array_equal(partially_erased[2:, :], data[2:, :])

    await opened.erase_chunks()
    await opened.erase_chunks()  # erasing absent chunks succeeds
    np.testing.assert_array_equal(await opened.getitem(Ellipsis), np.zeros_like(data))
    keys = await _store_keys(store)
    assert all(key.split("/")[-1] in METADATA_DOCUMENTS for key in keys)

    await opened.erase_metadata()
    await opened.erase_metadata()  # erasing absent metadata succeeds
    assert await _store_keys(store) == set()
    with pytest.raises(ArrayNotFoundError):
        await AsyncArrayIO.open(store, name="foo")


def test_chunks_before_metadata_sync() -> None:
    """The synchronous `ArrayIO` supports the same chunks-first flow."""
    store = MemoryStore()
    data = np.arange(16, dtype="uint16").reshape(4, 4)
    aio = ArrayIO.build_v3(
        store, name="foo", shape=data.shape, data_type=data.dtype, chunk_grid=(2, 2)
    )
    aio.setitem(Ellipsis, data)
    aio.store_metadata()
    opened = ArrayIO.open(store, name="foo")
    np.testing.assert_array_equal(opened.getitem(Ellipsis), data)
    np.testing.assert_array_equal(opened.to_array()[:], data)
    # every Array is built on the IO layer, reachable as .io
    np.testing.assert_array_equal(opened.to_array().io.getitem(Ellipsis), data)
    opened.erase_chunk((0, 0))
    opened.erase_chunks()
    opened.erase_metadata()
    with pytest.raises(ArrayNotFoundError):
        ArrayIO.open(store, name="foo")


def test_build_v3_unknown_chunk_grid_raises() -> None:
    with pytest.raises(ValueError, match="chunk grid"):
        ArrayIO.build_v3(
            MemoryStore(),
            name="foo",
            shape=(4,),
            data_type="uint8",
            chunk_grid={"name": "diagonal", "configuration": {}},
        )


async def test_open_missing_raises() -> None:
    with pytest.raises(ArrayNotFoundError):
        await AsyncArrayIO.open(MemoryStore(), name="foo")


async def test_document_work_without_codec_resolution() -> None:
    """An array whose codec this environment lacks still opens for document
    work — reading the document, verifying it, erasing chunks and metadata —
    and refuses only when the data path must be built."""
    store = MemoryStore()
    await _store_foreign_codec_array(store)

    aio = await AsyncArrayIO.open(store, name="foo")
    model = aio.metadata
    assert isinstance(model, ZarrV3ArrayMetadata)
    assert model.shape == (4,)
    assert model.must_understand_fields == {}
    await aio.verify()

    with pytest.raises(UnknownCodecError, match="zarr-python-does-not-know-this-codec"):
        await aio.getitem(Ellipsis)

    await aio.erase_chunk((0,))
    await aio.erase_chunks()
    await aio.store_metadata()
    await aio.erase_metadata()
    assert await _store_keys(store) == set()


async def test_verify_mismatch_raises() -> None:
    store = MemoryStore()
    aio = AsyncArrayIO.build_v3(store, shape=(4, 4), data_type="uint8", chunk_grid=(2, 2))
    await aio.store_metadata()
    other = AsyncArrayIO.build_v3(
        store, shape=(4, 4), data_type="uint8", chunk_grid=(4, 4), attributes={"a": 1}
    )
    with pytest.raises(ValueError, match="differing fields: attributes, chunk_grid"):
        await other.verify()


async def test_verify_format_mismatch_raises() -> None:
    store = MemoryStore()
    await AsyncArrayIO.build_v2(store, shape=(4,), dtype="uint8", chunks=(4,)).store_metadata()
    bound_v3 = AsyncArrayIO.build_v3(store, shape=(4,), data_type="uint8", chunk_grid=(4,))
    with pytest.raises(ValueError, match="stored array is Zarr format 2"):
        await bound_v3.verify()


async def test_verify_missing_raises() -> None:
    aio = AsyncArrayIO.build_v3(MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,))
    with pytest.raises(ArrayNotFoundError):
        await aio.verify()


async def test_erase_chunk_out_of_grid_raises() -> None:
    aio = AsyncArrayIO.build_v3(MemoryStore(), shape=(4, 4), data_type="uint8", chunk_grid=(2, 2))
    with pytest.raises(IndexError, match="outside the chunk grid"):
        await aio.erase_chunk((2, 0))
    with pytest.raises(IndexError, match="outside the chunk grid"):
        await aio.erase_chunk((0,))


async def test_with_methods_rebind() -> None:
    """with_metadata, with_store and with_config return new bindings with one
    part replaced and everything else kept."""
    store, other_store = MemoryStore(), MemoryStore()
    data = np.arange(4, dtype="uint8")
    aio = AsyncArrayIO.build_v3(store, name="foo", shape=(4,), data_type="uint8", chunk_grid=(2,))
    await aio.setitem(Ellipsis, data)
    await aio.store_metadata()

    relabeled = aio.with_metadata(aio.metadata.update(attributes={"a": 1}))
    assert relabeled.store_path == aio.store_path
    assert dict(relabeled.metadata.attributes) == {"a": 1}
    await relabeled.store_metadata()
    reread = await AsyncArrayIO.open(store, name="foo")
    assert reread.metadata == relabeled.metadata

    moved = aio.with_store(other_store)
    assert moved.store_path.path == aio.store_path.path
    await moved.setitem(Ellipsis, data)
    await moved.store_metadata()
    copied = await AsyncArrayIO.open(other_store, name="foo")
    np.testing.assert_array_equal(await copied.getitem(Ellipsis), data)

    configured = aio.with_config({"order": "F"})
    assert configured.config.order == "F"
    assert configured.metadata == aio.metadata


def test_binding_follows_the_global_config() -> None:
    """ArrayIO responds to the global configuration the way Array always has:
    the pipeline implementation is captured when the binding is made, and
    build_v2's order defaults from array.order."""
    with zarr.config.set(
        {
            "codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline",
            "array.order": "F",
        }
    ):
        v3 = ArrayIO.build_v3(MemoryStore(), shape=(2,), data_type="uint8", chunk_grid=(2,))
        v2 = ArrayIO.build_v2(MemoryStore(), shape=(2,), dtype="uint8", chunks=(2,))
    # outside the block, the choices made inside it hold, also for a binding
    # derived while the global configuration names a pipeline that cannot load
    assert isinstance(v3.to_array().async_array.codec_pipeline, FusedCodecPipeline)
    with zarr.config.set({"codec_pipeline.path": "nonexistent.module.Pipeline"}):
        derived = v3.with_config({"order": "C"})
    assert isinstance(derived.to_array().async_array.codec_pipeline, FusedCodecPipeline)
    legacy = v2.metadata_legacy
    assert isinstance(legacy, ArrayV2Metadata)
    assert legacy.order == "F"


async def test_array_state_changes_rebind_only_the_array() -> None:
    """A change through the array side (resize, update_attributes) shows on the
    array's current binding (`io`), not on a binding taken before it, and a
    derived binding stays independent."""
    store = MemoryStore()
    aio = AsyncArrayIO.build_v3(store, name="foo", shape=(4,), data_type="uint8", chunk_grid=(2,))
    await aio.store_metadata()
    arr = aio.to_array()

    derived = aio.with_config({"order": "F"})
    moved = aio.with_store(MemoryStore())

    await arr.resize((6,))
    assert arr.io.metadata.shape == (6,)
    assert aio.metadata.shape == (4,)

    await arr.update_attributes({"a": 1})
    assert dict(arr.io.metadata.attributes) == {"a": 1}

    # the derivatives were made before the changes and do not alias them
    assert derived.metadata_legacy.shape == (4,)
    assert dict(derived.metadata_legacy.attributes) == {}
    assert dict(moved.metadata_legacy.attributes) == {}

    # the same independence holds for a legacy-sourced binding, whose mutable
    # attributes dict must be copied, not shared
    legacy_arr = await zarr.api.asynchronous.create_array(
        MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,)
    )
    legacy_derivatives = [
        legacy_arr.io.with_config({"order": "F"}),
        legacy_arr.io.with_store(MemoryStore()),
        legacy_arr.io.with_metadata(legacy_arr.io.metadata_legacy),
    ]
    for legacy_derived in legacy_derivatives:
        _ = legacy_derived.metadata  # fill the model cache before the mutation
    await legacy_arr.update_attributes({"b": 2})
    assert dict(legacy_arr.io.metadata.attributes) == {"b": 2}
    for legacy_derived in legacy_derivatives:
        assert dict(legacy_derived.metadata_legacy.attributes) == {}
        assert dict(legacy_derived.metadata.attributes) == {}


async def test_store_metadata_of_model_without_attributes_clears_zattrs() -> None:
    """Storing a v2 model with no attributes removes a stale .zattrs."""
    store = MemoryStore()
    aio = AsyncArrayIO.build_v2(
        store, name="foo", shape=(4,), dtype="uint8", chunks=(4,), attributes={"stale": 1}
    )
    await aio.store_metadata()
    assert "foo/.zattrs" in await _store_keys(store)

    opened = await AsyncArrayIO.open(store, name="foo")
    model = opened.metadata
    assert isinstance(model, ModelV2)
    bare = opened.with_metadata(model.update(attributes=UNSET))
    await bare.store_metadata()
    assert "foo/.zattrs" not in await _store_keys(store)
    reread = await AsyncArrayIO.open(store, name="foo")
    assert dict(reread.metadata.attributes) == {}


def _bind_zero_chunk_by_constructor(metadata: ArrayV2Metadata) -> AsyncArrayIO:
    return AsyncArrayIO(metadata, StorePath(MemoryStore()))


def _bind_zero_chunk_by_build(metadata: ArrayV2Metadata) -> AsyncArrayIO:
    return AsyncArrayIO.build_v2(
        MemoryStore(),
        shape=metadata.shape,
        dtype=metadata.dtype,
        chunks=metadata.chunks,
        fill_value=metadata.fill_value,
        order=metadata.order,
    )


def _bind_zero_chunk_by_with_metadata(metadata: ArrayV2Metadata) -> AsyncArrayIO:
    declared = AsyncArrayIO.build_v2(MemoryStore(), shape=(3,), dtype="int16", chunks=(3,))
    return declared.with_metadata(metadata)


@pytest.mark.parametrize(
    "bind",
    [_bind_zero_chunk_by_constructor, _bind_zero_chunk_by_build, _bind_zero_chunk_by_with_metadata],
    ids=["constructor", "build_v2", "with_metadata"],
)
def test_zero_chunk_v2_metadata_binds_repaired(
    bind: Callable[[ArrayV2Metadata], AsyncArrayIO],
) -> None:
    """`ArrayV2Metadata` with a chunk size of 0 is read as repaired however it
    reaches a binding, as `parse_array_metadata` reads it."""
    metadata = ArrayV2Metadata(shape=(3,), chunks=(0,), dtype=Int16(), fill_value=0, order="C")
    assert bind(metadata).metadata_legacy.chunks == (1,)


async def test_v2_model_binding_keeps_attributes_writable() -> None:
    """A v2 binding read as a model resolves its attributes as plain, mutable
    JSON, so writes through it that re-encode the attributes succeed."""
    store = MemoryStore()
    attributes: dict[str, JSON] = {"k": [1, 2], "d": {"n": None}}
    aio = AsyncArrayIO.build_v2(
        store, name="foo", shape=(4,), dtype="uint8", chunks=(2,), attributes=attributes
    )
    await aio.store_metadata()
    arr = (await AsyncArrayIO.open(store, name="foo")).to_array()

    assert arr.metadata.attributes == attributes
    await arr.update_attributes({"new": 1})
    await arr.resize((6,))

    reread = await open_array(store=store, path="foo")
    assert reread.metadata.attributes == {**attributes, "new": 1}
    assert reread.shape == (6,)


def test_build_v2_vlen_without_object_codec_raises() -> None:
    with pytest.raises(ValueError, match="requires the numcodecs.VLenUTF8 codec"):
        AsyncArrayIO.build_v2(MemoryStore(), shape=(4,), dtype=str, chunks=(2,))


async def test_metadata_dict_binds_as_a_model_until_chunks_move() -> None:
    """A valid metadata dict binds as a model, so an extension zarr-python
    lacks fails only when chunks are read or written; metadata zarr can only
    read as repaired is read, repaired, when bound."""
    store = MemoryStore()
    doc: dict[str, JSON] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "bfloat16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "zarrs.zfp", "configuration": {"mode": "reversible"}}],
    }
    aio = AsyncArrayIO(cast("Any", doc), StorePath(store, "foo"))
    assert isinstance(aio.metadata, ZarrV3ArrayMetadata)
    await aio.store_metadata()
    await aio.verify()
    await aio.erase_chunks()
    with pytest.raises(ValueError, match="bfloat16"):
        await aio.getitem(Ellipsis)

    zero_chunk = {**doc, "data_type": "uint8", "codecs": [{"name": "bytes"}]}
    zero_chunk["chunk_grid"] = {"name": "regular", "configuration": {"chunk_shape": [0]}}
    with pytest.warns(zarr.errors.ZarrUserWarning, match="is read as"):
        repaired = AsyncArrayIO(cast("Any", zero_chunk), StorePath(store, "bar"))
    assert repaired.metadata_legacy.chunks == (1,)


async def test_chunk_and_selection_io_signatures() -> None:
    """erase_chunk takes grid coordinates positionally, as any sequence;
    setitem takes its prototype by keyword."""
    store = MemoryStore()
    aio = AsyncArrayIO.build_v3(store, name="foo", shape=(4,), data_type="uint8", chunk_grid=(2,))
    await aio.setitem(Ellipsis, np.arange(4, dtype="uint8"))
    await aio.erase_chunk([0])
    assert "foo/c/0" not in await _store_keys(store)
    with pytest.raises(TypeError):
        await aio.erase_chunk(chunk_coords=(1,))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        await aio.setitem(Ellipsis, np.arange(4, dtype="uint8"), default_buffer_prototype())  # type: ignore[call-arg]


async def test_v2_model_binding_without_attributes() -> None:
    store = MemoryStore()
    aio = AsyncArrayIO.build_v2(store, name="foo", shape=(4,), dtype="uint8", chunks=(2,))
    await aio.store_metadata()
    await store.delete("foo/.zattrs")
    arr = (await AsyncArrayIO.open(store, name="foo")).to_array()
    assert arr.metadata.attributes == {}
    await arr.update_attributes({"x": 1})
    assert (await open_array(store=store, path="foo")).metadata.attributes == {"x": 1}


async def _store_v2_document(store: MemoryStore, document: dict[str, Any]) -> None:
    prototype = default_buffer_prototype()
    await store.set("a/.zarray", prototype.buffer.from_bytes(json.dumps(document).encode()))


_V2_DOCUMENT: dict[str, Any] = {
    "zarr_format": 2,
    "shape": [4],
    "chunks": [4],
    "dtype": "<i4",
    "fill_value": 0,
    "order": "C",
    "filters": None,
    "compressor": None,
}


async def test_open_rejects_a_v2_chunk_size_zarr_reads_as_repaired() -> None:
    """The strict reads (`open`, `verify`) reject a v2 chunk size of 0, as they do a
    v3 one, though the v2 model accepts it, naming where and what kind."""
    store = MemoryStore()
    await _store_v2_document(store, {**_V2_DOCUMENT, "chunks": [0]})
    with pytest.raises(MetadataValidationError) as opened:
        await AsyncArrayIO.open(store, name="a")
    binding = AsyncArrayIO.build_v2(store, name="a", shape=(4,), dtype="<i4", chunks=(4,))
    with pytest.raises(MetadataValidationError) as verified:
        await binding.verify()
    for error in (opened.value, verified.value):
        assert [(p.loc, p.kind) for p in error.problems] == [(("chunks", 0), "invalid_value")]


async def test_a_strict_v2_read_reports_every_problem_with_its_chunk_sizes() -> None:
    store = MemoryStore()
    await _store_v2_document(store, {**_V2_DOCUMENT, "chunks": [0], "dtype": "nope"})
    with pytest.raises(MetadataValidationError) as error:
        await AsyncArrayIO.open(store, name="a")
    locations = [p.loc for p in error.value.problems]
    assert locations.count(("chunks", 0)) == 1
    assert any(loc[0] == "dtype" for loc in locations)


async def test_a_strict_v2_read_reports_a_chunk_size_once() -> None:
    """A chunk size the model rejects itself is reported once."""
    store = MemoryStore()
    await _store_v2_document(store, {**_V2_DOCUMENT, "chunks": [-1]})
    with pytest.raises(MetadataValidationError) as error:
        await AsyncArrayIO.open(store, name="a")
    assert [p.loc for p in error.value.problems] == [("chunks",)]


def test_with_metadata_rejects_what_is_not_metadata() -> None:
    aio = AsyncArrayIO.build_v3(MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,))
    with pytest.raises(TypeError, match="got str"):
        aio.with_metadata(cast("Any", "nonsense"))


def test_open_a_missing_path_creates_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.zarr"
    with pytest.raises(FileNotFoundError):
        ArrayIO.open(str(missing))
    assert not missing.exists()


def test_metadata_dict_with_storage_transformers_raises() -> None:
    doc: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "uint8",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "bytes"}],
        "storage_transformers": [{"name": "nope"}],
    }
    with pytest.raises(ValueError, match="storage transformers"):
        AsyncArrayIO(cast("Any", doc), StorePath(MemoryStore()))


async def test_bindings_take_metadata_json() -> None:
    """A binding takes array metadata as `zarr_metadata` types it in JSON (what
    zarrista's arrays hold), through the constructor and with_metadata."""
    store = MemoryStore()
    aio = AsyncArrayIO.build_v3(store, name="foo", shape=(4,), data_type="uint8", chunk_grid=(2,))
    json_metadata = aio.metadata.to_json()
    rebound = aio.with_metadata(json_metadata)
    assert AsyncArrayIO(json_metadata, aio.store_path) == aio
    assert rebound == aio
    await rebound.store_metadata()
    await rebound.setitem(Ellipsis, np.arange(4, dtype="uint8"))
    np.testing.assert_array_equal(await aio.getitem(Ellipsis), np.arange(4))


async def test_pickled_binding_rebuilds_its_pipeline(tmp_path: Path) -> None:
    """A pickled binding drops its built pipeline (a pipeline assigned to it is not
    carried either) and builds its class's pipeline on first use."""
    aio = AsyncArrayIO.build_v3(
        LocalStore(tmp_path), name="foo", shape=(4,), data_type="uint8", chunk_grid=(2,)
    )
    await aio.setitem(Ellipsis, np.arange(4, dtype="uint8"))
    revived = pickle.loads(pickle.dumps(aio))
    assert revived._codec_pipeline_cache is None
    np.testing.assert_array_equal(await revived.getitem(Ellipsis), np.arange(4))
    assert isinstance(revived._resolved_codec_pipeline, revived._pipeline_class)


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_erase_and_store_metadata_leave_a_group_in_place(zarr_format: ZarrFormat) -> None:
    """erase_metadata deletes only an array's documents, and storing a v2 model
    without attributes keeps a group's `.zattrs`."""
    store = MemoryStore()
    group = await zarr.api.asynchronous.open_group(
        store, path="g", zarr_format=zarr_format, mode="w"
    )
    await group.update_attributes({"owner": "group"})
    before = await _store_keys(store)
    declared = (
        AsyncArrayIO.build_v2(store, name="g", shape=(4,), dtype="uint8", chunks=(4,))
        if zarr_format == 2
        else AsyncArrayIO.build_v3(store, name="g", shape=(4,), data_type="uint8", chunk_grid=(4,))
    )
    await declared.erase_metadata()
    assert await _store_keys(store) == before
    if zarr_format == 2:
        model = declared.metadata
        assert isinstance(model, ModelV2)
        await declared.with_metadata(model.update(attributes=UNSET)).store_metadata()
        reopened = await zarr.api.asynchronous.open_group(store, path="g", mode="r", zarr_format=2)
        assert dict(reopened.attrs) == {"owner": "group"}


def test_constructor_never_shares_the_callers_metadata() -> None:
    mine = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(4,)).metadata
    aio = AsyncArrayIO(mine, StorePath(MemoryStore()))
    assert aio.metadata_legacy is not mine
    assert aio.metadata_legacy.attributes is not mine.attributes


async def test_metadata_dict_of_a_repaired_document_stores_its_repair() -> None:
    """An invalid stored document given to AsyncArrayIO as a dict is recorded as
    what the store holds, as AsyncArray records it, so the first chunk write
    stores the repair."""
    store = MemoryStore()
    doc: dict[str, Any] = {
        "zarr_format": 2,
        "shape": [3],
        "chunks": [0],
        "dtype": "<u2",
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "compressor": None,
    }
    prototype = default_buffer_prototype()
    await store.set("foo/.zarray", prototype.buffer.from_bytes(json.dumps(doc).encode()))
    with pytest.warns(zarr.errors.ZarrUserWarning, match="is read as"):
        aio = AsyncArrayIO(cast("Any", {**doc, "attributes": {}}), StorePath(store, "foo"))
    await aio.setitem(Ellipsis, np.array([7, 8, 9], dtype="uint16"))
    buffer = await store.get("foo/.zarray", prototype=prototype)
    assert buffer is not None
    assert json.loads(buffer.to_bytes())["chunks"] == [1]


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_verify_names_the_fields_that_differ(zarr_format: ZarrFormat) -> None:
    """Nested attributes (read-only in the model) do not stop the field naming."""
    store = MemoryStore()
    attributes: dict[str, JSON] = {"units": {"name": "m", "scale": [1, 2]}}
    aio = (
        AsyncArrayIO.build_v2(
            store,
            name="foo",
            shape=(4,),
            dtype="float32",
            chunks=(4,),
            fill_value=0.0,
            attributes=attributes,
        )
        if zarr_format == 2
        else AsyncArrayIO.build_v3(
            store,
            name="foo",
            shape=(4,),
            data_type="float32",
            chunk_grid=(4,),
            fill_value=0.0,
            attributes=attributes,
        )
    )
    await aio.store_metadata()
    other = aio.with_metadata(aio.metadata.update(fill_value=-0.0))
    with pytest.raises(ValueError, match=r"differing fields: fill_value\.$"):
        await other.verify()


def test_build_v3_keeps_empty_dimension_names_on_a_0d_array() -> None:
    """The builder keeps an empty name list for a 0-d array; the creation paths
    read it as no names, as they always have."""
    built = ArrayIO.build_v3(
        MemoryStore(), shape=(), data_type="int8", chunk_grid=(), dimension_names=[]
    )
    assert built.metadata_legacy.dimension_names == ()  # type: ignore[union-attr]
    created = zarr.create_array(MemoryStore(), shape=(), dtype="int8", dimension_names=[])
    assert created.metadata.dimension_names is None  # type: ignore[union-attr]


async def test_erase_chunk_takes_integer_coordinates() -> None:
    store = MemoryStore()
    aio = AsyncArrayIO.build_v3(store, name="foo", shape=(4,), data_type="uint8", chunk_grid=(2,))
    await aio.setitem(Ellipsis, np.arange(4, dtype="uint8"))
    with pytest.raises(TypeError):
        await aio.erase_chunk([cast("int", np.float64(1))])
    await aio.erase_chunk([cast("int", np.int64(1))])
    assert "foo/c/1" not in await _store_keys(store)


def test_equality_is_canonical_whatever_is_resolved() -> None:
    """Bindings compare by canonical metadata, before and after resolving, and
    whatever they were built from: a fill value of -0.0 differs from 0.0."""
    store_path = StorePath(MemoryStore(), "foo")
    positive = AsyncArrayIO.build_v3(
        store_path, shape=(4,), data_type="float32", chunk_grid=(4,), fill_value=0.0
    )
    negative = positive.with_metadata(positive.metadata.update(fill_value=-0.0))
    legacy_negative = AsyncArrayIO(negative.metadata_legacy, store_path)
    assert positive != negative
    assert positive != legacy_negative
    assert negative == legacy_negative
    _ = (positive.metadata_legacy, negative.metadata_legacy)
    assert positive != negative
    assert negative == legacy_negative


@pytest.mark.parametrize("bind", ["open", "model"])
async def test_storage_transformers_are_rejected_when_bound(bind: str) -> None:
    store = MemoryStore()
    doc: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "uint8",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "bytes"}],
        "storage_transformers": [{"name": "nope"}],
    }
    if bind == "open":
        prototype = default_buffer_prototype()
        await store.set("foo/zarr.json", prototype.buffer.from_bytes(json.dumps(doc).encode()))
        with pytest.raises(ValueError, match="storage transformers"):
            await AsyncArrayIO.open(store, name="foo")
    else:
        model = ZarrV3ArrayMetadata(doc)
        with pytest.raises(ValueError, match="storage transformers"):
            AsyncArrayIO(model, StorePath(store, "foo"))


@pytest.mark.parametrize(
    ("documents", "left"),
    [
        ({"foo/zarr.json": b"not json"}, set()),
        ({"foo/zarr.json": b'{"zarr_format": 3}'}, set()),
        ({"foo/zarr.json": b'{"zarr_format": 3, "node_type": "group"}'}, {"foo/zarr.json"}),
        ({"foo/.zattrs": b"{}"}, set()),
        (
            {"foo/.zarray": b"{}", "foo/.zattrs": b"{}", "foo/.zgroup": b"{}"},
            {"foo/.zattrs", "foo/.zgroup"},
        ),
    ],
    ids=["corrupt-v3", "v3-without-node-type", "v3-group", "stray-v2-zattrs", "v2-array-and-group"],
)
async def test_erase_metadata_removes_array_documents_however_written(
    documents: dict[str, bytes], left: set[str]
) -> None:
    """erase_metadata removes array documents even when they are corrupt, and
    leaves only a group's documents in place."""
    store = MemoryStore()
    prototype = default_buffer_prototype()
    for key, value in documents.items():
        await store.set(key, prototype.buffer.from_bytes(value))
    zarr_format: ZarrFormat = 3 if "foo/zarr.json" in documents else 2
    declared = (
        AsyncArrayIO.build_v2(store, name="foo", shape=(4,), dtype="uint8", chunks=(4,))
        if zarr_format == 2
        else AsyncArrayIO.build_v3(
            store, name="foo", shape=(4,), data_type="uint8", chunk_grid=(4,)
        )
    )
    await declared.erase_metadata()
    assert await _store_keys(store) == left


async def test_with_metadata_takes_any_mapping() -> None:
    aio = AsyncArrayIO.build_v3(
        MemoryStore(), name="foo", shape=(4,), data_type="uint8", chunk_grid=(4,)
    )
    rebound = aio.with_metadata(cast("Any", MappingProxyType(dict(aio.metadata.to_json()))))
    assert rebound == aio
    await rebound.store_metadata()


BYTES_JSON: dict[str, JSON] = {"name": "bytes", "configuration": {"endian": "little"}}


ZSTD_JSON: dict[str, JSON] = {"name": "zstd", "configuration": {"level": 0, "checksum": False}}


class _MarkedPipeline(BatchedCodecPipeline):
    """A pipeline class distinguishable from the default, for context tests."""


async def test_context_scopes_resolution() -> None:
    """Names resolve through the context when the data path is built, the
    context's pipeline wins over the global choice, and with_context rebinds."""
    store = MemoryStore()
    data = np.arange(16, dtype="uint16").reshape(4, 4)
    registry = DataTypeRegistry()
    registry.register(UInt16._zarr_v3_name, UInt16)
    context = Context(
        data_types=registry,
        codecs={"bytes": BytesCodec, "zstd": ZstdCodec},
        codec_pipeline=_MarkedPipeline,
    )
    aio = AsyncArrayIO.build_v3(
        store,
        name="foo",
        shape=data.shape,
        data_type="uint16",
        chunk_grid=(2, 2),
        codecs=[BYTES_JSON, ZSTD_JSON],
        context=context,
    )
    assert isinstance(aio.to_array().codec_pipeline, _MarkedPipeline)
    await aio.setitem(Ellipsis, data)
    await aio.store_metadata()

    opened = await AsyncArrayIO.open(store, name="foo", context=context)
    np.testing.assert_array_equal(await opened.getitem(Ellipsis), data)

    # the context survives derivation on both layers
    assert isinstance(opened.with_config({"order": "F"}).to_array().codec_pipeline, _MarkedPipeline)
    assert isinstance(opened.to_array().with_config({"order": "F"}).codec_pipeline, _MarkedPipeline)

    rebound = opened.with_context(None)
    assert not isinstance(rebound.to_array().codec_pipeline, _MarkedPipeline)
    np.testing.assert_array_equal(await rebound.getitem(Ellipsis), data)


async def test_context_unknown_codec_resolution_raises() -> None:
    """A stored codec name outside the context opens for metadata work and
    fails to resolve when the data path is built, naming the scope."""
    store = MemoryStore()
    aio = AsyncArrayIO.build_v3(store, name="foo", shape=(4,), data_type="uint8", chunk_grid=(4,))
    await aio.store_metadata()
    context = Context(data_types=data_type_registry, codecs={"bytes": BytesCodec})
    opened = await AsyncArrayIO.open(store, name="foo", context=context)
    await opened.verify()  # metadata work needs no resolution
    with pytest.raises(UnknownCodecError, match="Codecs in scope: \\['bytes'\\]"):
        await opened.getitem(Ellipsis)


def test_context_unknown_codec_on_build_raises() -> None:
    context = Context(data_types=data_type_registry, codecs={"bytes": BytesCodec})
    with pytest.raises(UnknownCodecError, match="'zstd' is not in this context"):
        AsyncArrayIO.build_v3(
            MemoryStore(),
            shape=(4,),
            data_type="uint8",
            chunk_grid=(4,),
            codecs=[BYTES_JSON, ZSTD_JSON],
            context=context,
        )


def test_context_unknown_dtype_raises() -> None:
    context = Context(data_types=DataTypeRegistry())
    with pytest.raises(ValueError, match="No Zarr data type found"):
        AsyncArrayIO.build_v3(
            MemoryStore(), shape=(4,), data_type="uint16", chunk_grid=(4,), context=context
        )
    # the str shortcut goes through the registry too
    with pytest.raises(ValueError, match="No Zarr data type found"):
        AsyncArrayIO.build_v3(
            MemoryStore(), shape=(4,), data_type=str, chunk_grid=(4,), context=context
        )


def test_context_pickles_with_codecs() -> None:
    """An array bound to a context with a codec scope pickles and unpickles
    with the scope intact."""
    context = Context(
        data_types=data_type_registry,
        codecs={"bytes": BytesCodec, "zstd": ZstdCodec},
        codec_pipeline=_MarkedPipeline,
    )
    arr = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(4,))
    rebound = arr.io.with_context(context).to_array()
    revived = pickle.loads(pickle.dumps(rebound))
    revived_context = revived._async_array._io._context
    assert revived_context is not None
    assert revived_context.codecs is not None
    assert set(revived_context.codecs) == {"bytes", "zstd"}
    assert revived_context.codec_pipeline is _MarkedPipeline
    np.testing.assert_array_equal(revived[:], np.zeros(4, dtype="uint8"))


async def test_with_context_reresolves_a_legacy_source() -> None:
    """A binding built from resolved metadata is re-resolved by with_context,
    so its stored names are checked against the scope."""
    arr = await zarr.api.asynchronous.create_array(
        MemoryStore(), shape=(4,), dtype="uint16", chunks=(4,)
    )
    full = Context(
        data_types=data_type_registry,
        codecs={"bytes": BytesCodec, "zstd": ZstdCodec},
        codec_pipeline=_MarkedPipeline,
    )
    rebound = arr.io.with_context(full)
    assert isinstance(rebound.to_array().codec_pipeline, _MarkedPipeline)
    data = np.arange(4, dtype="uint16")
    await rebound.setitem(Ellipsis, data)
    np.testing.assert_array_equal(await rebound.getitem(Ellipsis), data)


async def test_with_context_rejects_a_legacy_codec_outside_the_scope() -> None:
    arr = await zarr.api.asynchronous.create_array(
        MemoryStore(), shape=(4,), dtype="uint16", chunks=(4,)
    )
    narrow = Context(data_types=data_type_registry, codecs={"bytes": BytesCodec})
    with pytest.raises(UnknownCodecError, match="is not in this context"):
        arr.io.with_context(narrow)


async def test_with_context_rejects_a_legacy_dtype_outside_the_scope() -> None:
    arr = await zarr.api.asynchronous.create_array(
        MemoryStore(), shape=(4,), dtype="uint16", chunks=(4,)
    )
    with pytest.raises(ValueError, match="No Zarr data type found"):
        arr.io.with_context(Context(data_types=DataTypeRegistry()))


async def test_context_v2_scopes_dtype_resolution() -> None:
    """A v2 document resolves its dtype through the context on build and open."""
    store = MemoryStore()
    registry = DataTypeRegistry()
    registry.register(UInt16._zarr_v3_name, UInt16)
    context = Context(data_types=registry)
    data = np.arange(4, dtype="uint16")
    aio = AsyncArrayIO.build_v2(
        store, name="foo", shape=(4,), dtype="uint16", chunks=(4,), context=context
    )
    await aio.store_metadata()
    await aio.setitem(Ellipsis, data)
    opened = await AsyncArrayIO.open(store, name="foo", context=context)
    np.testing.assert_array_equal(await opened.getitem(Ellipsis), data)


def test_context_v2_unknown_dtype_raises() -> None:
    context = Context(data_types=DataTypeRegistry())
    with pytest.raises(ValueError, match="No Zarr data type found"):
        AsyncArrayIO.build_v2(
            MemoryStore(), name="foo", shape=(4,), dtype="uint16", chunks=(4,), context=context
        )


def test_context_default_codecs_outside_the_scope_raise() -> None:
    """The default codec chain resolves under the context too, so a closed
    scope never stores a chain it could not read back."""
    narrow = Context(data_types=data_type_registry, codecs={"bytes": BytesCodec})
    with pytest.raises(UnknownCodecError, match="is not in this context"):
        AsyncArrayIO.build_v3(
            MemoryStore(), shape=(4,), data_type="uint16", chunk_grid=(4,), context=narrow
        )


async def test_with_context_rebinding_stores_the_repair() -> None:
    """Rebinding a repaired legacy array carries nothing about the store: its
    store's cache still records the broken document, so the first write
    through the rebound binding stores the repair."""
    store = MemoryStore()
    doc: dict[str, Any] = {
        "zarr_format": 2,
        "shape": [3],
        "chunks": [0],
        "dtype": "<u2",
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "compressor": None,
    }
    prototype = default_buffer_prototype()
    await store.set("foo/.zarray", prototype.buffer.from_bytes(json.dumps(doc).encode()))
    with pytest.warns(ZarrUserWarning, match="is read as"):
        arr = await open_array(store=store, path="foo", mode="r+")
    context = Context(data_types=data_type_registry)
    rebound = arr.io.with_context(context)
    assert stored_documents(store).needing_repair("foo", 2) is not None

    await rebound.setitem(slice(None), np.array([7, 8, 9], dtype="uint16"))

    buffer = await store.get("foo/.zarray", prototype=prototype)
    assert buffer is not None
    assert json.loads(buffer.to_bytes())["chunks"] == [1]
    np.testing.assert_array_equal(await rebound.getitem(Ellipsis), [7, 8, 9])


class _ScopedUTF8(VariableLengthUTF8):
    """A variable-length UTF-8 data type distinguishable from the default."""


def test_parse_dtype_vlen_shortcut_in_context() -> None:
    """The variable-length-UTF8 shortcut returns what the context's registry
    matches, not the default class."""
    registry = DataTypeRegistry()
    registry.register(VariableLengthUTF8._zarr_v3_name, _ScopedUTF8)  # type: ignore[arg-type]
    context = Context(data_types=registry)
    assert isinstance(parse_dtype(str, zarr_format=2, context=context), _ScopedUTF8)
    assert isinstance(parse_dtype(str, zarr_format=3, context=context), _ScopedUTF8)


def test_with_metadata_under_a_context_rejects_a_codec_outside_it() -> None:
    narrow = Context(data_types=data_type_registry, codecs={"bytes": BytesCodec})
    aio = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint8",
        chunk_grid=(4,),
        codecs=[BYTES_JSON],
        context=narrow,
    )
    outside = zarr.create_array(
        MemoryStore(), shape=(4,), dtype="uint8", chunks=(4,)
    ).io.metadata_legacy
    with pytest.raises(UnknownCodecError, match="is not in this context"):
        aio.with_metadata(outside)


def test_builders_under_a_context_reject_instances_outside_it() -> None:
    """Data type and codec instances are checked against a closed context
    too, so a builder never binds what its own context cannot read back."""
    registry = DataTypeRegistry()
    registry.register(UInt16._zarr_v3_name, UInt16)
    context = Context(data_types=registry, codecs={"bytes": BytesCodec, "zstd": ZstdCodec})
    with pytest.raises(ValueError, match="No Zarr data type found"):
        AsyncArrayIO.build_v3(
            MemoryStore(), shape=(4,), data_type=Int16(), chunk_grid=(4,), context=context
        )
    with pytest.raises(UnknownCodecError, match="'gzip' is not in this context"):
        AsyncArrayIO.build_v3(
            MemoryStore(),
            shape=(4,),
            data_type="uint16",
            chunk_grid=(4,),
            codecs=[BytesCodec(), GzipCodec()],
            context=context,
        )


class _ScopedUInt16(UInt16):
    """A uint16 data type distinguishable from the default."""


def _scoped_uint16_registry() -> DataTypeRegistry:
    registry = DataTypeRegistry()
    registry.register(UInt16._zarr_v3_name, _ScopedUInt16)
    return registry


def test_v2_bindings_resolve_through_a_data_type_scope() -> None:
    """A Zarr format 2 legacy object entering a data type scope (constructor,
    with_metadata) binds the scope's class, and leaving it rebinds the default;
    a codec scope does not govern a Zarr format 2 array's numcodecs."""
    context = Context(data_types=_scoped_uint16_registry())
    legacy = zarr.create_array(
        MemoryStore(), shape=(4,), dtype="uint16", chunks=(4,), zarr_format=2
    ).io.metadata_legacy
    declared = AsyncArrayIO.build_v2(
        MemoryStore(), shape=(4,), dtype="uint16", chunks=(4,), context=context
    )
    for binding in (
        AsyncArrayIO(legacy, StorePath(MemoryStore()), context=context),
        declared.with_metadata(legacy),
    ):
        assert type(binding.metadata_legacy.dtype) is _ScopedUInt16
        assert type(binding.with_context(None).metadata_legacy.dtype) is UInt16

    codecs_only = Context(codecs={"bytes": BytesCodec})
    unscoped = AsyncArrayIO(legacy, StorePath(MemoryStore()), context=codecs_only)
    assert unscoped.metadata_legacy.to_dict() == legacy.to_dict()


def test_switching_between_scopes_of_different_kinds() -> None:
    """Moving from a codec scope to a data type scope reads the codecs again
    under the defaults and resolves the data type through the new scope."""
    codec_scoped = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint16",
        chunk_grid=(4,),
        codecs=[BYTES_JSON],
        context=Context(codecs={"bytes": _UnregisteredBytes}),
    )
    assert type(codec_scoped.metadata_legacy.codecs[0]) is _UnregisteredBytes  # type: ignore[union-attr]
    switched = codec_scoped.with_context(Context(data_types=_scoped_uint16_registry()))
    assert type(switched.metadata_legacy.codecs[0]) is BytesCodec  # type: ignore[union-attr]
    assert type(switched.metadata_legacy.dtype) is _ScopedUInt16


def test_bindings_resolve_through_the_context_they_are_under() -> None:
    """Whatever reaches a binding under a context, its data type is the
    context's class: a builder's declaration, a chunk-size-0 v2 declaration
    (read as repaired), and a legacy object given to with_metadata; leaving
    the context rebinds the default class."""
    context = Context(data_types=_scoped_uint16_registry())

    built = AsyncArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type="uint16", chunk_grid=(4,), context=context
    )
    zero_chunk = AsyncArrayIO.build_v2(
        MemoryStore(), shape=(3,), dtype="uint16", chunks=(0,), context=context
    )
    outside = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint16", chunks=(4,))
    rebound = built.with_metadata(outside.io.metadata_legacy)

    for binding in (built, zero_chunk, rebound):
        assert isinstance(binding.metadata_legacy.dtype, _ScopedUInt16)
    assert zero_chunk.metadata_legacy.chunks == (1,)
    assert type(built.with_context(None).metadata_legacy.dtype) is UInt16


def test_context_selects_a_pipeline_alone() -> None:
    """A context that only selects the codec pipeline resolves names from the
    default registries."""
    context = Context(codec_pipeline=_MarkedPipeline)
    aio = AsyncArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type="uint16", chunk_grid=(4,), context=context
    )
    assert isinstance(aio.to_array().codec_pipeline, _MarkedPipeline)
    assert type(aio.metadata_legacy.dtype) is UInt16


def test_bindings_under_different_contexts_differ() -> None:
    store = MemoryStore()
    aio = AsyncArrayIO.build_v3(store, name="foo", shape=(4,), data_type="uint8", chunk_grid=(4,))
    scoped = aio.with_context(Context(codec_pipeline=_MarkedPipeline))
    assert scoped != aio
    assert scoped.with_context(None) == aio


class _UnregisteredUInt16(UInt16):
    """A data type no registry holds."""


class _UnregisteredBytes(BytesCodec):
    """A codec no registry holds by its own name; it encodes as `bytes`."""


def test_pipeline_only_context_resolves_as_no_context() -> None:
    """A context that only selects the codec pipeline accepts exactly what no
    context accepts: unregistered data type and codec instances bind as given,
    for both formats, and survive with_config and with_context."""
    context = Context(codec_pipeline=_MarkedPipeline)
    v3 = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type=_UnregisteredUInt16(),
        chunk_grid=(4,),
        codecs=[_UnregisteredBytes()],
        context=context,
    )
    v2 = AsyncArrayIO.build_v2(
        MemoryStore(), shape=(4,), dtype=_UnregisteredUInt16(), chunks=(4,), context=context
    )
    v3_bindings = [
        v3,
        v3.with_config({"order": "F"}),
        v3.with_context(None),
        AsyncArrayIO(v3.metadata_legacy, StorePath(MemoryStore()), context=context),
        v3.with_metadata(v3.metadata_legacy),
    ]
    for binding in [*v3_bindings, v2]:
        assert type(binding.metadata_legacy.dtype) is _UnregisteredUInt16
    for binding in v3_bindings:
        assert type(binding.metadata_legacy.codecs[0]) is _UnregisteredBytes  # type: ignore[union-attr]
    assert isinstance(v3.to_array().codec_pipeline, _MarkedPipeline)


async def test_scoped_array_keeps_its_context() -> None:
    """An array under a closed context keeps it across with_config and
    pickling, reads chunk-size-0 v2 metadata as repaired without a warning,
    and binds the context's classes."""
    registry = DataTypeRegistry()
    registry.register(UInt16._zarr_v3_name, _ScopedUInt16)
    context = Context(data_types=registry, codec_pipeline=_MarkedPipeline)
    zero_chunk = ArrayV2Metadata(shape=(3,), chunks=(0,), dtype=UInt16(), fill_value=0, order="C")
    store = MemoryStore()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        arr = AsyncArray(zero_chunk, StorePath(store), context=context)
        declared = AsyncArrayIO.build_v2(store, shape=(3,), dtype="uint16", chunks=(3,))
        rebound = declared.with_context(context).with_metadata(zero_chunk)
        configured = arr.with_config({"order": "F"})
    revived = pickle.loads(pickle.dumps(arr))
    for array in (arr, configured, revived, rebound.to_array()):
        assert array.metadata.chunks == (1,)
        assert isinstance(array.metadata.dtype, _ScopedUInt16)
        assert array.io._context == context
        assert isinstance(array.codec_pipeline, _MarkedPipeline)


def test_default_context_is_the_empty_context() -> None:
    assert Context.default() == Context()
    assert not Context().limits_names
    assert Context(codecs={"bytes": BytesCodec}).limits_names


def test_closed_data_type_scope_checks_structured_fields() -> None:
    """A NumPy structured dtype is checked whole: a field type outside a closed
    data type scope is rejected at build, as a Struct instance is."""
    registry = DataTypeRegistry()
    registry.register(Int8._zarr_v3_name, Int8)
    registry.register(Struct._zarr_v3_name, Struct)
    context = Context(data_types=registry)
    with pytest.raises(ValueError, match="No Zarr data type found"):
        AsyncArrayIO.build_v3(
            MemoryStore(),
            shape=(4,),
            data_type=np.dtype([("a", "i1"), ("b", "f8")]),
            chunk_grid=(4,),
            context=context,
        )


def test_closed_codec_scope_keeps_admitted_instances() -> None:
    """An admitted codec instance binds as given, so it evolves against the data
    type as it does with no context."""
    closed = Context(codecs={"bytes": BytesCodec, "blosc": BloscCodec})
    stored = [
        AsyncArrayIO.build_v3(
            MemoryStore(),
            shape=(4,),
            data_type="int32",
            chunk_grid=(4,),
            codecs=[BytesCodec(), BloscCodec()],
            context=context,
        ).metadata_legacy.to_dict()["codecs"]
        for context in (closed, None)
    ]
    assert stored[0] == stored[1]


def test_leaving_a_closed_scope_rebinds_the_defaults() -> None:
    """Rebinding under any context that limits no names (or none) reads the
    names a closed scope resolved again under the defaults, and a context that
    sets nothing is no context."""
    scoped = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint8",
        chunk_grid=(4,),
        codecs=[BYTES_JSON],
        context=Context(codecs={"bytes": _UnregisteredBytes}),
    )
    for context in (None, Context(), Context(codec_pipeline=_MarkedPipeline)):
        rebound = scoped.with_context(context)
        assert type(rebound.metadata_legacy.codecs[0]) is BytesCodec  # type: ignore[union-attr]
    plain = scoped.with_context(None)
    assert plain.with_context(Context.default()) == plain


def test_with_metadata_under_a_pipeline_only_context_copies() -> None:
    declared = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint8",
        chunk_grid=(4,),
        context=Context(codec_pipeline=_MarkedPipeline),
    )
    mine = zarr.create_array(
        MemoryStore(), shape=(4,), dtype="uint8", chunks=(4,)
    ).io.metadata_legacy
    rebound = declared.with_metadata(mine)
    assert rebound.metadata_legacy is not mine
    assert rebound.metadata_legacy.attributes is not mine.attributes


def test_the_context_is_not_a_dataclass_field() -> None:
    """The dataclass fields are as they always were; `dataclasses.replace` builds
    from them, so its result has no context, and `with_config` keeps it."""
    context = Context(codec_pipeline=_MarkedPipeline)
    arr = AsyncArray(
        zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(4,)).metadata,
        StorePath(MemoryStore()),
        context=context,
    )
    order_f = parse_array_config({"order": "F"})
    assert dataclasses.replace(arr, config=order_f).io.context is None
    configured = arr.with_config(order_f)
    assert configured.io.context == context
    assert isinstance(configured.codec_pipeline, _MarkedPipeline)
    assert [f.name for f in dataclasses.fields(arr)] == [
        "metadata",
        "store_path",
        "codec_pipeline",
        "_chunk_grid",
        "config",
    ]


@pytest.mark.parametrize(
    "context",
    [
        None,
        Context(codec_pipeline=BatchedCodecPipeline),
        Context(data_types=data_type_registry),
    ],
)
def test_v3_from_dict_reports_errors_in_field_order(context: Context | None) -> None:
    """Unless a context limits codecs, codecs are parsed after the other fields,
    so a document with several bad fields reports the chunk grid first, as it
    always has."""
    doc: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "uint8",
        "chunk_grid": {"name": "nope", "configuration": {}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "nope"}],
    }
    with pytest.raises(ValueError, match="chunk grid"):
        ArrayV3Metadata.from_dict(doc, context=context)


def test_a_context_rechecks_only_the_kinds_it_limits() -> None:
    """A context that limits one kind leaves the other as given, entering it
    (constructor, with_metadata) and leaving it (with_context), for an instance
    no registry holds."""
    codecs_only = Context(codecs={"bytes": BytesCodec})
    declared = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type=_UnregisteredUInt16(),
        chunk_grid=(4,),
        codecs=[BytesCodec()],
    )
    legacy = declared.metadata_legacy
    bindings = [
        AsyncArrayIO(legacy, StorePath(MemoryStore()), context=codecs_only),
        declared.with_context(codecs_only).with_metadata(legacy),
        declared.with_context(codecs_only).with_context(None),
    ]
    for binding in bindings:
        assert type(binding.metadata_legacy.dtype) is _UnregisteredUInt16

    registry = DataTypeRegistry()
    registry.register(UInt16._zarr_v3_name, UInt16)
    data_types_only = Context(data_types=registry)
    unregistered_codec = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint16",
        chunk_grid=(4,),
        codecs=[_UnregisteredBytes()],
    )
    for binding in (
        unregistered_codec.with_context(data_types_only),
        unregistered_codec.with_context(data_types_only).with_context(None),
    ):
        assert type(binding.metadata_legacy.codecs[0]) is _UnregisteredBytes  # type: ignore[union-attr]


@dataclass(frozen=True)
class _LocalBytes(BytesCodec):
    """A serializer only a context holds, under a name of its own."""

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "local_bytes", "configuration": {"endian": "little"}}

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(endian="little")


async def test_repair_before_a_write_reads_under_the_context() -> None:
    """The repair check before a chunk write reads the stored document under the
    binding's context, so a codec only the context holds does not stop a write
    to an array whose stored chunk size needs a repair."""
    store = MemoryStore()
    doc: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [3],
        "data_type": "uint16",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [0]}},
        "chunk_key_encoding": {"name": "default"},
        "codecs": [{"name": "local_bytes", "configuration": {"endian": "little"}}],
        "attributes": {},
    }
    prototype = default_buffer_prototype()
    await store.set("a/zarr.json", prototype.buffer.from_bytes(json.dumps(doc).encode()))
    context = Context(codecs={"local_bytes": _LocalBytes})
    with pytest.warns(ZarrUserWarning, match="is read as"):
        arr = AsyncArray(cast("Any", doc), StorePath(store, "a"), context=context)
    await arr.setitem(slice(None), np.array([7, 8, 9], dtype="uint16"))
    np.testing.assert_array_equal(await arr.getitem(slice(None)), [7, 8, 9])


def test_closed_codec_scope_rejects_an_instance_of_another_class() -> None:
    """An instance of another class than the scope maps its name to is refused, not
    rebuilt (that would drop state its JSON does not hold)."""
    with pytest.raises(UnknownCodecError, match="maps 'bytes' to BytesCodec"):
        AsyncArrayIO.build_v3(
            MemoryStore(),
            shape=(4,),
            data_type="uint8",
            chunk_grid=(4,),
            codecs=[_UnregisteredBytes()],
            context=Context(codecs={"bytes": BytesCodec}),
        )


def test_a_context_copies_a_registry_it_is_given() -> None:
    registry = _scoped_uint16_registry()
    context = Context(data_types=registry)
    registry.register(Int16._zarr_v3_name, Int16)
    assert context.data_types is not None
    assert Int16._zarr_v3_name not in context.data_types.contents
    assert Context(data_types=data_type_registry).data_types is data_type_registry


def test_a_pickled_context_keeps_the_default_registry() -> None:
    """A context over the default data type registry unpickles over the
    receiving process's registry, not a snapshot; another registry is copied."""
    restored = pickle.loads(pickle.dumps(Context(data_types=data_type_registry)))
    assert restored.data_types is data_type_registry
    own = _scoped_uint16_registry()
    copied = pickle.loads(pickle.dumps(Context(data_types=own)))
    assert copied.data_types is not own
    assert copied.data_types is not None
    assert copied.data_types.get(UInt16._zarr_v3_name) is _ScopedUInt16


class _TaggedAsyncArray(AsyncArray[Any]):
    """An `AsyncArray` subclass with an instance attribute of its own."""

    tag: str

    def __init__(
        self, metadata: Any, store_path: StorePath, config: Any = None, tag: str = "default"
    ) -> None:
        super().__init__(metadata, store_path, config)
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
        assert set(vars(restored)) == {"_io", "tag"}

    # Rebuilt through the subclass's __init__ (which takes no context), as before.
    order_f = parse_array_config({"order": "F"})
    for rebuilt in (tagged.with_config(order_f), dataclasses.replace(tagged, config=order_f)):
        assert type(rebuilt) is _TaggedAsyncArray
        assert rebuilt.tag == "default"
        assert rebuilt.config.order == "F"


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


async def test_an_array_copied_by_its_dict_keeps_its_own_state() -> None:
    """An array made by copying another's `__dict__` (as subclasses do) is changed
    apart from it: a resize, an attribute update or a field set through either
    leaves the other, as when its fields were its only state; so does a binding
    taken from `io` before the change."""
    original = zarr.create_array(MemoryStore(), shape=(4,), dtype="int32", chunks=(2,))._async_array
    taken = original.io
    clone = object.__new__(AsyncArray)
    clone.__dict__.update(original.__dict__)
    await clone.resize((6,))
    assert (original.shape, clone.shape, taken.metadata_legacy.shape) == ((4,), (6,), (4,))
    await clone.update_attributes({"x": 1})
    assert original.attrs == {}
    object.__setattr__(clone, "metadata", clone.metadata.update_shape((3,)))
    assert original.shape == (4,)


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


def _bind_by_constructor(model: ModelV2) -> AsyncArrayIO:
    return AsyncArrayIO(model, StorePath(MemoryStore(), "a"))


def _bind_by_with_metadata(model: ModelV2) -> AsyncArrayIO:
    declared = AsyncArrayIO.build_v2(MemoryStore(), shape=(4,), dtype="<i4", chunks=(4,))
    return declared.with_metadata(model)


@pytest.mark.parametrize("bind", [_bind_by_constructor, _bind_by_with_metadata])
def test_binding_rejects_a_v2_model_with_chunk_size_zero(
    bind: Callable[[ModelV2], AsyncArrayIO],
) -> None:
    """The v2 model accepts a chunk size of 0, which zarr reads as repaired; bound
    as a model, it is rejected as the strict read rejects it."""
    model = ModelV2(_V2_CHUNK_ZERO)
    with pytest.raises(MetadataValidationError, match="chunks"):
        bind(model)


def test_a_tuple_valued_v2_chunk_size_zero_binds_as_repaired() -> None:
    """A metadata mapping written in Python (tuples, not lists) binds as `AsyncArray`
    reads it: a chunk size of 0 as repaired."""
    document = {**_V2_CHUNK_ZERO, "shape": (0,), "chunks": (0,)}
    binding = ArrayIO(cast("Any", document), StorePath(MemoryStore(), "a"))
    assert binding.metadata_legacy.chunks == (1,)
    binding.store_metadata()
    binding.verify()


async def test_an_array_over_a_model_shows_one_document() -> None:
    """An array over a model-backed binding hands out its metadata, which callers
    change in place (attribute updates through a copy); the binding's model then
    reads from it, so storing through the binding keeps the change."""
    store = MemoryStore()
    await AsyncArrayIO.build_v3(
        store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,), attributes={"v": 0}
    ).store_metadata()
    original = (await AsyncArrayIO.open(store, name="a")).to_array()
    _ = original.metadata
    for copied in (copy.copy(original), original.with_config({"order": "F"})):
        await copied.update_attributes({"v": id(copied)})
        assert dict(original.io.metadata.attributes) == original.attrs
        await original.io.store_metadata()
        await original.io.verify()


def test_bindings_of_different_classes_hash_as_they_compare() -> None:
    class _Binding(AsyncArrayIO):
        pass

    base = AsyncArrayIO.build_v3(
        MemoryStore(), name="a", shape=(4,), data_type="uint8", chunk_grid=(4,)
    )
    other = _Binding(base.metadata, base.store_path)
    assert other == base
    assert hash(other) == hash(base)


async def test_attribute_updates_through_an_array() -> None:
    """An attribute update through an array never reaches the binding the array was
    made from (`to_array`), shows through a binding taken from the array (`io`)
    consistently in both forms, and through an array copied by its `__dict__`, as
    it did when the fields were an array's only state."""
    store = MemoryStore()
    origin = AsyncArrayIO.build_v3(
        store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,), attributes={"v": 0}
    )
    await origin.store_metadata()
    array = origin.to_array()
    taken = array.io
    clone = object.__new__(AsyncArray)
    clone.__dict__.update(array.__dict__)
    await array.update_attributes({"v": 1})
    assert dict(origin.metadata.attributes) == {"v": 0}
    assert dict(origin.metadata_legacy.attributes) == {"v": 0}
    assert dict(taken.metadata.attributes) == taken.metadata_legacy.attributes == {"v": 1}
    assert clone.attrs == {"v": 1}
    await taken.store_metadata()
    await taken.verify()


@pytest.mark.parametrize(
    ("data_type", "fill_value"), [("float32", "0x7fc00001"), ("float64", "0x7ff8000000000001")]
)
async def test_reading_an_array_leaves_its_binding_as_stored(
    data_type: str, fill_value: str
) -> None:
    """Reading an array made from a binding (its shape, its data) leaves the binding
    holding the document as stored: it still verifies, compares equal to a fresh
    open, and stores that document unchanged (a NaN payload is kept)."""
    store = MemoryStore()
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": data_type,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": fill_value,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {},
    }
    stored = default_buffer_prototype().buffer.from_bytes(json.dumps(document).encode())
    await store.set("a/zarr.json", stored)
    binding = await AsyncArrayIO.open(store, name="a")
    array = binding.to_array()
    _ = array.shape
    await array.getitem(slice(None))
    assert binding == await AsyncArrayIO.open(store, name="a")
    await binding.verify()
    await binding.store_metadata()
    rewritten = await store.get("a/zarr.json", prototype=default_buffer_prototype())
    assert rewritten is not None
    assert json.loads(rewritten.to_bytes()) == document


class _LoggingBinding(AsyncArrayIO):
    """A binding subclass, as a backend would define one."""


def test_an_array_keeps_its_binding_class() -> None:
    """A change to an array's state, and pickling, keep the class of the binding
    it was made from."""
    binding = _LoggingBinding.build_v3(
        MemoryStore(), name="a", shape=(4,), data_type="uint8", chunk_grid=(4,)
    )
    array = binding.to_array()
    for kept in (array, pickle.loads(pickle.dumps(array))):
        sync(kept.resize((6,)))
        assert type(kept.io) is _LoggingBinding
    assert type(array.with_config({"order": "F"}).io) is _LoggingBinding


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


class _NativeBinding(AsyncArrayIO):
    """A binding that derives state of its own (a native handle's shape) in `_bind`."""

    native_shape: tuple[int, ...]
    native_where: str

    def _bind(self, source: Any, store_path: StorePath, *args: Any) -> None:
        super()._bind(source, store_path, *args)
        self.native_shape = tuple(source.shape)
        self.native_where = str(store_path)


async def test_every_route_rebinds_through_bind() -> None:
    """Every way a binding is made or changed goes through `_bind`, so state a
    subclass derives there follows the metadata, also after a resize."""
    store = MemoryStore()
    built = _NativeBinding.build_v3(store, name="a", shape=(4,), data_type="uint8", chunk_grid=(2,))
    await built.store_metadata()
    array = built.to_array()
    await array.resize((6,))
    bindings = [
        built,
        await _NativeBinding.open(store, name="a"),
        built.with_metadata(built.metadata),
        built.with_store(MemoryStore()),
        built.with_config({"order": "F"}),
        built.with_context(Context(codec_pipeline=_MarkedPipeline)),
        pickle.loads(pickle.dumps(built)),
        array.io,
        array.with_config({"order": "F"}).io,
        pickle.loads(pickle.dumps(array)).io,
        copy.copy(array).io,
    ]
    moved = copy.copy(array)
    object.__setattr__(moved, "store_path", StorePath(MemoryStore(), "b"))
    bindings.append(moved.io)
    for binding in bindings:
        assert isinstance(binding, _NativeBinding)
        assert binding.native_shape == binding.metadata.shape
        assert binding.native_where == str(binding.store_path)


@pytest.mark.parametrize("derive", ["with_config", "pickle", "deepcopy"])
async def test_an_array_over_a_model_keeps_the_document_as_stored(derive: str) -> None:
    """An array rebuilt from one over a model binding (`with_config`, pickling, a
    deep copy) keeps the model, so its binding still verifies against the store
    and stores the document unchanged (a NaN payload is kept)."""
    store = MemoryStore()
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": "0x7fc00001",
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {},
    }
    await store.set(
        "a/zarr.json", default_buffer_prototype().buffer.from_bytes(json.dumps(document).encode())
    )
    array = (await AsyncArrayIO.open(store, name="a")).to_array()
    if derive == "with_config":
        rebuilt = array.with_config({"order": "F"})
    elif derive == "pickle":
        rebuilt = pickle.loads(pickle.dumps(array))
    else:
        rebuilt = copy.deepcopy(array)
    await rebuilt.io.verify()
    await rebuilt.io.store_metadata()
    stored = await store.get("a/zarr.json", prototype=default_buffer_prototype())
    assert stored is not None
    assert json.loads(stored.to_bytes()) == document


async def test_bindings_compare_with_attributes_json_cannot_hold() -> None:
    """Equality of model bindings whose attributes were changed in place to values
    JSON cannot hold compares their documents without the attributes, then the
    attributes; reading the model names the value."""
    store = MemoryStore()
    await AsyncArrayIO.build_v3(
        store, name="a", shape=(4,), data_type="float32", chunk_grid=(4,), fill_value=0.0
    ).store_metadata()
    first, second = [(await AsyncArrayIO.open(store, name="a")).to_array() for _ in range(2)]
    for array in (first, second):
        array.metadata.attributes["n"] = np.int64(3)
    assert first.io == second.io
    second.metadata.attributes["n"] = np.int64(4)
    assert first.io != second.io
    with pytest.raises(TypeError, match="int64 is not JSON serializable"):
        _ = first.io.metadata


async def test_a_model_binding_reads_changed_attributes_once() -> None:
    """A model binding whose attributes were changed in place gives one model for
    them until they change again, also for attributes holding NaN."""
    store = MemoryStore()
    await AsyncArrayIO.build_v3(
        store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,)
    ).store_metadata()
    array = (await AsyncArrayIO.open(store, name="a")).to_array()
    array.metadata.attributes["x"] = float("nan")
    binding = array.io
    assert binding.metadata is binding.metadata
    assert math.isnan(cast("float", binding.metadata.attributes["x"]))


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


async def test_with_config_keeps_what_a_subclass_init_built() -> None:
    """`with_config` rebuilds through the constructor and keeps the metadata a
    subclass `__init__` built, as it always has, also over a model binding."""
    store = MemoryStore()
    await AsyncArrayIO.build_v3(
        store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,)
    ).store_metadata()
    for array in (
        _StampingAsyncArray(zarr.open_array(store, path="a").metadata, StorePath(store, "a")),
        _StampingAsyncArray._from_io(await AsyncArrayIO.open(store, name="a")),
    ):
        before = cast("int", array.attrs.get("n", 0))
        assert array.with_config({"order": "F"}).attrs["n"] == before + 1


async def test_a_pickled_array_carries_its_model_as_bytes() -> None:
    """The model a pickled array carries is its documents' bytes, so a zarr without
    zarr-metadata loads the pickle too."""
    store = MemoryStore()
    await AsyncArrayIO.build_v3(
        store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,)
    ).store_metadata()
    array = (await AsyncArrayIO.open(store, name="a")).to_array()
    pickled = pickle.dumps(array)
    assert b"zarr_metadata" not in pickled
    restored = pickle.loads(pickled).io
    assert restored == array.io
    assert not isinstance(restored._source, ArrayV2Metadata | ArrayV3Metadata)


class _UnpicklableBinding(AsyncArrayIO):
    """A binding deriving state that does not pickle, which it leaves out of its own."""

    lock: Any

    def _bind(self, *args: Any) -> None:
        super()._bind(*args)
        self.lock = threading.Lock()

    def __getstate__(self) -> dict[str, object]:
        state = super().__getstate__()
        state.pop("lock")
        return state


def test_a_binding_derives_its_state_again_when_unpickled() -> None:
    binding = _UnpicklableBinding.build_v3(
        MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,)
    )
    for restored in (pickle.loads(pickle.dumps(binding)), copy.deepcopy(binding)):
        assert restored == binding
        assert restored.lock is not binding.lock


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


class _AttributeNamedLikeTheCacheEntry(AsyncArrayIO):
    """A binding subclass with an attribute of the name a carried cache entry had."""

    _stored_entry: str


async def test_a_pickled_binding_keeps_its_attributes_apart_from_what_it_carries() -> None:
    """What a binding carries in a pickle takes no name of a subclass's, and the
    model derived from attributes changed in place is not pickled."""
    store = MemoryStore()
    await AsyncArrayIO.build_v3(
        store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,)
    ).store_metadata()
    binding = _AttributeNamedLikeTheCacheEntry.build_v3(
        store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,)
    )
    binding._stored_entry = "mine"
    assert pickle.loads(pickle.dumps(binding))._stored_entry == "mine"

    array = (await AsyncArrayIO.open(store, name="a")).to_array()
    array.metadata.attributes["changed"] = 1
    assert dict(array.io.metadata.attributes) == {"changed": 1}
    assert array.io.__getstate__()["_attributes_cache"] is None
    assert dict(pickle.loads(pickle.dumps(array.io)).metadata.attributes) == {"changed": 1}


async def test_attributes_changed_in_place_keep_their_order() -> None:
    store = MemoryStore()
    await AsyncArrayIO.build_v3(
        store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,)
    ).store_metadata()
    array = (await AsyncArrayIO.open(store, name="a")).to_array()
    array.metadata.attributes.update({"zeta": 1, "alpha": {"y": 1, "b": 2}})
    attributes = array.io.metadata.attributes
    assert list(attributes) == ["zeta", "alpha"]
    assert list(cast("Mapping[str, Any]", attributes["alpha"])) == ["y", "b"]

    # a change of order alone is a change
    stored = array.metadata.attributes
    alpha = stored.pop("zeta")
    stored["zeta"] = alpha
    assert list(array.io.metadata.attributes) == ["alpha", "zeta"]


async def test_a_strict_v2_read_reports_a_zero_chunk_size_beside_others() -> None:
    """A chunk size of 0, which the model accepts, is reported with the problems
    the model reports for the other chunk sizes."""
    store = MemoryStore()
    await _store_v2_document(store, {**_V2_DOCUMENT, "shape": [4, 4], "chunks": [-1, 0]})
    with pytest.raises(MetadataValidationError) as error:
        await AsyncArrayIO.open(store, name="a")
    assert ("chunks", 1) in [p.loc for p in error.value.problems]


async def test_a_pickle_whose_metadata_was_replaced_binds_the_new_metadata() -> None:
    """A pickle that another zarr loaded, resized and pickled again (keeping the
    binding's state as it was) binds the metadata it now holds, not the stale model."""
    store = MemoryStore()
    await AsyncArrayIO.build_v3(
        store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,)
    ).store_metadata()
    array = (await AsyncArrayIO.open(store, name="a")).to_array()
    state = cast("dict[str, Any]", array.__getstate__())
    resized = dataclasses.replace(state["metadata"], shape=(8,))
    revived = object.__new__(AsyncArray)
    revived.__setstate__({**state, "metadata": resized})
    assert revived.io.metadata.shape == (8,)


@pytest.mark.parametrize("chunks", [[False, 2], [0.0, 2]])
async def test_a_strict_v2_read_reports_a_rejected_chunk_size_once(chunks: list[Any]) -> None:
    store = MemoryStore()
    await _store_v2_document(store, {**_V2_DOCUMENT, "shape": [4, 4], "chunks": chunks})
    with pytest.raises(MetadataValidationError) as error:
        await AsyncArrayIO.open(store, name="a")
    assert [p.loc for p in error.value.problems] == [("chunks",)]


async def test_a_model_binding_reads_attributes_with_mixed_key_types() -> None:
    """Attributes changed in place to hold an int key read as JSON writes them."""
    store = MemoryStore()
    await AsyncArrayIO.build_v3(
        store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,), attributes={"b": 1}
    ).store_metadata()
    array = (await AsyncArrayIO.open(store, name="a")).to_array()
    cast("dict[Any, Any]", array.metadata.attributes)[1] = "x"
    assert dict(array.io.metadata.attributes) == {"b": 1, "1": "x"}


class _OwnPipeline(BatchedCodecPipeline):
    """A pipeline class a backend binding chooses for itself."""


class _OwnPipelineBinding(AsyncArrayIO):
    """A binding whose `_bind` runs a pipeline class of its own, as a backend might."""

    def _bind(self, *args: Any) -> None:
        *rest, _ = args
        bind: Any = super()._bind
        bind(*rest, _OwnPipeline)


def test_with_config_runs_the_pipeline_a_binding_chose() -> None:
    array = _OwnPipelineBinding.build_v3(
        MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,)
    ).to_array()
    configured = array.with_config({"order": "F"})
    assert configured.io._pipeline_class is _OwnPipeline
    assert isinstance(configured.codec_pipeline, _OwnPipeline)


class _AssignedPipeline(BatchedCodecPipeline):
    """A pipeline class a subclass `__init__` assigns."""


class _PipelineAssigningAsyncArray(AsyncArray[Any]):
    """A subclass whose `__init__` assigns a pipeline of its own."""

    def __init__(self, metadata: Any, store_path: StorePath, config: Any = None) -> None:
        super().__init__(metadata, store_path, config)
        pipeline = create_codec_pipeline(self.metadata, pipeline_class=_AssignedPipeline)
        object.__setattr__(self, "codec_pipeline", pipeline)


def test_with_config_prefers_a_binding_pipeline_over_an_assigned_one() -> None:
    """A binding that chose a pipeline class runs it, over a pipeline a subclass
    `__init__` assigned; without such a binding, the assigned pipeline runs."""
    binding = _OwnPipelineBinding.build_v3(
        MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,)
    )
    chosen = _PipelineAssigningAsyncArray._from_io(binding).with_config({"order": "F"})
    assert isinstance(chosen.codec_pipeline, _OwnPipeline)

    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    assigned = _PipelineAssigningAsyncArray(created.metadata, created.store_path)
    assert isinstance(assigned.with_config({"order": "F"}).codec_pipeline, _AssignedPipeline)


def test_bindings_expose_their_context() -> None:
    context = Context(codec_pipeline=_MarkedPipeline)
    aio = AsyncArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,), context=context
    )
    sync_io = ArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,), context=context
    )
    assert aio.context == sync_io.context == context
    assert aio.with_context(None).context is None
    assert aio.with_context(Context()).context is None


@dataclass(frozen=True, init=False)
class _DecoratedAsyncArray(AsyncArray[Any]):
    """A `@dataclass` subclass with a field and an `__init__` that takes no context."""

    tag: str = "default"

    def __init__(
        self, metadata: Any, store_path: StorePath, config: Any = None, tag: str = "default"
    ) -> None:
        super().__init__(metadata, store_path, config)
        object.__setattr__(self, "tag", tag)


def test_a_subclass_init_without_context_is_rebuilt() -> None:
    """`dataclasses.replace` and `with_config` call a subclass `__init__` that takes
    no context, decorated or not, as they always did; a decorated subclass's own
    field is passed by `replace`, and `with_config` keeps a context."""
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    order_f = parse_array_config({"order": "F"})
    context = Context(codec_pipeline=_MarkedPipeline)
    for cls, replaced_tag in ((_TaggedAsyncArray, "default"), (_DecoratedAsyncArray, "mine")):
        array = cls(created.metadata, created.store_path, tag="mine")
        for rebuilt, tag in (
            (dataclasses.replace(array, config=order_f), replaced_tag),
            (array.with_config(order_f), "default"),
        ):
            assert type(rebuilt) is cls
            assert rebuilt.tag == tag
            assert rebuilt.config.order == "F"
        scoped = cls._from_io(AsyncArrayIO(created.metadata, created.store_path, context=context))
        object.__setattr__(scoped, "tag", "mine")
        configured = scoped.with_config(order_f)
        assert type(configured) is cls
        assert configured.tag == "default"
        assert configured.io.context == context
        assert isinstance(configured.codec_pipeline, _MarkedPipeline)


def test_async_array_takes_context_by_keyword_only() -> None:
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    with pytest.raises(TypeError, match="positional"):
        AsyncArray(created.metadata, created.store_path, None, Context())  # type: ignore[call-overload]


def test_async_array_rejects_a_model_under_any_context() -> None:
    """`AsyncArray` takes legacy metadata or a mapping; a model is refused under a
    context that limits names as without one."""
    model = AsyncArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,)
    ).metadata
    closed = Context(codecs={"bytes": BytesCodec})
    for context in (None, closed):
        with pytest.raises(TypeError, match="got ZarrV3ArrayMetadata"):
            AsyncArray(cast("Any", model), StorePath(MemoryStore()), context=context)


def test_a_context_with_only_unset_kinds_binds_as_none() -> None:
    aio = AsyncArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,), context=Context()
    )
    assert aio.context is None


class _BuiltBinding(AsyncArrayIO):
    """A binding subclass whose own `__init__` sets up state, as a backend's might."""

    ready: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ready = True


def test_builders_run_a_binding_subclass_init() -> None:
    context = Context(codec_pipeline=_MarkedPipeline)
    built = (
        _BuiltBinding.build_v3(MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,)),
        _BuiltBinding.build_v2(
            MemoryStore(), shape=(4,), dtype="uint8", chunks=(4,), context=context
        ),
    )
    for binding in built:
        assert binding.ready


async def test_with_config_under_a_context_shares_metadata_and_builds_one_pipeline() -> None:
    """Under a context, `with_config` keeps the context and its pipeline, builds no
    other pipeline, and shares the metadata object, as it does with no context."""
    closed = Context(codecs={"bytes": BytesCodec}, codec_pipeline=_MarkedPipeline)
    for context in (None, closed):
        array = AsyncArrayIO.build_v3(
            MemoryStore(),
            shape=(4,),
            data_type="uint8",
            chunk_grid=(4,),
            codecs=[BYTES_JSON],
            context=context,
        ).to_array()
        configured = array.with_config({"order": "F"})
        assert configured.metadata is array.metadata
        assert configured.io.context == context
    array = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint8",
        chunk_grid=(4,),
        codecs=[BYTES_JSON],
        context=closed,
    ).to_array()
    with zarr.config.set({"codec_pipeline.path": "no.such.Pipeline"}):
        configured = array.with_config({"order": "F"})
    assert isinstance(configured.codec_pipeline, _MarkedPipeline)


def test_sync_bindings_take_a_context(tmp_path: Path) -> None:
    context = Context(codec_pipeline=_MarkedPipeline)
    built = ArrayIO.build_v2(
        LocalStore(tmp_path), name="a", shape=(4,), dtype="uint8", chunks=(4,), context=context
    )
    built.store_metadata()
    opened = ArrayIO.open(LocalStore(tmp_path), name="a", context=context)
    assert built.context == opened.context == context
    assert opened.with_context(None).context is None
    assert isinstance(opened.to_array().async_array.codec_pipeline, _MarkedPipeline)


@pytest.mark.parametrize("context", [None, Context(data_types=data_type_registry)])
@pytest.mark.parametrize(
    "native", [np.dtype(">u2"), np.dtype([("a", ">i4"), ("n", [("c", ">u2")])])]
)
def test_a_data_type_scope_keeps_the_byte_order(
    context: Context | None, native: np.dtype[Any]
) -> None:
    """Zarr format 3 data type JSON has no byte order; entering a data type scope, or
    leaving it, keeps a big-endian data type (or field) big-endian, as no context does."""
    built = AsyncArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type=native, chunk_grid=(4,), context=context
    )
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype=native)
    bound = AsyncArrayIO(created.metadata, created.store_path, context=context)
    for binding in (built, bound, bound.with_context(None)):
        assert binding.metadata_legacy.dtype.to_native_dtype() == native


@dataclass(frozen=True)
class _TunedZstd(ZstdCodec):
    """A zstd codec with state its JSON does not hold."""

    threads: int = 1


def test_a_closed_codec_scope_keeps_admitted_instances_whole() -> None:
    """An instance of the class the scope maps its name to is bound as given on
    every route, state its JSON does not hold included, also nested in sharding."""
    context = Context(codecs={"bytes": BytesCodec, "zstd": _TunedZstd})
    tuned = _TunedZstd(level=3, threads=8)
    built = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint8",
        chunk_grid=(4,),
        codecs=[BytesCodec(), tuned],
        context=context,
    )
    plain = AsyncArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,), codecs=[BytesCodec(), tuned]
    )
    legacy = plain.metadata_legacy
    for binding in (
        built,
        plain.with_context(context),
        AsyncArrayIO(legacy, plain.store_path, context=context),
        AsyncArray(legacy, plain.store_path, context=context).io,
    ):
        assert binding.metadata_legacy.codecs[1] == tuned  # type: ignore[union-attr]
        assert binding.metadata_legacy.codecs[1].threads == 8  # type: ignore[union-attr]
    sharded = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint8",
        chunk_grid=(4,),
        codecs=[ShardingCodec(chunk_shape=(2,), codecs=[BytesCodec(), tuned])],
    )
    scoped = sharded.with_context(Context(codecs={"sharding_indexed": ShardingCodec}))
    assert scoped.metadata_legacy.codecs[0].codecs[1].threads == 8  # type: ignore[union-attr]


@dataclass(frozen=True, kw_only=True)
class _AliasRegistry(DataTypeRegistry):
    """A registry that matches a name of its own."""

    def match_json(self, data: Any, *, zarr_format: ZarrFormat) -> Any:
        return super().match_json("float32" if data == "real32" else data, zarr_format=zarr_format)


def test_a_context_keeps_the_class_of_a_registry_it_copies() -> None:
    registry = _AliasRegistry()
    registry.register(Float32._zarr_v3_name, Float32)
    context = Context(data_types=registry)
    assert type(context.data_types) is _AliasRegistry
    assert type(parse_dtype("real32", zarr_format=3, context=context)) is Float32


def test_a_closed_data_type_scope_rejects_an_instance_of_another_class() -> None:
    """A data type instance is admitted only if the scope reads its JSON as its own
    class, as a codec instance is."""
    context = Context(data_types=_scoped_uint16_registry())
    with pytest.raises(ValueError, match="reads its JSON as, _ScopedUInt16"):
        parse_dtype(UInt16(), zarr_format=3, context=context)
    assert type(parse_dtype(_ScopedUInt16(), zarr_format=3, context=context)) is _ScopedUInt16


def test_matching_in_a_registry_does_not_copy_it(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _scoped_uint16_registry()
    monkeypatch.setattr(
        context_module,
        "_registry_snapshot",
        mock.Mock(side_effect=AssertionError("the registry was copied")),
    )
    assert type(registry.match_json("uint16", zarr_format=3)) is _ScopedUInt16


def test_with_config_chooses_the_pipeline_anew_unless_the_context_chose_it() -> None:
    """Under a context that sets no pipeline, `with_config` follows the global
    configuration, as with no context; a context's own pipeline is kept."""
    fused = "zarr.core.codec_pipeline.FusedCodecPipeline"
    for context, expected in (
        (Context(data_types=_scoped_uint16_registry()), FusedCodecPipeline),
        (Context(codec_pipeline=_MarkedPipeline), _MarkedPipeline),
    ):
        array = AsyncArrayIO.build_v3(
            MemoryStore(), shape=(4,), data_type="uint16", chunk_grid=(4,), context=context
        ).to_array()
        with zarr.config.set({"codec_pipeline.path": fused}):
            configured = array.with_config({"order": "F"})
        assert type(configured.codec_pipeline) is expected


@dataclass(frozen=True)
class _LazyEntryPoint:
    """An entry point that loads a data type class."""

    loaded: type[Any]

    def load(self) -> type[Any]:
        return self.loaded


def test_a_context_compares_the_same_before_and_after_its_registry_loads() -> None:
    registry = DataTypeRegistry()
    registry._lazy_load_list.append(cast("Any", _LazyEntryPoint(_ScopedUInt16)))
    context = Context(data_types=registry)
    binding = AsyncArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type="uint16", chunk_grid=(4,), context=context
    )
    _ = binding.metadata_legacy
    assert context == Context(data_types=registry)


def test_a_closed_data_type_scope_checks_struct_field_classes() -> None:
    """A structured data type instance is admitted only if the scope reads each of
    its fields as that field's own class."""
    context = Context(data_types=_scoped_uint16_registry_with_struct())
    with pytest.raises(ValueError, match="reads its JSON as"):
        parse_dtype(Struct(fields=(("a", UInt16()),)), zarr_format=3, context=context)
    admitted = Struct(fields=(("a", _ScopedUInt16()),))
    assert parse_dtype(admitted, zarr_format=3, context=context) is admitted


def _scoped_uint16_registry_with_struct() -> DataTypeRegistry:
    registry = _scoped_uint16_registry()
    registry.register(Struct._zarr_v3_name, Struct)
    return registry


def test_leaving_a_codec_scope_keeps_default_codecs_whole() -> None:
    """Leaving a scope rebinds only codecs the scope bound to other classes: a
    default-class codec is kept whole, with the codecs nested in it."""
    tuned = _TunedZstd(level=3, threads=8)
    sharded = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint8",
        chunk_grid=(4,),
        codecs=[ShardingCodec(chunk_shape=(2,), codecs=[BytesCodec(), tuned])],
    )
    left = sharded.with_context(Context(codecs={"sharding_indexed": ShardingCodec})).with_context(
        None
    )
    assert left.metadata_legacy.codecs[0].codecs[1].threads == 8  # type: ignore[union-attr]


def test_rebinding_resolved_metadata_reads_other_classes_by_name() -> None:
    """Metadata already resolved, entering a scope that maps a name to another class
    than its codec's, is read again as the scope's class (a builder refuses such an
    instance)."""
    plain = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint8",
        chunk_grid=(4,),
        codecs=[BytesCodec(), _TunedZstd(level=3, threads=8)],
    )
    scoped = plain.with_context(Context(codecs={"bytes": BytesCodec, "zstd": ZstdCodec}))
    assert type(scoped.metadata_legacy.codecs[1]) is ZstdCodec  # type: ignore[union-attr]


class _ScopedShuffle(Shuffle):
    """A shuffle codec class of a scope's own."""


async def test_a_scoped_numcodecs_class_survives_fitting_to_the_array() -> None:
    """A numcodecs codec fitted to the array's data type stays the scope's class."""
    store = MemoryStore()
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "int32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "numcodecs.shuffle", "configuration": {}},
        ],
        "attributes": {},
    }
    await store.set(
        "a/zarr.json", default_buffer_prototype().buffer.from_bytes(json.dumps(document).encode())
    )
    context = Context(codecs={"bytes": BytesCodec, "numcodecs.shuffle": _ScopedShuffle})
    binding = await AsyncArrayIO.open(store, name="a", context=context)
    assert type(binding.metadata_legacy.codecs[1]) is _ScopedShuffle  # type: ignore[union-attr]


async def test_a_scoped_write_passes_a_recorded_document_its_scope_cannot_read() -> None:
    """A document the store's cache records as needing a repair, which the writing
    binding's scope cannot read, describes another array: the write goes ahead."""
    store = MemoryStore()
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [3],
        "data_type": "int32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [0]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    await store.set(
        "a/zarr.json", default_buffer_prototype().buffer.from_bytes(json.dumps(document).encode())
    )
    with pytest.warns(ZarrUserWarning):
        await open_array(store=store, path="a")
    context = Context(data_types=_scoped_uint16_registry())
    binding = AsyncArrayIO.build_v3(
        store, name="a", shape=(3,), data_type="uint16", chunk_grid=(3,), context=context
    )
    await binding.setitem(slice(None), np.array([1, 2, 3], dtype="uint16"))
    # Another array's layout: its document is left as it was.
    stored = await store.get("a/zarr.json", prototype=default_buffer_prototype())
    assert stored is not None
    assert json.loads(stored.to_bytes())["chunk_grid"]["configuration"]["chunk_shape"] == [0]


class _ResubmittingAsyncArray(AsyncArray[Any]):
    """A subclass whose `__init__` gives its metadata as a dict."""

    def __init__(self, metadata: Any, store_path: StorePath, config: Any = None) -> None:
        super().__init__(metadata.to_dict(), store_path, config)


def test_with_config_resolves_substituted_metadata_under_the_context() -> None:
    context = Context(data_types=_scoped_uint16_registry())
    binding = AsyncArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type="uint16", chunk_grid=(4,), context=context
    )
    array = _ResubmittingAsyncArray._from_io(binding)
    configured = array.with_config({"order": "F"})
    assert type(configured.metadata.dtype) is _ScopedUInt16
    assert configured.io.context == context


def test_bindings_take_context_by_keyword_only() -> None:
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    for binding_class in (AsyncArrayIO, ArrayIO):
        with pytest.raises(TypeError, match="positional"):
            binding_class(created.metadata, created.store_path, None, Context())  # type: ignore[call-arg]


def test_with_config_without_a_context_runs_the_constructor() -> None:
    """Without a context, `with_config` builds the array through its constructor, as
    it always has."""
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,))
    array = AsyncArray(created.metadata, created.store_path)
    with mock.patch.object(
        AsyncArray, "__init__", autospec=True, side_effect=AsyncArray.__init__
    ) as init:
        array.with_config({"order": "F"})
    assert init.call_count == 1


def test_the_builder_keeps_a_scoped_numcodecs_class_through_fitting() -> None:
    """`build_array_metadata_v3` under a codec scope returns the scope's class for a
    codec that fitting to the array rebuilt, as reading a document does."""
    context = Context(codecs={"bytes": BytesCodec, "numcodecs.shuffle": _ScopedShuffle})
    metadata = build_array_metadata_v3(
        shape=(4,),
        data_type="int32",
        chunk_grid=(4,),
        codecs=[BYTES_JSON, {"name": "numcodecs.shuffle", "configuration": {}}],
        context=context,
    )
    assert type(metadata.codecs[1]) is _ScopedShuffle


def test_a_name_limiting_scope_shares_metadata_it_does_not_change() -> None:
    """Under a scope that reads every name as the metadata already holds it, an
    `AsyncArray` binds the object it is given, as with no context."""
    created = zarr.create_array(MemoryStore(), shape=(4,), dtype="uint8", chunks=(4,))
    context = Context(data_types=data_type_registry)
    array = AsyncArray(created.metadata, created.store_path, context=context)
    assert array.metadata is created.metadata


def test_with_config_builds_its_pipeline_over_metadata_resolved_under_the_scope() -> None:
    """A subclass `__init__` that substitutes metadata read under the defaults: the
    array is resolved again under its context, and its pipeline is built over that."""
    context = Context(codecs={"bytes": _UnregisteredBytes})
    binding = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint8",
        chunk_grid=(4,),
        codecs=[BYTES_JSON],
        context=context,
    )
    configured = _ResubmittingAsyncArray._from_io(binding).with_config({"order": "F"})
    pipeline = cast("BatchedCodecPipeline", configured.codec_pipeline)
    assert type(pipeline.array_bytes_codec) is _UnregisteredBytes


class _LocalPipelineBinding(AsyncArrayIO):
    """A binding whose `_bind` chooses its own pipeline for a local store only."""

    def _bind(self, *args: Any) -> None:
        *rest, pipeline_class = args
        local = isinstance(rest[1].store, LocalStore)
        bind: Any = super()._bind
        bind(*rest, _OwnPipeline if local else pipeline_class)


def test_a_binding_made_from_another_gets_its_contexts_pipeline(tmp_path: Path) -> None:
    """Under a context that chose a pipeline, a binding made from one whose `_bind`
    chose another class for its store gets the context's class again."""
    context = Context(codec_pipeline=_MarkedPipeline)
    local = _LocalPipelineBinding.build_v3(
        LocalStore(tmp_path), shape=(4,), data_type="uint8", chunk_grid=(4,), context=context
    )
    assert local._pipeline_class is _OwnPipeline
    assert local.with_store(MemoryStore())._pipeline_class is _MarkedPipeline


class _ShapePipelineBinding(AsyncArrayIO):
    """A binding whose `_bind` chooses its own pipeline for small arrays only."""

    def _bind(self, *args: Any) -> None:
        source, *rest, pipeline_class = args
        bind: Any = super()._bind
        bind(source, *rest, _OwnPipeline if source.shape[0] <= 4 else pipeline_class)


async def test_a_resize_runs_the_pipeline_its_binding_records() -> None:
    """A resize after which `_bind` chooses another pipeline class runs that class."""
    context = Context(codec_pipeline=_MarkedPipeline)
    array = _ShapePipelineBinding.build_v3(
        MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,), context=context
    ).to_array()
    assert isinstance(array.codec_pipeline, _OwnPipeline)
    await array.resize((8,))
    assert array.io._pipeline_class is _MarkedPipeline
    assert isinstance(array.codec_pipeline, _MarkedPipeline)


async def test_a_scoped_write_repairs_a_matching_document_its_scope_cannot_read() -> None:
    """A recorded document the scope cannot read is compared by its layout, read
    under the defaults, as with no context: one laid out as the binding is gets
    its repair stored before the write."""
    store = MemoryStore()
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [3],
        "data_type": "int32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [0]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    await store.set(
        "a/zarr.json", default_buffer_prototype().buffer.from_bytes(json.dumps(document).encode())
    )
    with pytest.warns(ZarrUserWarning):
        await open_array(store=store, path="a")
    context = Context(data_types=_scoped_uint16_registry())
    binding = AsyncArrayIO.build_v3(
        store, name="a", shape=(3,), data_type="uint16", chunk_grid=(1,), context=context
    )
    await binding.setitem(slice(None), np.array([1, 2, 3], dtype="uint16"))
    stored = await store.get("a/zarr.json", prototype=default_buffer_prototype())
    assert stored is not None
    chunk_grid = json.loads(stored.to_bytes())["chunk_grid"]
    assert chunk_grid["configuration"]["chunk_shape"] == [1]


def test_arrays_expose_their_context() -> None:
    context = Context(codec_pipeline=_MarkedPipeline)
    binding = ArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,), context=context
    )
    array = binding.to_array()
    assert array.async_array.context == array.io.context == context


class _OrderPipelineBinding(AsyncArrayIO):
    """A binding whose `_bind` chooses its own pipeline for C order only."""

    def _bind(self, source: Any, store_path: Any, config: Any, *rest: Any) -> None:
        *context, pipeline_class = rest
        chosen = _OwnPipeline if config.order == "C" else pipeline_class
        bind: Any = super()._bind
        bind(source, store_path, config, *context, chosen)


def test_setting_a_config_runs_the_pipeline_its_binding_records() -> None:
    """Setting an array's config, after which `_bind` chooses another pipeline class,
    runs that class."""
    context = Context(codec_pipeline=_MarkedPipeline)
    array = _OrderPipelineBinding.build_v3(
        MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,), context=context
    ).to_array()
    assert isinstance(array.codec_pipeline, _OwnPipeline)
    object.__setattr__(array, "config", parse_array_config({"order": "F"}))
    assert array.io._pipeline_class is _MarkedPipeline
    assert isinstance(array.codec_pipeline, _MarkedPipeline)


async def test_a_scoped_write_reads_a_newer_document_its_scope_cannot_read() -> None:
    """The document stored now is laid out under the defaults when the binding's
    scope cannot read it, as the recorded one is: a newer valid document another
    writer stored lets the write go ahead, as with no context."""
    store = MemoryStore()
    prototype = default_buffer_prototype()
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [3],
        "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [0]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    await store.set("a/zarr.json", prototype.buffer.from_bytes(json.dumps(document).encode()))
    with pytest.warns(ZarrUserWarning):
        await open_array(store=store, path="a")
    newer = {
        **document,
        "data_type": "int32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1]}},
    }
    await store.set("a/zarr.json", prototype.buffer.from_bytes(json.dumps(newer).encode()))
    context = Context(data_types=_scoped_uint16_registry())
    binding = AsyncArrayIO.build_v3(
        store, name="a", shape=(3,), data_type="uint16", chunk_grid=(1,), context=context
    )
    await binding.setitem(slice(None), np.array([1, 2, 3], dtype="uint16"))
    assert {"a/c/0", "a/c/1", "a/c/2"} <= await _store_keys(store)
    stored = await store.get("a/zarr.json", prototype=default_buffer_prototype())
    assert stored is not None
    assert json.loads(stored.to_bytes()) == newer


async def test_a_scoped_write_over_a_document_no_registry_reads_raises() -> None:
    """A recorded document that neither the writing binding's scope nor the
    defaults can read refuses the write, storing nothing."""
    store = MemoryStore()
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [3],
        "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [0]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "local_bytes", "configuration": {"endian": "little"}}],
        "attributes": {},
    }
    with pytest.warns(ZarrUserWarning):
        AsyncArray(
            cast("Any", document),
            StorePath(store, "a"),
            context=Context(codecs={"local_bytes": _LocalBytes}),
        )
    binding = AsyncArrayIO.build_v3(
        store,
        name="a",
        shape=(3,),
        data_type="uint16",
        chunk_grid=(1,),
        context=Context(data_types=_scoped_uint16_registry()),
    )
    with pytest.raises(UnknownCodecError):
        await binding.setitem(slice(None), np.array([1, 2, 3], dtype="uint16"))
    assert await _store_keys(store) == set()


def test_without_a_context_a_derived_binding_runs_the_class_bind_chose(tmp_path: Path) -> None:
    """With no context, the pipeline class a `_bind` chose is the one bindings made
    from that binding are given."""
    local = _LocalPipelineBinding.build_v3(
        LocalStore(tmp_path), shape=(4,), data_type="uint8", chunk_grid=(4,)
    )
    assert local._pipeline_class is _OwnPipeline
    assert local.with_store(MemoryStore())._pipeline_class is _OwnPipeline


def test_leaving_a_scope_fails_for_a_name_the_defaults_do_not_know() -> None:
    binding = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint16",
        chunk_grid=(4,),
        codecs=[{"name": "local_bytes", "configuration": {"endian": "little"}}],
        context=Context(codecs={"local_bytes": _LocalBytes}),
    )
    with pytest.raises(UnknownCodecError, match="local_bytes"):
        binding.with_context(None)


@dataclass(frozen=True)
class _PlainZstd(ZstdCodec):
    """A zstd class whose constructor takes no `checksum`."""

    def __init__(self, *, level: int = 0) -> None:
        super().__init__(level=level)


async def test_a_scoped_write_reads_a_document_a_scope_class_refuses_under_the_defaults() -> None:
    """A document a scope's class refuses to read (its constructor rejects the stored
    configuration) is laid out under the defaults, as with no context."""
    store = MemoryStore()
    prototype = default_buffer_prototype()
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [3],
        "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [0]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 0, "checksum": True}},
        ],
    }
    await store.set("a/zarr.json", prototype.buffer.from_bytes(json.dumps(document).encode()))
    with pytest.warns(ZarrUserWarning):
        await open_array(store=store, path="a")
    context = Context(codecs={"bytes": BytesCodec, "zstd": _PlainZstd})
    binding = AsyncArrayIO.build_v3(
        store,
        name="a",
        shape=(3,),
        data_type="uint16",
        chunk_grid=(1,),
        codecs=[BYTES_JSON],
        context=context,
    )
    await binding.setitem(slice(None), np.array([1, 2, 3], dtype="uint16"))
    assert {"a/c/0", "a/c/1", "a/c/2"} <= await _store_keys(store)


@dataclass(frozen=True)
class _ScopedGzip(GzipCodec):
    """A gzip class only a scope knows, under a name of its own."""

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "scoped_gzip", "configuration": {"level": self.level}}

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        configuration = cast("dict[str, JSON]", data["configuration"])
        return cls(level=cast("int", configuration["level"]))


async def test_a_scoped_write_reads_each_document_under_its_scope_first() -> None:
    """The current document is read under the binding's scope even when the recorded
    one needed the defaults."""
    store = MemoryStore()
    prototype = default_buffer_prototype()
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [3],
        "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [0]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "gzip", "configuration": {"level": 1}},
        ],
    }
    await store.set("a/zarr.json", prototype.buffer.from_bytes(json.dumps(document).encode()))
    with pytest.warns(ZarrUserWarning):
        await open_array(store=store, path="a")
    newer = {
        **document,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1]}},
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "scoped_gzip", "configuration": {"level": 1}},
        ],
    }
    await store.set("a/zarr.json", prototype.buffer.from_bytes(json.dumps(newer).encode()))
    context = Context(codecs={"bytes": BytesCodec, "scoped_gzip": _ScopedGzip})
    binding = AsyncArrayIO.build_v3(
        store,
        name="a",
        shape=(3,),
        data_type="uint16",
        chunk_grid=(1,),
        codecs=[BYTES_JSON],
        context=context,
    )
    await binding.setitem(slice(None), np.array([1, 2, 3], dtype="uint16"))
    assert {"a/c/0", "a/c/1", "a/c/2"} <= await _store_keys(store)


def test_a_codec_scope_held_under_an_alias_admits_no_instance() -> None:
    """A scope names codecs by the name each writes: one that holds `BytesCodec`
    only as `endian` admits no `BytesCodec` instance (it writes `bytes`, which the
    scope could not read back), whether built or entering the scope."""
    context = Context(codecs={"endian": BytesCodec})
    with pytest.raises(UnknownCodecError, match="'bytes' is not in this context"):
        AsyncArrayIO.build_v3(
            MemoryStore(),
            shape=(4,),
            data_type="uint16",
            chunk_grid=(4,),
            codecs=[BytesCodec()],
            context=context,
        )
    with pytest.raises(UnknownCodecError, match="'bytes' is not in this context") as raised:
        AsyncArrayIO.build_v3(
            MemoryStore(), shape=(4,), data_type="uint16", chunk_grid=(4,), codecs=[BytesCodec()]
        ).with_context(context)
    assert raised.value.__notes__ == ["'bytes' is the name a BytesCodec writes."]


async def test_a_rebind_that_switches_pipeline_builds_it_on_first_use() -> None:
    """A resize has stored its shape before the binding rebinds, so a pipeline class
    `_bind` switches to is built when next used, never failing the resize."""
    context = Context(codec_pipeline=_MarkedPipeline)
    array = _ShapePipelineBinding.build_v3(
        MemoryStore(), shape=(4,), data_type="uint8", chunk_grid=(4,), context=context
    ).to_array()
    await array.resize((8,))
    assert array.io._codec_pipeline_cache is None
    assert isinstance(array.codec_pipeline, _MarkedPipeline)


class _NamedBinding(AsyncArrayIO):
    """A binding whose constructor takes a signature of its own."""

    def __init__(self, name: str) -> None:
        raise AssertionError("derivations do not run the constructor")


def test_with_context_derives_without_the_constructor() -> None:
    """`with_context` makes its binding as the other derivations do, so a subclass
    whose constructor has its own signature derives, and a legacy source is
    never shared with the original binding."""
    built = AsyncArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type="uint16", chunk_grid=(4,), codecs=[BytesCodec()]
    )
    binding = object.__new__(_NamedBinding)
    binding._bind(
        built._source,
        built.store_path,
        built.config,
        None,
        built._pipeline_class,
    )
    context = Context(codecs={"bytes": BytesCodec})
    derived = binding.with_context(context)
    assert type(derived) is _NamedBinding
    assert derived.context == context
    assert derived.metadata_legacy == binding.metadata_legacy
    assert derived._source is not binding._source
    assert type(derived.with_context(None)) is _NamedBinding


class _ResolvedFirstElsewhere:
    """Stands in for `parse_array_metadata` while another thread stores its own
    resolution of `binding` first."""

    def __init__(self, binding: AsyncArrayIO, elsewhere: ArrayMetadata) -> None:
        self.binding = binding
        self.elsewhere = elsewhere

    def __call__(self, *args: Any, **kwargs: Any) -> ArrayMetadata:
        resolved = parse_array_metadata(*args, **kwargs)
        self.binding._resolved_cache = self.elsewhere
        return resolved


async def test_bindings_resolving_at_once_share_the_first_resolution() -> None:
    """A binding over a model keeps the legacy metadata resolved first, so threads
    resolving it at once share one object, whose attributes an array changes in place."""
    store = MemoryStore()
    built = AsyncArrayIO.build_v3(store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,))
    await built.store_metadata()
    binding = await AsyncArrayIO.open(store, name="a")
    elsewhere = parse_array_metadata(built.metadata_legacy.to_dict(), "a")
    with mock.patch(
        "zarr.core.array.parse_array_metadata", _ResolvedFirstElsewhere(binding, elsewhere)
    ):
        assert binding.metadata_legacy is elsewhere
    assert binding.metadata_legacy is elsewhere


def test_a_data_type_scope_rereads_without_unstable_specification_warnings() -> None:
    """Reading a resolved data type again under a scope, or leaving one, writes
    nothing, so it gives no warning of an unstable specification (warnings are
    errors in this suite); only storing the metadata does, as with no context."""
    registry = DataTypeRegistry()
    registry.register(NullTerminatedBytes._zarr_v3_name, NullTerminatedBytes)
    context = Context(data_types=registry)
    data_type = NullTerminatedBytes(length=4)
    scoped = AsyncArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type=data_type, chunk_grid=(4,), context=context
    )
    left = scoped.with_context(None)
    entered = AsyncArrayIO.build_v3(
        MemoryStore(), shape=(4,), data_type=data_type, chunk_grid=(4,)
    ).with_context(context)
    assert scoped.metadata_legacy.dtype == left.metadata_legacy.dtype == data_type
    assert entered.metadata_legacy.dtype == data_type


def test_bindings_that_run_different_pipelines_are_unequal() -> None:
    """A binding's pipeline class is part of it: bindings over one document made
    under different global pipeline settings run different pipelines, so they are
    unequal."""
    store = MemoryStore()
    default = AsyncArrayIO.build_v3(store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,))
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"}):
        fused = AsyncArrayIO.build_v3(
            store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,)
        )
    assert default._pipeline_class is not fused._pipeline_class
    assert default != fused


def test_a_data_type_scope_leaves_the_warning_filters_alone() -> None:
    """Silencing the unstable-specification warning for a reread touches no global
    warning state: a context-free warning shown once stays shown once around
    scoped reads, and the filters are unchanged."""
    registry = DataTypeRegistry()
    registry.register(NullTerminatedBytes._zarr_v3_name, NullTerminatedBytes)
    context = Context(data_types=registry)
    data_type = NullTerminatedBytes(length=4)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("default")
        filters = list(warnings.filters)
        for _ in range(3):
            data_type.to_json(zarr_format=3)
            parse_dtype(data_type, zarr_format=3, context=context)
        assert warnings.filters == filters
    assert [type(w.message) for w in shown] == [UnstableSpecificationWarning]


class _FixedSignatureBinding(AsyncArrayIO):
    """A binding subclass whose constructor was written without a `context`."""

    def __init__(
        self, metadata: Any, store_path: StorePath, config: ArrayConfigLike | None = None
    ) -> None:
        super().__init__(metadata, store_path, config)


async def test_builders_and_open_need_no_context_keyword_without_a_context() -> None:
    store = MemoryStore()
    built = _FixedSignatureBinding.build_v3(
        store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,)
    )
    await built.store_metadata()
    v2 = _FixedSignatureBinding.build_v2(MemoryStore(), shape=(4,), dtype="uint8", chunks=(4,))
    opened = await _FixedSignatureBinding.open(store, name="a")
    assert type(built) is type(v2) is type(opened) is _FixedSignatureBinding
    assert opened.metadata_legacy == built.metadata_legacy


async def test_an_opened_binding_entering_a_codec_scope_resolves_its_classes() -> None:
    """A binding over a model (one `open` made) entering a codec scope resolves its
    stored names through the scope when its legacy metadata is first built."""
    store = MemoryStore()
    await AsyncArrayIO.build_v3(
        store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,), codecs=[BytesCodec()]
    ).store_metadata()
    opened = await AsyncArrayIO.open(store, name="a")
    scoped = opened.with_context(Context(codecs={"bytes": _UnregisteredBytes}))
    assert [type(codec) for codec in scoped.metadata_legacy.codecs] == [_UnregisteredBytes]  # type: ignore[union-attr]


async def test_an_opened_binding_entering_a_scope_without_its_codec_fails_on_use() -> None:
    store = MemoryStore()
    await AsyncArrayIO.build_v3(
        store, name="a", shape=(4,), data_type="uint8", chunk_grid=(4,)
    ).store_metadata()
    opened = await AsyncArrayIO.open(store, name="a")
    scoped = opened.with_context(Context(codecs={"zstd": ZstdCodec}))
    with pytest.raises(UnknownCodecError, match="'bytes' is not in this context"):
        await scoped.getitem(slice(None))


def test_a_scope_class_refusing_a_codec_configuration_names_it() -> None:
    built = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint8",
        chunk_grid=(4,),
        codecs=[BytesCodec(), ZstdCodec(level=0, checksum=True)],
    )
    context = Context(codecs={"bytes": BytesCodec, "zstd": _PlainZstd})
    with pytest.raises(ZarrMetadataValidationError, match="'zstd'.*_PlainZstd, which refuses"):
        built.with_context(context)
    # The same refusal read from JSON reports alike.
    with pytest.raises(ZarrMetadataValidationError, match="'zstd'.*_PlainZstd, which refuses"):
        AsyncArrayIO.build_v3(
            MemoryStore(),
            shape=(4,),
            data_type="uint8",
            chunk_grid=(4,),
            codecs=[
                {"name": "bytes"},
                {"name": "zstd", "configuration": {"level": 0, "checksum": True}},
            ],
            context=context,
        )


class _RenamingZstd(ZstdCodec):
    """A zstd class that writes a name of its own."""

    def to_dict(self) -> dict[str, JSON]:
        return {**super().to_dict(), "name": "my.zstd"}


def test_a_scope_class_rebuilding_a_codec_must_read_back_in_the_scope() -> None:
    built = AsyncArrayIO.build_v3(
        MemoryStore(),
        shape=(4,),
        data_type="uint8",
        chunk_grid=(4,),
        codecs=[BytesCodec(), ZstdCodec()],
    )
    context = Context(codecs={"bytes": BytesCodec, "zstd": _RenamingZstd})
    with pytest.raises(UnknownCodecError, match="'my.zstd' is not in this context") as raised:
        built.with_context(context)
    assert raised.value.__notes__ == [
        "'my.zstd' is the name a _RenamingZstd writes, read from 'zstd'."
    ]
