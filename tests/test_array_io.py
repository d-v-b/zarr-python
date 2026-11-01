"""Tests for the explicit-IO array layer, `zarr.ArrayIO` and `zarr.AsyncArrayIO`."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import pickle
import threading
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING, Any, SupportsIndex, cast

import numpy as np
import pytest
from zarr_metadata import UNSET, MetadataValidationError, ZarrV3ArrayMetadata
from zarr_metadata import ZarrV2ArrayMetadata as ModelV2

import zarr
from zarr import ArrayIO, AsyncArray, AsyncArrayIO
from zarr.api.asynchronous import open_array
from zarr.core.buffer import default_buffer_prototype
from zarr.core.codec_pipeline import BatchedCodecPipeline, FusedCodecPipeline
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.stored import stored_documents
from zarr.core.sync import sync
from zarr.dtype import Int16
from zarr.errors import ArrayNotFoundError, UnknownCodecError, ZarrUserWarning
from zarr.storage import LocalStore, MemoryStore, StorePath

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

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
    from types import MappingProxyType

    aio = AsyncArrayIO.build_v3(
        MemoryStore(), name="foo", shape=(4,), data_type="uint8", chunk_grid=(4,)
    )
    rebound = aio.with_metadata(cast("Any", MappingProxyType(dict(aio.metadata.to_json()))))
    assert rebound == aio
    await rebound.store_metadata()


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
        assert set(vars(restored)) == {"_io", "tag"}


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
    import zarr.core.metadata.stored as stored_module

    def refuse(_: object) -> None:
        raise AssertionError("a written document was parsed")

    monkeypatch.setattr(stored_module, "buffer_to_json_object", refuse)
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
