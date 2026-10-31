"""Tests for the explicit-IO array layer, `zarr.ArrayIO` and `zarr.AsyncArrayIO`."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from zarr_metadata import UNSET, ZarrV3ArrayMetadata
from zarr_metadata import ZarrV2ArrayMetadata as ModelV2

import zarr
from zarr import ArrayIO, AsyncArrayIO
from zarr.api.asynchronous import open_array
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.codec_pipeline import BatchedCodecPipeline, FusedCodecPipeline
from zarr.core.dtype import UInt16, data_type_registry
from zarr.core.dtype.registry import DataTypeRegistry
from zarr.core.metadata import ArrayV2Metadata
from zarr.errors import ArrayNotFoundError, UnknownCodecError
from zarr.experimental import Context
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

BYTES_JSON: dict[str, JSON] = {"name": "bytes", "configuration": {"endian": "little"}}
ZSTD_JSON: dict[str, JSON] = {"name": "zstd", "configuration": {"level": 0, "checksum": False}}

METADATA_DOCUMENTS = ("zarr.json", ".zarray", ".zattrs", ".zgroup")


class _MarkedPipeline(BatchedCodecPipeline):
    """A pipeline class distinguishable from the default, for context tests."""


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
    # outside the block, the choices made inside it hold
    assert isinstance(v3.to_array().async_array.codec_pipeline, FusedCodecPipeline)
    legacy = v2.metadata_legacy
    assert isinstance(legacy, ArrayV2Metadata)
    assert legacy.order == "F"


async def test_array_state_propagates_to_the_shared_binding() -> None:
    """A change through the array side (resize, update_attributes) is visible
    on the shared binding's model, and a derived binding stays independent."""
    store = MemoryStore()
    aio = AsyncArrayIO.build_v3(store, name="foo", shape=(4,), data_type="uint8", chunk_grid=(2,))
    await aio.store_metadata()
    arr = aio.to_array()

    derived = aio.with_config({"order": "F"})
    moved = aio.with_store(MemoryStore())

    await arr.resize((6,))
    assert aio.metadata.shape == (6,)

    await arr.update_attributes({"a": 1})
    assert dict(aio.metadata.attributes) == {"a": 1}

    # the derivatives were made before the changes and do not alias them
    assert derived.metadata_legacy.shape == (4,)
    assert dict(derived.metadata_legacy.attributes) == {}
    assert dict(moved.metadata_legacy.attributes) == {}

    # the same independence holds for a legacy-sourced binding, whose mutable
    # attributes dict must be copied, not shared
    legacy_arr = await zarr.api.asynchronous.create_array(
        MemoryStore(), shape=(4,), dtype="uint8", chunks=(2,)
    )
    legacy_derived = legacy_arr.io.with_config({"order": "F"})
    _ = legacy_derived.metadata  # fill the model cache before the mutation
    await legacy_arr.update_attributes({"b": 2})
    assert dict(legacy_arr.io.metadata.attributes) == {"b": 2}
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
