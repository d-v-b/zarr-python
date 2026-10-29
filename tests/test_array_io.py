"""Tests for the explicit-IO array layer, `zarr.ArrayIO` and `zarr.AsyncArrayIO`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from zarr_metadata import ZarrV3ArrayMetadata

from zarr import ArrayIO, AsyncArrayIO
from zarr.api.asynchronous import open_array
from zarr.core.buffer import default_buffer_prototype
from zarr.errors import ArrayNotFoundError, UnknownCodecError
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat

METADATA_DOCUMENTS = ("zarr.json", ".zarray", ".zattrs", ".zgroup")


async def _store_keys(store: MemoryStore) -> set[str]:
    return {key async for key in store.list()}


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
    aio = await AsyncArrayIO.build(
        store,
        name="foo",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(2, 2),
        zarr_format=zarr_format,
        attributes={"answer": 42},
    )
    assert await _store_keys(store) == set()

    await aio.setitem(Ellipsis, data)
    keys = await _store_keys(store)
    assert len(keys) == 4
    assert not any(key.split("/")[-1] in METADATA_DOCUMENTS for key in keys)
    np.testing.assert_array_equal(await aio.getitem(Ellipsis), data)

    await aio.store_metadata()
    # the default zarr_format=None auto-detects the format, like zarrs
    opened = await AsyncArrayIO.open(store, name="foo")
    # the binding is a document; the stored and declared documents agree
    assert opened.document == aio.document
    assert opened.document.shape == data.shape
    assert opened.metadata.zarr_format == zarr_format
    assert opened.metadata.attributes == {"answer": 42}
    np.testing.assert_array_equal(await opened.getitem(Ellipsis), data)

    arr = await open_array(store=store, path="foo", mode="r", zarr_format=zarr_format)
    np.testing.assert_array_equal(await arr.getitem(Ellipsis), data)
    # every AsyncArray is built on an AsyncArrayIO, reachable as .io
    assert arr.io.metadata is arr.metadata
    np.testing.assert_array_equal(await arr.io.getitem(Ellipsis), data)

    # a writer that declared the same array independently can verify its assumption
    worker = await AsyncArrayIO.build(
        store,
        name="foo",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(2, 2),
        zarr_format=zarr_format,
        attributes={"answer": 42},
    )
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
    aio = ArrayIO.build(store, name="foo", shape=data.shape, dtype=data.dtype, chunks=(2, 2))
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


def test_build_v2_shards_raises() -> None:
    with pytest.raises(ValueError, match="shard_shape"):
        ArrayIO.build(
            MemoryStore(),
            name="foo",
            shape=(4,),
            dtype="uint8",
            chunks=(2,),
            shards=(4,),
            zarr_format=2,
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
    document = aio.document
    assert isinstance(document, ZarrV3ArrayMetadata)
    assert document.shape == (4,)
    assert document.must_understand_fields == {}
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
    aio = await AsyncArrayIO.build(store, shape=(4, 4), dtype="uint8", chunks=(2, 2))
    await aio.store_metadata()
    other = await AsyncArrayIO.build(
        store, shape=(4, 4), dtype="uint8", chunks=(4, 4), attributes={"a": 1}
    )
    with pytest.raises(ValueError, match="differing fields: attributes, chunk_grid"):
        await other.verify()


async def test_verify_format_mismatch_raises() -> None:
    store = MemoryStore()
    await (
        await AsyncArrayIO.build(store, shape=(4,), dtype="uint8", zarr_format=2)
    ).store_metadata()
    bound_v3 = await AsyncArrayIO.build(store, shape=(4,), dtype="uint8", zarr_format=3)
    with pytest.raises(ValueError, match="stored array is Zarr format 2"):
        await bound_v3.verify()


async def test_verify_missing_raises() -> None:
    aio = await AsyncArrayIO.build(MemoryStore(), shape=(4,), dtype="uint8")
    with pytest.raises(ArrayNotFoundError):
        await aio.verify()


async def test_erase_chunk_out_of_grid_raises() -> None:
    aio = await AsyncArrayIO.build(MemoryStore(), shape=(4, 4), dtype="uint8", chunks=(2, 2))
    with pytest.raises(IndexError, match="outside the chunk grid"):
        await aio.erase_chunk((2, 0))
    with pytest.raises(IndexError, match="outside the chunk grid"):
        await aio.erase_chunk((0,))
