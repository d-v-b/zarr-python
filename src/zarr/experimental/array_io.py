"""An explicit-IO layer for Zarr arrays.

`AsyncArrayIO` and `ArrayIO` separate declaring an array from storing it.
Constructing either class performs no IO: an instance binds an array metadata
document to a location in a store, and every read or write of that store is an
explicit method call. In particular, metadata is stored only when
`store_metadata` is called, so chunks can be written before metadata — a
pattern the eager creation API (`zarr.create_array`) cannot express. This
mirrors the array API of `zarrs`, where storing, retrieving and erasing
metadata and chunks are each explicit operations.

Every write has an erasing inverse: `store_metadata` ↔ `erase_metadata`,
and `setitem` ↔ `erase_chunk` / `erase_chunks`. `store_metadata` is the
publication point: until it is called, no Zarr reader resolves the array, so
a failed chunks-first run leaves no readable array. It does leave chunk keys
in the store, which occupy space and would be misread by a different array
stored later at the same location; `erase_chunks` removes them.

One exception to "no implicit metadata IO": an array opened from a stored
document that needed a repair stores that repair before its first chunk
write, exactly as `AsyncArray` does. Call `store_metadata` first to make
that write explicit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from zarr.core.array import (
    DEFAULT_FILL_VALUE,
    Array,
    AsyncArray,
    _iter_shard_keys,
    build_array_metadata,
    get_array_metadata,
    parse_array_metadata,
)
from zarr.core.common import concurrent_foreach, product
from zarr.core.config import config as zarr_config
from zarr.core.metadata.io import ARRAY_DOCUMENTS
from zarr.core.sync import sync
from zarr.storage._common import make_store_path

if TYPE_CHECKING:
    import numpy.typing as npt

    from zarr.core.array import (
        CompressorsLike,
        FiltersLike,
        SerializerLike,
        ShardsLike,
    )
    from zarr.core.array_spec import ArrayConfig, ArrayConfigLike
    from zarr.core.buffer import BufferPrototype, NDArrayLikeOrScalar
    from zarr.core.chunk_key_encodings import ChunkKeyEncodingLike
    from zarr.core.common import (
        JSON,
        ChunksLike,
        DimensionNamesLike,
        MemoryOrder,
        ShapeLike,
        ZarrFormat,
    )
    from zarr.core.dtype import ZDTypeLike
    from zarr.core.indexing import BasicSelection
    from zarr.core.metadata import ArrayMetadata, ArrayV2Metadata, ArrayV3Metadata
    from zarr.storage import StoreLike
    from zarr.storage._common import StorePath
    from zarr.types import AnyArray, AnyAsyncArray

__all__ = ["ArrayIO", "AsyncArrayIO"]


async def _delete_key(store_path: StorePath, key: str) -> None:
    await (store_path / key).delete()


class AsyncArrayIO:
    """Explicit asynchronous IO for one Zarr array.

    An `AsyncArrayIO` binds an array metadata document to a location in a
    store. Constructing one — directly, or with `build` — performs no IO:
    nothing is read from or written to the store until an IO method is called.
    `store_metadata` stores the metadata document, and `setitem` / `getitem`
    store and read chunks, in whatever order the caller chooses.

    Warning: the bound metadata is assumed, never verified against the store.
    Reads and partial writes decode stored chunks under it, so binding
    metadata that disagrees with how existing chunks at this location were
    encoded misreads or corrupts data — the hazard TensorStore documents for
    `assume_metadata`.
    """

    _array: AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]

    def __init__(
        self,
        metadata: ArrayV2Metadata | ArrayV3Metadata,
        store_path: StorePath,
        config: ArrayConfigLike | None = None,
    ) -> None:
        self._array = AsyncArray(metadata=metadata, store_path=store_path, config=config)

    @classmethod
    def _wrap(cls, array: AnyAsyncArray) -> AsyncArrayIO:
        io = object.__new__(cls)
        io._array = array
        return io

    @classmethod
    async def build(
        cls,
        store: StoreLike,
        *,
        name: str | None = None,
        shape: ShapeLike,
        dtype: ZDTypeLike,
        chunks: ChunksLike | Literal["auto"] = "auto",
        shards: ShardsLike | None = None,
        filters: FiltersLike = "auto",
        compressors: CompressorsLike = "auto",
        serializer: SerializerLike = "auto",
        fill_value: Any | None = DEFAULT_FILL_VALUE,
        order: MemoryOrder | None = None,
        zarr_format: ZarrFormat | None = 3,
        attributes: dict[str, JSON] | None = None,
        chunk_key_encoding: ChunkKeyEncodingLike | None = None,
        dimension_names: DimensionNamesLike = None,
        config: ArrayConfigLike | None = None,
    ) -> AsyncArrayIO:
        """Declare an array bound to a location in a store, without storing anything.

        The keyword arguments have the same meaning as in
        [`zarr.api.asynchronous.create_array`][]. The returned instance holds
        the normalized metadata in memory; `store_metadata` stores it.
        """
        metadata = build_array_metadata(
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            shards=shards,
            filters=filters,
            compressors=compressors,
            serializer=serializer,
            fill_value=fill_value,
            order=order,
            zarr_format=zarr_format,
            attributes=attributes,
            chunk_key_encoding=chunk_key_encoding,
            dimension_names=dimension_names,
        )
        store_path = await make_store_path(store, path=name)
        return cls(metadata, store_path, config)

    @classmethod
    async def open(
        cls,
        store: StoreLike,
        *,
        name: str | None = None,
        zarr_format: ZarrFormat | None = None,
        config: ArrayConfigLike | None = None,
    ) -> AsyncArrayIO:
        """Read the metadata of an existing array and bind to it.

        This is the one classmethod constructor that performs IO: the array's
        stored metadata document is read from the store. `zarr_format=None`
        (the default) reads Zarr format 3 and falls back to format 2, as
        `open` does in `zarrs`; pass 2 or 3 to require one format.
        """
        store_path = await make_store_path(store, path=name)
        array = await AsyncArray.open(store_path, zarr_format, config=config)
        return cls._wrap(array)

    @property
    def metadata(self) -> ArrayMetadata:
        """The in-memory metadata this instance is bound to."""
        return self._array.metadata

    @property
    def store_path(self) -> StorePath:
        """The location in a store this instance is bound to."""
        return self._array.store_path

    @property
    def config(self) -> ArrayConfig:
        """The runtime configuration of this instance."""
        return self._array.config

    def to_array(self) -> AnyAsyncArray:
        """Return an `AsyncArray` handle over the same metadata and location."""
        return self._array

    async def store_metadata(self, *, ensure_parents: bool = False) -> None:
        """Store this array's metadata document(s).

        The documents are written unconditionally, replacing the metadata of
        any existing node at this location, as `store_metadata` does in
        `zarrs`. Guarded creation, which raises when a node already exists,
        is [`zarr.api.asynchronous.create_array`][]'s job. Chunks of any
        previous array at this location are kept and would be read under the
        new metadata; call `erase_chunks` first when repurposing a location.
        With `ensure_parents`, missing parent groups are stored first.
        """
        await self._array._save_metadata(self._array.metadata, ensure_parents=ensure_parents)

    async def verify(self) -> None:
        """Check that the stored metadata declares the array this instance is bound to.

        Reads the metadata stored at this location — favoring Zarr format 3
        when documents of both formats exist — and raises `ValueError` naming
        each differing field when it does not equal the bound metadata.
        Raises [`zarr.errors.ArrayNotFoundError`][] when nothing is stored.
        This is the explicit counterpart of the verification TensorStore
        performs against spec constraints on open: a writer that declared its
        own metadata with `build` can assert, before writing chunks, that it
        agrees with what a coordinator stored.
        """
        bound = self.metadata
        stored = parse_array_metadata(
            await get_array_metadata(self._array.store_path, zarr_format=None),
            str(self._array.store_path),
        )
        if stored == bound:
            return
        if type(stored) is not type(bound):
            raise ValueError(
                f"Array {str(self.store_path)!r}: the stored array is Zarr format "
                f"{stored.zarr_format}; the bound metadata is Zarr format "
                f"{bound.zarr_format}."
            )
        stored_dict, bound_dict = stored.to_dict(), bound.to_dict()
        differing = sorted(
            key
            for key in stored_dict.keys() | bound_dict.keys()
            if stored_dict.get(key) != bound_dict.get(key)
        )
        raise ValueError(
            f"Array {str(self.store_path)!r}: the stored metadata does not match the "
            f"bound metadata; differing fields: {', '.join(differing)}."
        )

    async def erase_metadata(self) -> None:
        """Erase this array's metadata document(s), the inverse of `store_metadata`.

        Succeeds if the documents do not exist, as `erase_metadata` does in
        `zarrs`. Stored chunks are kept; `erase_chunks` removes those.
        """
        store_path = self._array.store_path
        for document in ARRAY_DOCUMENTS[self.metadata.zarr_format]:
            await (store_path / document).delete()
        self._array._stored_document_replaced()

    async def erase_chunk(self, chunk_coords: tuple[int, ...]) -> None:
        """Erase one stored chunk by its chunk grid coordinates.

        When sharding is used, the stored chunk is the shard. Succeeds if the
        chunk does not exist, as `erase_chunk` does in `zarrs`.
        """
        grid_shape = self._array._shard_grid_shape
        if len(chunk_coords) != len(grid_shape) or any(
            coord < 0 or coord >= size for coord, size in zip(chunk_coords, grid_shape, strict=True)
        ):
            raise IndexError(
                f"Chunk coordinates {chunk_coords} are outside the chunk grid "
                f"of shape {grid_shape}."
            )
        await _delete_key(self._array.store_path, self.metadata.encode_chunk_key(chunk_coords))

    async def erase_chunks(self) -> None:
        """Erase every stored chunk of this array, the inverse of its chunk writes.

        Chunks that do not exist are skipped; the metadata documents are kept.
        """
        n_chunks = product(self._array._shard_grid_shape)
        if n_chunks == 0:
            return
        limit = zarr_config.get("async.concurrency") or 1000
        store_path = self._array.store_path
        await concurrent_foreach(
            ((store_path, key) for key in _iter_shard_keys(self._array)),
            _delete_key,
            min(limit, n_chunks),
        )

    async def getitem(
        self,
        selection: BasicSelection,
        *,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """Read the selected region, reading absent chunks as the fill value."""
        return await self._array.getitem(selection, prototype=prototype)

    async def setitem(
        self,
        selection: BasicSelection,
        value: npt.ArrayLike,
        prototype: BufferPrototype | None = None,
    ) -> None:
        """Encode `value` and store the chunks covering `selection`."""
        await self._array.setitem(selection, value, prototype=prototype)

    def __repr__(self) -> str:
        return f"<AsyncArrayIO {self.store_path} shape={self.metadata.shape}>"


class ArrayIO:
    """Explicit synchronous IO for one Zarr array. See `AsyncArrayIO`."""

    _async_io: AsyncArrayIO

    def __init__(
        self,
        metadata: ArrayV2Metadata | ArrayV3Metadata,
        store_path: StorePath,
        config: ArrayConfigLike | None = None,
    ) -> None:
        self._async_io = AsyncArrayIO(metadata, store_path, config)

    @classmethod
    def _wrap(cls, async_io: AsyncArrayIO) -> ArrayIO:
        io = object.__new__(cls)
        io._async_io = async_io
        return io

    @classmethod
    def build(
        cls,
        store: StoreLike,
        *,
        name: str | None = None,
        shape: ShapeLike,
        dtype: ZDTypeLike,
        chunks: ChunksLike | Literal["auto"] = "auto",
        shards: ShardsLike | None = None,
        filters: FiltersLike = "auto",
        compressors: CompressorsLike = "auto",
        serializer: SerializerLike = "auto",
        fill_value: Any | None = DEFAULT_FILL_VALUE,
        order: MemoryOrder | None = None,
        zarr_format: ZarrFormat | None = 3,
        attributes: dict[str, JSON] | None = None,
        chunk_key_encoding: ChunkKeyEncodingLike | None = None,
        dimension_names: DimensionNamesLike = None,
        config: ArrayConfigLike | None = None,
    ) -> ArrayIO:
        """Declare an array bound to a location in a store, without storing anything.

        See `AsyncArrayIO.build`.
        """
        return cls._wrap(
            sync(
                AsyncArrayIO.build(
                    store,
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks,
                    shards=shards,
                    filters=filters,
                    compressors=compressors,
                    serializer=serializer,
                    fill_value=fill_value,
                    order=order,
                    zarr_format=zarr_format,
                    attributes=attributes,
                    chunk_key_encoding=chunk_key_encoding,
                    dimension_names=dimension_names,
                    config=config,
                )
            )
        )

    @classmethod
    def open(
        cls,
        store: StoreLike,
        *,
        name: str | None = None,
        zarr_format: ZarrFormat | None = None,
        config: ArrayConfigLike | None = None,
    ) -> ArrayIO:
        """Read the metadata of an existing array and bind to it.

        See `AsyncArrayIO.open`.
        """
        return cls._wrap(
            sync(AsyncArrayIO.open(store, name=name, zarr_format=zarr_format, config=config))
        )

    @property
    def metadata(self) -> ArrayMetadata:
        """The in-memory metadata this instance is bound to."""
        return self._async_io.metadata

    @property
    def store_path(self) -> StorePath:
        """The location in a store this instance is bound to."""
        return self._async_io.store_path

    @property
    def config(self) -> ArrayConfig:
        """The runtime configuration of this instance."""
        return self._async_io.config

    def to_array(self) -> AnyArray:
        """Return an `Array` handle over the same metadata and location."""
        return Array(self._async_io.to_array())

    def store_metadata(self, *, ensure_parents: bool = False) -> None:
        """Store this array's metadata document(s). See `AsyncArrayIO.store_metadata`."""
        sync(self._async_io.store_metadata(ensure_parents=ensure_parents))

    def verify(self) -> None:
        """Check that the stored metadata declares the array this instance is bound to.

        See `AsyncArrayIO.verify`.
        """
        sync(self._async_io.verify())

    def erase_metadata(self) -> None:
        """Erase this array's metadata document(s). See `AsyncArrayIO.erase_metadata`."""
        sync(self._async_io.erase_metadata())

    def erase_chunk(self, chunk_coords: tuple[int, ...]) -> None:
        """Erase one stored chunk by its chunk grid coordinates. See `AsyncArrayIO.erase_chunk`."""
        sync(self._async_io.erase_chunk(chunk_coords))

    def erase_chunks(self) -> None:
        """Erase every stored chunk of this array. See `AsyncArrayIO.erase_chunks`."""
        sync(self._async_io.erase_chunks())

    def getitem(
        self,
        selection: BasicSelection,
        *,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """Read the selected region, reading absent chunks as the fill value."""
        return sync(self._async_io.getitem(selection, prototype=prototype))

    def setitem(
        self,
        selection: BasicSelection,
        value: npt.ArrayLike,
        prototype: BufferPrototype | None = None,
    ) -> None:
        """Encode `value` and store the chunks covering `selection`."""
        sync(self._async_io.setitem(selection, value, prototype=prototype))

    def __repr__(self) -> str:
        return f"<ArrayIO {self.store_path} shape={self.metadata.shape}>"
