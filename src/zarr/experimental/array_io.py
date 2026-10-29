"""An explicit-IO layer for Zarr arrays.

`AsyncArrayIO` and `ArrayIO` separate declaring an array from storing it.
Constructing either class performs no IO: an instance binds an array metadata
document to a location in a store, and every read or write of that store is an
explicit method call. In particular, metadata is stored only when
`store_metadata` is called, so chunks can be written before metadata — a
pattern the eager creation API (`zarr.create_array`) cannot express. This
mirrors the array APIs of `zarrs` and `zarrita`, where storing metadata is a
separate step from storing chunks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from zarr.core.array import DEFAULT_FILL_VALUE, Array, AsyncArray, build_array_metadata
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


class AsyncArrayIO:
    """Explicit asynchronous IO for one Zarr array.

    An `AsyncArrayIO` binds an array metadata document to a location in a
    store. Constructing one — directly, or with `build` — performs no IO:
    nothing is read from or written to the store until an IO method is called.
    `store_metadata` stores the metadata document, and `setitem` / `getitem`
    store and read chunks, in whatever order the caller chooses.
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
        zarr_format: ZarrFormat | None = 3,
        config: ArrayConfigLike | None = None,
    ) -> AsyncArrayIO:
        """Read the metadata of an existing array and bind to it.

        This is the one classmethod constructor that performs IO: the array's
        stored metadata document is read from the store.
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
        is [`zarr.api.asynchronous.create_array`][]'s job. With
        `ensure_parents`, missing parent groups are stored first.
        """
        await self._array._save_metadata(self._array.metadata, ensure_parents=ensure_parents)

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
        zarr_format: ZarrFormat | None = 3,
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
