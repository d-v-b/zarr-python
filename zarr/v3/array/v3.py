# Notes on what I've changed here:
# 1. Split Array into AsyncArray and Array
# 2. Inherit from abc (SynchronousArray, AsynchronousArray)
# 3. Added .size and .attrs methods
# 4. Temporarily disabled the creation of ArrayV2
# 5. Added from_json to AsyncArray

# Questions to consider:
# 1. Was splitting the array into two classes really necessary?
# 2. Do we really need runtime_configuration? Specifically, the asyncio_loop seems problematic

from __future__ import annotations
from enum import Enum

import json
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
from attr import asdict, evolve, frozen, field

from zarr.v3.abc.array import SynchronousArray, AsynchronousArray
from zarr.v3.metadata import (
    DefaultChunkKeyEncodingConfigurationMetadata,
    RegularChunkGridConfigurationMetadata,
    RegularChunkGridMetadata,
    DefaultChunkKeyEncodingMetadata,
    DefaultChunkKeyEncodingMetadata,
    ChunkKeyEncodingMetadata,
    V2ChunkKeyEncodingConfigurationMetadata,
)
from zarr.v3.array.chunk import (
    V2ChunkKeyEncodingMetadata,
    read_chunk,
    write_chunk,
)

from zarr.v3.codecs import CodecMetadata, CodecPipeline, bytes_codec
from zarr.v3.common import (
    ZARR_JSON,
    Attributes,
    ChunkCoords,
    Selection,
    SliceSelection,
    concurrent_map,
    make_cattr,
)
from zarr.v3.array.indexing import BasicIndexer, all_chunk_coords, is_total_slice
from zarr.v3.array.base import (
    ChunkMetadata,
    RuntimeConfiguration,
    dtype_to_data_type,
)
from zarr.v3.codecs.sharding import ShardingCodec
from zarr.v3.store import StoreLike, StorePath, make_store_path
from zarr.v3.sync import sync


@frozen
class ArrayMetadata:
    shape: ChunkCoords
    data_type: np.dtype
    chunk_grid: RegularChunkGridMetadata
    chunk_key_encoding: ChunkKeyEncodingMetadata
    fill_value: Any
    codecs: list[CodecMetadata]
    attributes: Dict[str, Any] = field(factory=dict)
    dimension_names: Optional[Tuple[str, ...]] = None
    zarr_format: Literal[3] = 3
    node_type: Literal["array"] = "array"

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def get_core_metadata(self, runtime_configuration: RuntimeConfiguration) -> ChunkMetadata:
        return ChunkMetadata(
            array_shape=self.shape,
            chunk_shape=self.chunk_grid.configuration.chunk_shape,
            dtype=self.data_type,
            fill_value=self.fill_value,
        )

    def to_bytes(self) -> bytes:
        def _json_convert(o):
            if isinstance(o, np.dtype):
                return str(o)
            if isinstance(o, Enum):
                return o.name
            raise TypeError

        return json.dumps(
            asdict(
                self,
                filter=lambda attr, value: attr.name != "dimension_names" or value is not None,
            ),
            default=_json_convert,
        ).encode()

    @classmethod
    def from_json(cls, zarr_json: Any) -> ArrayMetadata:
        return make_cattr().structure(zarr_json, cls)


@frozen
class AsyncArray(AsynchronousArray):
    metadata: ArrayMetadata
    store_path: StorePath
    runtime_configuration: RuntimeConfiguration
    codec_pipeline: CodecPipeline

    @classmethod
    async def create(
        cls,
        store: StoreLike,
        *,
        shape: ChunkCoords,
        dtype: Union[str, np.dtype],
        chunk_shape: ChunkCoords,
        fill_value: Optional[Any] = None,
        chunk_key_encoding: Union[
            Tuple[Literal["default"], Literal[".", "/"]],
            Tuple[Literal["v2"], Literal[".", "/"]],
        ] = ("default", "/"),
        codecs: Optional[Iterable[CodecMetadata]] = None,
        dimension_names: Optional[Iterable[str]] = None,
        attributes: Dict[str, Any] = None,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
        exists_ok: bool = False,
    ) -> AsyncArray:
        store_path = make_store_path(store)
        if not exists_ok:
            assert not await (store_path / ZARR_JSON).exists_async()

            """     
            data_type = (
            DataType[dtype] if isinstance(dtype, str) else DataType[dtype_to_data_type[dtype.str]]
            )
            """
        if isinstance(dtype, str):
            data_type = np.dtype(dtype)
        else:
            data_type = dtype
        codecs = list(codecs) if codecs is not None else [bytes_codec()]

        if fill_value is None:
            if data_type == "bool":
                fill_value = False
            else:
                fill_value = 0

        metadata = ArrayMetadata(
            shape=shape,
            data_type=data_type,
            chunk_grid=RegularChunkGridMetadata(
                configuration=RegularChunkGridConfigurationMetadata(chunk_shape=chunk_shape)
            ),
            chunk_key_encoding=(
                V2ChunkKeyEncodingMetadata(
                    configuration=V2ChunkKeyEncodingConfigurationMetadata(
                        separator=chunk_key_encoding[1]
                    )
                )
                if chunk_key_encoding[0] == "v2"
                else DefaultChunkKeyEncodingMetadata(
                    configuration=DefaultChunkKeyEncodingConfigurationMetadata(
                        separator=chunk_key_encoding[1]
                    )
                )
            ),
            fill_value=fill_value,
            codecs=codecs,
            dimension_names=tuple(dimension_names) if dimension_names else None,
            attributes=attributes or {},
        )
        runtime_configuration = runtime_configuration or RuntimeConfiguration()

        array = cls(
            metadata=metadata,
            store_path=store_path,
            runtime_configuration=runtime_configuration,
            codec_pipeline=CodecPipeline.from_metadata(
                metadata.codecs, metadata.get_core_metadata(runtime_configuration)
            ),
        )

        await array._save_metadata()
        return array

    @classmethod
    def from_json(
        cls,
        store_path: StorePath,
        zarr_json: Any,
        runtime_configuration: RuntimeConfiguration,
    ) -> AsyncArray:
        metadata = ArrayMetadata.from_json(zarr_json)
        async_array = cls(
            metadata=metadata,
            store_path=store_path,
            runtime_configuration=runtime_configuration,
            codec_pipeline=CodecPipeline.from_metadata(
                metadata.codecs, metadata.get_core_metadata(runtime_configuration)
            ),
        )
        async_array._validate_metadata()
        return async_array

    @classmethod
    async def open(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> AsyncArray:
        store_path = make_store_path(store)
        zarr_json_bytes = await (store_path / ZARR_JSON).get_async()
        assert zarr_json_bytes is not None
        return cls.from_json(
            store_path,
            json.loads(zarr_json_bytes),
            runtime_configuration=runtime_configuration,
        )

    @classmethod
    async def open_auto(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> AsyncArray:  # TODO: Union[AsyncArray, ArrayV2]
        store_path = make_store_path(store)
        v3_metadata_bytes = await (store_path / ZARR_JSON).get_async()
        if v3_metadata_bytes is not None:
            return cls.from_json(
                store_path,
                json.loads(v3_metadata_bytes),
                runtime_configuration=runtime_configuration or RuntimeConfiguration(),
            )
        else:
            raise ValueError("no v2 support yet")
            # return await ArrayV2.open_async(store_path)

    @property
    def ndim(self) -> int:
        return len(self.metadata.shape)

    @property
    def shape(self) -> ChunkCoords:
        return self.metadata.shape

    @property
    def size(self) -> int:
        return np.prod(self.metadata.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.metadata.data_type

    @property
    def attrs(self) -> Attributes:
        return self.metadata.attributes

    async def getitem(self, selection: Selection):
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=self.metadata.chunk_grid.configuration.chunk_shape,
        )

        # setup output array
        out = np.zeros(
            indexer.shape,
            dtype=self.metadata.data_type,
            order=self.runtime_configuration.order,
        )

        # reading chunks and decoding them
        await concurrent_map(
            [
                (
                    self.metadata.chunk_key_encoding,
                    self.metadata.fill_value,
                    self.store_path,
                    self.codec_pipeline,
                    chunk_coords,
                    chunk_selection,
                    out_selection,
                    out,
                    self.runtime_configuration,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            read_chunk,
            self.runtime_configuration.concurrency,
        )

        if out.shape:
            return out
        else:
            return out[()]

    async def _save_metadata(self) -> None:
        self._validate_metadata()

        await (self.store_path / ZARR_JSON).set_async(self.metadata.to_bytes())

    def _validate_metadata(self) -> None:
        assert len(self.metadata.shape) == len(
            self.metadata.chunk_grid.configuration.chunk_shape
        ), "`chunk_shape` and `shape` need to have the same number of dimensions."
        assert self.metadata.dimension_names is None or len(self.metadata.shape) == len(
            self.metadata.dimension_names
        ), "`dimension_names` and `shape` need to have the same number of dimensions."
        assert self.metadata.fill_value is not None, "`fill_value` is required."

    async def setitem(self, selection: Selection, value: np.ndarray) -> None:
        chunk_shape = self.metadata.chunk_grid.configuration.chunk_shape
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=chunk_shape,
        )

        sel_shape = indexer.shape

        # check value shape
        if np.isscalar(value):
            # setting a scalar value
            pass
        else:
            if not hasattr(value, "shape"):
                value = np.asarray(value, self.metadata.dtype)
            assert value.shape == sel_shape
            if value.dtype.name != self.dtype.name:
                value = value.astype(self.dtype, order="A")

        # merging with existing data and encoding chunks
        await concurrent_map(
            [
                (
                    self.metadata.chunk_key_encoding,
                    self.store_path,
                    self.codec_pipeline,
                    value,
                    chunk_shape,
                    chunk_coords,
                    chunk_selection,
                    out_selection,
                    self.metadata.fill_value,
                    self.runtime_configuration,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            write_chunk,
            self.runtime_configuration.concurrency,
        )

    async def resize(self, new_shape: ChunkCoords) -> Array:
        assert len(new_shape) == len(self.metadata.shape)
        new_metadata = evolve(self.metadata, shape=new_shape)

        # Remove all chunks outside of the new shape
        chunk_shape = self.metadata.chunk_grid.configuration.chunk_shape
        chunk_key_encoding = self.metadata.chunk_key_encoding
        old_chunk_coords = set(all_chunk_coords(self.metadata.shape, chunk_shape))
        new_chunk_coords = set(all_chunk_coords(new_shape, chunk_shape))

        async def _delete_key(key: str) -> None:
            await (self.store_path / key).delete_async()

        await concurrent_map(
            [
                (chunk_key_encoding.encode_chunk_key(chunk_coords),)
                for chunk_coords in old_chunk_coords.difference(new_chunk_coords)
            ],
            _delete_key,
            self.runtime_configuration.concurrency,
        )

        # Write new metadata
        await (self.store_path / ZARR_JSON).set_async(new_metadata.to_bytes())
        return evolve(self, metadata=new_metadata)

    async def update_attributes(self, new_attributes: Dict[str, Any]) -> Array:
        new_metadata = evolve(self.metadata, attributes=new_attributes)

        # Write new metadata
        await (self.store_path / ZARR_JSON).set_async(new_metadata.to_bytes())
        return evolve(self, metadata=new_metadata)

    def __repr__(self):
        return f"<AsyncArray {self.store_path} shape={self.shape} dtype={self.dtype}>"

    async def info(self):
        return NotImplemented


@frozen
class Array(SynchronousArray):
    _async_array: AsyncArray

    @classmethod
    def create(
        cls,
        store: StoreLike,
        *,
        shape: ChunkCoords,
        dtype: Union[str, np.dtype],
        chunk_shape: ChunkCoords,
        fill_value: Optional[Any] = None,
        chunk_key_encoding: Union[
            Tuple[Literal["default"], Literal[".", "/"]],
            Tuple[Literal["v2"], Literal[".", "/"]],
        ] = ("default", "/"),
        codecs: Optional[Iterable[CodecMetadata]] = None,
        dimension_names: Optional[Iterable[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
        exists_ok: bool = False,
    ) -> Array:
        async_array = sync(
            AsyncArray.create(
                store=store,
                shape=shape,
                dtype=dtype,
                chunk_shape=chunk_shape,
                fill_value=fill_value,
                chunk_key_encoding=chunk_key_encoding,
                codecs=codecs,
                dimension_names=dimension_names,
                attributes=attributes,
                runtime_configuration=runtime_configuration,
                exists_ok=exists_ok,
            ),
            runtime_configuration.asyncio_loop,
        )
        return cls(async_array)

    @classmethod
    def from_json(
        cls,
        store_path: StorePath,
        zarr_json: Any,
        runtime_configuration: RuntimeConfiguration,
    ) -> Array:
        async_array = AsyncArray.from_json(
            store_path=store_path, zarr_json=zarr_json, runtime_configuration=runtime_configuration
        )
        return cls(async_array)

    @classmethod
    def open(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Array:

        async_array = sync(
            AsyncArray.open(store, runtime_configuration=runtime_configuration),
            runtime_configuration.asyncio_loop,
        )
        async_array._validate_metadata()
        return cls(async_array)

    @classmethod
    def open_auto(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Array:  # TODO: Union[Array, ArrayV2]:
        async_array = sync(
            AsyncArray.open_auto(store, runtime_configuration),
            runtime_configuration.asyncio_loop,
        )
        return cls(async_array)

    @property
    def ndim(self) -> int:
        return self._async_array.ndim

    @property
    def shape(self) -> ChunkCoords:
        return self._async_array.shape

    @property
    def size(self) -> int:
        return self._async_array.size

    @property
    def dtype(self) -> np.dtype:
        return self._async_array.dtype

    @property
    def attrs(self) -> dict:
        return self._async_array.attrs

    @property
    def metadata(self) -> ArrayMetadata:
        return self._async_array.metadata

    @property
    def store_path(self) -> str:
        return self._async_array.store_path

    def __getitem__(self, selection: Selection):
        return sync(
            self._async_array.getitem(selection),
            self._async_array.runtime_configuration.asyncio_loop,
        )

    def __setitem__(self, selection: Selection, value: np.ndarray) -> None:
        sync(
            self._async_array.setitem(selection, value),
            self._async_array.runtime_configuration.asyncio_loop,
        )

    def resize(self, new_shape: ChunkCoords) -> Array:
        return sync(
            self._async_array.resize(new_shape),
            self._async_array.runtime_configuration.asyncio_loop,
        )

    def update_attributes(self, new_attributes: Dict[str, Any]) -> Array:
        return sync(
            self._async_array.update_attributes(new_attributes),
            self._async_array.runtime_configuration.asyncio_loop,
        )

    def __repr__(self):
        return f"<Array {self.store_path} shape={self.shape} dtype={self.dtype}>"

    def info(self):
        return sync(
            self._async_array.info(),
            self._async_array.runtime_configuration.asyncio_loop,
        )
