from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from operator import itemgetter
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt

from zarr.abc.codec import (
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    Codec,
    CodecPipeline,
)
from zarr.abc.store import (
    ByteGetter,
    ByteRequest,
    ByteSetter,
    RangeByteRequest,
    SuffixByteRequest,
)
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import (
    Buffer,
    BufferPrototype,
    NDBuffer,
    default_buffer_prototype,
    numpy_buffer_prototype,
)
from zarr.core.chunk_grids import ChunkGrid, RegularChunkGrid
from zarr.core.common import (
    ShapeLike,
    parse_enum,
    parse_named_configuration,
    parse_shapelike,
    product,
)
from zarr.core.dtype.npy.int import UInt64
from zarr.core.indexing import (
    BasicIndexer,
    SelectorTuple,
    c_order_iter,
    get_indexer,
    morton_order_iter,
)
from zarr.core.metadata.v3 import parse_codecs
from zarr.registry import get_ndbuffer_class, get_pipeline_class

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

    from zarr.core.common import JSON
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

MAX_UINT_64 = 2**64 - 1


class ShardingCodecIndexLocation(Enum):
    """
    Enum for index location used by the sharding codec.
    """

    start = "start"
    end = "end"


def parse_index_location(data: object) -> ShardingCodecIndexLocation:
    return parse_enum(data, ShardingCodecIndexLocation)


@dataclass(frozen=True)
class _ShardingByteGetter(ByteGetter):
    """Adapts a shard to the ByteGetter protocol for a specific chunk."""

    shard: Shard | ShardBuilder
    chunk_coords: tuple[int, ...]

    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        assert byte_range is None, "byte_range is not supported within shards"
        assert prototype == default_buffer_prototype(), (
            f"prototype is not supported within shards currently. diff: {prototype} != {default_buffer_prototype()}"
        )
        if isinstance(self.shard, Shard):
            return self.shard.get_chunk_bytes(self.chunk_coords)
        else:
            # ShardBuilder - used for nested sharding during construction
            return self.shard.get(self.chunk_coords)


@dataclass(frozen=True)
class _ShardingByteSetter(_ShardingByteGetter, ByteSetter):
    """Adapts a shard builder to the ByteSetter protocol for a specific chunk."""

    shard: ShardBuilder

    async def set(self, value: Buffer, byte_range: ByteRequest | None = None) -> None:
        assert byte_range is None, "byte_range is not supported within shards"
        self.shard.append_chunk(self.chunk_coords, value)

    async def delete(self) -> None:
        # Delete means "don't include this chunk in the final shard"
        # Mark as tombstone so it won't be copied from old shard during merge
        self.shard._tombstones.add(self.chunk_coords)
        # Also remove from new chunks if it was previously added
        if self.chunk_coords in self.shard._chunks:
            del self.shard._chunks[self.chunk_coords]

    async def set_if_not_exists(self, default: Buffer) -> None:
        # Only set if chunk doesn't exist
        if self.shard.get_chunk(self.chunk_coords) is None:
            self.shard.append_chunk(self.chunk_coords, default)


class _ShardIndex(NamedTuple):
    # dtype uint64, shape (chunks_per_shard_0, chunks_per_shard_1, ..., 2)
    offsets_and_lengths: npt.NDArray[np.uint64]

    @property
    def chunks_per_shard(self) -> tuple[int, ...]:
        result = tuple(self.offsets_and_lengths.shape[0:-1])
        # The cast is required until https://github.com/numpy/numpy/pull/27211 is merged
        return cast("tuple[int, ...]", result)

    def _localize_chunk(self, chunk_coords: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            chunk_i % shard_i
            for chunk_i, shard_i in zip(chunk_coords, self.offsets_and_lengths.shape, strict=False)
        )

    def is_all_empty(self) -> bool:
        return bool(np.array_equiv(self.offsets_and_lengths, MAX_UINT_64))

    def get_full_chunk_map(self) -> npt.NDArray[np.bool_]:
        return np.not_equal(self.offsets_and_lengths[..., 0], MAX_UINT_64)

    def get_chunk_slice(self, chunk_coords: tuple[int, ...]) -> tuple[int, int] | None:
        localized_chunk = self._localize_chunk(chunk_coords)
        chunk_start, chunk_len = self.offsets_and_lengths[localized_chunk]
        if (chunk_start, chunk_len) == (MAX_UINT_64, MAX_UINT_64):
            return None
        else:
            return (int(chunk_start), int(chunk_start + chunk_len))

    def set_chunk_slice(self, chunk_coords: tuple[int, ...], chunk_slice: slice | None) -> None:
        localized_chunk = self._localize_chunk(chunk_coords)
        if chunk_slice is None:
            self.offsets_and_lengths[localized_chunk] = (MAX_UINT_64, MAX_UINT_64)
        else:
            self.offsets_and_lengths[localized_chunk] = (
                chunk_slice.start,
                chunk_slice.stop - chunk_slice.start,
            )

    def is_dense(self, chunk_byte_length: int) -> bool:
        sorted_offsets_and_lengths = sorted(
            [
                (offset, length)
                for offset, length in self.offsets_and_lengths
                if offset != MAX_UINT_64
            ],
            key=itemgetter(0),
        )

        # Are all non-empty offsets unique?
        if len(
            {offset for offset, _ in sorted_offsets_and_lengths if offset != MAX_UINT_64}
        ) != len(sorted_offsets_and_lengths):
            return False

        return all(
            offset % chunk_byte_length == 0 and length == chunk_byte_length
            for offset, length in sorted_offsets_and_lengths
        )

    @classmethod
    def create_empty(cls, chunks_per_shard: tuple[int, ...]) -> _ShardIndex:
        offsets_and_lengths = np.zeros(chunks_per_shard + (2,), dtype="<u8", order="C")
        offsets_and_lengths.fill(MAX_UINT_64)
        return cls(offsets_and_lengths)


# ============================================================================
# New Core Abstractions: Shard as Immutable Value Object
# ============================================================================


@dataclass(frozen=True)
class ChunkLocation:
    """Location of a chunk in a linear buffer."""

    offset: int
    length: int


@dataclass(frozen=True)
class Shard:
    """
    Immutable shard: represents a sparse N-D chunk grid backed by a linear buffer.

    A shard is fundamentally:
    - An index mapping chunk coordinates to byte ranges
    - A buffer containing the actual chunk data
    - Metadata describing the logical structure

    This is a value object - all operations are reads or transformations that
    return new Shards.
    """

    # Index: chunk coordinates → byte range
    index: _ShardIndex

    # Physical storage: linear byte buffer (already includes index if serialized)
    buffer: Buffer

    # Logical structure metadata
    chunks_per_shard: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    dtype: Any  # np.dtype
    fill_value: Any

    @classmethod
    async def from_buffer(
        cls,
        buffer: Buffer,
        chunks_per_shard: tuple[int, ...],
        chunk_shape: tuple[int, ...],
        dtype: Any,
        fill_value: Any,
        index_location: ShardingCodecIndexLocation,
        index_decoder: Callable[[Buffer], Awaitable[_ShardIndex]],
    ) -> Shard:
        """
        Parse a shard from a serialized buffer.

        The buffer contains both the index and chunk data, in the order
        specified by index_location.
        """
        # Decode index from buffer
        index = await index_decoder(buffer)

        return cls(
            index=index,
            buffer=buffer,
            chunks_per_shard=chunks_per_shard,
            chunk_shape=chunk_shape,
            dtype=dtype,
            fill_value=fill_value,
        )

    @classmethod
    def empty(
        cls,
        chunks_per_shard: tuple[int, ...],
        chunk_shape: tuple[int, ...],
        dtype: Any,
        fill_value: Any,
        buffer_prototype: BufferPrototype | None = None,
    ) -> Shard:
        """Create an empty shard with no chunks."""
        if buffer_prototype is None:
            buffer_prototype = default_buffer_prototype()

        return cls(
            index=_ShardIndex.create_empty(chunks_per_shard),
            buffer=buffer_prototype.buffer.create_zero_length(),
            chunks_per_shard=chunks_per_shard,
            chunk_shape=chunk_shape,
            dtype=dtype,
            fill_value=fill_value,
        )

    def get_chunk_location(self, coords: tuple[int, ...]) -> ChunkLocation | None:
        """Get the byte range for a chunk, or None if not present."""
        byte_range = self.index.get_chunk_slice(coords)
        if byte_range is None:
            return None
        return ChunkLocation(offset=byte_range[0], length=byte_range[1] - byte_range[0])

    def get_chunk_bytes(self, coords: tuple[int, ...]) -> Buffer | None:
        """Get raw encoded bytes for a chunk, or None if not present."""
        location = self.get_chunk_location(coords)
        if location is None:
            return None
        return self.buffer[location.offset : location.offset + location.length]

    def is_empty(self) -> bool:
        """Check if shard has no chunks."""
        return self.index.is_all_empty()

    def present_chunks(self) -> Iterator[tuple[int, ...]]:
        """Iterate over coordinates of all present chunks."""
        full_map = self.index.get_full_chunk_map()
        for coords in c_order_iter(self.chunks_per_shard):
            if full_map[coords]:
                yield coords


class ShardBuilder:
    """
    Mutable builder for constructing immutable Shards.

    Accumulates encoded chunks and metadata, then serializes them into
    a single buffer with an index.

    Usage:
        builder = ShardBuilder(...)
        builder.append_chunk((0, 0), encoded_bytes)
        builder.append_chunk((0, 1), encoded_bytes)
        shard = await builder.finalize(index_location, index_encoder)
    """

    def __init__(
        self,
        chunks_per_shard: tuple[int, ...],
        chunk_shape: tuple[int, ...],
        dtype: Any,
        fill_value: Any,
    ):
        self.chunks_per_shard = chunks_per_shard
        self.chunk_shape = chunk_shape
        self.dtype = dtype
        self.fill_value = fill_value

        # Accumulate chunks: coords → encoded bytes
        self._chunks: dict[tuple[int, ...], Buffer] = {}

        # Track explicitly deleted chunks (for merging with old shards)
        self._tombstones: set[tuple[int, ...]] = set()

    def append_chunk(self, coords: tuple[int, ...], encoded: Buffer) -> None:
        """Add an encoded chunk. Overwrites if coords already exist."""
        self._chunks[coords] = encoded

    def get_chunk(self, coords: tuple[int, ...]) -> Buffer | None:
        """Get a chunk that was added (for builder queries)."""
        return self._chunks.get(coords)

    def get(self, coords: tuple[int, ...], default: Buffer | None = None) -> Buffer | None:
        """Get chunk with optional default (for compatibility)."""
        return self._chunks.get(coords, default)

    async def finalize(
        self,
        index_location: ShardingCodecIndexLocation,
        index_encoder: Callable[[_ShardIndex], Awaitable[Buffer]],
    ) -> Shard:
        """
        Build the immutable Shard.

        This performs serialization:
        1. Orders chunks (Morton order for spatial locality)
        2. Builds index mapping coords → byte ranges
        3. Concatenates all chunks into single buffer
        4. Encodes and prepends/appends index
        5. Returns immutable Shard
        """
        if not self._chunks:
            # Empty shard
            return Shard.empty(
                self.chunks_per_shard,
                self.chunk_shape,
                self.dtype,
                self.fill_value,
            )

        # Order chunks in Morton order for better spatial locality
        ordered_coords = sorted(self._chunks.keys(), key=lambda c: morton_encode(c))

        # Build index and accumulate buffers
        index = _ShardIndex.create_empty(self.chunks_per_shard)
        chunk_buffers = []
        current_offset = 0

        for coords in ordered_coords:
            chunk_buffer = self._chunks[coords]
            length = len(chunk_buffer)

            # Record in index
            index.set_chunk_slice(coords, slice(current_offset, current_offset + length))

            chunk_buffers.append(chunk_buffer)
            current_offset += length

        # Encode index
        index_bytes = await index_encoder(index)

        # Combine based on index location
        if index_location == ShardingCodecIndexLocation.start:
            # Adjust all offsets to account for index at start
            empty_chunks_mask = index.offsets_and_lengths[..., 0] == MAX_UINT_64
            index.offsets_and_lengths[~empty_chunks_mask, 0] += len(index_bytes)
            # Re-encode with adjusted offsets
            index_bytes = await index_encoder(index)
            all_buffers = [index_bytes] + chunk_buffers
        else:
            all_buffers = chunk_buffers + [index_bytes]

        # Concatenate into single buffer
        if len(all_buffers) == 1:
            combined_buffer = all_buffers[0]
        else:
            combined_buffer = all_buffers[0].combine(all_buffers[1:])

        return Shard(
            index=index,
            buffer=combined_buffer,
            chunks_per_shard=self.chunks_per_shard,
            chunk_shape=self.chunk_shape,
            dtype=self.dtype,
            fill_value=self.fill_value,
        )


async def merge_shards(
    old_shard: Shard | None,
    new_chunks: dict[tuple[int, ...], Buffer],
    tombstones: set[tuple[int, ...]],
    index_location: ShardingCodecIndexLocation,
    index_encoder: Callable[[_ShardIndex], Awaitable[Buffer]],
) -> Shard:
    """
    Create a new shard by merging an old shard with updates.

    This is a pure function implementing the reconciliation operation:
        new_shard = old_shard - tombstones + new_chunks

    Args:
        old_shard: Existing shard to merge from (None = empty)
        new_chunks: New/updated chunks to add (coords → encoded bytes)
        tombstones: Chunk coordinates to delete
        index_location: Where to place index in serialized buffer
        index_encoder: Function to encode the index

    Returns:
        New immutable Shard with merged data
    """
    # Use old shard's metadata, or infer from new chunks if no old shard
    if old_shard is not None:
        chunks_per_shard = old_shard.chunks_per_shard
        chunk_shape = old_shard.chunk_shape
        dtype = old_shard.dtype
        fill_value = old_shard.fill_value
    else:
        # No old shard - must have new chunks to infer metadata
        # (In practice, caller should provide metadata)
        raise ValueError("Cannot merge with no old shard and no metadata")

    # Create builder
    builder = ShardBuilder(chunks_per_shard, chunk_shape, dtype, fill_value)

    # Iterate in Morton order for spatial locality
    for coords in morton_order_iter(chunks_per_shard):
        # Skip tombstones
        if coords in tombstones:
            continue

        # Prefer new chunks over old
        if coords in new_chunks:
            builder.append_chunk(coords, new_chunks[coords])
        elif old_shard is not None:
            # Copy from old shard
            chunk_bytes = old_shard.get_chunk_bytes(coords)
            if chunk_bytes is not None:
                builder.append_chunk(coords, chunk_bytes)

    return await builder.finalize(index_location, index_encoder)


def morton_encode(coords: tuple[int, ...]) -> int:
    """
    Encode multi-dimensional coordinates as a Morton (Z-order) number.

    This interleaves the bits of the coordinates to produce a single integer
    that preserves spatial locality.
    """
    # Simple implementation - could be optimized
    result = 0
    max_bits = max((c.bit_length() for c in coords), default=0)

    for bit in range(max_bits):
        for dim, coord in enumerate(coords):
            if coord & (1 << bit):
                result |= 1 << (bit * len(coords) + dim)

    return result


@dataclass(frozen=True)
class ShardingCodec(
    ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin, ArrayBytesCodecPartialEncodeMixin
):
    """Sharding codec"""

    chunk_shape: tuple[int, ...]
    codecs: tuple[Codec, ...]
    index_codecs: tuple[Codec, ...]
    index_location: ShardingCodecIndexLocation = ShardingCodecIndexLocation.end

    def __init__(
        self,
        *,
        chunk_shape: ShapeLike,
        codecs: Iterable[Codec | dict[str, JSON]] = (BytesCodec(),),
        index_codecs: Iterable[Codec | dict[str, JSON]] = (BytesCodec(), Crc32cCodec()),
        index_location: ShardingCodecIndexLocation | str = ShardingCodecIndexLocation.end,
    ) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)
        codecs_parsed = parse_codecs(codecs)
        index_codecs_parsed = parse_codecs(index_codecs)
        index_location_parsed = parse_index_location(index_location)

        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)
        object.__setattr__(self, "codecs", codecs_parsed)
        object.__setattr__(self, "index_codecs", index_codecs_parsed)
        object.__setattr__(self, "index_location", index_location_parsed)

        # Use instance-local lru_cache to avoid memory leaks

        # numpy void scalars are not hashable, which means an array spec with a fill value that is
        # a numpy void scalar will break the lru_cache. This is commented for now but should be
        # fixed. See https://github.com/zarr-developers/zarr-python/issues/3054
        # object.__setattr__(self, "_get_chunk_spec", lru_cache()(self._get_chunk_spec))
        object.__setattr__(self, "_get_index_chunk_spec", lru_cache()(self._get_index_chunk_spec))
        object.__setattr__(self, "_get_chunks_per_shard", lru_cache()(self._get_chunks_per_shard))

    # todo: typedict return type
    def __getstate__(self) -> dict[str, Any]:
        return self.to_dict()

    def __setstate__(self, state: dict[str, Any]) -> None:
        config = state["configuration"]
        object.__setattr__(self, "chunk_shape", parse_shapelike(config["chunk_shape"]))
        object.__setattr__(self, "codecs", parse_codecs(config["codecs"]))
        object.__setattr__(self, "index_codecs", parse_codecs(config["index_codecs"]))
        object.__setattr__(self, "index_location", parse_index_location(config["index_location"]))

        # Use instance-local lru_cache to avoid memory leaks
        # object.__setattr__(self, "_get_chunk_spec", lru_cache()(self._get_chunk_spec))
        object.__setattr__(self, "_get_index_chunk_spec", lru_cache()(self._get_index_chunk_spec))
        object.__setattr__(self, "_get_chunks_per_shard", lru_cache()(self._get_chunks_per_shard))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "sharding_indexed")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    @property
    def codec_pipeline(self) -> CodecPipeline:
        return get_pipeline_class().from_codecs(self.codecs)

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": self.chunk_shape,
                "codecs": tuple(s.to_dict() for s in self.codecs),
                "index_codecs": tuple(s.to_dict() for s in self.index_codecs),
                "index_location": self.index_location.value,
            },
        }

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        shard_spec = self._get_chunk_spec(array_spec)
        evolved_codecs = tuple(c.evolve_from_array_spec(array_spec=shard_spec) for c in self.codecs)
        if evolved_codecs != self.codecs:
            return replace(self, codecs=evolved_codecs)
        return self

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        if len(self.chunk_shape) != len(shape):
            raise ValueError(
                "The shard's `chunk_shape` and array's `shape` need to have the same number of dimensions."
            )
        if not isinstance(chunk_grid, RegularChunkGrid):
            raise TypeError("Sharding is only compatible with regular chunk grids.")
        if not all(
            s % c == 0
            for s, c in zip(
                chunk_grid.chunk_shape,
                self.chunk_shape,
                strict=False,
            )
        ):
            raise ValueError(
                f"The array's `chunk_shape` (got {chunk_grid.chunk_shape}) "
                f"needs to be divisible by the shard's inner `chunk_shape` (got {self.chunk_shape})."
            )

    async def _decode_single(
        self,
        shard_bytes: Buffer,
        shard_spec: ArraySpec,
    ) -> NDBuffer:
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)

        indexer = BasicIndexer(
            tuple(slice(0, s) for s in shard_shape),
            shape=shard_shape,
            chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
        )

        # setup output array
        out = chunk_spec.prototype.nd_buffer.empty(
            shape=shard_shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            order=shard_spec.order,
        )

        # Parse shard from buffer
        async def decode_index(buffer: Buffer) -> _ShardIndex:
            shard_index_size = self._shard_index_size(chunks_per_shard)
            if self.index_location == ShardingCodecIndexLocation.start:
                index_bytes = buffer[:shard_index_size]
            else:
                index_bytes = buffer[-shard_index_size:]
            return await self._decode_shard_index(index_bytes, chunks_per_shard)

        shard = await Shard.from_buffer(
            buffer=shard_bytes,
            chunks_per_shard=chunks_per_shard,
            chunk_shape=chunk_shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            fill_value=shard_spec.fill_value,
            index_location=self.index_location,
            index_decoder=decode_index,
        )

        if shard.is_empty():
            out.fill(shard_spec.fill_value)
            return out

        # decoding chunks and writing them into the output buffer
        await self.codec_pipeline.read(
            [
                (
                    _ShardingByteGetter(shard, chunk_coords),
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_shard,
                )
                for chunk_coords, chunk_selection, out_selection, is_complete_shard in indexer
            ],
            out,
        )

        return out

    async def _decode_partial_single(
        self,
        byte_getter: ByteGetter,
        selection: SelectorTuple,
        shard_spec: ArraySpec,
    ) -> NDBuffer | None:
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)

        indexer = get_indexer(
            selection,
            shape=shard_shape,
            chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
        )

        # setup output array
        out = shard_spec.prototype.nd_buffer.empty(
            shape=indexer.shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            order=shard_spec.order,
        )

        indexed_chunks = list(indexer)
        all_chunk_coords = {chunk_coords for chunk_coords, *_ in indexed_chunks}

        # reading bytes of all requested chunks
        shard_dict: Shard | dict[tuple[int, ...], Buffer]
        if self._is_total_shard(all_chunk_coords, chunks_per_shard):
            # read entire shard
            shard_dict_maybe = await self._load_full_shard_maybe(
                byte_getter=byte_getter,
                prototype=chunk_spec.prototype,
                chunks_per_shard=chunks_per_shard,
                dtype=shard_spec.dtype.to_native_dtype(),
                fill_value=shard_spec.fill_value,
            )
            if shard_dict_maybe is None:
                return None
            shard_dict = shard_dict_maybe
        else:
            # read some chunks within the shard
            shard_index = await self._load_shard_index_maybe(byte_getter, chunks_per_shard)
            if shard_index is None:
                return None
            shard_dict = {}
            for chunk_coords in all_chunk_coords:
                chunk_byte_slice = shard_index.get_chunk_slice(chunk_coords)
                if chunk_byte_slice:
                    chunk_bytes = await byte_getter.get(
                        prototype=chunk_spec.prototype,
                        byte_range=RangeByteRequest(chunk_byte_slice[0], chunk_byte_slice[1]),
                    )
                    if chunk_bytes:
                        shard_dict[chunk_coords] = chunk_bytes

        # decoding chunks and writing them into the output buffer
        await self.codec_pipeline.read(
            [
                (
                    _ShardingByteGetter(shard_dict, chunk_coords),
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_shard,
                )
                for chunk_coords, chunk_selection, out_selection, is_complete_shard in indexer
            ],
            out,
        )

        if hasattr(indexer, "sel_shape"):
            return out.reshape(indexer.sel_shape)
        else:
            return out

    async def _encode_single(
        self,
        shard_array: NDBuffer,
        shard_spec: ArraySpec,
    ) -> Buffer | None:
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)

        indexer = list(
            BasicIndexer(
                tuple(slice(0, s) for s in shard_shape),
                shape=shard_shape,
                chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
            )
        )

        shard_builder = ShardBuilder(
            chunks_per_shard=chunks_per_shard,
            chunk_shape=chunk_shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            fill_value=shard_spec.fill_value,
        )

        await self.codec_pipeline.write(
            [
                (
                    _ShardingByteSetter(shard_builder, chunk_coords),
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_shard,
                )
                for chunk_coords, chunk_selection, out_selection, is_complete_shard in indexer
            ],
            shard_array,
        )

        shard = await shard_builder.finalize(self.index_location, self._encode_shard_index)
        return shard.buffer

    async def _encode_partial_single(
        self,
        byte_setter: ByteSetter,
        shard_array: NDBuffer,
        selection: SelectorTuple,
        shard_spec: ArraySpec,
    ) -> None:
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)

        # Load existing shard if it exists
        old_shard = await self._load_full_shard_maybe(
            byte_getter=byte_setter,
            prototype=chunk_spec.prototype,
            chunks_per_shard=chunks_per_shard,
            dtype=shard_spec.dtype.to_native_dtype(),
            fill_value=shard_spec.fill_value,
        )

        # Collect new chunks
        new_chunk_builder = ShardBuilder(
            chunks_per_shard=chunks_per_shard,
            chunk_shape=chunk_shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            fill_value=shard_spec.fill_value,
        )

        indexer = list(
            get_indexer(
                selection, shape=shard_shape, chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape)
            )
        )

        await self.codec_pipeline.write(
            [
                (
                    _ShardingByteSetter(new_chunk_builder, chunk_coords),
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_shard,
                )
                for chunk_coords, chunk_selection, out_selection, is_complete_shard in indexer
            ],
            shard_array,
        )

        # Merge old and new using merge_shards()
        if old_shard is None and not new_chunk_builder._chunks:
            # Nothing to write - delete the shard
            await byte_setter.delete()
        elif old_shard is not None:
            # Merge old and new
            merged_shard = await merge_shards(
                old_shard=old_shard,
                new_chunks=new_chunk_builder._chunks,
                tombstones=new_chunk_builder._tombstones,
                index_location=self.index_location,
                index_encoder=self._encode_shard_index,
            )
            await byte_setter.set(merged_shard.buffer)
        else:
            # No old shard, just new chunks - finalize the builder directly
            shard = await new_chunk_builder.finalize(self.index_location, self._encode_shard_index)
            await byte_setter.set(shard.buffer)

    def _is_total_shard(
        self, all_chunk_coords: set[tuple[int, ...]], chunks_per_shard: tuple[int, ...]
    ) -> bool:
        return len(all_chunk_coords) == product(chunks_per_shard) and all(
            chunk_coords in all_chunk_coords for chunk_coords in c_order_iter(chunks_per_shard)
        )

    async def _decode_shard_index(
        self, index_bytes: Buffer, chunks_per_shard: tuple[int, ...]
    ) -> _ShardIndex:
        index_array = next(
            iter(
                await get_pipeline_class()
                .from_codecs(self.index_codecs)
                .decode(
                    [(index_bytes, self._get_index_chunk_spec(chunks_per_shard))],
                )
            )
        )
        assert index_array is not None
        return _ShardIndex(index_array.as_numpy_array())

    async def _encode_shard_index(self, index: _ShardIndex) -> Buffer:
        index_bytes = next(
            iter(
                await get_pipeline_class()
                .from_codecs(self.index_codecs)
                .encode(
                    [
                        (
                            get_ndbuffer_class().from_numpy_array(index.offsets_and_lengths),
                            self._get_index_chunk_spec(index.chunks_per_shard),
                        )
                    ],
                )
            )
        )
        assert index_bytes is not None
        assert isinstance(index_bytes, Buffer)
        return index_bytes

    def _shard_index_size(self, chunks_per_shard: tuple[int, ...]) -> int:
        return (
            get_pipeline_class()
            .from_codecs(self.index_codecs)
            .compute_encoded_size(
                16 * product(chunks_per_shard), self._get_index_chunk_spec(chunks_per_shard)
            )
        )

    def _get_index_chunk_spec(self, chunks_per_shard: tuple[int, ...]) -> ArraySpec:
        return ArraySpec(
            shape=chunks_per_shard + (2,),
            dtype=UInt64(endianness="little"),
            fill_value=MAX_UINT_64,
            config=ArrayConfig(
                order="C", write_empty_chunks=False
            ),  # Note: this is hard-coded for simplicity -- it is not surfaced into user code,
            prototype=default_buffer_prototype(),
        )

    def _get_chunk_spec(self, shard_spec: ArraySpec) -> ArraySpec:
        return ArraySpec(
            shape=self.chunk_shape,
            dtype=shard_spec.dtype,
            fill_value=shard_spec.fill_value,
            config=shard_spec.config,
            prototype=shard_spec.prototype,
        )

    def _get_chunks_per_shard(self, shard_spec: ArraySpec) -> tuple[int, ...]:
        return tuple(
            s // c
            for s, c in zip(
                shard_spec.shape,
                self.chunk_shape,
                strict=False,
            )
        )

    async def _load_shard_index_maybe(
        self, byte_getter: ByteGetter, chunks_per_shard: tuple[int, ...]
    ) -> _ShardIndex | None:
        shard_index_size = self._shard_index_size(chunks_per_shard)
        if self.index_location == ShardingCodecIndexLocation.start:
            index_bytes = await byte_getter.get(
                prototype=numpy_buffer_prototype(),
                byte_range=RangeByteRequest(0, shard_index_size),
            )
        else:
            index_bytes = await byte_getter.get(
                prototype=numpy_buffer_prototype(), byte_range=SuffixByteRequest(shard_index_size)
            )
        if index_bytes is not None:
            return await self._decode_shard_index(index_bytes, chunks_per_shard)
        return None

    async def _load_shard_index(
        self, byte_getter: ByteGetter, chunks_per_shard: tuple[int, ...]
    ) -> _ShardIndex:
        return (
            await self._load_shard_index_maybe(byte_getter, chunks_per_shard)
        ) or _ShardIndex.create_empty(chunks_per_shard)

    async def _load_full_shard_maybe(
        self,
        byte_getter: ByteGetter,
        prototype: BufferPrototype,
        chunks_per_shard: tuple[int, ...],
        dtype: Any,
        fill_value: Any,
    ) -> Shard | None:
        shard_bytes = await byte_getter.get(prototype=prototype)

        if not shard_bytes:
            return None

        # Decode the index from the buffer
        async def decode_index(buffer: Buffer) -> _ShardIndex:
            shard_index_size = self._shard_index_size(chunks_per_shard)
            if self.index_location == ShardingCodecIndexLocation.start:
                index_bytes = buffer[:shard_index_size]
            else:
                index_bytes = buffer[-shard_index_size:]
            return await self._decode_shard_index(index_bytes, chunks_per_shard)

        return await Shard.from_buffer(
            buffer=shard_bytes,
            chunks_per_shard=chunks_per_shard,
            chunk_shape=self.chunk_shape,
            dtype=dtype,
            fill_value=fill_value,
            index_location=self.index_location,
            index_decoder=decode_index,
        )

    def compute_encoded_size(self, input_byte_length: int, shard_spec: ArraySpec) -> int:
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        return input_byte_length + self._shard_index_size(chunks_per_shard)
