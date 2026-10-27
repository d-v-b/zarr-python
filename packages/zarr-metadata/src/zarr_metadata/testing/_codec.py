"""Strategies for each codec this package defines.

A codec handed an array draws what fits the chunk: a `bytes` codec an
`endian` for a type whose values take several bytes, a `transpose` an
`order` with an entry for each axis, a shard inner chunks that divide
it. Handed `Chunk()`, nothing is known of the chunk, and what is drawn
is what the codec's rules accept. The fields a codec holds are drawn
from the strategies: the data type a `cast_value` casts to, and a
shard's two pipelines, each drawn for the chunk the shard's definition
says it is handed.
"""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Any, Final

from hypothesis import reject
from hypothesis import strategies as st

from zarr_metadata.testing._data_type import FLOAT_WIDTHS, INTEGER_RANGES, finite_floats
from zarr_metadata.testing._draw import (
    briefly,
    data_type_fields,
    data_types,
    fill_values_of,
    pipelines_for,
    read,
)
from zarr_metadata.testing._strategy import CodecStrategy
from zarr_metadata.v3._definition import Chunk, DataTypeDefinition, storage_of
from zarr_metadata.v3.codec._arithmetic import FLOATING_POINT, NO_REAL_NUMBERS, read_name
from zarr_metadata.v3.codec.blosc import BLOSC_CNAME, BLOSC_CODEC, BLOSC_NO_SHUFFLE, BLOSC_SHUFFLE
from zarr_metadata.v3.codec.bytes import BYTES_CODEC, ENDIANNESS
from zarr_metadata.v3.codec.cast_value import (
    CAST_OUT_OF_RANGE_MODE,
    CAST_ROUNDING_MODE,
    CAST_VALUE_CODEC,
)
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.codec.scale_offset import SCALE_OFFSET_CODEC
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEX_LOCATION, SHARDING_INDEXED_CODEC
from zarr_metadata.v3.codec.transpose import TRANSPOSE_CODEC
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC, ZSTD_MAX_LEVEL, ZSTD_MIN_LEVEL

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata._common import JSONValue
    from zarr_metadata.testing._strategy import Configurations, Strategies
    from zarr_metadata.v3._definition import CodecDefinition, CodecSize, Lengths, Resolved
    from zarr_metadata.v3.codec.sharding_indexed import ShardingIndexedCodecConfiguration

_MAX_RANK: Final = 5
"""The most axes an `order` is drawn with, when the chunk does not say."""

_BLOSC_MAX_TYPESIZE: Final = 255
"""The widest element c-blosc takes: `BLOSC_MAX_TYPESIZE`."""

_BLOSC_MAX_BLOCKSIZE: Final = 2**31 - 1
"""The largest block c-blosc takes, a signed 32-bit size; 0 lets it choose."""

_MAX_INNER_LENGTH: Final = 16
"""The longest inner chunk drawn along an axis whose shard lengths are unknown."""

_MAX_SHARD_DEPTH: Final = 1
"""How deep shards are drawn inside a shard: one inside another, and none inside that."""

_MAX_SCALARS: Final = 3
"""The most entries each direction of a `cast_value` `scalar_map` is drawn with."""

_STATIC: Final[frozenset[CodecSize]] = frozenset({"static"})
"""What a shard index's codecs are: of static size, so a reader finds the index before reading it."""

_ENDIAN: Final = st.sampled_from(ENDIANNESS)

_SMALL_NUMBERS: Final = st.integers(0, 127)
"""Values every integral and floating-point data type holds: for a chunk whose data type is not known."""


def _bytes(chunk: Chunk, _strategies: Strategies) -> Configurations | None:
    """An `endian`, required for a data type whose values take several bytes; None for one whose values vary in size."""
    storage = None if chunk.data_type is None else storage_of(chunk.data_type)
    if storage == "variable_length":
        return None
    if storage == "multi_byte":
        return st.fixed_dictionaries({"endian": _ENDIAN})
    return st.fixed_dictionaries({}, optional={"endian": _ENDIAN})


@st.composite
def _blosc(draw: st.DrawFn) -> Mapping[str, JSONValue]:
    """A compressor, a level, a shuffle and a block size; and a `typesize`, which only `noshuffle` may leave out.

    Under `noshuffle` the spec says `typesize` "is ignored", so it is
    drawn there too, a spelling that is not the simplest.
    """
    shuffle = draw(st.sampled_from(BLOSC_SHUFFLE))
    typesize = st.integers(1, _BLOSC_MAX_TYPESIZE)
    required: dict[str, st.SearchStrategy[JSONValue]] = {
        "cname": st.sampled_from(BLOSC_CNAME),
        "clevel": st.integers(0, 9),
        "shuffle": st.just(shuffle),
        "blocksize": st.integers(0, _BLOSC_MAX_BLOCKSIZE),
    }
    if shuffle == BLOSC_NO_SHUFFLE:
        return draw(st.fixed_dictionaries(required, optional={"typesize": typesize}))
    return draw(st.fixed_dictionaries({**required, "typesize": typesize}))


def _order(rank: int) -> st.SearchStrategy[Mapping[str, JSONValue]]:
    return st.permutations(range(rank)).map(lambda order: {"order": tuple(order)})


def _transpose(chunk: Chunk, _strategies: Strategies) -> Configurations:
    """A permutation of the chunk's axes; of some number of axes, when the chunk does not say."""
    if chunk.rank is not None:
        return _order(chunk.rank)
    return st.integers(0, _MAX_RANK).flatmap(_order)


def _scale_offset(chunk: Chunk, _strategies: Strategies) -> Configurations | None:
    """An `offset` and a `scale`, each a finite value of the data type the codec is handed, the scale not zero.

    Each is a fill value of the chunk's data type
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md?plain=1#L39-L43),
    drawn finite, and the scale non-zero, so that decoding, which divides
    by it, is defined. Handed a data type the codec's registry entry does
    not list -- truth values, raw bits, complex numbers, numpy times --
    nothing is drawn: for some the spec refuses it, and for the others
    leaves it open. Handed a chunk whose data type is unknown, the values
    are small non-negative integers, which every listed type holds.
    """
    data_type = chunk.data_type
    is_read = data_type is not None and data_type.resolution == "read"
    definition = data_type.definition if data_type is not None and is_read else None
    if definition is None:
        offset: st.SearchStrategy[JSONValue] = _SMALL_NUMBERS
        scale: st.SearchStrategy[JSONValue] = st.integers(1, 127)
    elif definition in INTEGER_RANGES:
        low, high = INTEGER_RANGES[definition]
        offset = st.integers(low, high)
        scale = st.integers(low, high).filter(lambda value: value != 0)
    elif definition in FLOAT_WIDTHS:
        offset = finite_floats(FLOAT_WIDTHS[definition])
        scale = finite_floats(FLOAT_WIDTHS[definition]).filter(lambda value: value != 0)
    else:
        return None
    return st.fixed_dictionaries({}, optional={"offset": offset, "scale": scale})


def _unchunked(configurations: Configurations) -> CodecStrategy:
    """A codec whose configurations do not depend on the chunk it is handed."""
    return CodecStrategy(configurations=lambda _chunk, _strategies: configurations)


def _source_scalars(chunk: Chunk, strategies: Strategies) -> st.SearchStrategy[JSONValue]:
    """Values of the chunk's data type: its fill values, or small numbers when no scope read it or no strategy draws for it."""
    source = chunk.data_type
    if source is None or read_name(source) is None or source.definition not in strategies:
        return _SMALL_NUMBERS
    return fill_values_of(source, strategies)


@st.composite
def _cast_value_configuration(
    draw: st.DrawFn,
    targets: list[DataTypeDefinition[Any]],
    sources: st.SearchStrategy[JSONValue],
    strategies: Strategies,
) -> Mapping[str, JSONValue]:
    """A data type to cast to, a rounding, an out-of-range mode it allows, and a scalar map between the two types."""
    field = draw(data_type_fields(targets, strategies))
    target: Resolved[DataTypeDefinition[Any]] = read(field, DataTypeDefinition, strategies)
    values = fill_values_of(target, strategies)
    modes = CAST_OUT_OF_RANGE_MODE
    if read_name(target) in FLOATING_POINT:
        modes = tuple(mode for mode in modes if mode != "wrap")
    encode = st.lists(st.tuples(sources, values), max_size=_MAX_SCALARS).map(tuple)
    decode = st.lists(st.tuples(values, sources), max_size=_MAX_SCALARS).map(tuple)
    scalar_map = st.fixed_dictionaries({}, optional={"encode": encode, "decode": decode})
    optional: dict[str, st.SearchStrategy[Any]] = {
        "rounding": st.sampled_from(CAST_ROUNDING_MODE),
        "out_of_range": st.sampled_from(modes),
        "scalar_map": scalar_map,
    }
    return draw(st.fixed_dictionaries({"data_type": st.just(field)}, optional=optional))


def _cast_value(chunk: Chunk, strategies: Strategies) -> Configurations | None:
    """A cast to a data type that models real numbers, from one that does; None for a chunk of one that does not.

    The codec "is only defined for data types that model real numbers"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L5-L9),
    on either side of the cast. Each `scalar_map` entry pairs a value of
    the type it casts from with one of the type it casts to, in the order
    of its direction; `wrap` is drawn only for a type that is not
    floating-point.
    """
    if read_name(chunk.data_type) in NO_REAL_NUMBERS:
        return None
    targets = [d for d in data_types(strategies) if d.name not in NO_REAL_NUMBERS]
    if len(targets) == 0:
        return None
    return _cast_value_configuration(targets, _source_scalars(chunk, strategies), strategies)


def _divisors(lengths: frozenset[int] | None) -> st.SearchStrategy[int]:
    """An inner chunk length that divides each of `lengths`; any, when they are unknown or none but 0."""
    common = 0 if lengths is None else math.gcd(*lengths)
    if common == 0:
        return st.integers(1, _MAX_INNER_LENGTH)
    # Divisors come in pairs, one of each no greater than the square root.
    small = [n for n in range(1, math.isqrt(common) + 1) if common % n == 0]
    return st.sampled_from(sorted({*small, *(common // n for n in small)}))


def _inner_chunk_shapes(lengths: Lengths | None) -> st.SearchStrategy[tuple[int, ...]]:
    """An inner chunk length for each axis of the shard, dividing the lengths shards take along it."""
    if lengths is None:
        return st.lists(st.integers(1, _MAX_INNER_LENGTH), max_size=_MAX_RANK).map(tuple)
    return st.tuples(*(_divisors(axis) for axis in lengths))


@st.composite
def _sharding_configuration(
    draw: st.DrawFn, chunk: Chunk, strategies: Strategies, inner: Strategies
) -> Mapping[str, JSONValue]:
    """Inner chunks that divide the shard, and the two pipelines, each drawn for what the shard's definition says it is handed.

    The inner codecs are drawn from `inner`, which says how a shard
    inside this one is drawn, if at all.
    """
    chunk_shape = draw(_inner_chunk_shapes(chunk.lengths))
    sized: ShardingIndexedCodecConfiguration = {
        "chunk_shape": chunk_shape,
        "codecs": (),
        "index_codecs": (),
    }
    handed = SHARDING_INDEXED_CODEC.pipelines(sized, {}, chunk)
    codecs = pipelines_for(handed["codecs"], inner)
    index_codecs = pipelines_for(handed["index_codecs"], strategies, _STATIC)
    if codecs is None or index_codecs is None:
        reject()  # a codec that takes a chunk of the data type, but not of these lengths
    configuration: dict[str, st.SearchStrategy[Any]] = {
        "chunk_shape": st.just(chunk_shape),
        "codecs": codecs,
        "index_codecs": index_codecs,
    }
    optional = {"index_location": st.sampled_from(SHARDING_INDEX_LOCATION)}
    return draw(st.fixed_dictionaries(configuration, optional=optional))


_NO_SHAPE: Final[ShardingIndexedCodecConfiguration] = {
    "chunk_shape": (),
    "codecs": (),
    "index_codecs": (),
}
"""A shard configuration that says nothing but what its definition's `pipelines` reads: the data type each pipeline is handed."""


def _sharding(chunk: Chunk, strategies: Strategies, *, depth: int) -> Configurations | None:
    """A shard of the chunk, holding shards `depth` deep at most; None when no pipeline is drawn for its inner chunks or its index.

    Each pipeline needs a codec that holds no pipelines and takes the
    data type its definition hands it -- the shard's own for the inner
    chunks, and for the index one of static size; whether one does is
    asked of a chunk of that data type and nothing else known, since the
    inner lengths are drawn after. Its inner codecs are drawn from the
    same strategies but for a shard, which holds one level fewer, or is
    not drawn: shards drawn without end would nest without end.
    """
    handed = SHARDING_INDEXED_CODEC.pipelines(_NO_SHAPE, {}, Chunk(None, chunk.data_type))
    if pipelines_for(Chunk(None, handed["codecs"].data_type), strategies) is None:
        return None
    if pipelines_for(Chunk(None, handed["index_codecs"].data_type), strategies, _STATIC) is None:
        return None
    inner = {d: s for d, s in strategies.items() if d is not SHARDING_INDEXED_CODEC}
    if depth > 0:
        inner[SHARDING_INDEXED_CODEC] = _shards(depth - 1)
    return _sharding_configuration(chunk, strategies, briefly(inner))


def _shards(depth: int) -> CodecStrategy:
    """Shards that hold shards `depth` deep at most."""
    return CodecStrategy(configurations=partial(_sharding, depth=depth))


CODEC_STRATEGIES: Final[Mapping[CodecDefinition[Any], CodecStrategy]] = {
    BLOSC_CODEC: _unchunked(_blosc()),
    BYTES_CODEC: CodecStrategy(configurations=_bytes),
    CAST_VALUE_CODEC: CodecStrategy(configurations=_cast_value),
    CRC32C_CODEC: _unchunked(st.just({})),
    GZIP_CODEC: _unchunked(st.fixed_dictionaries({"level": st.integers(0, 9)})),
    SCALE_OFFSET_CODEC: CodecStrategy(configurations=_scale_offset),
    SHARDING_INDEXED_CODEC: _shards(_MAX_SHARD_DEPTH),
    TRANSPOSE_CODEC: CodecStrategy(configurations=_transpose),
    ZSTD_CODEC: _unchunked(
        st.fixed_dictionaries(
            {"level": st.integers(ZSTD_MIN_LEVEL, ZSTD_MAX_LEVEL)},
            optional={"checksum": st.booleans()},
        )
    ),
}
"""A strategy for each codec."""

__all__ = ["CODEC_STRATEGIES"]
