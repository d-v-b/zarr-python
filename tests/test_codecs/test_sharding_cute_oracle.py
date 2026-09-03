"""Cross-check the sharding codec's offset arithmetic against the CuTe layout algebra.

The Zarr v3 sharding spec mandates that a shard's inner chunk shape evenly divide
the shard shape. That makes shard -> inner chunk -> element a *regular nested
tiling*: data-independent, uniform, zero-origin. This is exactly the regime where
the CuTe layout algebra (Cecka, arXiv:2603.02298; pure-Python reference at
https://github.com/NVlabs/CuTe) is closed, so we can derive every offset the
sharding codec computes by hand from two algebra operations and use the result as
an independent oracle:

* ``logical_divide(shard, chunk_shape)`` factors the shard's element layout into
  ``((within_0, which_0), (within_1, which_1), ...)`` -- the ``chunk_coords *
  chunk_shape + offset`` arithmetic that ``BasicIndexer`` / the bulk decode path
  perform when scattering inner chunks into the shard.
* ``blocked_product(chunk_layout, grid_layout)`` builds the layout of the shard's
  *data section*: a compact within-chunk layout (row- or column-major, i.e. plain
  ``BytesCodec`` or ``TransposeCodec + BytesCodec``) repeated over the chunk grid
  in the codec's ``subchunk_write_order``. Its second-mode strides are the byte
  offsets ``_build_shard_layout`` writes into the shard index.

Both layouts share one hierarchical coordinate space, so evaluating them at the
same integer coordinate ``i`` yields ``(where element i lives in the blob, which
shard element it is)``. If zarr's hand-written arithmetic agrees with the algebra,
``blob[blob_layout(i)] == shard_layout(i)`` for every ``i`` and every combination
of shard/chunk shape, memory order, within-chunk order, write order and index
location.

Only the *lexicographic*, *colexicographic* and (power-of-two grid) *morton* write
orders are stride layouts; morton on a non-power-of-two grid is a rank-compacted
filter of a stride layout and ``unordered`` promises nothing, so neither is
checked here (they are covered by ``test_sharding.py``).

PyCuTe is a test-only dependency (installed only in the ``optional`` hatch test
matrix from a pinned commit); the module is skipped where it is absent. It must
never be imported at runtime.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pytest

pytest.importorskip(
    "pycute",
    reason="PyCuTe (https://github.com/NVlabs/CuTe) is only installed in the "
    "'optional' hatch test matrix",
)

from pycute import (
    Layout,
    blocked_product,
    idx2crd,
    logical_divide,
    make_ordered_layout,
    shape,
    size,
)

import zarr
from zarr.codecs import BytesCodec, ShardingCodec, TransposeCodec
from zarr.codecs.sharding import MAX_UINT_64, _ShardIndex
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.dtype import parse_dtype
from zarr.core.sync import sync
from zarr.storage import MemoryStore, StorePath

if TYPE_CHECKING:
    from zarr.codecs.sharding import IndexLocation

MemOrder = Literal["C", "F"]
WriteOrder = Literal["lexicographic", "colexicographic", "morton"]

# uint32 keeps a non-trivial item size in play so element/byte confusions show up.
DTYPE = "uint32"
ITEMSIZE = 4

# (shard_shape, chunk_shape). The first is the worked example from the task
# statement (a 24-long shard of 4-long chunks); the rest cover 1-3 dimensions,
# non-square grids, a unit grid extent, and power-of-two grids for morton.
CASES: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
    ((24,), (4,)),
    ((8, 6), (4, 3)),
    ((6, 4), (3, 4)),
    ((16, 8), (4, 2)),
    ((2, 3, 4), (1, 3, 2)),
    ((8, 8, 4), (2, 2, 2)),
)


# --------------------------------------------------------------------------- #
# The oracle: layouts built purely from the algebra, in units of *elements*.
# --------------------------------------------------------------------------- #


def _row_major(ndim: int) -> tuple[int, ...]:
    # make_ordered_layout gives stride 1 to the mode with the smallest order.
    return tuple(range(ndim))[::-1]


def _col_major(ndim: int) -> tuple[int, ...]:
    return tuple(range(ndim))


def _chunks_per_shard(
    shard_shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(s // c for s, c in zip(shard_shape, chunk_shape, strict=True))


def _is_pow2_grid(chunks_per_shard: tuple[int, ...]) -> bool:
    return all(c & (c - 1) == 0 for c in chunks_per_shard)


def _shard_layout(shard_shape: tuple[int, ...]) -> Any:
    """Element coordinate -> row-major flat index of the shard array.

    The flat index doubles as the *label* stored in each element (``np.arange``
    reshaped), so ``_shard_layout(coord)`` is the value we expect to find there.
    """
    return make_ordered_layout(shard_shape, _row_major(len(shard_shape)))


def _nested_layout(shard_shape: tuple[int, ...], chunk_shape: tuple[int, ...]) -> Any:
    """``logical_divide`` of the shard by its chunk shape, mode by mode.

    Shape ``((cs_0, cps_0), (cs_1, cps_1), ...)``: mode ``d`` is
    ``(within-chunk offset, chunk coordinate)`` along axis ``d``.
    """
    return logical_divide(_shard_layout(shard_shape), tuple(Layout(c) for c in chunk_shape))


def _morton_grid_layout(chunks_per_shard: tuple[int, ...]) -> Any:
    """Bit-interleaved (Z-order) layout of a power-of-two chunk grid.

    Mode ``d`` is split into one 2-extent sub-mode per bit; the stride of each
    bit is its position in the interleaved code, using the same interleave as
    ``zarr.core.indexing._morton_order`` (bit ``k`` of every dimension, in
    dimension order, before bit ``k + 1``; dimensions with fewer bits drop out).
    On a power-of-two grid every code is a valid coordinate, so the Morton rank
    of a coordinate is its code, and the map is a plain stride layout.
    """
    if not _is_pow2_grid(chunks_per_shard):
        raise ValueError(
            f"morton is a stride layout only on power-of-two grids, got {chunks_per_shard}"
        )
    bits = [(c - 1).bit_length() for c in chunks_per_shard]
    strides: list[list[int]] = [[0] * b for b in bits]
    out_bit = 0
    for coord_bit in range(max(bits, default=0)):
        for dim, nbits in enumerate(bits):
            if coord_bit < nbits:
                strides[dim][coord_bit] = 1 << out_bit
                out_bit += 1
    layout_shape = tuple((2,) * b if b else 1 for b in bits)
    layout_stride = tuple(tuple(s) if s else 0 for s in strides)
    return Layout(layout_shape, layout_stride)


def _grid_layout(chunks_per_shard: tuple[int, ...], write_order: WriteOrder) -> Any:
    """Chunk coordinate -> rank of that chunk in the shard's data section."""
    ndim = len(chunks_per_shard)
    if write_order == "lexicographic":
        return make_ordered_layout(chunks_per_shard, _row_major(ndim))
    if write_order == "colexicographic":
        return make_ordered_layout(chunks_per_shard, _col_major(ndim))
    return _morton_grid_layout(chunks_per_shard)


def _blob_layout(
    chunk_shape: tuple[int, ...],
    inner_order: MemOrder,
    chunks_per_shard: tuple[int, ...],
    write_order: WriteOrder,
) -> Any:
    """Hierarchical coordinate -> element offset within the shard's data section.

    ``blocked_product(tile, grid)`` repeats the compact within-chunk layout
    ``tile`` once per grid position, so its shape is
    ``((cs_0, g_0), (cs_1, g_1), ...)`` -- the same profile as ``_nested_layout``
    -- and its second-mode strides are multiples of ``size(tile)``.
    """
    ndim = len(chunk_shape)
    order = _row_major(ndim) if inner_order == "C" else _col_major(ndim)
    return blocked_product(
        make_ordered_layout(chunk_shape, order), _grid_layout(chunks_per_shard, write_order)
    )


def _grid_natural_coord(grid: Any, j: int) -> tuple[int, ...]:
    """The zarr chunk coordinate for the ``j``-th grid position.

    ``idx2crd`` decomposes ``j`` over the grid's (possibly bit-split, for morton)
    modes; collapsing each mode's coordinate back to an integer gives the plain
    chunk coordinate.
    """
    from pycute import crd2idx

    crd = idx2crd(j, shape(grid))
    return tuple(int(crd2idx(c, s)) for c, s in zip(crd, shape(grid), strict=True))


# --------------------------------------------------------------------------- #
# Zarr-side helpers.
# --------------------------------------------------------------------------- #


def _inner_codecs(ndim: int, inner_order: MemOrder) -> list[Any]:
    if inner_order == "C":
        return [BytesCodec(endian="little")]
    return [TransposeCodec(order=tuple(range(ndim))[::-1]), BytesCodec(endian="little")]


def _labels(shard_shape: tuple[int, ...], mem_order: MemOrder) -> np.ndarray[Any, Any]:
    n = int(np.prod(shard_shape))
    return np.asarray(np.arange(n, dtype=DTYPE).reshape(shard_shape), order=mem_order)


def _chunk_key(ndim: int) -> str:
    return "c/" + "/".join("0" * ndim) if ndim else "c"


def _shard_spec(shard_shape: tuple[int, ...], mem_order: MemOrder) -> ArraySpec:
    return ArraySpec(
        shape=shard_shape,
        dtype=parse_dtype(DTYPE, zarr_format=3),
        fill_value=0,
        config=ArrayConfig(order=mem_order, write_empty_chunks=False),
        prototype=default_buffer_prototype(),
    )


def _write_one_shard(
    codec: ShardingCodec,
    shard_shape: tuple[int, ...],
    mem_order: MemOrder,
) -> bytes:
    """Write a single fully-populated shard through the public API; return the raw blob."""
    store = MemoryStore()
    arr = zarr.create_array(
        StorePath(store),
        shape=shard_shape,
        dtype=DTYPE,
        chunks=shard_shape,
        serializer=codec,
        filters=None,
        compressors=None,
        fill_value=0,
        config={"order": mem_order},
    )
    arr[:] = _labels(shard_shape, mem_order)
    buf = sync(store.get(_chunk_key(len(shard_shape)), prototype=default_buffer_prototype()))
    assert buf is not None, "shard was not written"
    return buf.to_bytes()


def _param_grid() -> list[Any]:
    """(shard_shape, chunk_shape, inner_order, write_order), morton on pow2 grids only."""
    params = []
    for (shard_shape, chunk_shape), inner_order, write_order in itertools.product(
        CASES, ("C", "F"), ("lexicographic", "colexicographic", "morton")
    ):
        if write_order == "morton" and not _is_pow2_grid(
            _chunks_per_shard(shard_shape, chunk_shape)
        ):
            continue
        params.append(
            pytest.param(
                shard_shape,
                chunk_shape,
                inner_order,
                write_order,
                id=f"{shard_shape}-{chunk_shape}-inner{inner_order}-{write_order}",
            )
        )
    return params


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #


def test_oracle_worked_example() -> None:
    """Pin the algebra on the task's worked example, so a PyCuTe API/semantics
    change fails loudly here rather than as a confusing sharding failure."""
    div = logical_divide(Layout(24), Layout(4))
    assert div == Layout((4, 6), (1, 4))  # (within-chunk, which-chunk)
    nested = logical_divide[1](div, Layout(3))
    assert nested == Layout((4, (3, 2)), (1, (4, 12)))
    # and the same thing phrased the way the tests below use it
    assert _nested_layout((24,), (4,)) == Layout(((4, 6),), ((1, 4),))
    assert shape(_nested_layout((24,), (4,)))[0][1] == 6


@pytest.mark.parametrize(
    ("shard_shape", "chunk_shape", "inner_order", "write_order"), _param_grid()
)
@pytest.mark.parametrize("index_location", ["start", "end"])
@pytest.mark.parametrize("mem_order", ["C", "F"])
def test_encoded_shard_matches_layout_oracle(
    shard_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    inner_order: MemOrder,
    write_order: WriteOrder,
    index_location: IndexLocation,
    mem_order: MemOrder,
) -> None:
    """A shard written by zarr has every element at the byte offset the algebra
    predicts, and its index holds the offsets/lengths the grid layout predicts."""
    ndim = len(shard_shape)
    cps = _chunks_per_shard(shard_shape, chunk_shape)
    codec = ShardingCodec(
        chunk_shape=chunk_shape,
        codecs=_inner_codecs(ndim, inner_order),
        index_location=index_location,
        subchunk_write_order=write_order,
    )
    blob = _write_one_shard(codec, shard_shape, mem_order)

    nested = _nested_layout(shard_shape, chunk_shape)
    grid = _grid_layout(cps, write_order)
    blob_layout = _blob_layout(chunk_shape, inner_order, cps, write_order)

    # zarr's chunk grid is the "which chunk" mode of the divide, per axis
    assert codec._get_chunks_per_shard(_shard_spec(shard_shape, mem_order)) == cps
    assert tuple(shape(nested)[d][1] for d in range(ndim)) == cps

    n_elems = int(np.prod(shard_shape))
    n_inner = int(np.prod(chunk_shape))
    index_size = codec._shard_index_size(cps)
    data_start = index_size if index_location == "start" else 0
    assert len(blob) == n_elems * ITEMSIZE + index_size
    assert size(nested) == size(blob_layout) == n_elems

    # every element: value found where the blob layout says == label the divide says
    words = np.frombuffer(blob, dtype="<u4", count=n_elems, offset=data_start)
    for i in range(n_elems):
        assert words[blob_layout(i)] == nested(i), (
            f"element {idx2crd(i, shape(nested))}: blob has {words[blob_layout(i)]} "
            f"at data offset {blob_layout(i)}, algebra expects {nested(i)}"
        )

    # the shard index: chunk c is at data_start + rank(c) * chunk_bytes, length chunk_bytes
    index_bytes = blob[:index_size] if index_location == "start" else blob[-index_size:]
    index = codec._decode_shard_index_sync(
        default_buffer_prototype().buffer.from_bytes(index_bytes), cps
    )
    for j in range(size(grid)):
        c = _grid_natural_coord(grid, j)
        offset, length = (int(x) for x in index.offsets_and_lengths[c])
        assert (offset, length) == (
            data_start + grid(j) * n_inner * ITEMSIZE,
            n_inner * ITEMSIZE,
        ), f"chunk {c}: index says ({offset}, {length}), algebra rank {grid(j)}"
    # ...and the index array itself is stored row-major over chunks_per_shard + (2,)
    raw_entries = np.frombuffer(index_bytes, dtype="<u8", count=2 * int(np.prod(cps)))
    entry_layout = make_ordered_layout(cps + (2,), _row_major(ndim + 1))
    for j in range(size(grid)):
        c = _grid_natural_coord(grid, j)
        assert raw_entries[entry_layout((*c, 0))] == data_start + grid(j) * n_inner * ITEMSIZE
        assert raw_entries[entry_layout((*c, 1))] == n_inner * ITEMSIZE


@pytest.mark.parametrize(
    ("shard_shape", "chunk_shape", "inner_order", "write_order"), _param_grid()
)
@pytest.mark.parametrize("index_location", ["start", "end"])
@pytest.mark.parametrize("mem_order", ["C", "F"])
def test_decode_of_oracle_built_shard(
    shard_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    inner_order: MemOrder,
    write_order: WriteOrder,
    index_location: IndexLocation,
    mem_order: MemOrder,
) -> None:
    """The converse: a shard blob assembled *only* from the algebra (data section
    via the blob layout, index via the grid layout, no zarr involved) decodes to
    the labelled array through both the public read path and ``_decode_sync``."""
    ndim = len(shard_shape)
    cps = _chunks_per_shard(shard_shape, chunk_shape)
    n_elems = int(np.prod(shard_shape))
    n_inner = int(np.prod(chunk_shape))
    n_chunks = int(np.prod(cps))

    nested = _nested_layout(shard_shape, chunk_shape)
    grid = _grid_layout(cps, write_order)
    blob_layout = _blob_layout(chunk_shape, inner_order, cps, write_order)

    # index without a checksum so the whole blob can be built without zarr
    index_size = 16 * n_chunks
    data_start = index_size if index_location == "start" else 0

    words = np.empty(n_elems, dtype="<u4")
    for i in range(n_elems):
        words[blob_layout(i)] = nested(i)
    entries = np.full(cps + (2,), MAX_UINT_64, dtype="<u8", order="C")
    for j in range(size(grid)):
        c = _grid_natural_coord(grid, j)
        entries[c] = (data_start + grid(j) * n_inner * ITEMSIZE, n_inner * ITEMSIZE)
    data, index = words.tobytes(), entries.tobytes()
    blob = index + data if index_location == "start" else data + index

    codec = ShardingCodec(
        chunk_shape=chunk_shape,
        codecs=_inner_codecs(ndim, inner_order),
        index_codecs=[BytesCodec(endian="little")],
        index_location=index_location,
        subchunk_write_order=write_order,
    )
    assert codec._shard_index_size(cps) == index_size
    expected = _labels(shard_shape, "C")

    # public read path
    store = MemoryStore()
    arr = zarr.create_array(
        StorePath(store),
        shape=shard_shape,
        dtype=DTYPE,
        chunks=shard_shape,
        serializer=codec,
        filters=None,
        compressors=None,
        fill_value=0,
        config={"order": mem_order},
    )
    sync(store.set(_chunk_key(ndim), default_buffer_prototype().buffer.from_bytes(blob)))
    np.testing.assert_array_equal(arr[:], expected)
    # a strided read hits the partial-decode path
    sel = tuple(slice(0, s, 2) for s in shard_shape)
    np.testing.assert_array_equal(arr[sel], expected[sel])

    # sync whole-shard decode (used by the fused pipeline)
    spec = _shard_spec(shard_shape, mem_order)
    decoded = codec._decode_sync(default_buffer_prototype().buffer.from_bytes(blob), spec)
    np.testing.assert_array_equal(decoded.as_numpy_array(), expected)


@pytest.mark.parametrize(("shard_shape", "chunk_shape"), CASES)
@pytest.mark.parametrize("write_order", ["lexicographic", "colexicographic"])
def test_chunk_localization_is_divide_remainder(
    shard_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    write_order: WriteOrder,
) -> None:
    """Array-level chunk coordinates localize into a shard by ``% chunks_per_shard``.

    Dividing the array's chunk grid by the shard's chunk grid, axis by axis,
    ``logical_divide(Layout(n_chunks_d), Layout(cps_d))`` has shape
    ``(cps_d, n_shards_d)``: the first coordinate of ``idx2crd`` is the in-shard
    chunk coordinate. ``_ShardIndex._localize_chunk`` and the vectorized lookup
    must agree with it (checked over a 2-shards-per-axis array chunk grid).
    """
    cps = _chunks_per_shard(shard_shape, chunk_shape)
    n_shards = 2
    grid = _grid_layout(cps, write_order)
    axis_divides = [logical_divide(Layout(c * n_shards), Layout(c)) for c in cps]

    # an oracle-built index: chunk c at rank(c) (unit chunk length, offsets in ranks)
    index = _ShardIndex.create_empty(cps)
    for j in range(size(grid)):
        c = _grid_natural_coord(grid, j)
        index.offsets_and_lengths[c] = (grid(j), 1)

    abs_coords = list(itertools.product(*(range(c * n_shards) for c in cps)))
    starts, ends, valid = index.get_chunk_slices_vectorized(np.array(abs_coords, dtype=np.intp))
    assert valid.all()
    for k, abs_c in enumerate(abs_coords):
        local = tuple(idx2crd(a, shape(d))[0] for a, d in zip(abs_c, axis_divides, strict=True))
        assert index._localize_chunk(abs_c) == local
        assert index.get_chunk_slice(abs_c) == (grid(local), grid(local) + 1)
        assert (int(starts[k]), int(ends[k])) == (grid(local), grid(local) + 1)


def test_morton_grid_layout_rejects_non_pow2() -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        _morton_grid_layout((3, 2))


def test_grid_layouts_match_zarr_iteration_order() -> None:
    """The oracle's grid layouts rank chunks in the same order zarr writes them
    (``_subchunk_order_iter``); this ties ``_grid_layout`` to zarr's own
    definition of each write order, independent of any shard bytes."""
    codec = ShardingCodec(chunk_shape=(1, 1, 1))
    for cps, write_order in itertools.product(
        [(4, 2, 8), (1, 4, 2), (3, 2, 5)], ("lexicographic", "colexicographic", "morton")
    ):
        write_order = cast("WriteOrder", write_order)
        if write_order == "morton" and not _is_pow2_grid(cps):
            continue
        grid = _grid_layout(cps, write_order)
        ranks = [grid(c) for c in codec._subchunk_order_iter(cps, write_order)]
        assert ranks == list(range(int(np.prod(cps)))), (cps, write_order)
