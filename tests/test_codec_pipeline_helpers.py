"""Unit tests for the per-chunk primitives in zarr.core.codec_pipeline.

Each function exposed by ``codec_pipeline`` for SyncCodecPipeline composition
gets its own focused test. These tests don't construct stores, codecs, or
arrays — they exercise the primitive directly with mocks or trivial inputs.

Compare to ``test_pipeline_parity`` (integration: full pipeline vs. baseline)
and ``test_codec_invariants`` (contract: codecs satisfy ABCs). This file
sits below both: each helper, in isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np

from zarr.codecs.bytes import BytesCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.codec_pipeline import (
    ChunkTransform,
    delete_key,
    dispatch,
    merge_chunk,
    read_chunk,
    read_key,
    scatter_into,
    should_skip_chunk,
    write_chunk,
    write_key,
)
from zarr.core.dtype import get_data_type_from_native_dtype

if TYPE_CHECKING:
    from zarr.core.buffer import NDBuffer


def _spec(
    shape: tuple[int, ...] = (4,),
    dtype: str = "uint8",
    *,
    fill_value: int = 0,
    write_empty_chunks: bool = False,
) -> ArraySpec:
    zdtype = get_data_type_from_native_dtype(np.dtype(dtype))
    return ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(fill_value),
        config=ArrayConfig(order="C", write_empty_chunks=write_empty_chunks),
        prototype=default_buffer_prototype(),
    )


def _ndbuffer_from_array(arr: np.ndarray[Any, np.dtype[Any]]) -> NDBuffer:
    return default_buffer_prototype().nd_buffer.from_numpy_array(arr)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def test_dispatch_sequential_when_max_workers_is_one() -> None:
    """max_workers=1 runs in the calling thread, no pool involvement."""
    import threading

    main_tid = threading.get_ident()
    seen_tids: list[int] = []

    def fn(_x: int) -> int:
        seen_tids.append(threading.get_ident())
        return _x * 2

    out = dispatch([1, 2, 3], fn, max_workers=1)
    assert out == [2, 4, 6]
    assert all(t == main_tid for t in seen_tids)


def test_dispatch_threaded_when_max_workers_above_one() -> None:
    """max_workers>1 with len(items)>1 uses the pool."""
    import threading

    main_tid = threading.get_ident()
    seen_tids: list[int] = []
    lock = threading.Lock()

    def fn(_x: int) -> int:
        with lock:
            seen_tids.append(threading.get_ident())
        return _x * 2

    out = dispatch([1, 2, 3, 4], fn, max_workers=4)
    assert sorted(out) == [2, 4, 6, 8]
    # At least one task ran in a worker thread (not the main thread).
    assert any(t != main_tid for t in seen_tids)


def test_dispatch_single_item_runs_sequential_even_with_workers() -> None:
    """A single-item batch always runs inline (avoids pool overhead)."""
    import threading

    main_tid = threading.get_ident()
    captured = []

    def fn(_x: int) -> int:
        captured.append(threading.get_ident())
        return _x

    dispatch([42], fn, max_workers=8)
    assert captured == [main_tid]


# ---------------------------------------------------------------------------
# scatter_into
# ---------------------------------------------------------------------------


def test_scatter_into_basic_assignment() -> None:
    out = _ndbuffer_from_array(np.zeros(10, dtype="uint8"))
    chunk = _ndbuffer_from_array(np.array([1, 2, 3], dtype="uint8"))
    scatter_into(out, (slice(2, 5),), chunk)
    np.testing.assert_array_equal(
        out.as_ndarray_like(), np.array([0, 0, 1, 2, 3, 0, 0, 0, 0, 0], dtype="uint8")
    )


def test_scatter_into_drops_axes() -> None:
    """drop_axes squeezes the chunk before assignment."""
    out = _ndbuffer_from_array(np.zeros(4, dtype="uint8"))
    # Chunk has a singleton axis at position 0
    chunk = _ndbuffer_from_array(np.array([[1, 2, 3, 4]], dtype="uint8"))
    scatter_into(out, (slice(None),), chunk, drop_axes=(0,))
    np.testing.assert_array_equal(out.as_ndarray_like(), np.array([1, 2, 3, 4], dtype="uint8"))


# ---------------------------------------------------------------------------
# merge_chunk
# ---------------------------------------------------------------------------


def test_merge_chunk_returns_value_directly_for_complete_overwrite() -> None:
    """is_complete + matching shapes → the value is returned unchanged."""
    spec = _spec(shape=(4,))
    value = _ndbuffer_from_array(np.array([1, 2, 3, 4], dtype="uint8"))
    result = merge_chunk(
        existing=None,
        value=value,
        chunk_spec=spec,
        chunk_selection=(slice(None),),
        out_selection=(slice(None),),
        is_complete=True,
    )
    assert result is value


def test_merge_chunk_creates_fill_buffer_when_no_existing() -> None:
    """Partial write into a fresh chunk: untouched cells get fill_value."""
    spec = _spec(shape=(4,), fill_value=9)
    value = _ndbuffer_from_array(np.array([1, 2], dtype="uint8"))
    result = merge_chunk(
        existing=None,
        value=value,
        chunk_spec=spec,
        chunk_selection=(slice(0, 2),),
        out_selection=(slice(0, 2),),
        is_complete=False,
    )
    np.testing.assert_array_equal(result.as_ndarray_like(), np.array([1, 2, 9, 9], dtype="uint8"))


def test_merge_chunk_overwrites_existing_region() -> None:
    """Partial write merges into an existing chunk."""
    spec = _spec(shape=(4,), fill_value=0)
    existing = _ndbuffer_from_array(np.array([5, 5, 5, 5], dtype="uint8"))
    value = _ndbuffer_from_array(np.array([1, 2], dtype="uint8"))
    result = merge_chunk(
        existing=existing,
        value=value,
        chunk_spec=spec,
        chunk_selection=(slice(0, 2),),
        out_selection=(slice(0, 2),),
        is_complete=False,
    )
    np.testing.assert_array_equal(result.as_ndarray_like(), np.array([1, 2, 5, 5], dtype="uint8"))


# ---------------------------------------------------------------------------
# should_skip_chunk
# ---------------------------------------------------------------------------


def test_should_skip_chunk_true_when_all_fill_and_skip_empty() -> None:
    spec = _spec(shape=(4,), fill_value=7, write_empty_chunks=False)
    chunk = _ndbuffer_from_array(np.full(4, 7, dtype="uint8"))
    assert should_skip_chunk(chunk, spec)


def test_should_skip_chunk_false_when_write_empty_chunks_true() -> None:
    """write_empty_chunks=True means no chunk is ever 'empty'."""
    spec = _spec(shape=(4,), fill_value=7, write_empty_chunks=True)
    chunk = _ndbuffer_from_array(np.full(4, 7, dtype="uint8"))
    assert not should_skip_chunk(chunk, spec)


def test_should_skip_chunk_false_when_chunk_differs_from_fill() -> None:
    spec = _spec(shape=(4,), fill_value=0, write_empty_chunks=False)
    chunk = _ndbuffer_from_array(np.array([0, 1, 0, 0], dtype="uint8"))
    assert not should_skip_chunk(chunk, spec)


# ---------------------------------------------------------------------------
# read_key / write_key / delete_key
# ---------------------------------------------------------------------------


def test_read_key_calls_get_sync_with_prototype() -> None:
    bg = MagicMock()
    bg.get_sync.return_value = "fake_buffer"
    proto = default_buffer_prototype()
    result = read_key(bg, proto)
    assert result == "fake_buffer"
    bg.get_sync.assert_called_once_with(prototype=proto)


def test_write_key_calls_set_sync() -> None:
    bs = MagicMock()
    buf = MagicMock(spec=["__len__"])
    write_key(bs, buf)
    bs.set_sync.assert_called_once_with(buf)


def test_delete_key_calls_delete_sync() -> None:
    bs = MagicMock()
    delete_key(bs)
    bs.delete_sync.assert_called_once_with()


# ---------------------------------------------------------------------------
# read_chunk
# ---------------------------------------------------------------------------


def test_read_chunk_returns_none_for_missing_key() -> None:
    """When the byte_getter returns None, read_chunk returns None."""
    bg = MagicMock()
    bg.get_sync.return_value = None
    transform = ChunkTransform(codecs=(BytesCodec(),))
    spec = _spec(shape=(4,))
    result = read_chunk(bg, transform, spec, (slice(None),))
    assert result is None


def test_read_chunk_decodes_and_slices() -> None:
    """Round-trip: encode an array, then read_chunk decodes a slice of it."""
    transform = ChunkTransform(codecs=(BytesCodec(),))
    spec = _spec(shape=(8,))
    arr = _ndbuffer_from_array(np.arange(8, dtype="uint8"))
    encoded = transform.encode_chunk(arr, spec)
    assert encoded is not None

    bg = MagicMock()
    bg.get_sync.return_value = encoded
    result = read_chunk(bg, transform, spec, (slice(2, 5),))
    assert result is not None
    np.testing.assert_array_equal(result.as_ndarray_like(), np.array([2, 3, 4], dtype="uint8"))


# ---------------------------------------------------------------------------
# write_chunk
# ---------------------------------------------------------------------------


def test_write_chunk_full_overwrite_skips_existence_read() -> None:
    """is_complete=True must NOT call get_sync (no read of existing)."""
    transform = ChunkTransform(codecs=(BytesCodec(),))
    spec = _spec(shape=(4,))
    value = _ndbuffer_from_array(np.array([1, 2, 3, 4], dtype="uint8"))
    bs = MagicMock()
    write_chunk(
        bs,
        transform,
        spec,
        value,
        chunk_selection=(slice(None),),
        out_selection=(slice(None),),
        is_complete=True,
    )
    bs.get_sync.assert_not_called()
    bs.set_sync.assert_called_once()


def test_write_chunk_partial_reads_existing_first() -> None:
    """is_complete=False reads existing bytes before merging."""
    transform = ChunkTransform(codecs=(BytesCodec(),))
    spec = _spec(shape=(4,), fill_value=0)
    # Existing chunk: all zeros
    existing_arr = _ndbuffer_from_array(np.zeros(4, dtype="uint8"))
    encoded = transform.encode_chunk(existing_arr, spec)
    assert encoded is not None
    bs = MagicMock()
    bs.get_sync.return_value = encoded

    value = _ndbuffer_from_array(np.array([7, 7], dtype="uint8"))
    write_chunk(
        bs,
        transform,
        spec,
        value,
        chunk_selection=(slice(0, 2),),
        out_selection=(slice(0, 2),),
        is_complete=False,
    )
    bs.get_sync.assert_called_once()
    bs.set_sync.assert_called_once()


def test_write_chunk_deletes_when_merged_equals_fill() -> None:
    """write_empty_chunks=False + merged chunk all-fill → delete_sync."""
    transform = ChunkTransform(codecs=(BytesCodec(),))
    spec = _spec(shape=(4,), fill_value=0, write_empty_chunks=False)
    bs = MagicMock()
    value = _ndbuffer_from_array(np.zeros(4, dtype="uint8"))
    write_chunk(
        bs,
        transform,
        spec,
        value,
        chunk_selection=(slice(None),),
        out_selection=(slice(None),),
        is_complete=True,
    )
    bs.delete_sync.assert_called_once()
    bs.set_sync.assert_not_called()
