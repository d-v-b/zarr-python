from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import NDBuffer, default_buffer_prototype
from zarr.core.codec_pipeline import (
    BatchedCodecPipeline,
    Chunk,
    _choose_workers,
    _merge_chunk_array,
    fill_value_or_default,
)
from zarr.core.dtype import get_data_type_from_native_dtype


def _make_array_spec(shape: tuple[int, ...], dtype: np.dtype[np.generic]) -> ArraySpec:
    zdtype = get_data_type_from_native_dtype(dtype)
    return ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )


def _make_nd_buffer(arr: np.ndarray[Any, np.dtype[Any]]) -> NDBuffer:
    return default_buffer_prototype().nd_buffer.from_numpy_array(arr)


_DEFAULT_DTYPE = np.dtype("float64")


def _make_chain(
    codecs: list[Any], shape: tuple[int, ...] = (100,), dtype: np.dtype[Any] = _DEFAULT_DTYPE
) -> Chunk:
    spec = _make_array_spec(shape, dtype)
    return Chunk(codecs=tuple(codecs), chunk_spec=spec)


class TestChunk:
    def test_all_sync(self) -> None:
        chain = _make_chain([BytesCodec()])
        assert chain._all_sync is True

    def test_all_sync_with_compression(self) -> None:
        chain = _make_chain([BytesCodec(), GzipCodec()])
        assert chain._all_sync is True

    def test_all_sync_full_chain(self) -> None:
        chain = _make_chain([TransposeCodec(order=(1, 0)), BytesCodec(), ZstdCodec()], shape=(3, 4))
        assert chain._all_sync is True

    def test_encode_decode_roundtrip_bytes_only(self) -> None:
        arr = np.arange(100, dtype="float64")
        chain = _make_chain([BytesCodec()], shape=arr.shape, dtype=arr.dtype)
        nd_buf = _make_nd_buffer(arr)

        encoded = chain.encode_chunk(nd_buf)
        assert encoded is not None
        decoded = chain.decode_chunk(encoded)
        assert decoded is not None
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())

    def test_encode_decode_roundtrip_with_compression(self) -> None:
        arr = np.arange(100, dtype="float64")
        chain = _make_chain([BytesCodec(), GzipCodec(level=1)], shape=arr.shape, dtype=arr.dtype)
        nd_buf = _make_nd_buffer(arr)

        encoded = chain.encode_chunk(nd_buf)
        assert encoded is not None
        decoded = chain.decode_chunk(encoded)
        assert decoded is not None
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())

    def test_encode_decode_roundtrip_with_transpose(self) -> None:
        arr = np.arange(12, dtype="float64").reshape(3, 4)
        chain = _make_chain(
            [TransposeCodec(order=(1, 0)), BytesCodec(), ZstdCodec(level=1)],
            shape=arr.shape,
            dtype=arr.dtype,
        )
        nd_buf = _make_nd_buffer(arr)

        encoded = chain.encode_chunk(nd_buf)
        assert encoded is not None
        decoded = chain.decode_chunk(encoded)
        assert decoded is not None
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())

    def test_decode_none_returns_none(self) -> None:
        chain = _make_chain([BytesCodec()], shape=(10,))
        assert chain.decode_chunk(None) is None

    def test_encode_none_returns_none(self) -> None:
        chain = _make_chain([BytesCodec()], shape=(10,))
        assert chain.encode_chunk(None) is None

    def test_pre_resolved_metadata(self) -> None:
        chain = _make_chain([TransposeCodec(order=(1, 0)), BytesCodec(), ZstdCodec()], shape=(3, 4))
        assert len(chain.array_array_codecs) == 1
        assert isinstance(chain.array_array_codecs[0], TransposeCodec)
        assert isinstance(chain.array_bytes_codec, BytesCodec)
        assert len(chain.bytes_bytes_codecs) == 1
        assert isinstance(chain.bytes_bytes_codecs[0], ZstdCodec)


class TestBatchedCodecPipeline:
    def test_from_codecs(self) -> None:
        pipeline = BatchedCodecPipeline.from_codecs([BytesCodec()])
        assert isinstance(pipeline.array_bytes_codec, BytesCodec)
        assert pipeline.array_array_codecs == ()
        assert pipeline.bytes_bytes_codecs == ()

    def test_old_style_init(self) -> None:
        pipeline = BatchedCodecPipeline(
            array_array_codecs=(),
            array_bytes_codec=BytesCodec(),
            bytes_bytes_codecs=(GzipCodec(),),
        )
        assert isinstance(pipeline.array_bytes_codec, BytesCodec)
        assert len(pipeline.bytes_bytes_codecs) == 1

    def test_batch_size_deprecated(self) -> None:
        with pytest.warns(FutureWarning, match="batch_size"):
            BatchedCodecPipeline(
                array_bytes_codec=BytesCodec(),
                batch_size=10,
            )

    def test_supports_sync_io(self) -> None:
        pipeline = BatchedCodecPipeline.from_codecs([BytesCodec()])
        assert pipeline.supports_sync_io is True

    def test_iter(self) -> None:
        pipeline = BatchedCodecPipeline.from_codecs(
            [TransposeCodec(order=(1, 0)), BytesCodec(), GzipCodec()]
        )
        codecs = list(pipeline)
        assert len(codecs) == 3


class TestChooseWorkers:
    def test_returns_zero_for_single_chunk(self) -> None:
        assert _choose_workers(1, 1_000_000, [BytesCodec(), GzipCodec()]) == 0

    def test_returns_zero_for_no_compression(self) -> None:
        assert _choose_workers(10, 1_000_000, [BytesCodec()]) == 0

    def test_returns_zero_for_small_chunks(self) -> None:
        assert _choose_workers(10, 100, [BytesCodec(), GzipCodec()]) == 0


class TestMergeChunkArray:
    def test_complete_chunk_passthrough(self) -> None:
        spec = _make_array_spec((4,), np.dtype("float64"))
        value = _make_nd_buffer(np.ones(4, dtype="float64"))
        result = _merge_chunk_array(None, value, slice(None), spec, slice(None), True, ())
        np.testing.assert_array_equal(result.as_numpy_array(), np.ones(4))

    def test_partial_chunk_creates_fill(self) -> None:
        spec = _make_array_spec((4,), np.dtype("float64"))
        value = _make_nd_buffer(np.ones(4, dtype="float64"))
        result = _merge_chunk_array(None, value, slice(0, 2), spec, slice(0, 2), False, ())
        arr = result.as_numpy_array()
        np.testing.assert_array_equal(arr[:2], np.ones(2))
        np.testing.assert_array_equal(arr[2:], np.zeros(2))


class TestFillValueOrDefault:
    def test_with_fill_value(self) -> None:
        spec = _make_array_spec((10,), np.dtype("float64"))
        assert fill_value_or_default(spec) == 0

    def test_with_none_fill_value(self) -> None:
        zdtype = get_data_type_from_native_dtype(np.dtype("float64"))
        spec = ArraySpec(
            shape=(10,),
            dtype=zdtype,
            fill_value=None,
            config=ArrayConfig(order="C", write_empty_chunks=True),
            prototype=default_buffer_prototype(),
        )
        result = fill_value_or_default(spec)
        assert result is not None


class TestReadWriteSync:
    def test_read_sync_empty_batch(self) -> None:
        """read_sync with empty batch is a no-op."""
        pipeline = BatchedCodecPipeline.from_codecs([BytesCodec()])
        out = default_buffer_prototype().nd_buffer.create(
            shape=(10,), dtype=np.dtype("float64"), order="C", fill_value=0
        )
        pipeline.read_sync([], out)  # should not raise

    def test_write_sync_empty_batch(self) -> None:
        """write_sync with empty batch is a no-op."""
        pipeline = BatchedCodecPipeline.from_codecs([BytesCodec()])
        value = _make_nd_buffer(np.arange(10, dtype="float64"))
        pipeline.write_sync([], value)  # should not raise
