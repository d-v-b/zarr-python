from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import NDBuffer, default_buffer_prototype
from zarr.core.dtype import get_data_type_from_native_dtype

if TYPE_CHECKING:
    from zarr.abc.codec import Codec


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


class TestCodecChain:
    def test_from_codecs_bytes_only(self) -> None:
        from zarr.core.codec_pipeline import CodecChain

        chain = CodecChain.from_codecs([BytesCodec()])
        assert chain.array_array_codecs == ()
        assert isinstance(chain.array_bytes_codec, BytesCodec)
        assert chain.bytes_bytes_codecs == ()
        assert chain._all_sync is True

    def test_from_codecs_with_compression(self) -> None:
        from zarr.core.codec_pipeline import CodecChain

        chain = CodecChain.from_codecs([BytesCodec(), GzipCodec()])
        assert isinstance(chain.array_bytes_codec, BytesCodec)
        assert len(chain.bytes_bytes_codecs) == 1
        assert isinstance(chain.bytes_bytes_codecs[0], GzipCodec)
        assert chain._all_sync is True

    def test_from_codecs_with_transpose(self) -> None:
        from zarr.core.codec_pipeline import CodecChain

        chain = CodecChain.from_codecs([TransposeCodec(order=(1, 0)), BytesCodec()])
        assert len(chain.array_array_codecs) == 1
        assert isinstance(chain.array_array_codecs[0], TransposeCodec)
        assert isinstance(chain.array_bytes_codec, BytesCodec)
        assert chain._all_sync is True

    def test_from_codecs_full_chain(self) -> None:
        from zarr.core.codec_pipeline import CodecChain

        chain = CodecChain.from_codecs([TransposeCodec(order=(1, 0)), BytesCodec(), ZstdCodec()])
        assert len(chain.array_array_codecs) == 1
        assert isinstance(chain.array_bytes_codec, BytesCodec)
        assert len(chain.bytes_bytes_codecs) == 1
        assert chain._all_sync is True

    def test_iter(self) -> None:
        from zarr.core.codec_pipeline import CodecChain

        codecs: list[Codec] = [TransposeCodec(order=(1, 0)), BytesCodec(), GzipCodec()]
        chain = CodecChain.from_codecs(codecs)
        assert list(chain) == codecs

    def test_frozen(self) -> None:
        from zarr.core.codec_pipeline import CodecChain

        chain = CodecChain.from_codecs([BytesCodec()])
        with pytest.raises(AttributeError):
            chain.array_bytes_codec = BytesCodec()  # type: ignore[misc]

    def test_encode_decode_roundtrip_bytes_only(self) -> None:
        from zarr.core.codec_pipeline import CodecChain

        chain = CodecChain.from_codecs([BytesCodec()])
        arr = np.arange(100, dtype="float64")
        spec = _make_array_spec(arr.shape, arr.dtype)
        chain_evolved = CodecChain.from_codecs([c.evolve_from_array_spec(spec) for c in chain])
        nd_buf = _make_nd_buffer(arr)

        encoded = chain_evolved.encode_chunk(nd_buf, spec)
        assert encoded is not None
        decoded = chain_evolved.decode_chunk(encoded, spec)
        assert decoded is not None
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())

    def test_encode_decode_roundtrip_with_compression(self) -> None:
        from zarr.core.codec_pipeline import CodecChain

        chain = CodecChain.from_codecs([BytesCodec(), GzipCodec(level=1)])
        arr = np.arange(100, dtype="float64")
        spec = _make_array_spec(arr.shape, arr.dtype)
        chain_evolved = CodecChain.from_codecs([c.evolve_from_array_spec(spec) for c in chain])
        nd_buf = _make_nd_buffer(arr)

        encoded = chain_evolved.encode_chunk(nd_buf, spec)
        assert encoded is not None
        decoded = chain_evolved.decode_chunk(encoded, spec)
        assert decoded is not None
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())

    def test_encode_decode_roundtrip_with_transpose(self) -> None:
        from zarr.core.codec_pipeline import CodecChain

        chain = CodecChain.from_codecs(
            [TransposeCodec(order=(1, 0)), BytesCodec(), ZstdCodec(level=1)]
        )
        arr = np.arange(12, dtype="float64").reshape(3, 4)
        spec = _make_array_spec(arr.shape, arr.dtype)
        chain_evolved = CodecChain.from_codecs([c.evolve_from_array_spec(spec) for c in chain])
        nd_buf = _make_nd_buffer(arr)

        encoded = chain_evolved.encode_chunk(nd_buf, spec)
        assert encoded is not None
        decoded = chain_evolved.decode_chunk(encoded, spec)
        assert decoded is not None
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())

    def test_resolve_metadata_chain(self) -> None:
        from zarr.core.codec_pipeline import CodecChain

        chain = CodecChain.from_codecs([TransposeCodec(order=(1, 0)), BytesCodec(), GzipCodec()])
        arr = np.zeros((3, 4), dtype="float64")
        spec = _make_array_spec(arr.shape, arr.dtype)
        chain_evolved = CodecChain.from_codecs([c.evolve_from_array_spec(spec) for c in chain])

        aa_chain, ab_pair, bb_chain = chain_evolved.resolve_metadata_chain(spec)
        assert len(aa_chain) == 1
        assert aa_chain[0][1].shape == (3, 4)  # spec before transpose
        _ab_codec, ab_spec = ab_pair
        assert ab_spec.shape == (4, 3)  # spec after transpose
        assert len(bb_chain) == 1

    def test_resolve_metadata(self) -> None:
        from zarr.core.codec_pipeline import CodecChain

        chain = CodecChain.from_codecs([TransposeCodec(order=(1, 0)), BytesCodec()])
        spec = _make_array_spec((3, 4), np.dtype("float64"))
        chain_evolved = CodecChain.from_codecs([c.evolve_from_array_spec(spec) for c in chain])
        resolved = chain_evolved.resolve_metadata(spec)
        # After transpose (1,0) + bytes, shape should reflect the transpose
        assert resolved.shape == (4, 3)
