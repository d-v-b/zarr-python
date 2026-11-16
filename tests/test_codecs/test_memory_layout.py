"""
A chunk is a logical array: its values and shape are what a codec encodes, and the
way those values are laid out in memory is not part of it. These tests check that
every array-to-array and array-to-bytes codec gives the same result for the same
chunk in different memory layouts.

Every codec class that zarr registers must appear in `CASES`, so a new codec is
covered from the day it is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numcodecs
import numpy as np
import pytest

from zarr import create_array
from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec
from zarr.codecs import numcodecs as _numcodecs
from zarr.codecs._v2 import V2Codec
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.cast_value import CastValue
from zarr.codecs.scale_offset import ScaleOffset
from zarr.codecs.sharding import ShardingCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.vlen_utf8 import VLenBytesCodec, VLenUTF8Codec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import Buffer, NDBuffer, default_buffer_prototype
from zarr.core.dtype import VariableLengthBytes, get_data_type_from_native_dtype
from zarr.errors import UnknownCodecError
from zarr.registry import _codec_registries

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from zarr.core.dtype import ZDType

try:
    import cast_value_rs  # noqa: F401

    _HAS_CAST_VALUE_RS = True
except ModuleNotFoundError:
    _HAS_CAST_VALUE_RS = False

# No two axes have the same length, so a chunk that is read in the wrong axis order
# cannot pass by accident.
SHAPE = (3, 4, 5)


def _c_contiguous(arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.ascontiguousarray(arr)


def _f_contiguous(arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.asfortranarray(arr)


def _strided(arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Every other element of a larger array: neither C- nor F-contiguous."""
    step = (slice(None, None, 2),) * arr.ndim
    out = np.empty(tuple(2 * s for s in arr.shape), dtype=arr.dtype)[step]
    out[...] = arr
    return out


def _reversed(arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """A view with a negative stride along every axis."""
    flip = (slice(None, None, -1),) * arr.ndim
    return np.ascontiguousarray(arr[flip])[flip]


LAYOUTS: dict[str, Callable[[npt.NDArray[Any]], npt.NDArray[Any]]] = {
    "c": _c_contiguous,
    "f": _f_contiguous,
    "strided": _strided,
    "reversed": _reversed,
}


@dataclass(frozen=True)
class Case:
    """A codec, the data type of the chunks it is given, and the memory order of the array."""

    id: str
    codec: ArrayArrayCodec | ArrayBytesCodec
    dtype: str
    order: Literal["C", "F"] = "C"
    available: bool = True


AA_CASES: tuple[Case, ...] = (
    Case("transpose", TransposeCodec(order=(2, 0, 1)), "int32"),
    Case("scale_offset", ScaleOffset(offset=1, scale=2), "float64"),
    Case(
        "cast_value",
        CastValue(data_type="int16", rounding="nearest-even"),
        "float64",
        available=_HAS_CAST_VALUE_RS,
    ),
    Case("numcodecs.delta", _numcodecs.Delta(dtype="int64"), "int64"),
    Case(
        "numcodecs.fixedscaleoffset",
        _numcodecs.FixedScaleOffset(offset=0, scale=1, dtype="int64"),
        "int64",
    ),
    Case("numcodecs.quantize", _numcodecs.Quantize(digits=3, dtype="float32"), "float32"),
    Case("numcodecs.bitround", _numcodecs.BitRound(keepbits=10), "float32"),
    Case(
        "numcodecs.astype",
        _numcodecs.AsType(encode_dtype="float32", decode_dtype="float64"),
        "float64",
    ),
    Case("numcodecs.packbits", _numcodecs.PackBits(), "bool"),
)

AB_CASES: tuple[Case, ...] = (
    Case("bytes", BytesCodec(), "int32"),
    Case("vlen-utf8", VLenUTF8Codec(), "str"),
    Case("vlen-bytes", VLenBytesCodec(), "bytes"),
    Case("sharding", ShardingCodec(chunk_shape=(3, 2, 5)), "int32"),
    Case("numcodecs.zfpy", _numcodecs.ZFPY(), "float32"),
    Case("numcodecs.pcodec", _numcodecs.PCodec(), "float32"),
    Case("v2-c", V2Codec(filters=(numcodecs.Delta(dtype="<i8"),), compressor=None), "int64"),
    Case(
        "v2-f",
        V2Codec(filters=(numcodecs.Delta(dtype="<i8"),), compressor=None),
        "int64",
        order="F",
    ),
)

CASES = AA_CASES + AB_CASES


def _chunk(dtype: str, shape: tuple[int, ...] = SHAPE) -> npt.NDArray[Any]:
    """A C-contiguous chunk of distinct, unordered values, so no reordering of them is hidden."""
    rng = np.random.default_rng(0)
    size = int(np.prod(shape))
    values: npt.NDArray[Any]
    if dtype == "bool":
        values = rng.integers(0, 2, size).astype(bool)
    elif dtype == "str":
        values = np.array([f"s{i}" for i in rng.permutation(size)], dtype=np.dtypes.StringDType())
    elif dtype == "bytes":
        values = np.array([f"b{i}".encode() for i in rng.permutation(size)], dtype=object)
    else:
        values = rng.permutation(size).astype(dtype)
    return values.reshape(shape)


def _spec(case: Case, shape: tuple[int, ...] = SHAPE) -> ArraySpec:
    zdtype: ZDType[Any, Any]
    if case.dtype == "bytes":
        # numpy's object dtype does not say which zarr data type it holds
        zdtype = VariableLengthBytes()
    else:
        zdtype = get_data_type_from_native_dtype(_chunk(case.dtype, (1,)).dtype)
    return ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=zdtype.default_scalar(),
        config=ArrayConfig(order=case.order, write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )


def _require(case: Case) -> None:
    """Skip a case whose codec needs an optional dependency that is not installed."""
    if not case.available:
        pytest.skip(f"the codec for {case.id} is not installed")
    if isinstance(case.codec, _numcodecs._NumcodecsCodec):
        try:
            case.codec._codec  # noqa: B018
        except (UnknownCodecError, ImportError) as e:  # pragma: no cover
            pytest.skip(f"the codec for {case.id} is not installed: {e}")


def _nd_buffer(arr: npt.NDArray[Any]) -> NDBuffer:
    return default_buffer_prototype().nd_buffer.from_ndarray_like(arr)


async def _encode(case: Case, arr: npt.NDArray[Any]) -> NDBuffer | Buffer:
    spec = _spec(case)
    codec = case.codec.evolve_from_array_spec(spec)
    (encoded,) = await codec.encode([(_nd_buffer(arr), spec)])
    assert encoded is not None
    return encoded


def _assert_same_chunk(actual: NDBuffer | Buffer, expected: NDBuffer | Buffer) -> None:
    """The two encoded or decoded chunks hold the same values in the same shape."""
    if isinstance(expected, Buffer):
        assert isinstance(actual, Buffer)
        assert actual.to_bytes() == expected.to_bytes()
    else:
        assert isinstance(actual, NDBuffer)
        actual_arr = actual.as_numpy_array()
        expected_arr = expected.as_numpy_array()
        assert actual_arr.shape == expected_arr.shape
        assert actual_arr.dtype == expected_arr.dtype
        np.testing.assert_array_equal(actual_arr, expected_arr)


@pytest.mark.parametrize("dtype", ["int32", "bool", "str", "bytes"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_layouts(layout: str, dtype: str) -> None:
    """Each layout holds the values of the chunk it was made from, in the memory layout it names."""
    chunk = _chunk(dtype)
    laid_out = LAYOUTS[layout](chunk)

    np.testing.assert_array_equal(laid_out, chunk)
    observed = {
        "c": laid_out.flags.c_contiguous and not laid_out.flags.f_contiguous,
        "f": laid_out.flags.f_contiguous and not laid_out.flags.c_contiguous,
        "strided": not laid_out.flags.c_contiguous and not laid_out.flags.f_contiguous,
        "reversed": all(s < 0 for s in laid_out.strides),
    }
    assert observed[layout]


def test_every_registered_codec_has_a_case() -> None:
    """Each array-to-array and array-to-bytes codec class that zarr registers is in `CASES`."""
    registered = {
        cls
        for registry in _codec_registries.values()
        for cls in registry.values()
        if cls.__module__.startswith("zarr.") and issubclass(cls, ArrayArrayCodec | ArrayBytesCodec)
    }
    covered = {type(case.codec) for case in CASES}
    assert registered - covered == set()


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
async def test_encode_ignores_memory_layout(case: Case, layout: str) -> None:
    """Encoding a chunk gives the same result whatever the memory layout of the chunk."""
    _require(case)
    chunk = _chunk(case.dtype)

    expected = await _encode(case, chunk)
    actual = await _encode(case, LAYOUTS[layout](chunk))

    _assert_same_chunk(actual, expected)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("case", AA_CASES, ids=lambda case: case.id)
async def test_decode_ignores_memory_layout(case: Case, layout: str) -> None:
    """Decoding an encoded chunk gives the same result whatever its memory layout."""
    _require(case)
    spec = _spec(case)
    codec = case.codec.evolve_from_array_spec(spec)
    assert isinstance(codec, ArrayArrayCodec)
    encoded = (await _encode(case, _chunk(case.dtype))).as_numpy_array()

    (expected,) = await codec.decode([(_nd_buffer(encoded), spec)])
    (actual,) = await codec.decode([(_nd_buffer(LAYOUTS[layout](encoded)), spec)])

    assert expected is not None
    assert actual is not None
    _assert_same_chunk(actual, expected)


def _read_back(case: Case, filters: list[ArrayArrayCodec], data: npt.NDArray[Any]) -> Any:
    arr = create_array(
        {},
        shape=data.shape,
        chunks=data.shape,
        dtype=data.dtype,
        fill_value=_spec(case).fill_value,
        filters=filters,
        compressors=None,
    )
    arr[...] = data
    return arr[...]


def _transpose_params() -> list[Any]:
    """Each array-to-array case with a `TransposeCodec` before it and after it."""
    params = []
    for case in AA_CASES:
        for transpose_first in (True, False):
            marks = []
            if isinstance(case.codec, _numcodecs.PackBits) and not transpose_first:
                # `ArrayV3Metadata` validates each codec against the shape of the array
                # instead of the shape the codec before it produces, so it rejects a
                # 1-dimensional transpose after `PackBits`.
                marks.append(
                    pytest.mark.xfail(
                        raises=ValueError,
                        strict=True,
                        reason="codecs are validated against the array shape, not the encoded shape",
                    )
                )
            position = "transpose-first" if transpose_first else "transpose-last"
            params.append(
                pytest.param(case, transpose_first, id=f"{case.id}-{position}", marks=marks)
            )
    return params


@pytest.mark.filterwarnings("ignore:Numcodecs codecs are not in the Zarr version 3 specification")
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(("case", "transpose_first"), _transpose_params())
def test_transpose_does_not_change_values(case: Case, transpose_first: bool, layout: str) -> None:
    """
    An array reads back the same values with and without a `TransposeCodec` next to a filter.

    `TransposeCodec` hands the next codec a view that is not C-contiguous, on encode when
    it comes first and on decode when it comes last.
    """
    _require(case)
    assert isinstance(case.codec, ArrayArrayCodec)
    data = LAYOUTS[layout](_chunk(case.dtype))
    if transpose_first:
        filters = [TransposeCodec(order=(2, 1, 0)), case.codec]
    else:
        spec = _spec(case)
        encoded_ndim = len(case.codec.evolve_from_array_spec(spec).resolve_metadata(spec).shape)
        filters = [case.codec, TransposeCodec(order=tuple(reversed(range(encoded_ndim))))]

    expected = _read_back(case, [case.codec], data)
    actual = _read_back(case, filters, data)

    np.testing.assert_array_equal(actual, expected)
