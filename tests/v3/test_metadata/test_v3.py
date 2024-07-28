from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal, cast

from zarr.codecs.bytes import BytesCodec
from zarr.common import JSON, parse_dtype, parse_shapelike
from zarr.metadata.common import parse_attributes

if TYPE_CHECKING:
    from typing import Any

from collections.abc import Sequence

import numpy as np
import pytest

from zarr.metadata.v3 import (
    ArrayV3Metadata,
    ChunkKeyEncoding,
    RegularChunkGridDict,
    parse_chunk_grid,
    parse_chunk_key_encoding,
    parse_codecs,
    parse_dimension_names,
    parse_fill_value,
    parse_zarr_format,
)

bool_dtypes = ("bool",)

int_dtypes = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)

float_dtypes = (
    "float16",
    "float32",
    "float64",
)

complex_dtypes = ("complex64", "complex128")

dtypes = (*bool_dtypes, *int_dtypes, *float_dtypes, *complex_dtypes)


@pytest.mark.parametrize("data", [None, 1, 2, 4, 5, "3"])
def test_parse_zarr_format_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match=f"Invalid value. Expected 3. Got {data}"):
        parse_zarr_format(data)


def test_parse_zarr_format_valid() -> None:
    assert parse_zarr_format(3) == 3


@pytest.mark.parametrize("data", [(), [1, 2, "a"], {"foo": 10}])
def parse_dimension_names_invalid(data: Any) -> None:
    with pytest.raises(TypeError, match="Expected either None or iterable of str,"):
        parse_dimension_names(data)


@pytest.mark.parametrize("data", [None, ("a", "b", "c"), ["a", "a", "a"]])
def parse_dimension_names_valid(data: Sequence[str] | None) -> None:
    assert parse_dimension_names(data) == data


@pytest.mark.parametrize("dtype_str", dtypes)
def test_parse_auto_fill_value(dtype_str: str) -> None:
    """
    Test that parse_fill_value(None, dtype) results in the 0 value for the given dtype.
    """
    dtype = np.dtype(dtype_str)
    fill_value = None
    assert parse_fill_value(fill_value, dtype) == dtype.type(0)


@pytest.mark.parametrize("fill_value", [0, 1.11, False, True])
@pytest.mark.parametrize("dtype_str", dtypes)
def test_parse_fill_value_valid(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) casts fill_value to the given dtype.
    """
    dtype = np.dtype(dtype_str)
    assert parse_fill_value(fill_value, dtype) == dtype.type(fill_value)


@pytest.mark.parametrize("fill_value", ["not a valid value"])
@pytest.mark.parametrize("dtype_str", [*int_dtypes, *float_dtypes, *complex_dtypes])
def test_parse_fill_value_invalid_value(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) raises ValueError for invalid values.
    This test excludes bool because the bool constructor takes anything.
    """
    dtype = np.dtype(dtype_str)
    with pytest.raises(ValueError):
        parse_fill_value(fill_value, dtype)


@pytest.mark.parametrize("fill_value", [[1.0, 0.0], [0, 1], complex(1, 1), np.complex64(0)])
@pytest.mark.parametrize("dtype_str", [*complex_dtypes])
def test_parse_fill_value_complex(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) correctly handles complex values represented
    as length-2 sequences
    """
    dtype = np.dtype(dtype_str)
    if isinstance(fill_value, list):
        expected = dtype.type(complex(*fill_value))
    else:
        expected = dtype.type(fill_value)
    assert expected == parse_fill_value(fill_value, dtype)


@pytest.mark.parametrize("fill_value", [[1.0, 0.0, 3.0], [0, 1, 3], [1]])
@pytest.mark.parametrize("dtype_str", [*complex_dtypes])
def test_parse_fill_value_complex_invalid(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) correctly rejects sequences with length not
    equal to 2
    """
    dtype = np.dtype(dtype_str)
    match = (
        f"Got an invalid fill value for complex data type {dtype}."
        f"Expected a sequence with 2 elements, but {fill_value} has "
        f"length {len(fill_value)}."
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        parse_fill_value(fill_value=fill_value, dtype=dtype)


@pytest.mark.parametrize("fill_value", [{"foo": 10}])
@pytest.mark.parametrize("dtype_str", [*int_dtypes, *float_dtypes, *complex_dtypes])
def test_parse_fill_value_invalid_type(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) raises TypeError for invalid non-sequential types.
    This test excludes bool because the bool constructor takes anything.
    """
    dtype = np.dtype(dtype_str)
    match = "must be"
    with pytest.raises(TypeError, match=match):
        parse_fill_value(fill_value, dtype)


@pytest.mark.parametrize(
    "fill_value",
    [
        [
            1,
        ],
        (1, 23, 4),
    ],
)
@pytest.mark.parametrize("dtype_str", [*int_dtypes, *float_dtypes])
def test_parse_fill_value_invalid_type_sequence(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) raises TypeError for invalid sequential types.
    This test excludes bool because the bool constructor takes anything, and complex because
    complex values can be created from length-2 sequences.
    """
    dtype = np.dtype(dtype_str)
    match = f"Cannot parse non-string sequence {fill_value} as a scalar with type {dtype}"
    with pytest.raises(TypeError, match=re.escape(match)):
        parse_fill_value(fill_value, dtype)


def test_parse_attributes() -> None:
    """
    Test that parse_attributes passes dicts through unchanged.
    """
    data = {"foo": "bar"}
    assert parse_attributes(data) == data


def test_parse_null_attributes() -> None:
    """
    Test that parse_attributes normalizes `None` to an empty dictionary.
    """
    assert parse_attributes(None) == {}


@pytest.mark.parametrize(
    "shape, chunk_shape",
    [
        ((0,), (0,)),
        ((1,), (1,)),
        ((1, 2, 3), (1, 1, 1)),
    ],
)
@pytest.mark.parametrize("data_type", dtypes)
@pytest.mark.parametrize("chunk_grid", ["regular"])
@pytest.mark.parametrize("attributes", [None, {"foo": "bar"}])
@pytest.mark.parametrize("codecs", [[BytesCodec()]])
@pytest.mark.parametrize("fill_value", [0, 1])
@pytest.mark.parametrize("chunk_key_encoding", ["v2", "default"])
@pytest.mark.parametrize("dimension_separator", [".", "/", None])
@pytest.mark.parametrize("dimension_names", ["nones", "strings", "missing"])
@pytest.mark.parametrize("from_dict", [True, False])
def test_array_metadata_init(
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    data_type: str,
    chunk_grid: str,
    codecs: list[dict[str, Any]],
    fill_value: Any,
    chunk_key_encoding: Literal["v2", "default"],
    dimension_separator: Literal[".", "/"] | None,
    dimension_names: Literal["nones", "strings", "missing"],
    attributes: None | dict[str, Any],
    from_dict: bool,
) -> None:
    if chunk_grid == "regular":
        cgrid: RegularChunkGridDict = {
            "name": "regular",
            "configuration": {"chunk_shape": chunk_shape},
        }
        # fix type inference here
        _cgrid = cast(dict[str, JSON], cgrid)
    else:
        raise ValueError(f"Invalid chunk grid: {chunk_grid}")

    dnames: tuple[str | None, ...] | None

    if dimension_names == "strings":
        dnames = tuple(map(str, range(len(shape))))
    elif dimension_names == "missing":
        dnames = (None,) * len(shape)
    elif dimension_names == "nones":
        dnames = None

    cke: dict[str, Any]
    cke_name_dict = {"name": chunk_key_encoding}
    if dimension_separator is not None:
        cke = cke_name_dict | {"configuration": {"separator": dimension_separator}}
    else:
        cke = cke_name_dict

    # fix type inference here
    cke = cast(dict[str, JSON], cke)

    if from_dict:
        metadata_dict = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": shape,
            "chunk_grid": cgrid,
            "data_type": data_type,
            "chunk_key_encoding": cke,
            "codecs": codecs,
            "fill_value": fill_value,
        }
        if attributes is not None:
            metadata_dict["attributes"] = attributes
        if dnames is not None:
            metadata_dict["dimension_names"] = dnames

        metadata = ArrayV3Metadata.from_dict(metadata_dict)
    else:
        metadata = ArrayV3Metadata(
            shape=shape,
            chunk_grid=_cgrid,
            codecs=codecs,
            data_type=data_type,
            fill_value=fill_value,
            attributes=attributes,
            chunk_key_encoding=cke,
            dimension_names=dnames,
        )

    assert metadata.shape == parse_shapelike(shape)
    assert metadata.chunk_grid == parse_chunk_grid(cgrid)
    assert metadata.codecs == parse_codecs(codecs)
    assert metadata.data_type == parse_dtype(data_type)
    assert metadata.fill_value == parse_fill_value(fill_value, metadata.data_type)
    assert metadata.attributes == parse_attributes(attributes)
    assert metadata.chunk_key_encoding == parse_chunk_key_encoding(cke)
    assert metadata.dimension_names == parse_dimension_names(dnames)


@pytest.mark.parametrize("ndim", (0, 1, 5))
def test_shape_dimension_names(ndim: int) -> None:
    shape = tuple(range(1, ndim + 1))
    names_valid = tuple(map(str, shape))
    names_invalid = names_valid + ("extra",)
    cgrid = {"name": "regular", "configuration": {"chunk_shape": (1,) * len(shape)}}
    match = "The length of the `dimension_names` attribute must match the"

    with pytest.raises(ValueError, match=match):
        _ = ArrayV3Metadata(
            shape=shape,
            data_type="uint8",
            chunk_grid=cgrid,
            chunk_key_encoding={"name": "default"},
            fill_value=0,
            codecs=[],
            dimension_names=names_invalid,
            attributes=None,
        )


@pytest.mark.parametrize("chunk_grid", ["regular"])
@pytest.mark.parametrize("attributes", [None, {"foo": "bar"}])
@pytest.mark.parametrize("codecs", [[BytesCodec()]])
@pytest.mark.parametrize("fill_value", [0, 1])
@pytest.mark.parametrize("chunk_key_encoding", ["v2", "default"])
@pytest.mark.parametrize("dimension_separator", [".", "/", None])
@pytest.mark.parametrize("dimension_names", ["nones", "strings", "missing"])
def test_metadata_to_dict(
    chunk_grid: str,
    codecs: list[dict[str, Any]],
    fill_value: Any,
    chunk_key_encoding: Literal["v2", "default"],
    dimension_separator: Literal[".", "/"] | None,
    dimension_names: Literal["nones", "strings", "missing"],
    attributes: None | dict[str, Any],
) -> None:
    shape = (1, 2, 3)
    data_type = "uint8"
    if chunk_grid == "regular":
        cgrid = {"name": "regular", "configuration": {"chunk_shape": (1, 1, 1)}}

    cke: dict[str, Any]
    cke_name_dict = {"name": chunk_key_encoding}
    if dimension_separator is not None:
        cke = cke_name_dict | {"configuration": {"separator": dimension_separator}}
    else:
        cke = cke_name_dict
    dnames: tuple[str | None, ...] | None

    if dimension_names == "strings":
        dnames = tuple(map(str, range(len(shape))))
    elif dimension_names == "missing":
        dnames = (None,) * len(shape)
    elif dimension_names == "nones":
        dnames = None

    metadata_dict = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "chunk_grid": cgrid,
        "data_type": data_type,
        "chunk_key_encoding": cke,
        "codecs": tuple(c.to_dict() for c in codecs),
        "fill_value": fill_value,
    }

    if attributes is not None:
        metadata_dict["attributes"] = attributes
    if dnames is not None:
        metadata_dict["dimension_names"] = dnames

    metadata = ArrayV3Metadata.from_dict(metadata_dict)
    observed = metadata.to_dict()
    expected = metadata_dict
    if attributes is None:
        assert observed["attributes"] == {}
        observed.pop("attributes")
    if dimension_separator is None:
        assert (
            observed["chunk_key_encoding"]
            == ChunkKeyEncoding(name=chunk_key_encoding, configuration={"separator": "/"}).to_dict()
        )
        observed.pop("chunk_key_encoding")
        expected.pop("chunk_key_encoding")
    assert observed == expected
