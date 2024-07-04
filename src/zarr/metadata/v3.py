from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, overload

import numcodecs

if TYPE_CHECKING:
    from typing import Any, Literal

    import numpy.typing as npt
    from typing_extensions import Self

    _bool = bool
    from zarr.abc.codec import Codec

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from zarr.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.chunk_key_encodings import ChunkKeyEncoding
from zarr.codecs.registry import get_codec_class
from zarr.common import JSON, ZARR_JSON, ChunkCoords, parse_named_configuration
from zarr.config import config
from zarr.metadata.common import (
    ArrayMetadataBase,
    ArraySpec,
    ChunkGrid,
    RegularChunkGrid,
    parse_attributes,
    parse_dtype,
    parse_shapelike,
)

BOOL = np.bool_
BOOL_DTYPE = np.dtypes.BoolDType

INTEGER_DTYPE = (
    np.dtypes.Int8DType
    | np.dtypes.Int16DType
    | np.dtypes.Int32DType
    | np.dtypes.Int64DType
    | np.dtypes.UByteDType
)

INTEGER = np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64
FLOAT_DTYPE = np.dtypes.Float16DType | np.dtypes.Float32DType | np.dtypes.Float64DType
FLOAT = np.float16 | np.float32 | np.float64
COMPLEX_DTYPE = np.dtypes.Complex64DType | np.dtypes.Complex128DType
COMPLEX = np.complex64 | np.complex128


def parse_dimension_names(data: Any) -> tuple[str, ...] | None:
    if data is None:
        return data
    elif all(isinstance(x, str) for x in data):
        return tuple(data)

    msg = f"Expected either None or a iterable of strings, got {type(data)}"
    raise TypeError(msg)


@overload
def parse_fill_value(fill_value: Any, dtype: BOOL_DTYPE) -> BOOL: ...


@overload
def parse_fill_value(fill_value: Any, dtype: INTEGER_DTYPE) -> INTEGER: ...
@overload
def parse_fill_value(fill_value: Any, dtype: FLOAT_DTYPE) -> FLOAT: ...


@overload
def parse_fill_value(fill_value: Any, dtype: COMPLEX_DTYPE) -> COMPLEX: ...


def parse_fill_value(
    fill_value: Any, dtype: BOOL_DTYPE | INTEGER_DTYPE | FLOAT_DTYPE | COMPLEX_DTYPE
) -> BOOL | INTEGER | FLOAT | COMPLEX:
    if fill_value is None:
        return dtype.type(0)
    if isinstance(fill_value, Sequence) and not isinstance(fill_value, str):
        msg = f"Cannot parse non-string sequence {fill_value} as a scalar with type {dtype}"
        raise TypeError(msg)
    return dtype.type(fill_value)


def parse_zarr_format(data: Any) -> Literal[3]:
    if data == 3:
        return 3
    raise ValueError(f"Invalid value. Expected 3. Got {data}.")


def parse_node_type_array(data: Any) -> Literal["array"]:
    if data == "array":
        return "array"
    raise ValueError(f"Invalid value. Expected 'array'. Got {data}.")


def parse_codecs(data: Any) -> tuple[Codec, ...]:
    from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec

    out: tuple[Codec, ...] = ()

    if not isinstance(data, Iterable):
        raise TypeError(f"Expected iterable, got {type(data)}")

    for c in data:
        if isinstance(
            c, ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec
        ):  # Can't use Codec here because of mypy limitation
            out += (c,)
        else:
            name_parsed, _ = parse_named_configuration(c, require_configuration=False)
            out += (get_codec_class(name_parsed).from_dict(c),)

    return out


@dataclass(frozen=True, kw_only=True)
class ArrayV3Metadata(ArrayMetadataBase):
    shape: ChunkCoords
    data_type: np.dtype[Any]
    chunk_grid: ChunkGrid
    chunk_key_encoding: ChunkKeyEncoding
    fill_value: Any
    codecs: tuple[Codec, ...]
    attributes: dict[str, Any] = field(default_factory=dict)
    dimension_names: tuple[str, ...] | None = None
    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["array"] = field(default="array", init=False)

    def __init__(
        self,
        *,
        shape: Iterable[int],
        data_type: npt.DTypeLike,
        chunk_grid: dict[str, JSON] | ChunkGrid,
        chunk_key_encoding: dict[str, JSON] | ChunkKeyEncoding,
        fill_value: Any,
        codecs: Iterable[Codec | dict[str, JSON]],
        attributes: None | dict[str, JSON],
        dimension_names: None | Iterable[str],
    ) -> None:
        """
        Create an instance of ArrayMetadata, which models the metadata for a version 3 Zarr array.
        """
        # Because the class is a frozen dataclass, we set attributes using object.__setattr__
        shape_parsed = parse_shapelike(shape)
        data_type_parsed = parse_dtype(data_type)
        chunk_grid_parsed = ChunkGrid.from_dict(chunk_grid)
        chunk_key_encoding_parsed = ChunkKeyEncoding.from_dict(chunk_key_encoding)
        dimension_names_parsed = parse_dimension_names(dimension_names)
        fill_value_parsed = parse_fill_value(fill_value, data_type_parsed)
        attributes_parsed = parse_attributes(attributes)
        codecs_parsed_partial = parse_codecs(codecs)

        array_spec = ArraySpec(
            shape=shape_parsed,
            dtype=data_type_parsed,
            fill_value=fill_value_parsed,
            order="C",  # TODO: order is not needed here.
            prototype=default_buffer_prototype,  # TODO: prototype is not needed here.
        )
        codecs_parsed = [c.evolve_from_array_spec(array_spec) for c in codecs_parsed_partial]

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "data_type", data_type_parsed)
        object.__setattr__(self, "chunk_grid", chunk_grid_parsed)
        object.__setattr__(self, "chunk_key_encoding", chunk_key_encoding_parsed)
        object.__setattr__(self, "codecs", codecs_parsed)
        object.__setattr__(self, "dimension_names", dimension_names_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "attributes", attributes_parsed)

        self._validate_metadata()

    def _validate_metadata(self) -> None:
        if isinstance(self.chunk_grid, RegularChunkGrid) and len(self.shape) != len(
            self.chunk_grid.chunk_shape
        ):
            raise ValueError(
                "`chunk_shape` and `shape` need to have the same number of dimensions."
            )
        if self.dimension_names is not None and len(self.shape) != len(self.dimension_names):
            raise ValueError(
                "`dimension_names` and `shape` need to have the same number of dimensions."
            )

        if self.fill_value is None:
            raise ValueError("`fill_value` is required.")

        for codec in self.codecs:
            codec.validate(shape=self.shape, dtype=self.data_type, chunk_grid=self.chunk_grid)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data_type

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def get_chunk_spec(
        self, _chunk_coords: ChunkCoords, order: Literal["C", "F"], prototype: BufferPrototype
    ) -> ArraySpec:
        assert isinstance(
            self.chunk_grid, RegularChunkGrid
        ), "Currently, only regular chunk grid is supported"
        return ArraySpec(
            shape=self.chunk_grid.chunk_shape,
            dtype=self.dtype,
            fill_value=self.fill_value,
            order=order,
            prototype=prototype,
        )

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.chunk_key_encoding.encode_chunk_key(chunk_coords)

    def to_buffer_dict(self) -> dict[str, Buffer]:
        def _json_convert(o: np.dtype[Any] | Enum | Codec) -> str | dict[str, Any]:
            if isinstance(o, np.dtype) or np.isscalar(o):
                return str(o)
            if isinstance(o, Enum):
                return o.name
            # this serializes numcodecs compressors
            # todo: implement to_dict for codecs
            elif isinstance(o, numcodecs.abc.Codec):
                config: dict[str, Any] = o.get_config()
                return config
            raise TypeError

        json_indent = config.get("json_indent")
        return {
            ZARR_JSON: Buffer.from_bytes(
                json.dumps(self.to_dict(), default=_json_convert, indent=json_indent).encode()
            )
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> ArrayV3Metadata:
        # TODO: Remove the type: ignores[] comments below and use a TypedDict to type `data`
        # check that the zarr_format attribute is correct
        _ = parse_zarr_format(data.pop("zarr_format"))
        # check that the node_type attribute is correct
        _ = parse_node_type_array(data.pop("node_type"))

        data["dimension_names"] = data.pop("dimension_names", None)

        return cls(**data)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        out_dict = super().to_dict()

        if not isinstance(out_dict, dict):
            raise TypeError(f"Expected dict. Got {type(out_dict)}.")

        # if `dimension_names` is `None`, we do not include it in
        # the metadata document
        if out_dict["dimension_names"] is None:
            out_dict.pop("dimension_names")
        return out_dict

    def update_shape(self, shape: ChunkCoords) -> Self:
        return replace(self, shape=shape)

    def update_attributes(self, attributes: dict[str, JSON]) -> Self:
        return replace(self, attributes=attributes)


class DataType(Enum):
    bool = "bool"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    float32 = "float32"
    float64 = "float64"

    @property
    def byte_count(self) -> int:
        data_type_byte_counts = {
            DataType.bool: 1,
            DataType.int8: 1,
            DataType.int16: 2,
            DataType.int32: 4,
            DataType.int64: 8,
            DataType.uint8: 1,
            DataType.uint16: 2,
            DataType.uint32: 4,
            DataType.uint64: 8,
            DataType.float32: 4,
            DataType.float64: 8,
        }
        return data_type_byte_counts[self]

    @property
    def has_endianness(self) -> _bool:
        # This might change in the future, e.g. for a complex with 2 8-bit floats
        return self.byte_count != 1

    def to_numpy_shortname(self) -> str:
        data_type_to_numpy = {
            DataType.bool: "bool",
            DataType.int8: "i1",
            DataType.int16: "i2",
            DataType.int32: "i4",
            DataType.int64: "i8",
            DataType.uint8: "u1",
            DataType.uint16: "u2",
            DataType.uint32: "u4",
            DataType.uint64: "u8",
            DataType.float32: "f4",
            DataType.float64: "f8",
        }
        return data_type_to_numpy[self]

    @classmethod
    def from_dtype(cls, dtype: np.dtype[Any]) -> DataType:
        dtype_to_data_type = {
            "|b1": "bool",
            "bool": "bool",
            "|i1": "int8",
            "<i2": "int16",
            "<i4": "int32",
            "<i8": "int64",
            "|u1": "uint8",
            "<u2": "uint16",
            "<u4": "uint32",
            "<u8": "uint64",
            "<f4": "float32",
            "<f8": "float64",
        }
        return DataType[dtype_to_data_type[dtype.str]]
