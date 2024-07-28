from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict, TypeGuard, cast, overload

import numpy as np
import numpy.typing as npt

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec, Codec
from zarr.abc.metadata import Metadata
from zarr.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.config import config
from zarr.metadata.common import (
    ArrayMetadataBase,
    SeparatorLiteral,
    parse_attributes,
    parse_separator,
)
from zarr.registry import get_codec_class

ChunkKeyEncodingName = Literal["v2", "default"]
CHUNK_KEY_ENCODING_NAMES: Final = ("v2", "default")

if TYPE_CHECKING:
    from typing_extensions import Self

import numcodecs.abc

from zarr.array_spec import ArraySpec
from zarr.common import (
    JSON,
    ZARR_JSON,
    ChunkCoords,
    ChunkCoordsLike,
    parse_dtype,
    parse_named_configuration,
    parse_shapelike,
)

# For type checking
_bool = bool


def unpack_named_configuration(
    data: Any, allow_missing_config: bool = True
) -> tuple[str, dict[str, object] | None]:
    """
    Unpack a dict-like object with a `{'name': str, 'configuration': Any}` structure
    into a tuple where the first element is the value associated with the `name` key, and the
    second element is the value associated with the `configuration` key.

    Parameters
    ----------
    data: Any
        The data to be unpacked

    allow_missing_config: bool, default=False
        Whether a missing 'configuration' key is allowed. If this parameter is set to `False` and
        the `configuration` key is missing from `data`, then a KeyError will be raised. If this
        parameter is set to `True`, then the `configuration` will be assigned to `None`.

    Returns
    -------
    tuple
        A tuple containing the values associated with the `name` and `configuration` keys of `data`,
        respectively.
    """
    try:
        name = data["name"]
        if not isinstance(name, str):
            msg = (
                "The value associated with the `name` key must be a string."
                f"Got an object with type {type(name)} instead."
            )
            raise TypeError(msg)
        if "configuration" in data or allow_missing_config:
            config = data.get("configuration", None)
            return name, config
        else:
            msg = (
                f"Required key 'configuration' was missing from {data}."
                "Either add that key to your data, or "
                "call this function with `allow_missing_config` set to `True`"
                "to interpret a missing 'configuration' key as `None`."
            )
            raise KeyError(msg)
    except (KeyError, TypeError) as e:
        msg = (
            f"Data {data} is not structured correctly."
            "Ensure that it is a mapping with a 'name' key and a 'configuration' key."
        )
        raise ValueError(msg) from e


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


class RegularChunkGridConfigDict(TypedDict):
    chunk_shape: tuple[int, ...]


class RegularChunkGridDict(TypedDict):
    name: Literal["regular"]
    configuration: RegularChunkGridConfigDict


@dataclass(frozen=True)
class RegularChunkGridConfig(Metadata):
    chunk_shape: ChunkCoords

    def __init__(self, *, chunk_shape: ChunkCoordsLike) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)
        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)

    def to_dict(self) -> RegularChunkGridConfigDict:
        return {"chunk_shape": self.chunk_shape}


def parse_regular_chunk_grid_config(data: object) -> RegularChunkGridConfig:
    if isinstance(data, RegularChunkGridConfig):
        return data
    try:
        chunk_shape = data["chunk_shape"]  # type: ignore[index]
        return RegularChunkGridConfig(chunk_shape=chunk_shape)
    except TypeError as e:
        msg = (
            "Invalid configuration for regular chunk grid."
            f"Got type {type(data)}, expected an object supporting dict-style indexing."
        )
        raise ValueError(msg) from e
    except KeyError as e:
        msg = (
            "Invalid configuration for regular chunk grid."
            f"Required key `chunk_shape` is missing from {data}"
        )
        raise ValueError(msg) from e


@dataclass(frozen=True)
class RegularChunkGrid(Metadata):
    name: Literal["regular"] = field(init=False, default="regular")
    configuration: RegularChunkGridConfig

    def __init__(self, *, configuration: RegularChunkGridConfig | RegularChunkGridConfigDict):
        config_parsed = parse_regular_chunk_grid_config(configuration)
        object.__setattr__(self, "configuration", config_parsed)

    @classmethod
    def _from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "regular")

        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> RegularChunkGridDict:
        return super().to_dict()


def parse_chunk_grid(data: Any) -> RegularChunkGrid:
    if isinstance(data, RegularChunkGrid):
        return data

    name, config = unpack_named_configuration(data, allow_missing_config=False)

    if name == "regular":
        return RegularChunkGrid(configuration=config)  # type: ignore

    else:
        msg = f'Invalid chunk grid name. Got {name}, expected "regular".'
        raise ValueError(msg)


class ChunkKeyConfigDict(TypedDict):
    separator: SeparatorLiteral


class ChunkKeyEncodingDict(TypedDict):
    name: Literal["v2", "default"]
    configuration: ChunkKeyConfigDict


@dataclass(frozen=True)
class ChunkKeyConfig(Metadata):
    separator: SeparatorLiteral

    def __init__(self, *, separator: SeparatorLiteral):
        separator_parsed = parse_separator(separator)
        object.__setattr__(self, "separator", separator_parsed)

    def to_dict(self) -> ChunkKeyConfigDict:
        return super().to_dict()


def parse_chunk_key_config(data: object) -> ChunkKeyConfig:
    if isinstance(data, ChunkKeyConfig):
        return data
    if data is None:
        separator = "/"
    else:
        try:
            separator = data["separator"]
        except TypeError as e:
            msg = (
                "Invalid configuration for chunk key encoding."
                f"Got type {type(config)}, expected an object supporting dict-style indexing."
            )
            raise ValueError(msg) from e
        except KeyError as e:
            msg = (
                "Invalid configuration for chunk key encoding."
                f"Required key `separator` is missing from {config}"
            )
            raise ValueError(msg) from e

    return ChunkKeyConfig(separator=separator)


@dataclass(frozen=True)
class ChunkKeyEncoding(Metadata):
    name: Literal["v2", "default"]
    configuration: ChunkKeyConfig

    def __init__(
        self, *, name: ChunkKeyEncodingName, configuration: ChunkKeyConfigDict | ChunkKeyConfig
    ) -> None:
        name_parsed = parse_chunk_key_encoding_name(name)
        config_parsed = parse_chunk_key_config(configuration)

        object.__setattr__(self, "name", name_parsed)
        object.__setattr__(self, "configuration", config_parsed)

    def to_dict(self) -> ChunkKeyEncodingDict:
        return super().to_dict()


def is_chunk_key_encoding_name(data: object) -> TypeGuard[ChunkKeyEncodingName]:
    return data in CHUNK_KEY_ENCODING_NAMES


def parse_chunk_key_encoding_name(data: object) -> ChunkKeyEncodingName:
    if is_chunk_key_encoding_name(data):
        return data
    msg = (
        f"Invalid chunk key encoding name. Expected one of {CHUNK_KEY_ENCODING_NAMES}."
        f"Got {data} instead."
    )
    raise ValueError(msg)


def parse_chunk_key_encoding(data: object) -> ChunkKeyEncoding:
    if isinstance(data, ChunkKeyEncoding):
        return data

    name, config = unpack_named_configuration(data)
    return ChunkKeyEncoding(name=name, configuration=config)  # type: ignore


@dataclass(frozen=True, kw_only=True)
class ArrayV3Metadata(ArrayMetadataBase):
    shape: ChunkCoords
    data_type: np.dtype[Any]
    chunk_grid: RegularChunkGrid
    chunk_key_encoding: ChunkKeyEncoding
    fill_value: Any
    codecs: tuple[Codec, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)
    dimension_names: tuple[str | None, ...] | None
    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["array"] = field(default="array", init=False)

    def __init__(
        self,
        *,
        shape: Iterable[int],
        data_type: npt.DTypeLike,
        chunk_grid: dict[str, JSON] | RegularChunkGrid,
        chunk_key_encoding: dict[str, JSON] | ChunkKeyEncoding,
        fill_value: Any,
        codecs: Iterable[Codec | dict[str, JSON]],
        dimension_names: Iterable[str | None] | None,
        attributes: None | dict[str, JSON],
    ) -> None:
        """
        Because the class is a frozen dataclass, we set attributes using object.__setattr__
        """
        shape_parsed = parse_shapelike(shape)
        data_type_parsed = parse_dtype(data_type)
        chunk_grid_parsed = parse_chunk_grid(chunk_grid)
        chunk_key_encoding_parsed = parse_chunk_key_encoding(chunk_key_encoding)
        dimension_names_parsed = parse_dimension_names(dimension_names)
        fill_value_parsed = parse_fill_value(fill_value, dtype=data_type_parsed)
        attributes_parsed = parse_attributes(attributes)
        codecs_parsed_partial = parse_codecs(codecs)

        array_spec = ArraySpec(
            shape=shape_parsed,
            dtype=data_type_parsed,
            fill_value=fill_value_parsed,
            order="C",  # TODO: order is not needed here.
            prototype=default_buffer_prototype(),  # TODO: prototype is not needed here.
        )

        codecs_parsed = tuple(c.evolve_from_array_spec(array_spec) for c in codecs_parsed_partial)

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
        if len(self.shape) != len(self.chunk_grid.configuration.chunk_shape):
            raise ValueError(
                "`chunk_shape` and `shape` need to have the same number of dimensions."
            )
        if self.dimension_names is not None and len(self.shape) != len(self.dimension_names):
            msg = (
                "Invalid metadata. The length of the `dimension_names` attribute must match the "
                f"length of the `shape` attribute. Got `dimension_names`={self.dimension_names}, "
                f"with length={len(self.dimension_names)}, and `shape`={self.shape} with "
                f"length={len(self.shape)}"
            )

            raise ValueError(msg)

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
            shape=self.chunk_grid.configuration.chunk_shape,
            dtype=self.dtype,
            fill_value=self.fill_value,
            order=order,
            prototype=prototype,
        )

    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:
        def _json_convert(o: Any) -> Any:
            if isinstance(o, np.dtype):
                return str(o)
            if np.isscalar(o):
                # convert numpy scalar to python type, and pass
                # python types through
                out = getattr(o, "item", lambda: o)()
                if isinstance(out, complex):
                    # python complex types are not JSON serializable, so we use the
                    # serialization defined in the zarr v3 spec
                    return [out.real, out.imag]
                return out
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
            ZARR_JSON: prototype.buffer.from_bytes(
                json.dumps(self.to_dict(), default=_json_convert, indent=json_indent).encode()
            )
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> ArrayV3Metadata:
        data_copy = data.copy()
        # TODO: Remove the type: ignores[] comments below and use a TypedDict to type `data`
        # check that the zarr_format attribute is correct
        _ = parse_zarr_format(data_copy.pop("zarr_format"))  # type: ignore[arg-type]
        # check that the node_type attribute is correct
        _ = validate_node_type_array(data_copy.pop("node_type"))  # type: ignore[arg-type]

        # dimension_names is an optional key. If it is not present, then we set it to `None`
        data_copy["dimension_names"] = data_copy.pop("dimension_names", None)

        # attributes is an optional key. If it is not present, then we set it to None
        data_copy["attributes"] = data_copy.pop("attributes", None)

        return cls(**data_copy)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        out_dict = super().to_dict()

        # if `dimension_names` is `None`, we do not include it in
        # the metadata document
        if out_dict["dimension_names"] is None:
            out_dict.pop("dimension_names")
        return out_dict

    def update_shape(self, shape: ChunkCoords) -> Self:
        return replace(self, shape=shape)

    def update_attributes(self, attributes: Mapping[str, JSON]) -> Self:
        return replace(self, attributes=attributes)


def parse_dimension_names(data: Any) -> tuple[str | None, ...] | None:
    if data is None:
        return None
    elif all(isinstance(x, (str, type(None))) for x in data):
        return tuple(data)
    else:
        msg = f"Expected either None or a iterable of strings or `None`. Got {type(data)} instead."
        raise TypeError(msg)


def parse_zarr_format(data: Literal[3]) -> Literal[3]:
    if data == 3:
        return 3
    raise ValueError(f"Invalid value. Expected 3. Got {data}.")


def validate_node_type_array(data: Literal["array"]) -> Literal["array"]:
    if data == "array":
        return "array"
    raise ValueError(f"Invalid value. Expected 'array'. Got {data}.")


def parse_codecs(data: Iterable[Codec | dict[str, JSON]]) -> tuple[Codec, ...]:
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


BOOL = np.bool_
BOOL_DTYPE = np.dtypes.BoolDType

INTEGER_DTYPE = (
    np.dtypes.Int8DType
    | np.dtypes.Int16DType
    | np.dtypes.Int32DType
    | np.dtypes.Int64DType
    | np.dtypes.UByteDType
    | np.dtypes.UInt16DType
    | np.dtypes.UInt32DType
    | np.dtypes.UInt64DType
)

INTEGER = np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64
FLOAT_DTYPE = np.dtypes.Float16DType | np.dtypes.Float32DType | np.dtypes.Float64DType
FLOAT = np.float16 | np.float32 | np.float64
COMPLEX_DTYPE = np.dtypes.Complex64DType | np.dtypes.Complex128DType
COMPLEX = np.complex64 | np.complex128
# todo: r* dtypes


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
    """
    Parse `fill_value`, a potential fill value, into an instance of `dtype`, a data type.
    If `fill_value` is `None`, then this function will return the result of casting the value 0
    to the provided data type. Otherwise, `fill_value` will be cast to the provided data type.

    Note that some numpy dtypes use very permissive casting rules. For example,
    `np.bool_({'not remotely a bool'})` returns `True`. Thus this function should not be used for
    validating that the provided fill value is a valid instance of the data type.

    Parameters
    ----------
    fill_value: Any
        A potential fill value.
    dtype: BOOL_DTYPE | INTEGER_DTYPE | FLOAT_DTYPE | COMPLEX_DTYPE
        A numpy data type that models a data type defined in the Zarr V3 specification.

    Returns
    -------
    A scalar instance of `dtype`
    """
    if fill_value is None:
        return dtype.type(0)
    if isinstance(fill_value, Sequence) and not isinstance(fill_value, str):
        if dtype in (np.complex64, np.complex128):
            dtype = cast(COMPLEX_DTYPE, dtype)
            if len(fill_value) == 2:
                # complex datatypes serialize to JSON arrays with two elements
                return dtype.type(complex(*fill_value))
            else:
                msg = (
                    f"Got an invalid fill value for complex data type {dtype}."
                    f"Expected a sequence with 2 elements, but {fill_value} has "
                    f"length {len(fill_value)}."
                )
                raise ValueError(msg)
        msg = f"Cannot parse non-string sequence {fill_value} as a scalar with type {dtype}."
        raise TypeError(msg)
    return dtype.type(fill_value)
