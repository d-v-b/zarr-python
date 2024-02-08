from __future__ import annotations
from codecs import Codec
from dataclasses import asdict, dataclass, field

from cattr import Converter
from zarr.v3.abc.codec import CodecMetadata
from zarr.v3.codecs.registry import get_codec_class, get_codec_metadata_class
from zarr.v3.common import RuntimeConfiguration


import numpy as np
import numpy.typing as npt

import json
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from zarr.v3.common import parse_shape

from zarr.v3.types import JSON, Attributes, ChunkCoords


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


@dataclass(frozen=True)
class RegularChunkGridConfigurationMetadata:
    chunk_shape: ChunkCoords

    def __init__(self, chunk_shape: Any):
        chunk_shape_parsed = parse_shape(chunk_shape)
        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]):
        return cls(chunk_shape=data["chunk_shape"])


@dataclass(frozen=True)
class RegularChunkGridMetadata:
    configuration: RegularChunkGridConfigurationMetadata
    name: Literal["regular"] = field(default="regular", init=False)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]):
        if "name" not in data:
            raise ValueError(
                "Invalid `name`: `name` property is missing from the metadata document."
            )
        if (name := data.get("name")) != "regular":
            msg = f'Invalid `name` property; expected "regular", got {name}'
            raise ValueError(msg)
        return cls(
            configuration=RegularChunkGridConfigurationMetadata.from_dict(data["configuration"])
        )


def parse_separator(data: Any) -> Literal[".", "/"]:
    if data not in (".", "/"):
        msg = f'Expected one of {(".", "/")}, got {data} instead.'
        raise ValueError(msg)
    return data


@dataclass(frozen=True)
class DefaultChunkKeyEncodingConfigurationMetadata:
    separator: Literal[".", "/"] = "/"

    def __init__(self, separator: Literal[".", "/"] = "/"):
        separator_parsed = parse_separator(separator)
        object.__setattr__(self, "separator", separator_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]):
        return cls(separator=data.get("separator"))


@dataclass(frozen=True)
class DefaultChunkKeyEncodingMetadata:
    configuration: DefaultChunkKeyEncodingConfigurationMetadata = (
        DefaultChunkKeyEncodingConfigurationMetadata()
    )
    name: Literal["default"] = field(default="default", init=False)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]):
        return cls(
            configuration=DefaultChunkKeyEncodingConfigurationMetadata.from_dict(
                data["configuration"]
            )
        )

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        if chunk_key == "c":
            return ()
        return tuple(map(int, chunk_key[1:].split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.configuration.separator.join(map(str, ("c",) + chunk_coords))


@dataclass(frozen=True)
class V2ChunkKeyEncodingConfigurationMetadata:
    separator: Literal[".", "/"] = "."

    def __init__(self, separator: Literal[".", "/"]):
        separator_parsed = parse_separator(separator)
        object.__setattr__(self, "separator", separator_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]):
        return cls(separator=data["separator"])


@dataclass(frozen=True)
class V2ChunkKeyEncodingMetadata:
    configuration: V2ChunkKeyEncodingConfigurationMetadata = (
        V2ChunkKeyEncodingConfigurationMetadata()
    )
    name: Literal["v2"] = field(default="v2", init=False)

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        return tuple(map(int, chunk_key.split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.configuration.separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]):
        return cls(
            configuration=V2ChunkKeyEncodingConfigurationMetadata.from_dict(data["configuration"])
        )


ChunkKeyEncodingMetadata = Union[DefaultChunkKeyEncodingMetadata, V2ChunkKeyEncodingMetadata]


@dataclass(frozen=True)
class CoreArrayMetadata:
    shape: ChunkCoords
    chunk_shape: ChunkCoords
    data_type: DataType
    fill_value: Any
    runtime_configuration: RuntimeConfiguration

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data_type.value)

    @property
    def ndim(self) -> int:
        return len(self.shape)


def parse_data_type(data):
    return data


def parse_chunk_grid(data):
    return data


def parse_chunk_key_encoding(data):
    return data

# todo: handle None
def parse_fill_value(data):
    return data


def parse_codec(data: Any) -> Codec:
    if "name" in data:
        return get_codec_metadata_class(data["name"]).from_dict(data)
    msg = f'Expected a key called "name" in {data}, but it was not found.'
    raise ValueError(msg)


def parse_codecs_json(data: Dict[str, JSON]):
    if "codecs" not in data:
        msg = 'Expected key "codecs" was not found.'
        raise ValueError(msg)
    codecs_in = data["codecs"]
    if not isinstance(codecs_in, list):
        msg = f"Codecs must be a list, got {type(codecs_in)}"
        raise TypeError(msg)


def parse_codecs(data: Any) -> List[Codec]:
    if not isinstance(data, Iterable):
        msg = f"Expected an Iterable, got {type(data)}"
        raise TypeError(msg)
    return list(map(parse_codec, data))


def parse_dimension_names(data):
    if data is not None:
        return tuple(data)
    return data

def validate_dimensionality(
        shape: Tuple[int, ...], 
        chunk_grid: RegularChunkGridMetadata, 
        dimension_names: Optional[Tuple[str, ...]]):
    if len(shape) != len(chunk_grid.configuration.chunk_shape):
       msg = ("`chunk_grid.configuration.chunk_shape` and `shape` do not have the same length."
              f"`chunk_grid.configuration.chunk_shape` has length={len(chunk_grid.configuration.chunk_shape)}"
              f"but `shape` has length={len(shape)}")
       raise ValueError(msg)
    if dimension_names is not None and len(shape) != len(dimension_names):
        msg 
def _validate_metadata(self) -> None:
    assert len(self.metadata.shape) == len(
        self.metadata.chunk_grid.configuration.chunk_shape
    ), "`chunk_shape` and `shape` need to have the same number of dimensions."
    assert self.metadata.dimension_names is None or len(self.metadata.shape) == len(
        self.metadata.dimension_names
    ), "`dimension_names` and `shape` need to have the same number of dimensions."
    assert self.metadata.fill_value is not None, "`fill_value` is required."


@dataclass(frozen=True)
class ArrayMetadata:
    shape: ChunkCoords
    data_type: DataType
    chunk_grid: RegularChunkGridMetadata
    chunk_key_encoding: ChunkKeyEncodingMetadata
    fill_value: Any
    codecs: List[CodecMetadata]
    dimension_names: Optional[Tuple[str, ...]] = None
    attributes: Optional[Attributes] = field(default_factory=dict)
    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["array"] = field(default="array", init=False)

    def __init__(
        self,
        shape: ChunkCoords,
        data_type: npt.DtypeLike,
        chunk_grid: RegularChunkGridMetadata,
        chunk_key_encoding: ChunkKeyEncodingMetadata,
        fill_value: Any,
        codecs: List[CodecMetadata],
        dimension_names: Optional[Tuple[str, ...]] = None,
        attributes: Optional[Attributes] = field(default_factory=dict),
    ):

        shape_parsed = parse_shape(shape)
        object.__setattr__(self, "shape", shape_parsed)
        data_type_parsed = parse_data_type(data_type)
        object.__setattr__(self, "data_type", data_type_parsed)
        chunk_grid_parsed = parse_chunk_grid(chunk_grid)

        object.__setattr__(self, "chunk_grid", chunk_grid_parsed)
        chunk_key_encoding_parsed = parse_chunk_key_encoding(chunk_key_encoding)
        object.__setattr__(self, "chunk_key_encoding", chunk_key_encoding_parsed)
        fill_value_parsed = parse_fill_value(fill_value)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        codecs_parsed = parse_codecs(codecs)
        object.__setattr__(self, "codecs", codecs_parsed)
        dimension_names_parsed = parse_dimension_names(dimension_names)
        object.__setattr__(self, "dimension_names", dimension_names_parsed)
        object.__setattr__(self, "attributes", attributes)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data_type.value)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def get_core_metadata(self, runtime_configuration: RuntimeConfiguration) -> CoreArrayMetadata:
        return CoreArrayMetadata(
            shape=self.shape,
            chunk_shape=self.chunk_grid.configuration.chunk_shape,
            data_type=self.data_type,
            fill_value=self.fill_value,
            runtime_configuration=runtime_configuration,
        )

    def to_bytes(self) -> bytes:
        def _json_convert(o):
            if isinstance(o, Enum):
                return o.name
            raise TypeError

        self_dict = asdict(self)
        if self.dimension_names is None:
            self_dict.pop("dimension_names")
        return json.dumps(self_dict, default=_json_convert).encode()

    @classmethod
    def from_dict(
        cls, data: Dict[str, JSON], runtime_configuration: RuntimeConfiguration
    ) -> ArrayMetadata:
        node_type = data.pop("node_type", None)
        zarr_format = data.pop("zarr_format", None)
        if node_type in ("array", "group") and zarr_format is 3:
            data_type = DataType(data.get("data_type"))
            chunk_grid = RegularChunkGridMetadata.from_dict(data["chunk_grid"])
            shape = data["shape"]
            dimension_names = data.get("dimension_names", None)
            fill_value = data["fill_value"]
            attributes = data.get("attributes", None)
            array_metadata = CoreArrayMetadata(
                shape=shape,
                chunk_shape=chunk_grid.configuration.chunk_shape,
                fill_value=fill_value,
                data_type=data_type,
                runtime_configuration=runtime_configuration,
            )
            codecs = parse_codecs(data["codecs"])
            chunk_key_encoding = parse_chunk_key_encoding_json(data["chunk_key_encoding"])

            return cls(
                shape=shape,
                data_type=data_type,
                chunk_grid=chunk_grid,
                fill_value=fill_value,
                codecs=codecs,
                dimension_names=dimension_names,
                attributes=attributes,
                chunk_key_encoding=chunk_key_encoding,
            )
        raise ValueError("The JSON document provided is invalid for a Zarr V3 array.")


ShardingCodecIndexLocation = Literal["start", "end"]


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


def parse_chunk_key_encoding_json(data: Dict[str, JSON]) -> ChunkKeyEncodingMetadata:
    name = data["name"]
    if name == "v2":
        return V2ChunkKeyEncodingMetadata.from_dict(data)
    elif name == "default":
        return DefaultChunkKeyEncodingMetadata.from_dict(data)
    msg = (
        f'Invalid `chunk_grid.name` property in JSON. Expected one of ["v2", "default"], got {name}'
    )
    raise ValueError(msg)
