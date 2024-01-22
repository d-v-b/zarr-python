from __future__ import annotations
from codecs import Codec
from dataclasses import asdict, dataclass, field

from cattr import Converter
from zarr.v3.codecs.registry import get_codec_class, get_codec_metadata_class
from zarr.v3.common import RuntimeConfiguration


import numpy as np
import attr


import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union

from zarr.v3.types import JSON, Attributes, ChunkCoords


class CodecMetadata(Protocol):
    @property
    def name(self) -> str:
        pass


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

    @classmethod
    def from_json(cls, json_data: Dict[str, JSON]):
        return cls(chunk_shape=json_data['chunk_shape'])

@dataclass(frozen=True)
class RegularChunkGridMetadata:
    configuration: RegularChunkGridConfigurationMetadata
    name: Literal["regular"] = field(default="regular", init=False)

    def from_json(cls, json_data: Dict[str, JSON]):
        if 'name' not in json_data:
            raise ValueError('Invalid `name`: `name` property is missing from the metadata document.')
        if name := json_data.get('name') is not 'regular':
            msg = f'Invalid `name` property; expected "regular", got {name}'
            raise ValueError(msg)
        return cls(
            configuration=RegularChunkGridConfigurationMetadata.from_json(json_data['configuration'])
            )

@dataclass(frozen=True)
class DefaultChunkKeyEncodingConfigurationMetadata:
    separator: Literal[".", "/"] = "/"

    @classmethod
    def from_json(cls, json_data: Dict[str, JSON]):
        return cls(separator=json_data.get('separator'))


@dataclass(frozen=True)
class DefaultChunkKeyEncodingMetadata:
    configuration: DefaultChunkKeyEncodingConfigurationMetadata = (
        DefaultChunkKeyEncodingConfigurationMetadata()
    )
    name: Literal["default"] = field(default="default", init=False)
    
    @classmethod
    def from_json(cls, json_data: Dict[str, JSON]):
        return cls(
            configuration=DefaultChunkKeyEncodingConfigurationMetadata.from_json(json_data['configuration']))

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        if chunk_key == "c":
            return ()
        return tuple(map(int, chunk_key[1:].split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.configuration.separator.join(map(str, ("c",) + chunk_coords))


@dataclass(frozen=True)
class V2ChunkKeyEncodingConfigurationMetadata:
    separator: Literal[".", "/"] = "."

    @classmethod
    def from_json(cls, json_data: Dict[str, JSON]):
        return cls(separator=json_data['separator'])

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
    def from_json(cls, json_data: Dict[str, JSON]):
        return cls(configuration=V2ChunkKeyEncodingConfigurationMetadata.from_json(json_data['configuration']))
    
ChunkKeyEncodingMetadata = Union[DefaultChunkKeyEncodingMetadata, V2ChunkKeyEncodingMetadata]


""" def make_cattr():
    from zarr.v3.metadata.v3.array import (
        ChunkKeyEncodingMetadata,
    )
    from zarr.v3.codecs.registry import get_codec_metadata_class

    converter = Converter()

    def _structure_chunk_key_encoding_metadata(d: Dict[str, Any], _t) -> ChunkKeyEncodingMetadata:
        if d["name"] == "default":
            return converter.structure(d, DefaultChunkKeyEncodingMetadata)
        if d["name"] == "v2":
            return converter.structure(d, V2ChunkKeyEncodingMetadata)
        raise KeyError

    converter.register_structure_hook(
        ChunkKeyEncodingMetadata, _structure_chunk_key_encoding_metadata
    )

    def _structure_codec_metadata(d: Dict[str, Any], _t=None) -> CodecMetadata:
        codec_metadata_cls = get_codec_metadata_class(d["name"])
        return converter.structure(d, codec_metadata_cls)

    converter.register_structure_hook(CodecMetadata, _structure_codec_metadata)

    converter.register_structure_hook_factory(
        lambda t: str(t) == "ForwardRef('CodecMetadata')",
        lambda t: _structure_codec_metadata,
    )

    def _structure_order(d: Any, _t=None) -> Union[Literal["C", "F"], Tuple[int, ...]]:
        if d == "C":
            return "C"
        if d == "F":
            return "F"
        if isinstance(d, list):
            return tuple(d)
        raise KeyError

    converter.register_structure_hook_factory(
        lambda t: str(t) == "typing.Union[typing.Literal['C', 'F'], typing.Tuple[int, ...]]",
        lambda t: _structure_order,
    )

    # Needed for v2 fill_value
    def _structure_fill_value(d: Any, _t=None) -> Union[None, int, float]:
        if d is None:
            return None
        try:
            return int(d)
        except ValueError:
            pass
        try:
            return float(d)
        except ValueError:
            pass
        raise ValueError

    converter.register_structure_hook_factory(
        lambda t: str(t) == "typing.Union[NoneType, int, float]",
        lambda t: _structure_fill_value,
    )

    # Needed for v2 dtype
    converter.register_structure_hook(
        np.dtype,
        lambda d, _: np.dtype(d),
    )

    return converter """


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

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data_type.value)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def get_core_metadata(
            self, 
            runtime_configuration: RuntimeConfiguration) -> CoreArrayMetadata:
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
            self_dict.pop('dimension_names')
        return json.dumps(
            self_dict, 
            default=_json_convert
            ).encode()

    @classmethod
    def from_json(cls, json_data: Dict[str, JSON], runtime_configuration: RuntimeConfiguration) -> ArrayMetadata:
        node_type = json_data.pop('node_type', None)
        zarr_format = json_data.pop('zarr_format', None)
        if node_type is not None and zarr_format is not None:
            data_type = DataType(json_data.get('data_type'))
            chunk_grid = RegularChunkGridMetadata.from_json(json_data=['chunk_grid'])
            shape=json_data['shape']
            dimension_names = json_data.get('dimension_names', None)
            fill_value = json_data['fill_value']
            attributes = json_data.get('attributes', None)
            array_metadata = CoreArrayMetadata(
                shape=shape,
                chunk_shape=chunk_grid.configuration.chunk_shape 
                fill_value=fill_value,
                data_type=data_type,
                runtime_configuration=runtime_configuration)
            codecs: List[Codec] = []
            for json_codec in json_data['codecs']:
                codec_metadata = get_codec_metadata_class(json_codec['name']).from_json(json_codec)
                codec_class = get_codec_class(json_codec['name'])
                codecs.append(
                    codec_class.from_metadata(
                        codec_metadata=codec_metadata,
                        array_metadata=array_metadata))
            return cls(
                shape=shape,
                data_type=data_type,
                chunk_grid=chunk_grid,
                fill_value=fill_value,
                codecs=codecs,
                dimension_names=dimension_names,
                attributes=attributes
                )
        raise ValueError('The JSON document provided is invalid for a Zarr V3 array.')


class ShardingCodecIndexLocation(Enum):
    start = "start"
    end = "end"


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
