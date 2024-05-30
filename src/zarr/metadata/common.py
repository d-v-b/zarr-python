from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np
from typing_extensions import Self

from zarr.abc.codec import Codec, CodecPipeline
from zarr.abc.metadata import Metadata
from zarr.buffer import Buffer
from zarr.chunk_grids import ChunkGrid
from zarr.common import JSON, ArraySpec, ChunkCoords

# for type checking
_bool = bool


def _json_convert(o: np.dtype[Any] | Enum | Codec) -> str | dict[str, JSON]:
    import numcodecs

    if isinstance(o, np.dtype):
        return str(o)
    if isinstance(o, Enum):
        return o.name
    # this serializes numcodecs compressors
    # todo: implement to_dict for codecs
    elif isinstance(o, numcodecs.abc.Codec):
        return o.get_config()
    raise TypeError


@dataclass(frozen=True, kw_only=True)
class ArrayMetadataBase(Metadata, ABC):
    shape: ChunkCoords
    chunk_grid: ChunkGrid
    attributes: dict[str, JSON]

    @property
    @abstractmethod
    def dtype(self) -> np.dtype[Any]:
        pass

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @property
    @abstractmethod
    def codec_pipeline(self) -> CodecPipeline:
        pass

    @abstractmethod
    def get_chunk_spec(self, _chunk_coords: ChunkCoords, order: Literal["C", "F"]) -> ArraySpec:
        pass

    @abstractmethod
    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        pass

    @abstractmethod
    def to_buffer_dict(self) -> dict[str, Buffer]:
        pass

    @abstractmethod
    def update_shape(self, shape: ChunkCoords) -> Self:
        pass

    @abstractmethod
    def update_attributes(self, attributes: dict[str, JSON]) -> Self:
        pass


def parse_attributes(data: None | dict[str, JSON]) -> dict[str, JSON]:
    if data is None:
        return {}

    return data
