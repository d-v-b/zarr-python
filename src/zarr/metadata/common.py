from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypeGuard

if TYPE_CHECKING:
    from typing_extensions import Self

    from zarr.common import JSON, ChunkCoords, Mapping, ZarrFormat

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from zarr.abc.metadata import Metadata
from zarr.array_spec import ArraySpec
from zarr.buffer import Buffer, BufferPrototype

SeparatorLiteral = Literal[".", "/"]
SEPARATOR: Final = (".", "/")


@dataclass(frozen=True, kw_only=True)
class ArrayMetadataBase(Metadata, ABC):
    shape: ChunkCoords
    fill_value: Any
    attributes: Mapping[str, JSON]
    zarr_format: ZarrFormat

    @property
    @abstractmethod
    def dtype(self) -> np.dtype[Any]:
        pass

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @abstractmethod
    def get_chunk_spec(
        self, _chunk_coords: ChunkCoords, order: Literal["C", "F"], prototype: BufferPrototype
    ) -> ArraySpec:
        pass

    @abstractmethod
    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:
        pass

    @abstractmethod
    def update_shape(self, shape: ChunkCoords) -> Self:
        pass

    @abstractmethod
    def update_attributes(self, attributes: Mapping[str, JSON]) -> Self:
        pass


def parse_attributes(data: None | Mapping[str, JSON]) -> Mapping[str, JSON]:
    """
    Parse attributes. If the input is None, return an empty dict. Otherwise, the input is
    returned as-is. In the future, this function might apply more extensive data validation to
    handle more input types.
    """
    if data is None:
        return {}

    return data


def is_separator(data: object) -> TypeGuard[SeparatorLiteral]:
    return data in SEPARATOR


def parse_separator(data: Any) -> SeparatorLiteral:
    if data is None:
        return "/"

    if is_separator(data):
        return data

    msg = f"Invalid dimension separator. Expected one of {SEPARATOR}, got {data} instead."
    raise ValueError(msg)
