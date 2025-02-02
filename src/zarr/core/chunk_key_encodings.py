from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    TypeAlias,
    TypedDict,
    cast,
)

if TYPE_CHECKING:
    from typing import NotRequired

    from zarr.core.common import (
        JSON,
        ChunkCoords,
    )

from zarr.abc.metadata import NamedConfig

SeparatorLiteral = Literal[".", "/"]
DEFAULT_V3_SEPARATOR: SeparatorLiteral = "/"
DEFAULT_V2_SEPARATOR: SeparatorLiteral = "."


def parse_separator(data: JSON) -> SeparatorLiteral:
    if data not in (".", "/"):
        raise ValueError(f"Expected an '.' or '/' separator. Got {data} instead.")
    return cast(SeparatorLiteral, data)


def parse_chunk_key_encoding(
    data: ChunkKeyEncodingLike,
) -> V2ChunkKeyEncoding | DefaultChunkKeyEncoding:
    if isinstance(data, V2ChunkKeyEncoding | DefaultChunkKeyEncoding):
        return data
    name = data["name"]
    if name == "default":
        return DefaultChunkKeyEncoding.from_dict(data)  # type: ignore[arg-type]
    elif name == "v2":
        return V2ChunkKeyEncoding.from_dict(data)  # type: ignore[arg-type]
    msg = f"Unknown chunk key encoding. Got {name}, expected one of ('v2', 'default')."  # type: ignore[unreachable]
    raise ValueError(msg)


class ChunkKeyEncoding(ABC):
    @abstractmethod
    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        pass

    @abstractmethod
    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        pass


class DefaultChunkKeyEncodingConfig(TypedDict):
    separator: SeparatorLiteral


class DefaultChunkKeyEncodingMetadata(TypedDict):
    name: Literal["default"]
    configuration: NotRequired[DefaultChunkKeyEncodingConfig]


@dataclass(frozen=True, kw_only=True)
class DefaultChunkKeyEncoding(
    ChunkKeyEncoding, NamedConfig[Literal["default"], DefaultChunkKeyEncodingMetadata]
):
    """
    The default Zarr V3 chunk key encoding. Read the specification here:
    https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#chunk-key-encoding
    """

    name: ClassVar[Literal["default"]] = "default"
    separator: SeparatorLiteral = "/"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        if chunk_key == "c":
            return ()
        # map c/0/0 or c.0.0 to (0, 0)
        _, *rest = chunk_key.split(self.separator)
        return tuple(map(int, rest))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.separator.join(map(str, ("c",) + chunk_coords))


class V2ChunkKeyEncodingConfig(TypedDict):
    separator: SeparatorLiteral


class V2ChunkKeyEncodingMetadata(TypedDict):
    name: Literal["v2"]
    configuration: NotRequired[V2ChunkKeyEncodingConfig]


@dataclass(frozen=True, kw_only=True)
class V2ChunkKeyEncoding(ChunkKeyEncoding, NamedConfig[V2ChunkKeyEncodingMetadata]):
    """
    A chunk key encoding for Zarr V2 compatibility. Read the specification here:
    https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#chunk-key-encoding
    """

    name: ClassVar[Literal["v2"]] = "v2"
    separator: SeparatorLiteral = "."

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        return tuple(map(int, chunk_key.split(self.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier


ChunkKeyEncodingLike: TypeAlias = (
    V2ChunkKeyEncodingMetadata
    | DefaultChunkKeyEncodingMetadata
    | V2ChunkKeyEncoding
    | DefaultChunkKeyEncoding
)
