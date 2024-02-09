from __future__ import annotations
from dataclasses import dataclass, field

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Optional,
    Type,
)

import numpy as np

from crc32c import crc32c

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.abc.metadata import Metadata
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import NamedConfig

if TYPE_CHECKING:
    from zarr.v3.common import BytesLike, RuntimeConfiguration, ArraySpec


def parse_name(data: Any) -> Literal["crc32c"]:
    if data == "crc32c":
        return data
    msg = f"Expected 'crc32c', got {data} instead."
    raise ValueError(msg)


@dataclass(frozen=True)
class Crc32cCodecMetadata(Metadata):
    name: Literal["crc32c"] = field(default="crc32c", init=False)

    @classmethod
    def from_dict(cls, data: Any):
        _ = parse_name(data.pop("name"))
        return cls(**data)


@dataclass(frozen=True)
class Crc32cCodec(BytesBytesCodec):
    is_fixed_size = True

    @classmethod
    def from_metadata(cls, codec_metadata: NamedConfig) -> Crc32cCodec:
        assert isinstance(codec_metadata, Crc32cCodecMetadata)
        return cls()

    @classmethod
    def get_metadata_class(cls) -> Type[Crc32cCodecMetadata]:
        return Crc32cCodecMetadata

    async def decode(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        crc32_bytes = chunk_bytes[-4:]
        inner_bytes = chunk_bytes[:-4]

        assert np.uint32(crc32c(inner_bytes)).tobytes() == bytes(crc32_bytes)
        return inner_bytes

    async def encode(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        return chunk_bytes + np.uint32(crc32c(chunk_bytes)).tobytes()

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length + 4

    def to_dict(self) -> Dict[str, Any]:
        return Crc32cCodecMetadata()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(configuration=data["configuration"])


register_codec("crc32c", Crc32cCodec)
