from __future__ import annotations
from dataclasses import dataclass, field

from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    Optional,
    Type,
)

from numcodecs.gzip import GZip

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import to_thread
from zarr.v3.metadata.v3.array import CodecMetadata
from zarr.v3.types import JSON, BytesLike

if TYPE_CHECKING:
    from zarr.v3.metadata.v3.array import CoreArrayMetadata


@dataclass(frozen=True)
class GzipCodecConfigurationMetadata:
    level: int = 5

    @classmethod
    def from_json(cls, json_data: Dict[str, JSON]):
        return cls(level=json_data['level'])

@dataclass(frozen=True)
class GzipCodecMetadata:
    configuration: GzipCodecConfigurationMetadata
    name: Literal["gzip"] = field(default="gzip", init=False)

    @classmethod
    def from_json(cls, json_data: Dict[str, JSON]):
        return cls(configuration=GzipCodecConfigurationMetadata.from_json(json_data['configuration']))

@dataclass(frozen=True)
class GzipCodec(BytesBytesCodec):
    array_metadata: CoreArrayMetadata
    configuration: GzipCodecConfigurationMetadata
    is_fixed_size = field(default=True, init=False)

    @classmethod
    def from_metadata(
        cls, codec_metadata: CodecMetadata, array_metadata: CoreArrayMetadata
    ) -> GzipCodec:
        assert isinstance(codec_metadata, GzipCodecMetadata)

        return cls(
            array_metadata=array_metadata,
            configuration=codec_metadata.configuration,
        )

    @classmethod
    def get_metadata_class(cls) -> Type[GzipCodecMetadata]:
        return GzipCodecMetadata

    async def decode(
        self,
        chunk_bytes: bytes,
    ) -> BytesLike:
        return await to_thread(GZip(self.configuration.level).decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
    ) -> Optional[BytesLike]:
        return await to_thread(GZip(self.configuration.level).encode, chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


register_codec("gzip", GzipCodec)
