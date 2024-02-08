from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Optional,
    Type,
)

import numcodecs
import numpy as np
from dataclasses import asdict, dataclass, field, replace
from numcodecs.blosc import Blosc

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import to_thread
from zarr.v3.abc.codec import CodecMetadata
from zarr.v3.types import JSON, BytesLike

if TYPE_CHECKING:
    from zarr.v3.metadata.v3.array import CoreArrayMetadata


BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]
BloscCname = Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"]
# See https://zarr.readthedocs.io/en/stable/tutorial.html#configuring-blosc
numcodecs.blosc.use_threads = False

# todo: parametrize this with an upper and lower bound
def parse_int(data: Any) -> int:
    if not isinstance(data, int):
        msg = f"Expected an integer. Got {data} with type {type(data)} instead."
        raise ValueError(msg)
    return data


def parse_typesize(data: Any) -> int:
    return parse_int(data)


def parse_cname(data: Any) -> BloscCname:
    cnames = ["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"]
    if data not in cnames:
        msg = f"Cname must be one of {cnames}. Got {data} instead."
        raise ValueError(msg)
    return data


def parse_clevel(data: Any) -> int:
    return parse_int(data)


def parse_shuffle(data: Any) -> BloscShuffle:
    shuffles = ["noshuffle", "shuffle", "bitshuffle"]
    if data not in shuffles:
        msg = f"Shuffle must be one of {shuffles}. Got {data} instead."
        raise ValueError(msg)
    return data


@dataclass(frozen=True)
class BloscCodecConfigurationMetadata:
    typesize: int
    cname: BloscCname = "zstd"
    clevel: int = 5
    shuffle: BloscShuffle = "noshuffle"
    blocksize: int = 0

    @classmethod
    def from_json(cls, json_data: Dict[str, JSON]):
        typesize_parsed = parse_typesize(json_data["typesize"])
        cname_parsed = parse_cname(json_data["cname"])
        clevel_parsed = parse_clevel(json_data["clevel"])
        shuffle_parsed = parse_shuffle(json_data["shuffle"])
        blocksize_parsed = parse_blocksize(json_data["blocksize"])
        return cls(
            typesize=typesize_parsed,
            cname=cname_parsed,
            clevel=json_data["clevel"],
            shuffle=json_data["shuffle"],
            blocksize=json_data["blocksize"],
        )


blosc_shuffle_int_to_str: Dict[int, BloscShuffle] = {
    0: "noshuffle",
    1: "shuffle",
    2: "bitshuffle",
}


@dataclass(frozen=True)
class BloscCodecMetadata:
    configuration: BloscCodecConfigurationMetadata
    name: Literal["blosc"] = field(default="blosc", init=False)

    @classmethod
    def from_json(cls, json_data: Dict[str, JSON]):
        return cls(
            configuration=BloscCodecConfigurationMetadata.from_json(json_data["configuration"])
        )


@dataclass(frozen=True)
class BloscCodec(BytesBytesCodec):
    array_metadata: CoreArrayMetadata
    configuration: BloscCodecConfigurationMetadata
    blosc_codec: Blosc
    is_fixed_size: bool = field(default=False, init=False)

    @classmethod
    def from_metadata(
        cls, codec_metadata: CodecMetadata, array_metadata: CoreArrayMetadata
    ) -> BloscCodec:
        assert isinstance(codec_metadata, BloscCodecMetadata)
        configuration = codec_metadata.configuration
        if configuration.typesize == 0:
            configuration = replace(configuration, typesize=array_metadata.data_type.byte_count)
        config_dict = asdict(codec_metadata.configuration)
        config_dict.pop("typesize", None)
        map_shuffle_str_to_int = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
        config_dict["shuffle"] = map_shuffle_str_to_int[config_dict["shuffle"]]
        return cls(
            array_metadata=array_metadata,
            configuration=configuration,
            blosc_codec=Blosc.from_config(config_dict),
        )

    @classmethod
    def get_metadata_class(cls) -> Type[BloscCodecMetadata]:
        return BloscCodecMetadata

    async def decode(
        self,
        chunk_bytes: bytes,
    ) -> BytesLike:
        return await to_thread(self.blosc_codec.decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
    ) -> Optional[BytesLike]:
        chunk_array = np.frombuffer(chunk_bytes, dtype=self.array_metadata.dtype)
        return await to_thread(self.blosc_codec.encode, chunk_array)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


register_codec("blosc", BloscCodec)
