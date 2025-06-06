from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    TypedDict,
    cast,
    overload,
)

import numcodecs
from numcodecs.blosc import Blosc
from packaging.version import Version

from zarr.abc.codec import BytesBytesCodec
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import (
    JSON,
    JSON2,
    NamedConfig,
    ZarrFormat,
    parse_enum,
    parse_named_configuration,
)
from zarr.registry import register_codec

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer

_BloscCname = Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"]
BLOSC_CNAME: Final = ("lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib")

_BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]
BLOSC_SHUFFLE: Final = ("noshuffle", "shuffle", "bitshuffle")


class BloscMetaV2(TypedDict):
    id: Literal["blosc"]
    cname: str
    clevel: int
    shuffle: _BloscShuffle | None


def parse_bloscmetav2(data: JSON2) -> BloscMetaV2:
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected a dict, got {data!r} which has type {type(data)}")

    expect_keys = {"id", "cname", "clevel", "shuffle"}

    if (obs_keys := set(data.keys())) != expect_keys:
        msg = f"Object {data} has invalid keys. "
        missing = expect_keys - obs_keys
        extra = obs_keys - expect_keys
        if len(missing) > 0:
            msg += f"These keys were missing: {missing}. "
        if len(extra) > 0:
            msg += f"These extra keys were found: {extra}"
        raise ValueError(msg)

    if _id := data["id"] != "blosc":
        raise ValueError(f"ID field must be blosc, got {_id}")

    if data["cname"] not in BLOSC_CNAME:
        raise ValueError(f'value for "cname" is not one of {BLOSC_CNAME}')

    if data["shuffle"] not in BLOSC_SHUFFLE:
        raise ValueError(f'value for "cname" is not one of {BLOSC_CNAME}')

    if not isinstance(data["clevel"], int):
        raise ValueError("clevel is not an int.")

    return cast(BloscMetaV2, data)


def parse_bloscmetav3(data: JSON2) -> BloscMetaV2:
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected a dict, got {data!r} which has type {type(data)}")
    if _id := data["id"] != "blosc":
        raise ValueError(f"ID field must be blosc, got {_id}")
    if "cname" not in data:
        raise ValueError(f'Missing required key "cname" from {data}')
    if data["cname"] not in BLOSC_CNAME:
        raise ValueError(f'value for "cname" is not one of {BLOSC_CNAME}')
    if "shuffle" not in data:
        raise ValueError(f'Missing required key "shuffle" from {data}')
    if data["shuffle"] not in BLOSC_SHUFFLE:
        raise ValueError(f'value for "cname" is not one of {BLOSC_CNAME}')
    return cast(BloscMetaV2, data)


class BloscConfigV3(TypedDict):
    cname: _BloscCname
    clevel: int
    shuffle: _BloscShuffle | None
    blocksize: int


BloscMetaV3 = NamedConfig[Literal["blosc"], BloscConfigV3]


class BloscShuffle(Enum):
    """
    Enum for shuffle filter used by blosc.
    """

    noshuffle = "noshuffle"
    shuffle = "shuffle"
    bitshuffle = "bitshuffle"

    @classmethod
    def from_int(cls, num: int) -> BloscShuffle:
        blosc_shuffle_int_to_str = {
            0: "noshuffle",
            1: "shuffle",
            2: "bitshuffle",
        }
        if num not in blosc_shuffle_int_to_str:
            raise ValueError(f"Value must be between 0 and 2. Got {num}.")
        return BloscShuffle[blosc_shuffle_int_to_str[num]]


class BloscCname(Enum):
    """
    Enum for compression library used by blosc.
    """

    lz4 = "lz4"
    lz4hc = "lz4hc"
    blosclz = "blosclz"
    zstd = "zstd"
    snappy = "snappy"
    zlib = "zlib"


# See https://zarr.readthedocs.io/en/stable/user-guide/performance.html#configuring-blosc
numcodecs.blosc.use_threads = False


def parse_typesize(data: JSON) -> int:
    if isinstance(data, int):
        if data > 0:
            return data
        else:
            raise ValueError(
                f"Value must be greater than 0. Got {data}, which is less or equal to 0."
            )
    raise TypeError(f"Value must be an int. Got {type(data)} instead.")


# todo: real validation
def parse_clevel(data: JSON) -> int:
    if isinstance(data, int):
        return data
    raise TypeError(f"Value should be an int. Got {type(data)} instead.")


def parse_blocksize(data: JSON) -> int:
    if isinstance(data, int):
        return data
    raise TypeError(f"Value should be an int. Got {type(data)} instead.")


@dataclass(frozen=True)
class BloscCodec(BytesBytesCodec):
    is_fixed_size = False

    typesize: int
    cname: BloscCname
    clevel: int
    shuffle: BloscShuffle | None
    blocksize: int

    def __init__(
        self,
        *,
        typesize: int | None = None,
        cname: BloscCname | str = BloscCname.zstd,
        clevel: int = 5,
        shuffle: BloscShuffle | str | None = None,
        blocksize: int = 0,
    ) -> None:
        typesize_parsed = parse_typesize(typesize) if typesize is not None else None
        cname_parsed = parse_enum(cname, BloscCname)
        clevel_parsed = parse_clevel(clevel)
        shuffle_parsed = parse_enum(shuffle, BloscShuffle) if shuffle is not None else None
        blocksize_parsed = parse_blocksize(blocksize)

        object.__setattr__(self, "typesize", typesize_parsed)
        object.__setattr__(self, "cname", cname_parsed)
        object.__setattr__(self, "clevel", clevel_parsed)
        object.__setattr__(self, "shuffle", shuffle_parsed)
        object.__setattr__(self, "blocksize", blocksize_parsed)

    @overload
    def to_json(self, *, zarr_format: Literal[2]) -> BloscMetaV2: ...

    @overload
    def to_json(self, *, zarr_format: Literal[3]) -> BloscMetaV3: ...

    def to_json(self, *, zarr_format: ZarrFormat) -> BloscMetaV2 | BloscMetaV3:
        if zarr_format == 2:
            return {
                "id": "blosc",
                "cname": self.cname.value,
                "clevel": self.clevel,
                "shuffle": self.shuffle.value,
            }
        elif zarr_format == 3:
            return {
                "name": "blosc",
                "configuration": {
                    "cname": self.cname.value,
                    "clevel": self.clevel,
                    "shuffle": self.shuffle.value,
                    "blocksize": self.blocksize,
                },
            }
        else:
            raise ValueError(f"Unsupported zarr format: {zarr_format}")  # pragma: no cover

    @classmethod
    def from_json(cls, data: JSON2): ...

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "blosc")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        if self.typesize is None:
            raise ValueError("`typesize` needs to be set for serialization.")
        if self.shuffle is None:
            raise ValueError("`shuffle` needs to be set for serialization.")
        return {
            "name": "blosc",
            "configuration": {
                "typesize": self.typesize,
                "cname": self.cname.value,
                "clevel": self.clevel,
                "shuffle": self.shuffle.value,
                "blocksize": self.blocksize,
            },
        }

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        dtype = array_spec.dtype
        new_codec = self
        if new_codec.typesize is None:
            new_codec = replace(new_codec, typesize=dtype.itemsize)
        if new_codec.shuffle is None:
            new_codec = replace(
                new_codec,
                shuffle=(BloscShuffle.bitshuffle if dtype.itemsize == 1 else BloscShuffle.shuffle),
            )

        return new_codec

    @cached_property
    def _blosc_codec(self) -> Blosc:
        if self.shuffle is None:
            raise ValueError("`shuffle` needs to be set for decoding and encoding.")
        map_shuffle_str_to_int = {
            BloscShuffle.noshuffle: 0,
            BloscShuffle.shuffle: 1,
            BloscShuffle.bitshuffle: 2,
        }
        config_dict = {
            "cname": self.cname.name,
            "clevel": self.clevel,
            "shuffle": map_shuffle_str_to_int[self.shuffle],
            "blocksize": self.blocksize,
        }
        # See https://github.com/zarr-developers/numcodecs/pull/713
        if Version(numcodecs.__version__) >= Version("0.16.0"):
            config_dict["typesize"] = self.typesize
        return Blosc.from_config(config_dict)

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return await asyncio.to_thread(
            as_numpy_array_wrapper, self._blosc_codec.decode, chunk_bytes, chunk_spec.prototype
        )

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        # Since blosc only support host memory, we convert the input and output of the encoding
        # between numpy array and buffer
        return await asyncio.to_thread(
            lambda chunk: chunk_spec.prototype.buffer.from_bytes(
                self._blosc_codec.encode(chunk.as_numpy_array())
            ),
            chunk_bytes,
        )

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


register_codec("blosc", BloscCodec)
