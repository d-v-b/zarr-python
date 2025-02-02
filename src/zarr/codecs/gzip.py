from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from numcodecs.gzip import GZip

from zarr.abc.codec import BytesBytesCodec
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON
from zarr.registry import register_codec

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


def parse_gzip_level(data: JSON) -> int:
    if not isinstance(data, (int)):
        raise TypeError(f"Expected int, got {type(data)}")
    if data not in range(10):
        raise ValueError(
            f"Expected an integer from the inclusive range (0, 9). Got {data} instead."
        )
    return data


@dataclass(frozen=True)
class GzipCodec(BytesBytesCodec):
    is_fixed_size: ClassVar[Literal[False]] = False
    name: ClassVar[Literal["gzip"]] = "gzip"
    level: int = 5

    def __init__(self, *, level: int = 5) -> None:
        level_parsed = parse_gzip_level(level)
        object.__setattr__(self, "level", level_parsed)

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return await asyncio.to_thread(
            as_numpy_array_wrapper, GZip(self.level).decode, chunk_bytes, chunk_spec.prototype
        )

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return await asyncio.to_thread(
            as_numpy_array_wrapper, GZip(self.level).encode, chunk_bytes, chunk_spec.prototype
        )

    def compute_encoded_size(
        self,
        _input_byte_length: int,
        _chunk_spec: ArraySpec,
    ) -> int:
        raise NotImplementedError


register_codec("gzip", GzipCodec)
