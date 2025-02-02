from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, cast

import numpy as np
import typing_extensions
from crc32c import crc32c

from zarr.abc.codec import BytesBytesCodec
from zarr.registry import register_codec

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


@dataclass(frozen=True)
class Crc32cCodec(BytesBytesCodec):
    name: ClassVar[Literal["crc32c"]] = "crc32c"
    is_fixed_size: ClassVar[Literal[True]] = True

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        data = chunk_bytes.as_numpy_array()
        crc32_bytes = data[-4:]
        inner_bytes = data[:-4]

        # Need to do a manual cast until https://github.com/numpy/numpy/issues/26783 is resolved
        computed_checksum = np.uint32(crc32c(cast(typing_extensions.Buffer, inner_bytes))).tobytes()
        stored_checksum = bytes(crc32_bytes)
        if computed_checksum != stored_checksum:
            raise ValueError(
                f"Stored and computed checksum do not match. Stored: {stored_checksum!r}. Computed: {computed_checksum!r}."
            )
        return chunk_spec.prototype.buffer.from_array_like(inner_bytes)

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        data = chunk_bytes.as_numpy_array()
        # Calculate the checksum and "cast" it to a numpy array
        checksum = np.array([crc32c(cast(typing_extensions.Buffer, data))], dtype=np.uint32)
        # Append the checksum (as bytes) to the data
        return chunk_spec.prototype.buffer.from_array_like(np.append(data, checksum.view("b")))

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length + 4


register_codec("crc32c", Crc32cCodec)
