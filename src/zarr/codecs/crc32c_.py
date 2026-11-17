from __future__ import annotations

from collections.abc import Buffer as ABCBuffer
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import google_crc32c
import numpy as np
import numpy.typing as npt

from zarr.abc.codec import BytesBytesCodec
from zarr.core.common import JSON, parse_named_configuration

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


def _checksum_bytes(data: npt.NDArray[Any]) -> bytes:
    """
    Return the CRC32C checksum of ``data`` as the 4 bytes the crc32c codec stores.

    The crc32c codec specification stores the checksum as a little-endian
    unsigned 32-bit integer, independent of the byte order of the host.
    """
    checksum: int = google_crc32c.value(cast(ABCBuffer, data))
    return checksum.to_bytes(4, "little")


@dataclass(frozen=True)
class Crc32cCodec(BytesBytesCodec):
    """crc32c codec"""

    is_fixed_size = True

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        parse_named_configuration(data, "crc32c", require_configuration=False)
        return cls()

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "crc32c"}

    def _decode_sync(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        data = chunk_bytes.as_numpy_array()
        crc32_bytes = data[-4:]
        inner_bytes = data[:-4]

        computed_checksum = _checksum_bytes(inner_bytes)
        stored_checksum = bytes(crc32_bytes)
        if computed_checksum != stored_checksum:
            raise ValueError(
                f"Stored and computed checksum do not match. Stored: {stored_checksum!r}. Computed: {computed_checksum!r}."
            )
        return chunk_spec.prototype.buffer.from_array_like(inner_bytes)

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return self._decode_sync(chunk_bytes, chunk_spec)

    def _encode_sync(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        data = chunk_bytes.as_numpy_array()
        checksum = np.frombuffer(_checksum_bytes(data), dtype=np.uint8)
        return chunk_spec.prototype.buffer.from_array_like(np.append(data, checksum))

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return self._encode_sync(chunk_bytes, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length + 4
