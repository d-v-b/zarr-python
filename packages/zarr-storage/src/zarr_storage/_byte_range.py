"""Resolving a byte range against the size of a value."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_storage._errors import InvalidByteRangeError
from zarr_storage._types import OffsetRequest, RangeRequest, SuffixRequest

if TYPE_CHECKING:
    from zarr_storage._types import ByteRange

__all__ = ["resolve"]


def resolve(byte_range: ByteRange | None, size: int) -> tuple[int, int]:
    """The `(start, end)` a byte range covers in a value of `size` bytes.

    An end past the value is clamped to `size`. A start past the value raises
    `InvalidByteRangeError`; a start equal to `size` selects nothing.
    """
    match byte_range:
        case None:
            return 0, size
        case RangeRequest(start=start, end=end):
            if start > size:
                raise InvalidByteRangeError(
                    f"The range starts at byte {start}, past the end of a {size}-byte value."
                )
            return start, min(end, size)
        case OffsetRequest(start=start):
            if start > size:
                raise InvalidByteRangeError(
                    f"The range starts at byte {start}, past the end of a {size}-byte value."
                )
            return start, size
        case SuffixRequest(length=length):
            return max(size - length, 0), size
