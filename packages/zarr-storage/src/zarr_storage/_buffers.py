"""Host-memory buffers: the byte views stores take values as, and copying a read into one."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_storage._types import Copied, Found, Missing

if TYPE_CHECKING:
    from typing_extensions import Buffer

__all__ = ["byte_view", "copy_into"]


def byte_view(value: Buffer, *, writable: bool) -> memoryview:
    """A flat, C-contiguous byte view of a buffer.

    Raises `ValueError` if the buffer is not C-contiguous, or if `writable` and it is read-only.
    """
    view = memoryview(value)
    if not view.c_contiguous:
        raise ValueError("The buffer must be C-contiguous.")
    if writable and view.readonly:
        raise ValueError("The output buffer must be writable.")
    return view.cast("B") if view.format != "B" or view.ndim != 1 else view


def copy_into(result: Found | Missing, out: memoryview) -> Copied | Missing:
    """The `get_into` outcome of a `get`, copying a found value into `out`.

    For a backend that can't read into a buffer directly.
    """
    match result:
        case Found(value=value, generation=generation, read_at=read_at):
            n = len(value)
            if n > len(out):
                raise ValueError(f"The output buffer holds {len(out)} bytes; the value has {n}.")
            out[:n] = value
            return Copied(n, generation, read_at)
        case Missing():
            return result
