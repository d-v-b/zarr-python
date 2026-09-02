"""Checked affine coordinate arithmetic."""

from __future__ import annotations

from typing import Any, overload

import numpy as np
import numpy.typing as npt

_INTP_INFO = np.iinfo(np.intp)


def _fits_intp(value: int) -> bool:
    return _INTP_INFO.min <= value <= _INTP_INFO.max


_DTYPE_LIMITS: dict[np.dtype[Any], tuple[bool, int]] = {}


def _dtype_limits(dtype: np.dtype[Any]) -> tuple[bool, int]:
    """Whether every value of an integer ``dtype`` fits ``np.intp``, and the dtype's largest magnitude."""
    limits = _DTYPE_LIMITS.get(dtype)
    if limits is None:
        info = np.iinfo(dtype)
        fits = int(info.min) >= _INTP_INFO.min and int(info.max) <= _INTP_INFO.max
        limits = (fits, max(-int(info.min), int(info.max)))
        _DTYPE_LIMITS[dtype] = limits
    return limits


@overload
def checked_affine(offset: int, stride: int, coordinates: int) -> int: ...


@overload
def checked_affine(
    offset: int,
    stride: int,
    coordinates: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.intp]: ...


def checked_affine(
    offset: int,
    stride: int,
    coordinates: int | npt.NDArray[np.integer[Any]],
) -> int | npt.NDArray[np.intp]:
    """Evaluate ``offset + stride * coordinates`` without integer overflow.

    Bounds are established with Python integers before coordinates are cast or
    NumPy performs fixed-width arithmetic. The common representable case then
    uses an ``np.intp`` fast path whose multiplication and addition were proven
    safe; cancellation cases use exact object arithmetic.

    Two shortcuts avoid scanning the array at all. The identity affine
    (``offset == 0 and stride == 1``) of an array whose dtype fits ``np.intp``
    cannot overflow, so it is returned as-is (re-typed). And when the dtype's
    own bounds already prove ``offset + stride * value`` representable for
    every value the dtype can hold, the fixed-width path is taken directly.
    Chunk resolution builds many small maps, where the scan's cost is the
    call overhead rather than the elements.
    """
    offset = int(offset)
    stride = int(stride)
    if not isinstance(coordinates, np.ndarray):
        mapped = offset + stride * int(coordinates)
        if not _fits_intp(mapped):
            raise OverflowError(f"output coordinate {mapped} is outside np.intp range")
        return mapped

    if coordinates.size == 0:
        return np.empty(coordinates.shape, dtype=np.intp)

    dtype_fits, dtype_bound = _dtype_limits(coordinates.dtype)
    if offset == 0 and stride == 1:
        if dtype_fits:
            return np.asarray(coordinates, dtype=np.intp)
    elif abs(offset) + abs(stride) * dtype_bound <= _INTP_INFO.max:
        intp_coordinates = coordinates.astype(np.intp, copy=False)
        return np.asarray(offset + stride * intp_coordinates, dtype=np.intp)

    coordinate_min = int(coordinates.min())
    coordinate_max = int(coordinates.max())
    product_at_min = stride * coordinate_min
    product_at_max = stride * coordinate_max
    mapped_at_min = offset + product_at_min
    mapped_at_max = offset + product_at_max
    mapped_min = min(mapped_at_min, mapped_at_max)
    mapped_max = max(mapped_at_min, mapped_at_max)
    if not _fits_intp(mapped_min) or not _fits_intp(mapped_max):
        invalid = mapped_min if not _fits_intp(mapped_min) else mapped_max
        raise OverflowError(f"output coordinate {invalid} is outside np.intp range")

    safe_fixed_width = (
        _fits_intp(coordinate_min)
        and _fits_intp(coordinate_max)
        and _fits_intp(offset)
        and _fits_intp(stride)
        and _fits_intp(product_at_min)
        and _fits_intp(product_at_max)
    )
    if safe_fixed_width:
        intp_coordinates = coordinates.astype(np.intp, copy=False)
        return np.asarray(offset + stride * intp_coordinates, dtype=np.intp)

    exact = offset + stride * coordinates.astype(object)
    return np.asarray(exact, dtype=np.intp)
