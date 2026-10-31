"""NumPy encodings that both Zarr formats use in fill values: numpy time ticks and `NaT`, and base64 byte strings.

A v3 data type and a v2 dtype of the same NumPy family write their fill
values the same way, so each encoding is defined once here.
"""

import re
from collections.abc import Iterator
from typing import Annotated, Final, Literal, NewType

from annotated_types import Interval

from zarr_metadata._json import ValidationProblem, shown

NUMPY_TIME_UNITS: Final = (
    "Y",
    "M",
    "W",
    "D",
    "h",
    "m",
    "s",
    "ms",
    "us",
    "μs",
    "ns",
    "ps",
    "fs",
    "as",
)
"""The units NumPy accepts for a datetime64 or timedelta64 (https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units). The v3 types also accept `"generic"`."""

NUMPY_TIME_MAX_SCALE_FACTOR: Final = 2**31 - 1
"""The largest `scale_factor` that numpy stores, because the field is a signed int32."""

NumpyTimeScaleFactor = Annotated[int, Interval(ge=1, le=NUMPY_TIME_MAX_SCALE_FACTOR)]
"""The number of units in one tick: a positive int32."""

NumpyTimeTicks = Annotated[int, Interval(ge=-(2**63), le=2**63 - 1)]
"""A count of ticks, as a fill value writes it: a signed 64-bit integer.

https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L109-L112
"""

NOT_A_TIME_TICKS: Final = -(2**63)
"""The tick count that stores `NaT`. A fill value may write this count for `NaT`.

"`"fill_value": "NaT"` and `"fill_value": -9223372036854775808` should
be treated as equivalent representations of the same scalar value"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L114-L116).
The `numpy.timedelta64` spec says the same
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.timedelta64/README.md?plain=1#L117-L119).
"""


def numpy_time_fill_value_canonical(
    configuration: object, nested: object, value: NumpyTimeTicks | Literal["NaT"]
) -> NumpyTimeTicks | Literal["NaT"]:
    """Return the canonical spelling of a numpy time fill value: `"NaT"` for `NaT` however it is written, and any other count of ticks as written."""
    return "NaT" if value in ("NaT", NOT_A_TIME_TICKS) else value


Base64Bytes = NewType("Base64Bytes", str)
"""A standard-alphabet base64-encoded byte sequence."""

_BASE64_RE: Final = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")


def base64_bytes(value: str) -> Base64Bytes:
    """Validate `value` as a Base64Bytes and brand it.

    Raises ValueError if `value` is not standard-alphabet base64
    (length must be a multiple of 4 once padded; only `A-Z`, `a-z`,
    `0-9`, `+`, `/`, and trailing `=` padding are permitted).
    """
    if len(value) % 4 != 0 or not _BASE64_RE.fullmatch(value):
        raise ValueError(f"Expected standard-alphabet base64, got {value!r}")
    return Base64Bytes(value)


def base64_fill_value_rules(
    configuration: object, nested: object, value: object
) -> Iterator[ValidationProblem]:
    """Check that a fill value written as a string is standard-alphabet base64: a data type's `fill_value_rules`.

    `Base64Bytes` is a `NewType` of `str`, so a fill value annotated with
    it accepts any string. Pass this rule as the data type's
    `fill_value_rules` to reject a string that is not base64. A value of
    any other shape is not checked here, so the rule also serves a fill
    value that may be written another way, such as
    `tuple[ByteValue, ...] | Base64Bytes`. A rule that checks more, such
    as the number of bytes, yields from this one first.
    """
    if not isinstance(value, str):
        return
    try:
        base64_bytes(value)
    except ValueError:
        yield ValidationProblem(
            (), f"expected standard-alphabet base64, got {shown(value)}", "invalid_value"
        )


__all__ = [
    "NOT_A_TIME_TICKS",
    "NUMPY_TIME_MAX_SCALE_FACTOR",
    "NUMPY_TIME_UNITS",
    "Base64Bytes",
    "NumpyTimeScaleFactor",
    "NumpyTimeTicks",
    "base64_bytes",
    "base64_fill_value_rules",
    "numpy_time_fill_value_canonical",
]
