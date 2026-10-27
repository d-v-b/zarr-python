"""Strategies for each data type this package defines.

A fill value is drawn in every spelling the data type's spec gives it:
a float's as a number, a named non-finite value or a hex string of its
bits, a `bytes` value's as an array of byte values or base64. A struct's
field types are drawn from the strategies, and its fill value is one
for each field, drawn by that field's own type.
"""

from __future__ import annotations

import base64
from functools import partial
from typing import TYPE_CHECKING, Any, Final, Literal, cast, get_args

from hypothesis import strategies as st

from zarr_metadata.testing._draw import (
    briefly,
    data_type_fields,
    data_types,
    fill_values_of,
    read,
)
from zarr_metadata.testing._strategy import DataTypeStrategy
from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration, storage_of
from zarr_metadata.v3.data_type._float import FloatSpecialFillValue
from zarr_metadata.v3.data_type._numpy_time import NUMPY_TIME_MAX_SCALE_FACTOR
from zarr_metadata.v3.data_type.bool import BOOL_DATA_TYPE
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.complex64 import COMPLEX64_DATA_TYPE
from zarr_metadata.v3.data_type.complex128 import COMPLEX128_DATA_TYPE
from zarr_metadata.v3.data_type.float16 import FLOAT16_DATA_TYPE
from zarr_metadata.v3.data_type.float32 import FLOAT32_DATA_TYPE
from zarr_metadata.v3.data_type.float64 import FLOAT64_DATA_TYPE
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE
from zarr_metadata.v3.data_type.int16 import INT16_DATA_TYPE
from zarr_metadata.v3.data_type.int32 import INT32_DATA_TYPE
from zarr_metadata.v3.data_type.int64 import INT64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_datetime64 import NUMPY_DATETIME64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_timedelta64 import (
    NUMPY_TIME_UNIT,
    NUMPY_TIMEDELTA64_DATA_TYPE,
)
from zarr_metadata.v3.data_type.raw import RAW_BYTES_DATA_TYPE, RawBytesConfiguration
from zarr_metadata.v3.data_type.string import STRING_DATA_TYPE
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE, StructConfiguration
from zarr_metadata.v3.data_type.uint8 import UINT8_DATA_TYPE
from zarr_metadata.v3.data_type.uint16 import UINT16_DATA_TYPE
from zarr_metadata.v3.data_type.uint32 import UINT32_DATA_TYPE
from zarr_metadata.v3.data_type.uint64 import UINT64_DATA_TYPE

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata._common import JSONValue
    from zarr_metadata.testing._strategy import Configurations, Strategies
    from zarr_metadata.v3.definition import ZarrV3MetadataFieldJSON

FloatWidth = Literal[16, 32, 64]

INTEGER_RANGES: Final[Mapping[DataTypeDefinition[Any], tuple[int, int]]] = {
    INT8_DATA_TYPE: (-(2**7), 2**7 - 1),
    INT16_DATA_TYPE: (-(2**15), 2**15 - 1),
    INT32_DATA_TYPE: (-(2**31), 2**31 - 1),
    INT64_DATA_TYPE: (-(2**63), 2**63 - 1),
    UINT8_DATA_TYPE: (0, 2**8 - 1),
    UINT16_DATA_TYPE: (0, 2**16 - 1),
    UINT32_DATA_TYPE: (0, 2**32 - 1),
    UINT64_DATA_TYPE: (0, 2**64 - 1),
}
"""The values each integer data type holds."""

FLOAT_WIDTHS: Final[Mapping[DataTypeDefinition[Any], FloatWidth]] = {
    FLOAT16_DATA_TYPE: 16,
    FLOAT32_DATA_TYPE: 32,
    FLOAT64_DATA_TYPE: 64,
}
"""The width in bits of each floating-point data type."""

_EMPTY: Final = st.just(EmptyConfiguration())

_MAX_RAW_BYTES: Final = 32
"""The widest raw bits drawn, in bytes: `r256`."""

_MAX_BYTES: Final = 16
"""The longest `bytes` fill value drawn."""

_MAX_STRUCT_FIELDS: Final = 4
"""The most fields a struct is drawn with."""

_MAX_STRUCT_DEPTH: Final = 1
"""How deep structs are drawn inside a struct: one inside another, and none inside that."""


def finite_floats(width: FloatWidth) -> st.SearchStrategy[float]:
    """Finite numbers a float of `width` bits holds exactly."""
    return st.floats(width=width, allow_nan=False, allow_infinity=False)


def _hex_spellings(digits: str) -> st.SearchStrategy[str]:
    """`0x` and the digits, the letters in either case."""
    return st.sampled_from((f"0x{digits}", f"0x{digits.upper()}"))


def _float_fill_values(width: FloatWidth) -> st.SearchStrategy[JSONValue]:
    """A number, an integer the type holds exactly, a named non-finite value, or the bits in hex.

    A float's fill value is any of these
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L63-L79).
    Numbers are drawn from those the type holds, so none is rounded: a
    `float16` takes `1e300` by the letter of the spec, and a reader is
    free to make of it what it will.
    """
    # Every integer of magnitude up to 2**11 is a float16, the narrowest type.
    numbers = finite_floats(width) | st.integers(-(2**11), 2**11)
    named = st.sampled_from(get_args(FloatSpecialFillValue))
    bits = st.integers(0, 2**width - 1).flatmap(
        lambda value: _hex_spellings(f"{value:0{width // 4}x}")
    )
    return st.one_of(numbers, named, bits)


def _fill_values_of(values: st.SearchStrategy[JSONValue]) -> DataTypeStrategy:
    """A data type with nothing to configure, whose fill values are `values`."""
    return DataTypeStrategy(
        configurations=lambda _strategies: _EMPTY,
        fill_values=lambda _configuration, _strategies: values,
    )


def _byte_values(size: int) -> st.SearchStrategy[JSONValue]:
    """`size` byte values, each an integer in `[0, 255]`, as a JSON array."""
    return st.lists(st.integers(0, 255), min_size=size, max_size=size).map(
        lambda values: cast("JSONValue", tuple(values))
    )


def _raw_bytes_fill_values(
    configuration: Mapping[str, JSONValue], _strategies: Strategies
) -> st.SearchStrategy[JSONValue]:
    """One byte value for each 8 bits of the size."""
    return _byte_values(cast("RawBytesConfiguration", configuration)["bits"] // 8)


def _base64(value: bytes) -> JSONValue:
    return base64.b64encode(value).decode("ascii")


def _complex_fill_values(width: FloatWidth) -> st.SearchStrategy[JSONValue]:
    """A pair of float fill values of `width` bits, real then imaginary."""
    component = _float_fill_values(width)
    return st.tuples(component, component).map(lambda pair: cast("JSONValue", pair))


_BYTES_FILL_VALUES: Final = st.integers(0, _MAX_BYTES).flatmap(_byte_values) | st.binary(
    max_size=_MAX_BYTES
).map(_base64)
"""A `bytes` fill value: byte values, or the bytes in standard-alphabet base64."""

_CHARACTERS: Final = st.characters(exclude_categories=("Cs",))
"""Characters a JSON document can hold: no lone surrogate, which is no Unicode scalar value."""

_TEXT: Final = st.text(_CHARACTERS, max_size=16)

_NUMPY_TIME: Final = DataTypeStrategy(
    configurations=lambda _strategies: st.fixed_dictionaries(
        {
            "unit": st.sampled_from(NUMPY_TIME_UNIT),
            "scale_factor": st.integers(1, NUMPY_TIME_MAX_SCALE_FACTOR),
        }
    ),
    fill_values=lambda _configuration, _strategies: (
        st.integers(-(2**63), 2**63 - 1) | st.just("NaT")
    ),
)
"""Both numpy time types: a unit, and how many of it one tick is; a signed 64-bit count of ticks, or `"NaT"`."""


def _fixed_size(strategies: Strategies) -> st.SearchStrategy[ZarrV3MetadataFieldJSON]:
    """A data type field whose values are of a fixed size, which a struct's field types must be.

    "Variable-length data types (e.g. "string") MUST NOT be used as field
    types"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/struct/README.md?plain=1#L42-L49).
    """
    return data_type_fields(data_types(strategies), strategies).filter(
        lambda field: storage_of(read(field, DataTypeDefinition, strategies)) != "variable_length"
    )


def _struct(strategies: Strategies, *, depth: int) -> Configurations:
    """Fields with distinct names, none empty, each of a data type of fixed size; structs among them `depth` deep at most.

    The field types are drawn from the same strategies but for a struct,
    which holds one level fewer, or is not drawn: structs drawn without
    end would nest without end.
    """
    inner = {d: s for d, s in strategies.items() if d is not STRUCT_DATA_TYPE}
    if depth > 0:
        inner[STRUCT_DATA_TYPE] = _structs(depth - 1)
    member = st.fixed_dictionaries(
        {
            "name": st.text(_CHARACTERS, min_size=1, max_size=16),
            "data_type": _fixed_size(briefly(inner)),
        }
    )
    members = st.lists(
        member,
        min_size=1,
        max_size=_MAX_STRUCT_FIELDS,
        unique_by=lambda entry: cast("str", entry["name"]),
    )
    return members.map(lambda entries: {"fields": tuple(entries)})


def _struct_fill_values(
    configuration: Mapping[str, JSONValue], strategies: Strategies
) -> st.SearchStrategy[JSONValue]:
    """A fill value for each field, by its name, drawn by the field's own data type."""
    members = cast("StructConfiguration", configuration)["fields"]
    return st.fixed_dictionaries(
        {
            member["name"]: fill_values_of(
                read(member["data_type"], DataTypeDefinition, strategies), strategies
            )
            for member in members
        }
    ).map(lambda value: cast("JSONValue", value))


def _structs(depth: int) -> DataTypeStrategy:
    """Structs that hold structs `depth` deep at most."""
    return DataTypeStrategy(
        configurations=partial(_struct, depth=depth), fill_values=_struct_fill_values
    )


DATA_TYPE_STRATEGIES: Final[Mapping[DataTypeDefinition[Any], DataTypeStrategy]] = {
    BOOL_DATA_TYPE: _fill_values_of(st.booleans()),
    **{
        definition: _fill_values_of(st.integers(low, high))
        for definition, (low, high) in INTEGER_RANGES.items()
    },
    **{
        definition: _fill_values_of(_float_fill_values(width))
        for definition, width in FLOAT_WIDTHS.items()
    },
    COMPLEX64_DATA_TYPE: _fill_values_of(_complex_fill_values(32)),
    COMPLEX128_DATA_TYPE: _fill_values_of(_complex_fill_values(64)),
    RAW_BYTES_DATA_TYPE: DataTypeStrategy(
        configurations=lambda _strategies: st.integers(1, _MAX_RAW_BYTES).map(
            lambda size: RawBytesConfiguration(bits=8 * size)
        ),
        fill_values=_raw_bytes_fill_values,
    ),
    BYTES_DATA_TYPE: _fill_values_of(_BYTES_FILL_VALUES),
    STRING_DATA_TYPE: _fill_values_of(_TEXT),
    NUMPY_DATETIME64_DATA_TYPE: _NUMPY_TIME,
    NUMPY_TIMEDELTA64_DATA_TYPE: _NUMPY_TIME,
    STRUCT_DATA_TYPE: _structs(_MAX_STRUCT_DEPTH),
}
"""A strategy for each data type."""

__all__ = ["DATA_TYPE_STRATEGIES", "FLOAT_WIDTHS", "INTEGER_RANGES", "finite_floats"]
