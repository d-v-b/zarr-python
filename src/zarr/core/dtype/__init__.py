from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final, cast

from zarr.core.dtype.common import _UNSTABLE_WARNING_SILENCED, HasEndianness
from zarr.core.dtype.npy.bool import Bool
from zarr.core.dtype.npy.bytes import (
    NullTerminatedBytes,
    NullterminatedBytesJSON_V2,
    NullTerminatedBytesJSON_V3,
    RawBytes,
    RawBytesJSON_V2,
    RawBytesJSON_V3,
    VariableLengthBytes,
    VariableLengthBytesJSON_V2,
)
from zarr.core.dtype.npy.complex import Complex64, Complex128
from zarr.core.dtype.npy.float import Float16, Float32, Float64
from zarr.core.dtype.npy.int import Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
from zarr.core.dtype.npy.structured import (
    Struct,
    StructJSON_V3,
    Structured,
    StructuredJSON_V2,
    StructuredJSON_V3,
)
from zarr.core.dtype.npy.time import (
    DateTime64,
    DateTime64JSON_V2,
    DateTime64JSON_V3,
    TimeDelta64,
    TimeDelta64JSON_V2,
    TimeDelta64JSON_V3,
)

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat
    from zarr.core.context import Context
    from zarr.core.dtype.common import DTypeJSON

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from zarr.core.common import JSON
from zarr.core.dtype.npy.string import (
    FixedLengthUTF32,
    FixedLengthUTF32JSON_V2,
    FixedLengthUTF32JSON_V3,
    VariableLengthUTF8,
    VariableLengthUTF8JSON_V2,
)
from zarr.core.dtype.registry import DataTypeRegistry
from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

__all__ = [
    "Bool",
    "Complex64",
    "Complex128",
    "DataTypeRegistry",
    "DateTime64",
    "DateTime64JSON_V2",
    "DateTime64JSON_V3",
    "FixedLengthUTF32",
    "FixedLengthUTF32JSON_V2",
    "FixedLengthUTF32JSON_V3",
    "Float16",
    "Float32",
    "Float64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "NullTerminatedBytes",
    "NullTerminatedBytesJSON_V3",
    "NullterminatedBytesJSON_V2",
    "RawBytes",
    "RawBytesJSON_V2",
    "RawBytesJSON_V3",
    "Struct",
    "StructJSON_V3",
    "Structured",
    "StructuredJSON_V2",
    "StructuredJSON_V3",
    "TBaseDType",
    "TBaseScalar",
    "TimeDelta64",
    "TimeDelta64JSON_V2",
    "TimeDelta64JSON_V3",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "VariableLengthBytes",
    "VariableLengthBytesJSON_V2",
    "VariableLengthUTF8",
    "VariableLengthUTF8JSON_V2",
    "ZDType",
    "data_type_registry",
    "parse_data_type",
    "parse_dtype",
]

data_type_registry = DataTypeRegistry()

IntegerDType = Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
INTEGER_DTYPE: Final = Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64

FloatDType = Float16 | Float32 | Float64
FLOAT_DTYPE: Final = Float16, Float32, Float64

ComplexFloatDType = Complex64 | Complex128
COMPLEX_FLOAT_DTYPE: Final = Complex64, Complex128

StringDType = FixedLengthUTF32 | VariableLengthUTF8
STRING_DTYPE: Final = FixedLengthUTF32, VariableLengthUTF8

TimeDType = DateTime64 | TimeDelta64
TIME_DTYPE: Final = DateTime64, TimeDelta64

BytesDType = RawBytes | NullTerminatedBytes | VariableLengthBytes
BYTES_DTYPE: Final = RawBytes, NullTerminatedBytes, VariableLengthBytes

AnyDType = (
    Bool
    | IntegerDType
    | FloatDType
    | ComplexFloatDType
    | StringDType
    | BytesDType
    | Struct
    | TimeDType
    | VariableLengthBytes
)
# mypy has trouble inferring the type of variablelengthstring dtype, because its class definition
# depends on the installed numpy version. That's why the type: ignore statement is needed here.
ANY_DTYPE: Final = (
    Bool,
    *INTEGER_DTYPE,
    *FLOAT_DTYPE,
    *COMPLEX_FLOAT_DTYPE,
    *STRING_DTYPE,
    *BYTES_DTYPE,
    Struct,
    *TIME_DTYPE,
    VariableLengthBytes,
)

# These are aliases for variable-length UTF-8 strings
# We handle them when a user requests a data type instead of using NumPy's dtype inferece because
# the default NumPy behavior -- to inspect the user-provided array data and choose
# an appropriately sized U dtype -- is unworkable for Zarr.
VLEN_UTF8_ALIAS: Final = ("str", str, "string")

# This type models inputs that can be coerced to a ZDType
type ZDTypeLike = npt.DTypeLike | ZDType[TBaseDType, TBaseScalar] | Mapping[str, JSON] | str

for dtype in ANY_DTYPE:
    # mypy does not know that all the elements of ANY_DTYPE are subclasses of ZDType
    data_type_registry.register(dtype._zarr_v3_name, dtype)  # type: ignore[arg-type]


# TODO: find a better name for this function
def get_data_type_from_native_dtype(
    dtype: npt.DTypeLike, *, context: Context | None = None
) -> ZDType[TBaseDType, TBaseScalar]:
    """
    Get a data type wrapper (an instance of ``ZDType``) from a native data type, e.g. a numpy dtype.

    With a `context`, the match is made against the data types it holds instead of the
    default data type registry.
    """
    if not isinstance(dtype, np.dtype):
        na_dtype: np.dtype[np.generic]
        if isinstance(dtype, list):
            # this is a valid _VoidDTypeLike check
            na_dtype = np.dtype([tuple(d) for d in dtype])
        else:
            na_dtype = np.dtype(dtype)
    else:
        na_dtype = dtype
    # A context's own data types, if it sets them; otherwise the default registry.
    registry = data_type_registry if context is None else context.data_type_registry()
    return registry.match_dtype(dtype=na_dtype)


def get_data_type_from_json(
    dtype_spec: DTypeJSON, *, zarr_format: ZarrFormat, context: Context | None = None
) -> ZDType[TBaseDType, TBaseScalar]:
    """
    Given a JSON representation of a data type and a Zarr format version,
    attempt to create a ZDType instance from the registered ZDType classes.

    With a `context`, the match is made against the data types it holds instead of the
    default data type registry.
    """
    # A context's own data types, if it sets them; otherwise the default registry.
    registry = data_type_registry if context is None else context.data_type_registry()
    return registry.match_json(dtype_spec, zarr_format=zarr_format)


def parse_data_type(
    dtype_spec: ZDTypeLike,
    *,
    zarr_format: ZarrFormat,
) -> ZDType[TBaseDType, TBaseScalar]:
    """
    Interpret the input as a ZDType.

    This function wraps ``parse_dtype``. The only difference is the function name. This function may
    be deprecated in a future version of Zarr Python in favor of ``parse_dtype``.

    Parameters
    ----------
    dtype_spec : ZDTypeLike
        The input to be interpreted as a ZDType. This could be a ZDType, which will be returned
        directly, or a JSON representation of a ZDType, or a native dtype, or a python object that
        can be converted into a native dtype.
    zarr_format : ZarrFormat
        The Zarr format version. This parameter is required because this function will attempt to
        parse the JSON representation of a data type, and the JSON representation of data types
        varies between Zarr 2 and Zarr 3.

    Returns
    -------
    ZDType[TBaseDType, TBaseScalar]
        The ZDType corresponding to the input.

    Examples
    --------
    ```python
    from zarr.dtype import parse_data_type
    import numpy as np
    parse_data_type("int32", zarr_format=2)
    # Int32(endianness='little')
    parse_data_type(np.dtype('S10'), zarr_format=2)
    # NullTerminatedBytes(length=10)
    parse_data_type({"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 10}}, zarr_format=3)
    # DateTime64(endianness='little', scale_factor=10, unit='s')
    ```
    """
    return parse_dtype(dtype_spec, zarr_format=zarr_format)


def parse_dtype(
    dtype_spec: ZDTypeLike,
    *,
    zarr_format: ZarrFormat,
    context: Context | None = None,
) -> ZDType[TBaseDType, TBaseScalar]:
    """
    Convert the input as a ZDType.

    Parameters
    ----------
    dtype_spec : ZDTypeLike
        The input to be converted to a ZDType. This could be a ZDType, which will be returned
        directly, or a JSON representation of a ZDType, or a numpy dtype, or a python object that
        can be converted into a native dtype.
    zarr_format : ZarrFormat
        The Zarr format version. This parameter is required because this function will attempt to
        parse the JSON representation of a data type, and the JSON representation of data types
        varies between Zarr 2 and Zarr 3.

    Returns
    -------
    ZDType[TBaseDType, TBaseScalar]
        The ZDType corresponding to the input.

    Examples
    --------
    ```python
    from zarr.dtype import parse_dtype
    import numpy as np
    parse_dtype("int32", zarr_format=2)
    # Int32(endianness='little')
    parse_dtype(np.dtype('S10'), zarr_format=2)
    # NullTerminatedBytes(length=10)
    parse_dtype({"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 10}}, zarr_format=3)
    # DateTime64(endianness='little', scale_factor=10, unit='s')
    ```
    """
    if isinstance(dtype_spec, ZDType):
        # Without a data type scope, an instance stands as given, as it always has.
        if context is None or context.data_types is None:
            return dtype_spec
        # A closed world checks instances too: the scope must read the instance's
        # JSON as the instance's own class, as a closed codec scope requires.
        matched = _matched_in_scope(dtype_spec, context, zarr_format)
        if not _same_classes(matched, dtype_spec):
            raise ValueError(
                f"Data type {dtype_spec!r} is not of the classes this context's scope "
                f"reads its JSON as, {matched!r}. Give an instance of those classes, or "
                "the data type's name."
            )
        return dtype_spec
    # First attempt to interpret the input as JSON
    if isinstance(dtype_spec, Mapping | str | Sequence):
        try:
            return get_data_type_from_json(dtype_spec, zarr_format=zarr_format, context=context)
        except ValueError:
            # no data type matched this JSON-like input
            pass
    if dtype_spec in VLEN_UTF8_ALIAS:
        # If the dtype request is one of the aliases for variable-length UTF-8 strings,
        # return that dtype.
        utf8 = VariableLengthUTF8()
        if context is not None and context.data_types is not None:
            # The shortcut must not widen a closed world, and the context may
            # map this JSON to its own class: return the registry's match.
            return _matched_in_scope(
                cast("ZDType[TBaseDType, TBaseScalar]", utf8), context, zarr_format
            )
        return utf8  # type: ignore[return-value]
    # otherwise, we have either a numpy dtype string, or a zarr v3 dtype string, and in either case
    # we can create a native dtype from it, and do the dtype inference from that
    zdtype = get_data_type_from_native_dtype(dtype_spec, context=context)  # type: ignore[arg-type]
    if context is not None and context.data_types is not None:
        # A native data type can hold others (a structured dtype's fields), which
        # resolve from the defaults: check the whole through the closed world.
        return _matched_in_scope(zdtype, context, zarr_format)
    return zdtype


def __getattr__(name: str) -> object:
    if name == "DataTypeValidationError":
        import warnings

        from zarr.errors import DataTypeValidationError, ZarrDeprecationWarning

        warnings.warn(
            "Importing DataTypeValidationError from zarr.core.dtype is deprecated. "
            "Use zarr.errors.DataTypeValidationError instead.",
            ZarrDeprecationWarning,
            stacklevel=2,
        )
        return DataTypeValidationError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _json_to_reread(zdtype: ZDType[TBaseDType, TBaseScalar], zarr_format: ZarrFormat) -> DTypeJSON:
    """`zdtype`'s JSON, for reading it again: without the warning a data type with
    no stable specification gives when it is written (nothing is written). Only the
    warning given through `v3_unstable_dtype_warning` is silenced; a data type that
    calls `warnings.warn` itself warns here as well."""
    token = _UNSTABLE_WARNING_SILENCED.set(True)
    try:
        return zdtype.to_json(zarr_format=zarr_format)
    finally:
        _UNSTABLE_WARNING_SILENCED.reset(token)


def _matched_in_scope(
    zdtype: ZDType[TBaseDType, TBaseScalar], context: Context, zarr_format: ZarrFormat
) -> ZDType[TBaseDType, TBaseScalar]:
    """The data type a closed data type scope matches `zdtype`'s JSON to, in the byte
    order of `zdtype` (see `_with_byte_order_of`): how a resolved data type is read
    again under a scope, which rebinds the scope's class for its name."""
    matched = context.data_type_registry().match_json(
        _json_to_reread(zdtype, zarr_format), zarr_format=zarr_format
    )
    return _with_byte_order_of(matched, zdtype)


def _same_classes(
    zdtype: ZDType[TBaseDType, TBaseScalar], other: ZDType[TBaseDType, TBaseScalar]
) -> bool:
    """Whether `zdtype` and `other` are of the same class, and so are their fields, for
    a structured data type."""
    if type(zdtype) is not type(other):
        return False
    if isinstance(zdtype, Structured) and isinstance(other, Structured):
        return len(zdtype.fields) == len(other.fields) and all(
            _same_classes(field, other_field)
            for (_, field), (_, other_field) in zip(zdtype.fields, other.fields, strict=True)
        )
    return True


def _with_byte_order_of(
    zdtype: ZDType[TBaseDType, TBaseScalar], source: ZDType[TBaseDType, TBaseScalar]
) -> ZDType[TBaseDType, TBaseScalar]:
    """`zdtype`, read again from `source`'s JSON, in `source`'s byte order (its fields',
    for a structured data type): Zarr format 3 data type JSON carries none, so reading
    it alone gives the host's byte order."""
    if (
        isinstance(source, Structured)
        and isinstance(zdtype, Structured)
        and len(source.fields) == len(zdtype.fields)
    ):
        fields = tuple(
            (name, _with_byte_order_of(field, source_field))
            for (name, field), (_, source_field) in zip(zdtype.fields, source.fields, strict=True)
        )
        return cast("ZDType[TBaseDType, TBaseScalar]", replace(cast("Any", zdtype), fields=fields))
    if isinstance(source, HasEndianness) and isinstance(zdtype, HasEndianness):
        return cast(
            "ZDType[TBaseDType, TBaseScalar]",
            replace(cast("Any", zdtype), endianness=source.endianness),
        )
    return zdtype
