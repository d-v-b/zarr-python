"""
# Overview

This module provides a proof-of-concept standalone interface for managing dtypes in the zarr-python codebase.

The `ZarrDType` class introduced in this module effectively acts as a replacement for `np.dtype` throughout the
zarr-python codebase. It attempts to encapsulate all relevant runtime information necessary for working with
dtypes in the context of the Zarr V3 specification (e.g. is this a core dtype or not, how many bytes and what
endianness is the dtype etc). By providing this abstraction, the module aims to:

- Simplify dtype management within zarr-python
- Support runtime flexibility and custom extensions
- Remove unnecessary dependencies on the numpy API

## Extensibility

The module attempts to support user-driven extensions, allowing developers to introduce custom dtypes
without requiring immediate changes to zarr-python. Extensions can leverage the current entrypoint mechanism,
enabling integration of experimental features. Over time, widely adopted extensions may be formalized through
inclusion in zarr-python or standardized via a Zarr Enhancement Proposal (ZEP), but this is not essential.

## Examples

### Core `dtype` Registration

The following example demonstrates how to register a built-in `dtype` in the core codebase:

```python
from zarr.core.dtype import ZarrDType
from zarr.registry import register_v3dtype

class Float16(ZarrDType):
    zarr_spec_format = "3"
    experimental = False
    endianness = "little"
    byte_count = 2
    to_numpy = np.dtype('float16')

register_v3dtype(Float16)
```

### Entrypoint Extension

The following example demonstrates how users can register a new `bfloat16` dtype for Zarr.
This approach adheres to the existing Zarr entrypoint pattern as much as possible, ensuring
consistency with other extensions. The code below would typically be part of a Python package
that specifies the entrypoints for the extension:

```python
import ml_dtypes
from zarr.core.dtype import ZarrDType  # User inherits from ZarrDType when creating their dtype

class Bfloat16(ZarrDType):
    zarr_spec_format = "3"
    experimental = True
    endianness = "little"
    byte_count = 2
    to_numpy = np.dtype('bfloat16')  # Enabled by importing ml_dtypes
    configuration_v3 = {
        "version": "example_value",
        "author": "example_value",
        "ml_dtypes_version": "example_value"
    }
```

### dtype lookup

The following examples demonstrate how to perform a lookup for the relevant ZarrDType, given
a string that matches the dtype Zarr specification ID, or a numpy dtype object:

```
from zarr.registry import get_v3dtype_class, get_v3dtype_class_from_numpy

get_v3dtype_class('complex64')  # returns little-endian Complex64 ZarrDType
get_v3dtype_class('not_registered_dtype')  # ValueError

get_v3dtype_class_from_numpy('>i2')  # returns big-endian Int16 ZarrDType
get_v3dtype_class_from_numpy(np.dtype('float32'))  # returns little-endian Float32 ZarrDType
get_v3dtype_class_from_numpy('i10')  # ValueError
```

### String dtypes

The following indicates one possibility for supporting variable-length strings. It is via the
entrypoint mechanism as in a previous example. The Apache Arrow specification does not currently
include a dtype for fixed-length strings (only for fixed-length bytes) and so I am using string
here to implicitly refer to a variable-length string data (there may be some subtleties with codecs
that means this needs to be refined further):

```python
import numpy as np
from zarr.core.dtype import ZarrDType  # User inherits from ZarrDType when creating their dtype

try:
    to_numpy = np.dtypes.StringDType()
except AttributeError:
    to_numpy = np.dtypes.ObjectDType()

class String(ZarrDType):
    zarr_spec_format = "3"
    experimental = True
    endianness = 'little'
    byte_count = None  # None is defined to mean variable
    to_numpy = to_numpy
```

### int4 dtype

There is currently considerable interest in the AI community in 'quantising' models - storing
models at reduced precision, while minimising loss of information content. There are a number
of sub-byte dtypes that the community are using e.g. int4. Unfortunately numpy does not
currently have support for handling such sub-byte dtypes in an easy way. However, they can
still be held in a numpy array and then passed (in a zero-copy way) to something like pytorch
which can handle appropriately:

```python
import numpy as np
from zarr.core.dtype import ZarrDType  # User inherits from ZarrDType when creating their dtype

class Int4(ZarrDType):
    zarr_spec_format = "3"
    experimental = True
    endianness = 'little'
    byte_count = 1  # this is ugly, but I could change this from byte_count to bit_count if there was consensus
    to_numpy = np.dtype('B')  # could also be np.dtype('V1'), but this would prevent bit-twiddling
    configuration_v3 = {
        "version": "example_value",
        "author": "example_value",
    }
```
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


# perhaps over-complicating, but I don't want to allow the attributes to be patched
class FrozenClassVariables(type):
    def __setattr__(cls, attr: str, value: object) -> None:
        if hasattr(cls, attr):
            raise ValueError(f"Attribute {attr} on ZarrDType class can not be changed once set.")


Endianness = Literal["big", "little"]


class ZarrDType(metaclass=FrozenClassVariables):
    name: str
    byte_count: int | None  # None indicates variable count
    to_numpy: np.dtype[Any]  # may involve installing a a numpy extension e.g. ml_dtypes;

    def __init_subclass__(  # enforces all required fields are set and basic sanity checks
        cls,
        **kwargs: object,
    ) -> None:
        required_attrs = [
            "name",
            "endianness",
            "byte_count",
            "to_numpy",
        ]
        for attr in required_attrs:
            if not hasattr(cls, attr):
                raise ValueError(f"{attr} is a required attribute for a Zarr dtype.")

        cls._validate()  # sanity check on basic requirements

        super().__init_subclass__(**kwargs)

    # TODO: add further checks
    @classmethod
    def _validate(cls) -> None:
        if cls.byte_count is not None and cls.byte_count <= 0:
            raise ValueError("byte_count must be a positive integer.")


# create numpy dtypes
class Bool(ZarrDType):
    name = "bool"
    byte_count = 1
    to_numpy = np.dtype("bool_")


class Int8(ZarrDType):
    name = "int8"
    byte_count = 1
    to_numpy = np.dtype("int8")


class Uint8(ZarrDType):
    name = "uint8"
    byte_count = 1
    to_numpy = np.dtype("uint8")


class Int16(ZarrDType):
    name = "int16"
    byte_count = 2
    to_numpy = np.dtype("int16")


class Uint16(ZarrDType):
    name = "uint16"
    byte_count = 2
    to_numpy = np.dtype("uint16")


class Int32(ZarrDType):
    name = "int32"
    byte_count = 4
    to_numpy = np.dtype("int32")


class Uint32(ZarrDType):
    name = "uint32"
    byte_count = 4
    to_numpy = np.dtype("uint32")


class Int64(ZarrDType):
    name = "int64"
    byte_count = 8
    to_numpy = np.dtype("int64")


class Uint64(ZarrDType):
    name = "uint64"
    byte_count = 8
    to_numpy = np.dtype("uint64")


class Float16(ZarrDType):
    name = "float16"
    byte_count = 2
    to_numpy = np.dtype("float16")


class Float32(ZarrDType):
    name = "float32"
    byte_count = 4
    to_numpy = np.dtype("float32")


class Float64(ZarrDType):
    name = "float64"
    byte_count = 8
    to_numpy = np.dtype("float64")


class Complex64(ZarrDType):
    name = "complex64"
    byte_count = 8
    to_numpy = np.dtype("complex64")


class Complex128(ZarrDType):
    name = "complex128"
    byte_count = 16
    to_numpy = np.dtype("complex128")


DateUnit = Literal["Y", "M", "W", "D"]
TimeUnit = Literal["h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as"]


class DateTime64Y(ZarrDType):
    name = "numpy/datetime64"
    byte_count = 8
    unit: Literal["Y"] = "Y"
    to_numpy = np.dtype("datetime64[Y]")


class DateTime64M(ZarrDType):
    name = "numpy/datetime64"
    byte_count = 8
    unit: Literal["M"] = "M"
    to_numpy = np.dtype("datetime64[M]")


class DateTime64W(ZarrDType):
    name = "numpy/datetime64"
    byte_count = 8
    unit: Literal["W"] = "W"
    to_numpy = np.dtype("datetime64[W]")


class DateTime64D(ZarrDType):
    name = "numpy/datetime64"
    byte_count = 8
    unit: Literal["D"] = "D"
    to_numpy = np.dtype("datetime64[D]")


class DateTime64H(ZarrDType):
    name = "numpy/datetime64"
    byte_count = 8
    unit: Literal["h"] = "h"
    to_numpy = np.dtype("datetime64[h]")


class DateTime64m(ZarrDType):
    name = "numpy/datetime64"
    byte_count = 8
    unit: Literal["m"] = "m"
    to_numpy = np.dtype("datetime64[m]")


class DateTime64s(ZarrDType):
    name = "numpy/datetime64"
    byte_count = 8
    unit: Literal["s"] = "s"
    to_numpy = np.dtype("datetime64[s]")


class DateTime64ms(ZarrDType):
    name = "numpy/datetime64"
    byte_count = 8
    unit: Literal["ms"] = "ms"
    to_numpy = np.dtype("datetime64[ms]")


class DateTime64us(ZarrDType):
    name = "numpy/datetime64"
    byte_count = 8
    unit: Literal["us"] = "us"
    to_numpy = np.dtype("datetime64[us]")


class DateTime64ns(ZarrDType):
    name = "numpy/datetime64"
    byte_count = 8
    unit: Literal["ns"] = "ns"
    to_numpy = np.dtype("datetime64[ns]")


class DateTime64ps(ZarrDType):
    name = "numpy/datetime64"
    byte_count = 8
    unit: Literal["ps"] = "ps"
    to_numpy = np.dtype("datetime64[ps]")


class DateTime64fs(ZarrDType):
    name = "numpy/datetime64"
    byte_count = 8
    unit: Literal["fs"] = "fs"
    to_numpy = np.dtype("datetime64[fs]")


class DateTime64as(ZarrDType):
    name = "numpy/datetime64"
    byte_count = 8
    unit: Literal["as"] = "as"
    to_numpy = np.dtype("datetime64[as]")


def get_fixed_length_bytestring_dtype(length: int) -> type[ZarrDType]:
    return type(
        f"Bytes{length}",
        (ZarrDType,),
        {"name": "numpy/bytes", "byte_count": length, "to_numpy": np.dtype(f"S{length}")},
    )


def get_fixed_length_void_dtype(length: int) -> type[ZarrDType]:
    return type(
        f"Bytes{length}",
        (ZarrDType,),
        {"name": "numpy/void", "byte_count": length, "to_numpy": np.dtype(f"V{length}")},
    )
