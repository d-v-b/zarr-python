"""The scope zarr-python reads v3 metadata in: zarr-metadata's, and zarr-python's own data types.

zarr-metadata models the data types the spec and the zarr-extensions
registry define. zarr-python writes four more, and each is here as an
entity, registered through `Context.extended_with` as any extension is,
so that the resolved pipeline carries it as it carries a core type. Each
judges a fill value by asking zarr-python's own data type, which is the
authority on the encodings these names use.

No future import: zarr-metadata resolves these annotations at
registration, in this module's namespace.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import ClassVar

from zarr_metadata.v3.entity import (
    CORE_AND_EXTENSIONS,
    Configuration,
    Context,
    DataTypeEntity,
    Loc,
    StorageClass,
    ValidationProblem,
)


def _fill_value_problems(
    entity: DataTypeEntity, value: object, loc: Loc
) -> tuple[ValidationProblem, ...]:
    """Why zarr-python's own data type of this name refuses `value` as a fill value."""
    from zarr.core.dtype import get_data_type_from_json

    dtype = get_data_type_from_json(entity.to_json(), zarr_format=3)
    try:
        dtype.from_json_scalar(value, zarr_format=3)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        return (ValidationProblem(loc, str(error), "invalid_value"),)
    return ()


@dataclass(frozen=True)
class LengthBytesOptions(Configuration):
    """A fixed width, in bytes."""

    length_bytes: int

    def problems(self) -> Iterator[ValidationProblem]:
        if self.length_bytes < 1:
            yield ValidationProblem(
                ("length_bytes",),
                f"expected a positive integer, got {self.length_bytes}",
                "invalid_value",
            )


@dataclass(frozen=True)
class Utf32Options(Configuration):
    """A fixed width, in bytes, of whole UTF-32 code units."""

    length_bytes: int

    def problems(self) -> Iterator[ValidationProblem]:
        if self.length_bytes < 1 or self.length_bytes % 4 != 0:
            yield ValidationProblem(
                ("length_bytes",),
                f"expected a positive multiple of 4, got {self.length_bytes}",
                "invalid_value",
            )


@dataclass(frozen=True)
class FixedLengthUtf32DataType(DataTypeEntity):
    """`fixed_length_utf32`: NumPy's `U` type, UTF-32 code units."""

    configuration: Utf32Options

    identifier: ClassVar[str] = "fixed_length_utf32"
    scalar_storage: ClassVar[StorageClass] = "multi_byte"
    twos_complement: ClassVar[bool] = False

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        return _fill_value_problems(self, value, loc)


@dataclass(frozen=True)
class NullTerminatedBytesDataType(DataTypeEntity):
    """`null_terminated_bytes`: NumPy's `S` type."""

    configuration: LengthBytesOptions

    identifier: ClassVar[str] = "null_terminated_bytes"
    scalar_storage: ClassVar[StorageClass] = "single_byte"
    twos_complement: ClassVar[bool] = False

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        return _fill_value_problems(self, value, loc)


@dataclass(frozen=True)
class ZarrPythonRawBytesDataType(DataTypeEntity):
    """`raw_bytes`: NumPy's unstructured `V` type."""

    configuration: LengthBytesOptions

    identifier: ClassVar[str] = "raw_bytes"
    scalar_storage: ClassVar[StorageClass] = "single_byte"
    twos_complement: ClassVar[bool] = False

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        return _fill_value_problems(self, value, loc)


@dataclass(frozen=True)
class VariableLengthBytesDataType(DataTypeEntity):
    """`variable_length_bytes`: NumPy object arrays of `bytes`."""

    configuration: Configuration = field(default_factory=Configuration)

    identifier: ClassVar[str] = "variable_length_bytes"
    scalar_storage: ClassVar[StorageClass] = "variable_length"
    twos_complement: ClassVar[bool] = False

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        return _fill_value_problems(self, value, loc)


SCOPE: Context = CORE_AND_EXTENSIONS.extended_with(
    FixedLengthUtf32DataType,
    NullTerminatedBytesDataType,
    ZarrPythonRawBytesDataType,
    VariableLengthBytesDataType,
)
"""What zarr-python reads v3 metadata in."""
