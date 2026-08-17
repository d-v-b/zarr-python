"""Composition rules for the `struct` data type.

`StructField`'s own docstring promises field names are unique within a
struct and non-empty. Neither is expressible in a TypedDict, so both are
composition judgments and live here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._registry import entity_rule
from zarr_metadata.v3._extension_points import DATA_TYPE
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE_NAME

if TYPE_CHECKING:
    from zarr_metadata.rules._spec import ArraySpec

_ARRAY_V3 = "zarr_v3_array"


def _field_names(configuration: Mapping[str, object]) -> tuple[tuple[int, str], ...]:
    """`(index, name)` for each field with a string name, else nothing.

    Anything the shape validator would reject is skipped: it owns that
    complaint, and judging names inside a malformed field list is noise.
    """
    fields = configuration.get("fields")
    if not isinstance(fields, tuple):
        return ()
    named: list[tuple[int, str]] = []
    for index, field in enumerate(cast("tuple[object, ...]", fields)):
        if not isinstance(field, Mapping):
            continue
        name = cast("Mapping[object, object]", field).get("name")
        if isinstance(name, str):
            named.append((index, name))
    return tuple(named)


@entity_rule(_ARRAY_V3, DATA_TYPE, STRUCT_DATA_TYPE_NAME)
def field_names_are_non_empty(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    """A struct field must be addressable, so its name cannot be empty."""
    return tuple(
        ValidationProblem(
            ("fields", index, "name"), "expected a non-empty field name", "invalid_value"
        )
        for index, name in _field_names(configuration)
        if name == ""
    )


@entity_rule(_ARRAY_V3, DATA_TYPE, STRUCT_DATA_TYPE_NAME)
def field_names_are_unique(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    """Duplicate field names make a fill value's per-field mapping ambiguous."""
    seen: dict[str, int] = {}
    problems: list[ValidationProblem] = []
    for index, name in _field_names(configuration):
        first = seen.get(name)
        if first is None:
            seen[name] = index
            continue
        problems.append(
            ValidationProblem(
                ("fields", index, "name"),
                f"duplicate field name {name!r}, already used by field {first}",
                "invalid_value",
            )
        )
    return tuple(problems)
