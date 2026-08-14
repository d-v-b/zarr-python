"""Whole-document validation: structure plus composition, one entry point.

These trios mirror the model layer's `validate_*` / `is_*` / `parse_*`
grammar and carry the same names, scoped by module: the model layer's
functions judge *structure* only, while the functions here run the
structural validator **and** the composition rules and report every
problem from both passes together. They are the front door for readers —
"is this loaded `zarr.json` a document I should act on?" — the same
combined judgment the `create_*` factories apply at construction time.

One deliberate difference from the model grammar: the `is_*` functions
here return plain `bool`, not `TypeIs`. The TypedDicts encode structure,
so only the structural layer can narrow honestly; a composition-aware
guard is stricter than the type, and a `TypeIs` built on it would wrongly
exclude the type in its negative branch (a document with `fill_value`
300 for `uint8` *is* a `ZarrV3ArrayMetadataJSON` — it is just not a
valid one). Use the model layer's `is_*` to narrow, and these to judge.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import (
    MetadataValidationError,
    arrays_to_tuples,
)
from zarr_metadata.model._validation import (
    validate_array_metadata_v2 as _validate_structure_v2,
)
from zarr_metadata.model._validation import (
    validate_array_metadata_v3 as _validate_structure_v3,
)
from zarr_metadata.rules._engine import run_rules
from zarr_metadata.rules._v2_array import ZARR_V2_ARRAY_RULES
from zarr_metadata.rules._v3_array import ZARR_V3_ARRAY_RULES

if TYPE_CHECKING:
    from zarr_metadata.model._validation import ValidationProblem
    from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON


def validate_array_metadata_v3(value: object) -> list[ValidationProblem]:
    """Every reason `value` is not a valid v3 array document.

    Structural problems (from the model layer) and composition problems
    (from `ZARR_V3_ARRAY_RULES`) are reported together.
    """
    problems = _validate_structure_v3(value)
    if isinstance(value, Mapping):
        document = cast("Mapping[str, object]", value)
        problems = problems + run_rules(ZARR_V3_ARRAY_RULES, document)
    return problems


def is_array_metadata_v3(value: object) -> bool:
    """Whether `value` is a structurally and compositionally valid v3 array doc.

    Deliberately not a `TypeIs` guard — see the module docstring. Use
    `zarr_metadata.model.is_array_metadata_v3` to narrow.
    """
    return not validate_array_metadata_v3(value)


def parse_array_metadata_v3(value: object) -> ZarrV3ArrayMetadataJSON:
    """Return `value` as a valid `ZarrV3ArrayMetadataJSON`, or raise.

    Normalizes JSON arrays to tuples, then raises a single
    `MetadataValidationError` carrying every structural and composition
    problem found.
    """
    normalized = arrays_to_tuples(value)
    problems = validate_array_metadata_v3(normalized)
    if problems:
        raise MetadataValidationError(problems)
    return cast("ZarrV3ArrayMetadataJSON", normalized)


def validate_array_metadata_v2(value: object) -> list[ValidationProblem]:
    """Every reason `value` is not a valid v2 array document (merged form)."""
    problems = _validate_structure_v2(value)
    if isinstance(value, Mapping):
        document = cast("Mapping[str, object]", value)
        problems = problems + run_rules(ZARR_V2_ARRAY_RULES, document)
    return problems


def is_array_metadata_v2(value: object) -> bool:
    """Whether `value` is a structurally and compositionally valid v2 array doc.

    Deliberately not a `TypeIs` guard — see the module docstring.
    """
    return not validate_array_metadata_v2(value)


def parse_array_metadata_v2(value: object) -> ZarrV2ArrayMetadataJSON:
    """Return `value` as a valid `ZarrV2ArrayMetadataJSON`, or raise.

    Normalizes JSON arrays to tuples, then raises a single
    `MetadataValidationError` carrying every structural and composition
    problem found.
    """
    normalized = arrays_to_tuples(value)
    problems = validate_array_metadata_v2(normalized)
    if problems:
        raise MetadataValidationError(problems)
    return cast("ZarrV2ArrayMetadataJSON", normalized)


__all__ = [
    "is_array_metadata_v2",
    "is_array_metadata_v3",
    "parse_array_metadata_v2",
    "parse_array_metadata_v3",
    "validate_array_metadata_v2",
    "validate_array_metadata_v3",
]
