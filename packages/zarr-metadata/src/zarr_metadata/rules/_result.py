"""A tagged union carrying either a validated document or its problems.

`validate_*` answers "what is wrong", `parse_*` answers "give me the
document or raise". Neither lets a caller hold both outcomes in one
value and let the type checker enforce that they looked: `validate_*`
returns a collection whose emptiness the checker cannot connect to the
document's validity, and `parse_*` moves the failure into control flow.

`check_*` closes that gap. It returns `Valid[T] | Invalid`, and because
the two arms differ in a `Literal[bool]` discriminant, narrowing on
`result.valid` gives the checker `result.document` in one branch and
`result.problems` in the other — reading the wrong one is a static
error, not a runtime `None`.

    result = check_array_metadata_v3(loaded)
    if result.valid:
        store(result.document)      # typed ZarrV3ArrayMetadataJSON
    else:
        report(result.problems)     # non-empty tuple of problems

This is offered *alongside* the trios rather than replacing them. The
trios mirror the model layer's grammar name-for-name, which is worth
keeping: a reader who knows `zarr_metadata.model.validate_array_metadata_v3`
should not have to learn a second shape to get the composition-aware
judgment. Prefer `check_*` when the document is the point and the
problems are the exception; prefer `validate_*` when collecting problems
across many documents; prefer `parse_*` at a trust boundary where
invalid input should abort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, TypeVar, cast

from zarr_metadata.model._validation import ValidationProblem, arrays_to_tuples
from zarr_metadata.rules._documents import (
    validate_array_metadata_v2,
    validate_array_metadata_v3,
    validate_group_metadata_v2,
    validate_group_metadata_v3,
)
from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON  # noqa: TC001
from zarr_metadata.v2.group import ZarrV2GroupMetadataJSON  # noqa: TC001
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON  # noqa: TC001
from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON  # noqa: TC001

DocumentT = TypeVar("DocumentT")


@dataclass(frozen=True, slots=True)
class Valid(Generic[DocumentT]):
    """A document that passed structural and composition validation."""

    document: DocumentT
    valid: Literal[True] = True


@dataclass(frozen=True, slots=True)
class Invalid:
    """Every reason a document failed validation.

    `problems` is never empty: an empty report is a `Valid`.
    """

    problems: tuple[ValidationProblem, ...]
    valid: Literal[False] = False

    def __post_init__(self) -> None:
        if len(self.problems) == 0:
            msg = "Invalid requires at least one validation problem"
            raise ValueError(msg)


ValidationResult: TypeAlias = Valid[DocumentT] | Invalid
"""Either a validated document or the problems that disqualified it."""


def check_array_metadata_v3(value: object) -> ValidationResult[ZarrV3ArrayMetadataJSON]:
    """`value` as a valid v3 array document, or the problems disqualifying it.

    A `Valid` carries the normalized document (JSON arrays as tuples),
    exactly as `parse_array_metadata_v3` returns it.
    """
    problems = validate_array_metadata_v3(value)
    if len(problems) != 0:
        return Invalid(problems)
    return Valid(cast("ZarrV3ArrayMetadataJSON", arrays_to_tuples(value)))


def check_array_metadata_v2(value: object) -> ValidationResult[ZarrV2ArrayMetadataJSON]:
    """`value` as a valid v2 array document, or the problems disqualifying it."""
    problems = validate_array_metadata_v2(value)
    if len(problems) != 0:
        return Invalid(problems)
    return Valid(cast("ZarrV2ArrayMetadataJSON", arrays_to_tuples(value)))


def check_group_metadata_v3(value: object) -> ValidationResult[ZarrV3GroupMetadataJSON]:
    """`value` as a valid v3 group document, or the problems disqualifying it."""
    problems = validate_group_metadata_v3(value)
    if len(problems) != 0:
        return Invalid(problems)
    return Valid(cast("ZarrV3GroupMetadataJSON", arrays_to_tuples(value)))


def check_group_metadata_v2(value: object) -> ValidationResult[ZarrV2GroupMetadataJSON]:
    """`value` as a valid v2 group document, or the problems disqualifying it."""
    problems = validate_group_metadata_v2(value)
    if len(problems) != 0:
        return Invalid(problems)
    return Valid(cast("ZarrV2GroupMetadataJSON", arrays_to_tuples(value)))


__all__ = [
    "Invalid",
    "Valid",
    "ValidationResult",
    "check_array_metadata_v2",
    "check_array_metadata_v3",
    "check_group_metadata_v2",
    "check_group_metadata_v3",
]
