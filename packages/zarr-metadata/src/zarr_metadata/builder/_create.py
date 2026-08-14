"""One-shot factories for Zarr metadata documents.

Package rule: **every public document TypedDict gets a `create_*` factory**
whose signature is `**kwargs: Unpack[<TypedDict>]`. Unpacking the *total*
TypedDict makes a missing required key a static error at the call site —
strictly stronger than any runtime completeness check — while wrong value
types and unknown keys are already static errors. At runtime each factory
deep-copies its inputs, materializes JSON arrays as tuples, and (where the
package defines them) runs the structural validator and the semantic rules,
raising one `MetadataValidationError` carrying every problem found.

The rule is deliberately scoped to *document* TypedDicts. Entity TypedDicts
(`BloscCodecObject`, `RegularChunkGridConfiguration`, ...) do not get
factories: TypedDict constructor syntax (`BloscCodecObject(name=..., ...)`)
already enforces their shape statically, no semantic rules apply to an
entity in isolation, and a do-nothing factory would falsely suggest that
factory-built entities are validated while literal-built ones are not.

The v3 array and group documents are open (PEP 728 `extra_items`), but
type checkers without PEP 728 support reject unknown keyword names
statically — so those factories take an `extensions=` mapping for
extension fields, mirroring `ZarrV3ArrayMetadataBuilder.evolve_extension`.

`DOCUMENT_FACTORIES` maps each document TypedDict name to its factory and
is the source of truth the drift test checks against, so a new document
TypedDict cannot ship without one.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Final, cast

from typing_extensions import Unpack

from zarr_metadata.builder._rules import ZARR_V3_ARRAY_RULES, run_rules
from zarr_metadata.model._validation import (
    ARRAY_METADATA_STANDARD_KEYS_V3,
    GROUP_METADATA_STANDARD_KEYS_V3,
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    parse_array_metadata_v2,
    parse_array_metadata_v3,
    parse_group_metadata_v2,
    parse_group_metadata_v3,
    validate_consolidated_metadata_v3,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from collections.abc import Set as AbstractSet

    from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON, ZarrV2ZArrayJSON
    from zarr_metadata.v2.consolidated import ZarrV2ConsolidatedMetadataJSON
    from zarr_metadata.v2.group import ZarrV2GroupMetadataJSON, ZarrV2ZGroupJSON
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON, ZarrV3ExtensionField
    from zarr_metadata.v3.consolidated import ZarrV3ConsolidatedMetadataJSON
    from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON


def _merged_with_extensions(
    kwargs: Mapping[str, object],
    extensions: Mapping[str, ZarrV3ExtensionField] | None,
    standard_keys: AbstractSet[str],
) -> tuple[dict[str, object], list[ValidationProblem]]:
    """`kwargs` plus `extensions`, refusing extension names that shadow standard keys.

    A colliding name is reported and *not* merged, so a later structural or
    semantic pass judges the standard field the caller actually passed
    rather than a value smuggled in through the extension hatch.
    """
    document = dict(kwargs)
    problems: list[ValidationProblem] = []
    for name, value in (extensions or {}).items():
        if name in standard_keys:
            problems.append(
                ValidationProblem(
                    (name,),
                    f"{name!r} is a standard metadata key; pass it as a keyword argument",
                    "invalid_value",
                )
            )
        else:
            document[name] = value
    return document, problems


def _normalized(document: Mapping[str, object]) -> dict[str, object]:
    """A deep copy of `document` with JSON arrays materialized as tuples.

    Deep-copying first means the returned document shares no mutable state
    with the caller's arguments: mutating an input after the factory
    returns cannot alter the validated result.
    """
    return cast("dict[str, object]", arrays_to_tuples(copy.deepcopy(dict(document))))


def _raise_if_problems(problems: list[ValidationProblem]) -> None:
    if problems:
        raise MetadataValidationError(problems)


def create_zarr_v3_array_metadata_json(
    *,
    extensions: Mapping[str, ZarrV3ExtensionField] | None = None,
    **kwargs: Unpack[ZarrV3ArrayMetadataJSON],
) -> ZarrV3ArrayMetadataJSON:
    """A validated v3 array metadata document (the `zarr.json` content for an array).

    Required keys are enforced statically by the signature; at runtime the
    document is checked structurally (via the model layer's parser) and
    semantically (via `ZARR_V3_ARRAY_RULES`), and every problem from both
    passes is raised together in one `MetadataValidationError`. Extension
    fields go in `extensions`; names that shadow standard keys are rejected.
    """
    document, problems = _merged_with_extensions(
        kwargs, extensions, ARRAY_METADATA_STANDARD_KEYS_V3
    )
    normalized = _normalized(document)
    parsed: ZarrV3ArrayMetadataJSON | None = None
    try:
        parsed = parse_array_metadata_v3(normalized)
    except MetadataValidationError as error:
        problems += error.problems
    problems += run_rules(ZARR_V3_ARRAY_RULES, normalized)
    _raise_if_problems(problems)
    assert parsed is not None
    return parsed


def create_zarr_v3_group_metadata_json(
    *,
    extensions: Mapping[str, ZarrV3ExtensionField] | None = None,
    **kwargs: Unpack[ZarrV3GroupMetadataJSON],
) -> ZarrV3GroupMetadataJSON:
    """A validated v3 group metadata document (the `zarr.json` content for a group).

    Extension fields go in `extensions`; names that shadow standard keys
    are rejected. No semantic rules apply to group documents today, so the
    runtime pass is structural only.
    """
    document, problems = _merged_with_extensions(
        kwargs, extensions, GROUP_METADATA_STANDARD_KEYS_V3
    )
    normalized = _normalized(document)
    parsed: ZarrV3GroupMetadataJSON | None = None
    try:
        parsed = parse_group_metadata_v3(normalized)
    except MetadataValidationError as error:
        problems += error.problems
    _raise_if_problems(problems)
    assert parsed is not None
    return parsed


def create_zarr_v3_consolidated_metadata_json(
    **kwargs: Unpack[ZarrV3ConsolidatedMetadataJSON],
) -> ZarrV3ConsolidatedMetadataJSON:
    """A validated v3 inline consolidated metadata object.

    This is the value embedded in a v3 group document under the
    `consolidated_metadata` key, not a store document of its own.
    """
    normalized = _normalized(kwargs)
    _raise_if_problems(validate_consolidated_metadata_v3(normalized))
    return cast("ZarrV3ConsolidatedMetadataJSON", normalized)


def create_zarr_v2_array_metadata_json(
    **kwargs: Unpack[ZarrV2ArrayMetadataJSON],
) -> ZarrV2ArrayMetadataJSON:
    """A validated v2 array metadata document, in-memory merged form.

    Models `.zarray` plus the sibling `.zattrs` folded in as `attributes`.
    For the strict on-disk `.zarray` shape use `create_zarr_v2_z_array_json`.
    """
    parsed: ZarrV2ArrayMetadataJSON | None = None
    problems: list[ValidationProblem] = []
    try:
        parsed = parse_array_metadata_v2(_normalized(kwargs))
    except MetadataValidationError as error:
        problems += error.problems
    _raise_if_problems(problems)
    assert parsed is not None
    return parsed


def create_zarr_v2_group_metadata_json(
    **kwargs: Unpack[ZarrV2GroupMetadataJSON],
) -> ZarrV2GroupMetadataJSON:
    """A validated v2 group metadata document, in-memory merged form.

    Models `.zgroup` plus the sibling `.zattrs` folded in as `attributes`.
    For the strict on-disk `.zgroup` shape use `create_zarr_v2_z_group_json`.
    """
    parsed: ZarrV2GroupMetadataJSON | None = None
    problems: list[ValidationProblem] = []
    try:
        parsed = parse_group_metadata_v2(_normalized(kwargs))
    except MetadataValidationError as error:
        problems += error.problems
    _raise_if_problems(problems)
    assert parsed is not None
    return parsed


def create_zarr_v2_z_array_json(**kwargs: Unpack[ZarrV2ZArrayJSON]) -> ZarrV2ZArrayJSON:
    """A validated on-disk `.zarray` document (strict form, no `attributes`).

    Structurally checked with the merged-form parser: the strict shape is
    the merged shape minus `attributes`, which the signature already
    excludes statically.
    """
    parsed: ZarrV2ArrayMetadataJSON | None = None
    problems: list[ValidationProblem] = []
    try:
        parsed = parse_array_metadata_v2(_normalized(kwargs))
    except MetadataValidationError as error:
        problems += error.problems
    _raise_if_problems(problems)
    assert parsed is not None
    return cast("ZarrV2ZArrayJSON", parsed)


def create_zarr_v2_z_group_json(**kwargs: Unpack[ZarrV2ZGroupJSON]) -> ZarrV2ZGroupJSON:
    """A validated on-disk `.zgroup` document (strict form, no `attributes`).

    Structurally checked with the merged-form parser: the strict shape is
    the merged shape minus `attributes`, which the signature already
    excludes statically.
    """
    parsed: ZarrV2GroupMetadataJSON | None = None
    problems: list[ValidationProblem] = []
    try:
        parsed = parse_group_metadata_v2(_normalized(kwargs))
    except MetadataValidationError as error:
        problems += error.problems
    _raise_if_problems(problems)
    assert parsed is not None
    return cast("ZarrV2ZGroupJSON", parsed)


def create_zarr_v2_consolidated_metadata_json(
    **kwargs: Unpack[ZarrV2ConsolidatedMetadataJSON],
) -> ZarrV2ConsolidatedMetadataJSON:
    """A `.zmetadata` consolidated metadata document.

    The package defines no structural validator or semantic rules for the
    v2 consolidated envelope, so the runtime pass is normalization only;
    the signature enforces the envelope shape statically. The nested
    per-path documents are typed but not re-validated here — validate them
    individually with the model layer when reading untrusted data.
    """
    return cast("ZarrV2ConsolidatedMetadataJSON", _normalized(kwargs))


DOCUMENT_FACTORIES: Final[Mapping[str, Callable[..., Mapping[str, object]]]] = {
    "ZarrV2ArrayMetadataJSON": create_zarr_v2_array_metadata_json,
    "ZarrV2ConsolidatedMetadataJSON": create_zarr_v2_consolidated_metadata_json,
    "ZarrV2GroupMetadataJSON": create_zarr_v2_group_metadata_json,
    "ZarrV2ZArrayJSON": create_zarr_v2_z_array_json,
    "ZarrV2ZGroupJSON": create_zarr_v2_z_group_json,
    "ZarrV3ArrayMetadataJSON": create_zarr_v3_array_metadata_json,
    "ZarrV3ConsolidatedMetadataJSON": create_zarr_v3_consolidated_metadata_json,
    "ZarrV3GroupMetadataJSON": create_zarr_v3_group_metadata_json,
}
"""Every public document TypedDict, mapped to its factory.

The drift test derives each expected factory name mechanically from the
TypedDict name and checks this mapping is exactly the set of public
document TypedDicts, so a new document type cannot ship without a factory
(or a factory without its type).
"""


__all__ = [
    "DOCUMENT_FACTORIES",
    "create_zarr_v2_array_metadata_json",
    "create_zarr_v2_consolidated_metadata_json",
    "create_zarr_v2_group_metadata_json",
    "create_zarr_v2_z_array_json",
    "create_zarr_v2_z_group_json",
    "create_zarr_v3_array_metadata_json",
    "create_zarr_v3_consolidated_metadata_json",
    "create_zarr_v3_group_metadata_json",
]
