"""One-shot factories for Zarr metadata documents.

Package rule: **every public document TypedDict gets a `create_*` factory**
whose signature is `**kwargs: Unpack[<TypedDict>]`. At a literal-keyword
call site, unpacking the *total* TypedDict makes a missing required key and
a wrong value type static errors. That guarantee is call-site-shaped, not
absolute: a `**`-splatted mapping bypasses required-key coverage in both
pyright and mypy, and for the open v3 documents a PEP 728 checker accepts
unknown keyword names as extension items while a non-PEP 728 checker
rejects them. The runtime pass exists for exactly the callers the static
story cannot see: each factory deep-copies its inputs, materializes JSON
arrays as tuples, and (where the package defines them) runs the structural
validator and the semantic rules, raising one `MetadataValidationError`
carrying every problem found.

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
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, cast

from typing_extensions import Unpack

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
    validate_json,
)
from zarr_metadata.rules import (
    ZARR_V2_ARRAY_RULES,
    ZARR_V3_ARRAY_RULES,
    ZARR_V3_GROUP_RULES,
    run_rules,
)
from zarr_metadata.rules._v3_group import consolidated_entries_problems

if TYPE_CHECKING:
    from collections.abc import Callable
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


def _raise_if_problems(problems: Sequence[ValidationProblem]) -> None:
    if len(problems) != 0:
        raise MetadataValidationError(problems)


def _reject_attributes(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """Problems for an `attributes` key in a strict on-disk v2 document.

    The strict `.zarray` / `.zgroup` shapes exclude `attributes` (it lives
    in the sibling `.zattrs` file). The signature enforces that statically
    at keyword call sites; this is the runtime backstop for `**`-splatted
    and untyped callers, without which the merged-form parser would accept
    the key and the returned value would not be the type it claims.
    """
    if "attributes" not in document:
        return ()
    return (
        ValidationProblem(
            ("attributes",),
            "'attributes' is not part of the on-disk document (it belongs to the "
            "sibling .zattrs file); use the merged-form factory instead",
            "invalid_value",
        ),
    )


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
        problems.extend(error.problems)
    problems.extend(run_rules(ZARR_V3_ARRAY_RULES, normalized))
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
    are rejected. The composition rules recurse into inline consolidated
    metadata, so an embedded child document invalid under its own rules
    is reported here, at its path.
    """
    document, problems = _merged_with_extensions(
        kwargs, extensions, GROUP_METADATA_STANDARD_KEYS_V3
    )
    normalized = _normalized(document)
    parsed: ZarrV3GroupMetadataJSON | None = None
    try:
        parsed = parse_group_metadata_v3(normalized)
    except MetadataValidationError as error:
        problems.extend(error.problems)
    problems.extend(run_rules(ZARR_V3_GROUP_RULES, normalized))
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
    _raise_if_problems(
        validate_consolidated_metadata_v3(normalized) + consolidated_entries_problems(normalized)
    )
    return cast("ZarrV3ConsolidatedMetadataJSON", normalized)


def create_zarr_v2_array_metadata_json(
    **kwargs: Unpack[ZarrV2ArrayMetadataJSON],
) -> ZarrV2ArrayMetadataJSON:
    """A validated v2 array metadata document, in-memory merged form.

    Models `.zarray` plus the sibling `.zattrs` folded in as `attributes`.
    For the strict on-disk `.zarray` shape use `create_zarr_v2_z_array_json`.
    """
    normalized = _normalized(kwargs)
    parsed: ZarrV2ArrayMetadataJSON | None = None
    problems: list[ValidationProblem] = []
    try:
        parsed = parse_array_metadata_v2(normalized)
    except MetadataValidationError as error:
        problems.extend(error.problems)
    problems.extend(run_rules(ZARR_V2_ARRAY_RULES, normalized))
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
        problems.extend(error.problems)
    _raise_if_problems(problems)
    assert parsed is not None
    return parsed


def create_zarr_v2_z_array_json(**kwargs: Unpack[ZarrV2ZArrayJSON]) -> ZarrV2ZArrayJSON:
    """A validated on-disk `.zarray` document (strict form, no `attributes`).

    Structurally checked with the merged-form parser plus a runtime
    rejection of `attributes`: the strict shape is the merged shape minus
    `attributes`, and the runtime check holds for callers the signature's
    static exclusion cannot see (`**`-splatted mappings, untyped code).
    """
    normalized = _normalized(kwargs)
    problems: list[ValidationProblem] = list(_reject_attributes(normalized))
    parsed: ZarrV2ArrayMetadataJSON | None = None
    try:
        parsed = parse_array_metadata_v2(normalized)
    except MetadataValidationError as error:
        problems.extend(error.problems)
    problems.extend(run_rules(ZARR_V2_ARRAY_RULES, normalized))
    _raise_if_problems(problems)
    assert parsed is not None
    return cast("ZarrV2ZArrayJSON", parsed)


def create_zarr_v2_z_group_json(**kwargs: Unpack[ZarrV2ZGroupJSON]) -> ZarrV2ZGroupJSON:
    """A validated on-disk `.zgroup` document (strict form, no `attributes`).

    Structurally checked with the merged-form parser plus a runtime
    rejection of `attributes`: the strict shape is the merged shape minus
    `attributes`, and the runtime check holds for callers the signature's
    static exclusion cannot see (`**`-splatted mappings, untyped code).
    """
    normalized = _normalized(kwargs)
    problems: list[ValidationProblem] = list(_reject_attributes(normalized))
    parsed: ZarrV2GroupMetadataJSON | None = None
    try:
        parsed = parse_group_metadata_v2(normalized)
    except MetadataValidationError as error:
        problems.extend(error.problems)
    _raise_if_problems(problems)
    assert parsed is not None
    return cast("ZarrV2ZGroupJSON", parsed)


def _validate_v2_consolidated_envelope(
    document: Mapping[str, object],
) -> tuple[ValidationProblem, ...]:
    """Every reason `document` is not a `.zmetadata` envelope.

    Checks the envelope only: both keys present, an integer format marker,
    and `metadata` a string-keyed mapping of JSON objects. The nested
    per-path documents are typed but deliberately not deep-validated —
    which file shape each value must have is keyed on its path suffix, a
    store-layout concern; validate entries individually with the model
    layer when reading untrusted data.
    """
    problems: list[ValidationProblem] = []
    fmt = document.get("zarr_consolidated_format")
    if "zarr_consolidated_format" not in document:
        problems.append(
            ValidationProblem(("zarr_consolidated_format",), "missing key", "missing_key")
        )
    elif isinstance(fmt, bool) or not isinstance(fmt, int):
        problems.append(
            ValidationProblem(
                ("zarr_consolidated_format",), f"expected an integer, got {fmt!r}", "invalid_type"
            )
        )
    elif fmt != 1:
        problems.append(
            ValidationProblem(
                ("zarr_consolidated_format",),
                f"expected consolidated format 1, got {fmt!r}",
                "invalid_value",
            )
        )
    if "metadata" not in document:
        problems.append(ValidationProblem(("metadata",), "missing key", "missing_key"))
    else:
        entries = document["metadata"]
        if not isinstance(entries, Mapping):
            problems.append(ValidationProblem(("metadata",), "expected a mapping", "invalid_type"))
        else:
            for key, entry in cast("Mapping[object, object]", entries).items():
                if not isinstance(key, str):
                    problems.append(
                        ValidationProblem(
                            ("metadata",), f"expected string keys, got {key!r}", "invalid_type"
                        )
                    )
                elif not isinstance(entry, Mapping):
                    problems.append(
                        ValidationProblem(
                            ("metadata", key), "expected a JSON object", "invalid_type"
                        )
                    )
                else:
                    problems.extend(
                        ValidationProblem(("metadata", key, *found.loc), found.message, found.kind)
                        for found in validate_json(cast("object", entry))
                    )
    return tuple(problems)


def create_zarr_v2_consolidated_metadata_json(
    **kwargs: Unpack[ZarrV2ConsolidatedMetadataJSON],
) -> ZarrV2ConsolidatedMetadataJSON:
    """A validated `.zmetadata` consolidated metadata document.

    The runtime pass checks the envelope — key presence, an integer
    format marker, `metadata` as a string-keyed mapping of JSON objects —
    for the callers the signature's static enforcement cannot see
    (`**`-splatted mappings, untyped code). The nested per-path documents
    are typed but not deep-validated here; validate them individually
    with the model layer when reading untrusted data.
    """
    normalized = _normalized(kwargs)
    _raise_if_problems(_validate_v2_consolidated_envelope(normalized))
    return cast("ZarrV2ConsolidatedMetadataJSON", normalized)


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
