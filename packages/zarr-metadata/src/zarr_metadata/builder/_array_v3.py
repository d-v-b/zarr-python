"""Incremental, validated construction of v3 array metadata documents."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal, Self, cast

from typing_extensions import Unpack

from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import (
    ARRAY_METADATA_STANDARD_KEYS_V3,
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    parse_array_metadata_v3,
)
from zarr_metadata.rules import ZARR_V3_ARRAY_RULES, applicable, run_rules

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from zarr_metadata._common import JSONValue
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
    from zarr_metadata.v3.array import (
        ZarrV3ArrayMetadataJSON,
        ZarrV3ArrayMetadataJSONPartial,
        ZarrV3ExtensionField,
    )


def _normalized(inner: ZarrV3ArrayMetadataJSONPartial) -> ZarrV3ArrayMetadataJSONPartial:
    """Copy `inner` and convert JSON arrays to tuples."""
    return cast("ZarrV3ArrayMetadataJSONPartial", arrays_to_tuples(copy.deepcopy(inner)))


class ZarrV3ArrayMetadataBuilder:
    """Immutable accumulator for a v3 array metadata document.

    Holds a possibly incomplete `ZarrV3ArrayMetadataJSONPartial` and
    returns updated copies:

        doc = (
            ZarrV3ArrayMetadataBuilder()
            .with_fields(zarr_format=3, node_type="array")
            .with_fields(shape=(4, 4), data_type="uint8", fill_value=0)
            .with_fields(
                chunk_grid={"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
                chunk_key_encoding="default",
                codecs=("bytes",),
            )
            .build()
        )

    `with_fields` types standard fields; PEP 728 checkers also accept extension
    fields. `with_extension` works across checkers and rejects standard
    names. `without_fields` removes keys. Properties return `UNSET` for absent
    fields, distinct from JSON `null`.

    Applicable composition rules run after every change. `build` adds
    structural validation and returns a complete document. Inputs and
    outputs are copied, and JSON arrays normalize to tuples.
    """

    __slots__ = ("_inner",)
    _inner: ZarrV3ArrayMetadataJSONPartial

    def __init__(self, inner: ZarrV3ArrayMetadataJSONPartial | None = None) -> None:
        self._inner = _normalized(inner) if inner is not None else {}
        self._check(changed=frozenset(self._inner.keys()))

    # -- updates ------------------------------------------------------------

    def with_fields(self, **kwargs: Unpack[ZarrV3ArrayMetadataJSONPartial]) -> Self:
        """A new builder with the given fields replaced.

        Each given field fully replaces its previous value. Raises
        `MetadataValidationError` if the merged state violates any
        dependency-complete semantic rule.
        """
        return self._evolved(cast("ZarrV3ArrayMetadataJSONPartial", dict(kwargs)))

    def with_extension(self, name: str, value: ZarrV3ExtensionField) -> Self:
        """A new builder with extension field `name` set to `value`.

        Extension fields are the document keys outside the standard v3
        array metadata keys; `name` must not collide with a standard key.
        This method supports extension names on checkers without PEP 728.
        """
        if name in ARRAY_METADATA_STANDARD_KEYS_V3:
            raise MetadataValidationError(
                [
                    ValidationProblem(
                        (name,),
                        f"{name!r} is a standard v3 array metadata key; set it via with_fields()",
                        "invalid_value",
                    )
                ]
            )
        return self._evolved(cast("ZarrV3ArrayMetadataJSONPartial", {name: value}))

    def without_fields(self, *keys: str) -> Self:
        """A new builder with the given keys absent.

        Removing an already-absent key is a no-op. This is the only way to
        unset a field: `with_fields` never stores an "unset" sentinel, so a key
        is UNSET exactly when it is not in the document.
        """
        inner = cast(
            "ZarrV3ArrayMetadataJSONPartial",
            {key: value for key, value in self._inner.items() if key not in keys},
        )
        return type(self)(inner)

    def _evolved(self, changes: ZarrV3ArrayMetadataJSONPartial) -> Self:
        new = type(self).__new__(type(self))
        new._inner = _normalized(cast("ZarrV3ArrayMetadataJSONPartial", {**self._inner, **changes}))
        new._check(changed=frozenset(changes.keys()))
        return new

    def _check(self, changed: AbstractSet[str]) -> None:
        """Run every dependency-complete rule; raise with all problems found.

        `changed` names the keys set by the triggering call, used to
        attribute a conflict between a just-set field and one set earlier.
        """
        problems: list[ValidationProblem] = []
        for rule in applicable(ZARR_V3_ARRAY_RULES, self._inner.keys()):
            found = rule.check(self._inner)
            if len(found) == 0:
                continue
            earlier = rule.requires - changed
            just_set = rule.requires & changed
            if len(earlier) != 0 and len(just_set) != 0:
                hint = (
                    f" [{', '.join(sorted(just_set))} set in this call conflicts with "
                    f"{', '.join(sorted(earlier))} set earlier; with_fields() can change "
                    "both at once]"
                )
                found = tuple(
                    ValidationProblem(problem.loc, problem.message + hint, problem.kind)
                    for problem in found
                )
            problems.extend(found)
        if len(problems) != 0:
            raise MetadataValidationError(problems)

    # -- output -------------------------------------------------------------

    def build(self) -> ZarrV3ArrayMetadataJSON:
        """The complete, validated metadata document.

        Validates structurally (required keys derived from the TypedDict,
        field shapes) via the model layer's parser, then semantically via
        the full rule set, and raises `MetadataValidationError` carrying
        every problem from both passes. The returned dict shares no
        mutable state with the builder.
        """
        structural: tuple[ValidationProblem, ...] = ()
        parsed: ZarrV3ArrayMetadataJSON | None = None
        try:
            parsed = parse_array_metadata_v3(copy.deepcopy(dict(self._inner)))
        except MetadataValidationError as error:
            structural = error.problems
        semantic = run_rules(ZARR_V3_ARRAY_RULES, self._inner)
        if len(structural) != 0 or len(semantic) != 0:
            raise MetadataValidationError(structural + semantic)
        assert parsed is not None
        return parsed

    def to_partial_json(self) -> dict[str, JSONValue]:
        """The accumulated document fragment, under an honest partial type.

        Unlike `build`, this always succeeds. The name says "partial"
        because the output is not necessarily a valid metadata document —
        do not persist it to a store as one. Key absence is preserved
        exactly: a key missing from the builder is missing here, and a
        stored `None` (JSON `null`) is emitted as `null`, at any depth.
        """
        return cast("dict[str, JSONValue]", copy.deepcopy(dict(self._inner)))

    # -- introspection ------------------------------------------------------

    @property
    def zarr_format(self) -> Literal[3] | UNSET:
        return copy.deepcopy(self._inner.get("zarr_format", UNSET))

    @property
    def node_type(self) -> Literal["array"] | UNSET:
        return copy.deepcopy(self._inner.get("node_type", UNSET))

    @property
    def shape(self) -> tuple[int, ...] | UNSET:
        return copy.deepcopy(self._inner.get("shape", UNSET))

    @property
    def data_type(self) -> ZarrV3MetadataFieldJSON | UNSET:
        return copy.deepcopy(self._inner.get("data_type", UNSET))

    @property
    def chunk_grid(self) -> ZarrV3MetadataFieldJSON | UNSET:
        return copy.deepcopy(self._inner.get("chunk_grid", UNSET))

    @property
    def chunk_key_encoding(self) -> ZarrV3MetadataFieldJSON | UNSET:
        return copy.deepcopy(self._inner.get("chunk_key_encoding", UNSET))

    @property
    def fill_value(self) -> JSONValue | UNSET:
        return copy.deepcopy(self._inner.get("fill_value", UNSET))

    @property
    def codecs(self) -> tuple[ZarrV3MetadataFieldJSON, ...] | UNSET:
        return copy.deepcopy(self._inner.get("codecs", UNSET))

    @property
    def attributes(self) -> Mapping[str, JSONValue] | UNSET:
        return copy.deepcopy(self._inner.get("attributes", UNSET))

    @property
    def storage_transformers(self) -> tuple[ZarrV3MetadataFieldJSON, ...] | UNSET:
        return copy.deepcopy(self._inner.get("storage_transformers", UNSET))

    @property
    def dimension_names(self) -> tuple[str | None, ...] | UNSET:
        return copy.deepcopy(self._inner.get("dimension_names", UNSET))

    @property
    def extension_fields(self) -> dict[str, ZarrV3ExtensionField]:
        """The accumulated extension fields (keys outside the standard set)."""
        return {
            key: copy.deepcopy(cast("ZarrV3ExtensionField", value))
            for key, value in self._inner.items()
            if key not in ARRAY_METADATA_STANDARD_KEYS_V3
        }

    # -- value semantics ----------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZarrV3ArrayMetadataBuilder):
            return NotImplemented
        return self._inner == other._inner

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_partial_json()!r})"


__all__ = [
    "ZarrV3ArrayMetadataBuilder",
]
