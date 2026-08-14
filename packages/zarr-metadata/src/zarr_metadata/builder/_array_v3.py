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
    """A deep copy of `inner` with JSON arrays materialized as tuples.

    Normalizing at ingestion (rather than deep-copying accessors' return
    values alone) is what makes the no-shared-mutable-state contract hold
    for runtime callers the type checker never saw: a list smuggled in as
    `shape` becomes a tuple here, so no property can hand out a reference
    whose mutation would corrupt the builder behind the eager rules' back.
    """
    return cast("ZarrV3ArrayMetadataJSONPartial", arrays_to_tuples(copy.deepcopy(inner)))


class ZarrV3ArrayMetadataBuilder:
    """Immutable accumulator for a v3 array metadata document.

    Holds a (possibly incomplete) `ZarrV3ArrayMetadataJSONPartial` — the
    same plain-JSON shapes the document itself uses, never wrapper objects
    — and hands out evolved copies:

        doc = (
            ZarrV3ArrayMetadataBuilder()
            .evolve(zarr_format=3, node_type="array")
            .evolve(shape=(4, 4), data_type="uint8", fill_value=0)
            .evolve(
                chunk_grid={"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
                chunk_key_encoding="default",
                codecs=("bytes",),
            )
            .build()
        )

    `evolve` is the only setter for standard fields; it is typed via
    `Unpack[ZarrV3ArrayMetadataJSONPartial]`, so at literal-keyword call
    sites wrong value types are rejected statically, and checkers without
    PEP 728 support reject unknown keyword names too (PEP 728 checkers
    accept them as extension items; `**`-splatted calls bypass both).
    Extension fields go through `evolve_extension`, which also enforces
    the no-shadowing rule at runtime. Keys are removed — not set to a
    sentinel — with `without`; an absent key means UNSET, while a stored
    `None` means JSON `null`, and the two never convert into one another.

    After every change the semantic rules whose dependencies are all
    present fire over the merged state (all of them, not just rules
    touching the changed fields — a rule that passed under an old value
    must be re-judged under the new one), raising
    `MetadataValidationError` with every problem found. Fields may be set
    in any order; coupled fields are checked as soon as all of them exist,
    and a conflict between two fields is escaped by evolving both at once.

    `build` validates structurally and semantically and returns the plain
    `ZarrV3ArrayMetadataJSON` dict; `to_partial_json` returns whatever is
    accumulated so far under an honest partial type. There is no
    `build_unchecked`.

    Instances never share mutable state: input mappings are deep-copied in
    with JSON arrays materialized as tuples at every depth, and every
    accessor and serializer deep-copies out. The tuple normalization also
    makes equality spelling-insensitive: two builders holding the same
    document compare equal whether their arrays arrived as lists (e.g.
    straight from `json.loads`) or as tuples. Reading a field via its
    property answers `UNSET` (PEP 661 sentinel; test with `is UNSET`)
    when the key is absent.
    """

    __slots__ = ("_inner",)
    _inner: ZarrV3ArrayMetadataJSONPartial

    def __init__(self, inner: ZarrV3ArrayMetadataJSONPartial | None = None) -> None:
        self._inner = _normalized(inner) if inner is not None else {}
        self._check(changed=frozenset(self._inner.keys()))

    # -- evolution ----------------------------------------------------------

    def evolve(self, **kwargs: Unpack[ZarrV3ArrayMetadataJSONPartial]) -> Self:
        """A new builder with the given standard fields replaced.

        Each given field fully replaces its previous value. Raises
        `MetadataValidationError` if the merged state violates any
        dependency-complete semantic rule.
        """
        return self._evolved(cast("ZarrV3ArrayMetadataJSONPartial", dict(kwargs)))

    def evolve_extension(self, name: str, value: ZarrV3ExtensionField) -> Self:
        """A new builder with extension field `name` set to `value`.

        Extension fields are the document keys outside the standard v3
        array metadata keys; `name` must not collide with a standard key.
        This is a separate method (rather than extra `evolve` kwargs)
        because type checkers without PEP 728 support reject unknown
        keyword names statically.
        """
        if name in ARRAY_METADATA_STANDARD_KEYS_V3:
            raise MetadataValidationError(
                [
                    ValidationProblem(
                        (name,),
                        f"{name!r} is a standard v3 array metadata key; set it with evolve()",
                        "invalid_value",
                    )
                ]
            )
        return self._evolved(cast("ZarrV3ArrayMetadataJSONPartial", {name: value}))

    def without(self, *keys: str) -> Self:
        """A new builder with the given keys absent.

        Removing an already-absent key is a no-op. This is the only way to
        unset a field: `evolve` never stores an "unset" sentinel, so a key
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
            earlier = rule.keys - changed
            just_set = rule.keys & changed
            if len(earlier) != 0 and len(just_set) != 0:
                hint = (
                    f" [{', '.join(sorted(just_set))} set in this call conflicts with "
                    f"{', '.join(sorted(earlier))} set earlier; evolve() can change "
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
