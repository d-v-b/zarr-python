"""
Top-level cross-version primitives for Zarr metadata.

Version-specific types live under `zarr_metadata.v2` and `zarr_metadata.v3`.
Codec and dtype spec types live under `zarr_metadata.v3.codec` and
`zarr_metadata.v3.data_type`.
"""

from collections.abc import Mapping, Sequence
from typing import NotRequired

from typing_extensions import ReadOnly, TypeAliasType, TypedDict

JSONValue = TypeAliasType(
    "JSONValue",
    int | float | bool | str | Sequence["JSONValue"] | Mapping[str, "JSONValue"] | None,
)
"""A recursive type alias for JSON-encodable values.

Defined via `TypeAliasType` (rather than a plain `TypeAlias`) so the
self-reference is a named recursion point that pydantic can resolve when
building a `TypeAdapter`; a bare recursive `TypeAlias` raises
`PydanticUserError`/`RecursionError` at validation time.

The array arm is the covariant `Sequence` rather than the invariant
`list["JSONValue"] | tuple["JSONValue", ...]`, so values typed with a
*narrower* element type still count as JSON values: a `list[str]` field on a
TypedDict is assignable to `JSONValue` under `Sequence` but not under
`list[JSONValue]` (`list` is invariant in its element type, and pyright's
diagnostic for that failure suggests exactly this change). This is what lets
downstream TypedDicts give their fields precise types (`Sequence[str]`,
`list[int]`, ...) while remaining assignable to `Mapping[str, JSONValue]`.
The type-level cost, accepted deliberately: `Sequence` says nothing about the
concrete container, and it admits `str`/`bytes` (`str` was already a union
arm); runtime code narrowing a JSON array must exclude `str`/`bytes`/
`bytearray` regardless of how this alias is spelled.
"""


class ZarrV3NamedConfigJSON(TypedDict, closed=True):
    """
    Externally-tagged union member for a metadata field.

    The optional `configuration` mapping holds arbitrary JSON-encodable
    values. `must_understand` is implicitly true when absent.

    `name` and `configuration` are `ReadOnly` (PEP 705) so that concrete
    entity types — `BloscCodecObject`, `RegularChunkGridObject`, and the
    rest — are assignable to this type, and therefore to
    `ZarrV3MetadataFieldJSON`. Without `ReadOnly` both items are invariant,
    so a concrete `name: Literal["blosc"]` does not satisfy `name: str`, and
    a required `configuration` does not satisfy a `NotRequired` one. That
    made the package's own codec types unusable in the very fields they
    describe (`codecs`, `data_type`, `chunk_grid`, ...), and made
    `TypeIs`-based codec classification impossible to declare, since `TypeIs`
    requires the narrowed type to be assignable to the input type.

    `must_understand` stays writable: nothing needs to narrow it, and
    keeping it mutable lets writers set it on an already-constructed field.

    The type is `closed` (PEP 728): the spec's named-configuration envelope
    has exactly these three members, and closing it is also what makes this
    type — and every concrete entity type embedding it, e.g. the
    `sharding_indexed` configuration's inner `codecs` list — assignable to
    `Mapping[str, JSONValue]` (i.e. usable as a `JSONValue`).
    """

    name: ReadOnly[str]
    configuration: NotRequired[ReadOnly[Mapping[str, JSONValue]]]
    must_understand: NotRequired[bool]
