"""Zarr v3 array metadata types."""

from collections.abc import Mapping
from typing import Final, Literal, NotRequired, TypeAlias

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON

ZarrV3ExtensionField: TypeAlias = JSONValue
"""The JSON value of an unknown top-level v3 metadata field.

An object carrying the literal member `must_understand: false` may be ignored.
Every other JSON shape implicitly requires understanding; recognition itself
belongs to the reader rather than this structural type.
"""


class ZarrV3ArrayMetadataJSON(TypedDict, extra_items=ZarrV3ExtensionField):
    """
    Zarr v3 array metadata document (the `zarr.json` content for an array).

    Extra keys may contain arbitrary JSON values.

    See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array-metadata
    """

    zarr_format: Literal[3]
    """The Zarr format version, `3`."""
    node_type: Literal["array"]
    """The kind of node the document describes, `"array"`."""
    data_type: ZarrV3MetadataFieldJSON
    """The data type of the array's elements, e.g. `"float64"`."""
    shape: tuple[int, ...]
    """The length of each dimension of the array."""
    chunk_grid: ZarrV3MetadataFieldJSON
    """How the array is divided into chunks, e.g. a `"regular"` grid of one chunk shape."""
    chunk_key_encoding: ZarrV3MetadataFieldJSON
    """How a chunk's grid coordinates become its key in the store."""
    fill_value: JSONValue
    """The value of an element in a chunk that was never written.

    Which JSON values it may take depends on `data_type`.
    """
    codecs: tuple[ZarrV3MetadataFieldJSON, ...]
    """The codecs that encode each chunk, in encoding order.

    Zero or more array -> array codecs, then exactly one array -> bytes codec,
    then zero or more bytes -> bytes codecs, so the list is never empty
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L974-L983).
    """
    attributes: NotRequired[Mapping[str, JSONValue]]
    """User attributes: arbitrary JSON values by string key."""
    storage_transformers: NotRequired[tuple[ZarrV3MetadataFieldJSON, ...]]
    """The storage transformers between the array and its store.

    Absent means none, as does an empty list.
    """
    dimension_names: NotRequired[tuple[str | None, ...]]
    """A name for each dimension, `null` for an unnamed one.

    As long as `shape`; absent means every dimension is unnamed.
    """


class ZarrV3ArrayMetadataJSONPartial(TypedDict, total=False, extra_items=ZarrV3ExtensionField):
    """
    Partial form of `ZarrV3ArrayMetadataJSON`: every field is `NotRequired`.

    Field annotations and `extra_items=` mirror `ZarrV3ArrayMetadataJSON` exactly.
    The only difference is `total=False`, which makes every key optional
    at the type level.

    Use this when typing dicts that intentionally hold a subset of a complete
    v3 array metadata document — e.g. test fixtures that override only a few
    fields of a base template, or callers that build a fragment to be merged
    into a complete document elsewhere.

    The `NotRequired[...]` wrappers on `attributes`, `storage_transformers`,
    and `dimension_names` are intentional: keeping them preserves byte-identical
    `__annotations__` with `ZarrV3ArrayMetadataJSON` so the `==` check in
    `tests/test_partial_equivalence.py` passes without special-casing those
    fields (PEP 655 explicitly permits `NotRequired` inside `total=False`).

    Drift between this type and `ZarrV3ArrayMetadataJSON` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[3]
    """The Zarr format version, `3`."""
    node_type: Literal["array"]
    """The kind of node the document describes, `"array"`."""
    data_type: ZarrV3MetadataFieldJSON
    """The data type of the array's elements, e.g. `"float64"`."""
    shape: tuple[int, ...]
    """The length of each dimension of the array."""
    chunk_grid: ZarrV3MetadataFieldJSON
    """How the array is divided into chunks, e.g. a `"regular"` grid of one chunk shape."""
    chunk_key_encoding: ZarrV3MetadataFieldJSON
    """How a chunk's grid coordinates become its key in the store."""
    fill_value: JSONValue
    """The value of an element in a chunk that was never written.

    Which JSON values it may take depends on `data_type`.
    """
    codecs: tuple[ZarrV3MetadataFieldJSON, ...]
    """The codecs that encode each chunk, in encoding order.

    Zero or more array -> array codecs, then exactly one array -> bytes codec,
    then zero or more bytes -> bytes codecs, so the list is never empty
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L974-L983).
    """
    attributes: NotRequired[Mapping[str, JSONValue]]
    """User attributes: arbitrary JSON values by string key."""
    storage_transformers: NotRequired[tuple[ZarrV3MetadataFieldJSON, ...]]
    """The storage transformers between the array and its store.

    Absent means none, as does an empty list.
    """
    dimension_names: NotRequired[tuple[str | None, ...]]
    """A name for each dimension, `null` for an unnamed one.

    As long as `shape`; absent means every dimension is unnamed.
    """


ZarrV3ArrayMetadataStoreKey = Literal["zarr.json"]
"""Literal type of the store key holding a v3 array's metadata document."""

ZARR_V3_ARRAY_METADATA_STORE_KEY: Final[ZarrV3ArrayMetadataStoreKey] = "zarr.json"
"""The store key a v3 array's metadata document is persisted under.

v3 uses one key for both node types; the document's `node_type` field
distinguishes an array from a group.
"""


__all__ = [
    "ZARR_V3_ARRAY_METADATA_STORE_KEY",
    "ZarrV3ArrayMetadataJSON",
    "ZarrV3ArrayMetadataJSONPartial",
    "ZarrV3ArrayMetadataStoreKey",
    "ZarrV3ExtensionField",
]
