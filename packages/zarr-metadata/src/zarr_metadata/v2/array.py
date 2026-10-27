"""Zarr v2 array metadata types."""

from collections.abc import Mapping
from typing import Final, Literal, NotRequired

from typing_extensions import TypeAliasType, TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v2.codec import ZarrV2CodecMetadata

ZarrV2DataTypeMetadata = TypeAliasType(
    "ZarrV2DataTypeMetadata",
    str
    | tuple[
        tuple[str, "ZarrV2DataTypeMetadata"]
        | tuple[str, "ZarrV2DataTypeMetadata", tuple[int, ...]],
        ...,
    ],
)
"""The v2 dtype representation.

Either a numpy-style dtype string (e.g. `"<f8"`, `"|S10"`) or a tuple of
field records describing a structured dtype. Each field record is either
a 2-tuple `(name, datatype)` or a 3-tuple `(name, datatype, shape)`
(the 3-tuple form indicates a subarray field). A field datatype may itself
be another structured dtype.

Endianness is encoded in the prefix character of the dtype string;
parsing it out is a caller concern, not part of this type.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#data-type-encoding
"""

ZarrV2ArrayOrder = Literal["C", "F"]
"""Literal type of permitted values for the `order` field of v2 array metadata.

`"C"` (row-major) or `"F"` (column-major) — the in-chunk byte layout.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""

ZARR_V2_ARRAY_ORDER: Final = ("C", "F")
"""Tuple of permitted values for the `order` field of v2 array metadata."""

ZarrV2ArrayDimensionSeparator = Literal[".", "/"]
"""Literal type of permitted values for the `dimension_separator` field of v2 array metadata.

`"."` (legacy default) joins chunk grid coordinates as `0.0`, `0.1`, ...
`"/"` joins them as `0/0`, `0/1`, ... yielding nested directories.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""

ZARR_V2_ARRAY_DIMENSION_SEPARATOR: Final = (".", "/")
"""Tuple of permitted values for the `dimension_separator` field of v2 array metadata."""


class ZarrV2ZArrayJSON(TypedDict):
    """
    On-disk `.zarray` file content.

    Strict shape of the JSON document persisted at `<path>/.zarray` for
    a v2 array. User attributes live in a sibling `.zattrs` file and are
    NOT part of this type; see `ZarrV2ZAttrsJSON`.

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    zarr_format: Literal[2]
    """The Zarr format version, `2`."""
    shape: tuple[int, ...]
    """The length of each dimension of the array."""
    chunks: tuple[int, ...]
    """The length of each dimension of a chunk.

    Every chunk has this shape, so a chunk at the array's edge overhangs it
    where a length does not divide evenly; what the overhang holds is undefined.
    """
    dtype: ZarrV2DataTypeMetadata
    """The data type of the array's elements.

    A NumPy typestr such as `"<f8"`, or a list of `[name, datatype]` or
    `[name, datatype, shape]` field records for a structured data type.
    """
    compressor: ZarrV2CodecMetadata | None
    """The configuration of the codec that compresses each chunk, or `null` for none."""
    fill_value: JSONValue
    """The value of an element in a chunk that was never written, or `null` for none.

    With `null`, what such a chunk holds is undefined. A non-finite float is
    `"NaN"`, `"Infinity"` or `"-Infinity"`, and the fill value of a fixed-length
    byte string or structured data type is a Base64 string
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L181-L194).
    """
    order: ZarrV2ArrayOrder
    """The layout of elements within each chunk: `"C"` (row-major) or `"F"` (column-major)."""
    filters: tuple[ZarrV2CodecMetadata, ...] | None
    """Codecs applied to each chunk before the compressor, in order, or `null` for none."""
    dimension_separator: NotRequired[ZarrV2ArrayDimensionSeparator]
    """The separator between a chunk's grid coordinates in its key.

    Absent means `"."`, giving keys like `0.0`; `"/"` gives keys like `0/0`.
    """


class ZarrV2ArrayMetadataJSON(TypedDict, extra_items=JSONValue):
    """
    Zarr v2 array metadata document, in-memory merged form.

    Models the union of `.zarray` (the spec-defined fields, https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L51-L92)
    and `.zattrs` (user attributes, https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L323-L330). On disk, attributes live in a sibling `.zattrs` file
    and are not part of `.zarray`; this type folds them in as the
    `attributes` field so a single TypedDict represents the complete
    in-memory state of a v2 array node. Consumers that read or write a
    real `.zarray` file should split / merge `attributes` accordingly,
    or use `ZarrV2ZArrayJSON` (strict on-disk) plus `ZarrV2ZAttrsJSON` directly.

    Open: other keys "SHOULD NOT be present ... and SHOULD be ignored by
    implementations" (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L91-L92), and each
    other member is a JSON value.

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    zarr_format: Literal[2]
    """The Zarr format version, `2`."""
    shape: tuple[int, ...]
    """The length of each dimension of the array."""
    chunks: tuple[int, ...]
    """The length of each dimension of a chunk.

    Every chunk has this shape, so a chunk at the array's edge overhangs it
    where a length does not divide evenly; what the overhang holds is undefined.
    """
    dtype: ZarrV2DataTypeMetadata
    """The data type of the array's elements.

    A NumPy typestr such as `"<f8"`, or a list of `[name, datatype]` or
    `[name, datatype, shape]` field records for a structured data type.
    """
    compressor: ZarrV2CodecMetadata | None
    """The configuration of the codec that compresses each chunk, or `null` for none."""
    fill_value: JSONValue
    """The value of an element in a chunk that was never written, or `null` for none.

    With `null`, what such a chunk holds is undefined. A non-finite float is
    `"NaN"`, `"Infinity"` or `"-Infinity"`, and the fill value of a fixed-length
    byte string or structured data type is a Base64 string
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L181-L194).
    """
    order: ZarrV2ArrayOrder
    """The layout of elements within each chunk: `"C"` (row-major) or `"F"` (column-major)."""
    filters: tuple[ZarrV2CodecMetadata, ...] | None
    """Codecs applied to each chunk before the compressor, in order, or `null` for none."""
    dimension_separator: NotRequired[ZarrV2ArrayDimensionSeparator]
    """The separator between a chunk's grid coordinates in its key.

    Absent means `"."`, giving keys like `0.0`; `"/"` gives keys like `0/0`.
    """
    attributes: NotRequired[Mapping[str, JSONValue]]
    """User attributes from the sibling `.zattrs` file (not part of `.zarray`).

    See the class docstring for the rationale behind the merged representation.
    """


class ZarrV2ArrayMetadataJSONPartial(TypedDict, total=False, extra_items=JSONValue):
    """
    Partial form of `ZarrV2ArrayMetadataJSON`: every field is `NotRequired`.

    Field annotations mirror `ZarrV2ArrayMetadataJSON` exactly. The only difference is
    `total=False`, which makes every key optional at the type level.

    Use this when typing dicts that intentionally hold a subset of a complete
    v2 array metadata document — e.g. test fixtures that override only a few
    fields of a base template, or callers that build a fragment to be merged
    into a complete document elsewhere.

    The `NotRequired[...]` wrappers on `dimension_separator` and `attributes`
    are intentional: keeping them preserves byte-identical `__annotations__`
    with `ZarrV2ArrayMetadataJSON` so the `==` check in
    `tests/test_partial_equivalence.py` passes without special-casing those
    fields (PEP 655 explicitly permits `NotRequired` inside `total=False`).

    Like `ZarrV2ArrayMetadataJSON`, it is open: the v2 spec has no
    extension-field concept, and other `.zarray` keys "SHOULD be ignored by
    implementations" (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L91-L92), so each other member is a
    JSON value and none is modeled further.

    Drift between this type and `ZarrV2ArrayMetadataJSON` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[2]
    """The Zarr format version, `2`."""
    shape: tuple[int, ...]
    """The length of each dimension of the array."""
    chunks: tuple[int, ...]
    """The length of each dimension of a chunk.

    Every chunk has this shape, so a chunk at the array's edge overhangs it
    where a length does not divide evenly; what the overhang holds is undefined.
    """
    dtype: ZarrV2DataTypeMetadata
    """The data type of the array's elements.

    A NumPy typestr such as `"<f8"`, or a list of `[name, datatype]` or
    `[name, datatype, shape]` field records for a structured data type.
    """
    compressor: ZarrV2CodecMetadata | None
    """The configuration of the codec that compresses each chunk, or `null` for none."""
    fill_value: JSONValue
    """The value of an element in a chunk that was never written, or `null` for none.

    With `null`, what such a chunk holds is undefined. A non-finite float is
    `"NaN"`, `"Infinity"` or `"-Infinity"`, and the fill value of a fixed-length
    byte string or structured data type is a Base64 string
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L181-L194).
    """
    order: ZarrV2ArrayOrder
    """The layout of elements within each chunk: `"C"` (row-major) or `"F"` (column-major)."""
    filters: tuple[ZarrV2CodecMetadata, ...] | None
    """Codecs applied to each chunk before the compressor, in order, or `null` for none."""
    dimension_separator: NotRequired[ZarrV2ArrayDimensionSeparator]
    """The separator between a chunk's grid coordinates in its key.

    Absent means `"."`, giving keys like `0.0`; `"/"` gives keys like `0/0`.
    """
    attributes: NotRequired[Mapping[str, JSONValue]]
    """User attributes from the sibling `.zattrs` file (not part of `.zarray`).

    See the class docstring for the rationale behind the merged representation.
    """


ZarrV2ArrayMetadataStoreKey = Literal[".zarray"]
"""Literal type of the store key holding a v2 array's metadata document."""

ZARR_V2_ARRAY_METADATA_STORE_KEY: Final[ZarrV2ArrayMetadataStoreKey] = ".zarray"
"""The store key a v2 array's metadata document is persisted under."""


__all__ = [
    "ZARR_V2_ARRAY_DIMENSION_SEPARATOR",
    "ZARR_V2_ARRAY_METADATA_STORE_KEY",
    "ZARR_V2_ARRAY_ORDER",
    "ZarrV2ArrayDimensionSeparator",
    "ZarrV2ArrayMetadataJSON",
    "ZarrV2ArrayMetadataJSONPartial",
    "ZarrV2ArrayMetadataStoreKey",
    "ZarrV2ArrayOrder",
    "ZarrV2DataTypeMetadata",
    "ZarrV2ZArrayJSON",
]
