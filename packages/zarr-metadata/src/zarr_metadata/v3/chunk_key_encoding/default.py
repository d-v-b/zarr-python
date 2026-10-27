"""
Default chunk key encoding (Zarr v3 core spec).

The chunk key for a chunk with grid index `(k, j, i, ...)` is formed
by appending `c<sep>k<sep>j<sep>i...` (where `<sep>` is `separator`).

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.v3._definition import ChunkKeyEncodingDefinition

DEFAULT_CHUNK_KEY_ENCODING_NAME: Final = "default"
"""The `name` field value of the default chunk key encoding."""

DefaultChunkKeyEncodingName = Literal["default"]
"""Literal type of the `name` field of the default chunk key encoding."""

DefaultChunkKeyEncodingSeparator = Literal["/", "."]
"""Literal type of permitted `separator` values for the default chunk key encoding.

Defaults to `"/"` if absent.
"""

DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR: Final = ("/", ".")
"""Tuple of permitted values for the `separator` field of the default chunk key encoding."""


class DefaultChunkKeyEncodingConfiguration(TypedDict, closed=True):
    """Configuration for the default chunk key encoding.

    `separator` is optional and defaults to `"/"` per spec.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-key-encodings/default/index.rst#L27-L29
    """

    separator: NotRequired[DefaultChunkKeyEncodingSeparator]
    """The character between the parts of a chunk key: `c/1/23/45` or `c.1.23.45`.

    Absent means `"/"`.
    """


class DefaultChunkKeyEncodingObject(TypedDict, closed=True):
    """Default chunk key encoding metadata in object form."""

    name: DefaultChunkKeyEncodingName
    """The chunk key encoding's name, `"default"`."""
    configuration: NotRequired[DefaultChunkKeyEncodingConfiguration]
    """The chunk key encoding's parameters.

    Absent means every parameter takes its default: a `separator` of `"/"`.
    """
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this chunk key encoding must refuse the document.

    Absent means `true`. The spec does not permit `false` for a chunk key encoding
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581).
    """


DefaultChunkKeyEncodingMetadata = DefaultChunkKeyEncodingObject | DefaultChunkKeyEncodingName
"""Permitted JSON shapes for the default chunk-key encoding metadata.

The configuration has no required keys (`separator` defaults to `"/"`),
so the short-hand-name form is permitted in addition to the object form.
"""

DEFAULT_CHUNK_KEY_ENCODING: Final = ChunkKeyEncodingDefinition(
    name=DEFAULT_CHUNK_KEY_ENCODING_NAME, configuration=DefaultChunkKeyEncodingConfiguration
)
"""The `default` chunk key encoding; its `separator` is typed, so it has no rule of its own."""


__all__ = [
    "DEFAULT_CHUNK_KEY_ENCODING",
    "DEFAULT_CHUNK_KEY_ENCODING_NAME",
    "DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR",
    "DefaultChunkKeyEncodingConfiguration",
    "DefaultChunkKeyEncodingMetadata",
    "DefaultChunkKeyEncodingName",
    "DefaultChunkKeyEncodingObject",
    "DefaultChunkKeyEncodingSeparator",
]
