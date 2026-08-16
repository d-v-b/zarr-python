"""
Constructing chunk key encodings from Zarr v3 JSON metadata.

Dispatch is over a **closed set**: the two encodings the v3 core spec
defines. There is deliberately no registration API, no entry point group,
and no notion of a third-party encoding here.

Chunk key encoding is an extension point, so an open set is where this ends
up eventually -- but the machinery for that (registration, discovery,
support tiers, fault isolation around third-party code) is the same for
every v3 extension point, and belongs in one shared place rather than
reinvented per package. Until that package exists, this one stays a library
of the spec-defined encodings. Note that Zarr v3 already has an established
entry point group for this extension point, ``zarr.chunk_key_encoding``,
which `zarr` itself scans; adding a second, competing one is exactly the
commitment worth not making early, since an entry point group is a
compatibility ratchet the moment a third party publishes against it.

The signatures below are the ones an open set would use too, so growing into
a registry later does not change them.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, NotRequired, cast

from typing_extensions import TypeAliasType, TypedDict

from zarr_chunk_key_encoding.abc import ChunkKeyEncoding, ChunkKeyEncodingJSON
from zarr_chunk_key_encoding.default import DefaultChunkKeyEncoding
from zarr_chunk_key_encoding.errors import (
    ChunkKeyConfigurationError,
    UnknownChunkKeyEncodingError,
)
from zarr_chunk_key_encoding.separator import Separator
from zarr_chunk_key_encoding.v2 import V2ChunkKeyEncoding

if TYPE_CHECKING:
    from zarr_metadata import JSONValue

__all__ = [
    "CHUNK_KEY_ENCODINGS",
    "ChunkKeyEncodingLike",
    "ChunkKeyEncodingParams",
    "chunk_key_encoding_from_json",
    "parse_chunk_key_encoding",
]

CHUNK_KEY_ENCODINGS: Final[Mapping[str, type[ChunkKeyEncoding]]] = {
    DefaultChunkKeyEncoding.name: DefaultChunkKeyEncoding,
    V2ChunkKeyEncoding.name: V2ChunkKeyEncoding,
}
"""The chunk key encodings defined by the Zarr v3 core spec, by name.

A closed mapping, not a registry: it is not mutable, and nothing discovers
additions to it.
"""


class ChunkKeyEncodingParams(TypedDict, closed=True):
    """Flat keyword form for specifying a chunk key encoding.

    A convenience input shape (``{"name": ..., "separator": ...}``) accepted
    by `parse_chunk_key_encoding`; it is not valid Zarr metadata. Closed
    (PEP 728): exactly ``name`` and optionally ``separator``.
    """

    name: str
    separator: NotRequired[Separator]


ChunkKeyEncodingLike = TypeAliasType(
    "ChunkKeyEncodingLike",
    ChunkKeyEncoding | ChunkKeyEncodingParams | ChunkKeyEncodingJSON,
)
"""Anything `parse_chunk_key_encoding` can turn into a `ChunkKeyEncoding`:
an instance, the flat params form, a short-hand name string, or the
named-configuration JSON object form."""


def get_chunk_key_encoding_class(name: str) -> type[ChunkKeyEncoding]:
    """Look up a spec-defined chunk key encoding class by name.

    Parameters
    ----------
    name : str
        The ``name`` field of the encoding's v3 metadata.

    Returns
    -------
    type[ChunkKeyEncoding]
        The class implementing that encoding.

    Raises
    ------
    UnknownChunkKeyEncodingError
        If the name is not one the Zarr v3 core spec defines.
    """
    try:
        return CHUNK_KEY_ENCODINGS[name]
    except KeyError:
        raise UnknownChunkKeyEncodingError(name, tuple(CHUNK_KEY_ENCODINGS)) from None


def chunk_key_encoding_from_json(data: ChunkKeyEncodingJSON) -> ChunkKeyEncoding:
    """Construct a chunk key encoding from Zarr v3 JSON metadata.

    Resolves the class by ``name``, then delegates to its ``from_json``.

    Parameters
    ----------
    data : ChunkKeyEncodingJSON
        The short-hand name string or named-configuration object form.

    Returns
    -------
    ChunkKeyEncoding
        The constructed encoding.

    Raises
    ------
    ChunkKeyConfigurationError
        If the metadata carries no usable ``name``, or the named class
        rejects the metadata.
    UnknownChunkKeyEncodingError
        If the name is not one the Zarr v3 core spec defines.
    """
    # Widen to `object` (the cast defeats assignment narrowing) so the shape
    # checks below stay meaningful runtime guards for untyped callers instead
    # of being flagged as unnecessary.
    data_obj = cast("object", data)
    if isinstance(data_obj, str):
        name = data_obj
    elif isinstance(data_obj, Mapping):
        # Decoded JSON objects always have string keys; assert that view for
        # the type checker.
        name = cast("Mapping[str, JSONValue]", data_obj).get("name")
        if not isinstance(name, str):
            raise ChunkKeyConfigurationError(
                f"Invalid chunk key encoding metadata: expected a 'name' key "
                f"with a string value in {data!r}."
            )
    else:
        raise ChunkKeyConfigurationError(
            f"Invalid chunk key encoding metadata: expected a name string or "
            f"a JSON object, got {data!r}."
        )
    return get_chunk_key_encoding_class(name).from_json(data)


def parse_chunk_key_encoding(data: ChunkKeyEncodingLike) -> ChunkKeyEncoding:
    """Coerce any chunk-key-encoding-like value to a `ChunkKeyEncoding`.

    Accepts, in addition to everything `chunk_key_encoding_from_json`
    accepts: existing `ChunkKeyEncoding` instances (returned unchanged) and
    the flat `ChunkKeyEncodingParams` form
    (``{"name": ..., "separator": ...}``).

    Parameters
    ----------
    data : ChunkKeyEncodingLike
        The value to coerce.

    Returns
    -------
    ChunkKeyEncoding
        The coerced encoding.

    Raises
    ------
    ChunkKeyConfigurationError
        If the input does not describe a valid encoding.
    UnknownChunkKeyEncodingError
        If the name is not one the Zarr v3 core spec defines.
    """
    if isinstance(data, ChunkKeyEncoding):
        return data
    if isinstance(data, Mapping) and "separator" in data:
        keys = set(data)
        if keys == {"name", "separator"}:
            return chunk_key_encoding_from_json(
                {"name": data["name"], "configuration": {"separator": data["separator"]}}
            )
    return chunk_key_encoding_from_json(data)
