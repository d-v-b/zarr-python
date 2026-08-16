"""
The chunk key encoding registry.

A process-wide mapping from encoding name to `ChunkKeyEncoding` subclass,
modeled on the runtime plugin registry in the `zarrs_chunk_key_encoding`
Rust crate: encodings register under their metadata ``name``, JSON metadata
is resolved to a class by name (`chunk_key_encoding_from_json`), and
registrations can be reversed (`unregister_chunk_key_encoding`).

Third-party packages can register encodings in two ways:

- imperatively, by calling `register_chunk_key_encoding` at import time, or
- declaratively, via the ``zarr_chunk_key_encoding`` entry point group. Each
  entry point must load to a `ChunkKeyEncoding` subclass; entry points are
  loaded lazily, the first time a name lookup would otherwise fail.

The built-in ``default`` and ``v2`` encodings are registered when the
`zarr_chunk_key_encoding` package is imported.

Every registered encoding carries a
`zarr_chunk_key_encoding.support.ChunkKeyEncodingSupport` level saying
whether it is defined by the core spec, registered in `zarr-extensions`, or
neither, so a consumer can accept only the tiers it is willing to honour --
see `registered_chunk_key_encodings` and `get_chunk_key_encoding_support`.
"""

from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Final, NotRequired, cast

from typing_extensions import TypeAliasType, TypedDict

from zarr_chunk_key_encoding.abc import ChunkKeyEncoding, ChunkKeyEncodingJSON
from zarr_chunk_key_encoding.errors import (
    ChunkKeyConfigurationError,
    ChunkKeyRegistryError,
    UnknownChunkKeyEncodingError,
)
from zarr_chunk_key_encoding.separator import Separator
from zarr_chunk_key_encoding.support import (
    CORE_CHUNK_KEY_ENCODING_NAMES,
    ChunkKeyEncodingSupport,
)

if TYPE_CHECKING:
    from zarr_metadata import JSONValue

__all__ = [
    "ENTRY_POINT_GROUP",
    "ChunkKeyEncodingLike",
    "ChunkKeyEncodingParams",
    "chunk_key_encoding_from_json",
    "get_chunk_key_encoding_class",
    "get_chunk_key_encoding_support",
    "parse_chunk_key_encoding",
    "register_chunk_key_encoding",
    "registered_chunk_key_encodings",
    "unregister_chunk_key_encoding",
]

ENTRY_POINT_GROUP: Final = "zarr_chunk_key_encoding"
"""The entry point group scanned for third-party chunk key encodings."""

_registry: dict[str, type[ChunkKeyEncoding]] = {}
_entry_points_loaded: bool = False


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


def _load_entry_points() -> None:
    """Register encodings declared via the entry point group, at most once.

    Entry points never displace explicit registrations: a name that is
    already registered is skipped.
    """
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        cls = entry_point.load()
        if not (isinstance(cls, type) and issubclass(cls, ChunkKeyEncoding)):
            raise ChunkKeyRegistryError(
                f"Entry point {entry_point.name!r} in group {ENTRY_POINT_GROUP!r} "
                f"loaded {cls!r}, which is not a ChunkKeyEncoding subclass."
            )
        if cls.name not in _registry:
            _registry[cls.name] = cls


def register_chunk_key_encoding(cls: type[ChunkKeyEncoding], *, overwrite: bool = False) -> None:
    """Register a chunk key encoding class under its ``name``.

    Registering the same class again is a no-op. Registering a different
    class under an already-registered name requires ``overwrite=True``.

    Parameters
    ----------
    cls : type[ChunkKeyEncoding]
        The encoding class to register. Its ``name`` class variable is the
        registry key.
    overwrite : bool
        Whether to replace an existing registration for the same name.

    Raises
    ------
    ChunkKeyRegistryError
        If the name is already registered to a different class and
        ``overwrite`` is false, or if the class claims
        `ChunkKeyEncodingSupport.CORE` for a name the Zarr v3 core spec does
        not define.
    """
    # Checked so that CORE means "the spec defines this" rather than "the
    # author said so" -- otherwise the tier a consumer gates on would be
    # self-asserted by the very code it is trying to gate.
    if (
        cls.support is ChunkKeyEncodingSupport.CORE
        and cls.name not in CORE_CHUNK_KEY_ENCODING_NAMES
    ):
        raise ChunkKeyRegistryError(
            f"Chunk key encoding {cls.name!r} declares support level "
            f"{ChunkKeyEncodingSupport.CORE!r}, but the Zarr v3 core spec "
            f"defines only {sorted(CORE_CHUNK_KEY_ENCODING_NAMES)}. Use "
            f"{ChunkKeyEncodingSupport.EXTENSION!r} if it is registered in "
            f"zarr-extensions, or {ChunkKeyEncodingSupport.CUSTOM!r} otherwise."
        )
    existing = _registry.get(cls.name)
    if existing is not None and existing is not cls and not overwrite:
        raise ChunkKeyRegistryError(
            f"A chunk key encoding named {cls.name!r} is already registered "
            f"({existing!r}). Pass overwrite=True to replace it."
        )
    _registry[cls.name] = cls


def unregister_chunk_key_encoding(name: str) -> None:
    """Remove a chunk key encoding from the registry.

    Parameters
    ----------
    name : str
        The registered name to remove.

    Raises
    ------
    UnknownChunkKeyEncodingError
        If no encoding is registered under ``name``.
    """
    if name not in _registry:
        raise UnknownChunkKeyEncodingError(name, registered_chunk_key_encodings())
    del _registry[name]


def get_chunk_key_encoding_class(name: str) -> type[ChunkKeyEncoding]:
    """Look up a chunk key encoding class by its registered name.

    Entry points are loaded on the first lookup that would otherwise fail.

    Parameters
    ----------
    name : str
        The registered name of the encoding.

    Returns
    -------
    type[ChunkKeyEncoding]
        The registered class.

    Raises
    ------
    UnknownChunkKeyEncodingError
        If no encoding is registered under ``name``.
    """
    if name not in _registry:
        _load_entry_points()
    try:
        return _registry[name]
    except KeyError:
        raise UnknownChunkKeyEncodingError(name, registered_chunk_key_encodings()) from None


def get_chunk_key_encoding_support(name: str) -> ChunkKeyEncodingSupport:
    """Look up a registered encoding's support level.

    Entry points are loaded on the first lookup that would otherwise fail.

    Parameters
    ----------
    name : str
        The registered name of the encoding.

    Returns
    -------
    ChunkKeyEncodingSupport
        Whether the encoding is core, a registered extension, or custom.

    Raises
    ------
    UnknownChunkKeyEncodingError
        If no encoding is registered under ``name``.
    """
    return get_chunk_key_encoding_class(name).support


def registered_chunk_key_encodings(
    *, support: ChunkKeyEncodingSupport | None = None
) -> tuple[str, ...]:
    """Return the names of registered chunk key encodings, sorted.

    Parameters
    ----------
    support : ChunkKeyEncodingSupport or None
        If given, return only encodings at this support level. This is the
        hook for gating: a consumer that only accepts spec-defined encodings
        can take ``support=ChunkKeyEncodingSupport.CORE`` as its allowlist.
        Note that entry points are *not* force-loaded here, so a filtered
        listing reflects what has been registered so far.

    Returns
    -------
    tuple of str
        The matching names, sorted.
    """
    if support is None:
        return tuple(sorted(_registry))
    return tuple(sorted(name for name, cls in _registry.items() if cls.support is support))


def chunk_key_encoding_from_json(data: ChunkKeyEncodingJSON) -> ChunkKeyEncoding:
    """Construct a chunk key encoding from Zarr v3 JSON metadata.

    Resolves the encoding class by ``name`` via the registry, then delegates
    to that class's ``from_json``.

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
        If the named encoding is not registered.
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
        If the named encoding is not registered.
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
