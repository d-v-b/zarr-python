"""
The abstract base class for chunk key encodings.

A chunk key encoding maps the grid index of a chunk (a tuple of non-negative
integers identifying a cell in the chunk grid) to the string key under which
that chunk is stored, and — when possible — back again.

The design follows the `zarrs` Rust implementation, which models chunk key
encodings as a small standalone crate with a name-keyed plugin registry:

- Encodings are identified by a registered ``name`` (a class variable here, a
  plugin identifier in `zarrs`).
- Instances are constructed from Zarr v3 JSON metadata (``from_json``, the
  analogue of `zarrs`' fallible ``create``) and can always report their
  metadata back (``to_json``, the analogue of ``configuration``).
- ``encode`` is required; ``decode`` is optional, since an encoding need not
  be injective in general (`zarrs` omits decoding entirely).

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, NewType, Self

from typing_extensions import TypeAliasType
from zarr_metadata import JSONValue

from zarr_chunk_key_encoding.support import ChunkKeyEncodingSupport

if TYPE_CHECKING:
    from zarr_chunk_key_encoding.bounded import BoundedChunkKeyEncoding

__all__ = [
    "ChunkKey",
    "ChunkKeyEncoding",
    "ChunkKeyEncodingJSON",
]

ChunkKey = NewType("ChunkKey", str)
"""A store key produced by a chunk key encoding.

The static analogue of the validated ``StoreKey`` newtype in the `zarrs`
Rust implementation: a plain ``str`` at runtime, but a distinct type to
type checkers, so that key-consuming code can require proof that a string
came out of an encoding rather than accepting any string.

``encode`` is the blessed producer: its output is valid by construction.
``decode`` deliberately accepts plain ``str``, since its job is judging
untrusted input. Code that has validated a candidate string by other means
(for example a successful membership test against a
`zarr_chunk_key_encoding.bounded.BoundedChunkKeyEncoding`) may brand it
directly with ``ChunkKey(candidate)``.
"""

ChunkKeyEncodingJSON = TypeAliasType("ChunkKeyEncodingJSON", str | Mapping[str, JSONValue])
"""Unvalidated JSON input for constructing a chunk key encoding.

Either the spec's short-hand name string or the named-configuration object
form. Deliberately looser than the precise per-encoding output types from
`zarr_metadata` (e.g. ``DefaultChunkKeyEncodingObject``): ``from_json``
accepts arbitrary decoded JSON and validates it; ``to_json`` returns the
precise shape.
"""


@dataclass(frozen=True)
class ChunkKeyEncoding(ABC):
    """Maps chunk grid indices to store keys, and (optionally) back.

    Subclasses must define the ``name`` class variable — the encoding's
    registered v3 metadata name — and implement ``from_json``, ``to_json``,
    and ``encode``. Implementing ``decode`` is optional: encodings that are
    not injective (or for which decoding is simply not needed) may leave the
    default implementation, which raises ``NotImplementedError``.

    Subclasses defined outside this package should also set ``support`` to
    `ChunkKeyEncodingSupport.EXTENSION` if the encoding is registered in the
    `zarr-extensions` repository; the default, ``CUSTOM``, is the safe
    assumption for anything that is not.
    """

    name: ClassVar[str]

    support: ClassVar[ChunkKeyEncodingSupport] = ChunkKeyEncodingSupport.CUSTOM
    """Where this encoding comes from — core spec, registered extension, or
    neither. Declared on the class so that an encoding registered through
    the entry point group carries its own provenance, and so that holders of
    an instance can gate on it without a registry lookup. Registering a class
    that claims ``CORE`` for a name the spec does not define is an error."""

    @classmethod
    @abstractmethod
    def from_json(cls, data: ChunkKeyEncodingJSON) -> Self:
        """Construct this encoding from Zarr v3 JSON metadata.

        Parameters
        ----------
        data : ChunkKeyEncodingJSON
            The short-hand name string or named-configuration object.

        Returns
        -------
        Self
            The constructed encoding.

        Raises
        ------
        ChunkKeyConfigurationError
            If the metadata is not a valid description of this encoding.
        """

    @abstractmethod
    def to_json(self) -> Mapping[str, JSONValue]:
        """Return the Zarr v3 JSON metadata describing this encoding.

        Returns
        -------
        Mapping[str, JSONValue]
            The named-configuration object form
            (`zarr_metadata.ZarrV3NamedConfigJSON`). Concrete encodings
            narrow the return type to their precise `zarr_metadata` object
            type.
        """

    @abstractmethod
    def encode(self, chunk_coords: Sequence[int]) -> ChunkKey:
        """Encode chunk grid indices into a store key.

        Parameters
        ----------
        chunk_coords : Sequence[int]
            The grid index of the chunk. May be empty, for the single chunk
            of a zero-dimensional array.

        Returns
        -------
        ChunkKey
            The store key for the chunk, relative to the array's prefix.

        Raises
        ------
        InvalidChunkCoordsError
            If the coordinates are not non-negative integers.
        """

    def decode(self, chunk_key: str) -> tuple[int, ...]:
        """Decode a store key into chunk grid indices.

        This is the inverse of ``encode`` where an inverse exists. The base
        implementation raises ``NotImplementedError``; encodings for which
        decoding is well-defined should override it.

        Parameters
        ----------
        chunk_key : str
            The store key for the chunk, relative to the array's prefix.

        Returns
        -------
        tuple of int
            The grid index of the chunk.

        Raises
        ------
        ChunkKeyDecodeError
            If the key is not a valid output of ``encode``.
        NotImplementedError
            If this encoding does not support decoding.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement decode.")

    def bind(self, grid_shape: Sequence[int]) -> "BoundedChunkKeyEncoding":
        """Bind this encoding to a chunk grid, restricting its domain.

        The result is a `zarr_chunk_key_encoding.bounded.BoundedChunkKeyEncoding`:
        its ``encode`` and ``decode`` reject coordinates and keys outside the
        grid, its ``decode`` is a total inverse of ``encode`` (resolving the
        ``v2`` encoding's rank-zero ambiguity), and its valid key set is a
        finite collection supporting ``in``, iteration, and ``len``.

        Parameters
        ----------
        grid_shape : Sequence[int]
            The number of chunks along each dimension. For arrays using
            sharding, pass the shard grid shape, since shards are the unit
            of storage.

        Returns
        -------
        BoundedChunkKeyEncoding
            This encoding, restricted to the given grid.

        Raises
        ------
        ChunkKeyConfigurationError
            If any grid shape entry is not a non-negative integer.
        """
        from zarr_chunk_key_encoding.bounded import BoundedChunkKeyEncoding

        return BoundedChunkKeyEncoding(encoding=self, grid_shape=tuple(grid_shape))
