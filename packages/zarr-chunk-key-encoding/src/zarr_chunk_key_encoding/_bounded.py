"""
Chunk key encodings bound to a known chunk grid.

A plain `ChunkKeyEncoding` maps *any* tuple of non-negative integers to a
key. Once an array's chunk grid shape is known, the meaningful domain shrinks
to the finite set of valid grid indices, and the image becomes a finite set
of keys. `BoundedChunkKeyEncoding` models that restriction:

- `encode` and `decode` reject coordinates and keys outside the grid
  (`ChunkCoordsOutOfBoundsError` and `ChunkKeyOutOfBoundsError`).
- `decode` becomes a total inverse of `encode`. In particular the `v2`
  encoding's rank-zero ambiguity — `"0"` is the key for both `()` and
  `(0,)` — disappears, because the grid rank is known.
- The valid key set is a first-class finite collection: membership testing
  (`key in bounded`), iteration, and `len`.

Consumers that validate candidate store keys against an array — an HTTP
server routing requests, for example — get the full check (grammar, rank,
bounds, canonical spelling) as a single membership test.

A bounded encoding is not a Zarr v3 metadata extension: it has no `name`
and is not something an array's `chunk_key_encoding` field can hold, since
the grid shape lives elsewhere in the array metadata. It does have a JSON
form of its own, though — `{"grid_shape": [...], "chunk_key_encoding": ...}`
— so that a consumer can persist or transmit the bound object as a unit
(`to_json` / `from_json`, typed as `BoundedChunkKeyEncodingJSON`).
"""

import itertools
import math
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, cast

from typing_extensions import TypedDict

from zarr_chunk_key_encoding._abc import ChunkKey, ChunkKeyEncoding, ChunkKeyEncodingJSON
from zarr_chunk_key_encoding._errors import (
    ChunkCoordsOutOfBoundsError,
    ChunkKeyConfigurationError,
    ChunkKeyDecodeError,
    ChunkKeyOutOfBoundsError,
    InvalidChunkCoordsError,
)
from zarr_chunk_key_encoding._parsing import normalize_chunk_coords

if TYPE_CHECKING:
    from zarr_metadata import JSONValue

__all__ = [
    "BoundedChunkKeyEncoding",
    "BoundedChunkKeyEncodingJSON",
]


class BoundedChunkKeyEncodingJSON(TypedDict, closed=True):
    """The JSON form of a `BoundedChunkKeyEncoding`.

    Closed (PEP 728): exactly these two keys. `chunk_key_encoding` holds the
    underlying encoding's own metadata, in either the short-hand name string
    or named-configuration object form.
    """

    grid_shape: list[int]
    chunk_key_encoding: ChunkKeyEncodingJSON


def _parse_grid_shape(grid_shape: Sequence[int]) -> tuple[int, ...]:
    """Normalize a chunk grid shape to a tuple of built-in non-negative ints.

    Parameters
    ----------
    grid_shape : Sequence[int]
        The number of chunks along each dimension.

    Returns
    -------
    tuple of int
        The normalized grid shape.

    Raises
    ------
    ChunkKeyConfigurationError
        If any entry is not a non-negative integer.
    """
    try:
        return normalize_chunk_coords(grid_shape)
    except InvalidChunkCoordsError as e:
        raise ChunkKeyConfigurationError(
            f"Invalid chunk grid shape {grid_shape!r}: entries must be non-negative integers."
        ) from e


@dataclass(frozen=True)
class BoundedChunkKeyEncoding(Collection[ChunkKey]):
    """A chunk key encoding restricted to a known chunk grid.

    Construct with `ChunkKeyEncoding.bind_grid_shape`, or directly. The valid key set
    is finite, so instances are collections of keys: `key in bounded`
    checks grammar, rank, bounds, and canonical spelling in one test,
    `iter` enumerates every valid key, and `len` counts the chunks.
    Membership testing and iteration require nothing beyond the underlying
    encoding, except that membership delegates to `decode` and therefore
    raises `NotImplementedError` for encodings that do not implement it
    (rank-zero grids excepted, where the single valid key is compared
    directly against `encode(())`).

    Attributes
    ----------
    encoding : ChunkKeyEncoding
        The underlying (unbounded) encoding.
    grid_shape : tuple of int
        The number of chunks along each dimension. May be empty, for the
        single chunk of a zero-dimensional array, and may contain zeros,
        for an array with no chunks along some dimension.

    Examples
    --------
    >>> from zarr_chunk_key_encoding import DefaultChunkKeyEncoding
    >>> bounded = DefaultChunkKeyEncoding().bind_grid_shape((2, 3))
    >>> bounded.encode((1, 2))
    'c/1/2'
    >>> bounded.decode("c/1/2")
    (1, 2)
    >>> "c/1/2" in bounded
    True
    >>> "c/9/0" in bounded
    False
    >>> len(bounded)
    6
    >>> bounded.to_json()
    {'grid_shape': [2, 3], 'chunk_key_encoding': {'name': 'default', 'configuration': {'separator': '/'}}}
    """

    encoding: ChunkKeyEncoding
    grid_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid_shape", _parse_grid_shape(self.grid_shape))

    @classmethod
    def from_json(cls, data: object) -> Self:
        """Construct from the JSON form produced by `to_json`.

        Parameters
        ----------
        data : object
            Unvalidated JSON: an object with exactly the keys `grid_shape`
            (a list of non-negative integers) and `chunk_key_encoding` (the
            underlying encoding's own metadata, in either form).

        Returns
        -------
        Self
            The bound encoding.

        Raises
        ------
        ChunkKeyConfigurationError
            If the envelope is not that shape, or the grid shape or nested
            encoding metadata is invalid.
        UnknownChunkKeyEncodingError
            If the nested encoding names one this package does not know.
        """
        # Deferred: `_from_json` imports `_default` and `_v2`, which import
        # `_abc`, whose `bind_grid_shape` imports this module. Nothing here is needed at
        # import time, so resolve it at call time and keep the graph acyclic.
        from zarr_chunk_key_encoding._from_json import chunk_key_encoding_from_json

        if not isinstance(data, Mapping):
            raise ChunkKeyConfigurationError(
                f"Invalid bounded chunk key encoding metadata: expected a JSON "
                f"object with keys 'grid_shape' and 'chunk_key_encoding', got {data!r}."
            )
        mapping = cast("Mapping[str, JSONValue]", data)
        expected = {"grid_shape", "chunk_key_encoding"}
        if set(mapping) != expected:
            raise ChunkKeyConfigurationError(
                f"Invalid bounded chunk key encoding metadata: expected exactly "
                f"the keys {sorted(expected)}, got {sorted(mapping, key=repr)}."
            )
        grid_shape = mapping["grid_shape"]
        if not isinstance(grid_shape, Sequence) or isinstance(grid_shape, str):
            raise ChunkKeyConfigurationError(
                f"Invalid bounded chunk key encoding metadata: 'grid_shape' must "
                f"be a list of integers, got {grid_shape!r}."
            )
        # `_parse_grid_shape` (via `__init__`) validates the entries; the
        # sequence check above only rules out shapes it could not iterate.
        encoding = chunk_key_encoding_from_json(
            cast("ChunkKeyEncodingJSON", mapping["chunk_key_encoding"])
        )
        return cls(encoding=encoding, grid_shape=tuple(cast("Sequence[int]", grid_shape)))

    def to_json(self) -> BoundedChunkKeyEncodingJSON:
        """Return the JSON form: the grid shape and the encoding's own metadata.

        Returns
        -------
        BoundedChunkKeyEncodingJSON
            `{"grid_shape": [...], "chunk_key_encoding": {...}}`. The grid
            shape is a list, since JSON has no tuple; `from_json` accepts
            either.
        """
        return BoundedChunkKeyEncodingJSON(
            grid_shape=list(self.grid_shape),
            chunk_key_encoding=self.encoding.to_json(),
        )

    def _in_grid(self, coords: tuple[int, ...]) -> bool:
        """Whether non-negative *coords* name a cell of the grid."""
        return len(coords) == len(self.grid_shape) and all(
            c < g for c, g in zip(coords, self.grid_shape, strict=True)
        )

    def encode(self, chunk_coords: Sequence[int]) -> ChunkKey:
        """Encode chunk grid indices into a store key.

        Parameters
        ----------
        chunk_coords : Sequence[int]
            The grid index of the chunk.

        Returns
        -------
        ChunkKey
            The store key for the chunk, relative to the array's prefix.

        Raises
        ------
        InvalidChunkCoordsError
            If the coordinates are not non-negative integers.
        ChunkCoordsOutOfBoundsError
            If the coordinates do not name a cell of the grid: wrong rank,
            or any index at or beyond the grid extent.
        """
        indices = normalize_chunk_coords(chunk_coords)
        if not self._in_grid(indices):
            raise ChunkCoordsOutOfBoundsError(
                f"Chunk coordinates {indices!r} do not name a cell of the "
                f"chunk grid with shape {self.grid_shape!r}."
            )
        return self.encoding.encode(indices)

    def decode(self, chunk_key: str) -> tuple[int, ...]:
        """Decode a store key into chunk grid indices. Total inverse of `encode`.

        Because the grid rank is known, this resolves ambiguities the
        unbounded `decode` cannot: on a zero-dimensional grid the only
        valid key is `encode(())`, so the `v2` encoding's `"0"`
        decodes to `()` here rather than `(0,)`.

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
            If the key is not a valid output of the underlying encoding.
        ChunkKeyOutOfBoundsError
            If the key is well-formed but names a chunk outside the grid.
        NotImplementedError
            If the underlying encoding does not support decoding.
        """
        if self.grid_shape == () and chunk_key == self.encoding.encode(()):
            return ()
        coords = self.encoding.decode(chunk_key)
        if not self._in_grid(coords):
            raise ChunkKeyOutOfBoundsError(
                f"Chunk key {chunk_key!r} names a chunk outside the chunk "
                f"grid with shape {self.grid_shape!r}."
            )
        return coords

    def __contains__(self, key: object) -> bool:
        """Whether *key* is the canonical store key of a chunk in the grid."""
        if not isinstance(key, str):
            return False
        try:
            self.decode(key)
        except ChunkKeyDecodeError:
            return False
        return True

    def __iter__(self) -> Iterator[ChunkKey]:
        """Iterate over all valid chunk keys, in row-major order of their grid indices."""
        for coords in itertools.product(*(range(g) for g in self.grid_shape)):
            yield self.encoding.encode(coords)

    def __len__(self) -> int:
        """The number of chunks in the grid."""
        return math.prod(self.grid_shape)
