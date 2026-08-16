"""
Chunk key encodings bound to a known chunk grid.

A plain `ChunkKeyEncoding` maps *any* tuple of non-negative integers to a
key. Once an array's chunk grid shape is known, the meaningful domain shrinks
to the finite set of valid grid indices, and the image becomes a finite set
of keys. `BoundedChunkKeyEncoding` models that restriction:

- ``encode`` and ``decode`` reject coordinates and keys outside the grid
  (`ChunkCoordsOutOfBoundsError` and `ChunkKeyOutOfBoundsError`).
- ``decode`` becomes a total inverse of ``encode``. In particular the ``v2``
  encoding's rank-zero ambiguity — ``"0"`` is the key for both ``()`` and
  ``(0,)`` — disappears, because the grid rank is known.
- The valid key set is a first-class finite collection: membership testing
  (``key in bounded``), iteration, and ``len``.

Consumers that validate candidate store keys against an array — an HTTP
server routing requests, for example — get the full check (grammar, rank,
bounds, canonical spelling) as a single membership test.

Bounded encodings are a runtime construct, not a metadata one: the grid
shape lives in the array metadata, not in the chunk key encoding metadata,
so `BoundedChunkKeyEncoding` has no JSON form and no registry entry.
"""

import itertools
import math
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass

from zarr_chunk_key_encoding._parsing import normalize_chunk_coords
from zarr_chunk_key_encoding.abc import ChunkKey, ChunkKeyEncoding
from zarr_chunk_key_encoding.errors import (
    ChunkCoordsOutOfBoundsError,
    ChunkKeyConfigurationError,
    ChunkKeyDecodeError,
    ChunkKeyOutOfBoundsError,
    InvalidChunkCoordsError,
)

__all__ = [
    "BoundedChunkKeyEncoding",
]


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

    Construct with `ChunkKeyEncoding.bind`, or directly. The valid key set
    is finite, so instances are collections of keys: ``key in bounded``
    checks grammar, rank, bounds, and canonical spelling in one test,
    ``iter`` enumerates every valid key, and ``len`` counts the chunks.
    Membership testing and iteration require nothing beyond the underlying
    encoding, except that membership delegates to ``decode`` and therefore
    raises ``NotImplementedError`` for encodings that do not implement it
    (rank-zero grids excepted, where the single valid key is compared
    directly against ``encode(())``).

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
    >>> bounded = DefaultChunkKeyEncoding().bind((2, 3))
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
    """

    encoding: ChunkKeyEncoding
    grid_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid_shape", _parse_grid_shape(self.grid_shape))

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
        """Decode a store key into chunk grid indices. Total inverse of ``encode``.

        Because the grid rank is known, this resolves ambiguities the
        unbounded ``decode`` cannot: on a zero-dimensional grid the only
        valid key is ``encode(())``, so the ``v2`` encoding's ``"0"``
        decodes to ``()`` here rather than ``(0,)``.

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
