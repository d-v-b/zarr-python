"""
Exception types for `zarr-chunk-key-encoding`.

Every error raised by this package derives from `ChunkKeyEncodingError`, so
consumers can catch a single type. The leaf classes also derive from
`ValueError` for compatibility with code that catches standard exceptions.
"""

__all__ = [
    "ChunkCoordsOutOfBoundsError",
    "ChunkKeyConfigurationError",
    "ChunkKeyDecodeError",
    "ChunkKeyEncodingError",
    "ChunkKeyOutOfBoundsError",
    "InvalidChunkCoordsError",
    "UnknownChunkKeyEncodingError",
]


class ChunkKeyEncodingError(Exception):
    """Base class for all errors raised by this package."""


class ChunkKeyConfigurationError(ChunkKeyEncodingError, ValueError):
    """Raised when chunk key encoding JSON metadata or parameters are invalid.

    Examples: a separator outside the permitted set, a `name` field that does
    not match the encoding being constructed, or unexpected metadata keys.
    """


class UnknownChunkKeyEncodingError(ChunkKeyEncodingError, ValueError):
    """Raised when a chunk key encoding name is not one this package knows.

    The set of known names is closed: the encodings the Zarr v3 core spec
    defines. Chunk key encoding is an extension point, so a name outside that
    set is not necessarily invalid -- only unknown here.
    """

    def __init__(self, name: str, known: tuple[str, ...]) -> None:
        self.name = name
        self.known = known
        super().__init__(
            f"Unknown chunk key encoding {name!r}. Known chunk key encodings: {sorted(known)}."
        )


class ChunkKeyDecodeError(ChunkKeyEncodingError, ValueError):
    """Raised when a chunk key cannot be decoded into chunk grid indices."""


class ChunkKeyOutOfBoundsError(ChunkKeyDecodeError):
    """Raised when a well-formed chunk key names a chunk outside a bounded encoding's grid.

    Distinguishes "not a chunk key at all" (the parent `ChunkKeyDecodeError`)
    from "a chunk key, but not for this grid": the key is a valid output of
    the underlying encoding, but its coordinates have the wrong rank or fall
    outside the grid extent.
    """


class InvalidChunkCoordsError(ChunkKeyEncodingError, ValueError):
    """Raised when chunk grid indices passed to `encode` are not non-negative integers."""


class ChunkCoordsOutOfBoundsError(InvalidChunkCoordsError):
    """Raised when chunk grid indices fall outside a bounded encoding's grid.

    The coordinates are valid in isolation (non-negative integers), but have
    the wrong rank for the grid or an index at or beyond the grid extent.
    """
