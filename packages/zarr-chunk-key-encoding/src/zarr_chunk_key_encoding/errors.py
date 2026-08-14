"""
Exception types for `zarr-chunk-key-encoding`.

Every error raised by this package derives from `ChunkKeyEncodingError`, so
consumers can catch a single type. The leaf classes also derive from
`ValueError` for compatibility with code that catches standard exceptions.
"""

__all__ = [
    "ChunkKeyConfigurationError",
    "ChunkKeyDecodeError",
    "ChunkKeyEncodingError",
    "ChunkKeyRegistryError",
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
    """Raised when a chunk key encoding name is not present in the registry."""

    def __init__(self, name: str, registered: tuple[str, ...]) -> None:
        self.name = name
        self.registered = registered
        super().__init__(
            f"Unknown chunk key encoding {name!r}. "
            f"Registered chunk key encodings: {sorted(registered)}."
        )


class ChunkKeyRegistryError(ChunkKeyEncodingError, ValueError):
    """Raised when a registration conflicts with an existing registry entry."""


class ChunkKeyDecodeError(ChunkKeyEncodingError, ValueError):
    """Raised when a chunk key cannot be decoded into chunk grid indices."""


class InvalidChunkCoordsError(ChunkKeyEncodingError, ValueError):
    """Raised when chunk grid indices passed to `encode` are not non-negative integers."""
