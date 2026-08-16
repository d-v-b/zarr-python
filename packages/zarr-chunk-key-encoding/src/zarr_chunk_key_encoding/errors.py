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
    "ChunkKeyPluginWarning",
    "ChunkKeyRegistryError",
    "InvalidChunkCoordsError",
    "UnknownChunkKeyEncodingError",
]


class ChunkKeyPluginWarning(UserWarning):
    """Warned when an entry point in the plugin group cannot be registered.

    Discovery skips the offending entry point rather than failing, so one
    broken third-party package cannot make unrelated lookups raise. Callers
    who would rather treat a broken plugin as fatal can escalate this
    category::

        warnings.simplefilter("error", ChunkKeyPluginWarning)
    """


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
