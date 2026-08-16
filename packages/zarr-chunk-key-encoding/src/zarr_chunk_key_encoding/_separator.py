"""
The chunk key separator: the character placed between coordinates in a chunk key.

Both encodings defined by the Zarr v3 core spec (`default` and `v2`) accept
the same two separators, `"/"` and `"."`; they differ only in which one they
default to. `zarr-metadata` types the per-encoding separator fields
(`DefaultChunkKeyEncodingSeparator`, `V2ChunkKeyEncodingSeparator`); this
module provides the shared runtime alias and validator.
"""

from typing import Final, Literal

from zarr_chunk_key_encoding._errors import ChunkKeyConfigurationError

__all__ = [
    "SEPARATORS",
    "Separator",
    "parse_separator",
]

Separator = Literal[".", "/"]
"""Literal type of the permitted chunk key separators."""

SEPARATORS: Final = (".", "/")
"""Tuple of the permitted chunk key separators."""


def parse_separator(data: object) -> Separator:
    """Validate a chunk key separator.

    Parameters
    ----------
    data : object
        The value to validate.

    Returns
    -------
    Separator
        The input, narrowed to the `Separator` type.

    Raises
    ------
    ChunkKeyConfigurationError
        If the input is not `"."` or `"/"`.
    """
    if data not in SEPARATORS:
        raise ChunkKeyConfigurationError(
            f"Invalid chunk key separator: {data!r}. Expected one of {SEPARATORS}."
        )
    # pyright narrows `data` to Separator via the membership test above.
    return data
