"""Values that stores take and return. All are immutable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

__all__ = [
    "ByteRange",
    "Copied",
    "Deleted",
    "Entry",
    "Found",
    "FoundRanges",
    "Generation",
    "Key",
    "ListItem",
    "Missing",
    "ObjectMetadata",
    "OffsetRequest",
    "Prefix",
    "RangeRequest",
    "Rejected",
    "SuffixRequest",
    "Unchanged",
    "Written",
]

Generation = NewType("Generation", object)
"""An opaque token for the state of one key, compared only by equality.

Backends mint it: a counter (memory), `(device, inode, mtime_ns, size)` (local files), an ETag
(object stores). Two reads that return equal generations read the same value.
"""


def _check_offsets(**offsets: int) -> None:
    for name, value in offsets.items():
        if type(value) is not int:
            raise TypeError(f"{name} must be an int; got {type(value).__name__}.")


@dataclass(frozen=True, slots=True)
class RangeRequest:
    """The bytes from `start` (inclusive) to `end` (exclusive)."""

    start: int
    end: int

    def __post_init__(self) -> None:
        _check_offsets(start=self.start, end=self.end)
        if self.start < 0 or self.end < self.start:
            raise ValueError(
                f"A range needs 0 <= start <= end; got start={self.start}, end={self.end}."
            )


@dataclass(frozen=True, slots=True)
class OffsetRequest:
    """The bytes from `start` (inclusive) to the end of the value."""

    start: int

    def __post_init__(self) -> None:
        _check_offsets(start=self.start)
        if self.start < 0:
            raise ValueError(f"An offset must be >= 0; got {self.start}.")


@dataclass(frozen=True, slots=True)
class SuffixRequest:
    """The last `length` bytes of the value, or the whole value if it is shorter."""

    length: int

    def __post_init__(self) -> None:
        _check_offsets(length=self.length)
        if self.length < 0:
            raise ValueError(f"A suffix length must be >= 0; got {self.length}.")


ByteRange = RangeRequest | OffsetRequest | SuffixRequest
"""A part of a value. An end past the value is clamped; a start past the value is an error."""


@dataclass(frozen=True, slots=True)
class Found:
    """A read found the key: its bytes (or the requested part) and their generation."""

    value: memoryview
    generation: Generation
    read_at: float | None = None
    """When the store read the value, in seconds since the UNIX epoch, if the backend knows."""


@dataclass(frozen=True, slots=True)
class FoundRanges:
    """A read found the key: the bytes of several ranges, all from one generation."""

    values: tuple[memoryview, ...]
    generation: Generation
    read_at: float | None = None


@dataclass(frozen=True, slots=True)
class Copied:
    """A read found the key and copied `size` bytes into the caller's buffer."""

    size: int
    generation: Generation
    read_at: float | None = None


@dataclass(frozen=True, slots=True)
class Missing:
    """The key holds no value (an HTTP store's `404`).

    `read_at` records when the store looked, if the backend knows, so an absence can be cached like
    a value.
    """

    read_at: float | None = None


@dataclass(frozen=True, slots=True)
class Unchanged:
    """`get_if_changed` named the key's current generation, so no bytes were read (an HTTP `304`)."""

    generation: Generation
    read_at: float | None = None


@dataclass(frozen=True, slots=True)
class Written:
    """A write stored its value, which has `generation`."""

    generation: Generation


@dataclass(frozen=True, slots=True)
class Deleted:
    """A delete removed the key's value, which had `generation`."""

    generation: Generation


@dataclass(frozen=True, slots=True)
class Rejected:
    """A conditional write or delete changed nothing: the key holds a value with `generation`.

    `put_if_absent` is rejected by any value; `put_if_match` and `delete_if_match` by a value whose
    generation is not the one they named.
    """

    generation: Generation


@dataclass(frozen=True, slots=True)
class ObjectMetadata:
    """What a store knows about one key without reading its value."""

    size: int
    generation: Generation
    last_modified: float | None
    """Seconds since the UNIX epoch, or `None` if the backend does not track it."""


@dataclass(frozen=True, slots=True)
class Key:
    """A key directly inside the listed directory. `name` has no `/`."""

    name: str


@dataclass(frozen=True, slots=True)
class Prefix:
    """A directory directly inside the listed directory. `name` has no `/`."""

    name: str


@dataclass(frozen=True, slots=True)
class Entry:
    """A child of the listed directory that the backend can't classify as a key or a directory."""

    name: str


ListItem = Key | Prefix | Entry
"""One child of a listed directory. A name can appear both as a `Key` and as a `Prefix`."""
