"""Values that stores take and return. All are immutable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

__all__ = [
    "ByteRange",
    "DeleteResult",
    "Entry",
    "Generation",
    "Key",
    "ListItem",
    "ObjectMetadata",
    "OffsetRequest",
    "Prefix",
    "PutResult",
    "RangeRequest",
    "RangesResult",
    "ReadResult",
    "SuffixRequest",
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
class ReadResult:
    """The bytes a read returned, and the generation they belong to."""

    value: memoryview
    generation: Generation


@dataclass(frozen=True, slots=True)
class RangesResult:
    """The bytes of several ranges of one value, all from one read of one generation."""

    values: tuple[memoryview, ...]
    generation: Generation


@dataclass(frozen=True, slots=True)
class PutResult:
    """The outcome of a write.

    `applied` is `False` when a precondition (`if_match`, `if_none_match`) failed; `generation` is
    then the key's current generation, or `None` if the key is absent. When `applied` is `True`,
    `generation` is the generation of the value just written.
    """

    generation: Generation | None
    applied: bool = True


@dataclass(frozen=True, slots=True)
class DeleteResult:
    """The outcome of a delete.

    | case | `generation` | `applied` |
    |---|---|---|
    | the key was absent | `None` | `False` |
    | `if_match` did not match | the current generation | `False` |
    | the key was deleted | the generation of the deleted value | `True` |
    """

    generation: Generation | None
    applied: bool


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
