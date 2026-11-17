"""Capability protocols that backends implement.

A backend works on *backend paths*: keys and directories already joined with a store's root and
already checked against the key grammar. A backend never joins, strips or validates paths beyond
what it reserves for itself. `Store` does all of that, once, for every backend.

Directories are passed normalized: `""` for the backend's root, otherwise a key without a trailing
`/`.

Each capability has a sync protocol and an async twin whose methods end in `_async`. A backend
implements whichever it can; `Store`'s methods are typed so that calling one the backend lacks is a
type error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from zarr_storage._types import (
        ByteRange,
        DeleteResult,
        Generation,
        ListItem,
        ObjectMetadata,
        PutResult,
        RangesResult,
        ReadResult,
    )

__all__ = [
    "Backend",
    "Listable",
    "ListableAsync",
    "Readable",
    "ReadableAsync",
    "ReadableInto",
    "ReadableIntoAsync",
    "Writable",
    "WritableAsync",
]


class Backend(Protocol):
    """What every backend provides: a lifecycle and whether it accepts writes at all."""

    @property
    def closed(self) -> bool:
        """Whether `close` has been called."""
        ...

    @property
    def writable(self) -> bool:
        """Whether the backend can write. A store over a backend that can't is read-only."""
        ...

    def close(self) -> None:
        """Release the backend's resources. Idempotent."""
        ...


class Readable(Backend, Protocol):
    def get(self, path: str, byte_range: ByteRange | None) -> ReadResult | None:
        """The value at `path`, or the requested part of it; `None` if the key is absent."""
        ...

    def get_ranges(self, path: str, ranges: Sequence[ByteRange]) -> RangesResult | None:
        """Several parts of the value at `path`, from one read; `None` if the key is absent."""
        ...

    def head(self, path: str) -> ObjectMetadata | None:
        """The size and generation of the value at `path`; `None` if the key is absent."""
        ...


class ReadableInto(Backend, Protocol):
    def get_into(self, path: str, out: memoryview, byte_range: ByteRange | None) -> int | None:
        """Copy the value (or part) into `out`; the number of bytes, or `None` if absent.

        Raises `ValueError` if `out` is shorter than the bytes to copy.
        """
        ...


class Writable(Backend, Protocol):
    def put(
        self, path: str, value: memoryview, *, if_match: Generation | None, if_none_match: bool
    ) -> PutResult:
        """Write `value` at `path`, if the preconditions hold."""
        ...

    def delete(self, path: str, *, if_match: Generation | None) -> DeleteResult:
        """Delete the key at `path`, if the precondition holds. A directory is not a key."""
        ...

    def delete_dir(self, directory: str) -> None:
        """Delete every key under `directory`."""
        ...


class Listable(Backend, Protocol):
    def list(self, directory: str) -> Iterator[str]:
        """Every key under `directory`, as backend paths, in no particular order."""
        ...

    def list_dir(self, directory: str) -> Iterator[ListItem]:
        """The direct children of `directory`, named relative to it."""
        ...


class ReadableAsync(Backend, Protocol):
    async def get_async(self, path: str, byte_range: ByteRange | None) -> ReadResult | None: ...

    async def get_ranges_async(
        self, path: str, ranges: Sequence[ByteRange]
    ) -> RangesResult | None: ...

    async def head_async(self, path: str) -> ObjectMetadata | None: ...


class ReadableIntoAsync(Backend, Protocol):
    async def get_into_async(
        self, path: str, out: memoryview, byte_range: ByteRange | None
    ) -> int | None: ...


class WritableAsync(Backend, Protocol):
    async def put_async(
        self, path: str, value: memoryview, *, if_match: Generation | None, if_none_match: bool
    ) -> PutResult: ...

    async def delete_async(self, path: str, *, if_match: Generation | None) -> DeleteResult: ...

    async def delete_dir_async(self, directory: str) -> None: ...


class ListableAsync(Backend, Protocol):
    def list_async(self, directory: str) -> AsyncIterator[str]: ...

    def list_dir_async(self, directory: str) -> AsyncIterator[ListItem]: ...
