"""Capability protocols that backends implement.

A backend works on *backend paths*: keys and directories already joined with a store's root and
already checked against the key grammar. A backend never joins, strips or validates paths beyond
what it reserves for itself. `Store` does all of that, once, for every backend.

Directories are passed normalized: `""` for the backend's root, otherwise a key without a trailing
`/`.

Each capability has a sync protocol and an async twin whose methods end in `_async`. A backend
implements whichever it can; `Store`'s methods are typed so that calling one the backend lacks is a
type error. Compare-and-swap on a generation (`WritableIfMatch`, `DeletableIfMatch`) is a capability
of its own, since many stores can't do it; `put_if_absent` is part of `Writable`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from zarr_storage._types import (
        ByteRange,
        Copied,
        Deleted,
        Found,
        FoundRanges,
        Generation,
        ListItem,
        Missing,
        ObjectMetadata,
        Rejected,
        Unchanged,
        Written,
    )

__all__ = [
    "Backend",
    "DeletableIfMatch",
    "DeletableIfMatchAsync",
    "Listable",
    "ListableAsync",
    "Readable",
    "ReadableAsync",
    "ReadableInto",
    "ReadableIntoAsync",
    "Writable",
    "WritableAsync",
    "WritableIfMatch",
    "WritableIfMatchAsync",
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
    def get(self, path: str, byte_range: ByteRange | None) -> Found | Missing:
        """The value at `path`, or the requested part of it."""
        ...

    def get_if_changed(
        self, path: str, generation: Generation, byte_range: ByteRange | None
    ) -> Found | Missing | Unchanged:
        """`get`, or `Unchanged` without reading any bytes if the value has `generation`."""
        ...

    def get_ranges(self, path: str, ranges: Sequence[ByteRange]) -> FoundRanges | Missing:
        """Several parts of the value at `path`, all from one generation."""
        ...

    def head(self, path: str) -> ObjectMetadata | Missing:
        """The size and generation of the value at `path`."""
        ...


class ReadableInto(Backend, Protocol):
    def get_into(
        self, path: str, out: memoryview, byte_range: ByteRange | None
    ) -> Copied | Missing:
        """Copy the value (or part) into `out`.

        Raises `ValueError` if `out` is shorter than the bytes to copy.
        """
        ...


class Writable(Backend, Protocol):
    def put(self, path: str, value: memoryview) -> Written:
        """Write `value` at `path`."""
        ...

    def put_if_absent(self, path: str, value: memoryview) -> Written | Rejected:
        """Write `value` at `path` if the key holds no value."""
        ...

    def delete(self, path: str) -> Deleted | Missing:
        """Delete the key at `path`. A directory is not a key."""
        ...

    def delete_dir(self, directory: str) -> None:
        """Delete every key under `directory`."""
        ...


class WritableIfMatch(Backend, Protocol):
    def put_if_match(
        self, path: str, value: memoryview, generation: Generation
    ) -> Written | Rejected | Missing:
        """Write `value` at `path` if the key's value has `generation`."""
        ...


class DeletableIfMatch(Backend, Protocol):
    def delete_if_match(self, path: str, generation: Generation) -> Deleted | Rejected | Missing:
        """Delete the key at `path` if its value has `generation`."""
        ...


class Listable(Backend, Protocol):
    def list(self, directory: str) -> Iterator[str]:
        """Every key under `directory`, as backend paths, in no particular order."""
        ...

    def list_dir(self, directory: str) -> Iterator[ListItem]:
        """The direct children of `directory`, named relative to it."""
        ...


class ReadableAsync(Backend, Protocol):
    async def get_async(self, path: str, byte_range: ByteRange | None) -> Found | Missing: ...

    async def get_if_changed_async(
        self, path: str, generation: Generation, byte_range: ByteRange | None
    ) -> Found | Missing | Unchanged: ...

    async def get_ranges_async(
        self, path: str, ranges: Sequence[ByteRange]
    ) -> FoundRanges | Missing: ...

    async def head_async(self, path: str) -> ObjectMetadata | Missing: ...


class ReadableIntoAsync(Backend, Protocol):
    async def get_into_async(
        self, path: str, out: memoryview, byte_range: ByteRange | None
    ) -> Copied | Missing: ...


class WritableAsync(Backend, Protocol):
    async def put_async(self, path: str, value: memoryview) -> Written: ...

    async def put_if_absent_async(self, path: str, value: memoryview) -> Written | Rejected: ...

    async def delete_async(self, path: str) -> Deleted | Missing: ...

    async def delete_dir_async(self, directory: str) -> None: ...


class WritableIfMatchAsync(Backend, Protocol):
    async def put_if_match_async(
        self, path: str, value: memoryview, generation: Generation
    ) -> Written | Rejected | Missing: ...


class DeletableIfMatchAsync(Backend, Protocol):
    async def delete_if_match_async(
        self, path: str, generation: Generation
    ) -> Deleted | Rejected | Missing: ...


class ListableAsync(Backend, Protocol):
    def list_async(self, directory: str) -> AsyncIterator[str]: ...

    def list_dir_async(self, directory: str) -> AsyncIterator[ListItem]: ...
