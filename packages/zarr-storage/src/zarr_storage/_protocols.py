"""Capability protocols that backends implement.

A backend works on *backend paths*: keys and directories already joined with a store's root and
already checked against the key grammar. A backend never joins, strips or validates paths beyond
what it reserves for itself. `Store` does all of that, once, for every backend.

Directories are passed normalized: `""` for the backend's root, otherwise a key without a trailing
`/`.

Each capability has a sync protocol and an async twin whose methods end in `_async`. A backend
implements whichever it can. `Store` calls the sync protocols. `AsyncStore` calls the async twin
where the backend has it and otherwise runs the sync method on its context's thread pool, so a
backend that only blocks (a filesystem) needs no async code. Methods are typed so that calling one
the backend lacks is a type error. Compare-and-swap on a generation (`WritableIfMatch`,
`DeletableIfMatch`) is a capability of its own, since many stores can't do it; `put_if_absent` is
part of `Writable`.

Backends hold no thread pools or limits: the store schedules and limits every operation with the
resources of its `Context`, under the backend's `resource_key`. Backends must be thread-safe, since
a store runs their sync methods on many threads at once.

The batch capabilities (`ReadableMany`, `ReadableIntoMany`, `WritableMany` and their async twins)
are for batching the store itself can do (one request for many keys, one GPU submission): without
them, `Store.get_many` and its siblings run single operations concurrently under the context's
resources. A batch method receives that context, so it can draw on the same resources; it yields
`(index, result)` pairs as operations finish and follows the failure rule of `zarr_storage._batch`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from zarr_storage._resources import Context
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
    "ClosableAsync",
    "DeletableIfMatch",
    "DeletableIfMatchAsync",
    "Listable",
    "ListableAsync",
    "Readable",
    "ReadableAsync",
    "ReadableInto",
    "ReadableIntoAsync",
    "ReadableIntoMany",
    "ReadableIntoManyAsync",
    "ReadableMany",
    "ReadableManyAsync",
    "Writable",
    "WritableAsync",
    "WritableIfMatch",
    "WritableIfMatchAsync",
    "WritableMany",
    "WritableManyAsync",
]


O_contra = TypeVar("O_contra", contravariant=True)
"""The type of buffer a backend's `get_into` writes into."""


class Backend(Protocol):
    """What every backend provides: a lifecycle, whether it accepts writes, and the key its
    operations are limited under."""

    @property
    def resource_key(self) -> str:
        """The key of the `Limiter` that stores take from their context for this backend.

        Backends that should share a limit by default share a key: every obstore S3 backend uses
        `"obstore.S3Store"`, so a process's S3 requests are bounded together.
        """
        ...

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


@runtime_checkable
class ClosableAsync(Backend, Protocol):
    """A backend whose cleanup awaits something, such as closing a client session.

    `AsyncStore.aclose` awaits `aclose` on such a backend, and calls `close` on any other.
    """

    async def aclose(self) -> None:
        """`close`, as a coroutine. Idempotent."""
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


class ReadableInto(Backend, Protocol[O_contra]):
    """Reads into a buffer the caller allocated, of the type the backend names.

    A backend over host memory takes any `Buffer` (`ReadableInto[Buffer]`). A backend that reads
    into another device's memory, such as GPU direct storage, takes that device's buffer type
    instead, and `Store.get_into` then accepts exactly that type. Where the bytes land is the
    backend's business; `Copied` reports how many and from which generation.
    """

    def get_into(self, path: str, out: O_contra, byte_range: ByteRange | None) -> Copied | Missing:
        """Copy the value (or part) into `out`.

        Raises `ValueError` if `out` is shorter than the bytes to copy, or can't be written.
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


@runtime_checkable
class ReadableMany(Backend, Protocol):
    def get_many(
        self, requests: Sequence[tuple[str, ByteRange | None]], context: Context
    ) -> Iterator[tuple[int, Found | Missing]]:
        """`get` for each `(path, byte_range)`, yielding `(index, result)` as each finishes."""
        ...


@runtime_checkable
class ReadableIntoMany(Backend, Protocol[O_contra]):
    def get_into_many(
        self, requests: Sequence[tuple[str, O_contra, ByteRange | None]], context: Context
    ) -> Iterator[tuple[int, Copied | Missing]]:
        """`get_into` for each `(path, out, byte_range)`, yielding `(index, result)`."""
        ...


@runtime_checkable
class WritableMany(Backend, Protocol):
    def put_many(
        self, requests: Sequence[tuple[str, memoryview]], context: Context
    ) -> Iterator[tuple[int, Written]]:
        """`put` for each `(path, value)`, yielding `(index, result)` as each finishes."""
        ...

    def delete_many(
        self, paths: Sequence[str], context: Context
    ) -> Iterator[tuple[int, Deleted | Missing]]:
        """`delete` for each path, yielding `(index, result)` as each finishes."""
        ...


class Listable(Backend, Protocol):
    def list(self, directory: str) -> Iterator[str]:
        """Every key under `directory`, as backend paths, in no particular order."""
        ...

    def list_dir(self, directory: str) -> Iterator[ListItem]:
        """The direct children of `directory`, named relative to it."""
        ...


@runtime_checkable
class ReadableAsync(Backend, Protocol):
    async def get_async(self, path: str, byte_range: ByteRange | None) -> Found | Missing: ...

    async def get_if_changed_async(
        self, path: str, generation: Generation, byte_range: ByteRange | None
    ) -> Found | Missing | Unchanged: ...

    async def get_ranges_async(
        self, path: str, ranges: Sequence[ByteRange]
    ) -> FoundRanges | Missing: ...

    async def head_async(self, path: str) -> ObjectMetadata | Missing: ...


@runtime_checkable
class ReadableIntoAsync(Backend, Protocol[O_contra]):
    async def get_into_async(
        self, path: str, out: O_contra, byte_range: ByteRange | None
    ) -> Copied | Missing: ...


@runtime_checkable
class WritableAsync(Backend, Protocol):
    async def put_async(self, path: str, value: memoryview) -> Written: ...

    async def put_if_absent_async(self, path: str, value: memoryview) -> Written | Rejected: ...

    async def delete_async(self, path: str) -> Deleted | Missing: ...

    async def delete_dir_async(self, directory: str) -> None: ...


@runtime_checkable
class WritableIfMatchAsync(Backend, Protocol):
    async def put_if_match_async(
        self, path: str, value: memoryview, generation: Generation
    ) -> Written | Rejected | Missing: ...


@runtime_checkable
class DeletableIfMatchAsync(Backend, Protocol):
    async def delete_if_match_async(
        self, path: str, generation: Generation
    ) -> Deleted | Rejected | Missing: ...


@runtime_checkable
class ReadableManyAsync(Backend, Protocol):
    def get_many_async(
        self, requests: Sequence[tuple[str, ByteRange | None]], context: Context
    ) -> AsyncIterator[tuple[int, Found | Missing]]: ...


@runtime_checkable
class ReadableIntoManyAsync(Backend, Protocol[O_contra]):
    def get_into_many_async(
        self, requests: Sequence[tuple[str, O_contra, ByteRange | None]], context: Context
    ) -> AsyncIterator[tuple[int, Copied | Missing]]: ...


@runtime_checkable
class WritableManyAsync(Backend, Protocol):
    def put_many_async(
        self, requests: Sequence[tuple[str, memoryview]], context: Context
    ) -> AsyncIterator[tuple[int, Written]]: ...

    def delete_many_async(
        self, paths: Sequence[str], context: Context
    ) -> AsyncIterator[tuple[int, Deleted | Missing]]: ...


@runtime_checkable
class ListableAsync(Backend, Protocol):
    def list_async(self, directory: str) -> AsyncIterator[str]: ...

    def list_dir_async(self, directory: str) -> AsyncIterator[ListItem]: ...
