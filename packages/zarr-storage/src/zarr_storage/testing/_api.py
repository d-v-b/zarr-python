"""The store surface the suite tests, and an adapter that drives it through `AsyncStore`.

Every suite test runs twice: once against `Store`, once against `AsyncAPI`, which gives the same
sync-looking surface but performs every call through the `AsyncStore` method of the same name on
one event loop. The sync and async surfaces therefore meet one contract, checked by one set of tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, TypeVar, cast

from typing_extensions import Buffer

from zarr_storage._protocols import (
    Listable,
    Readable,
    ReadableInto,
    Writable,
    WritableIfMatch,
)
from zarr_storage._store import AsyncStore

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
    from types import TracebackType

    from zarr_storage._protocols import DeletableIfMatch, DeletableIfMatchAsync
    from zarr_storage._store import Store
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

__all__ = ["AsyncAPI", "FullBackend", "StoreAPI", "async_api", "delete_if_match", "unwrap"]

T = TypeVar("T")
Q = TypeVar("Q", bound="str | tuple[str, ByteRange | None]")


class FullBackend(
    Readable,
    ReadableInto[Buffer],
    Writable,
    WritableIfMatch,
    Listable,
    Protocol,
):
    """A backend with every capability the suite needs of all backends.

    Only the sync protocols: `AsyncStore` uses a backend's async twins where it has them and runs
    its sync methods on its context's thread pool otherwise, so the suite's async run covers both.
    `DeletableIfMatch` is left out, since object stores lack it; `delete_if_match` reaches it.
    """


class StoreAPI(Protocol):
    """The operations the suite calls; `Store` and `AsyncAPI` both provide them."""

    @property
    def root(self) -> str: ...
    @property
    def read_only(self) -> bool: ...
    @property
    def closed(self) -> bool: ...

    def get(self, key: str, *, byte_range: ByteRange | None = None) -> Found | Missing: ...
    def get_if_changed(
        self, key: str, generation: Generation, *, byte_range: ByteRange | None = None
    ) -> Found | Missing | Unchanged: ...
    def get_ranges(self, key: str, ranges: Sequence[ByteRange]) -> FoundRanges | Missing: ...
    def get_into(
        self, key: str, out: Buffer, *, byte_range: ByteRange | None = None
    ) -> Copied | Missing: ...
    def head(self, key: str) -> ObjectMetadata | Missing: ...
    def exists(self, key: str) -> bool: ...
    def put(self, key: str, value: Buffer) -> Written: ...
    def put_if_absent(self, key: str, value: Buffer) -> Written | Rejected: ...
    def put_if_match(
        self, key: str, value: Buffer, generation: Generation
    ) -> Written | Rejected | Missing: ...
    def delete(self, key: str) -> Deleted | Missing: ...
    def delete_dir(self, directory: str = "") -> None: ...
    def get_many(self, requests: Iterable[Q]) -> Iterator[tuple[Q, Found | Missing]]: ...
    def get_into_many(
        self, requests: Iterable[tuple[str, Buffer] | tuple[str, Buffer, ByteRange | None]]
    ) -> Iterator[
        tuple[tuple[str, Buffer] | tuple[str, Buffer, ByteRange | None], Copied | Missing]
    ]: ...
    def put_many(self, items: Iterable[tuple[str, Buffer]]) -> list[tuple[str, Written]]: ...
    def delete_many(self, keys: Iterable[str]) -> list[tuple[str, Deleted | Missing]]: ...
    def list(self, directory: str = "") -> Iterator[str]: ...
    def list_dir(self, directory: str = "") -> Iterator[ListItem]: ...
    def __truediv__(self, directory: str) -> Self: ...
    def with_read_only(self, read_only: bool = True) -> Self: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


def unwrap(store: StoreAPI) -> Store[FullBackend]:
    """The `Store` behind a `StoreAPI`: a sync view of an `AsyncAPI`'s store."""
    if isinstance(store, AsyncAPI):
        return store.store.as_sync()
    return store  # pyright: ignore[reportReturnType]


def delete_if_match(
    store: StoreAPI, key: str, generation: Generation
) -> Deleted | Rejected | Missing:
    """`delete_if_match` through `store`'s surface, for a backend that has it."""
    if isinstance(store, AsyncAPI):
        return store.delete_if_match(key, generation)
    return cast("Store[DeletableIfMatch]", store).delete_if_match(key, generation)


async def _next(items: AsyncIterator[T]) -> T:
    return await anext(items)


def async_api(store: Store[FullBackend], runner: asyncio.Runner) -> AsyncAPI:
    """An `AsyncAPI` over an `AsyncStore` that owns `store`'s backend, at its root."""
    return AsyncAPI(AsyncStore(store.backend, root=store.root, read_only=store.read_only), runner)


class AsyncAPI:
    """`Store`'s surface, performed through an `AsyncStore` on one event loop.

    Each method runs the `AsyncStore` method of the same name to completion; listings and lazy
    batches are taken one item per run.
    """

    def __init__(self, store: AsyncStore[FullBackend], runner: asyncio.Runner) -> None:
        self._store = store
        self._runner = runner

    @property
    def store(self) -> AsyncStore[FullBackend]:
        return self._store

    @property
    def root(self) -> str:
        return self._store.root

    @property
    def read_only(self) -> bool:
        return self._store.read_only

    @property
    def closed(self) -> bool:
        return self._store.closed

    def get(self, key: str, *, byte_range: ByteRange | None = None) -> Found | Missing:
        return self._runner.run(self._store.get(key, byte_range=byte_range))

    def get_if_changed(
        self, key: str, generation: Generation, *, byte_range: ByteRange | None = None
    ) -> Found | Missing | Unchanged:
        return self._runner.run(self._store.get_if_changed(key, generation, byte_range=byte_range))

    def get_ranges(self, key: str, ranges: Sequence[ByteRange]) -> FoundRanges | Missing:
        return self._runner.run(self._store.get_ranges(key, ranges))

    def get_into(
        self, key: str, out: Buffer, *, byte_range: ByteRange | None = None
    ) -> Copied | Missing:
        return self._runner.run(self._store.get_into(key, out, byte_range=byte_range))

    def head(self, key: str) -> ObjectMetadata | Missing:
        return self._runner.run(self._store.head(key))

    def exists(self, key: str) -> bool:
        return self._runner.run(self._store.exists(key))

    def put(self, key: str, value: Buffer) -> Written:
        return self._runner.run(self._store.put(key, value))

    def put_if_absent(self, key: str, value: Buffer) -> Written | Rejected:
        return self._runner.run(self._store.put_if_absent(key, value))

    def put_if_match(
        self, key: str, value: Buffer, generation: Generation
    ) -> Written | Rejected | Missing:
        return self._runner.run(self._store.put_if_match(key, value, generation))

    def delete(self, key: str) -> Deleted | Missing:
        return self._runner.run(self._store.delete(key))

    def delete_if_match(self, key: str, generation: Generation) -> Deleted | Rejected | Missing:
        store = cast("AsyncStore[DeletableIfMatchAsync]", self._store)
        return self._runner.run(store.delete_if_match(key, generation))

    def delete_dir(self, directory: str = "") -> None:
        self._runner.run(self._store.delete_dir(directory))

    def get_many(self, requests: Iterable[Q]) -> Iterator[tuple[Q, Found | Missing]]:
        # `get_many` checks its requests when called; the results come one at a time.
        return self._iterate(self._store.get_many(requests))

    def get_into_many(
        self, requests: Iterable[tuple[str, Buffer] | tuple[str, Buffer, ByteRange | None]]
    ) -> Iterator[
        tuple[tuple[str, Buffer] | tuple[str, Buffer, ByteRange | None], Copied | Missing]
    ]:
        return self._iterate(self._store.get_into_many(requests))

    def put_many(self, items: Iterable[tuple[str, Buffer]]) -> list[tuple[str, Written]]:
        return self._runner.run(self._store.put_many(items))

    def delete_many(self, keys: Iterable[str]) -> list[tuple[str, Deleted | Missing]]:
        return self._runner.run(self._store.delete_many(keys))

    def list(self, directory: str = "") -> Iterator[str]:
        # `list` checks its argument when called; the items come one at a time.
        return self._iterate(self._store.list(directory))

    def list_dir(self, directory: str = "") -> Iterator[ListItem]:
        return self._iterate(self._store.list_dir(directory))

    def _iterate(self, items: AsyncIterator[T]) -> Iterator[T]:
        while True:
            try:
                yield self._runner.run(_next(items))
            except StopAsyncIteration:
                return

    def __truediv__(self, directory: str) -> Self:
        return type(self)(self._store / directory, self._runner)

    def with_read_only(self, read_only: bool = True) -> Self:
        return type(self)(self._store.with_read_only(read_only), self._runner)

    def close(self) -> None:
        self._runner.run(self._store.aclose())

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
