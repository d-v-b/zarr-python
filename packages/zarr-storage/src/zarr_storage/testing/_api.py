"""The store surface the suite tests, and an adapter that drives it through the async methods.

Every suite test runs twice: once against `Store`'s sync methods, once against `AsyncAPI`, which
gives the same sync-looking surface but performs every call through the matching `_async` method on
one event loop. The sync and async surfaces therefore meet one contract, checked by one set of tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self

from zarr_storage._protocols import (
    Listable,
    ListableAsync,
    Readable,
    ReadableAsync,
    Writable,
    WritableAsync,
)

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncIterator, Iterator, Sequence
    from types import TracebackType

    from typing_extensions import Buffer

    from zarr_storage._store import Store
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

__all__ = ["AsyncAPI", "FullBackend", "StoreAPI", "unwrap"]


class FullBackend(
    Readable, Writable, Listable, ReadableAsync, WritableAsync, ListableAsync, Protocol
):
    """A backend with every sync and async capability."""


class StoreAPI(Protocol):
    """The operations the suite calls; `Store` and `AsyncAPI` both provide them."""

    @property
    def root(self) -> str: ...
    @property
    def read_only(self) -> bool: ...
    @property
    def closed(self) -> bool: ...

    def get(self, key: str, *, byte_range: ByteRange | None = None) -> ReadResult | None: ...
    def get_ranges(self, key: str, ranges: Sequence[ByteRange]) -> RangesResult | None: ...
    def get_into(
        self, key: str, out: Buffer, *, byte_range: ByteRange | None = None
    ) -> int | None: ...
    def head(self, key: str) -> ObjectMetadata | None: ...
    def exists(self, key: str) -> bool: ...
    def put(
        self,
        key: str,
        value: Buffer,
        *,
        if_match: Generation | None = None,
        if_none_match: bool = False,
    ) -> PutResult: ...
    def delete(self, key: str, *, if_match: Generation | None = None) -> DeleteResult: ...
    def delete_dir(self, directory: str = "") -> None: ...
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
    """The `Store` behind a `StoreAPI`."""
    if isinstance(store, AsyncAPI):
        return store.store
    return store  # pyright: ignore[reportReturnType]


async def _next(items: AsyncIterator[object]) -> object:
    return await anext(items)


class AsyncAPI:
    """`Store`'s sync-looking surface, performed through its `_async` methods on one event loop."""

    def __init__(self, store: Store[FullBackend], runner: asyncio.Runner) -> None:
        self._store = store
        self._runner = runner

    @property
    def store(self) -> Store[FullBackend]:
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

    def get(self, key: str, *, byte_range: ByteRange | None = None) -> ReadResult | None:
        return self._runner.run(self._store.get_async(key, byte_range=byte_range))

    def get_ranges(self, key: str, ranges: Sequence[ByteRange]) -> RangesResult | None:
        return self._runner.run(self._store.get_ranges_async(key, ranges))

    def get_into(self, key: str, out: Buffer, *, byte_range: ByteRange | None = None) -> int | None:
        return self._runner.run(self._store.get_into_async(key, out, byte_range=byte_range))

    def head(self, key: str) -> ObjectMetadata | None:
        return self._runner.run(self._store.head_async(key))

    def exists(self, key: str) -> bool:
        return self._runner.run(self._store.exists_async(key))

    def put(
        self,
        key: str,
        value: Buffer,
        *,
        if_match: Generation | None = None,
        if_none_match: bool = False,
    ) -> PutResult:
        return self._runner.run(
            self._store.put_async(key, value, if_match=if_match, if_none_match=if_none_match)
        )

    def delete(self, key: str, *, if_match: Generation | None = None) -> DeleteResult:
        return self._runner.run(self._store.delete_async(key, if_match=if_match))

    def delete_dir(self, directory: str = "") -> None:
        self._runner.run(self._store.delete_dir_async(directory))

    def list(self, directory: str = "") -> Iterator[str]:
        # `list_async` checks its argument when called; the items come one at a time.
        return self._iterate(self._store.list_async(directory))  # pyright: ignore[reportReturnType]

    def list_dir(self, directory: str = "") -> Iterator[ListItem]:
        return self._iterate(self._store.list_dir_async(directory))  # pyright: ignore[reportReturnType]

    def _iterate(self, items: AsyncIterator[object]) -> Iterator[object]:
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
        self._store.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
