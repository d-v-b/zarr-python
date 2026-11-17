"""`Store`: a view of a backend at a root.

`Store` is where the key grammar, the root, read-only access and the closed check are enforced, once,
for every backend. A backend only sees backend paths.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Final, Generic, Self, TypeVar, cast, final

from zarr_storage._batch import run_many, run_many_async
from zarr_storage._buffers import byte_view
from zarr_storage._errors import BatchError, ReadOnlyError, StoreClosedError
from zarr_storage._keys import check_key, is_within, join, normalize_directory, relativize
from zarr_storage._protocols import (
    Backend,
    ReadableIntoMany,
    ReadableIntoManyAsync,
    ReadableMany,
    ReadableManyAsync,
    WritableMany,
    WritableManyAsync,
)
from zarr_storage._types import Missing

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
    from types import TracebackType

    from typing_extensions import Buffer

    from zarr_storage._protocols import (
        DeletableIfMatch,
        DeletableIfMatchAsync,
        Listable,
        ListableAsync,
        Readable,
        ReadableAsync,
        ReadableInto,
        ReadableIntoAsync,
        Writable,
        WritableAsync,
        WritableIfMatch,
        WritableIfMatchAsync,
    )
    from zarr_storage._types import (
        ByteRange,
        Copied,
        Deleted,
        Found,
        FoundRanges,
        Generation,
        ListItem,
        ObjectMetadata,
        Rejected,
        Unchanged,
        Written,
    )

__all__ = ["Store"]

B_co = TypeVar("B_co", bound=Backend, covariant=True)
O = TypeVar("O")
T = TypeVar("T")
R = TypeVar("R")
Q = TypeVar("Q", bound="str | tuple[str, ByteRange | None]")
"""A `get_many` request: a key, or a key and the part of its value to read."""


@final
class Store(Generic[B_co]):
    """A backend seen from a root directory.

    Construct a store from a backend; the store you construct owns the backend and closes it.
    `store / "a"` and `with_read_only()` return views on the same backend, which share its state and
    never close it. A view of a read-only store is read-only too.

    `backend` gives direct access to the backend, which bypasses the key grammar, the root, and
    read-only and closed checks.

    Stores have no value equality: `==` is identity. Use `shares_backend` and `overlaps` to compare
    stores.

    Parameters
    ----------
    backend
        The backend to read and write through.
    root
        The directory within the backend that this store's keys are relative to.
    read_only
        Refuse writes through this store. A backend that can't write makes every store over it
        read-only.
    """

    __slots__ = ("_backend", "_owner", "_read_only", "_root")

    def __init__(self, backend: B_co, *, root: str = "", read_only: bool = False) -> None:
        self._backend: Final[B_co] = backend
        self._root: Final[str] = normalize_directory(root)
        self._read_only: Final[bool] = read_only or not backend.writable
        self._owner = True

    # ----------------------------------------------------------------- views and identity

    @property
    def backend(self) -> B_co:
        """The backend this store reads and writes through."""
        return self._backend

    @property
    def root(self) -> str:
        """This store's directory within the backend; `""` is the backend's root."""
        return self._root

    @property
    def read_only(self) -> bool:
        """Whether writes through this store are refused."""
        return self._read_only

    @property
    def closed(self) -> bool:
        """Whether the backend has been closed."""
        return self._backend.closed

    def _view(self, *, root: str, read_only: bool) -> Self:
        view = type(self)(self._backend, root=root, read_only=read_only)
        view._owner = False
        return view

    def __truediv__(self, directory: str) -> Self:
        """A view on the same backend, rooted at `directory` under this store's root."""
        return self._view(
            root=join(self._root, normalize_directory(directory)), read_only=self._read_only
        )

    def with_read_only(self, read_only: bool = True) -> Self:
        """A view on the same backend and root that refuses writes, or accepts them.

        A read-only store can't produce a writable view: `with_read_only(False)` on one raises
        `ReadOnlyError`.
        """
        if self._read_only and not read_only:
            raise ReadOnlyError(f"{self!r} is read-only; its views are read-only too.")
        return self._view(root=self._root, read_only=read_only)

    def shares_backend(self, other: Store[Backend]) -> bool:
        """Whether `other` reads and writes through the same backend object."""
        return self._backend is other._backend

    def overlaps(self, other: Store[Backend]) -> bool:
        """Whether a key could be visible through both stores: same backend, nested roots."""
        return self.shares_backend(other) and (
            is_within(self._root, other._root) or is_within(other._root, self._root)
        )

    def __repr__(self) -> str:
        flags = ", read_only=True" if self._read_only else ""
        return f"{type(self).__name__}({self._backend!r}, root={self._root!r}{flags})"

    def __reduce__(self) -> tuple[object, ...]:
        return (_restore_store, (type(self), self._backend, self._root, self._read_only))

    # ----------------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Close the backend, if this store owns it (it was constructed, not derived)."""
        if self._owner:
            self._backend.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ----------------------------------------------------------------- path handling

    def _check_open(self) -> None:
        if self._backend.closed:
            raise StoreClosedError(f"{self!r} is closed.")

    def _path(self, key: str) -> str:
        self._check_open()
        return join(self._root, check_key(key))

    def _directory(self, directory: str) -> str:
        self._check_open()
        return join(self._root, normalize_directory(directory))

    def _open_items(self, items: Iterator[T]) -> Iterator[T]:
        """`items`, refusing to continue once the backend closes."""
        for item in items:
            self._check_open()
            yield item

    async def _open_items_async(self, items: AsyncIterator[T]) -> AsyncIterator[T]:
        async for item in items:
            self._check_open()
            yield item

    def _writable_path(self, key: str) -> str:
        path = self._path(key)
        if self._read_only:
            raise ReadOnlyError(f"{self!r} is read-only.")
        return path

    def _writable_directory(self, directory: str) -> str:
        path = self._directory(directory)
        if self._read_only:
            raise ReadOnlyError(f"{self!r} is read-only.")
        return path

    def _read_request(
        self, request: str | tuple[str, ByteRange | None]
    ) -> tuple[str, ByteRange | None]:
        if isinstance(request, str):
            return self._path(request), None
        key, byte_range = request
        return self._path(key), byte_range

    def _into_request(
        self, request: tuple[str, O] | tuple[str, O, ByteRange | None]
    ) -> tuple[str, O, ByteRange | None]:
        if len(request) == 2:
            key, out = request
            return self._path(key), out, None
        key, out, byte_range = request
        return self._path(key), out, byte_range

    def _writable_paths(self, keys: Sequence[str]) -> list[str]:
        """The backend path of each key, refusing a key named twice: its outcome would be a race."""
        self._check_open()
        if self._read_only:
            raise ReadOnlyError(f"{self!r} is read-only.")
        paths = [self._writable_path(key) for key in keys]
        seen: set[str] = set()
        for key, path in zip(keys, paths, strict=True):
            if path in seen:
                raise ValueError(f"The batch names the key {key!r} more than once.")
            seen.add(path)
        return paths

    def _streamed(
        self, batch: Sequence[T], results: Iterator[tuple[int, R]]
    ) -> Iterator[tuple[T, R]]:
        """`results` paired with their requests; a failure raises `BatchError`."""
        done: set[int] = set()
        try:
            while True:
                # Checked before each result is taken, which may start more reads.
                self._check_open()
                pair = next(results, None)
                if pair is None:
                    return
                index, result = pair
                done.add(index)
                yield batch[index], result
        except ExceptionGroup as group:
            raise _batch_error(group, batch, {}, done) from None
        finally:
            # A caller that stops early must not leave the batch's reads to the garbage collector.
            if isinstance(results, Generator):
                results.close()

    async def _streamed_async(
        self, batch: Sequence[T], results: AsyncIterator[tuple[int, R]]
    ) -> AsyncIterator[tuple[T, R]]:
        done: set[int] = set()
        try:
            while True:
                self._check_open()
                pair = await anext(results, None)
                if pair is None:
                    return
                index, result = pair
                done.add(index)
                yield batch[index], result
        except ExceptionGroup as group:
            raise _batch_error(group, batch, {}, done) from None
        finally:
            if isinstance(results, AsyncGenerator):
                await results.aclose()

    @staticmethod
    def _finished(batch: Sequence[T], results: Iterator[tuple[int, R]]) -> list[tuple[T, R]]:
        """Every result, in request order; a failure raises `BatchError` with those that landed."""
        completed: dict[int, R] = {}
        try:
            completed.update(results)  # one pair at a time, so a failure keeps those before it
        except ExceptionGroup as group:
            raise _batch_error(group, batch, completed, completed.keys()) from None
        return [(request, completed[index]) for index, request in enumerate(batch)]

    @staticmethod
    async def _finished_async(
        batch: Sequence[T], results: AsyncIterator[tuple[int, R]]
    ) -> list[tuple[T, R]]:
        completed: dict[int, R] = {}
        try:
            async for index, result in results:
                completed[index] = result  # noqa: PERF403 -- a failure keeps those before it
        except ExceptionGroup as group:
            raise _batch_error(group, batch, completed, completed.keys()) from None
        return [(request, completed[index]) for index, request in enumerate(batch)]

    # ----------------------------------------------------------------- sync reads

    def get(
        self: Store[Readable], key: str, *, byte_range: ByteRange | None = None
    ) -> Found | Missing:
        """The value at `key`, or the part `byte_range` selects."""
        return self._backend.get(self._path(key), byte_range)

    def get_if_changed(
        self: Store[Readable],
        key: str,
        generation: Generation,
        *,
        byte_range: ByteRange | None = None,
    ) -> Found | Missing | Unchanged:
        """`get`, unless the value still has `generation`: then `Unchanged`, and no bytes are read.

        This revalidates a cached copy, as an HTTP `If-None-Match` request does.
        """
        return self._backend.get_if_changed(self._path(key), generation, byte_range)

    def get_ranges(
        self: Store[Readable], key: str, ranges: Sequence[ByteRange]
    ) -> FoundRanges | Missing:
        """Several parts of the value at `key`, all from one generation."""
        return self._backend.get_ranges(self._path(key), ranges)

    def get_into(
        self: Store[ReadableInto[O]], key: str, out: O, *, byte_range: ByteRange | None = None
    ) -> Copied | Missing:
        """Copy the value at `key` (or part of it) into `out`, a buffer the caller allocated.

        `out` is whatever buffer type the backend reads into: any writable C-contiguous `Buffer`
        for the built-in backends, device memory for a GPU direct storage backend. Raises
        `ValueError` if `out` is shorter than the bytes to copy or can't be written.
        """
        return self._backend.get_into(self._path(key), out, byte_range)

    def head(self: Store[Readable], key: str) -> ObjectMetadata | Missing:
        """The size and generation of the value at `key`."""
        return self._backend.head(self._path(key))

    def exists(self: Store[Readable], key: str) -> bool:
        """Whether `key` holds a value."""
        return not isinstance(self.head(key), Missing)

    # ----------------------------------------------------------------- sync writes

    def put(self: Store[Writable], key: str, value: Buffer) -> Written:
        """Write `value` at `key`."""
        return self._backend.put(self._writable_path(key), byte_view(value, writable=False))

    def put_if_absent(self: Store[Writable], key: str, value: Buffer) -> Written | Rejected:
        """Write `value` at `key` if the key holds no value; `Rejected` if it holds one."""
        return self._backend.put_if_absent(
            self._writable_path(key), byte_view(value, writable=False)
        )

    def put_if_match(
        self: Store[WritableIfMatch], key: str, value: Buffer, generation: Generation
    ) -> Written | Rejected | Missing:
        """Write `value` at `key` if the key's value has `generation`.

        `Rejected` if the key holds another generation, `Missing` if it holds no value.
        """
        return self._backend.put_if_match(
            self._writable_path(key), byte_view(value, writable=False), generation
        )

    def delete(self: Store[Writable], key: str) -> Deleted | Missing:
        """Delete the key `key`. A directory is not a key: deleting one finds it `Missing`."""
        return self._backend.delete(self._writable_path(key))

    def delete_if_match(
        self: Store[DeletableIfMatch], key: str, generation: Generation
    ) -> Deleted | Rejected | Missing:
        """Delete the key `key` if its value has `generation`."""
        return self._backend.delete_if_match(self._writable_path(key), generation)

    def delete_dir(self: Store[Writable], directory: str = "") -> None:
        """Delete every key under `directory` (by default, every key in this store)."""
        self._backend.delete_dir(self._writable_directory(directory))

    # ----------------------------------------------------------------- sync batches

    def get_many(
        self: Store[Readable], requests: Iterable[Q]
    ) -> Iterator[tuple[Q, Found | Missing]]:
        """`get` for many keys, yielding `(request, result)` pairs as reads finish.

        A request is a key, or a `(key, byte_range)` pair, and comes back as it was given, so
        `dict(store.get_many(keys))` maps keys to results. Keys are checked when `get_many` is
        called; the reads happen as the pairs are taken, in no particular order. A batch is not
        atomic: each read is its own `get`, from whatever generation it finds (`get_ranges` reads
        several parts of one value from one generation).

        If reads raise, the iterator yields the reads already running that succeed, then raises
        `BatchError` with the requests left in `remaining`; its `completed` is empty, since those
        results were yielded.
        """
        batch = list(requests)
        self._check_open()
        paths = [self._read_request(request) for request in batch]
        backend = self._backend
        if isinstance(backend, ReadableMany):
            return self._streamed(batch, backend.get_many(paths))
        return self._streamed(batch, run_many(backend.get, paths))

    def get_into_many(
        self: Store[ReadableInto[O]],
        requests: Iterable[tuple[str, O] | tuple[str, O, ByteRange | None]],
    ) -> Iterator[tuple[tuple[str, O] | tuple[str, O, ByteRange | None], Copied | Missing]]:
        """`get_into` for many `(key, out)` or `(key, out, byte_range)` requests, as `get_many`.

        Each request names its own buffer; two requests sharing one write it in no set order.
        """
        batch = list(requests)
        self._check_open()
        paths = [self._into_request(request) for request in batch]
        backend = self._backend
        if isinstance(backend, ReadableIntoMany):
            many = cast("ReadableIntoMany[O]", backend)
            return self._streamed(batch, many.get_into_many(paths))
        return self._streamed(batch, run_many(backend.get_into, paths))

    def put_many(
        self: Store[Writable], items: Iterable[tuple[str, Buffer]]
    ) -> list[tuple[str, Written]]:
        """`put` each `(key, value)`, returning `(key, result)` pairs in the order given.

        Every key is checked, and every value viewed, before anything is written; a key named twice
        raises `ValueError`. A batch is not atomic. If writes raise, the writes already running
        finish, no others start, and `BatchError` reports the keys written (`completed`, with their
        results) and the rest (`remaining`).
        """
        batch = list(items)
        keys = [key for key, _ in batch]
        paths = self._writable_paths(keys)
        requests = [
            (path, byte_view(value, writable=False))
            for path, (_, value) in zip(paths, batch, strict=True)
        ]
        backend = self._backend
        if isinstance(backend, WritableMany):
            return self._finished(keys, backend.put_many(requests))
        return self._finished(keys, run_many(backend.put, requests))

    def delete_many(
        self: Store[Writable], keys: Iterable[str]
    ) -> list[tuple[str, Deleted | Missing]]:
        """`delete` each key, returning `(key, result)` pairs in the order given, as `put_many`."""
        batch = list(keys)
        paths = self._writable_paths(batch)
        backend = self._backend
        if isinstance(backend, WritableMany):
            return self._finished(batch, backend.delete_many(paths))
        return self._finished(batch, run_many(backend.delete, [(path,) for path in paths]))

    # ----------------------------------------------------------------- sync listing

    def list(self: Store[Listable], directory: str = "") -> Iterator[str]:
        """Every key under `directory`, relative to this store's root, in no particular order.

        The directory is checked when `list` is called; iterating after the store closes raises
        `StoreClosedError`.
        """
        paths = self._backend.list(self._directory(directory))
        return self._open_items(relativize(path, self._root) for path in paths)

    def list_dir(self: Store[Listable], directory: str = "") -> Iterator[ListItem]:
        """The direct children of `directory`, named relative to it.

        A backend may report a `Prefix` that holds no keys, such as an empty directory on a
        filesystem.
        """
        return self._open_items(self._backend.list_dir(self._directory(directory)))

    # ----------------------------------------------------------------- async reads

    async def get_async(
        self: Store[ReadableAsync], key: str, *, byte_range: ByteRange | None = None
    ) -> Found | Missing:
        """Async `get`."""
        return await self._backend.get_async(self._path(key), byte_range)

    async def get_if_changed_async(
        self: Store[ReadableAsync],
        key: str,
        generation: Generation,
        *,
        byte_range: ByteRange | None = None,
    ) -> Found | Missing | Unchanged:
        """Async `get_if_changed`."""
        return await self._backend.get_if_changed_async(self._path(key), generation, byte_range)

    async def get_ranges_async(
        self: Store[ReadableAsync], key: str, ranges: Sequence[ByteRange]
    ) -> FoundRanges | Missing:
        """Async `get_ranges`."""
        return await self._backend.get_ranges_async(self._path(key), ranges)

    async def get_into_async(
        self: Store[ReadableIntoAsync[O]],
        key: str,
        out: O,
        *,
        byte_range: ByteRange | None = None,
    ) -> Copied | Missing:
        """Async `get_into`."""
        return await self._backend.get_into_async(self._path(key), out, byte_range)

    async def head_async(self: Store[ReadableAsync], key: str) -> ObjectMetadata | Missing:
        """Async `head`."""
        return await self._backend.head_async(self._path(key))

    async def exists_async(self: Store[ReadableAsync], key: str) -> bool:
        """Async `exists`."""
        return not isinstance(await self.head_async(key), Missing)

    # ----------------------------------------------------------------- async writes

    async def put_async(self: Store[WritableAsync], key: str, value: Buffer) -> Written:
        """Async `put`."""
        return await self._backend.put_async(
            self._writable_path(key), byte_view(value, writable=False)
        )

    async def put_if_absent_async(
        self: Store[WritableAsync], key: str, value: Buffer
    ) -> Written | Rejected:
        """Async `put_if_absent`."""
        return await self._backend.put_if_absent_async(
            self._writable_path(key), byte_view(value, writable=False)
        )

    async def put_if_match_async(
        self: Store[WritableIfMatchAsync], key: str, value: Buffer, generation: Generation
    ) -> Written | Rejected | Missing:
        """Async `put_if_match`."""
        return await self._backend.put_if_match_async(
            self._writable_path(key), byte_view(value, writable=False), generation
        )

    async def delete_async(self: Store[WritableAsync], key: str) -> Deleted | Missing:
        """Async `delete`."""
        return await self._backend.delete_async(self._writable_path(key))

    async def delete_if_match_async(
        self: Store[DeletableIfMatchAsync], key: str, generation: Generation
    ) -> Deleted | Rejected | Missing:
        """Async `delete_if_match`."""
        return await self._backend.delete_if_match_async(self._writable_path(key), generation)

    async def delete_dir_async(self: Store[WritableAsync], directory: str = "") -> None:
        """Async `delete_dir`."""
        await self._backend.delete_dir_async(self._writable_directory(directory))

    # ----------------------------------------------------------------- async batches

    def get_many_async(
        self: Store[ReadableAsync], requests: Iterable[Q]
    ) -> AsyncIterator[tuple[Q, Found | Missing]]:
        """Async `get_many`."""
        batch = list(requests)
        self._check_open()
        paths = [self._read_request(request) for request in batch]
        backend = self._backend
        if isinstance(backend, ReadableManyAsync):
            return self._streamed_async(batch, backend.get_many_async(paths))
        return self._streamed_async(batch, run_many_async(backend.get_async, paths))

    def get_into_many_async(
        self: Store[ReadableIntoAsync[O]],
        requests: Iterable[tuple[str, O] | tuple[str, O, ByteRange | None]],
    ) -> AsyncIterator[tuple[tuple[str, O] | tuple[str, O, ByteRange | None], Copied | Missing]]:
        """Async `get_into_many`."""
        batch = list(requests)
        self._check_open()
        paths = [self._into_request(request) for request in batch]
        backend = self._backend
        if isinstance(backend, ReadableIntoManyAsync):
            many = cast("ReadableIntoManyAsync[O]", backend)
            return self._streamed_async(batch, many.get_into_many_async(paths))
        return self._streamed_async(batch, run_many_async(backend.get_into_async, paths))

    async def put_many_async(
        self: Store[WritableAsync], items: Iterable[tuple[str, Buffer]]
    ) -> list[tuple[str, Written]]:
        """Async `put_many`."""
        batch = list(items)
        keys = [key for key, _ in batch]
        paths = self._writable_paths(keys)
        requests = [
            (path, byte_view(value, writable=False))
            for path, (_, value) in zip(paths, batch, strict=True)
        ]
        backend = self._backend
        if isinstance(backend, WritableManyAsync):
            return await self._finished_async(keys, backend.put_many_async(requests))
        return await self._finished_async(keys, run_many_async(backend.put_async, requests))

    async def delete_many_async(
        self: Store[WritableAsync], keys: Iterable[str]
    ) -> list[tuple[str, Deleted | Missing]]:
        """Async `delete_many`."""
        batch = list(keys)
        paths = self._writable_paths(batch)
        backend = self._backend
        if isinstance(backend, WritableManyAsync):
            return await self._finished_async(batch, backend.delete_many_async(paths))
        return await self._finished_async(
            batch, run_many_async(backend.delete_async, [(path,) for path in paths])
        )

    # ----------------------------------------------------------------- async listing

    def list_async(self: Store[ListableAsync], directory: str = "") -> AsyncIterator[str]:
        """Async `list`."""
        paths = self._backend.list_async(self._directory(directory))
        return self._open_items_async(_relativized(paths, self._root))

    def list_dir_async(self: Store[ListableAsync], directory: str = "") -> AsyncIterator[ListItem]:
        """Async `list_dir`."""
        return self._open_items_async(self._backend.list_dir_async(self._directory(directory)))


def _batch_error(
    group: ExceptionGroup[Exception],
    batch: Sequence[T],
    completed: dict[int, R],
    done: Iterable[int],
) -> BatchError:
    finished = set(done)
    return BatchError(
        f"{len(group.exceptions)} of {len(batch)} operations in a batch failed.",
        group.exceptions,
        completed=tuple((batch[index], result) for index, result in sorted(completed.items())),
        remaining=tuple(request for index, request in enumerate(batch) if index not in finished),
    )


async def _relativized(paths: AsyncIterator[str], root: str) -> AsyncIterator[str]:
    async for path in paths:
        yield relativize(path, root)


def _restore_store(
    cls: type[Store[Backend]], backend: Backend, root: str, read_only: bool
) -> Store[Backend]:
    return cls(backend, root=root, read_only=read_only)
