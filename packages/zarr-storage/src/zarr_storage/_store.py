"""`Store` and `AsyncStore`: views of a backend at a root, under a context, through sync methods or
coroutines.

They are where the key grammar, the root, read-only access and the closed check are enforced,
once, for every backend; a backend only sees backend paths. They are also where operations are
scheduled and limited: every operation holds a slot of the view's context's limiter for the
backend's `resource_key`, and blocking work runs on the context's thread pool.

A slot is taken where the work runs. A sync backend's operation called through an `AsyncStore`
takes its slot on the pool thread that runs it, so no operation ever holds a slot while it waits
for a thread: pool threads waiting for slots and slot holders waiting for threads can't block each
other.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Final, Generic, Self, TypeVar, TypeVarTuple, cast, final

from zarr_storage._batch import run_many, run_many_async
from zarr_storage._buffers import byte_view
from zarr_storage._errors import BatchError, ReadOnlyError, StoreClosedError
from zarr_storage._keys import check_key, is_within, join, normalize_directory, relativize
from zarr_storage._protocols import (
    Backend,
    ClosableAsync,
    DeletableIfMatchAsync,
    ListableAsync,
    ReadableAsync,
    ReadableIntoAsync,
    ReadableIntoMany,
    ReadableIntoManyAsync,
    ReadableMany,
    ReadableManyAsync,
    WritableAsync,
    WritableIfMatchAsync,
    WritableMany,
    WritableManyAsync,
)
from zarr_storage._resources import default_context
from zarr_storage._types import Missing

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
    from types import TracebackType

    from typing_extensions import Buffer

    from zarr_storage._protocols import (
        DeletableIfMatch,
        Listable,
        Readable,
        ReadableInto,
        Writable,
        WritableIfMatch,
    )
    from zarr_storage._resources import Context, Limiter, Slot
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

__all__ = ["AsyncStore", "Store"]

B_co = TypeVar("B_co", bound=Backend, covariant=True)
O = TypeVar("O")
T = TypeVar("T")
R = TypeVar("R")
Ts = TypeVarTuple("Ts")
Q = TypeVar("Q", bound="str | tuple[str, ByteRange | None]")
"""A `get_many` request: a key, or a key and the part of its value to read."""


def _limited(limiter: Limiter, fn: Callable[[*Ts], R], *args: *Ts) -> R:
    """`fn(*args)`, holding a slot of `limiter`, on the calling thread."""
    with limiter.slot():
        return fn(*args)


def _list_all(fn: Callable[[str], Iterator[T]], directory: str) -> list[T]:
    return list(fn(directory))


class _View(Generic[B_co]):
    """What `Store` and `AsyncStore` share: a backend, a root, read-only access, a context, and
    ownership."""

    __slots__ = ("_backend", "_context", "_limiter", "_owner", "_read_only", "_root")

    def __init__(
        self,
        backend: B_co,
        *,
        root: str = "",
        read_only: bool = False,
        context: Context | None = None,
    ) -> None:
        self._backend: Final[B_co] = backend
        self._root: Final[str] = normalize_directory(root)
        self._read_only: Final[bool] = read_only or not backend.writable
        self._context: Final[Context] = default_context() if context is None else context
        self._limiter: Final[Limiter] = self._context.limiter(backend.resource_key)
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
    def context(self) -> Context:
        """The resources this store's operations draw on."""
        return self._context

    @property
    def closed(self) -> bool:
        """Whether the backend has been closed."""
        return self._backend.closed

    def _view(self, *, root: str, read_only: bool, context: Context) -> Self:
        view = type(self)(self._backend, root=root, read_only=read_only, context=context)
        view._owner = False
        return view

    def __truediv__(self, directory: str) -> Self:
        """A view on the same backend, rooted at `directory` under this store's root."""
        return self._view(
            root=join(self._root, normalize_directory(directory)),
            read_only=self._read_only,
            context=self._context,
        )

    def with_read_only(self, read_only: bool = True) -> Self:
        """A view on the same backend and root that refuses writes, or accepts them.

        A read-only store can't produce a writable view: `with_read_only(False)` on one raises
        `ReadOnlyError`.
        """
        if self._read_only and not read_only:
            raise ReadOnlyError(f"{self!r} is read-only; its views are read-only too.")
        return self._view(root=self._root, read_only=read_only, context=self._context)

    def with_context(self, context: Context) -> Self:
        """A view on the same backend and root whose operations draw on `context`."""
        return self._view(root=self._root, read_only=self._read_only, context=context)

    def limited(self, limit: int) -> Self:
        """A view whose operations, together, run at most `limit` at once.

        The limit narrows this view's: an operation holds a slot of both. Batches through the view
        start at most `limit` operations at once.
        """
        return self.with_context(self._context.limited(self._backend.resource_key, limit))

    def shares_backend(self, other: _View[Backend]) -> bool:
        """Whether `other` reads and writes through the same backend object."""
        return self._backend is other._backend

    def overlaps(self, other: _View[Backend]) -> bool:
        """Whether a key could be visible through both stores: same backend, nested roots."""
        return self.shares_backend(other) and (
            is_within(self._root, other._root) or is_within(other._root, self._root)
        )

    def __repr__(self) -> str:
        flags = ", read_only=True" if self._read_only else ""
        return f"{type(self).__name__}({self._backend!r}, root={self._root!r}{flags})"

    def __reduce__(self) -> tuple[object, ...]:
        return (
            _restore_store,
            (type(self), self._backend, self._root, self._read_only, self._context),
        )

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

    def _slot(self) -> Slot:
        return self._limiter.slot()

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

    def _put_requests(
        self, items: Iterable[tuple[str, Buffer]]
    ) -> tuple[list[str], list[tuple[str, memoryview]]]:
        batch = list(items)
        keys = [key for key, _ in batch]
        paths = self._writable_paths(keys)
        views = [byte_view(value, writable=False) for _, value in batch]
        return keys, list(zip(paths, views, strict=True))


@final
class Store(_View[B_co]):
    """A backend seen from a root directory, under a context, through sync methods.

    Construct a store from a backend; the store you construct owns the backend and closes it.
    `store / "a"`, `with_read_only()`, `with_context()`, `limited()` and `as_async()` return views
    on the same backend, which share its state and never close it. A view of a read-only store is
    read-only too.

    Which methods a store offers depends on its backend's capabilities: a method whose capability
    the backend lacks is a type error. `AsyncStore` offers the same operations as coroutines.

    Every operation holds a slot of the context's limiter for the backend's `resource_key`; batches
    run their operations on the context's thread pool, at most the limiter's limit at once.

    `backend` gives direct access to the backend, which bypasses the key grammar, the root, the
    read-only and closed checks, and the context's limits.

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
    context
        The resources operations draw on. Left out, `Context.default()`.
    """

    __slots__ = ()

    def __init__(
        self,
        backend: B_co,
        *,
        root: str = "",
        read_only: bool = False,
        context: Context | None = None,
    ) -> None:
        super().__init__(backend, root=root, read_only=read_only, context=context)

    def as_async(self) -> AsyncStore[B_co]:
        """An `AsyncStore` view on the same backend, root and context; it doesn't own the backend."""
        view = AsyncStore(
            self._backend, root=self._root, read_only=self._read_only, context=self._context
        )
        view._owner = False
        return view

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

    # ----------------------------------------------------------------- batch plumbing

    def _run_many(
        self, fn: Callable[[*Ts], R], requests: Sequence[tuple[*Ts]]
    ) -> Iterator[tuple[int, R]]:
        """`fn` over `requests` on the context's pool, at most the limiter's limit at once."""
        return run_many(fn, requests, self._context.threads, self._limiter.limit)

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

    @staticmethod
    def _finished(batch: Sequence[T], results: Iterator[tuple[int, R]]) -> list[tuple[T, R]]:
        """Every result, in request order; a failure raises `BatchError` with those that landed."""
        completed: dict[int, R] = {}
        try:
            completed.update(results)  # one pair at a time, so a failure keeps those before it
        except ExceptionGroup as group:
            raise _batch_error(group, batch, completed, completed.keys()) from None
        return [(request, completed[index]) for index, request in enumerate(batch)]

    # ----------------------------------------------------------------- operations on backend paths

    def _get(self: Store[Readable], path: str, byte_range: ByteRange | None) -> Found | Missing:
        with self._slot():
            return self._backend.get(path, byte_range)

    def _get_into(
        self: Store[ReadableInto[O]], path: str, out: O, byte_range: ByteRange | None
    ) -> Copied | Missing:
        with self._slot():
            return self._backend.get_into(path, out, byte_range)

    def _put(self: Store[Writable], path: str, value: memoryview) -> Written:
        with self._slot():
            return self._backend.put(path, value)

    def _delete(self: Store[Writable], path: str) -> Deleted | Missing:
        with self._slot():
            return self._backend.delete(path)

    # ----------------------------------------------------------------- reads

    def get(
        self: Store[Readable], key: str, *, byte_range: ByteRange | None = None
    ) -> Found | Missing:
        """The value at `key`, or the part `byte_range` selects."""
        return self._get(self._path(key), byte_range)

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
        path = self._path(key)
        with self._slot():
            return self._backend.get_if_changed(path, generation, byte_range)

    def get_ranges(
        self: Store[Readable], key: str, ranges: Sequence[ByteRange]
    ) -> FoundRanges | Missing:
        """Several parts of the value at `key`, all from one generation."""
        path = self._path(key)
        with self._slot():
            return self._backend.get_ranges(path, ranges)

    def get_into(
        self: Store[ReadableInto[O]], key: str, out: O, *, byte_range: ByteRange | None = None
    ) -> Copied | Missing:
        """Copy the value at `key` (or part of it) into `out`, a buffer the caller allocated.

        `out` is whatever buffer type the backend reads into: any writable C-contiguous `Buffer`
        for the built-in backends, device memory for a GPU direct storage backend. Raises
        `ValueError` if `out` is shorter than the bytes to copy or can't be written.
        """
        return self._get_into(self._path(key), out, byte_range)

    def head(self: Store[Readable], key: str) -> ObjectMetadata | Missing:
        """The size and generation of the value at `key`."""
        path = self._path(key)
        with self._slot():
            return self._backend.head(path)

    def exists(self: Store[Readable], key: str) -> bool:
        """Whether `key` holds a value."""
        return not isinstance(self.head(key), Missing)

    # ----------------------------------------------------------------- writes

    def put(self: Store[Writable], key: str, value: Buffer) -> Written:
        """Write `value` at `key`."""
        return self._put(self._writable_path(key), byte_view(value, writable=False))

    def put_if_absent(self: Store[Writable], key: str, value: Buffer) -> Written | Rejected:
        """Write `value` at `key` if the key holds no value; `Rejected` if it holds one."""
        path = self._writable_path(key)
        view = byte_view(value, writable=False)
        with self._slot():
            return self._backend.put_if_absent(path, view)

    def put_if_match(
        self: Store[WritableIfMatch], key: str, value: Buffer, generation: Generation
    ) -> Written | Rejected | Missing:
        """Write `value` at `key` if the key's value has `generation`.

        `Rejected` if the key holds another generation, `Missing` if it holds no value.
        """
        path = self._writable_path(key)
        view = byte_view(value, writable=False)
        with self._slot():
            return self._backend.put_if_match(path, view, generation)

    def delete(self: Store[Writable], key: str) -> Deleted | Missing:
        """Delete the key `key`. A directory is not a key: deleting one finds it `Missing`."""
        return self._delete(self._writable_path(key))

    def delete_if_match(
        self: Store[DeletableIfMatch], key: str, generation: Generation
    ) -> Deleted | Rejected | Missing:
        """Delete the key `key` if its value has `generation`."""
        path = self._writable_path(key)
        with self._slot():
            return self._backend.delete_if_match(path, generation)

    def delete_dir(self: Store[Writable], directory: str = "") -> None:
        """Delete every key under `directory` (by default, every key in this store)."""
        path = self._writable_directory(directory)
        with self._slot():
            self._backend.delete_dir(path)

    # ----------------------------------------------------------------- batches

    def get_many(
        self: Store[Readable], requests: Iterable[Q]
    ) -> Iterator[tuple[Q, Found | Missing]]:
        """`get` for many keys, yielding `(request, result)` pairs as reads finish.

        A request is a key, or a `(key, byte_range)` pair, and comes back as it was given, so
        `dict(store.get_many(keys))` maps keys to results. Keys are checked when `get_many` is
        called; the reads run on the context's thread pool as the pairs are taken, at most the
        limiter's limit at once, in no particular order. A batch is not atomic: each read is its own
        `get`, from whatever generation it finds (`get_ranges` reads several parts of one value from
        one generation).

        If reads raise, the iterator yields the reads already running that succeed, then raises
        `BatchError` with the requests left in `remaining`; its `completed` is empty, since those
        results were yielded.
        """
        batch = list(requests)
        self._check_open()
        paths = [self._read_request(request) for request in batch]
        backend = self._backend
        if isinstance(backend, ReadableMany):
            return self._streamed(batch, backend.get_many(paths, self._context))
        return self._streamed(batch, self._run_many(self._get, paths))

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
            return self._streamed(batch, many.get_into_many(paths, self._context))
        return self._streamed(batch, self._run_many(self._get_into, paths))

    def put_many(
        self: Store[Writable], items: Iterable[tuple[str, Buffer]]
    ) -> list[tuple[str, Written]]:
        """`put` each `(key, value)`, returning `(key, result)` pairs in the order given.

        Every key is checked, and every value viewed, before anything is written; a key named twice
        raises `ValueError`. A batch is not atomic. If writes raise, the writes already running
        finish, no others start, and `BatchError` reports the keys written (`completed`, with their
        results) and the rest (`remaining`).
        """
        keys, requests = self._put_requests(items)
        backend = self._backend
        if isinstance(backend, WritableMany):
            return self._finished(keys, backend.put_many(requests, self._context))
        return self._finished(keys, self._run_many(self._put, requests))

    def delete_many(
        self: Store[Writable], keys: Iterable[str]
    ) -> list[tuple[str, Deleted | Missing]]:
        """`delete` each key, returning `(key, result)` pairs in the order given, as `put_many`."""
        batch = list(keys)
        paths = self._writable_paths(batch)
        backend = self._backend
        if isinstance(backend, WritableMany):
            return self._finished(batch, backend.delete_many(paths, self._context))
        return self._finished(batch, self._run_many(self._delete, [(path,) for path in paths]))

    # ----------------------------------------------------------------- listing

    def _open_items(self, items: Iterator[T]) -> Iterator[T]:
        """`items`, refusing to continue once the backend closes."""
        for item in items:
            self._check_open()
            yield item

    def list(self: Store[Listable], directory: str = "") -> Iterator[str]:
        """Every key under `directory`, relative to this store's root, in no particular order.

        The directory is checked when `list` is called; iterating after the store closes raises
        `StoreClosedError`. Listings hold no slot of the context's limiter.
        """
        paths = self._backend.list(self._directory(directory))
        return self._open_items(relativize(path, self._root) for path in paths)

    def list_dir(self: Store[Listable], directory: str = "") -> Iterator[ListItem]:
        """The direct children of `directory`, named relative to it.

        A backend may report a `Prefix` that holds no keys, such as an empty directory on a
        filesystem.
        """
        return self._open_items(self._backend.list_dir(self._directory(directory)))


@final
class AsyncStore(_View[B_co]):
    """A backend seen from a root directory, under a context, through coroutines.

    The async twin of `Store`: the same operations under the same names, awaited. A backend's async
    methods are used where it has them; otherwise its sync methods run on the context's thread
    pool, so a backend that only blocks needs no async code. Listings and lazy batches are async
    iterators. Construct one from a backend, or take `store.as_async()`; `as_sync()` goes back.
    Close it with `await store.aclose()` or `async with`.

    Parameters
    ----------
    backend
        The backend to read and write through.
    root
        The directory within the backend that this store's keys are relative to.
    read_only
        Refuse writes through this store. A backend that can't write makes every store over it
        read-only.
    context
        The resources operations draw on. Left out, `Context.default()`.
    """

    __slots__ = ()

    def __init__(
        self,
        backend: B_co,
        *,
        root: str = "",
        read_only: bool = False,
        context: Context | None = None,
    ) -> None:
        super().__init__(backend, root=root, read_only=read_only, context=context)

    def as_sync(self) -> Store[B_co]:
        """A `Store` view on the same backend, root and context; it doesn't own the backend."""
        view = Store(
            self._backend, root=self._root, read_only=self._read_only, context=self._context
        )
        view._owner = False
        return view

    # ----------------------------------------------------------------- lifecycle

    async def aclose(self) -> None:
        """Close the backend, if this store owns it (it was constructed, not derived).

        A backend that is `ClosableAsync` is closed with `aclose`; any other with `close`.
        """
        if not self._owner:
            return
        backend = self._backend
        if isinstance(backend, ClosableAsync):
            await backend.aclose()
        else:
            backend.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ----------------------------------------------------------------- batch plumbing

    async def _in_thread(self, fn: Callable[[*Ts], R], *args: *Ts) -> R:
        """`fn(*args)` on the context's pool, holding a slot taken on the pool thread."""
        return await self._context.threads.run(_limited, self._limiter, fn, *args)

    async def _streamed(
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
    async def _finished(
        batch: Sequence[T], results: AsyncIterator[tuple[int, R]]
    ) -> list[tuple[T, R]]:
        completed: dict[int, R] = {}
        try:
            async for index, result in results:
                completed[index] = result  # noqa: PERF403 -- a failure keeps those before it
        except ExceptionGroup as group:
            raise _batch_error(group, batch, completed, completed.keys()) from None
        return [(request, completed[index]) for index, request in enumerate(batch)]

    # ----------------------------------------------------------------- operations on backend paths

    async def _get(
        self: AsyncStore[ReadableAsync] | AsyncStore[Readable],
        path: str,
        byte_range: ByteRange | None,
    ) -> Found | Missing:
        backend = self._backend
        if isinstance(backend, ReadableAsync):
            async with self._slot():
                return await backend.get_async(path, byte_range)
        return await self._in_thread(backend.get, path, byte_range)

    async def _get_into(
        self: AsyncStore[ReadableIntoAsync[O]] | AsyncStore[ReadableInto[O]],
        path: str,
        out: O,
        byte_range: ByteRange | None,
    ) -> Copied | Missing:
        backend = self._backend
        if isinstance(backend, ReadableIntoAsync):
            async with self._slot():
                return await backend.get_into_async(path, out, byte_range)
        return await self._in_thread(backend.get_into, path, out, byte_range)

    async def _put(
        self: AsyncStore[WritableAsync] | AsyncStore[Writable], path: str, value: memoryview
    ) -> Written:
        backend = self._backend
        if isinstance(backend, WritableAsync):
            async with self._slot():
                return await backend.put_async(path, value)
        return await self._in_thread(backend.put, path, value)

    async def _delete(
        self: AsyncStore[WritableAsync] | AsyncStore[Writable], path: str
    ) -> Deleted | Missing:
        backend = self._backend
        if isinstance(backend, WritableAsync):
            async with self._slot():
                return await backend.delete_async(path)
        return await self._in_thread(backend.delete, path)

    # ----------------------------------------------------------------- reads

    async def get(
        self: AsyncStore[ReadableAsync] | AsyncStore[Readable],
        key: str,
        *,
        byte_range: ByteRange | None = None,
    ) -> Found | Missing:
        """As `Store.get`."""
        return await self._get(self._path(key), byte_range)

    async def get_if_changed(
        self: AsyncStore[ReadableAsync] | AsyncStore[Readable],
        key: str,
        generation: Generation,
        *,
        byte_range: ByteRange | None = None,
    ) -> Found | Missing | Unchanged:
        """As `Store.get_if_changed`."""
        path = self._path(key)
        backend = self._backend
        if isinstance(backend, ReadableAsync):
            async with self._slot():
                return await backend.get_if_changed_async(path, generation, byte_range)
        return await self._in_thread(backend.get_if_changed, path, generation, byte_range)

    async def get_ranges(
        self: AsyncStore[ReadableAsync] | AsyncStore[Readable],
        key: str,
        ranges: Sequence[ByteRange],
    ) -> FoundRanges | Missing:
        """As `Store.get_ranges`."""
        path = self._path(key)
        backend = self._backend
        if isinstance(backend, ReadableAsync):
            async with self._slot():
                return await backend.get_ranges_async(path, ranges)
        return await self._in_thread(backend.get_ranges, path, ranges)

    async def get_into(
        self: AsyncStore[ReadableIntoAsync[O]] | AsyncStore[ReadableInto[O]],
        key: str,
        out: O,
        *,
        byte_range: ByteRange | None = None,
    ) -> Copied | Missing:
        """As `Store.get_into`."""
        return await self._get_into(self._path(key), out, byte_range)

    async def head(
        self: AsyncStore[ReadableAsync] | AsyncStore[Readable], key: str
    ) -> ObjectMetadata | Missing:
        """As `Store.head`."""
        path = self._path(key)
        backend = self._backend
        if isinstance(backend, ReadableAsync):
            async with self._slot():
                return await backend.head_async(path)
        return await self._in_thread(backend.head, path)

    async def exists(self: AsyncStore[ReadableAsync] | AsyncStore[Readable], key: str) -> bool:
        """As `Store.exists`."""
        return not isinstance(await self.head(key), Missing)

    # ----------------------------------------------------------------- writes

    async def put(
        self: AsyncStore[WritableAsync] | AsyncStore[Writable], key: str, value: Buffer
    ) -> Written:
        """As `Store.put`."""
        return await self._put(self._writable_path(key), byte_view(value, writable=False))

    async def put_if_absent(
        self: AsyncStore[WritableAsync] | AsyncStore[Writable], key: str, value: Buffer
    ) -> Written | Rejected:
        """As `Store.put_if_absent`."""
        path = self._writable_path(key)
        view = byte_view(value, writable=False)
        backend = self._backend
        if isinstance(backend, WritableAsync):
            async with self._slot():
                return await backend.put_if_absent_async(path, view)
        return await self._in_thread(backend.put_if_absent, path, view)

    async def put_if_match(
        self: AsyncStore[WritableIfMatchAsync] | AsyncStore[WritableIfMatch],
        key: str,
        value: Buffer,
        generation: Generation,
    ) -> Written | Rejected | Missing:
        """As `Store.put_if_match`."""
        path = self._writable_path(key)
        view = byte_view(value, writable=False)
        backend = self._backend
        if isinstance(backend, WritableIfMatchAsync):
            async with self._slot():
                return await backend.put_if_match_async(path, view, generation)
        return await self._in_thread(backend.put_if_match, path, view, generation)

    async def delete(
        self: AsyncStore[WritableAsync] | AsyncStore[Writable], key: str
    ) -> Deleted | Missing:
        """As `Store.delete`."""
        return await self._delete(self._writable_path(key))

    async def delete_if_match(
        self: AsyncStore[DeletableIfMatchAsync] | AsyncStore[DeletableIfMatch],
        key: str,
        generation: Generation,
    ) -> Deleted | Rejected | Missing:
        """As `Store.delete_if_match`."""
        path = self._writable_path(key)
        backend = self._backend
        if isinstance(backend, DeletableIfMatchAsync):
            async with self._slot():
                return await backend.delete_if_match_async(path, generation)
        return await self._in_thread(backend.delete_if_match, path, generation)

    async def delete_dir(
        self: AsyncStore[WritableAsync] | AsyncStore[Writable], directory: str = ""
    ) -> None:
        """As `Store.delete_dir`."""
        path = self._writable_directory(directory)
        backend = self._backend
        if isinstance(backend, WritableAsync):
            async with self._slot():
                await backend.delete_dir_async(path)
            return
        await self._in_thread(backend.delete_dir, path)

    # ----------------------------------------------------------------- batches

    def get_many(
        self: AsyncStore[ReadableAsync] | AsyncStore[Readable], requests: Iterable[Q]
    ) -> AsyncIterator[tuple[Q, Found | Missing]]:
        """As `Store.get_many`; the reads run as tasks, at most the limiter's limit at once."""
        batch = list(requests)
        self._check_open()
        paths = [self._read_request(request) for request in batch]
        backend = self._backend
        if isinstance(backend, ReadableManyAsync):
            return self._streamed(batch, backend.get_many_async(paths, self._context))
        return self._streamed(batch, run_many_async(self._get, paths, self._limiter.limit))

    def get_into_many(
        self: AsyncStore[ReadableIntoAsync[O]] | AsyncStore[ReadableInto[O]],
        requests: Iterable[tuple[str, O] | tuple[str, O, ByteRange | None]],
    ) -> AsyncIterator[tuple[tuple[str, O] | tuple[str, O, ByteRange | None], Copied | Missing]]:
        """As `Store.get_into_many`."""
        batch = list(requests)
        self._check_open()
        paths = [self._into_request(request) for request in batch]
        backend = self._backend
        if isinstance(backend, ReadableIntoManyAsync):
            many = cast("ReadableIntoManyAsync[O]", backend)
            return self._streamed(batch, many.get_into_many_async(paths, self._context))
        return self._streamed(batch, run_many_async(self._get_into, paths, self._limiter.limit))

    async def put_many(
        self: AsyncStore[WritableAsync] | AsyncStore[Writable],
        items: Iterable[tuple[str, Buffer]],
    ) -> list[tuple[str, Written]]:
        """As `Store.put_many`."""
        keys, requests = self._put_requests(items)
        backend = self._backend
        if isinstance(backend, WritableManyAsync):
            return await self._finished(keys, backend.put_many_async(requests, self._context))
        return await self._finished(keys, run_many_async(self._put, requests, self._limiter.limit))

    async def delete_many(
        self: AsyncStore[WritableAsync] | AsyncStore[Writable], keys: Iterable[str]
    ) -> list[tuple[str, Deleted | Missing]]:
        """As `Store.delete_many`."""
        batch = list(keys)
        paths = self._writable_paths(batch)
        backend = self._backend
        if isinstance(backend, WritableManyAsync):
            return await self._finished(batch, backend.delete_many_async(paths, self._context))
        requests = [(path,) for path in paths]
        return await self._finished(
            batch, run_many_async(self._delete, requests, self._limiter.limit)
        )

    # ----------------------------------------------------------------- listing

    async def _open_items(self, items: AsyncIterator[T]) -> AsyncIterator[T]:
        async for item in items:
            self._check_open()
            yield item

    async def _listed_in_thread(
        self, fn: Callable[[str], Iterator[T]], directory: str
    ) -> AsyncIterator[T]:
        """A sync listing, taken whole on the context's pool."""
        for item in await self._context.threads.run(_list_all, fn, directory):
            yield item

    def list(
        self: AsyncStore[ListableAsync] | AsyncStore[Listable], directory: str = ""
    ) -> AsyncIterator[str]:
        """As `Store.list`."""
        path = self._directory(directory)
        backend = self._backend
        if isinstance(backend, ListableAsync):
            paths = backend.list_async(path)
        else:
            paths = self._listed_in_thread(backend.list, path)
        return self._open_items(_relativized(paths, self._root))

    def list_dir(
        self: AsyncStore[ListableAsync] | AsyncStore[Listable], directory: str = ""
    ) -> AsyncIterator[ListItem]:
        """As `Store.list_dir`."""
        path = self._directory(directory)
        backend = self._backend
        if isinstance(backend, ListableAsync):
            return self._open_items(backend.list_dir_async(path))
        return self._open_items(self._listed_in_thread(backend.list_dir, path))


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
    cls: type[_View[Backend]], backend: Backend, root: str, read_only: bool, context: Context
) -> _View[Backend]:
    return cls(backend, root=root, read_only=read_only, context=context)
