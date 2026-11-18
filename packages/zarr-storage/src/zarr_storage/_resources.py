"""The resources a store's operations draw on: thread pools, limits on operations in flight, and the
context that holds them.

Modeled on TensorStore's `Context`: a context maps resource keys to resource objects, a store view
carries one, and every operation through the view takes its resources from it. A backend names the
key its operations are limited under (`Backend.resource_key`) and never holds resources itself.
Resources change how fast operations run, never what they do, so a context is not part of a store's
identity.

A context left out is `Context.default()`, which resolves every key to a process-wide default, as
TensorStore's thread pools and request limits are shared across independent contexts by default.

Pickling a resource pickles its settings. Pickle stores each object once, so stores that shared a
resource before pickling share one new resource after; a default unpickles as the receiving
process's default.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeVar, TypeVarTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from concurrent.futures import Future
    from types import TracebackType

__all__ = [
    "Context",
    "Limiter",
    "Slot",
    "ThreadPool",
    "default_context",
    "default_limiter",
    "default_thread_pool",
]

R = TypeVar("R")
Ts = TypeVarTuple("Ts")

DEFAULT_THREADS: Final = max(4, os.cpu_count() or 1)
"""Threads in the default pool, as TensorStore's `file_io_concurrency` default."""

DEFAULT_LIMIT: Final = 32
"""Operations in flight under a default limiter, as TensorStore's `s3_request_concurrency`."""


def _check_size(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int; got {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} must be >= 1; got {value}.")


# ---------------------------------------------------------------------------- thread pools

_worker = threading.local()
"""Per thread: the `ThreadPool` the thread works for, if any."""


def _mark_worker(pool: ThreadPool) -> None:
    _worker.pool = pool


class ThreadPool:
    """Threads that run blocking operations, such as file I/O.

    The threads start when first needed. Work runs in a copy of the submitter's `contextvars`
    context, as `asyncio.to_thread` does. Whoever creates a pool owns it and may `shutdown` it once
    nothing needs it; the default pool lives as long as the process.

    Parameters
    ----------
    max_workers
        The most threads the pool runs at once.
    """

    def __init__(self, max_workers: int) -> None:
        _check_size("max_workers", max_workers)
        self._max_workers = max_workers
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None

    @classmethod
    def default(cls) -> ThreadPool:
        """The process-wide pool, of `DEFAULT_THREADS` threads."""
        return default_thread_pool()

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def __repr__(self) -> str:
        return f"{type(self).__name__}(max_workers={self._max_workers})"

    def __reduce__(self) -> tuple[object, ...]:
        return (ThreadPool, (self._max_workers,))

    def _pool(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    self._max_workers,
                    thread_name_prefix="zarr-storage",
                    initializer=_mark_worker,
                    initargs=(self,),
                )
            return self._executor

    def in_worker(self) -> bool:
        """Whether the calling thread is one of this pool's.

        Work started from one of the pool's own threads must not wait for other work queued on the
        pool, which may never start; stores run such work inline.
        """
        return getattr(_worker, "pool", None) is self

    def submit(self, fn: Callable[[*Ts], R], /, *args: *Ts) -> Future[R]:
        """Run `fn(*args)` on one of the pool's threads."""
        context = contextvars.copy_context()
        return self._pool().submit(context.run, fn, *args)

    async def run(self, fn: Callable[[*Ts], R], /, *args: *Ts) -> R:
        """Await `fn(*args)`, run on one of the pool's threads."""
        context = contextvars.copy_context()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool(), partial(context.run, fn, *args))

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop the pool's threads; with `wait`, after the work already given them.

        The pool starts new threads if it is used again. Raises `ValueError` for the default pool,
        which other code may be using.
        """
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=wait)


class _DefaultThreadPool(ThreadPool):
    """The process-wide pool: never shut down, and unpickled as the receiver's own default."""

    def __reduce__(self) -> tuple[object, ...]:
        return (default_thread_pool, ())

    def shutdown(self, *, wait: bool = True) -> None:
        raise ValueError("The default thread pool is shared by the whole process.")


_defaults_lock = threading.Lock()
_default_pool: list[ThreadPool] = []


def default_thread_pool() -> ThreadPool:
    """The process-wide `ThreadPool`, of `DEFAULT_THREADS` threads."""
    with _defaults_lock:
        if not _default_pool:
            _default_pool.append(_DefaultThreadPool(DEFAULT_THREADS))
        return _default_pool[0]


# ---------------------------------------------------------------------------- limiters


class _ThreadWaiter:
    """A thread blocked on a limiter."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def grant(self) -> bool:
        self._event.set()
        return True

    def wait(self) -> None:
        self._event.wait()


def _resolve(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


class _TaskWaiter:
    """A task blocked on a limiter, on its own event loop. Fields change under the limiter's lock."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.future: asyncio.Future[None] = loop.create_future()
        self.granted = False
        self.cancelled = False

    def grant(self) -> bool:
        if self.cancelled:
            return False
        try:
            self.loop.call_soon_threadsafe(_resolve, self.future)
        except RuntimeError:  # the loop is closed; nothing waits any more
            return False
        self.granted = True
        return True


def _limiter(limit: int, within: Limiter | None) -> Limiter:
    return Limiter(limit, within=within)


class Limiter:
    """A limit on how many operations run at once.

    Threads and tasks on any event loop wait in one first-come, first-served queue. A limiter
    `within` another takes a slot of that one too, so a scoped limit can't exceed the limit it
    narrows.

    Parameters
    ----------
    limit
        The most operations that run at once.
    within
        A limiter whose slots this one's operations also hold.
    """

    def __init__(self, limit: int, *, within: Limiter | None = None) -> None:
        _check_size("limit", limit)
        self._limit = limit
        self._within = within
        self._lock = threading.Lock()
        self._in_use = 0
        self._waiters: deque[_ThreadWaiter | _TaskWaiter] = deque()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def within(self) -> Limiter | None:
        """The limiter whose slots this one's operations also hold, if any."""
        return self._within

    @property
    def in_use(self) -> int:
        """How many slots are held now."""
        return self._in_use

    def __repr__(self) -> str:
        return f"{type(self).__name__}(limit={self._limit}, in_use={self._in_use})"

    def __reduce__(self) -> tuple[object, ...]:
        return (_limiter, (self._limit, self._within))

    def chain(self) -> tuple[Limiter, ...]:
        """This limiter and each it lies within, innermost first."""
        chain: list[Limiter] = []
        current: Limiter | None = self
        while current is not None:
            chain.append(current)
            current = current.within
        return tuple(chain)

    def slot(self) -> Slot:
        """A context manager, sync (`with`) or async (`async with`), holding a slot of this limiter
        and of each limiter it lies within.

        Slots are taken innermost first, so an operation waiting on a narrow limit holds none of the
        wider one it narrows. A slot is not re-entrant: code holding one that starts work waiting
        for another slot of the same limiter can wait forever, as with a lock. Stores take one slot
        per operation; to bound a burst of operations, use a limited view (`Store.limited`).
        """
        return Slot(self)

    def _take(self) -> bool:
        """Take a free slot if no one is waiting; called under the lock."""
        if self._in_use < self._limit and not self._waiters:
            self._in_use += 1
            return True
        return False

    def acquire(self) -> None:
        """Take a slot of this limiter alone, blocking the calling thread until one is free."""
        with self._lock:
            if self._take():
                return
            waiter = _ThreadWaiter()
            self._waiters.append(waiter)
        waiter.wait()

    async def acquire_async(self) -> None:
        """Take a slot of this limiter alone, waiting without blocking the event loop."""
        with self._lock:
            if self._take():
                return
            waiter = _TaskWaiter(asyncio.get_running_loop())
            self._waiters.append(waiter)
        try:
            await waiter.future
        except asyncio.CancelledError:
            with self._lock:
                granted = waiter.granted
                if not granted:
                    waiter.cancelled = True
                    self._waiters.remove(waiter)
            if granted:
                self.release()
            raise

    def release(self) -> None:
        """Give back a slot of this limiter, to the longest waiter if there is one.

        Raises `RuntimeError` if no slot is held.
        """
        with self._lock:
            while self._waiters:
                if self._waiters.popleft().grant():
                    return  # the slot passes to the waiter
            if self._in_use == 0:
                raise RuntimeError("Released a limiter slot that was not held.")
            self._in_use -= 1


def _release_all(limiters: list[Limiter]) -> None:
    for limiter in reversed(limiters):
        limiter.release()


class Slot:
    """One use of a `Limiter`: holds a slot of it and of each limiter it lies within, innermost
    first, from entry to exit."""

    __slots__ = ("_chain", "_taken")

    def __init__(self, limiter: Limiter) -> None:
        self._chain = limiter.chain()
        self._taken: list[Limiter] = []

    def __enter__(self) -> None:
        taken: list[Limiter] = []
        try:
            for limiter in self._chain:
                limiter.acquire()
                taken.append(limiter)
        except BaseException:
            _release_all(taken)
            raise
        self._taken = taken

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        taken, self._taken = self._taken, []
        _release_all(taken)

    async def __aenter__(self) -> None:
        taken: list[Limiter] = []
        try:
            for limiter in self._chain:
                await limiter.acquire_async()
                taken.append(limiter)
        except BaseException:
            _release_all(taken)
            raise
        self._taken = taken

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        taken, self._taken = self._taken, []
        _release_all(taken)


class _DefaultLimiter(Limiter):
    """A process-wide limiter for one key; unpickled as the receiver's limiter for that key."""

    def __init__(self, key: str) -> None:
        super().__init__(DEFAULT_LIMIT)
        self._key = key

    def __reduce__(self) -> tuple[object, ...]:
        return (default_limiter, (self._key,))


_default_limiters: dict[str, Limiter] = {}


def default_limiter(key: str) -> Limiter:
    """The process-wide `Limiter` for `key`, of `DEFAULT_LIMIT`."""
    with _defaults_lock:
        limiter = _default_limiters.get(key)
        if limiter is None:
            limiter = _default_limiters[key] = _DefaultLimiter(key)
        return limiter


# ---------------------------------------------------------------------------- contexts


def _context(
    threads: ThreadPool | None, limits: Mapping[str, Limiter], parent: Context | None
) -> Context:
    return Context(threads=threads, limits=limits, parent=parent)


class Context:
    """The resources operations draw on: a thread pool, and a limiter per resource key.

    A context is immutable. `child` makes one that overrides some resources and takes the rest from
    its parent; a context with no parent takes the rest from the process-wide defaults. Every store
    view carries a context (`Store.context`); `with_context` and `limited` return views under
    another one.

    Parameters
    ----------
    threads
        The pool that runs blocking work: a sync backend's operations called through an
        `AsyncStore`, and the operations of batches. Left out, the parent's.
    limits
        A `Limiter` for each resource key it names (a backend's `resource_key`). Keys left out
        resolve through the parent.
    parent
        The context to take unnamed resources from. Left out, the process-wide defaults.
    """

    __slots__ = ("_limits", "_parent", "_threads")

    def __init__(
        self,
        *,
        threads: ThreadPool | None = None,
        limits: Mapping[str, Limiter] | None = None,
        parent: Context | None = None,
    ) -> None:
        own = {} if limits is None else dict(limits)
        for key, limiter in own.items():
            if not isinstance(limiter, Limiter):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(f"limits[{key!r}] must be a Limiter; got {type(limiter).__name__}.")
        if threads is not None and not isinstance(threads, ThreadPool):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"threads must be a ThreadPool; got {type(threads).__name__}.")
        self._threads = threads
        self._limits: Mapping[str, Limiter] = MappingProxyType(own)
        self._parent = parent

    @classmethod
    def default(cls) -> Context:
        """The process-wide context: every key resolves to its process-wide default."""
        return default_context()

    @property
    def parent(self) -> Context | None:
        return self._parent

    @property
    def threads(self) -> ThreadPool:
        """The pool that runs blocking work under this context."""
        context: Context | None = self
        while context is not None:
            if context._threads is not None:
                return context._threads
            context = context._parent
        return default_thread_pool()

    def limiter(self, key: str) -> Limiter:
        """The limiter for resource key `key` under this context."""
        context: Context | None = self
        while context is not None:
            limiter = context._limits.get(key)
            if limiter is not None:
                return limiter
            context = context._parent
        return default_limiter(key)

    def child(
        self, *, threads: ThreadPool | None = None, limits: Mapping[str, Limiter] | None = None
    ) -> Context:
        """A context with these resources, taking the rest from this one."""
        return Context(threads=threads, limits=limits, parent=self)

    def limited(self, key: str, limit: int) -> Context:
        """A child whose limiter for `key` allows `limit` operations, within this context's."""
        return self.child(limits={key: Limiter(limit, within=self.limiter(key))})

    def __repr__(self) -> str:
        own = ", ".join(f"{key!r}: {limiter!r}" for key, limiter in self._limits.items())
        return f"{type(self).__name__}(threads={self._threads!r}, limits={{{own}}})"

    def __reduce__(self) -> tuple[object, ...]:
        return (_context, (self._threads, dict(self._limits), self._parent))


class _DefaultContext(Context):
    """The process-wide context; unpickled as the receiver's own."""

    __slots__ = ()

    def __reduce__(self) -> tuple[object, ...]:
        return (default_context, ())


_default_context: list[Context] = []


def default_context() -> Context:
    """The process-wide `Context`."""
    with _defaults_lock:
        if not _default_context:
            _default_context.append(_DefaultContext())
        return _default_context[0]
