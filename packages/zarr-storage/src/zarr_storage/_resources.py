"""Resources that backends share: thread pools and limits on operations in flight.

Modeled on TensorStore's context resources (`file_io_concurrency`, `s3_request_concurrency`): a
resource is an object you create and pass to as many backends as should share it. A backend never
closes a resource it was given. Left out, a backend uses a process-wide default, as TensorStore
does, because threads and request slots are process-wide in the end; pass your own to isolate a
backend from the rest of the process. A resource changes how fast operations run, never what they
do.

Pickling a resource pickles its settings. Pickle stores each object once, so backends that shared
a resource before pickling share one new resource after; a default resource unpickles as the
default of the receiving process.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from functools import partial
from typing import TYPE_CHECKING, Final, TypeVar, TypeVarTuple

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future
    from contextvars import Token
    from types import TracebackType

__all__ = ["Limiter", "Slot", "ThreadPool", "default_limiter", "default_thread_pool"]

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
    """Threads that run blocking operations, such as file I/O, for the backends given it.

    The threads start when first needed. Whoever creates a pool owns it and may `shutdown` it once
    no backend needs it; the default pool lives as long as the process.

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
        """The process-wide pool that backends use when given none."""
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
        pool, which may never start; the batch runners run such work inline.
        """
        return getattr(_worker, "pool", None) is self

    def submit(self, fn: Callable[[*Ts], R], /, *args: *Ts) -> Future[R]:
        """Run `fn(*args)` on one of the pool's threads."""
        return self._pool().submit(fn, *args)

    async def run(self, fn: Callable[[*Ts], R], /, *args: *Ts) -> R:
        """Await `fn(*args)`, run on one of the pool's threads."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool(), partial(fn, *args))

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


_default_pool_lock = threading.Lock()
_default_pool: list[ThreadPool] = []


def default_thread_pool() -> ThreadPool:
    """The process-wide `ThreadPool`, with `DEFAULT_THREADS` threads."""
    with _default_pool_lock:
        if not _default_pool:
            _default_pool.append(_DefaultThreadPool(DEFAULT_THREADS))
        return _default_pool[0]


# ---------------------------------------------------------------------------- limiters

_held: ContextVar[frozenset[Limiter]] = ContextVar(
    "zarr_storage_held_limiters", default=frozenset()
)
"""The limiters whose slots the current operation holds, so nested operations don't wait again."""


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


class Limiter:
    """A limit on how many operations run at once, across every backend given it.

    Threads and tasks on any event loop wait in one first-come, first-served queue.

    Parameters
    ----------
    limit
        The most operations that run at once.
    """

    def __init__(self, limit: int) -> None:
        _check_size("limit", limit)
        self._limit = limit
        self._lock = threading.Lock()
        self._in_use = 0
        self._waiters: deque[_ThreadWaiter | _TaskWaiter] = deque()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_use(self) -> int:
        """How many slots are held now."""
        return self._in_use

    def __repr__(self) -> str:
        return f"{type(self).__name__}(limit={self._limit}, in_use={self._in_use})"

    def __reduce__(self) -> tuple[object, ...]:
        return (Limiter, (self._limit,))

    def slot(self) -> Slot:
        """A context manager, sync (`with`) or async (`async with`), that holds one slot.

        Prefer it to `acquire` and `release`: it releases on error, and inside an operation that
        already holds a slot of this limiter it takes no second one, so an operation that calls
        another can't wait on itself.
        """
        return Slot(self)

    def _take(self) -> bool:
        """Take a free slot if no one is waiting; called under the lock."""
        if self._in_use < self._limit and not self._waiters:
            self._in_use += 1
            return True
        return False

    def acquire(self) -> None:
        """Take a slot, blocking the calling thread until one is free."""
        with self._lock:
            if self._take():
                return
            waiter = _ThreadWaiter()
            self._waiters.append(waiter)
        waiter.wait()

    async def acquire_async(self) -> None:
        """Take a slot, waiting without blocking the event loop until one is free."""
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
        """Give back a slot, to the longest waiter if there is one.

        Raises `RuntimeError` if no slot is held.
        """
        with self._lock:
            while self._waiters:
                if self._waiters.popleft().grant():
                    return  # the slot passes to the waiter
            if self._in_use == 0:
                raise RuntimeError("Released a limiter slot that was not held.")
            self._in_use -= 1


class Slot:
    """One use of a `Limiter`: holds a slot from entry to exit, unless the operation holds one."""

    __slots__ = ("_limiter", "_token")

    def __init__(self, limiter: Limiter) -> None:
        self._limiter = limiter
        self._token: Token[frozenset[Limiter]] | None = None

    def _needed(self) -> bool:
        return self._limiter not in _held.get()

    def _mark(self) -> None:
        self._token = _held.set(_held.get() | {self._limiter})

    def _unmark(self) -> None:
        if self._token is None:
            return
        _held.reset(self._token)
        self._token = None
        self._limiter.release()

    def __enter__(self) -> None:
        if self._needed():
            self._limiter.acquire()
            self._mark()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._unmark()

    async def __aenter__(self) -> None:
        if self._needed():
            await self._limiter.acquire_async()
            self._mark()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._unmark()


class _DefaultLimiter(Limiter):
    """A process-wide limiter for one key; unpickled as the receiver's limiter for that key."""

    def __init__(self, key: str) -> None:
        super().__init__(DEFAULT_LIMIT)
        self._key = key

    def __reduce__(self) -> tuple[object, ...]:
        return (default_limiter, (self._key,))


_default_limiters_lock = threading.Lock()
_default_limiters: dict[str, Limiter] = {}


def default_limiter(key: str) -> Limiter:
    """The process-wide `Limiter` for `key` (such as an obstore store class), of `DEFAULT_LIMIT`."""
    with _default_limiters_lock:
        limiter = _default_limiters.get(key)
        if limiter is None:
            limiter = _default_limiters[key] = _DefaultLimiter(key)
        return limiter
