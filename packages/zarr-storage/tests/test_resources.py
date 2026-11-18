"""Shared resources: thread pools and limiters, alone and as backends use them."""

from __future__ import annotations

import asyncio
import pickle
import threading
import time
from functools import partial
from typing import TYPE_CHECKING

import pytest
from obstore.store import LocalStore, MemoryStore

from zarr_storage import Copied, Deleted, Found, Limiter, Missing, Store, ThreadPool
from zarr_storage.backends import LocalBackend, ObjectStoreBackend

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from zarr_storage import ByteRange


class _Peak:
    """Counts callers inside a region, and the most there ever were at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inside = 0
        self.most = 0
        self.done = 0

    def enter(self) -> None:
        with self._lock:
            self._inside += 1
            self.most = max(self.most, self._inside)

    def leave(self) -> None:
        with self._lock:
            self._inside -= 1
            self.done += 1


def _hold_sync(limiter: Limiter, peak: _Peak) -> None:
    with limiter.slot():
        peak.enter()
        time.sleep(0.002)
        peak.leave()


async def _hold_async(limiter: Limiter, peak: _Peak) -> None:
    async with limiter.slot():
        peak.enter()
        await asyncio.sleep(0.002)
        peak.leave()


async def _many_async(limiter: Limiter, peak: _Peak, n: int) -> None:
    await asyncio.gather(*(_hold_async(limiter, peak) for _ in range(n)))


def _run_loop(limiter: Limiter, peak: _Peak, n: int) -> None:
    asyncio.run(_many_async(limiter, peak, n))


def _finishes(fn: Callable[[], object]) -> bool:
    """Whether `fn()` returns within ten seconds; a deadlock fails a test instead of hanging it."""
    worker = threading.Thread(target=fn, daemon=True)
    worker.start()
    worker.join(10)
    return not worker.is_alive()


# ---------------------------------------------------------------------------- Limiter


@pytest.mark.parametrize("limit", [1, 3])
def test_limiter_bounds_threads_and_tasks_together(limit: int) -> None:
    """Threads and tasks on two event loops never hold more than `limit` slots at once."""
    limiter, peak = Limiter(limit), _Peak()
    workers = [threading.Thread(target=_hold_sync, args=(limiter, peak)) for _ in range(8)]
    workers += [threading.Thread(target=_run_loop, args=(limiter, peak, 8)) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert (peak.done, peak.most, limiter.in_use) == (24, limit, 0)


async def _nested_async(limiter: Limiter) -> int:
    async with limiter.slot(), limiter.slot():
        return limiter.in_use


def _nested_sync(limiter: Limiter, seen: list[int]) -> None:
    with limiter.slot(), limiter.slot():
        seen.append(limiter.in_use)
    seen.append(asyncio.run(_nested_async(limiter)))


def test_slot_inside_a_slot_takes_no_second_slot() -> None:
    limiter, seen = Limiter(1), []
    assert _finishes(partial(_nested_sync, limiter, seen))
    assert (seen, limiter.in_use) == ([1, 1], 0)


async def _take_in_order(limiter: Limiter, name: str, order: list[str]) -> None:
    async with limiter.slot():
        order.append(name)


async def _first_come_first_served() -> list[str]:
    limiter, order = Limiter(1), []
    await limiter.acquire_async()
    tasks = [asyncio.create_task(_take_in_order(limiter, name, order)) for name in "abc"]
    await asyncio.sleep(0)
    limiter.release()
    await asyncio.gather(*tasks)
    return order


def test_limiter_serves_waiters_in_order() -> None:
    assert asyncio.run(_first_come_first_served()) == ["a", "b", "c"]


async def _cancel_while_waiting(granted_first: bool) -> int:
    limiter = Limiter(1)
    await limiter.acquire_async()
    waiting = asyncio.create_task(limiter.acquire_async())
    await asyncio.sleep(0)
    if granted_first:
        limiter.release()  # the slot passes to the waiter, which is then cancelled
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    if not granted_first:
        limiter.release()
    return limiter.in_use


@pytest.mark.parametrize("granted_first", [False, True])
def test_cancelled_waiter_gives_back_its_slot(granted_first: bool) -> None:
    assert asyncio.run(_cancel_while_waiting(granted_first)) == 0


@pytest.mark.parametrize(("limit", "error"), [(0, ValueError), (1.5, TypeError), (True, TypeError)])
def test_limiter_of_a_bad_limit_raises(limit: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        Limiter(limit)  # pyright: ignore[reportArgumentType]


def test_releasing_a_slot_not_held_raises() -> None:
    with pytest.raises(RuntimeError, match="not held"):
        Limiter(1).release()


# ---------------------------------------------------------------------------- ThreadPool


def _worker_view(pool: ThreadPool, other: ThreadPool) -> tuple[bool, bool]:
    return pool.in_worker(), other.in_worker()


def test_thread_pool_knows_its_own_threads() -> None:
    pool, other = ThreadPool(1), ThreadPool(1)
    try:
        assert not pool.in_worker()
        assert pool.submit(_worker_view, pool, other).result() == (True, False)
    finally:
        pool.shutdown()
        other.shutdown()


class _RecordingBackend(LocalBackend):
    """A local backend that records which threads its reads run on."""

    def __init__(self, root: Path, threads: ThreadPool) -> None:
        super().__init__(root, threads=threads)
        self.read_on: set[int] = set()

    def get(self, path: str, byte_range: ByteRange | None) -> Found | Missing:
        self.read_on.add(threading.get_ident())
        return super().get(path, byte_range)


def _batch_here(store: Store[_RecordingBackend], keys: list[str]) -> int:
    assert all(isinstance(result, Found) for _, result in store.get_many(keys))
    return threading.get_ident()


def test_batch_from_a_thread_of_its_own_pool_runs_on_that_thread(tmp_path: Path) -> None:
    """A batch started on one of its pool's threads must not wait for the pool's other threads,
    which may all be waiting the same way; it runs its reads on the thread that started it."""
    pool = ThreadPool(2)
    backend = _RecordingBackend(tmp_path, pool)
    store = Store(backend)
    keys = [f"k{i}" for i in range(4)]
    store.put_many((key, b"x") for key in keys)
    try:
        worker = pool.submit(_batch_here, store, keys).result(timeout=10)
    finally:
        pool.shutdown(wait=False)
    assert backend.read_on == {worker}


@pytest.mark.parametrize(("size", "error"), [(0, ValueError), (2.0, TypeError)])
def test_thread_pool_of_a_bad_size_raises(size: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        ThreadPool(size)  # pyright: ignore[reportArgumentType]


def test_shutting_down_the_default_pool_raises() -> None:
    with pytest.raises(ValueError, match="shared"):
        ThreadPool.default().shutdown()


# ---------------------------------------------------------------------------- sharing


def test_backends_share_resources_as_given(tmp_path: Path) -> None:
    """Left out, resources are process-wide defaults; given, they are exactly what was given."""
    assert LocalBackend(tmp_path / "a").threads is ThreadPool.default()
    pool = ThreadPool(2)
    assert LocalBackend(tmp_path / "b", threads=pool).threads is pool
    memory = [ObjectStoreBackend(MemoryStore()).concurrency for _ in range(2)]
    local = ObjectStoreBackend(LocalStore(str(tmp_path))).concurrency
    assert memory[0] is memory[1]
    assert local is not memory[0]
    limiter = Limiter(3)
    assert ObjectStoreBackend(MemoryStore(), concurrency=limiter).concurrency is limiter


def test_pickling_keeps_resources_shared(tmp_path: Path) -> None:
    """Backends that shared a resource share one after a round trip; defaults stay defaults."""
    pool, limiter = ThreadPool(2), Limiter(3)
    objects = str(tmp_path)  # obstore's `MemoryStore` can't be pickled; its `LocalStore` can
    backends = (
        LocalBackend(tmp_path / "a", threads=pool),
        LocalBackend(tmp_path / "b", threads=pool),
        ObjectStoreBackend(LocalStore(objects), concurrency=limiter),
        ObjectStoreBackend(LocalStore(objects), concurrency=limiter),
        LocalBackend(tmp_path / "c"),
        ObjectStoreBackend(LocalStore(objects)),
    )
    a, b, c, d, e, f = pickle.loads(pickle.dumps(backends))
    assert a.threads is b.threads
    assert a.threads is not pool
    assert a.threads.max_workers == 2
    assert c.concurrency is d.concurrency
    assert c.concurrency is not limiter
    assert c.concurrency.limit == 3
    assert e.threads is ThreadPool.default()
    assert f.concurrency is backends[5].concurrency


def _get_in_thread(store: Store[ObjectStoreBackend], results: list[object]) -> None:
    results.append(store.get("k"))


async def _get_async_while_held(store: Store[ObjectStoreBackend]) -> bool:
    try:
        await asyncio.wait_for(store.as_async().get("k"), 0.05)
    except TimeoutError:
        return True
    return False


def test_obstore_operations_wait_for_a_shared_limit() -> None:
    """Every operation, sync or async, of every backend sharing a limiter waits for a slot."""
    limiter = Limiter(1)
    first = Store(ObjectStoreBackend(MemoryStore(), concurrency=limiter))
    second = Store(ObjectStoreBackend(MemoryStore(), concurrency=limiter))
    second.put("k", b"x")
    limiter.acquire()  # another backend's operation holds the only slot
    results: list[object] = []
    reader = threading.Thread(target=_get_in_thread, args=(second, results))
    reader.start()
    reader.join(0.05)
    assert reader.is_alive()
    assert asyncio.run(_get_async_while_held(first))
    limiter.release()
    reader.join(10)
    assert isinstance(results[0], Found)
    assert limiter.in_use == 0


def _nested_operations(store: Store[ObjectStoreBackend], results: list[object]) -> None:
    store.put("k", b"xy")
    results.append(store.get_into("k", bytearray(2)))
    results.append(store.delete("k"))


def test_obstore_operation_that_calls_another_takes_one_slot() -> None:
    """`get_into` reads through `get` and `delete` looks with `head`: with one slot, neither waits."""
    store = Store(ObjectStoreBackend(MemoryStore(), concurrency=Limiter(1)))
    results: list[object] = []
    assert _finishes(partial(_nested_operations, store, results))
    assert [type(r) for r in results] == [Copied, Deleted]
