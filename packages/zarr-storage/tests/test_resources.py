"""Contexts and their resources, alone and as stores draw on them."""

from __future__ import annotations

import asyncio
import pickle
import threading
import time
from functools import partial
from typing import TYPE_CHECKING

import pytest
from obstore.store import MemoryStore

from zarr_storage import Context, Found, Limiter, Missing, Store, ThreadPool
from zarr_storage._resources import default_limiter
from zarr_storage.backends import LocalBackend, MemoryBackend, ObjectStoreBackend

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from zarr_storage import ByteRange
    from zarr_storage.testing import FullBackend


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


def _hold_sync(limiter: Limiter, peaks: list[_Peak]) -> None:
    with limiter.slot():
        for peak in peaks:
            peak.enter()
        time.sleep(0.002)
        for peak in peaks:
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


def _join_all(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


# ---------------------------------------------------------------------------- Limiter


@pytest.mark.parametrize("limit", [1, 3])
def test_limiter_bounds_threads_and_tasks_together(limit: int) -> None:
    """Threads and tasks on two event loops never hold more than `limit` slots at once."""
    limiter, peak = Limiter(limit), _Peak()
    workers = [threading.Thread(target=_hold_sync, args=(limiter, [peak])) for _ in range(8)]
    workers += [threading.Thread(target=_run_loop, args=(limiter, peak, 8)) for _ in range(2)]
    _join_all(workers)
    assert (peak.done, peak.most, limiter.in_use) == (24, limit, 0)


def test_nested_limiter_holds_slots_of_both() -> None:
    """A limiter `within` another is bounded by its own limit, and shares the wider one."""
    wide = Limiter(3)
    narrow = Limiter(2, within=wide)
    narrow_peak, wide_peak = _Peak(), _Peak()
    workers = [
        threading.Thread(target=_hold_sync, args=(narrow, [narrow_peak, wide_peak]))
        for _ in range(8)
    ]
    workers += [threading.Thread(target=_hold_sync, args=(wide, [wide_peak])) for _ in range(8)]
    _join_all(workers)
    assert (narrow_peak.most, wide_peak.most, wide_peak.done) == (2, 3, 16)
    assert (narrow.in_use, wide.in_use) == (0, 0)


def _wait_for_slot(limiter: Limiter) -> None:
    with limiter.slot():
        pass


def test_waiting_on_a_narrow_limit_holds_none_of_the_wide_one() -> None:
    """Slots are taken innermost first, so a burst queued on a narrow limit leaves the wide one
    free for everyone else."""
    wide = Limiter(2)
    narrow = Limiter(1, within=wide)
    with narrow.slot():
        waiter = threading.Thread(target=_wait_for_slot, args=(narrow,))
        waiter.start()
        waiter.join(0.05)
        assert waiter.is_alive()
        assert wide.in_use == 1
    waiter.join(10)
    assert (narrow.in_use, wide.in_use) == (0, 0)


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


async def _cancel_between_slots() -> tuple[int, int]:
    wide = Limiter(1)
    narrow = Limiter(1, within=wide)
    await wide.acquire_async()
    entering = asyncio.create_task(_wait_for_slot_async(narrow))
    await asyncio.sleep(0)  # holds the narrow slot, waits for the wide one
    entering.cancel()
    with pytest.raises(asyncio.CancelledError):
        await entering
    wide.release()
    return narrow.in_use, wide.in_use


async def _wait_for_slot_async(limiter: Limiter) -> None:
    async with limiter.slot():
        pass


def test_cancelled_while_taking_a_nested_slot_gives_back_the_inner_one() -> None:
    assert asyncio.run(_cancel_between_slots()) == (0, 0)


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


@pytest.mark.parametrize(("size", "error"), [(0, ValueError), (2.0, TypeError)])
def test_thread_pool_of_a_bad_size_raises(size: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        ThreadPool(size)  # pyright: ignore[reportArgumentType]


def test_shutting_down_the_default_pool_raises() -> None:
    with pytest.raises(ValueError, match="shared"):
        ThreadPool.default().shutdown()


# ---------------------------------------------------------------------------- Context


def test_context_resolves_resources_through_its_parents() -> None:
    """Own resources first, then the parent's, then the process-wide defaults."""
    pool, a, b = ThreadPool(2), Limiter(4), Limiter(5)
    default = Context.default()
    assert default.threads is ThreadPool.default()
    assert default.limiter("x") is default_limiter("x")
    own = Context(threads=pool, limits={"a": a})
    assert (own.threads, own.limiter("a"), own.limiter("x")) == (pool, a, default_limiter("x"))
    child = own.child(limits={"b": b})
    assert (child.threads, child.limiter("a"), child.limiter("b")) == (pool, a, b)
    assert child.child(threads=ThreadPool.default()).threads is ThreadPool.default()
    narrowed = child.limited("a", 2).limiter("a")
    assert (narrowed.limit, narrowed.within) == (2, a)


@pytest.mark.parametrize(
    "make",
    [
        partial(Context, limits={"a": 4}),  # pyright: ignore[reportArgumentType]
        partial(Context, threads=4),  # pyright: ignore[reportArgumentType]
    ],
    ids=["limit-not-a-limiter", "threads-not-a-pool"],
)
def test_context_of_a_bad_resource_raises(make: Callable[[], Context]) -> None:
    with pytest.raises(TypeError):
        make()


def test_views_carry_their_context() -> None:
    """Views keep their context; `with_context` and `limited` change it and nothing else."""
    context = Context(limits={"memory": Limiter(8)})
    store = Store(MemoryBackend(), root="r", context=context)
    for view in (store / "a", store.with_read_only(), store.as_async(), store.as_async().as_sync()):
        assert view.context is context
    other = Context()
    assert store.with_context(other).context is other
    limited = store.limited(2)
    assert limited.context.parent is context
    assert limited.context.limiter("memory").limit == 2
    assert limited.context.limiter("memory").within is context.limiter("memory")
    assert (limited.root, limited.shares_backend(store)) == ("r", True)


def test_pickling_keeps_resources_shared(tmp_path: Path) -> None:
    """Stores that shared a context share one after a round trip; defaults stay defaults."""
    wide = Limiter(8)
    context = Context(threads=ThreadPool(2), limits={"local": Limiter(3, within=wide)})
    stores = (
        Store(LocalBackend(tmp_path / "a"), context=context),
        Store(LocalBackend(tmp_path / "b"), context=context),
        Store(LocalBackend(tmp_path / "c")),
    )
    a, b, c = pickle.loads(pickle.dumps(stores))
    assert a.context is b.context
    assert a.context is not context
    limiter = a.context.limiter("local")
    assert (a.context.threads.max_workers, limiter.limit, limiter.within.limit) == (2, 3, 8)
    assert c.context is Context.default()


# ---------------------------------------------------------------------------- stores under limits


def _get_in_thread(store: Store[FullBackend], results: list[object]) -> None:
    results.append(store.get("k"))


async def _get_async_while_held(store: Store[FullBackend]) -> bool:
    try:
        await asyncio.wait_for(store.as_async().get("k"), 0.05)
    except TimeoutError:
        return True
    return False


@pytest.mark.parametrize("backend", ["local", "obstore"])
def test_operations_wait_for_the_contexts_limiter(tmp_path: Path, backend: str) -> None:
    """Sync and async operations, on native-async and sync-only backends, wait for a slot."""
    raw: FullBackend = (
        LocalBackend(tmp_path) if backend == "local" else ObjectStoreBackend(MemoryStore())
    )
    pool, limiter = ThreadPool(2), Limiter(1)
    store = Store(raw, context=Context(threads=pool, limits={raw.resource_key: limiter}))
    store.put("k", b"x")
    limiter.acquire()  # some other operation under this context holds the only slot
    results: list[object] = []
    reader = threading.Thread(target=_get_in_thread, args=(store, results))
    reader.start()
    reader.join(0.05)
    assert reader.is_alive()
    assert asyncio.run(_get_async_while_held(store))
    limiter.release()
    reader.join(10)
    pool.shutdown()  # the timed-out async read's thread finishes too
    assert isinstance(results[0], Found)
    assert limiter.in_use == 0


class _PeakBackend(MemoryBackend):
    """A memory backend whose reads take a moment and record how many overlap."""

    def __init__(self) -> None:
        super().__init__()
        self.peak = _Peak()

    def get(self, path: str, byte_range: ByteRange | None) -> Found | Missing:
        self.peak.enter()
        time.sleep(0.005)
        self.peak.leave()
        return super().get(path, byte_range)


def test_limited_view_bounds_its_batches() -> None:
    backend = _PeakBackend()
    store = Store(backend, context=Context(threads=ThreadPool(8)))
    keys = [f"k{i}" for i in range(24)]
    store.put_many((key, b"x") for key in keys)
    assert len(list(store.limited(2).get_many(keys))) == 24
    assert backend.peak.most == 2
    backend.peak.most = 0
    assert len(list(store.get_many(keys))) == 24
    assert backend.peak.most > 2


class _RecordingBackend(LocalBackend):
    """A local backend that records which threads its reads run on."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
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
    backend = _RecordingBackend(tmp_path)
    store = Store(backend, context=Context(threads=pool))
    keys = [f"k{i}" for i in range(4)]
    store.put_many((key, b"x") for key in keys)
    backend.read_on.clear()
    try:
        worker = pool.submit(_batch_here, store, keys).result(timeout=10)
    finally:
        pool.shutdown(wait=False)
    assert backend.read_on == {worker}


def _read_batch(store: Store[LocalBackend], results: list[object]) -> None:
    results.extend(result for _, result in store.get_many(["k"]))


def _read_async(store: Store[LocalBackend], results: list[object]) -> None:
    results.append(asyncio.run(store.as_async().get("k")))


def _settle() -> None:
    time.sleep(0.05)


def test_async_read_of_a_sync_backend_holds_no_slot_while_it_waits_for_a_thread(
    tmp_path: Path,
) -> None:
    """One thread, one slot. A sync batch's read is queued on the thread, then an async read of the
    sync backend is queued behind it. Had the async read taken the slot before its turn on the
    thread, the batch's read, which runs first, would wait for that slot forever."""
    pool, limiter = ThreadPool(1), Limiter(1)
    store = Store(LocalBackend(tmp_path), context=Context(threads=pool, limits={"local": limiter}))
    store.put("k", b"x")
    gate = threading.Event()
    pool.submit(gate.wait)  # keeps the only thread busy while the reads queue up
    batch: list[object] = []
    single: list[object] = []
    batch_reader = threading.Thread(target=_read_batch, args=(store, batch), daemon=True)
    async_reader = threading.Thread(target=_read_async, args=(store, single), daemon=True)
    batch_reader.start()
    _settle()  # the batch's read is queued
    async_reader.start()
    _settle()  # the async read is queued behind it
    gate.set()
    batch_reader.join(10)
    async_reader.join(10)
    assert not batch_reader.is_alive()
    assert not async_reader.is_alive()
    assert [type(r) for r in batch + single] == [Found, Found]
    assert limiter.in_use == 0
