"""The runners behind every batch: windows, ordering and the failure rule."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING

import pytest

from zarr_storage import BatchError
from zarr_storage._batch import run_many, run_many_async
from zarr_storage._resources import ThreadPool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _Tracker:
    """Squares its argument, recording what started and the most calls ever running at once."""

    def __init__(self, fails: int | None = None) -> None:
        self._fails = fails
        self._lock = threading.Lock()
        self._running = 0
        self.most = 0
        self.started: set[int] = set()

    def _enter(self, x: int) -> None:
        with self._lock:
            self.started.add(x)
            self._running += 1
            self.most = max(self.most, self._running)

    def _exit(self) -> None:
        with self._lock:
            self._running -= 1

    def __call__(self, x: int) -> int:
        self._enter(x)
        try:
            if x == self._fails:
                raise ValueError(x)
            time.sleep(0.01)
            return x * x
        finally:
            self._exit()

    async def call_async(self, x: int) -> int:
        self._enter(x)
        try:
            if x == self._fails:
                raise ValueError(x)
            await asyncio.sleep(0.01)
            return x * x
        finally:
            self._exit()


def _run(tracker: _Tracker, mode: str, window: int, n: int, into: list[tuple[int, int]]) -> None:
    """Run `tracker` over `range(n)`, appending each result to `into` as it is yielded."""
    requests = [(x,) for x in range(n)]
    if mode == "serial":
        into.extend(run_many(tracker, requests))
    elif mode == "threads":
        pool = ThreadPool(8)
        try:
            into.extend(run_many(tracker, requests, pool, window))
        finally:
            pool.shutdown()
    else:
        asyncio.run(_collect(run_many_async(tracker.call_async, requests, window), into))


async def _collect(results: AsyncIterator[tuple[int, int]], into: list[tuple[int, int]]) -> None:
    async for pair in results:
        into.append(pair)  # noqa: PERF401 -- a failure keeps the pairs before it


@pytest.mark.parametrize(
    ("mode", "window"),
    [("serial", 1), ("threads", 1), ("threads", 3), ("async", 1), ("async", 3)],
)
@pytest.mark.parametrize("n", [0, 1, 7])
def test_runner(mode: str, window: int, n: int) -> None:
    """Every request runs once, its result keyed by its index, never more than `window` at once."""
    tracker = _Tracker()
    results: list[tuple[int, int]] = []
    _run(tracker, mode, window, n, results)
    assert sorted(results) == [(x, x * x) for x in range(n)]
    assert tracker.most <= window


@pytest.mark.parametrize(("mode", "window"), [("serial", 1), ("threads", 3), ("async", 3)])
def test_runner_stops_starting_after_a_failure(mode: str, window: int) -> None:
    """The failure's peers already running finish and are yielded; nothing new starts."""
    tracker = _Tracker(fails=0)
    results: list[tuple[int, int]] = []
    with pytest.raises(ExceptionGroup) as caught:
        _run(tracker, mode, window, 10, results)
    assert [type(e) for e in caught.value.exceptions] == [ValueError]
    assert tracker.started == set(range(window))
    assert sorted(results) == [(x, x * x) for x in range(1, window)]


def test_batch_error_keeps_its_report_when_split() -> None:
    """`except*` splits a group; each part is still a `BatchError` with the batch's report."""
    error = BatchError("x", [ValueError(), KeyError()], completed=(("a", 1),), remaining=("b",))
    values, rest = error.split(ValueError)
    for part, kind in [(values, ValueError), (rest, KeyError)]:
        assert isinstance(part, BatchError)
        assert [type(e) for e in part.exceptions] == [kind]
        assert (part.completed, part.remaining) == ((("a", 1),), ("b",))
