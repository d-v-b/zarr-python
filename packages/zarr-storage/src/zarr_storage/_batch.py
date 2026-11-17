"""Running one operation over many requests, for `Store`'s fallbacks and backends' batch methods.

Both runners yield `(index, result)` pairs as operations finish, so callers map results back to
their requests. A runner keeps at most `window` operations in flight. After the first failure it
starts no new operations, lets the running ones finish (yielding those that succeed), then raises
an `ExceptionGroup` of every failure. `Store` turns that group into a `BatchError`.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import FIRST_COMPLETED, wait
from typing import TYPE_CHECKING, Any, TypeVar, TypeVarTuple, Unpack

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Sequence
    from concurrent.futures import Executor, Future

__all__ = ["run_many", "run_many_async"]

R = TypeVar("R")
Ts = TypeVarTuple("Ts")

_MESSAGE = "Operations of a batch failed."


def run_many(
    fn: Callable[[Unpack[Ts]], R],
    requests: Sequence[tuple[*Ts]],
    executor: Executor | None = None,
    window: int = 1,
) -> Iterator[tuple[int, R]]:
    """Call `fn(*request)` for each request: one at a time, or on `executor`, `window` at once."""
    if executor is None:
        return _run_serially(fn, requests)
    return _run_on(executor, fn, requests, window)


def _run_serially(
    fn: Callable[[Unpack[Ts]], R], requests: Sequence[tuple[*Ts]]
) -> Iterator[tuple[int, R]]:
    for index, request in enumerate(requests):
        try:
            result = fn(*request)
        except Exception as error:  # noqa: BLE001 -- every failure is reported, in the group
            raise ExceptionGroup(_MESSAGE, [error]) from None
        yield index, result


def _run_on(
    executor: Executor,
    fn: Callable[[Unpack[Ts]], R],
    requests: Sequence[tuple[*Ts]],
    window: int,
) -> Iterator[tuple[int, R]]:
    pending: dict[Future[R], int] = {}
    errors: list[Exception] = []
    started = 0
    try:
        while pending or (not errors and started < len(requests)):
            while not errors and started < len(requests) and len(pending) < window:
                pending[executor.submit(fn, *requests[started])] = started
                started += 1
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                error = future.exception()
                if error is None:
                    yield index, future.result()
                elif isinstance(error, Exception):
                    errors.append(error)
                else:
                    raise error
    finally:
        for future in pending:
            future.cancel()
    if errors:
        raise ExceptionGroup(_MESSAGE, errors)


async def run_many_async(
    fn: Callable[[Unpack[Ts]], Coroutine[Any, Any, R]],
    requests: Sequence[tuple[*Ts]],
    window: int = 1,
) -> AsyncIterator[tuple[int, R]]:
    """Await `fn(*request)` for each request, `window` at once."""
    pending: dict[asyncio.Task[R], int] = {}
    errors: list[Exception] = []
    started = 0
    try:
        while pending or (not errors and started < len(requests)):
            while not errors and started < len(requests) and len(pending) < window:
                pending[asyncio.ensure_future(fn(*requests[started]))] = started
                started += 1
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                index = pending.pop(task)
                error = task.exception()
                if error is None:
                    yield index, task.result()
                elif isinstance(error, Exception):
                    errors.append(error)
                else:
                    raise error
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    if errors:
        raise ExceptionGroup(_MESSAGE, errors)
