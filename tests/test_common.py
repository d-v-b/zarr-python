from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, get_args

import numpy as np
import pytest

from tests.conftest import Expect
from zarr.core.common import (
    ANY_ACCESS_MODE,
    AccessModeLiteral,
    ceildiv,
    concurrent_foreach,
    concurrent_iter,
    concurrent_map,
    parse_bool,
    parse_int,
    parse_name,
    parse_order,
    parse_shapelike,
    product,
    to_thread_to_completion,
)
from zarr.core.config import parse_indexing_order

if TYPE_CHECKING:
    from typing import Any, Literal


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [(0, 0, 0), (7, 3, 3), (7.5, 2, 4), (2**62 - 1, 1, 2**62)],
)
def test_ceildiv(a: float, b: float, expected: int) -> None:
    """The original helper retains its floating-point division behavior."""
    assert ceildiv(a, b) == expected


@pytest.mark.parametrize(
    "case",
    [
        Expect(input=(0, 3), output=0, id="zero"),
        Expect(input=(7, 3), output=3, id="round-up"),
        Expect(input=(9, 3), output=3, id="exact"),
        Expect(input=(-7, 3), output=-2, id="negative-numerator"),
        Expect(input=(7, -3), output=-2, id="negative-divisor"),
        Expect(input=(-7, -3), output=3, id="both-negative"),
        Expect(input=(2**62 - 1, 1), output=2**62 - 1, id="large-exact"),
        Expect(input=(2**62 + 1, 2), output=2**61 + 1, id="large-round-up"),
        Expect(input=(2**60 + 3, 1), output=2**60 + 3, id="large-low-bits"),
        Expect(
            input=(np.int64(2**62 - 1), np.int64(1)),
            output=2**62 - 1,
            id="numpy-signed",
        ),
        Expect(
            input=(np.int64(-(2**63)), np.int64(-1)),
            output=2**63,
            id="numpy-signed-minimum",
        ),
        Expect(
            input=(np.uint64(2**64 - 1), np.uint64(2)),
            output=2**63,
            id="numpy-unsigned-maximum",
        ),
    ],
    ids=lambda case: case.id,
)
def test_ceildiv_int(case: Expect[tuple[int, int], int]) -> None:
    from zarr.core.common import ceildiv_int

    result = ceildiv_int(*case.input)
    assert result == case.output
    assert isinstance(result, int)


@pytest.mark.parametrize("numerator", [0, 1])
def test_ceildiv_int_zero_divisor(numerator: int) -> None:
    from zarr.core.common import ceildiv_int

    with pytest.raises(ZeroDivisionError):
        ceildiv_int(numerator, 0)


@pytest.mark.parametrize("data", [(0, 0, 0, 0), (1, 3, 4, 5, 6), (2, 4)])
def test_product(data: tuple[int, ...]) -> None:
    assert product(data) == np.prod(data)


def test_access_modes() -> None:
    """
    Test that the access modes type and variable for run-time checking are equivalent.
    """
    assert set(ANY_ACCESS_MODE) == set(get_args(AccessModeLiteral))


async def test_concurrent_iter_schedules_eagerly() -> None:
    """`concurrent_iter` must return already-scheduled tasks, not a lazy generator.

    Its docstring promises `func(*item)` is launched concurrently for every
    item up front; a caller that awaits the returned tasks one at a time
    (rather than via `gather`/`as_completed`, which force iteration) relies
    on that eager scheduling to get any overlap at all.
    """
    started = [False, False, False]

    async def mark(i: int) -> int:
        started[i] = True
        return i

    tasks = concurrent_iter([(0,), (1,), (2,)], mark)

    # Give the event loop one chance to run before awaiting anything
    # individually. If `concurrent_iter` were lazy, nothing would have been
    # scheduled yet and `started` would still be all-False here.
    await asyncio.sleep(0)
    assert started == [True, True, True]

    results = [await t for t in tasks]
    assert results == [0, 1, 2]


async def test_concurrent_map() -> None:
    """Results come back in input order, whatever order the calls finish in."""

    async def work(i: int) -> int:
        for _ in range(3 - i):
            await asyncio.sleep(0)
        return i

    assert await concurrent_map([(0,), (1,), (2,)], work) == [0, 1, 2]
    assert await concurrent_map([(0,), (1,), (2,)], work, limit=1) == [0, 1, 2]


async def test_concurrent_map_failure_stops_other_calls_before_raising() -> None:
    """When a call fails, every call that started has ended, and the rest never start.

    Item 1 is in flight when item 0 fails and needs several event-loop iterations to
    wind down. The failure frees a slot, so one queued item may take a first step
    before it is cancelled too, but the last items are never reached.
    """
    n_items = 6
    item_1_started = asyncio.Event()
    never = asyncio.Event()
    started: list[int] = []
    running: set[int] = set()

    async def work(i: int) -> None:
        started.append(i)
        if i == 0:
            await item_1_started.wait()
            raise ValueError("failed")
        running.add(i)
        item_1_started.set()
        try:
            await never.wait()
        finally:
            for _ in range(5):
                await asyncio.sleep(0)
            running.remove(i)

    with pytest.raises(ValueError, match="failed"):
        await concurrent_map([(i,) for i in range(n_items)], work, limit=2)

    assert 1 in started
    assert len(started) < n_items
    assert running == set()


async def test_to_thread_to_completion() -> None:
    """Passes arguments through and returns the result, like `asyncio.to_thread`."""

    def add(a: int, *, b: int) -> int:
        return a + b

    assert await to_thread_to_completion(add, 1, b=2) == 3


async def test_to_thread_to_completion_raises_func_exception() -> None:
    def fail() -> None:
        raise ValueError("failed")

    with pytest.raises(ValueError, match="failed"):
        await to_thread_to_completion(fail)


async def test_to_thread_to_completion_holds_cancellation_until_func_returns() -> None:
    func_started = threading.Event()
    release_func = threading.Event()

    def func() -> None:
        func_started.set()
        release_func.wait(timeout=30)

    task = asyncio.ensure_future(to_thread_to_completion(func))
    await asyncio.to_thread(func_started.wait, 30)
    task.cancel()
    # Cancelled more than once, as an impatient caller might: still held back.
    for _ in range(3):
        await asyncio.sleep(0)
        task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release_func.set()
    with pytest.raises(asyncio.CancelledError):
        await task


class _Abort(BaseException):
    """A BaseException that is neither KeyboardInterrupt nor SystemExit."""


@pytest.mark.parametrize(("n_items", "limit"), [(0, 1), (1, 4), (7, 1), (7, 3), (7, 7), (7, 50)])
async def test_concurrent_foreach(n_items: int, limit: int) -> None:
    """Every item is processed exactly once, lazily, with at most `limit` in flight."""
    seen: list[int] = []
    in_flight = 0
    peak = 0
    pulled = 0

    def items() -> Iterable[tuple[int]]:
        nonlocal pulled
        for i in range(n_items):
            pulled += 1
            yield (i,)

    async def work(i: int) -> None:
        nonlocal in_flight, peak
        # Laziness: an item is pulled only when a worker is free to take it.
        assert pulled <= i + limit
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        seen.append(i)

    await concurrent_foreach(items(), work, limit)
    assert sorted(seen) == list(range(n_items))
    assert peak == min(n_items, limit)


async def test_concurrent_foreach_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be at least 1"):
        await concurrent_foreach([(0,)], asyncio.sleep, 0)


async def test_concurrent_foreach_single_failure_is_raised_bare() -> None:
    """One failing call surfaces as itself and stops the rest of the stream."""
    started: list[int] = []

    async def work(i: int) -> None:
        started.append(i)
        await asyncio.sleep(0)
        if i == 2:
            raise OSError("boom")

    with pytest.raises(OSError, match="boom") as info:
        await concurrent_foreach(((i,) for i in range(100)), work, 2)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__
    assert len(started) < 100


async def test_concurrent_foreach_fast_failure_is_not_repeated() -> None:
    """A call that fails before its first await stops the other workers from
    taking items, so a read-only store raises once, not once per worker."""
    calls = 0

    async def work(i: int) -> None:
        nonlocal calls
        calls += 1
        raise PermissionError("read only")

    with pytest.raises(PermissionError, match="read only") as info:
        await concurrent_foreach(((i,) for i in range(100)), work, 8)
    assert calls == 1
    assert info.value.__suppress_context__


async def test_concurrent_foreach_simultaneous_failures_keep_the_group() -> None:
    """Failures from the same loop iteration are all kept on the raised exception."""

    async def work(i: int) -> None:
        await asyncio.sleep(0)
        if i == 0:
            raise OSError("first")
        raise ValueError("second")

    with pytest.raises(OSError, match="first") as info:
        await concurrent_foreach([(0,), (1,)], work, 2)
    group = info.value.__cause__
    assert isinstance(group, ExceptionGroup)
    assert {type(e) for e in group.exceptions} == {OSError, ValueError}


async def test_concurrent_foreach_base_exception_is_raised_bare() -> None:
    async def work(i: int) -> None:
        raise _Abort

    with pytest.raises(_Abort):
        await concurrent_foreach([(0,), (1,)], work, 1)


# todo: test
def test_to_thread() -> None: ...


# todo: test
def test_enum_names() -> None: ...


# todo: test
def test_parse_enum() -> None: ...


@pytest.mark.parametrize("data", [("foo", "bar"), (10, 11)])
def test_parse_name_invalid(data: tuple[Any, Any]) -> None:
    observed, expected = data
    if isinstance(observed, str):
        with pytest.raises(ValueError, match=f"Expected '{expected}'. Got {observed} instead."):
            parse_name(observed, expected)
    else:
        with pytest.raises(
            TypeError, match=f"Expected a string, got an instance of {type(observed)}."
        ):
            parse_name(observed, expected)


@pytest.mark.parametrize("data", [("foo", "foo"), ("10", "10")])
def test_parse_name_valid(data: tuple[Any, Any]) -> None:
    observed, expected = data
    assert parse_name(observed, expected) == observed


@pytest.mark.parametrize("data", [0, 1, "hello", "f"])
def test_parse_indexing_order_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match="Failed to parse input for 'order'"):
        parse_indexing_order(data)


@pytest.mark.parametrize("data", [0, 1, "hello", "f"])
def test_parse_order_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match="Failed to parse input for 'order'"):
        parse_order(data)


@pytest.mark.parametrize("data", [0, 1, "true", None, [True]])
def test_parse_bool_invalid(data: Any) -> None:
    """Non-bool values are rejected with a ValueError."""
    with pytest.raises(ValueError, match="Expected instance of bool"):
        parse_bool(data)


@pytest.mark.parametrize("data", [True, False])
def test_parse_bool_valid(data: bool) -> None:
    assert parse_bool(data) is data


@pytest.mark.parametrize("data", ["1", 1.0, True, False, None, [1], (1,)])
def test_parse_int_invalid(data: Any) -> None:
    """Non-int values (including bools, which are int subclasses) are rejected."""
    with pytest.raises(ValueError, match="Expected int"):
        parse_int(data)


@pytest.mark.parametrize("data", [0, 1, -1, 2**63])
def test_parse_int_valid(data: int) -> None:
    assert parse_int(data) == data


@pytest.mark.parametrize("data", ["C", "F"])
def parse_indexing_order_valid(data: Literal["C", "F"]) -> None:
    assert parse_indexing_order(data) == data


@pytest.mark.parametrize("data", [lambda v: v, slice(None)])
def test_parse_shapelike_invalid_single_type(data: Any) -> None:
    """
    Test that we get the expected error message when passing in a value that is not an integer
    or an iterable of integers.
    """
    with pytest.raises(TypeError, match="Expected an integer or an iterable of integers."):
        parse_shapelike(data)


def test_parse_shapelike_invalid_single_value() -> None:
    """
    Test that we get the expected error message when passing in a negative integer.
    """
    with pytest.raises(ValueError, match="Expected a non-negative integer."):
        parse_shapelike(-1)


@pytest.mark.parametrize("data", ["shape", ("0", 1, 2, 3), {"0": "0"}, ((1, 2), (2, 2)), (4.0, 2)])
def test_parse_shapelike_invalid_iterable_types(data: Any) -> None:
    """
    Test that we get the expected error message when passing in an iterable containing
    non-integer elements
    """
    with pytest.raises(TypeError, match="Expected an iterable of integers"):
        parse_shapelike(data)


@pytest.mark.parametrize("data", [(1, 2, 3, -1), (-10,)])
def test_parse_shapelike_invalid_iterable_values(data: Any) -> None:
    """
    Test that we get the expected error message when passing in an iterable containing negative
    integers
    """
    with pytest.raises(ValueError, match="Expected all values to be non-negative."):
        parse_shapelike(data)


@pytest.mark.parametrize(
    "data", [range(10), [0, 1, 2, np.uint64(3)], (3, 4, 5), (), 1, np.uint8(1)]
)
def test_parse_shapelike_valid(data: Iterable[int] | int) -> None:
    if isinstance(data, Iterable):
        expected = tuple(data)
    else:
        expected = (data,)
    assert parse_shapelike(data) == expected


# todo: more dtypes
@pytest.mark.parametrize("data", [("uint8", np.uint8), ("float64", np.float64)])
def parse_dtype(data: tuple[str, np.dtype[Any]]) -> None:
    unparsed, parsed = data
    assert parse_dtype(unparsed) == parsed


# todo: figure out what it means to test this
def test_parse_fill_value() -> None: ...
