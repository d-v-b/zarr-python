from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import obstore
import pytest
from obstore.exceptions import GenericError
from obstore.store import LocalStore, MemoryStore

from zarr_storage import ObjectMetadata, RangeRequest, Store
from zarr_storage.backends import ObjectStoreBackend
from zarr_storage.backends._obstore import _empty_read_or_raise, _generation
from zarr_storage.testing import (
    ConditionalPutSuite,
    KeyAndDirectorySuite,
    KeyDirectoryConflictSuite,
    NoConditionalDeleteSuite,
    NoConditionalPutSuite,
    PickleSuite,
    StoreSuite,
    UniqueGenerationsSuite,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path


_SYNC_CALLS = ["get", "get_ranges", "head", "put", "delete", "list_with_delimiter", "copy"]


class _NotOnTheEventLoop:
    """An obstore function that refuses to run on a thread that is running an event loop."""

    def __init__(self, real: Callable[..., object]) -> None:
        self._real = real

    def __call__(self, *args: object, **kwargs: object) -> object:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._real(*args, **kwargs)
        raise AssertionError("A blocking obstore call ran inside an async method.")


class NoBlockingCallsInAsyncRuns:
    """Fail any async run of a test that makes a blocking obstore call."""

    @pytest.fixture(autouse=True)
    def _forbid_blocking_calls(
        self, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        callspec = getattr(request.node, "callspec", None)
        params = {} if callspec is None else callspec.params
        if "async" in (params.get("store"), params.get("api")):
            for name in _SYNC_CALLS:
                monkeypatch.setattr(obstore, name, _NotOnTheEventLoop(getattr(obstore, name)))


class TestObjectStoreMemory(
    StoreSuite,
    UniqueGenerationsSuite,
    ConditionalPutSuite,
    NoConditionalDeleteSuite,
    KeyAndDirectorySuite,
    NoBlockingCallsInAsyncRuns,
):
    def make_store(self, directory: Path) -> Store[ObjectStoreBackend]:
        return Store(ObjectStoreBackend(MemoryStore()))


class TestObjectStoreLocal(
    StoreSuite,
    NoConditionalPutSuite,
    NoConditionalDeleteSuite,
    KeyDirectoryConflictSuite,
    PickleSuite,
    NoBlockingCallsInAsyncRuns,
):
    def make_store(self, directory: Path) -> Store[ObjectStoreBackend]:
        return Store(ObjectStoreBackend(LocalStore(directory)))


_META = ObjectMetadata(10, _generation("e", None), None)


def test_refused_empty_read_is_empty() -> None:
    result = _empty_read_or_raise(_META, RangeRequest(10, 12), GenericError("refused"))
    assert result is not None
    assert (bytes(result.value), result.generation) == (b"", _META.generation)


def test_refused_readable_range_reraises() -> None:
    error = GenericError("refused")
    with pytest.raises(GenericError, match="refused"):
        _empty_read_or_raise(_META, RangeRequest(2, 5), error)


class _FailFor:
    """An obstore function that raises a transient error for one path and works for the rest."""

    def __init__(self, real: Callable[..., object], path: str) -> None:
        self._real = real
        self._path = path

    def __call__(
        self, store: object, path: str | None = None, *args: object, **kwargs: object
    ) -> object:
        if path == self._path:
            raise GenericError("injected: 503 Service Unavailable")
        return self._real(store, path, *args, **kwargs)


def _list_a(store: Store[ObjectStoreBackend]) -> object:
    return list(store.list("a"))


def _list_dir_a(store: Store[ObjectStoreBackend]) -> object:
    return list(store.list_dir("a"))


def _get_ab(store: Store[ObjectStoreBackend]) -> object:
    return store.get("a/b")


def _head_ab(store: Store[ObjectStoreBackend]) -> object:
    return store.head("a/b")


async def _collect(items: AsyncIterator[object]) -> list[object]:
    return [item async for item in items]


def _list_a_async(store: Store[ObjectStoreBackend]) -> object:
    return asyncio.run(_collect(store.list_async("a")))


def _list_dir_a_async(store: Store[ObjectStoreBackend]) -> object:
    return asyncio.run(_collect(store.list_dir_async("a")))


def _get_ab_async(store: Store[ObjectStoreBackend]) -> object:
    return asyncio.run(store.get_async("a/b"))


@pytest.mark.parametrize(
    ("operation", "failing", "path"),
    [
        (_list_a, "list", "a"),
        (_list_dir_a, "list_with_delimiter", "a"),
        (_get_ab, "get", "a/b"),
        (_head_ab, "head", "a/b"),
        (_list_a_async, "list", "a"),
        (_list_dir_a_async, "list_with_delimiter_async", "a"),
        (_get_ab_async, "get_async", "a/b"),
    ],
    ids=["list", "list_dir", "get", "head", "list_async", "list_dir_async", "get_async"],
)
def test_errors_on_object_stores_are_not_read_as_absent(
    operation: Callable[[Store[ObjectStoreBackend]], object],
    failing: str,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a filesystem store's refusals mean "under a key"; elsewhere an error is an error.

    Only the operation's own call fails, so a check of the path's parents would succeed and could
    wrongly turn the error into "absent".
    """
    store = Store(ObjectStoreBackend(MemoryStore()))
    store.put("a", b"x")
    store.put("a/b", b"x")
    monkeypatch.setattr(obstore, failing, _FailFor(getattr(obstore, failing), path))
    with pytest.raises(GenericError):
        operation(store)
