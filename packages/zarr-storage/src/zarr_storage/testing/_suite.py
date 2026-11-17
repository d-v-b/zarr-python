"""The storage contract, as tests.

`StoreSuite` holds what every backend must do. Capabilities a backend may lack come as mixins: a
backend's test class combines `StoreSuite` with the mixin for each capability it has, or with the
mixin that checks it refuses that capability cleanly. Every test that takes the `store` fixture runs
twice, through the sync methods and through the async methods (see `AsyncAPI`).

Each behavior has one table-driven test for the expected outputs and one test per error case.
"""

from __future__ import annotations

import array
import asyncio
import pickle
import threading
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import run_state_machine_as_test

from zarr_storage._errors import (
    InvalidByteRangeError,
    InvalidKeyError,
    KeyConflictError,
    ReadOnlyError,
    StoreClosedError,
    UnsupportedOperationError,
)
from zarr_storage._types import (
    Generation,
    Key,
    OffsetRequest,
    Prefix,
    RangeRequest,
    SuffixRequest,
)
from zarr_storage.testing._api import AsyncAPI, FullBackend, StoreAPI, unwrap
from zarr_storage.testing._stateful import Capabilities, StoreMachine

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    from typing_extensions import Buffer

    from zarr_storage._store import Store
    from zarr_storage._types import ByteRange, ListItem

__all__ = [
    "ConditionalDeleteSuite",
    "ConditionalPutSuite",
    "ConditionalRaceSuite",
    "KeyAndDirectorySuite",
    "KeyDirectoryConflictSuite",
    "NoConditionalDeleteSuite",
    "NoConditionalPutSuite",
    "PickleSuite",
    "StoreSuite",
    "UniqueGenerationsSuite",
]

VALUE = b"0123456789"

# ---------------------------------------------------------------------------- tables

VALID_KEYS = ["a", "a/b", "a/b/c", ".a", "a.b", "..a", "a..x", "__a", "zarr.json", "c/0/0", "a b"]
INVALID_KEYS = ["", "/", "/a", "a/", "a//b", ".", "..", "./a", "a/.", "a/../b", "a\x00b"]
INVALID_DIRECTORIES = ["//", "/a", "a//b", "..", "a/./b", "a\x00"]

VALUES: list[tuple[str, Buffer, bytes]] = [
    ("bytes", VALUE, VALUE),
    ("empty", b"", b""),
    ("bytearray", bytearray(VALUE), VALUE),
    ("memoryview", memoryview(VALUE), VALUE),
    ("int-array", array.array("i", [1, 2]), array.array("i", [1, 2]).tobytes()),
    ("2d-view", memoryview(VALUE).cast("B", (2, 5)), VALUE),
]

BYTE_RANGES: list[tuple[ByteRange | None, bytes]] = [
    (None, VALUE),
    (RangeRequest(0, 10), VALUE),
    (RangeRequest(2, 5), b"234"),
    (RangeRequest(5, 5), b""),
    (RangeRequest(8, 100), b"89"),
    (RangeRequest(10, 12), b""),
    (OffsetRequest(0), VALUE),
    (OffsetRequest(7), b"789"),
    (OffsetRequest(10), b""),
    (SuffixRequest(3), b"789"),
    (SuffixRequest(0), b""),
    (SuffixRequest(100), VALUE),
]


@dataclass(frozen=True)
class PutCase:
    present: bool
    precondition: Literal["none", "if_none_match", "if_match_current", "if_match_stale"]
    applied: bool
    generation: Literal["new", "current", "none"]


UNCONDITIONAL_PUT_CASES = [
    PutCase(present=False, precondition="none", applied=True, generation="new"),
    PutCase(present=True, precondition="none", applied=True, generation="new"),
    PutCase(present=False, precondition="if_none_match", applied=True, generation="new"),
    PutCase(present=True, precondition="if_none_match", applied=False, generation="current"),
]
IF_MATCH_PUT_CASES = [
    PutCase(present=True, precondition="if_match_current", applied=True, generation="new"),
    PutCase(present=True, precondition="if_match_stale", applied=False, generation="current"),
    PutCase(present=False, precondition="if_match_stale", applied=False, generation="none"),
]


@dataclass(frozen=True)
class DeleteCase:
    present: bool
    precondition: Literal["none", "if_match_current", "if_match_stale"]
    applied: bool
    generation: Literal["current", "none"]


UNCONDITIONAL_DELETE_CASES = [
    DeleteCase(present=False, precondition="none", applied=False, generation="none"),
    DeleteCase(present=True, precondition="none", applied=True, generation="current"),
]
IF_MATCH_DELETE_CASES = [
    DeleteCase(present=True, precondition="if_match_current", applied=True, generation="current"),
    DeleteCase(present=True, precondition="if_match_stale", applied=False, generation="current"),
    DeleteCase(present=False, precondition="if_match_stale", applied=False, generation="none"),
]

TREE = [
    "zarr.json",
    "a/zarr.json",
    "a/c/0",
    "a/c/1",
    "ab/zarr.json",
    "lat/zarr.json",
    "lat_bnds/zarr.json",
]


def _without(*removed: str) -> set[str]:
    return set(TREE) - set(removed)


DELETE_DIR_CASES: list[tuple[str, set[str]]] = [
    ("a", _without("a/zarr.json", "a/c/0", "a/c/1")),
    ("a/", _without("a/zarr.json", "a/c/0", "a/c/1")),
    ("lat", _without("lat/zarr.json")),
    ("a/c", _without("a/c/0", "a/c/1")),
    ("a/c/0", _without()),
    ("missing", _without()),
    ("", set()),
]

LIST_DIRECTORIES = ["", "a", "a/", "a/c", "lat", "lat/", "missing", "a/c/0"]

VIEW_ROOTS: list[tuple[list[str], str]] = [
    ([], ""),
    ([""], ""),
    (["a"], "a"),
    (["a/"], "a"),
    (["a", "b"], "a/b"),
    (["a/b", ""], "a/b"),
    (["a", "b/c/"], "a/b/c"),
    (["/"], ""),
    (["a", "/"], "a"),
]

KEY_SETS = st.sets(
    st.lists(st.sampled_from(["a", "b", "ab", "c.d", "_"]), min_size=1, max_size=3).map("/".join),
    max_size=8,
)
LISTED_DIRECTORIES = st.sampled_from(["", "a", "b", "a/a", "ab", "a/b/"])


def _model_list(keys: Iterable[str], directory: str) -> set[str]:
    directory = directory.removesuffix("/")
    if directory == "":
        return set(keys)
    return {k for k in keys if k.startswith(f"{directory}/")}


def _model_list_dir(keys: Iterable[str], directory: str) -> set[ListItem]:
    directory = directory.removesuffix("/")
    items: set[ListItem] = set()
    for key in _model_list(keys, directory):
        rest = key if directory == "" else key.removeprefix(f"{directory}/")
        name, sep, _ = rest.partition("/")
        items.add(Prefix(name) if sep else Key(name))
    return items


def _without_key_directories(keys: set[str]) -> set[str]:
    """`keys` minus those that are also another key's directory."""
    return {k for k in keys if not any(other.startswith(f"{k}/") for other in keys)}


def _stale() -> Generation:
    return Generation(object())


# ---------------------------------------------------------------------------- operations


def _get_key(store: StoreAPI, key: str) -> object:
    return store.get(key)


def _head_key(store: StoreAPI, key: str) -> object:
    return store.head(key)


def _put_key(store: StoreAPI, key: str) -> object:
    return store.put(key, VALUE)


def _delete_key(store: StoreAPI, key: str) -> object:
    return store.delete(key)


KEY_OPERATIONS: list[Callable[[StoreAPI, str], object]] = [
    _get_key,
    _head_key,
    _put_key,
    _delete_key,
]
KEY_OPERATION_IDS = ["get", "head", "put", "delete"]


def _view_of(store: StoreAPI, directory: str) -> object:
    return store / directory


def _list_of(store: StoreAPI, directory: str) -> object:
    return store.list(directory)


def _list_dir_of(store: StoreAPI, directory: str) -> object:
    return store.list_dir(directory)


def _delete_dir_of(store: StoreAPI, directory: str) -> object:
    store.delete_dir(directory)
    return None


DIRECTORY_OPERATIONS: list[Callable[[StoreAPI, str], object]] = [
    _view_of,
    _list_of,
    _list_dir_of,
    _delete_dir_of,
]
DIRECTORY_OPERATION_IDS = ["view", "list", "list_dir", "delete_dir"]


def _missing_get(store: StoreAPI) -> object:
    return store.get("missing")


def _missing_get_directory(store: StoreAPI) -> object:
    return store.get("a")


def _missing_get_range(store: StoreAPI) -> object:
    return store.get("missing", byte_range=RangeRequest(0, 1))


def _missing_get_ranges(store: StoreAPI) -> object:
    return store.get_ranges("missing", [RangeRequest(0, 1)])


def _missing_get_into(store: StoreAPI) -> object:
    return store.get_into("missing", bytearray(1))


def _missing_head(store: StoreAPI) -> object:
    return store.head("a")


def _missing_exists(store: StoreAPI) -> object:
    return store.exists("a")


MISSING_READS: list[tuple[Callable[[StoreAPI], object], object]] = [
    (_missing_get, None),
    (_missing_get_directory, None),
    (_missing_get_range, None),
    (_missing_get_ranges, None),
    (_missing_get_into, None),
    (_missing_head, None),
    (_missing_exists, False),
]
MISSING_READ_IDS = ["get", "get-directory", "get-range", "get_ranges", "get_into", "head", "exists"]


def _put(store: StoreAPI) -> object:
    return store.put("k", VALUE)


def _delete(store: StoreAPI) -> object:
    return store.delete("k")


def _delete_dir(store: StoreAPI) -> object:
    store.delete_dir("")
    return None


WRITES: list[Callable[[StoreAPI], object]] = [_put, _delete, _delete_dir]
WRITE_IDS = ["put", "delete", "delete_dir"]


def _get(store: StoreAPI) -> object:
    return store.get("k")


def _get_ranges(store: StoreAPI) -> object:
    return store.get_ranges("k", [SuffixRequest(1)])


def _get_into(store: StoreAPI) -> object:
    return store.get_into("k", bytearray(1))


def _head(store: StoreAPI) -> object:
    return store.head("k")


def _list(store: StoreAPI) -> object:
    return store.list()


def _list_dir(store: StoreAPI) -> object:
    return store.list_dir()


CLOSED_OPERATIONS: list[Callable[[StoreAPI], object]] = [
    _get,
    _get_ranges,
    _get_into,
    _head,
    _put,
    _delete,
    _delete_dir,
    _list,
    _list_dir,
]
CLOSED_IDS = [
    "get",
    "get_ranges",
    "get_into",
    "head",
    "put",
    "delete",
    "delete_dir",
    "list",
    "list_dir",
]


def _write_repeatedly(
    store: Store[FullBackend], stop: threading.Event, written: dict[Generation, int]
) -> None:
    fill = 0
    while not stop.is_set():
        fill = (fill + 1) % 256
        generation = store.put("k", bytes([fill]) * 1000).generation
        assert generation is not None
        written[generation] = fill


def _create(
    store: Store[FullBackend], key: str, barrier: threading.Barrier, results: list[bool]
) -> None:
    barrier.wait()
    results.append(store.put(key, VALUE, if_none_match=True).applied)


def _check_put(store: StoreAPI, case: PutCase) -> None:
    current = store.put("k", VALUE).generation if case.present else None
    match case.precondition:
        case "none":
            result = store.put("k", b"new")
        case "if_none_match":
            result = store.put("k", b"new", if_none_match=True)
        case "if_match_current":
            assert current is not None
            result = store.put("k", b"new", if_match=current)
        case "if_match_stale":
            result = store.put("k", b"new", if_match=_stale())
    assert result.applied is case.applied
    match case.generation:
        case "new":
            assert result.generation is not None
            assert result.generation != current
        case "current":
            assert result.generation == current
        case "none":
            assert result.generation is None
    stored = store.get("k")
    expected = b"new" if case.applied else (VALUE if case.present else None)
    assert (None if stored is None else bytes(stored.value)) == expected


def _check_delete(store: StoreAPI, case: DeleteCase) -> None:
    current = store.put("k", VALUE).generation if case.present else None
    match case.precondition:
        case "none":
            result = store.delete("k")
        case "if_match_current":
            assert current is not None
            result = store.delete("k", if_match=current)
        case "if_match_stale":
            result = store.delete("k", if_match=_stale())
    assert result.applied is case.applied
    assert result.generation == (current if case.generation == "current" else None)
    assert store.exists("k") is (case.present and not case.applied)


def _check_listing(store: Store[FullBackend], keys: set[str], directory: str) -> None:
    for key in keys:
        store.put(key, VALUE)
    assert set(store.list(directory)) == _model_list(keys, directory)
    assert set(store.list_dir(directory)) == _model_list_dir(keys, directory)


def _machine_store(
    suite: StoreSuite,
    directories: pytest.TempPathFactory,
    api: Literal["sync", "async"],
    runner: asyncio.Runner,
) -> StoreAPI:
    store = suite.make_store(directories.mktemp("machine"))
    return store if api == "sync" else AsyncAPI(store, runner)


# ---------------------------------------------------------------------------- the core suite


class StoreSuite:
    """What every backend must do.

    Override `make_store` to return an empty, writable store over a fresh backend. `directory` is a
    new empty directory a filesystem backend can use; other backends ignore it.
    """

    def make_store(self, directory: Path) -> Store[FullBackend]:
        raise NotImplementedError("Override make_store to return an empty store.")

    @pytest.fixture(params=["sync", "async"])
    def store(self, request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StoreAPI]:
        store = self.make_store(tmp_path)
        if request.param == "sync":
            yield store
            return
        with asyncio.Runner() as runner:
            yield AsyncAPI(store, runner)

    # ------------------------------------------------------------------ everything at once

    @pytest.mark.parametrize("api", ["sync", "async"])
    def test_state_machine(
        self, tmp_path_factory: pytest.TempPathFactory, api: Literal["sync", "async"]
    ) -> None:
        """Random sequences of operations through random views agree with a dict model."""
        capabilities = Capabilities(
            conditional_put=isinstance(self, ConditionalPutSuite),
            conditional_delete=isinstance(self, ConditionalDeleteSuite),
            holds_key_and_directory=isinstance(self, KeyAndDirectorySuite),
            leaves_empty_directories=isinstance(self, KeyDirectoryConflictSuite),
            unique_generations=isinstance(self, UniqueGenerationsSuite),
        )
        with asyncio.Runner() as runner:
            make_store = partial(_machine_store, self, tmp_path_factory, api, runner)
            run_state_machine_as_test(
                # Hypothesis calls the factory with no arguments; a partial is a fine factory.
                partial(StoreMachine, make_store, capabilities),  # pyright: ignore[reportArgumentType]
                settings=settings(
                    max_examples=40,
                    stateful_step_count=30,
                    deadline=None,
                    suppress_health_check=[HealthCheck.differing_executors],
                ),
            )

    # ------------------------------------------------------------------ keys

    @pytest.mark.parametrize("key", VALID_KEYS)
    def test_valid_key_round_trips(self, store: StoreAPI, key: str) -> None:
        store.put(key, VALUE)
        result = store.get(key)
        assert result is not None
        assert bytes(result.value) == VALUE
        assert key in set(store.list())

    @pytest.mark.parametrize("key", INVALID_KEYS)
    @pytest.mark.parametrize("operation", KEY_OPERATIONS, ids=KEY_OPERATION_IDS)
    def test_invalid_key_raises(
        self, store: StoreAPI, key: str, operation: Callable[[StoreAPI, str], object]
    ) -> None:
        with pytest.raises(InvalidKeyError):
            operation(store, key)

    @pytest.mark.parametrize("directory", INVALID_DIRECTORIES)
    @pytest.mark.parametrize("operation", DIRECTORY_OPERATIONS, ids=DIRECTORY_OPERATION_IDS)
    def test_invalid_directory_raises(
        self, store: StoreAPI, directory: str, operation: Callable[[StoreAPI, str], object]
    ) -> None:
        with pytest.raises(InvalidKeyError):
            operation(store, directory)

    # ------------------------------------------------------------------ get / put

    @pytest.mark.parametrize(
        ("value", "expected"), [v[1:] for v in VALUES], ids=[v[0] for v in VALUES]
    )
    def test_put_then_get(self, store: StoreAPI, value: Buffer, expected: bytes) -> None:
        written = store.put("k", value)
        assert written.applied
        result = store.get("k")
        assert result is not None
        assert (bytes(result.value), result.generation) == (expected, written.generation)
        meta = store.head("k")
        assert meta is not None
        assert (meta.size, meta.generation) == (len(expected), written.generation)
        assert meta.last_modified is None or abs(meta.last_modified - time.time()) < 3600
        assert store.exists("k")

    def test_put_copies_the_value(self, store: StoreAPI) -> None:
        value = bytearray(VALUE)
        store.put("k", value)
        value[0:1] = b"X"
        result = store.get("k")
        assert result is not None
        assert bytes(result.value) == VALUE

    @pytest.mark.parametrize(("read", "expected"), MISSING_READS, ids=MISSING_READ_IDS)
    def test_reads_of_missing_key(
        self, store: StoreAPI, read: Callable[[StoreAPI], object], expected: object
    ) -> None:
        store.put("a/k", VALUE)
        assert read(store) == expected

    # ------------------------------------------------------------------ byte ranges

    @pytest.mark.parametrize(("byte_range", "expected"), BYTE_RANGES)
    def test_byte_range(
        self, store: StoreAPI, byte_range: ByteRange | None, expected: bytes
    ) -> None:
        store.put("k", VALUE)
        result = store.get("k", byte_range=byte_range)
        assert result is not None
        assert bytes(result.value) == expected
        out = bytearray(len(VALUE) + 1)
        n = store.get_into("k", out, byte_range=byte_range)
        assert n == len(expected)
        assert bytes(out[:n]) == expected
        if byte_range is not None:
            ranges = store.get_ranges("k", [byte_range, SuffixRequest(1)])
            assert ranges is not None
            assert [bytes(v) for v in ranges.values] == [expected, b"9"]
            assert ranges.generation == result.generation

    @pytest.mark.parametrize("byte_range", [RangeRequest(11, 12), OffsetRequest(11)])
    def test_byte_range_past_end_raises(self, store: StoreAPI, byte_range: ByteRange) -> None:
        store.put("k", VALUE)
        with pytest.raises(InvalidByteRangeError):
            store.get("k", byte_range=byte_range)

    def test_get_into_short_buffer_raises(self, store: StoreAPI) -> None:
        store.put("k", VALUE)
        with pytest.raises(ValueError, match="holds"):
            store.get_into("k", bytearray(len(VALUE) - 1))

    def test_get_into_read_only_buffer_raises(self, store: StoreAPI) -> None:
        store.put("k", VALUE)
        with pytest.raises(ValueError, match="writable"):
            store.get_into("k", memoryview(bytes(len(VALUE))))

    def test_get_into_non_contiguous_buffer_raises(self, store: StoreAPI) -> None:
        store.put("k", VALUE)
        with pytest.raises(ValueError, match="contiguous"):
            store.get_into("k", memoryview(bytearray(2 * len(VALUE)))[::2])

    @pytest.mark.parametrize("api", ["sync", "async"])
    def test_get_ranges_reads_one_generation(self, tmp_path: Path, api: str) -> None:
        """Ranges read while another thread rewrites the value all come from the write whose
        generation the result reports."""
        store = self.make_store(tmp_path)
        written: dict[Generation, int] = {}
        put = store.put("k", bytes(1000))
        assert put.generation is not None
        written[put.generation] = 0
        stop = threading.Event()
        writer = threading.Thread(target=_write_repeatedly, args=(store, stop, written))
        seen: list[tuple[Generation, bytes, bytes]] = []
        writer.start()
        try:
            with asyncio.Runner() as runner:
                reader: StoreAPI = store if api == "sync" else AsyncAPI(store, runner)
                for _ in range(200):
                    result = reader.get_ranges("k", [RangeRequest(0, 1), SuffixRequest(1)])
                    assert result is not None
                    first, last = (bytes(v) for v in result.values)
                    seen.append((result.generation, first, last))
        finally:
            stop.set()
            writer.join()
        for generation, first, last in seen:
            assert first == last
            if isinstance(self, UniqueGenerationsSuite):
                assert first == bytes([written[generation]])

    # ------------------------------------------------------------------ writes

    @pytest.mark.parametrize("case", UNCONDITIONAL_PUT_CASES, ids=repr)
    def test_put(self, store: StoreAPI, case: PutCase) -> None:
        _check_put(store, case)

    def test_put_with_both_preconditions_raises(self, store: StoreAPI) -> None:
        with pytest.raises(ValueError, match="at most one"):
            store.put("k", VALUE, if_match=_stale(), if_none_match=True)

    def test_concurrent_creates_apply_once(self, tmp_path: Path) -> None:
        """Of many racing `if_none_match` writes to one key, exactly one applies."""
        store = self.make_store(tmp_path)
        for round_ in range(20):
            barrier = threading.Barrier(8)
            results: list[bool] = []
            threads = [
                threading.Thread(target=_create, args=(store, f"k{round_}", barrier, results))
                for _ in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert results.count(True) == 1

    @pytest.mark.parametrize("case", UNCONDITIONAL_DELETE_CASES, ids=repr)
    def test_delete(self, store: StoreAPI, case: DeleteCase) -> None:
        _check_delete(store, case)

    def test_delete_of_a_directory_deletes_nothing(self, store: StoreAPI) -> None:
        store.put("a/b", VALUE)
        result = store.delete("a")
        assert (result.applied, result.generation) == (False, None)
        assert store.exists("a/b")

    @pytest.mark.parametrize(("directory", "remaining"), DELETE_DIR_CASES)
    def test_delete_dir(self, store: StoreAPI, directory: str, remaining: set[str]) -> None:
        for key in TREE:
            store.put(key, VALUE)
        store.delete_dir(directory)
        assert set(store.list()) == remaining

    # ------------------------------------------------------------------ listing

    @pytest.mark.parametrize("directory", LIST_DIRECTORIES)
    def test_list(self, store: StoreAPI, directory: str) -> None:
        for key in TREE:
            store.put(key, VALUE)
        listed, children = list(store.list(directory)), list(store.list_dir(directory))
        assert len(listed) == len(set(listed))
        assert len(children) == len(set(children))
        assert set(listed) == _model_list(TREE, directory)
        assert set(children) == _model_list_dir(TREE, directory)

    @given(keys=KEY_SETS, directory=LISTED_DIRECTORIES)
    # Each backend's subclass runs this test; that is the point of the suite, not a flaky executor.
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.differing_executors]
    )
    def test_listing_matches_a_model(
        self, tmp_path_factory: pytest.TempPathFactory, keys: set[str], directory: str
    ) -> None:
        store = self.make_store(tmp_path_factory.mktemp("store"))
        _check_listing(store, _without_key_directories(keys), directory)

    # ------------------------------------------------------------------ views

    @pytest.mark.parametrize(("path", "root"), VIEW_ROOTS)
    def test_view_root(self, store: StoreAPI, path: list[str], root: str) -> None:
        view = store
        for directory in path:
            view = view / directory
        assert view.root == root

    def test_views_share_the_backend(self, store: StoreAPI) -> None:
        child = store / "a"
        child.put("k", VALUE)
        assert store.exists("a/k")
        store.put("a/b/c", VALUE)
        assert set(child.list()) == {"k", "b/c"}
        assert set(child.list("b")) == {"b/c"}
        assert set(child.list_dir()) == {Key("k"), Prefix("b")}
        result = child.get_ranges("b/c", [SuffixRequest(1)])
        assert result is not None
        assert [bytes(v) for v in result.values] == [b"9"]
        meta = child.head("b/c")
        assert meta is not None
        assert meta.size == len(VALUE)
        assert set((store / "lat").list()) == set()
        child.delete_dir()
        assert set(store.list()) == set()

    def test_identity(self, store: StoreAPI, tmp_path_factory: pytest.TempPathFactory) -> None:
        other = self.make_store(tmp_path_factory.mktemp("other"))
        base = unwrap(store)
        a, ab, lat, lat_bnds = base / "a", base / "a/b", base / "lat", base / "lat_bnds"
        table = [
            (base, base / "", True, True),
            (a, ab, True, True),
            (ab, a, True, True),
            (base, lat, True, True),
            (lat, lat_bnds, True, False),
            (a, lat, True, False),
            (a, a.with_read_only(), True, True),
            (base, other, False, False),
            (a, other / "a", False, False),
        ]
        for left, right, shares, overlaps in table:
            assert left.shares_backend(right) is shares
            assert left.overlaps(right) is overlaps
        assert base != base / ""

    def test_read_only_views(self, store: StoreAPI) -> None:
        store.put("k", VALUE)
        reader = store.with_read_only()
        assert (store.read_only, reader.read_only, (reader / "x").read_only) == (False, True, True)
        assert reader.exists("k")
        assert not store.with_read_only(False).read_only

    def test_writable_view_of_read_only_store_raises(self, store: StoreAPI) -> None:
        with pytest.raises(ReadOnlyError):
            store.with_read_only().with_read_only(False)

    @pytest.mark.parametrize("write", WRITES, ids=WRITE_IDS)
    def test_write_through_read_only_view_raises(
        self, store: StoreAPI, write: Callable[[StoreAPI], object]
    ) -> None:
        store.put("k", VALUE)
        with pytest.raises(ReadOnlyError):
            write(store.with_read_only())
        assert store.exists("k")

    # ------------------------------------------------------------------ lifecycle

    def test_closing_a_derived_view_leaves_the_backend_open(self, store: StoreAPI) -> None:
        store.put("k", VALUE)
        (store / "a").close()
        store.with_read_only().close()
        assert not store.closed
        assert store.exists("k")

    @pytest.mark.parametrize("operation", CLOSED_OPERATIONS, ids=CLOSED_IDS)
    def test_use_after_owner_closes_raises(
        self, store: StoreAPI, operation: Callable[[StoreAPI], object]
    ) -> None:
        view = store / "a"
        with store:
            store.put("a/k", VALUE)
        assert view.closed
        with pytest.raises(StoreClosedError):
            operation(view)

    @pytest.mark.parametrize("listing", ["list", "list_dir"])
    def test_iterating_after_close_raises(self, store: StoreAPI, listing: str) -> None:
        store.put("a", VALUE)
        store.put("b", VALUE)
        items = store.list() if listing == "list" else store.list_dir()
        store.close()
        with pytest.raises(StoreClosedError):
            next(iter(items))


# ---------------------------------------------------------------------------- capability mixins


class UniqueGenerationsSuite:
    """For backends that never hand out a generation twice, even after a delete.

    obstore's `LocalStore` doesn't qualify: its ETag comes from the inode and mtime, which a
    filesystem such as ext4 can repeat.
    """

    def make_store(self, directory: Path) -> Store[FullBackend]:
        raise NotImplementedError("Override make_store to return an empty store.")

    def test_rewrite_changes_generation(self, store: StoreAPI) -> None:
        generations = [store.put("k", value).generation for value in [VALUE, VALUE, b"x" * 10]]
        assert len(set(generations)) == 3

    def test_generations_are_not_reused_after_deletes(self, tmp_path: Path) -> None:
        store = self.make_store(tmp_path)
        seen: set[Generation] = set()
        for i in range(200):
            generation = store.put(f"k{i % 3}", VALUE).generation
            assert generation is not None
            assert generation not in seen
            seen.add(generation)
            store.delete(f"k{(i + 1) % 3}")


class ConditionalPutSuite:
    """For backends that support `put(if_match=...)`."""

    @pytest.mark.parametrize("case", IF_MATCH_PUT_CASES, ids=repr)
    def test_put_if_match(self, store: StoreAPI, case: PutCase) -> None:
        _check_put(store, case)


class NoConditionalPutSuite:
    """For backends that refuse `put(if_match=...)`."""

    def test_put_if_match_raises(self, store: StoreAPI) -> None:
        current = store.put("k", VALUE).generation
        assert current is not None
        with pytest.raises(UnsupportedOperationError):
            store.put("k", b"new", if_match=current)


class ConditionalDeleteSuite:
    """For backends that support `delete(if_match=...)`."""

    @pytest.mark.parametrize("case", IF_MATCH_DELETE_CASES, ids=repr)
    def test_delete_if_match(self, store: StoreAPI, case: DeleteCase) -> None:
        _check_delete(store, case)


def _race_put(
    store: Store[FullBackend],
    generation: Generation,
    barrier: threading.Barrier,
    results: list[str],
) -> None:
    barrier.wait()
    if store.put("d/k", b"new", if_match=generation).applied:
        results.append("put")


def _race_delete(
    store: Store[FullBackend],
    generation: Generation,
    barrier: threading.Barrier,
    results: list[str],
) -> None:
    barrier.wait()
    if store.delete("d/k", if_match=generation).applied:
        results.append("delete")


class ConditionalRaceSuite:
    """For backends with conditional puts and deletes: racing ones from one generation apply once."""

    def make_store(self, directory: Path) -> Store[FullBackend]:
        raise NotImplementedError("Override make_store to return an empty store.")

    def test_conditional_put_and_delete_apply_once(self, tmp_path: Path) -> None:
        store = self.make_store(tmp_path)
        for _ in range(100):
            generation = store.put("d/k", VALUE).generation
            assert generation is not None
            barrier = threading.Barrier(4)
            results: list[str] = []
            threads = [
                threading.Thread(target=target, args=(store, generation, barrier, results))
                for target in (_race_put, _race_delete, _race_put, _race_delete)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert len(results) == 1


class NoConditionalDeleteSuite:
    """For backends that refuse `delete(if_match=...)`."""

    def test_delete_if_match_raises(self, store: StoreAPI) -> None:
        current = store.put("k", VALUE).generation
        assert current is not None
        with pytest.raises(UnsupportedOperationError):
            store.delete("k", if_match=current)


class KeyAndDirectorySuite:
    """For backends that hold a key `a` and a key `a/b` at once (object stores, memory)."""

    def make_store(self, directory: Path) -> Store[FullBackend]:
        raise NotImplementedError("Override make_store to return an empty store.")

    def test_list_dir_reports_a_name_that_is_both_key_and_prefix(self, store: StoreAPI) -> None:
        store.put("a/b", VALUE)
        store.put("a", VALUE)
        assert set(store.list_dir()) == {Key("a"), Prefix("a")}
        assert set(store.list()) == {"a", "a/b"}

    def test_delete_dir_keeps_a_key_with_the_directory_name(self, store: StoreAPI) -> None:
        store.put("a", VALUE)
        store.put("a/b", VALUE)
        store.delete_dir("a")
        assert set(store.list()) == {"a"}

    @given(keys=KEY_SETS, directory=LISTED_DIRECTORIES)
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.differing_executors]
    )
    def test_listing_with_key_directories_matches_a_model(
        self, tmp_path_factory: pytest.TempPathFactory, keys: set[str], directory: str
    ) -> None:
        _check_listing(self.make_store(tmp_path_factory.mktemp("store")), keys, directory)


class KeyDirectoryConflictSuite:
    """For backends that can't hold a key and a directory with one name (filesystems)."""

    def test_key_over_a_directory_raises(self, store: StoreAPI) -> None:
        store.put("a/b", VALUE)
        with pytest.raises(KeyConflictError):
            store.put("a", VALUE)

    def test_directory_under_a_key_raises(self, store: StoreAPI) -> None:
        store.put("a", VALUE)
        with pytest.raises(KeyConflictError):
            store.put("a/b", VALUE)

    def test_reads_under_a_key_find_nothing(self, store: StoreAPI) -> None:
        store.put("a", VALUE)
        assert store.get("a/b") is None
        assert store.get_into("a/b", bytearray(len(VALUE))) is None
        assert store.head("a/b") is None
        assert not store.exists("a/b")
        assert store.get_ranges("a/b", [SuffixRequest(1)]) is None
        assert not store.delete("a/b").applied
        assert set(store.list("a")) == set()
        assert set(store.list_dir("a")) == set()
        assert set(store.list_dir("a/b")) == set()


class PickleSuite:
    """For backends whose stores can be pickled."""

    @pytest.mark.parametrize(("root", "read_only"), [("", False), ("a", True), ("a/b", False)])
    def test_pickle_round_trip(self, store: StoreAPI, root: str, read_only: bool) -> None:
        store.put("a/b/k", VALUE)
        view = unwrap((store / root).with_read_only(read_only))
        restored = pickle.loads(pickle.dumps(view))
        assert (restored.root, restored.read_only, restored.closed) == (root, read_only, False)
        assert set(restored.list()) == set(view.list())

    def test_pickle_keeps_closed(self, store: StoreAPI) -> None:
        store.close()
        assert pickle.loads(pickle.dumps(unwrap(store))).closed
