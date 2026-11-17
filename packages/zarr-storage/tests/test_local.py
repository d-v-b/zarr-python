from __future__ import annotations

import errno
import multiprocessing
import os
import threading
import time
from typing import TYPE_CHECKING

import pytest

from zarr_storage import (
    Generation,
    InvalidKeyError,
    Key,
    KeyConflictError,
    Prefix,
    Store,
    UnsupportedOperationError,
)
from zarr_storage.backends import LocalBackend
from zarr_storage.backends._local import RESERVED_SUFFIX, _directory_lock, segment_problem
from zarr_storage.testing import (
    ConditionalDeleteSuite,
    ConditionalPutSuite,
    ConditionalRaceSuite,
    KeyDirectoryConflictSuite,
    PickleSuite,
    StoreSuite,
    UniqueGenerationsSuite,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class TestLocalBackend(
    StoreSuite,
    UniqueGenerationsSuite,
    ConditionalPutSuite,
    ConditionalDeleteSuite,
    ConditionalRaceSuite,
    KeyDirectoryConflictSuite,
    PickleSuite,
):
    def make_store(self, directory: Path) -> Store[LocalBackend]:
        return Store(LocalBackend(directory))


def test_listing_skips_backend_files(tmp_path: Path) -> None:
    """Lock files and temporary files are never listed."""
    store = Store(LocalBackend(tmp_path))
    store.put("a/k", b"x")
    (tmp_path / "a" / f"k.123{RESERVED_SUFFIX}").write_bytes(b"partial")
    assert (tmp_path / "a" / f"lock{RESERVED_SUFFIX}").exists()
    assert set(store.list()) == {"a/k"}
    assert set(store.list_dir("a")) == {Key("k")}


def test_delete_leaves_the_directory(tmp_path: Path) -> None:
    """Deleting a key leaves its directory, which `list_dir` reports as an empty `Prefix`."""
    store = Store(LocalBackend(tmp_path))
    store.put("a/b/c", b"x")
    store.delete("a/b/c")
    assert set(store.list()) == set()
    assert set(store.list_dir("a")) == {Prefix("b")}


def test_failed_put_leaves_no_staged_files(tmp_path: Path) -> None:
    store = Store(LocalBackend(tmp_path))
    store.put("k", b"x")
    assert not store.put("k", b"y", if_none_match=True).applied
    assert list((tmp_path / f"staging{RESERVED_SUFFIX}").iterdir()) == []


def test_delete_dir_of_the_root_keeps_backend_files(tmp_path: Path) -> None:
    store = Store(LocalBackend(tmp_path))
    store.put("k", b"x")
    store.delete_dir("")
    assert {p.name for p in tmp_path.iterdir()} == {
        f"lock{RESERVED_SUFFIX}",
        f"staging{RESERVED_SUFFIX}",
    }
    assert store.put("k", b"y", if_none_match=True).applied


def test_put_recreates_a_removed_root(tmp_path: Path) -> None:
    """An outer backend can remove a nested backend's root; the nested one recreates it."""
    outer = Store(LocalBackend(tmp_path))
    inner = Store(LocalBackend(tmp_path / "sub"))
    outer.delete_dir("sub")
    assert inner.put("k", b"x").applied
    assert outer.exists("sub/k")


@pytest.mark.parametrize(
    ("segment", "windows", "problem"),
    [
        ("a", False, None),
        ("a", True, None),
        ("a.b", True, None),
        ("con", False, None),
        ("a:b", False, None),
        ("a\\b", False, None),
        (f"a{RESERVED_SUFFIX}", False, "reserved"),
        (f"a{RESERVED_SUFFIX}", True, "reserved"),
        ("a:b", True, "forbids"),
        ("a\\b", True, "forbids"),
        ("a?", True, "forbids"),
        ("a\x01", True, "forbids"),
        ("NUL", True, "device"),
        ("con", True, "device"),
        ("com1.txt", True, "device"),
        ("LPT9", True, "device"),
        ("COM¹", True, "device"),
        ("a.", True, "strips"),
        ("a ", True, "strips"),
    ],
)
def test_segment_problem(segment: str, windows: bool, problem: str | None) -> None:
    found = segment_problem(segment, windows=windows)
    if problem is None:
        assert found is None
    else:
        assert found is not None
        assert problem in found


def test_missing_root_without_create_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LocalBackend(tmp_path / "missing", create=False)


def test_reserved_key_raises(tmp_path: Path) -> None:
    with pytest.raises(InvalidKeyError, match="reserved"):
        Store(LocalBackend(tmp_path)).put(f"a{RESERVED_SUFFIX}/b", b"x")


def test_overlong_key_raises(tmp_path: Path) -> None:
    with pytest.raises(InvalidKeyError, match="too long"):
        Store(LocalBackend(tmp_path)).put("a" * 1000, b"x")


def test_zero_concurrency_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        LocalBackend(tmp_path, concurrency=0)


def _create(
    store: Store[LocalBackend], key: str, barrier: threading.Barrier, results: list[bool]
) -> None:
    barrier.wait()
    results.append(store.put(key, b"x", if_none_match=True).applied)


@pytest.mark.parametrize("overlapping_roots", [False, True])
def test_concurrent_creates_through_overlapping_roots(
    tmp_path: Path, overlapping_roots: bool
) -> None:
    """Of many racing `if_none_match` writes to one key, exactly one applies, even through
    backends whose roots overlap."""
    outer = Store(LocalBackend(tmp_path))
    inner = Store(LocalBackend(tmp_path / "sub")) if overlapping_roots else outer / "sub"
    for round_ in range(50):
        key = f"k{round_}"
        barrier = threading.Barrier(8)
        results: list[bool] = []
        threads = [
            threading.Thread(target=_create, args=(store, path, barrier, results))
            for store, path in [(outer, f"sub/{key}"), (inner, key)] * 4
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results.count(True) == 1


def _create_in_process(root: str, key: str, start: threading.Event) -> bool:
    start.wait()
    return Store(LocalBackend(root)).put(key, b"x", if_none_match=True).applied


def test_concurrent_creates_apply_once_across_processes(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as manager, ctx.Pool(4) as pool:
        for round_ in range(5):
            start = manager.Event()
            pending = [
                pool.apply_async(_create_in_process, (str(tmp_path), f"k{round_}", start))
                for _ in range(4)
            ]
            start.set()
            assert [p.get(timeout=60) for p in pending].count(True) == 1


def _put_many(store: Store[LocalBackend], errors: list[BaseException]) -> None:
    try:
        for i in range(200):
            store.put(f"d/k{i}", b"x")
    except BaseException as e:  # noqa: BLE001 -- the test reports any failure
        errors.append(e)


def test_delete_dir_racing_put_never_breaks_the_put(tmp_path: Path) -> None:
    """A put whose directory a concurrent `delete_dir` removes starts over instead of failing."""
    store = Store(LocalBackend(tmp_path))
    errors: list[BaseException] = []
    writer = threading.Thread(target=_put_many, args=(store, errors))
    writer.start()
    while writer.is_alive():
        store.delete_dir("d")
        time.sleep(0.0001)
    writer.join()
    assert errors == []


def _hold_lock(directory: Path, held: threading.Event, release: threading.Event) -> None:
    with _directory_lock(directory):
        held.set()
        release.wait()


def _record_lock_file(directory: Path, entered: threading.Event, saw: list[bool]) -> None:
    with _directory_lock(directory):
        saw.append((directory / f"lock{RESERVED_SUFFIX}").exists())
        entered.set()


def test_lock_waiter_retries_when_the_lock_file_is_removed(tmp_path: Path) -> None:
    """A waiter that wakes holding a lock file someone removed takes the new lock file instead."""
    held, release, entered = threading.Event(), threading.Event(), threading.Event()
    saw: list[bool] = []
    holder = threading.Thread(target=_hold_lock, args=(tmp_path, held, release))
    holder.start()
    held.wait()
    waiter = threading.Thread(target=_record_lock_file, args=(tmp_path, entered, saw))
    waiter.start()
    time.sleep(0.1)
    assert not entered.is_set()
    (tmp_path / f"lock{RESERVED_SUFFIX}").unlink()
    release.set()
    holder.join()
    waiter.join(timeout=5)
    assert saw == [True]


class _FlakyReplace:
    """`os.replace` that fails with the given errors first, then renames."""

    def __init__(self, failures: list[OSError]) -> None:
        self._failures = failures
        self._real = os.replace

    def __call__(self, src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if self._failures:
            raise self._failures.pop()
        self._real(src, dst)


def test_put_retries_a_rename_into_a_removed_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(LocalBackend(tmp_path))
    failures = [OSError(errno.ENOENT, "removed"), OSError(errno.EINVAL, "removed")]
    monkeypatch.setattr(os, "replace", _FlakyReplace(failures))
    assert store.put("a/k", b"x").applied
    result = store.get("a/k")
    assert result is not None
    assert bytes(result.value) == b"x"


class _PausedReplace:
    """`os.replace` that, for one destination name, waits for a signal before renaming."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._real = os.replace
        self.reached = threading.Event()
        self.release = threading.Event()

    def __call__(self, src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if os.fspath(dst).endswith(self._name):
            self.reached.set()
            self.release.wait()
        self._real(src, dst)


@pytest.mark.parametrize("ancestor", ["", "P", "P/D"])
def test_delete_dir_waits_for_a_write_in_progress(
    tmp_path: Path, ancestor: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`delete_dir` of any ancestor waits for a write that holds its directory's lock, then
    removes what it wrote; a create after the delete applies."""
    store = Store(LocalBackend(tmp_path))
    store.put("P/D/other", b"x")
    paused = _PausedReplace("/k")
    monkeypatch.setattr(os, "replace", paused)
    outcomes: dict[str, bool] = {}
    barrier = threading.Barrier(1)
    writer = threading.Thread(target=_put_outcome, args=(store, "P/D/k", barrier, outcomes))
    writer.start()
    assert paused.reached.wait(5)
    deleter = threading.Thread(target=store.delete_dir, args=(ancestor,))
    deleter.start()
    time.sleep(0.2)
    assert deleter.is_alive()
    paused.release.set()
    writer.join(5)
    deleter.join(5)
    assert outcomes == {"P/D/k": True}
    assert not store.exists("P/D/k")
    assert store.put("P/D/k", b"y", if_none_match=True).applied


def test_delete_dir_leaves_a_symlinked_directory_alone(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_bytes(b"x")
    store = Store(LocalBackend(tmp_path / "root"))
    (tmp_path / "root" / "link").symlink_to(outside)
    store.delete_dir("link")
    assert (outside / "keep").exists()


def test_delete_dir_removes_a_nested_backends_staging(tmp_path: Path) -> None:
    outer = Store(LocalBackend(tmp_path))
    inner = Store(LocalBackend(tmp_path / "sub"))
    inner.put("k", b"x")
    outer.delete_dir("sub")
    assert not (tmp_path / "sub").exists()


def _put_outcome(
    store: Store[LocalBackend], key: str, barrier: threading.Barrier, outcomes: dict[str, bool]
) -> None:
    barrier.wait()
    try:
        outcomes[key] = store.put(key, b"x").applied
    except KeyConflictError:
        outcomes[key] = False


def test_reclaim_never_removes_an_acknowledged_key(tmp_path: Path) -> None:
    """A put of `a` racing a put of `a/b`, over a directory `a` a delete left empty: exactly one
    applies, and the one that applied is stored."""
    store = Store(LocalBackend(tmp_path))
    for _ in range(300):
        store.delete_dir("")
        store.put("a/x", b"x")
        store.delete("a/x")
        barrier = threading.Barrier(2)
        outcomes: dict[str, bool] = {}
        threads = [
            threading.Thread(target=_put_outcome, args=(store, key, barrier, outcomes))
            for key in ("a", "a/b")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        stored = set(store.list())
        assert sum(outcomes.values()) == 1
        assert stored == {key for key, applied in outcomes.items() if applied}


def _coarse(self: LocalBackend) -> bool:
    return False


@pytest.mark.parametrize(
    "write",
    [
        lambda s: s.put("k", b"x", if_none_match=True),
        lambda s: s.put("k", b"x", if_match=Generation(object())),
        lambda s: s.delete("k", if_match=Generation(object())),
    ],
    ids=["put-if_none_match", "put-if_match", "delete-if_match"],
)
def test_conditional_writes_on_a_filesystem_that_rounds_times_raise(
    tmp_path: Path, write: Callable[[Store[LocalBackend]], object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(LocalBackend, "_probe_exact_times", _coarse)
    store = Store(LocalBackend(tmp_path))
    with pytest.raises(UnsupportedOperationError, match="rounds file times"):
        write(store)
    assert store.put("k", b"x").applied
