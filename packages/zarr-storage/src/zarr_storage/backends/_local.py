"""A backend over a directory on the local filesystem."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import shutil
import stat
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypedDict, TypeVar

from zarr_storage._batch import run_many, run_many_async
from zarr_storage._buffers import byte_view
from zarr_storage._byte_range import resolve
from zarr_storage._errors import (
    InvalidKeyError,
    KeyConflictError,
    StoreClosedError,
    UnsupportedOperationError,
)
from zarr_storage._types import (
    Copied,
    Deleted,
    Found,
    FoundRanges,
    Generation,
    Key,
    Missing,
    ObjectMetadata,
    Prefix,
    Unchanged,
    Written,
)
from zarr_storage.backends._common import IfMatch, always, if_absent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Generator, Iterator, Sequence
    from io import BufferedReader

    from typing_extensions import Buffer

    from zarr_storage._types import ByteRange, ListItem, Rejected

__all__ = ["RESERVED_SUFFIX", "LocalBackend", "segment_problem"]

RESERVED_SUFFIX = ".__zarr_local"
"""File names ending with this suffix belong to the backend: temporary files and lock files.

Keys with a segment ending in it are refused, and listings skip such names, as TensorStore's file
driver does for its own lock files.
"""

_LOCK_NAME = f"lock{RESERVED_SUFFIX}"
_STAGING_NAME = f"staging{RESERVED_SUFFIX}"
"""The directory under the root where writes stage values before renaming them into place."""

_WRITE_ATTEMPTS = 64
"""How many times a write starts over after a concurrent `delete_dir` removed its directory."""

_WINDOWS = sys.platform == "win32"

_WINDOWS_DEVICE_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"{name}{i}" for name in ("COM", "LPT") for i in [*range(1, 10), "¹", "²", "³"]]
)
_WINDOWS_FORBIDDEN_CHARACTERS = frozenset('<>:"\\|?*') | frozenset(chr(i) for i in range(32))

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class _State(TypedDict):
    root: str
    concurrency: int
    closed: bool


class _Retry(Exception):
    """A concurrent `delete_dir` removed a directory a write was using; start the write over."""


_TRANSIENT_ERRNOS = (errno.ENOENT, errno.EINVAL)
"""What creating a file in, or renaming into, a directory being removed reports.

macOS reports EINVAL rather than ENOENT for a file created in a directory being removed.
"""


def _generation(st: os.stat_result) -> Generation:
    return Generation((st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size))


_stamp_lock = threading.Lock()
_last_stamp = 0


def _unique_stamp() -> int:
    """A nanosecond time, at least now, and later than any this process has handed out.

    Every write stamps its file's mtime with one. A filesystem can reuse an inode as soon as a file
    is deleted (ext4 does) and only advances mtimes once per clock tick, so without the stamp a
    generation `(dev, inode, mtime_ns, size)` could come back for a different value, and a stale
    `if_match` would succeed.
    """
    global _last_stamp
    with _stamp_lock:
        _last_stamp = max(time.time_ns(), _last_stamp + 1)
        return _last_stamp


def _set_mtime(fd: int, path: Path, stamp: int) -> None:
    if os.utime in os.supports_fd:
        os.utime(fd, ns=(stamp, stamp))
    else:
        os.utime(path, ns=(stamp, stamp))


_PROBE_STAMP = 1_700_000_000_123_456_789
"""An mtime with every digit significant, to see whether a filesystem keeps nanoseconds."""


def segment_problem(segment: str, *, windows: bool) -> str | None:
    """Why a filesystem can't hold `segment` as one file or directory name, or `None` if it can.

    On every platform the backend's reserved suffix is refused. With `windows=True`, so are the
    names Windows reads as something else: separators, `:` (a drive or stream), other forbidden
    characters, device names such as `NUL` or `com1.txt`, and names ending in `.` or a space, which
    Windows strips.
    """
    if segment.endswith(RESERVED_SUFFIX):
        return f"names ending in {RESERVED_SUFFIX!r} are reserved by the local backend"
    if not windows:
        return None
    if not _WINDOWS_FORBIDDEN_CHARACTERS.isdisjoint(segment):
        return 'Windows forbids the characters <>:"|?*\\ and control characters in names'
    if segment.split(".")[0].upper() in _WINDOWS_DEVICE_NAMES:
        return "Windows reserves device names such as NUL, CON and COM1"
    if segment.endswith((".", " ")):
        return "Windows strips a trailing '.' or space from names"
    return None


def _check_segments(path: str) -> None:
    for segment in path.split("/"):
        problem = segment_problem(segment, windows=_WINDOWS)
        if problem is not None:
            raise InvalidKeyError(f"{path!r} can't be stored on this filesystem: {problem}.")


@contextlib.contextmanager
def _key_errors(path: str) -> Generator[None, None, None]:
    """Report a name the filesystem can't hold as an invalid key."""
    try:
        yield
    except OSError as e:
        if e.errno == errno.ENAMETOOLONG:
            raise InvalidKeyError(f"{path!r} is too long for this filesystem.") from e
        raise


def _open_key(file: Path) -> BufferedReader | None:
    """The file at `file` opened for reading, or `None` if it is absent or is a directory."""
    try:
        return open(file, "rb")
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None
    except PermissionError:
        # Windows refuses to open a directory with PermissionError.
        if file.is_dir():
            return None
        raise


def _read(f: BufferedReader, st: os.stat_result, byte_range: ByteRange | None) -> Found:
    """The part of the open file `f`, whose stat is `st`, that `byte_range` selects."""
    start, end = resolve(byte_range, st.st_size)
    f.seek(start)
    return Found(memoryview(f.read(end - start)), _generation(st), time.time())


def _read_exact_into(f: BufferedReader, out: memoryview) -> None:
    filled = 0
    while filled < len(out):
        n = f.readinto(out[filled:])
        if n == 0:
            raise OSError("The file shrank while it was being read.")
        filled += n


def _stat_key(file: Path) -> os.stat_result | None:
    """The file's stat, or `None` if it is absent or is not a regular file."""
    try:
        st = file.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    return st if stat.S_ISREG(st.st_mode) else None


@contextlib.contextmanager
def _directory_lock(directory: Path) -> Generator[None, None, None]:
    """Hold the exclusive lock of one directory.

    Every write and delete of a key holds the lock of the key's directory while it checks
    preconditions and renames or unlinks. The lock lives in the directory itself, so two backends
    whose roots overlap use the same lock for the same key. `delete_dir` removes a directory only
    while holding its lock; a waiter that wakes holding the removed lock file tries again.

    Locks are taken ancestor before descendant, never the other way, so they can't deadlock.

    Raises `_Retry` if the directory is being removed.
    """
    lock_path = directory / _LOCK_NAME
    while True:
        try:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        except NotADirectoryError as e:
            # The directory was removed and a key of the same name written in its place.
            raise _Retry from e
        except OSError as e:
            if e.errno in _TRANSIENT_ERRNOS:
                raise _Retry from e
            raise
        try:
            _lock(fd)
            held = os.fstat(fd)
            try:
                current = os.stat(lock_path)
            except (FileNotFoundError, NotADirectoryError):
                current = None
            if current is not None and (current.st_dev, current.st_ino) == (
                held.st_dev,
                held.st_ino,
            ):
                break
            _unlock(fd)
        except BaseException:
            os.close(fd)
            raise
        os.close(fd)
    try:
        yield
    finally:
        _unlock(fd)
        os.close(fd)


def _replace_if(
    file: Path,
    path: str,
    temp: Path,
    staged: Generation,
    check: Callable[[Generation | None], R | None],
) -> Written | R:
    """Rename `temp` to `file` if `check` lets it. The caller holds the directory's lock.

    `staged` is the generation of `temp`; a rename keeps the inode, mtime and size, so it is the
    generation of the written key too. A directory in the way is a conflict, whatever the
    preconditions, unless it holds no keys; then it gives way.
    """
    if _is_real_directory(file) and not _reclaim(file):
        raise KeyConflictError(f"{path!r} is a directory, so it can't also be a key.")
    st = _stat_key(file)
    refusal = check(None if st is None else _generation(st))
    if refusal is not None:
        return refusal
    try:
        os.replace(temp, file)
    except (IsADirectoryError, PermissionError) as e:
        if _is_real_directory(file):
            # Deletes leave directories behind; one that holds no keys gives way to the key.
            if not _reclaim(file):
                raise KeyConflictError(
                    f"{path!r} is a directory, so it can't also be a key."
                ) from e
            raise _Retry from e
        if isinstance(e, IsADirectoryError):
            raise _Retry from e  # The directory in the way vanished meanwhile.
        raise
    except OSError as e:
        if e.errno in _TRANSIENT_ERRNOS:
            raise _Retry from e
        raise
    return Written(staged)


def _is_real_directory(path: Path) -> bool:
    """Whether `path` is a directory itself, not a symlink to one (a symlink is a key's file)."""
    return path.is_dir() and not path.is_symlink()


def _reclaim(directory: Path) -> bool:
    """Remove `directory` if no key lies under it, and say whether it is gone.

    Only empty directories are removed, with `rmdir`, deepest first, each under its own lock
    (ancestor before descendant). A file anywhere stops it, and so does a writer adding one meanwhile:
    `rmdir` refuses a directory that isn't empty, so a reclaim can never remove a key. Another
    operation can recreate a directory's lock file between its removal and the `rmdir`; then the
    directory is scanned again.
    """
    for _ in range(_WRITE_ATTEMPTS):
        try:
            with _directory_lock(directory):
                with os.scandir(directory) as entries:
                    children = [e for e in entries if e.name != _LOCK_NAME]
                for entry in children:
                    if not entry.is_dir(follow_symlinks=False) or entry.name.endswith(
                        RESERVED_SUFFIX
                    ):
                        return False
                    if not _reclaim(Path(entry.path)):
                        return False
                os.unlink(directory / _LOCK_NAME)
                os.rmdir(directory)
        except (_Retry, FileNotFoundError):
            return True
        except OSError as e:
            if e.errno == errno.ENOTEMPTY:
                continue
            raise
        return True
    return False


def _remove_tree(directory: Path, *, remove_self: bool) -> None:
    """Delete every key under `directory`, and its subdirectories, each under its own lock.

    The directory's lock is held while its children are removed, and a child's lock is taken while
    the parent's is held (ancestor before descendant), so a writer holding a directory's lock always
    finishes before the directory goes. With `remove_self`, `directory` is removed too, unless a
    concurrent writer has added to it meanwhile; directories aren't keys, so that is not an error.
    """
    try:
        with _directory_lock(directory):
            with os.scandir(directory) as entries:
                children = [e for e in entries if e.name != _LOCK_NAME]
            for entry in children:
                if entry.name.endswith(RESERVED_SUFFIX):
                    # Below the root, this is a nested backend's staging directory; at the root,
                    # it is this backend's own and stays.
                    if remove_self:
                        shutil.rmtree(entry.path, ignore_errors=True)
                    continue
                if entry.is_dir(follow_symlinks=False):
                    _remove_tree(Path(entry.path), remove_self=True)
                else:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(entry.path)
            if remove_self:
                os.unlink(directory / _LOCK_NAME)
                try:
                    os.rmdir(directory)
                except OSError as e:
                    if e.errno != errno.ENOTEMPTY:
                        raise
    except (_Retry, FileNotFoundError, NotADirectoryError):
        return  # The directory is already gone.


class LocalBackend:
    """Keys stored as files under a root directory.

    A write goes to a temporary file in a staging directory under the root and is renamed into
    place, so readers see the old value or the new one, never part of either. The root must not
    span filesystems (no mount points under it), since a rename can't cross one. Writes are not
    synced to disk.

    Writes and deletes hold the lock of the key's directory while they check preconditions and
    rename, which makes conditional writes atomic across threads, processes and backends whose
    roots overlap, on one machine.

    Deleting a key leaves its directory in place, so `list_dir` can report a `Prefix` that holds no
    keys; a later write of a key with that directory's name removes it if no key lies under it. `delete_dir` removes the
    tree bottom-up, taking each directory's lock first, so a write in progress finishes before its
    directory goes; it is not atomic, and a key written while it runs may survive it. A write whose
    directory a concurrent `delete_dir` removes starts over, with a short backoff, up to 64 times.

    Temporary files and lock files end in `RESERVED_SUFFIX` and are never listed. A crash can leave
    staged files behind in the staging directory; they are never read.

    Key names follow the filesystem: on case-insensitive or Unicode-normalizing filesystems (the
    defaults on macOS and Windows), keys that differ only in case or normalization are the same key.

    Parameters
    ----------
    root
        The directory that holds the keys.
    create
        Create `root` if it doesn't exist. With `create=False`, a missing root raises
        `FileNotFoundError`.
    concurrency
        The number of threads that run the async methods' file operations, and the number of
        operations a batch (`get_many`, `put_many`, ...) runs at once.
    """

    def __init__(
        self, root: str | os.PathLike[str], *, create: bool = True, concurrency: int = 16
    ) -> None:
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1; got {concurrency}.")
        self._root = Path(root).resolve()
        if create:
            self._root.mkdir(parents=True, exist_ok=True)
        elif not self._root.is_dir():
            raise FileNotFoundError(f"The root directory {str(self._root)!r} does not exist.")
        self._concurrency = concurrency
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._exact_times: bool | None = None
        self._closed = False

    @property
    def root(self) -> Path:
        """The directory that holds the keys."""
        return self._root

    def __repr__(self) -> str:
        return f"LocalBackend({str(self._root)!r})"

    def __getstate__(self) -> _State:
        return {"root": str(self._root), "concurrency": self._concurrency, "closed": self._closed}

    def __setstate__(self, state: _State) -> None:
        self._root = Path(state["root"])
        self._concurrency = state["concurrency"]
        self._executor = None
        self._executor_lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._exact_times = None
        self._closed = state["closed"]

    # --- Backend

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def writable(self) -> bool:
        return True

    def close(self) -> None:
        with self._executor_lock:
            self._closed = True
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)

    # --- paths

    def _file(self, path: str) -> Path:
        _check_segments(path)
        return self._root / path

    def _directory(self, directory: str) -> Path:
        if directory == "":
            return self._root
        _check_segments(directory)
        return self._root / directory

    # --- Readable

    def get(self, path: str, byte_range: ByteRange | None) -> Found | Missing:
        with _key_errors(path):
            f = _open_key(self._file(path))
        if f is None:
            return Missing(time.time())
        with f:
            return _read(f, os.fstat(f.fileno()), byte_range)

    def get_if_changed(
        self, path: str, generation: Generation, byte_range: ByteRange | None
    ) -> Found | Missing | Unchanged:
        with _key_errors(path):
            f = _open_key(self._file(path))
        if f is None:
            return Missing(time.time())
        with f:
            st = os.fstat(f.fileno())
            if _generation(st) == generation:
                return Unchanged(generation, time.time())
            return _read(f, st, byte_range)

    def get_ranges(self, path: str, ranges: Sequence[ByteRange]) -> FoundRanges | Missing:
        with _key_errors(path):
            f = _open_key(self._file(path))
        if f is None:
            return Missing(time.time())
        parts: list[memoryview] = []
        with f:
            st = os.fstat(f.fileno())
            for byte_range in ranges:
                start, end = resolve(byte_range, st.st_size)
                f.seek(start)
                parts.append(memoryview(f.read(end - start)))
        return FoundRanges(tuple(parts), _generation(st), time.time())

    def get_into(self, path: str, out: Buffer, byte_range: ByteRange | None) -> Copied | Missing:
        view = byte_view(out, writable=True)
        with _key_errors(path):
            f = _open_key(self._file(path))
        if f is None:
            return Missing(time.time())
        with f:
            st = os.fstat(f.fileno())
            start, end = resolve(byte_range, st.st_size)
            n = end - start
            if n > len(view):
                raise ValueError(f"The output buffer holds {len(view)} bytes; the value has {n}.")
            f.seek(start)
            _read_exact_into(f, view[:n])
        return Copied(n, _generation(st), time.time())

    def head(self, path: str) -> ObjectMetadata | Missing:
        with _key_errors(path):
            st = _stat_key(self._file(path))
        if st is None:
            return Missing(time.time())
        return ObjectMetadata(st.st_size, _generation(st), st.st_mtime)

    # --- Writable

    def put(self, path: str, value: memoryview) -> Written:
        return self._put(path, value, always)

    def put_if_absent(self, path: str, value: memoryview) -> Written | Rejected:
        return self._put(path, value, if_absent)

    def put_if_match(
        self, path: str, value: memoryview, generation: Generation
    ) -> Written | Rejected | Missing:
        self._require_exact_times()
        return self._put(path, value, IfMatch(generation))

    def _put(
        self, path: str, value: memoryview, check: Callable[[Generation | None], R | None]
    ) -> Written | R:
        file = self._file(path)
        last = _Retry()
        with _key_errors(path):
            for attempt in range(_WRITE_ATTEMPTS):
                temp: Path | None = None
                try:
                    temp, staged = self._stage(value)
                    return self._commit(file, path, temp, staged, check)
                except _Retry as retry:
                    last = retry
                    time.sleep(min(1e-4 * 2**attempt, 1e-2))
                finally:
                    if temp is not None:
                        with contextlib.suppress(OSError):
                            temp.unlink(missing_ok=True)
        raise OSError(
            f"Gave up writing {path!r}: a concurrent delete kept removing its directory."
        ) from last.__cause__

    def _require_exact_times(self) -> None:
        """Refuse conditional writes on a filesystem that rounds file times.

        Generations need the nanosecond stamps writes set; a filesystem that rounds them (FAT,
        HFS+, some network filesystems) could hand an old generation out again.
        """
        with self._probe_lock:
            if self._exact_times is None:
                self._exact_times = self._probe_exact_times()
        if not self._exact_times:
            raise UnsupportedOperationError(
                f"The filesystem under {str(self._root)!r} rounds file times, so generations "
                "aren't unique enough for conditional writes."
            )

    def _probe_exact_times(self) -> bool:
        staging = self._root / _STAGING_NAME
        staging.mkdir(parents=True, exist_ok=True)
        probe = staging / f"probe-{uuid.uuid4().hex}"
        try:
            with open(probe, "xb") as f:
                _set_mtime(f.fileno(), probe, _PROBE_STAMP)
                return os.fstat(f.fileno()).st_mtime_ns == _PROBE_STAMP
        finally:
            probe.unlink(missing_ok=True)

    def _stage(self, value: memoryview) -> tuple[Path, Generation]:
        """Write `value` to a new file in the staging directory, outside every key's directory.

        Raises `_Retry` if a concurrent delete removed the root or the staging directory.
        """
        staging = self._root / _STAGING_NAME
        temp = staging / uuid.uuid4().hex
        try:
            staging.mkdir(parents=True, exist_ok=True)
            with open(temp, "xb") as f:
                f.write(value)
                f.flush()
                _set_mtime(f.fileno(), temp, _unique_stamp())
                staged = _generation(os.fstat(f.fileno()))
        except (FileExistsError, NotADirectoryError) as e:
            if self._root.exists() and not self._root.is_dir():
                raise KeyConflictError(
                    f"The root {str(self._root)!r} is now a file (a key of an enclosing store)."
                ) from e
            # `mkdir` saw the directory, then a concurrent delete removed it.
            raise _Retry from e
        except OSError as e:
            if e.errno in _TRANSIENT_ERRNOS:
                raise _Retry from e
            raise
        return temp, staged

    def _has_key_parent(self, file: Path) -> bool:
        """Whether a directory between the root and `file` is a regular file (a key)."""
        parent = file.parent
        while parent != self._root:
            if _stat_key(parent) is not None:
                return True
            parent = parent.parent
        return False

    def _commit(
        self,
        file: Path,
        path: str,
        temp: Path,
        staged: Generation,
        check: Callable[[Generation | None], R | None],
    ) -> Written | R:
        """Rename `temp` to `file` if `check` lets it, under the lock of `file`'s directory.

        Raises `_Retry` if a concurrent `delete_dir` removed the directory on the way.
        """
        try:
            file.parent.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, NotADirectoryError) as e:
            if self._has_key_parent(file):
                raise KeyConflictError(
                    f"A parent of {path!r} is a key, so it can't also be a directory."
                ) from e
            # `mkdir` saw the directory, then a concurrent delete removed it before `mkdir` checked
            # that it was one.
            raise _Retry from e
        except OSError as e:
            if e.errno in _TRANSIENT_ERRNOS:
                raise _Retry from e
            raise
        with _directory_lock(file.parent):
            return _replace_if(file, path, temp, staged, check)

    def delete(self, path: str) -> Deleted | Missing:
        return self._delete(path, always)

    def delete_if_match(self, path: str, generation: Generation) -> Deleted | Rejected | Missing:
        self._require_exact_times()
        return self._delete(path, IfMatch(generation))

    def _delete(
        self, path: str, check: Callable[[Generation | None], R | None]
    ) -> Deleted | Missing | R:
        file = self._file(path)
        with _key_errors(path):
            try:
                with _directory_lock(file.parent):
                    st = _stat_key(file)
                    if st is None:
                        return Missing(time.time())
                    current = _generation(st)
                    refusal = check(current)
                    if refusal is not None:
                        return refusal
                    file.unlink()
            except (_Retry, NotADirectoryError):
                # The key's directory doesn't exist, or a parent is a key: the key is absent.
                return Missing(time.time())
        return Deleted(current)

    def delete_dir(self, directory: str) -> None:
        target = self._directory(directory)
        with _key_errors(directory):
            if target == self._root:
                _remove_tree(self._root, remove_self=False)
            elif target.is_dir() and not target.is_symlink():
                _remove_tree(target, remove_self=True)

    # --- Listable

    def list(self, directory: str) -> Iterator[str]:
        target = self._directory(directory)
        prefix = "" if directory == "" else f"{directory}/"
        with _key_errors(directory):
            return iter(list(_walk(target, prefix)))

    def list_dir(self, directory: str) -> Iterator[ListItem]:
        target = self._directory(directory)
        with _key_errors(directory):
            try:
                entries = list(os.scandir(target))
            except (FileNotFoundError, NotADirectoryError):
                return iter(())
        return _classify(entries)

    # --- batches: the operations above, `concurrency` at once on the backend's threads

    def get_many(
        self, requests: Sequence[tuple[str, ByteRange | None]]
    ) -> Iterator[tuple[int, Found | Missing]]:
        return run_many(self.get, requests, self._pool(), self._concurrency)

    def get_into_many(
        self, requests: Sequence[tuple[str, Buffer, ByteRange | None]]
    ) -> Iterator[tuple[int, Copied | Missing]]:
        return run_many(self.get_into, requests, self._pool(), self._concurrency)

    def put_many(self, requests: Sequence[tuple[str, memoryview]]) -> Iterator[tuple[int, Written]]:
        return run_many(self.put, requests, self._pool(), self._concurrency)

    def delete_many(self, paths: Sequence[str]) -> Iterator[tuple[int, Deleted | Missing]]:
        return run_many(self.delete, [(path,) for path in paths], self._pool(), self._concurrency)

    # --- async twins: run the sync methods on the backend's threads

    def _pool(self) -> ThreadPoolExecutor:
        with self._executor_lock:
            if self._closed:
                raise StoreClosedError(f"{self!r} is closed.")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._concurrency, thread_name_prefix="zarr-storage-local"
                )
            return self._executor

    async def _run(self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool(), partial(fn, *args, **kwargs))

    async def get_async(self, path: str, byte_range: ByteRange | None) -> Found | Missing:
        return await self._run(self.get, path, byte_range)

    async def get_if_changed_async(
        self, path: str, generation: Generation, byte_range: ByteRange | None
    ) -> Found | Missing | Unchanged:
        return await self._run(self.get_if_changed, path, generation, byte_range)

    async def get_ranges_async(
        self, path: str, ranges: Sequence[ByteRange]
    ) -> FoundRanges | Missing:
        return await self._run(self.get_ranges, path, ranges)

    async def head_async(self, path: str) -> ObjectMetadata | Missing:
        return await self._run(self.head, path)

    async def get_into_async(
        self, path: str, out: Buffer, byte_range: ByteRange | None
    ) -> Copied | Missing:
        return await self._run(self.get_into, path, out, byte_range)

    async def put_async(self, path: str, value: memoryview) -> Written:
        return await self._run(self.put, path, value)

    async def put_if_absent_async(self, path: str, value: memoryview) -> Written | Rejected:
        return await self._run(self.put_if_absent, path, value)

    async def put_if_match_async(
        self, path: str, value: memoryview, generation: Generation
    ) -> Written | Rejected | Missing:
        return await self._run(self.put_if_match, path, value, generation)

    async def delete_async(self, path: str) -> Deleted | Missing:
        return await self._run(self.delete, path)

    async def delete_if_match_async(
        self, path: str, generation: Generation
    ) -> Deleted | Rejected | Missing:
        return await self._run(self.delete_if_match, path, generation)

    async def delete_dir_async(self, directory: str) -> None:
        await self._run(self.delete_dir, directory)

    def get_many_async(
        self, requests: Sequence[tuple[str, ByteRange | None]]
    ) -> AsyncIterator[tuple[int, Found | Missing]]:
        return run_many_async(self.get_async, requests, self._concurrency)

    def get_into_many_async(
        self, requests: Sequence[tuple[str, Buffer, ByteRange | None]]
    ) -> AsyncIterator[tuple[int, Copied | Missing]]:
        return run_many_async(self.get_into_async, requests, self._concurrency)

    def put_many_async(
        self, requests: Sequence[tuple[str, memoryview]]
    ) -> AsyncIterator[tuple[int, Written]]:
        return run_many_async(self.put_async, requests, self._concurrency)

    def delete_many_async(
        self, paths: Sequence[str]
    ) -> AsyncIterator[tuple[int, Deleted | Missing]]:
        return run_many_async(self.delete_async, [(path,) for path in paths], self._concurrency)

    async def list_async(self, directory: str) -> AsyncIterator[str]:
        for path in await self._run(_materialize, self.list(directory)):
            yield path

    async def list_dir_async(self, directory: str) -> AsyncIterator[ListItem]:
        for item in await self._run(_materialize, self.list_dir(directory)):
            yield item


def _materialize(items: Iterator[T]) -> list[T]:
    return list(items)


def _walk(directory: Path, prefix: str) -> Iterator[str]:
    try:
        entries = list(os.scandir(directory))
    except (FileNotFoundError, NotADirectoryError):
        return
    for entry in entries:
        if entry.name.endswith(RESERVED_SUFFIX):
            continue
        if entry.is_dir(follow_symlinks=False):
            yield from _walk(Path(entry.path), f"{prefix}{entry.name}/")
        else:
            yield f"{prefix}{entry.name}"


def _classify(entries: list[os.DirEntry[str]]) -> Iterator[ListItem]:
    for entry in entries:
        if entry.name.endswith(RESERVED_SUFFIX):
            continue
        if entry.is_dir(follow_symlinks=False):
            yield Prefix(entry.name)
        else:
            yield Key(entry.name)


if sys.platform == "win32":
    import msvcrt

    def _lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            except OSError:
                continue
            return

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
