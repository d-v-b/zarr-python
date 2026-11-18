"""A backend over an obstore store (S3, GCS, Azure, HTTP, local, memory).

Each operation is written once as plain logic around obstore calls; the sync method and its async
twin differ only in which obstore function they await. The decisions (which put mode, whether an
error means "absent", whether a read is consistent) live in module functions both share.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

import obstore
from obstore.exceptions import (
    AlreadyExistsError,
    GenericError,
    NotModifiedError,
    PreconditionError,
)
from obstore.store import HTTPStore, LocalStore

from zarr_storage._batch import run_many_async
from zarr_storage._buffers import byte_view, copy_into
from zarr_storage._byte_range import resolve
from zarr_storage._errors import InvalidKeyError, KeyConflictError, UnsupportedOperationError
from zarr_storage._keys import check_key, relativize
from zarr_storage._resources import Limiter, default_limiter
from zarr_storage._types import (
    Deleted,
    Found,
    FoundRanges,
    Generation,
    Key,
    Missing,
    ObjectMetadata,
    OffsetRequest,
    Prefix,
    RangeRequest,
    Rejected,
    SuffixRequest,
    Unchanged,
    Written,
)
from zarr_storage.backends._common import IfMatch, always, if_absent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence

    from obstore import GetOptions, GetResult, ListResult, ListStream, ObjectMeta, PutMode
    from obstore.store import ObjectStore
    from typing_extensions import Buffer

    from zarr_storage._types import ByteRange, Copied, ListItem

__all__ = ["ObjectStoreBackend"]

R = TypeVar("R")

_EMPTY = memoryview(b"")

_NOT_A_DIRECTORY = f"(os error {errno.ENOTDIR})"
"""How a filesystem store's refusal reads when a parent of the path is a file."""

_PUT_ATTEMPTS = 16
"""How many times a write starts over when the store refused it for a state that then went away."""


# ---------------------------------------------------------------------------- generations


def _generation(e_tag: str | None, version: str | None) -> Generation:
    return Generation(("obstore", e_tag, version))


def _metadata(meta: ObjectMeta) -> ObjectMetadata:
    return ObjectMetadata(
        meta["size"],
        _generation(meta["e_tag"], meta["version"]),
        meta["last_modified"].timestamp(),
    )


def _version_of(generation: object) -> tuple[str | None, str | None] | None:
    """The `(e_tag, version)` in a generation this backend minted, or `None` for any other."""
    match generation:
        case ("obstore", str() | None as e_tag, str() | None as version):
            return e_tag, version
        case _:
            return None


def _if_match_mode(generation: Generation) -> PutMode | None:
    """The put mode that writes only over `generation`; `None` for a generation this backend did
    not mint, which no stored value has."""
    version = _version_of(generation)
    if version is None:
        return None
    e_tag, object_version = version
    if e_tag is None and object_version is None:
        raise UnsupportedOperationError(
            "The store reported neither an ETag nor a version, so writes can't be conditional."
        )
    mode = {k: v for k, v in (("e_tag", e_tag), ("version", object_version)) if v is not None}
    return cast("PutMode", mode)


def _refuse(current: ObjectMetadata | Missing) -> Rejected | Missing:
    """The outcome of a write conditioned on a generation this backend did not mint."""
    return current if isinstance(current, Missing) else Rejected(current.generation)


def _found(result: GetResult, data: memoryview) -> Found:
    return Found(data, _generation(result.meta["e_tag"], result.meta["version"]), time.time())


def _if_none_match(generation: Generation) -> GetOptions:
    """Read options that skip the read if the value has `generation`, where the store can tell."""
    version = _version_of(generation)
    if version is None or version[0] is None:
        return {}
    return {"if_none_match": version[0]}


# ---------------------------------------------------------------------------- reads


def _get_options(byte_range: ByteRange | None) -> GetOptions:
    match byte_range:
        case None:
            return {}
        case RangeRequest(start=start, end=end):
            return {"range": (start, end)}
        case OffsetRequest(start=start):
            return {"range": {"offset": start}}
        case SuffixRequest(length=length):
            return {"range": {"suffix": length}}


def _is_empty_request(byte_range: ByteRange | None) -> bool:
    """Whether the range selects no bytes regardless of the value (object stores refuse those)."""
    match byte_range:
        case RangeRequest(start=start, end=end):
            return start == end
        case SuffixRequest(length=0):
            return True
        case _:
            return False


def _empty_read_or_raise(
    meta: ObjectMetadata | Missing, byte_range: ByteRange | None, error: BaseException | None
) -> Found | Missing:
    """The result of a read the object store refused or that selects nothing.

    Object stores refuse a range that starts at the end of the value; the contract reads that as
    empty. A range past the end raises `InvalidByteRangeError` (from `resolve`). Any other refusal
    re-raises the store's error.
    """
    if isinstance(meta, Missing):
        return meta
    start, end = resolve(byte_range, meta.size)
    if start == end:
        return Found(_EMPTY, meta.generation, time.time())
    if error is not None:
        raise error
    raise AssertionError("A non-empty range was treated as empty.")


def _pinned(generation: Generation) -> GetOptions | None:
    """Read options that make a read fail unless the value still has `generation`.

    `None` when the store reported neither an ETag nor a version, so reads can't be pinned.
    """
    version = _version_of(generation)
    if version is None:
        return None
    e_tag, object_version = version
    if e_tag is not None:
        return {"if_match": e_tag}
    if object_version is not None:
        return {"version": object_version}
    return None


def _slices(data: memoryview, ranges: Sequence[ByteRange]) -> tuple[memoryview, ...]:
    return tuple(data[slice(*resolve(r, len(data)))] for r in ranges)


# ---------------------------------------------------------------------------- listing

# obstore's `list` overloads mention Arrow types from an optional dependency, which pyright can't
# resolve without it; these wrappers pin the non-Arrow overload in one place.


def _list(store: ObjectStore, prefix: str | None) -> ListStream[Sequence[ObjectMeta]]:
    return obstore.list(store, prefix, return_arrow=False)  # pyright: ignore[reportUnknownMemberType]


def _list_with_delimiter(
    store: ObjectStore, prefix: str | None
) -> ListResult[Sequence[ObjectMeta]]:
    return obstore.list_with_delimiter(store, prefix, return_arrow=False)  # pyright: ignore[reportUnknownMemberType]


async def _list_with_delimiter_async(
    store: ObjectStore, prefix: str | None
) -> ListResult[Sequence[ObjectMeta]]:
    return await obstore.list_with_delimiter_async(store, prefix, return_arrow=False)  # pyright: ignore[reportUnknownMemberType]


def _prefix(directory: str) -> str | None:
    return None if directory == "" else directory


def _valid_keys(paths: Iterable[str], directory: str) -> Iterator[str]:
    """The paths that follow the key grammar and lie strictly under `directory`."""
    for path in paths:
        if path == directory:
            continue
        try:
            yield check_key(path)
        except InvalidKeyError:
            continue


def _children(
    objects: Iterable[ObjectMeta], prefixes: Iterable[str], directory: str
) -> list[ListItem]:
    items: list[ListItem] = [
        Key(relativize(path, directory))
        for path in _valid_keys((m["path"] for m in objects), directory)
    ]
    items.extend(Prefix(relativize(p.removesuffix("/"), directory)) for p in prefixes)
    return items


def _ancestors(path: str) -> list[str]:
    """`path`'s parent directories, nearest last: `a/b/c` gives `a`, `a/b`."""
    segments = path.split("/")
    return ["/".join(segments[:i]) for i in range(1, len(segments))]


class ObjectStoreBackend:
    """Keys stored in an obstore store.

    Generations are the store's ETag and version. `put_if_match` needs a store that supports it (S3,
    GCS, Azure and obstore's `MemoryStore` do; obstore's `LocalStore` doesn't); otherwise it raises
    `UnsupportedOperationError`. Object stores can't delete conditionally, so this backend has no
    `delete_if_match`. `get_if_changed` sends `If-None-Match` with the generation's ETag.

    `get_ranges` reads the first range, then the others pinned (`If-Match`) to the generation and
    size that first response reported. If the value changed in between, or the store reports
    neither ETags nor versions, it reads the value once, whole, and slices it.

    obstore's `LocalStore` refuses paths that lie under a file; for it alone, such a refusal reads
    as an absent key. Every other store's errors are raised. obstore's `LocalStore` has no locking,
    so concurrent writes that create and remove the same directories can still surface its
    errors; `LocalBackend` is the backend for that.

    Parameters
    ----------
    store
        The obstore store.
    writable
        Whether to accept writes. Defaults to `False` for `HTTPStore`, `True` otherwise.
    concurrency
        The limit on this backend's operations in flight, sync and async, single and batched
        (listings are not counted). Share one `Limiter` between backends to bound them together.
        The default is a process-wide limiter for the store's class, so that every S3 backend in
        the process shares one limit, as TensorStore's `s3_request_concurrency` does. Async batches
        start at most `concurrency.limit` operations at once; sync batches run one at a time.
    """

    def __init__(
        self,
        store: ObjectStore,
        *,
        writable: bool | None = None,
        concurrency: Limiter | None = None,
    ) -> None:
        self._store = store
        self._concurrency = (
            default_limiter(f"obstore.{type(store).__name__}")
            if concurrency is None
            else concurrency
        )
        self._writable = not isinstance(store, HTTPStore) if writable is None else writable
        # Only a filesystem refuses a path that lies under a file; elsewhere a refusal is an error.
        self._filesystem = isinstance(store, LocalStore)
        self._closed = False

    @property
    def store(self) -> ObjectStore:
        """The obstore store."""
        return self._store

    @property
    def concurrency(self) -> Limiter:
        """The limit on this backend's operations in flight."""
        return self._concurrency

    def __repr__(self) -> str:
        return f"ObjectStoreBackend({self._store!r})"

    # --- Backend

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def writable(self) -> bool:
        return self._writable

    def close(self) -> None:
        self._closed = True

    # --- "absent" for filesystem-backed stores, which refuse paths under a file

    def _under_a_key(self, path: str) -> bool:
        """Whether this is a filesystem store and a parent of `path` is a key."""
        return self._filesystem and any(self._head_raw(p) is not None for p in _ancestors(path))

    async def _under_a_key_async(self, path: str) -> bool:
        if not self._filesystem:
            return False
        for p in _ancestors(path):
            if await self._head_raw_async(p) is not None:
                return True
        return False

    def _refused_under_a_key(self, error: GenericError, path: str) -> bool:
        """Whether a filesystem store refused `path` because a parent of it is a key.

        `ENOTDIR` in the refusal says so as of the refusal. Looking at the parents afterwards can
        race a concurrent delete of that parent, so that is only the fallback.
        """
        return self._filesystem and (_NOT_A_DIRECTORY in str(error) or self._under_a_key(path))

    async def _refused_under_a_key_async(self, error: GenericError, path: str) -> bool:
        return self._filesystem and (
            _NOT_A_DIRECTORY in str(error) or await self._under_a_key_async(path)
        )

    def _blocked_by_a_key(self, directory: str) -> bool:
        """Whether a filesystem store refused to list `directory` because it is, or lies under, a key."""
        return self._filesystem and (
            self._head_raw(directory) is not None or self._under_a_key(directory)
        )

    async def _blocked_by_a_key_async(self, directory: str) -> bool:
        return self._filesystem and (
            await self._head_raw_async(directory) is not None
            or await self._under_a_key_async(directory)
        )

    def _head_raw(self, path: str) -> ObjectMetadata | None:
        try:
            return _metadata(obstore.head(self._store, path))
        except (FileNotFoundError, GenericError):
            return None

    async def _head_raw_async(self, path: str) -> ObjectMetadata | None:
        try:
            return _metadata(await obstore.head_async(self._store, path))
        except (FileNotFoundError, GenericError):
            return None

    def _is_directory(self, path: str) -> bool:
        """Whether `path` is a directory, on a filesystem store (even one holding only uploads)."""
        if not isinstance(self._store, LocalStore) or self._store.prefix is None:
            return False
        return (Path(self._store.prefix) / path).is_dir()

    def _reclaim_empty_directory(self, path: str) -> bool:
        """Remove a directory at `path` that holds no files, on a filesystem store.

        obstore's `LocalStore` leaves the directories of deleted keys behind, and one in the way of a
        key would otherwise make the write fail. Only empty directories are removed, with `rmdir`,
        deepest first; `rmdir` refuses a directory a writer has just added to, so this never removes
        a key. Returns whether the directory is gone.
        """
        if not isinstance(self._store, LocalStore) or self._store.prefix is None:
            return False
        return _remove_empty_directories(Path(self._store.prefix) / path)

    def _conflicts(self, path: str) -> bool:
        """Whether this is a filesystem store and `path` collides with a key or a directory.

        Answers `False` if the check itself fails, so the caller raises the write's own error.
        """
        if not self._filesystem:
            return False
        try:
            if self._under_a_key(path):
                return True
            listed = _list_with_delimiter(self._store, path)
        except Exception:  # noqa: BLE001 -- only a diagnosis; the write's error is raised instead
            return False
        return len(listed["objects"]) > 0 or len(listed["common_prefixes"]) > 0

    async def _conflicts_async(self, path: str) -> bool:
        if not self._filesystem:
            return False
        try:
            if await self._under_a_key_async(path):
                return True
            listed = await _list_with_delimiter_async(self._store, path)
        except Exception:  # noqa: BLE001 -- only a diagnosis; the write's error is raised instead
            return False
        return len(listed["objects"]) > 0 or len(listed["common_prefixes"]) > 0

    # --- Readable

    def head(self, path: str) -> ObjectMetadata | Missing:
        with self._concurrency.slot():
            try:
                return _metadata(obstore.head(self._store, path))
            except FileNotFoundError:
                return Missing(time.time())
            except GenericError as e:
                if self._refused_under_a_key(e, path):
                    return Missing(time.time())
                raise

    async def head_async(self, path: str) -> ObjectMetadata | Missing:
        async with self._concurrency.slot():
            try:
                return _metadata(await obstore.head_async(self._store, path))
            except FileNotFoundError:
                return Missing(time.time())
            except GenericError as e:
                if await self._refused_under_a_key_async(e, path):
                    return Missing(time.time())
                raise

    def get(self, path: str, byte_range: ByteRange | None) -> Found | Missing:
        with self._concurrency.slot():
            return self._get(path, byte_range, {})

    async def get_async(self, path: str, byte_range: ByteRange | None) -> Found | Missing:
        async with self._concurrency.slot():
            return await self._get_async(path, byte_range, {})

    def get_into(self, path: str, out: Buffer, byte_range: ByteRange | None) -> Copied | Missing:
        """A read, copied into `out`: obstore can't read into a caller's buffer."""
        with self._concurrency.slot():
            view = byte_view(out, writable=True)
            return copy_into(self.get(path, byte_range), view)

    async def get_into_async(
        self, path: str, out: Buffer, byte_range: ByteRange | None
    ) -> Copied | Missing:
        async with self._concurrency.slot():
            view = byte_view(out, writable=True)
            return copy_into(await self.get_async(path, byte_range), view)

    def get_if_changed(
        self, path: str, generation: Generation, byte_range: ByteRange | None
    ) -> Found | Missing | Unchanged:
        with self._concurrency.slot():
            try:
                result = self._get(path, byte_range, _if_none_match(generation))
            except NotModifiedError:
                return Unchanged(generation, time.time())
            return _unless_unchanged(result, generation)

    async def get_if_changed_async(
        self, path: str, generation: Generation, byte_range: ByteRange | None
    ) -> Found | Missing | Unchanged:
        async with self._concurrency.slot():
            try:
                result = await self._get_async(path, byte_range, _if_none_match(generation))
            except NotModifiedError:
                return Unchanged(generation, time.time())
            return _unless_unchanged(result, generation)

    def _get(self, path: str, byte_range: ByteRange | None, options: GetOptions) -> Found | Missing:
        """A read with extra `options`; raises `NotModifiedError` if they include a matching
        `if_none_match`."""
        if _is_empty_request(byte_range):
            return _empty_read_or_raise(self.head(path), byte_range, None)
        try:
            result = obstore.get(self._store, path, options={**options, **_get_options(byte_range)})
        except FileNotFoundError:
            return Missing(time.time())
        except GenericError as e:
            return _empty_read_or_raise(self.head(path), byte_range, e)
        return _found(result, memoryview(result.bytes()))

    async def _get_async(
        self, path: str, byte_range: ByteRange | None, options: GetOptions
    ) -> Found | Missing:
        if _is_empty_request(byte_range):
            return _empty_read_or_raise(await self.head_async(path), byte_range, None)
        try:
            result = await obstore.get_async(
                self._store, path, options={**options, **_get_options(byte_range)}
            )
        except FileNotFoundError:
            return Missing(time.time())
        except GenericError as e:
            return _empty_read_or_raise(await self.head_async(path), byte_range, e)
        return _found(result, memoryview(await result.bytes_async()))

    def get_ranges(self, path: str, ranges: Sequence[ByteRange]) -> FoundRanges | Missing:
        with self._concurrency.slot():
            if len(ranges) == 0 or _is_empty_request(ranges[0]):
                return self._read_whole_and_slice(path, ranges)
            try:
                first = obstore.get(self._store, path, options=_get_options(ranges[0]))
            except FileNotFoundError:
                return Missing(time.time())
            except GenericError:
                # The first range starts at the end of the value; one whole read answers every range.
                return self._read_whole_and_slice(path, ranges)
            meta = first.meta
            generation = _generation(meta["e_tag"], meta["version"])
            pinned = _pinned(generation)
            if pinned is None:
                return self._read_whole_and_slice(path, ranges)
            try:
                rest = [
                    self._read_pinned(path, pinned, *resolve(r, meta["size"])) for r in ranges[1:]
                ]
            except (PreconditionError, FileNotFoundError):
                # The value changed after the first read; one whole read is consistent by itself.
                return self._read_whole_and_slice(path, ranges)
            return FoundRanges((memoryview(first.bytes()), *rest), generation, time.time())

    def _read_pinned(self, path: str, pinned: GetOptions, start: int, end: int) -> memoryview:
        if start == end:
            return _EMPTY
        result = obstore.get(self._store, path, options={**pinned, "range": (start, end)})
        _check_pinned(result.meta, pinned)
        return memoryview(result.bytes())

    def _read_whole_and_slice(
        self, path: str, ranges: Sequence[ByteRange]
    ) -> FoundRanges | Missing:
        try:
            result = obstore.get(self._store, path)
        except FileNotFoundError:
            return Missing(time.time())
        except GenericError as e:
            if self._refused_under_a_key(e, path):
                return Missing(time.time())
            raise
        generation = _generation(result.meta["e_tag"], result.meta["version"])
        return FoundRanges(_slices(memoryview(result.bytes()), ranges), generation, time.time())

    async def get_ranges_async(
        self, path: str, ranges: Sequence[ByteRange]
    ) -> FoundRanges | Missing:
        async with self._concurrency.slot():
            if len(ranges) == 0 or _is_empty_request(ranges[0]):
                return await self._read_whole_and_slice_async(path, ranges)
            try:
                first = await obstore.get_async(self._store, path, options=_get_options(ranges[0]))
            except FileNotFoundError:
                return Missing(time.time())
            except GenericError:
                return await self._read_whole_and_slice_async(path, ranges)
            meta = first.meta
            generation = _generation(meta["e_tag"], meta["version"])
            pinned = _pinned(generation)
            if pinned is None:
                return await self._read_whole_and_slice_async(path, ranges)
            try:
                rest = await asyncio.gather(
                    *(
                        _read_pinned_async(self._store, path, pinned, *resolve(r, meta["size"]))
                        for r in ranges[1:]
                    )
                )
            except (PreconditionError, FileNotFoundError):
                return await self._read_whole_and_slice_async(path, ranges)
            return FoundRanges(
                (memoryview(await first.bytes_async()), *rest), generation, time.time()
            )

    async def _read_whole_and_slice_async(
        self, path: str, ranges: Sequence[ByteRange]
    ) -> FoundRanges | Missing:
        try:
            result = await obstore.get_async(self._store, path)
        except FileNotFoundError:
            return Missing(time.time())
        except GenericError as e:
            if await self._refused_under_a_key_async(e, path):
                return Missing(time.time())
            raise
        generation = _generation(result.meta["e_tag"], result.meta["version"])
        data = memoryview(await result.bytes_async())
        return FoundRanges(_slices(data, ranges), generation, time.time())

    # --- Writable

    def put(self, path: str, value: memoryview) -> Written:
        with self._concurrency.slot():
            return self._put(path, value, None, always)

    async def put_async(self, path: str, value: memoryview) -> Written:
        async with self._concurrency.slot():
            return await self._put_async(path, value, None, always)

    def put_if_absent(self, path: str, value: memoryview) -> Written | Rejected:
        with self._concurrency.slot():
            return self._put(path, value, "create", if_absent)

    async def put_if_absent_async(self, path: str, value: memoryview) -> Written | Rejected:
        async with self._concurrency.slot():
            return await self._put_async(path, value, "create", if_absent)

    def put_if_match(
        self, path: str, value: memoryview, generation: Generation
    ) -> Written | Rejected | Missing:
        with self._concurrency.slot():
            mode = _if_match_mode(generation)
            if mode is None:
                return _refuse(self.head(path))
            return self._put(path, value, mode, IfMatch(generation))

    async def put_if_match_async(
        self, path: str, value: memoryview, generation: Generation
    ) -> Written | Rejected | Missing:
        async with self._concurrency.slot():
            mode = _if_match_mode(generation)
            if mode is None:
                return _refuse(await self.head_async(path))
            return await self._put_async(path, value, mode, IfMatch(generation))

    def _put(
        self,
        path: str,
        value: memoryview,
        mode: PutMode | None,
        check: Callable[[Generation | None], R | None],
    ) -> Written | R:
        """Write with `mode`. When the store refuses, `check` the key's current generation: its
        refusal is the outcome, and if it has none, the state that made the store refuse went away
        and the write starts over."""
        for _ in range(_PUT_ATTEMPTS):
            try:
                # obstore keeps a reference to the buffer it's given; copy so later changes to the
                # caller's buffer don't reach the stored value.
                result = obstore.put(self._store, path, bytes(value), mode=mode)
            except (AlreadyExistsError, PreconditionError):
                current = self.head(path)
                if isinstance(current, Missing):
                    if self._reclaim_empty_directory(path):
                        continue
                    if self._conflicts(path):
                        raise _conflict(path, self._store) from None
                refusal = check(None if isinstance(current, Missing) else current.generation)
                if refusal is not None:
                    return refusal
                continue
            except NotImplementedError as e:
                raise UnsupportedOperationError(
                    f"{self._store!r} does not support this conditional write."
                ) from e
            except GenericError as e:
                if self._reclaim_empty_directory(path):
                    continue
                if self._conflicts(path) or self._is_directory(path):
                    raise _conflict(path, self._store) from e
                raise
            return Written(_generation(result["e_tag"], result["version"]))
        raise OSError(f"Gave up writing {path!r}: the store kept refusing it.")

    async def _put_async(
        self,
        path: str,
        value: memoryview,
        mode: PutMode | None,
        check: Callable[[Generation | None], R | None],
    ) -> Written | R:
        for _ in range(_PUT_ATTEMPTS):
            try:
                result = await obstore.put_async(self._store, path, bytes(value), mode=mode)
            except (AlreadyExistsError, PreconditionError):
                current = await self.head_async(path)
                if isinstance(current, Missing):
                    if await asyncio.to_thread(self._reclaim_empty_directory, path):
                        continue
                    if await self._conflicts_async(path):
                        raise _conflict(path, self._store) from None
                refusal = check(None if isinstance(current, Missing) else current.generation)
                if refusal is not None:
                    return refusal
                continue
            except NotImplementedError as e:
                raise UnsupportedOperationError(
                    f"{self._store!r} does not support this conditional write."
                ) from e
            except GenericError as e:
                if await asyncio.to_thread(self._reclaim_empty_directory, path):
                    continue
                if await self._conflicts_async(path) or await asyncio.to_thread(
                    self._is_directory, path
                ):
                    raise _conflict(path, self._store) from e
                raise
            return Written(_generation(result["e_tag"], result["version"]))
        raise OSError(f"Gave up writing {path!r}: the store kept refusing it.")

    def delete(self, path: str) -> Deleted | Missing:
        with self._concurrency.slot():
            meta = self.head(path)
            if isinstance(meta, Missing):
                return meta
            try:
                obstore.delete(self._store, path)
            except FileNotFoundError:
                return Missing(time.time())
            return Deleted(meta.generation)

    async def delete_async(self, path: str) -> Deleted | Missing:
        async with self._concurrency.slot():
            meta = await self.head_async(path)
            if isinstance(meta, Missing):
                return meta
            try:
                await obstore.delete_async(self._store, path)
            except FileNotFoundError:
                return Missing(time.time())
            return Deleted(meta.generation)

    def delete_dir(self, directory: str) -> None:
        with self._concurrency.slot():
            paths = list(self._paths(directory))
            try:
                if paths:
                    obstore.delete(self._store, paths)
            except FileNotFoundError:
                # A concurrent delete removed some first; delete the rest one at a time.
                for path in paths:
                    with contextlib.suppress(FileNotFoundError):
                        obstore.delete(self._store, path)

    async def delete_dir_async(self, directory: str) -> None:
        async with self._concurrency.slot():
            paths = [path async for path in self._paths_async(directory)]
            try:
                if paths:
                    await obstore.delete_async(self._store, paths)
            except FileNotFoundError:
                for path in paths:
                    with contextlib.suppress(FileNotFoundError):
                        await obstore.delete_async(self._store, path)

    # --- async batches: the async operations, at most `concurrency.limit` started at once

    def get_many_async(
        self, requests: Sequence[tuple[str, ByteRange | None]]
    ) -> AsyncIterator[tuple[int, Found | Missing]]:
        return run_many_async(self.get_async, requests, self._concurrency.limit)

    def get_into_many_async(
        self, requests: Sequence[tuple[str, Buffer, ByteRange | None]]
    ) -> AsyncIterator[tuple[int, Copied | Missing]]:
        return run_many_async(self.get_into_async, requests, self._concurrency.limit)

    def put_many_async(
        self, requests: Sequence[tuple[str, memoryview]]
    ) -> AsyncIterator[tuple[int, Written]]:
        return run_many_async(self.put_async, requests, self._concurrency.limit)

    def delete_many_async(
        self, paths: Sequence[str]
    ) -> AsyncIterator[tuple[int, Deleted | Missing]]:
        return run_many_async(
            self.delete_async, [(path,) for path in paths], self._concurrency.limit
        )

    # --- Listable

    def _paths(self, directory: str) -> Iterator[str]:
        """Every stored path under `directory`, whether or not it follows the key grammar."""
        try:
            for chunk in _list(self._store, _prefix(directory)):
                yield from (meta["path"] for meta in chunk if meta["path"] != directory)
        except GenericError:
            # A filesystem store can't walk a path that is, or lies under, a file; such a
            # directory has no children.
            if not self._blocked_by_a_key(directory):
                raise

    async def _paths_async(self, directory: str) -> AsyncIterator[str]:
        try:
            async for chunk in _list(self._store, _prefix(directory)):
                for meta in chunk:
                    if meta["path"] != directory:
                        yield meta["path"]
        except GenericError:
            if not await self._blocked_by_a_key_async(directory):
                raise

    def list(self, directory: str) -> Iterator[str]:
        return _valid_keys(self._paths(directory), directory)

    async def list_async(self, directory: str) -> AsyncIterator[str]:
        async for path in self._paths_async(directory):
            for key in _valid_keys([path], directory):
                yield key

    def list_dir(self, directory: str) -> Iterator[ListItem]:
        try:
            result = _list_with_delimiter(self._store, _prefix(directory))
        except GenericError:
            if not self._blocked_by_a_key(directory):
                raise
            return iter(())
        return iter(_children(result["objects"], result["common_prefixes"], directory))

    async def list_dir_async(self, directory: str) -> AsyncIterator[ListItem]:
        try:
            result = await _list_with_delimiter_async(self._store, _prefix(directory))
        except GenericError:
            if not await self._blocked_by_a_key_async(directory):
                raise
            return
        for item in _children(result["objects"], result["common_prefixes"], directory):
            yield item


async def _read_pinned_async(
    store: ObjectStore, path: str, pinned: GetOptions, start: int, end: int
) -> memoryview:
    if start == end:
        return _EMPTY
    result = await obstore.get_async(store, path, options={**pinned, "range": (start, end)})
    _check_pinned(result.meta, pinned)
    return memoryview(await result.bytes_async())


def _check_pinned(meta: ObjectMeta, pinned: GetOptions) -> None:
    """Treat a response from another generation as a failed precondition.

    A store that ignored `If-Match` would otherwise mix parts of two values.
    """
    if meta["e_tag"] != pinned.get("if_match", meta["e_tag"]) or meta["version"] != pinned.get(
        "version", meta["version"]
    ):
        raise PreconditionError("The value changed between reads.")


def _remove_empty_directories(target: Path) -> bool:
    """Remove `target` and the directories under it if none holds a file; whether it is gone."""
    if not target.is_dir():
        return False
    try:
        for directory, _, files in os.walk(target, topdown=False):
            if files:
                return False
            os.rmdir(directory)
    except FileNotFoundError:
        return not target.exists()
    except NotADirectoryError:
        return False  # A key was written in the directory's place meanwhile.
    except OSError as e:
        if e.errno == errno.ENOTEMPTY:
            return False
        raise
    return True


def _unless_unchanged(
    result: Found | Missing, generation: Generation
) -> Found | Missing | Unchanged:
    """`Unchanged` for a value read whole although it has `generation`.

    That happens when the store can't skip the read (no ETag to send, or it ignores
    `If-None-Match`); the bytes are discarded so the outcome doesn't depend on the store.
    """
    if isinstance(result, Found) and result.generation == generation:
        return Unchanged(generation, result.read_at)
    return result


def _conflict(path: str, store: ObjectStore) -> KeyConflictError:
    return KeyConflictError(
        f"{path!r} collides with a directory or key of the same name, which {store!r} can't hold "
        "side by side."
    )
