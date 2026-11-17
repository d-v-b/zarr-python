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
from pathlib import Path
from typing import TYPE_CHECKING, cast

import obstore
from obstore.exceptions import AlreadyExistsError, GenericError, PreconditionError
from obstore.store import HTTPStore, LocalStore

from zarr_storage._byte_range import resolve
from zarr_storage._errors import InvalidKeyError, KeyConflictError, UnsupportedOperationError
from zarr_storage._keys import check_key, relativize
from zarr_storage._types import (
    DeleteResult,
    Generation,
    Key,
    ObjectMetadata,
    OffsetRequest,
    Prefix,
    PutResult,
    RangeRequest,
    RangesResult,
    ReadResult,
    SuffixRequest,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Iterator, Sequence

    from obstore import GetOptions, ListResult, ListStream, ObjectMeta, PutMode
    from obstore.store import ObjectStore

    from zarr_storage._types import ByteRange, ListItem

__all__ = ["ObjectStoreBackend"]

_EMPTY = memoryview(b"")


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


class _NeverMatches:
    """The precondition names a generation this backend did not mint, so no stored value has it."""


_NEVER_MATCHES = _NeverMatches()


def _put_mode(if_match: Generation | None, if_none_match: bool) -> PutMode | _NeverMatches | None:
    if if_none_match:
        return "create"
    if if_match is None:
        return None
    version = _version_of(if_match)
    if version is None:
        return _NEVER_MATCHES
    e_tag, object_version = version
    if e_tag is None and object_version is None:
        raise UnsupportedOperationError(
            "The store reported neither an ETag nor a version, so writes can't be conditional."
        )
    mode = {k: v for k, v in (("e_tag", e_tag), ("version", object_version)) if v is not None}
    return cast("PutMode", mode)


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
    meta: ObjectMetadata | None, byte_range: ByteRange | None, error: BaseException | None
) -> ReadResult | None:
    """The result of a read the object store refused or that selects nothing.

    Object stores refuse a range that starts at the end of the value; the contract reads that as
    empty. A range past the end raises `InvalidByteRangeError` (from `resolve`). Any other refusal
    re-raises the store's error.
    """
    if meta is None:
        return None
    start, end = resolve(byte_range, meta.size)
    if start == end:
        return ReadResult(_EMPTY, meta.generation)
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

    Generations are the store's ETag and version. Conditional writes need a store that supports them
    (S3, GCS, Azure and obstore's `MemoryStore` do; obstore's `LocalStore` supports only
    `if_none_match`); otherwise they raise `UnsupportedOperationError`. Conditional deletes are not
    supported.

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
    """

    def __init__(self, store: ObjectStore, *, writable: bool | None = None) -> None:
        self._store = store
        self._writable = not isinstance(store, HTTPStore) if writable is None else writable
        # Only a filesystem refuses a path that lies under a file; elsewhere a refusal is an error.
        self._filesystem = isinstance(store, LocalStore)
        self._closed = False

    @property
    def store(self) -> ObjectStore:
        """The obstore store."""
        return self._store

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

    def head(self, path: str) -> ObjectMetadata | None:
        try:
            return _metadata(obstore.head(self._store, path))
        except FileNotFoundError:
            return None
        except GenericError:
            if self._under_a_key(path):
                return None
            raise

    async def head_async(self, path: str) -> ObjectMetadata | None:
        try:
            return _metadata(await obstore.head_async(self._store, path))
        except FileNotFoundError:
            return None
        except GenericError:
            if await self._under_a_key_async(path):
                return None
            raise

    def get(self, path: str, byte_range: ByteRange | None) -> ReadResult | None:
        if _is_empty_request(byte_range):
            return _empty_read_or_raise(self.head(path), byte_range, None)
        try:
            result = obstore.get(self._store, path, options=_get_options(byte_range))
        except FileNotFoundError:
            return None
        except GenericError as e:
            return _empty_read_or_raise(self.head(path), byte_range, e)
        meta = result.meta
        return ReadResult(memoryview(result.bytes()), _generation(meta["e_tag"], meta["version"]))

    async def get_async(self, path: str, byte_range: ByteRange | None) -> ReadResult | None:
        if _is_empty_request(byte_range):
            return _empty_read_or_raise(await self.head_async(path), byte_range, None)
        try:
            result = await obstore.get_async(self._store, path, options=_get_options(byte_range))
        except FileNotFoundError:
            return None
        except GenericError as e:
            return _empty_read_or_raise(await self.head_async(path), byte_range, e)
        meta = result.meta
        data = await result.bytes_async()
        return ReadResult(memoryview(data), _generation(meta["e_tag"], meta["version"]))

    def get_ranges(self, path: str, ranges: Sequence[ByteRange]) -> RangesResult | None:
        if len(ranges) == 0 or _is_empty_request(ranges[0]):
            return self._read_whole_and_slice(path, ranges)
        try:
            first = obstore.get(self._store, path, options=_get_options(ranges[0]))
        except FileNotFoundError:
            return None
        except GenericError:
            # The first range starts at the end of the value; one whole read answers every range.
            return self._read_whole_and_slice(path, ranges)
        meta = first.meta
        generation = _generation(meta["e_tag"], meta["version"])
        pinned = _pinned(generation)
        if pinned is None:
            return self._read_whole_and_slice(path, ranges)
        try:
            rest = [self._read_pinned(path, pinned, *resolve(r, meta["size"])) for r in ranges[1:]]
        except (PreconditionError, FileNotFoundError):
            # The value changed after the first read; one whole read is consistent by itself.
            return self._read_whole_and_slice(path, ranges)
        return RangesResult((memoryview(first.bytes()), *rest), generation)

    def _read_pinned(self, path: str, pinned: GetOptions, start: int, end: int) -> memoryview:
        if start == end:
            return _EMPTY
        result = obstore.get(self._store, path, options={**pinned, "range": (start, end)})
        _check_pinned(result.meta, pinned)
        return memoryview(result.bytes())

    def _read_whole_and_slice(self, path: str, ranges: Sequence[ByteRange]) -> RangesResult | None:
        try:
            result = obstore.get(self._store, path)
        except FileNotFoundError:
            return None
        except GenericError:
            if self._under_a_key(path):
                return None
            raise
        generation = _generation(result.meta["e_tag"], result.meta["version"])
        return RangesResult(_slices(memoryview(result.bytes()), ranges), generation)

    async def get_ranges_async(self, path: str, ranges: Sequence[ByteRange]) -> RangesResult | None:
        if len(ranges) == 0 or _is_empty_request(ranges[0]):
            return await self._read_whole_and_slice_async(path, ranges)
        try:
            first = await obstore.get_async(self._store, path, options=_get_options(ranges[0]))
        except FileNotFoundError:
            return None
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
        return RangesResult((memoryview(await first.bytes_async()), *rest), generation)

    async def _read_whole_and_slice_async(
        self, path: str, ranges: Sequence[ByteRange]
    ) -> RangesResult | None:
        try:
            result = await obstore.get_async(self._store, path)
        except FileNotFoundError:
            return None
        except GenericError:
            if await self._under_a_key_async(path):
                return None
            raise
        generation = _generation(result.meta["e_tag"], result.meta["version"])
        return RangesResult(_slices(memoryview(await result.bytes_async()), ranges), generation)

    # --- Writable

    def put(
        self, path: str, value: memoryview, *, if_match: Generation | None, if_none_match: bool
    ) -> PutResult:
        mode = _put_mode(if_match, if_none_match)
        if isinstance(mode, _NeverMatches):
            return PutResult(_generation_or_none(self.head(path)), applied=False)
        try:
            # obstore keeps a reference to the buffer it's given; copy so later changes to the
            # caller's buffer don't reach the stored value.
            result = obstore.put(self._store, path, bytes(value), mode=mode)
        except (AlreadyExistsError, PreconditionError):
            current = self.head(path)
            if current is None and self._reclaim_empty_directory(path):
                return self.put(path, value, if_match=if_match, if_none_match=if_none_match)
            if current is None and self._conflicts(path):
                raise _conflict(path, self._store) from None
            return PutResult(_generation_or_none(current), applied=False)
        except NotImplementedError as e:
            raise UnsupportedOperationError(
                f"{self._store!r} does not support this conditional write."
            ) from e
        except GenericError as e:
            if self._reclaim_empty_directory(path):
                return self.put(path, value, if_match=if_match, if_none_match=if_none_match)
            if self._conflicts(path) or self._is_directory(path):
                raise _conflict(path, self._store) from e
            raise
        return PutResult(_generation(result["e_tag"], result["version"]))

    async def put_async(
        self, path: str, value: memoryview, *, if_match: Generation | None, if_none_match: bool
    ) -> PutResult:
        mode = _put_mode(if_match, if_none_match)
        if isinstance(mode, _NeverMatches):
            return PutResult(_generation_or_none(await self.head_async(path)), applied=False)
        try:
            result = await obstore.put_async(self._store, path, bytes(value), mode=mode)
        except (AlreadyExistsError, PreconditionError):
            current = await self.head_async(path)
            if current is None and await asyncio.to_thread(self._reclaim_empty_directory, path):
                return await self.put_async(
                    path, value, if_match=if_match, if_none_match=if_none_match
                )
            if current is None and await self._conflicts_async(path):
                raise _conflict(path, self._store) from None
            return PutResult(_generation_or_none(current), applied=False)
        except NotImplementedError as e:
            raise UnsupportedOperationError(
                f"{self._store!r} does not support this conditional write."
            ) from e
        except GenericError as e:
            if await asyncio.to_thread(self._reclaim_empty_directory, path):
                return await self.put_async(
                    path, value, if_match=if_match, if_none_match=if_none_match
                )
            if await self._conflicts_async(path) or await asyncio.to_thread(
                self._is_directory, path
            ):
                raise _conflict(path, self._store) from e
            raise
        return PutResult(_generation(result["e_tag"], result["version"]))

    def delete(self, path: str, *, if_match: Generation | None) -> DeleteResult:
        _refuse_conditional_delete(if_match)
        meta = self.head(path)
        if meta is None:
            return DeleteResult(None, applied=False)
        try:
            obstore.delete(self._store, path)
        except FileNotFoundError:
            return DeleteResult(None, applied=False)
        return DeleteResult(meta.generation, applied=True)

    async def delete_async(self, path: str, *, if_match: Generation | None) -> DeleteResult:
        _refuse_conditional_delete(if_match)
        meta = await self.head_async(path)
        if meta is None:
            return DeleteResult(None, applied=False)
        try:
            await obstore.delete_async(self._store, path)
        except FileNotFoundError:
            return DeleteResult(None, applied=False)
        return DeleteResult(meta.generation, applied=True)

    def delete_dir(self, directory: str) -> None:
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
        paths = [path async for path in self._paths_async(directory)]
        try:
            if paths:
                await obstore.delete_async(self._store, paths)
        except FileNotFoundError:
            for path in paths:
                with contextlib.suppress(FileNotFoundError):
                    await obstore.delete_async(self._store, path)

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


def _generation_or_none(meta: ObjectMetadata | None) -> Generation | None:
    return None if meta is None else meta.generation


def _refuse_conditional_delete(if_match: Generation | None) -> None:
    if if_match is not None:
        raise UnsupportedOperationError("Object stores do not support conditional deletes.")


def _conflict(path: str, store: ObjectStore) -> KeyConflictError:
    return KeyConflictError(
        f"{path!r} collides with a directory or key of the same name, which {store!r} can't hold "
        "side by side."
    )
