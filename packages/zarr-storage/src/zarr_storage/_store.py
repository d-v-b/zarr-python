"""`Store`: a view of a backend at a root.

`Store` is where the key grammar, the root, read-only access and the closed check are enforced, once,
for every backend. A backend only sees backend paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Generic, Self, TypeVar, cast, final

from typing_extensions import Buffer

from zarr_storage._errors import ReadOnlyError, StoreClosedError
from zarr_storage._keys import check_key, is_within, join, normalize_directory, relativize
from zarr_storage._protocols import Backend
from zarr_storage._types import Copied, Found, Missing

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence
    from types import TracebackType

    from zarr_storage._protocols import (
        DeletableIfMatch,
        DeletableIfMatchAsync,
        Listable,
        ListableAsync,
        Readable,
        ReadableAsync,
        ReadableInto,
        ReadableIntoAsync,
        Writable,
        WritableAsync,
        WritableIfMatch,
        WritableIfMatchAsync,
    )
    from zarr_storage._types import (
        ByteRange,
        Deleted,
        FoundRanges,
        Generation,
        ListItem,
        ObjectMetadata,
        Rejected,
        Unchanged,
        Written,
    )

__all__ = ["Store"]

B_co = TypeVar("B_co", bound=Backend, covariant=True)
T = TypeVar("T")


def _as_bytes_view(value: Buffer, *, writable: bool) -> memoryview:
    """A flat, C-contiguous byte view of a buffer."""
    view = memoryview(value)
    if not view.c_contiguous:
        raise ValueError("The buffer must be C-contiguous.")
    if writable and view.readonly:
        raise ValueError("The output buffer must be writable.")
    return view.cast("B") if view.format != "B" or view.ndim != 1 else view


@final
class Store(Generic[B_co]):
    """A backend seen from a root directory.

    Construct a store from a backend; the store you construct owns the backend and closes it.
    `store / "a"` and `with_read_only()` return views on the same backend, which share its state and
    never close it. A view of a read-only store is read-only too.

    `backend` gives direct access to the backend, which bypasses the key grammar, the root, and
    read-only and closed checks.

    Stores have no value equality: `==` is identity. Use `shares_backend` and `overlaps` to compare
    stores.

    Parameters
    ----------
    backend
        The backend to read and write through.
    root
        The directory within the backend that this store's keys are relative to.
    read_only
        Refuse writes through this store. A backend that can't write makes every store over it
        read-only.
    """

    __slots__ = ("_backend", "_owner", "_read_only", "_root")

    def __init__(self, backend: B_co, *, root: str = "", read_only: bool = False) -> None:
        self._backend: Final[B_co] = backend
        self._root: Final[str] = normalize_directory(root)
        self._read_only: Final[bool] = read_only or not backend.writable
        self._owner = True

    # ----------------------------------------------------------------- views and identity

    @property
    def backend(self) -> B_co:
        """The backend this store reads and writes through."""
        return self._backend

    @property
    def root(self) -> str:
        """This store's directory within the backend; `""` is the backend's root."""
        return self._root

    @property
    def read_only(self) -> bool:
        """Whether writes through this store are refused."""
        return self._read_only

    @property
    def closed(self) -> bool:
        """Whether the backend has been closed."""
        return self._backend.closed

    def _view(self, *, root: str, read_only: bool) -> Self:
        view = type(self)(self._backend, root=root, read_only=read_only)
        view._owner = False
        return view

    def __truediv__(self, directory: str) -> Self:
        """A view on the same backend, rooted at `directory` under this store's root."""
        return self._view(
            root=join(self._root, normalize_directory(directory)), read_only=self._read_only
        )

    def with_read_only(self, read_only: bool = True) -> Self:
        """A view on the same backend and root that refuses writes, or accepts them.

        A read-only store can't produce a writable view: `with_read_only(False)` on one raises
        `ReadOnlyError`.
        """
        if self._read_only and not read_only:
            raise ReadOnlyError(f"{self!r} is read-only; its views are read-only too.")
        return self._view(root=self._root, read_only=read_only)

    def shares_backend(self, other: Store[Backend]) -> bool:
        """Whether `other` reads and writes through the same backend object."""
        return self._backend is other._backend

    def overlaps(self, other: Store[Backend]) -> bool:
        """Whether a key could be visible through both stores: same backend, nested roots."""
        return self.shares_backend(other) and (
            is_within(self._root, other._root) or is_within(other._root, self._root)
        )

    def __repr__(self) -> str:
        flags = ", read_only=True" if self._read_only else ""
        return f"{type(self).__name__}({self._backend!r}, root={self._root!r}{flags})"

    def __reduce__(self) -> tuple[object, ...]:
        return (_restore_store, (type(self), self._backend, self._root, self._read_only))

    # ----------------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Close the backend, if this store owns it (it was constructed, not derived)."""
        if self._owner:
            self._backend.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ----------------------------------------------------------------- path handling

    def _check_open(self) -> None:
        if self._backend.closed:
            raise StoreClosedError(f"{self!r} is closed.")

    def _path(self, key: str) -> str:
        self._check_open()
        return join(self._root, check_key(key))

    def _directory(self, directory: str) -> str:
        self._check_open()
        return join(self._root, normalize_directory(directory))

    def _open_items(self, items: Iterator[T]) -> Iterator[T]:
        """`items`, refusing to continue once the backend closes."""
        for item in items:
            self._check_open()
            yield item

    async def _open_items_async(self, items: AsyncIterator[T]) -> AsyncIterator[T]:
        async for item in items:
            self._check_open()
            yield item

    def _writable_path(self, key: str) -> str:
        path = self._path(key)
        if self._read_only:
            raise ReadOnlyError(f"{self!r} is read-only.")
        return path

    def _writable_directory(self, directory: str) -> str:
        path = self._directory(directory)
        if self._read_only:
            raise ReadOnlyError(f"{self!r} is read-only.")
        return path

    # ----------------------------------------------------------------- sync reads

    def get(
        self: Store[Readable], key: str, *, byte_range: ByteRange | None = None
    ) -> Found | Missing:
        """The value at `key`, or the part `byte_range` selects."""
        return self._backend.get(self._path(key), byte_range)

    def get_if_changed(
        self: Store[Readable],
        key: str,
        generation: Generation,
        *,
        byte_range: ByteRange | None = None,
    ) -> Found | Missing | Unchanged:
        """`get`, unless the value still has `generation`: then `Unchanged`, and no bytes are read.

        This revalidates a cached copy, as an HTTP `If-None-Match` request does.
        """
        return self._backend.get_if_changed(self._path(key), generation, byte_range)

    def get_ranges(
        self: Store[Readable], key: str, ranges: Sequence[ByteRange]
    ) -> FoundRanges | Missing:
        """Several parts of the value at `key`, all from one generation."""
        return self._backend.get_ranges(self._path(key), ranges)

    def get_into(
        self: Store[Readable], key: str, out: Buffer, *, byte_range: ByteRange | None = None
    ) -> Copied | Missing:
        """Copy the value at `key` (or part of it) into `out`.

        Raises `ValueError` if `out` is shorter than the bytes to copy, read-only, or not
        C-contiguous.
        """
        path = self._path(key)
        view = _as_bytes_view(out, writable=True)
        if hasattr(self._backend, "get_into"):
            backend = cast("ReadableInto", self._backend)
            return backend.get_into(path, view, byte_range)
        return _copy_into(self._backend.get(path, byte_range), view)

    def head(self: Store[Readable], key: str) -> ObjectMetadata | Missing:
        """The size and generation of the value at `key`."""
        return self._backend.head(self._path(key))

    def exists(self: Store[Readable], key: str) -> bool:
        """Whether `key` holds a value."""
        return not isinstance(self.head(key), Missing)

    # ----------------------------------------------------------------- sync writes

    def put(self: Store[Writable], key: str, value: Buffer) -> Written:
        """Write `value` at `key`."""
        return self._backend.put(self._writable_path(key), _as_bytes_view(value, writable=False))

    def put_if_absent(self: Store[Writable], key: str, value: Buffer) -> Written | Rejected:
        """Write `value` at `key` if the key holds no value; `Rejected` if it holds one."""
        return self._backend.put_if_absent(
            self._writable_path(key), _as_bytes_view(value, writable=False)
        )

    def put_if_match(
        self: Store[WritableIfMatch], key: str, value: Buffer, generation: Generation
    ) -> Written | Rejected | Missing:
        """Write `value` at `key` if the key's value has `generation`.

        `Rejected` if the key holds another generation, `Missing` if it holds no value.
        """
        return self._backend.put_if_match(
            self._writable_path(key), _as_bytes_view(value, writable=False), generation
        )

    def delete(self: Store[Writable], key: str) -> Deleted | Missing:
        """Delete the key `key`. A directory is not a key: deleting one finds it `Missing`."""
        return self._backend.delete(self._writable_path(key))

    def delete_if_match(
        self: Store[DeletableIfMatch], key: str, generation: Generation
    ) -> Deleted | Rejected | Missing:
        """Delete the key `key` if its value has `generation`."""
        return self._backend.delete_if_match(self._writable_path(key), generation)

    def delete_dir(self: Store[Writable], directory: str = "") -> None:
        """Delete every key under `directory` (by default, every key in this store)."""
        self._backend.delete_dir(self._writable_directory(directory))

    # ----------------------------------------------------------------- sync listing

    def list(self: Store[Listable], directory: str = "") -> Iterator[str]:
        """Every key under `directory`, relative to this store's root, in no particular order.

        The directory is checked when `list` is called; iterating after the store closes raises
        `StoreClosedError`.
        """
        paths = self._backend.list(self._directory(directory))
        return self._open_items(relativize(path, self._root) for path in paths)

    def list_dir(self: Store[Listable], directory: str = "") -> Iterator[ListItem]:
        """The direct children of `directory`, named relative to it.

        A backend may report a `Prefix` that holds no keys, such as an empty directory on a
        filesystem.
        """
        return self._open_items(self._backend.list_dir(self._directory(directory)))

    # ----------------------------------------------------------------- async reads

    async def get_async(
        self: Store[ReadableAsync], key: str, *, byte_range: ByteRange | None = None
    ) -> Found | Missing:
        """Async `get`."""
        return await self._backend.get_async(self._path(key), byte_range)

    async def get_if_changed_async(
        self: Store[ReadableAsync],
        key: str,
        generation: Generation,
        *,
        byte_range: ByteRange | None = None,
    ) -> Found | Missing | Unchanged:
        """Async `get_if_changed`."""
        return await self._backend.get_if_changed_async(self._path(key), generation, byte_range)

    async def get_ranges_async(
        self: Store[ReadableAsync], key: str, ranges: Sequence[ByteRange]
    ) -> FoundRanges | Missing:
        """Async `get_ranges`."""
        return await self._backend.get_ranges_async(self._path(key), ranges)

    async def get_into_async(
        self: Store[ReadableAsync], key: str, out: Buffer, *, byte_range: ByteRange | None = None
    ) -> Copied | Missing:
        """Async `get_into`."""
        path = self._path(key)
        view = _as_bytes_view(out, writable=True)
        if hasattr(self._backend, "get_into_async"):
            backend = cast("ReadableIntoAsync", self._backend)
            return await backend.get_into_async(path, view, byte_range)
        return _copy_into(await self._backend.get_async(path, byte_range), view)

    async def head_async(self: Store[ReadableAsync], key: str) -> ObjectMetadata | Missing:
        """Async `head`."""
        return await self._backend.head_async(self._path(key))

    async def exists_async(self: Store[ReadableAsync], key: str) -> bool:
        """Async `exists`."""
        return not isinstance(await self.head_async(key), Missing)

    # ----------------------------------------------------------------- async writes

    async def put_async(self: Store[WritableAsync], key: str, value: Buffer) -> Written:
        """Async `put`."""
        return await self._backend.put_async(
            self._writable_path(key), _as_bytes_view(value, writable=False)
        )

    async def put_if_absent_async(
        self: Store[WritableAsync], key: str, value: Buffer
    ) -> Written | Rejected:
        """Async `put_if_absent`."""
        return await self._backend.put_if_absent_async(
            self._writable_path(key), _as_bytes_view(value, writable=False)
        )

    async def put_if_match_async(
        self: Store[WritableIfMatchAsync], key: str, value: Buffer, generation: Generation
    ) -> Written | Rejected | Missing:
        """Async `put_if_match`."""
        return await self._backend.put_if_match_async(
            self._writable_path(key), _as_bytes_view(value, writable=False), generation
        )

    async def delete_async(self: Store[WritableAsync], key: str) -> Deleted | Missing:
        """Async `delete`."""
        return await self._backend.delete_async(self._writable_path(key))

    async def delete_if_match_async(
        self: Store[DeletableIfMatchAsync], key: str, generation: Generation
    ) -> Deleted | Rejected | Missing:
        """Async `delete_if_match`."""
        return await self._backend.delete_if_match_async(self._writable_path(key), generation)

    async def delete_dir_async(self: Store[WritableAsync], directory: str = "") -> None:
        """Async `delete_dir`."""
        await self._backend.delete_dir_async(self._writable_directory(directory))

    # ----------------------------------------------------------------- async listing

    def list_async(self: Store[ListableAsync], directory: str = "") -> AsyncIterator[str]:
        """Async `list`."""
        paths = self._backend.list_async(self._directory(directory))
        return self._open_items_async(_relativized(paths, self._root))

    def list_dir_async(self: Store[ListableAsync], directory: str = "") -> AsyncIterator[ListItem]:
        """Async `list_dir`."""
        return self._open_items_async(self._backend.list_dir_async(self._directory(directory)))


async def _relativized(paths: AsyncIterator[str], root: str) -> AsyncIterator[str]:
    async for path in paths:
        yield relativize(path, root)


def _copy_into(result: Found | Missing, out: memoryview) -> Copied | Missing:
    """The `get_into` outcome of a `get`, copying a found value into `out`."""
    match result:
        case Found(value=value, generation=generation, read_at=read_at):
            n = len(value)
            if n > len(out):
                raise ValueError(f"The output buffer holds {len(out)} bytes; the value has {n}.")
            out[:n] = value
            return Copied(n, generation, read_at)
        case Missing():
            return result


def _restore_store(
    cls: type[Store[Backend]], backend: Backend, root: str, read_only: bool
) -> Store[Backend]:
    return cls(backend, root=root, read_only=read_only)
