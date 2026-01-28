from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import starmap
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from zarr.core.sync import sync

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable
    from types import TracebackType
    from typing import Any, Self, TypeAlias

    from zarr.core.buffer import Buffer, BufferPrototype

__all__ = ["ByteGetter", "ByteSetter", "Store", "set_or_delete"]


@dataclass
class RangeByteRequest:
    """Request a specific byte range"""

    start: int
    """The start of the byte range request (inclusive)."""
    end: int
    """The end of the byte range request (exclusive)."""


@dataclass
class OffsetByteRequest:
    """Request all bytes starting from a given byte offset"""

    offset: int
    """The byte offset for the offset range request."""


@dataclass
class SuffixByteRequest:
    """Request up to the last `n` bytes"""

    suffix: int
    """The number of bytes from the suffix to request."""


ByteRequest: TypeAlias = RangeByteRequest | OffsetByteRequest | SuffixByteRequest


def _dereference_path(root: str, path: str) -> str:
    """Combine a root path and a sub-path into a single path string."""
    if not isinstance(root, str):
        msg = f"{root=} is not a string ({type(root)=})"  # type: ignore[unreachable]
        raise TypeError(msg)
    if not isinstance(path, str):
        msg = f"{path=} is not a string ({type(path)=})"  # type: ignore[unreachable]
        raise TypeError(msg)
    root = root.rstrip("/")
    path = f"{root}/{path}" if root else path
    return path.rstrip("/")


class Store(ABC):
    """
    Abstract base class for Zarr stores.
    """

    _read_only: bool
    _is_open: bool
    _path: str

    def __init__(self, *, read_only: bool = False, path: str = "") -> None:
        self._is_open = False
        self._read_only = read_only
        self._path = path.strip("/")

    @property
    def path(self) -> str:
        """The path prefix for this store."""
        return self._path

    @classmethod
    async def open(cls, *args: Any, **kwargs: Any) -> Self:
        """
        Create and open the store.

        Parameters
        ----------
        *args : Any
            Positional arguments to pass to the store constructor.
        **kwargs : Any
            Keyword arguments to pass to the store constructor.

        Returns
        -------
        Store
            The opened store instance.
        """
        store = cls(*args, **kwargs)
        await store._open()
        return store

    def with_read_only(self, read_only: bool = False) -> Store:
        """
        Return a new store with a new read_only setting.

        The new store points to the same location with the specified new read_only state.
        The returned Store is not automatically opened, and this store is
        not automatically closed.

        Parameters
        ----------
        read_only
            If True, the store will be created in read-only mode. Defaults to False.

        Returns
        -------
            A new store of the same type with the new read only attribute.
        """
        raise NotImplementedError(
            f"with_read_only is not implemented for the {type(self)} store type."
        )

    def with_path(self, path: str) -> Store:
        """
        Return a new store with the given path, replacing the current path.

        The new store points to the same storage backend but with the specified path.
        The returned Store is not automatically opened, and this store is
        not automatically closed.

        Parameters
        ----------
        path : str
            The new path for the store.

        Returns
        -------
            A new store of the same type with the new path.
        """
        raise NotImplementedError(f"with_path is not implemented for the {type(self)} store type.")

    def __truediv__(self, other: str) -> Store:
        """Combine this store's path with another path segment."""
        from zarr.storage._utils import normalize_path

        new_path = _dereference_path(self.path, normalize_path(other))
        return self.with_path(new_path)

    def __enter__(self) -> Self:
        """Enter a context manager that will close the store upon exiting."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the store."""
        self.close()

    async def _open(self) -> None:
        """
        Open the store.

        Raises
        ------
        ValueError
            If the store is already open.
        """
        if self._is_open:
            raise ValueError("store is already open")
        self._is_open = True

    async def _ensure_open(self) -> None:
        """Open the store if it is not already open."""
        if not self._is_open:
            await self._open()

    # -------------------------------------------------------------------------
    # Path-scoped public API
    #
    # These methods auto-prepend self.path to key arguments before dispatching
    # to the abstract _-prefixed methods that concrete stores implement.
    # -------------------------------------------------------------------------

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Retrieve the value associated with a given key.

        The key is resolved relative to this store's path.

        Parameters
        ----------
        key : str
        prototype : BufferPrototype
        byte_range : ByteRequest, optional

        Returns
        -------
        Buffer
        """
        await self._ensure_open()
        full_key = _dereference_path(self.path, key) if self.path else key
        return await self._get(full_key, prototype=prototype, byte_range=byte_range)

    async def set(self, key: str, value: Buffer) -> None:
        """Store a (key, value) pair.

        The key is resolved relative to this store's path.

        Parameters
        ----------
        key : str
        value : Buffer
        """
        await self._ensure_open()
        full_key = _dereference_path(self.path, key) if self.path else key
        await self._set(full_key, value)

    async def delete(self, key: str) -> None:
        """Remove a key from the store.

        The key is resolved relative to this store's path.

        Parameters
        ----------
        key : str
        """
        await self._ensure_open()
        full_key = _dereference_path(self.path, key) if self.path else key
        await self._delete(full_key)

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the store.

        The key is resolved relative to this store's path.

        Parameters
        ----------
        key : str

        Returns
        -------
        bool
        """
        await self._ensure_open()
        full_key = _dereference_path(self.path, key) if self.path else key
        return await self._exists(full_key)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Retrieve possibly partial values from given key_ranges.

        Keys are resolved relative to this store's path.

        Parameters
        ----------
        prototype : BufferPrototype
        key_ranges : Iterable[tuple[str, ByteRequest | None]]

        Returns
        -------
        list of values, in the order of the key_ranges, may contain null/none for missing keys
        """
        await self._ensure_open()
        if self.path:
            key_ranges = [(_dereference_path(self.path, k), r) for k, r in key_ranges]
        return await self._get_partial_values(prototype=prototype, key_ranges=key_ranges)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        """
        Store a key to ``value`` if the key is not already present.

        The key is resolved relative to this store's path.

        Parameters
        ----------
        key : str
        value : Buffer
        """
        await self._ensure_open()
        full_key = _dereference_path(self.path, key) if self.path else key
        await self._set_if_not_exists(full_key, value)

    async def delete_dir(self, prefix: str) -> None:
        """
        Remove all keys and prefixes in the store that begin with a given prefix.

        The prefix is resolved relative to this store's path.
        """
        await self._ensure_open()
        full_prefix = _dereference_path(self.path, prefix) if self.path else prefix
        await self._delete_dir(full_prefix)

    async def _delete_dir(self, prefix: str) -> None:
        """
        Remove all keys and prefixes in the store that begin with a given prefix.

        The default implementation lists and deletes keys one by one.
        Concrete stores may override this for more efficient behavior.

        Parameters
        ----------
        prefix : str
            The fully-qualified prefix (self.path already prepended).
        """
        if not self.supports_deletes:
            raise NotImplementedError
        if not self.supports_listing:
            raise NotImplementedError
        self._check_writable()
        if prefix != "" and not prefix.endswith("/"):
            prefix += "/"
        async for key in self._list_prefix(prefix):
            await self._delete(key)

    async def is_empty(self, prefix: str) -> bool:
        """
        Check if the directory is empty.

        The prefix is resolved relative to this store's path.

        Parameters
        ----------
        prefix : str
            Prefix of keys to check.

        Returns
        -------
        bool
            True if the store is empty, False otherwise.
        """
        if not self.supports_listing:
            raise NotImplementedError
        await self._ensure_open()
        full_prefix = _dereference_path(self.path, prefix) if self.path else prefix
        if full_prefix != "" and not full_prefix.endswith("/"):
            full_prefix += "/"
        async for _ in self._list_prefix(full_prefix):
            return False
        return True

    def list(self) -> AsyncIterator[str]:
        """Retrieve all keys in the store, relative to this store's path.

        Returns
        -------
        AsyncIterator[str]
        """
        if self.path:
            return self._list_prefix_relative(self.path)
        return self._list()

    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """
        Retrieve all keys in the store that begin with a given prefix.
        Keys are returned relative to this store's path.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        AsyncIterator[str]
        """
        full_prefix = _dereference_path(self.path, prefix) if self.path else prefix
        if self.path:
            return self._list_prefix_relative(full_prefix)
        return self._list_prefix(full_prefix)

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        """
        Retrieve all keys and prefixes with a given prefix and which do not contain the character
        "/" after the given prefix.

        The prefix is resolved relative to this store's path.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        AsyncIterator[str]
        """
        full_prefix = _dereference_path(self.path, prefix) if self.path else prefix
        return self._list_dir(full_prefix)

    async def _list_prefix_relative(self, full_prefix: str) -> AsyncIterator[str]:
        """List keys under full_prefix but strip self.path prefix from results."""
        strip_prefix = self.path + "/"
        async for key in self._list_prefix(full_prefix):
            if key.startswith(strip_prefix):
                yield key[len(strip_prefix) :]
            else:
                yield key

    # -------------------------------------------------------------------------
    # Key-less methods (satisfy ByteGetter/ByteSetter protocols)
    #
    # These operate on self.path directly, without a key argument.
    # StorePath used to provide these; now Store provides them directly.
    # -------------------------------------------------------------------------

    async def getb(
        self,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Read bytes at this store's path (no key argument).

        Parameters
        ----------
        prototype : BufferPrototype, optional
        byte_range : ByteRequest, optional

        Returns
        -------
        Buffer or None
        """
        from zarr.core.buffer import default_buffer_prototype

        if prototype is None:
            prototype = default_buffer_prototype()
        await self._ensure_open()
        return await self._get(self.path, prototype=prototype, byte_range=byte_range)

    async def setb(self, value: Buffer) -> None:
        """Write bytes at this store's path (no key argument).

        Parameters
        ----------
        value : Buffer
        """
        await self._ensure_open()
        await self._set(self.path, value)

    async def deleteb(self) -> None:
        """Delete the value at this store's path (no key argument)."""
        await self._ensure_open()
        await self._delete(self.path)

    async def existsb(self) -> bool:
        """Check if a value exists at this store's path (no key argument).

        Returns
        -------
        bool
        """
        await self._ensure_open()
        return await self._exists(self.path)

    async def set_if_not_existsb(self, default: Buffer) -> None:
        """Store a value at this store's path if not already present (no key argument).

        Parameters
        ----------
        default : Buffer
        """
        await self._ensure_open()
        await self._set_if_not_exists(self.path, default)

    async def is_emptyb(self) -> bool:
        """Check if any keys exist under this store's path (no key argument).

        Returns
        -------
        bool
        """
        return await self.is_empty("")

    async def delete_dirb(self) -> None:
        """Delete all keys under this store's path (no key argument)."""
        await self.delete_dir("")

    # -------------------------------------------------------------------------
    # Abstract methods for concrete store implementations
    #
    # These receive fully-qualified keys (with self.path already prepended).
    # Concrete stores implement these without worrying about path scoping.
    # -------------------------------------------------------------------------

    @abstractmethod
    async def _get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Retrieve the value associated with a given key.

        Parameters
        ----------
        key : str
        prototype : BufferPrototype
        byte_range : ByteRequest, optional

        Returns
        -------
        Buffer
        """
        ...

    @abstractmethod
    async def _set(self, key: str, value: Buffer) -> None:
        """Store a (key, value) pair.

        Parameters
        ----------
        key : str
        value : Buffer
        """
        ...

    @abstractmethod
    async def _delete(self, key: str) -> None:
        """Remove a key from the store

        Parameters
        ----------
        key : str
        """
        ...

    @abstractmethod
    async def _exists(self, key: str) -> bool:
        """Check if a key exists in the store.

        Parameters
        ----------
        key : str

        Returns
        -------
        bool
        """
        ...

    @abstractmethod
    async def _get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Retrieve possibly partial values from given key_ranges.

        Parameters
        ----------
        prototype : BufferPrototype
        key_ranges : Iterable[tuple[str, ByteRequest | None]]

        Returns
        -------
        list of values, in the order of the key_ranges, may contain null/none for missing keys
        """
        ...

    async def _set_if_not_exists(self, key: str, value: Buffer) -> None:
        """
        Store a key to ``value`` if the key is not already present.

        Parameters
        ----------
        key : str
        value : Buffer
        """
        # Note for implementers: the default implementation provided here
        # is not safe for concurrent writers. There's a race condition between
        # the `exists` check and the `set` where another writer could set some
        # value at `key` or delete `key`.
        if not await self._exists(key):
            await self._set(key, value)

    @abstractmethod
    def _list(self) -> AsyncIterator[str]:
        """Retrieve all keys in the store.

        Returns
        -------
        AsyncIterator[str]
        """
        # This method should be async, like overridden methods in child classes.
        # However, that's not straightforward:
        # https://stackoverflow.com/questions/68905848

    @abstractmethod
    def _list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """
        Retrieve all keys in the store that begin with a given prefix. Keys are returned relative
        to the root of the store.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        AsyncIterator[str]
        """
        # This method should be async, like overridden methods in child classes.
        # However, that's not straightforward:
        # https://stackoverflow.com/questions/68905848

    @abstractmethod
    def _list_dir(self, prefix: str) -> AsyncIterator[str]:
        """
        Retrieve all keys and prefixes with a given prefix and which do not contain the character
        "/" after the given prefix.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        AsyncIterator[str]
        """
        # This method should be async, like overridden methods in child classes.
        # However, that's not straightforward:
        # https://stackoverflow.com/questions/68905848

    # -------------------------------------------------------------------------
    # Remaining concrete methods
    # -------------------------------------------------------------------------

    async def clear(self) -> None:
        """
        Clear the store.

        Remove all keys and values from the store.
        """
        if not self.supports_deletes:
            raise NotImplementedError
        if not self.supports_listing:
            raise NotImplementedError
        self._check_writable()
        if self.path:
            await self.delete_dir("")
        else:
            # delete everything
            async for key in self._list():
                await self._delete(key)

    @property
    def read_only(self) -> bool:
        """Is the store read-only?"""
        return self._read_only

    def _check_writable(self) -> None:
        """Raise an exception if the store is not writable."""
        if self.read_only:
            raise ValueError("store was opened in read-only mode and does not support writing")

    @abstractmethod
    def __eq__(self, value: object) -> bool:
        """Equality comparison."""
        ...

    @property
    @abstractmethod
    def supports_writes(self) -> bool:
        """Does the store support writes?"""
        ...

    @property
    def supports_consolidated_metadata(self) -> bool:
        """
        Does the store support consolidated metadata?.

        If it doesn't an error will be raised on requests to consolidate the metadata.
        Returning `False` can be useful for stores which implement their own
        consolidation mechanism outside of the zarr-python implementation.
        """

        return True

    @property
    @abstractmethod
    def supports_deletes(self) -> bool:
        """Does the store support deletes?"""
        ...

    @property
    def supports_partial_writes(self) -> Literal[False]:
        """Does the store support partial writes?

        Partial writes are no longer used by Zarr, so this is always false.
        """
        return False

    @property
    @abstractmethod
    def supports_listing(self) -> bool:
        """Does the store support listing?"""
        ...

    def close(self) -> None:
        """Close the store."""
        self._is_open = False

    async def _get_bytes(
        self, key: str, *, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> bytes:
        """
        Retrieve raw bytes from the store asynchronously.

        This is a convenience method that wraps ``get()`` and converts the result
        to bytes. Use this when you need the raw byte content of a stored value.

        Parameters
        ----------
        key : str
            The key identifying the data to retrieve.
        prototype : BufferPrototype
            The buffer prototype to use for reading the data.
        byte_range : ByteRequest, optional
            If specified, only retrieve a portion of the stored data.
            Can be a ``RangeByteRequest``, ``OffsetByteRequest``, or ``SuffixByteRequest``.

        Returns
        -------
        bytes
            The raw bytes stored at the given key.

        Raises
        ------
        FileNotFoundError
            If the key does not exist in the store.
        """
        buffer = await self.get(key, prototype, byte_range)
        if buffer is None:
            raise FileNotFoundError(key)
        return buffer.to_bytes()

    def _get_bytes_sync(
        self, key: str = "", *, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> bytes:
        """
        Retrieve raw bytes from the store synchronously.

        Parameters
        ----------
        key : str, optional
            The key identifying the data to retrieve. Defaults to an empty string.
        prototype : BufferPrototype
            The buffer prototype to use for reading the data.
        byte_range : ByteRequest, optional
            If specified, only retrieve a portion of the stored data.

        Returns
        -------
        bytes
            The raw bytes stored at the given key.

        Raises
        ------
        FileNotFoundError
            If the key does not exist in the store.
        """

        return sync(self._get_bytes(key, prototype=prototype, byte_range=byte_range))

    async def _get_json(
        self, key: str, *, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Any:
        """
        Retrieve and parse JSON data from the store asynchronously.

        Parameters
        ----------
        key : str
            The key identifying the JSON data to retrieve.
        prototype : BufferPrototype
            The buffer prototype to use for reading the data.
        byte_range : ByteRequest, optional
            If specified, only retrieve a portion of the stored data.

        Returns
        -------
        Any
            The parsed JSON data.

        Raises
        ------
        FileNotFoundError
            If the key does not exist in the store.
        json.JSONDecodeError
            If the stored data is not valid JSON.
        """

        return json.loads(await self._get_bytes(key, prototype=prototype, byte_range=byte_range))

    def _get_json_sync(
        self, key: str = "", *, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Any:
        """
        Retrieve and parse JSON data from the store synchronously.

        Parameters
        ----------
        key : str, optional
            The key identifying the JSON data to retrieve. Defaults to an empty string.
        prototype : BufferPrototype
            The buffer prototype to use for reading the data.
        byte_range : ByteRequest, optional
            If specified, only retrieve a portion of the stored data.

        Returns
        -------
        Any
            The parsed JSON data.

        Raises
        ------
        FileNotFoundError
            If the key does not exist in the store.
        json.JSONDecodeError
            If the stored data is not valid JSON.
        """

        return sync(self._get_json(key, prototype=prototype, byte_range=byte_range))

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        """
        Insert multiple (key, value) pairs into storage.
        """
        await asyncio.gather(*starmap(self.set, values))

    async def _get_many(
        self, requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]]
    ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
        """
        Retrieve a collection of objects from storage. In general this method does not guarantee
        that objects will be retrieved in the order in which they were requested, so this method
        yields tuple[str, Buffer | None] instead of just Buffer | None
        """
        for req in requests:
            yield (req[0], await self.get(*req))

    async def getsize(self, key: str) -> int:
        """
        Return the size, in bytes, of a value in a Store.

        Parameters
        ----------
        key : str

        Returns
        -------
        nbytes : int
            The size of the value (in bytes).

        Raises
        ------
        FileNotFoundError
            When the given key does not exist in the store.
        """
        # Note to implementers: this default implementation is very inefficient since
        # it requires reading the entire object. Many systems will have ways to get the
        # size of an object without reading it.
        # avoid circular import
        from zarr.core.buffer.core import default_buffer_prototype

        value = await self.get(key, prototype=default_buffer_prototype())
        if value is None:
            raise FileNotFoundError(key)
        return len(value)

    async def getsize_prefix(self, prefix: str) -> int:
        """
        Return the size, in bytes, of all values under a prefix.

        Parameters
        ----------
        prefix : str
            The prefix of the directory to measure.

        Returns
        -------
        nbytes : int
            The sum of the sizes of the values in the directory (in bytes).

        See Also
        --------
        zarr.Array.nbytes_stored
        Store.getsize
        """
        # avoid circular import
        from zarr.core.common import concurrent_map
        from zarr.core.config import config

        keys = [(x,) async for x in self.list_prefix(prefix)]
        limit = config.get("async.concurrency")
        sizes = await concurrent_map(keys, self.getsize, limit=limit)
        return sum(sizes)


@runtime_checkable
class ByteGetter(Protocol):
    async def getb(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None: ...


@runtime_checkable
class ByteSetter(Protocol):
    async def getb(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None: ...

    async def setb(self, value: Buffer) -> None: ...

    async def deleteb(self) -> None: ...

    async def set_if_not_existsb(self, default: Buffer) -> None: ...


async def set_or_delete(byte_setter: ByteSetter, value: Buffer | None) -> None:
    """Set or delete a value in a byte setter

    Parameters
    ----------
    byte_setter : ByteSetter
    value : Buffer | None

    Notes
    -----
    If value is None, the key will be deleted.
    """
    if value is None:
        await byte_setter.deleteb()
    else:
        await byte_setter.setb(value)
