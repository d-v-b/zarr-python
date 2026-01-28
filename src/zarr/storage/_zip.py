from __future__ import annotations

import os
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

ZipStoreAccessModeLiteral = Literal["r", "w", "a"]


class ZipStore(Store):
    """
    Store using a ZIP file.

    Parameters
    ----------
    zip_path : str or Path
        Location of the ZIP file.
    mode : str, optional
        One of 'r' to read an existing file, 'w' to truncate and write a new
        file, 'a' to append to an existing file, or 'x' to exclusively create
        and write a new file.
    compression : int, optional
        Compression method to use when writing to the archive.
    allowZip64 : bool, optional
        If True (the default) will create ZIP files that use the ZIP64
        extensions when the zipfile is larger than 2 GiB. If False
        will raise an exception when the ZIP file would require ZIP64
        extensions.

    Attributes
    ----------
    allowed_exceptions
    supports_writes
    supports_deletes
    supports_listing
    zip_path
    compression
    allowZip64
    """

    supports_writes: bool = True
    supports_deletes: bool = False
    supports_listing: bool = True

    zip_path: Path
    compression: int
    allowZip64: bool

    _zf: zipfile.ZipFile
    _lock: threading.RLock

    def __init__(
        self,
        zip_path: Path | str,
        *,
        mode: ZipStoreAccessModeLiteral = "r",
        read_only: bool | None = None,
        compression: int = zipfile.ZIP_STORED,
        allowZip64: bool = True,
        path: str = "",
    ) -> None:
        if read_only is None:
            read_only = mode == "r"

        super().__init__(read_only=read_only, path=path)

        if isinstance(zip_path, str):
            zip_path = Path(zip_path)
        assert isinstance(zip_path, Path)
        self.zip_path = zip_path

        self._zmode = mode
        self.compression = compression
        self.allowZip64 = allowZip64

    def with_path(self, path: str) -> Self:
        raise NotImplementedError(
            "ZipStore does not support with_path. "
            "Use StorePath(store, path) instead for path-scoped access."
        )

    def _sync_open(self) -> None:
        if self._is_open:
            raise ValueError("store is already open")

        self._lock = threading.RLock()

        self._zf = zipfile.ZipFile(
            self.zip_path,
            mode=self._zmode,
            compression=self.compression,
            allowZip64=self.allowZip64,
        )

        self._is_open = True

    async def _open(self) -> None:
        self._sync_open()

    def __getstate__(self) -> dict[str, Any]:
        # We need a copy to not modify the state of the original store
        state = self.__dict__.copy()
        for attr in ["_zf", "_lock"]:
            state.pop(attr, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state
        self._is_open = False
        self._sync_open()

    def close(self) -> None:
        # docstring inherited
        super().close()
        with self._lock:
            self._zf.close()

    async def clear(self) -> None:
        # docstring inherited
        with self._lock:
            self._check_writable()
            self._zf.close()
            os.remove(self.zip_path)
            self._zf = zipfile.ZipFile(
                self.zip_path, mode="w", compression=self.compression, allowZip64=self.allowZip64
            )

    def __str__(self) -> str:
        return f"zip://{self.zip_path}"

    def __repr__(self) -> str:
        return f"ZipStore('{self}')"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.zip_path == other.zip_path

    def _read_item(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Synchronous helper to read from the zip file (must be called under lock)."""
        if not self._is_open:
            self._sync_open()
        try:
            with self._zf.open(key) as f:  # will raise KeyError
                if byte_range is None:
                    return prototype.buffer.from_bytes(f.read())
                elif isinstance(byte_range, RangeByteRequest):
                    f.seek(byte_range.start)
                    return prototype.buffer.from_bytes(f.read(byte_range.end - f.tell()))
                size = f.seek(0, os.SEEK_END)
                if isinstance(byte_range, OffsetByteRequest):
                    f.seek(byte_range.offset)
                elif isinstance(byte_range, SuffixByteRequest):
                    f.seek(max(0, size - byte_range.suffix))
                else:
                    raise TypeError(f"Unexpected byte_range, got {byte_range}.")
                return prototype.buffer.from_bytes(f.read())
        except KeyError:
            return None

    async def _get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # docstring inherited
        assert isinstance(key, str)

        with self._lock:
            return self._read_item(key, prototype=prototype, byte_range=byte_range)

    async def _get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # docstring inherited
        out = []
        with self._lock:
            for key, byte_range in key_ranges:
                out.append(self._read_item(key, prototype=prototype, byte_range=byte_range))
        return out

    def _write_item(self, key: str, value: Buffer) -> None:
        """Synchronous helper to write to the zip file (must be called under lock)."""
        if not self._is_open:
            self._sync_open()
        # generally, this should be called inside a lock
        keyinfo = zipfile.ZipInfo(filename=key, date_time=time.localtime(time.time())[:6])
        keyinfo.compress_type = self.compression
        if keyinfo.filename[-1] == os.sep:
            keyinfo.external_attr = 0o40775 << 16  # drwxrwxr-x
            keyinfo.external_attr |= 0x10  # MS-DOS directory flag
        else:
            keyinfo.external_attr = 0o644 << 16  # ?rw-r--r--
        self._zf.writestr(keyinfo, value.to_bytes())

    async def _set(self, key: str, value: Buffer) -> None:
        # docstring inherited
        self._check_writable()
        if not self._is_open:
            self._sync_open()
        assert isinstance(key, str)
        if not isinstance(value, Buffer):
            raise TypeError(
                f"ZipStore.set(): `value` must be a Buffer instance. Got an instance of {type(value)} instead."
            )
        with self._lock:
            self._write_item(key, value)

    async def _set_if_not_exists(self, key: str, value: Buffer) -> None:
        self._check_writable()
        with self._lock:
            members = self._zf.namelist()
            if key not in members:
                self._write_item(key, value)

    async def _delete_dir(self, prefix: str) -> None:
        # only raise NotImplementedError if any keys are found
        self._check_writable()
        if prefix != "" and not prefix.endswith("/"):
            prefix += "/"
        async for _ in self._list_prefix(prefix):
            raise NotImplementedError

    async def _delete(self, key: str) -> None:
        # docstring inherited
        # we choose to only raise NotImplementedError here if the key exists
        # this allows the array/group APIs to avoid the overhead of existence checks
        self._check_writable()
        if await self._exists(key):
            raise NotImplementedError

    async def _exists(self, key: str) -> bool:
        # docstring inherited
        with self._lock:
            try:
                self._zf.getinfo(key)
            except KeyError:
                return False
            else:
                return True

    async def _list(self) -> AsyncIterator[str]:
        # docstring inherited
        with self._lock:
            for key in self._zf.namelist():
                yield key

    async def _list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        async for key in self._list():
            if key.startswith(prefix):
                yield key

    async def _list_dir(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        prefix = prefix.rstrip("/")

        keys = self._zf.namelist()
        seen = set()
        if prefix == "":
            keys_unique = {k.split("/")[0] for k in keys}
            for key in keys_unique:
                if key not in seen:
                    seen.add(key)
                    yield key
        else:
            for key in keys:
                if key.startswith(prefix + "/") and key.strip("/") != prefix:
                    k = key.removeprefix(prefix + "/").split("/")[0]
                    if k not in seen:
                        seen.add(k)
                        yield k

    async def move(self, new_zip_path: Path | str) -> None:
        """
        Move the store to another path.
        """
        if isinstance(new_zip_path, str):
            new_zip_path = Path(new_zip_path)
        self.close()
        os.makedirs(new_zip_path.parent, exist_ok=True)
        shutil.move(self.zip_path, new_zip_path)
        self.zip_path = new_zip_path
        await self._open()
