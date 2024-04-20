from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Optional, MutableMapping, List, Tuple

from zarr.v3.common import BytesLike
from zarr.v3.abc.store import Store


class MemoryStore(Store):
    supports_writes: bool = True
    supports_partial_writes: bool = True
    supports_listing: bool = True

    _store_dict: MutableMapping[str, bytes]

    def __init__(self, store_dict: Optional[MutableMapping[str, bytes]] = None):
        self._store_dict = store_dict or {}

    def __str__(self) -> str:
        return f"memory://{id(self._store_dict)}"

    def __repr__(self) -> str:
        return f"MemoryStore({repr(str(self))})"

    async def get(
        self, key: str, byte_range: Optional[Tuple[int, Optional[int]]] = None
    ) -> Optional[BytesLike]:
        assert isinstance(key, str)
        try:
            value = self._store_dict[key]
            if byte_range is not None:
                value = value[byte_range[0] : byte_range[1]]
            return value
        except KeyError:
            return None

    async def get_metadata(self, key: str) -> Optional[dict[str, Any]]:
        try:
            return self._store_dict[key]
        except KeyError:
            return None

    async def get_partial_values(
        self, key_ranges: List[Tuple[str, Tuple[int, int]]]
    ) -> List[bytes]:
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        return key in self._store_dict

    async def set(
        self, key: str, value: BytesLike, byte_range: Optional[Tuple[int, int]] = None
    ) -> None:
        assert isinstance(key, str)
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(f"Expected BytesLike. Got {type(value)}.")

        if byte_range is not None:
            buf = bytearray(self._store_dict[key])
            buf[byte_range[0] : byte_range[1]] = value
            self._store_dict[key] = buf
        else:
            self._store_dict[key] = value

    async def set_metadata(self, key: str, metadata: dict[str, Any]) -> None:
        self._store_dict[key] = metadata

    async def delete(self, key: str) -> None:
        try:
            del self._store_dict[key]
        except KeyError:
            pass  # Q(JH): why not raise?

    async def set_partial_values(self, key_start_values: List[Tuple[str, int, bytes]]) -> None:
        raise NotImplementedError

    async def list(self) -> AsyncGenerator[str, None]:
        for key in self._store_dict:
            yield key

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        for key in self._store_dict:
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        for key in self._store_dict:
            if key.startswith(prefix + "/") and key != prefix:
                yield key.removeprefix(prefix + "/").rsplit("/", maxsplit=1)[0]
