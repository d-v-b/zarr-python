"""An in-process backend over a dict."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from zarr_storage._byte_range import resolve
from zarr_storage._types import (
    DeleteResult,
    Generation,
    ObjectMetadata,
    PutResult,
    RangesResult,
    ReadResult,
)
from zarr_storage.backends._common import children, strictly_within

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from zarr_storage._types import ByteRange, ListItem

__all__ = ["MemoryBackend"]


@dataclass(frozen=True, slots=True)
class _Value:
    data: bytes
    generation: int
    last_modified: float


class _State(TypedDict):
    values: dict[str, _Value]
    next_generation: int
    closed: bool


class MemoryBackend:
    """Values held in a dict in this process.

    Thread-safe. Generations are a counter that increases with every write. A pickled
    `MemoryBackend` carries a copy of its values; the copy and the original are independent.
    """

    def __init__(self) -> None:
        self._values: dict[str, _Value] = {}
        self._next_generation = 1
        self._lock = threading.Lock()
        self._closed = False

    def __repr__(self) -> str:
        return f"MemoryBackend(<{len(self._values)} keys>)"

    def __getstate__(self) -> _State:
        with self._lock:
            return {
                "values": dict(self._values),
                "next_generation": self._next_generation,
                "closed": self._closed,
            }

    def __setstate__(self, state: _State) -> None:
        self._values = state["values"]
        self._next_generation = state["next_generation"]
        self._lock = threading.Lock()
        self._closed = state["closed"]

    # --- Backend

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def writable(self) -> bool:
        return True

    def close(self) -> None:
        self._closed = True

    # --- Readable

    def get(self, path: str, byte_range: ByteRange | None) -> ReadResult | None:
        value = self._values.get(path)
        if value is None:
            return None
        start, end = resolve(byte_range, len(value.data))
        return ReadResult(memoryview(value.data)[start:end], Generation(value.generation))

    def get_ranges(self, path: str, ranges: Sequence[ByteRange]) -> RangesResult | None:
        value = self._values.get(path)
        if value is None:
            return None
        view = memoryview(value.data)
        size = len(value.data)
        parts = tuple(view[slice(*resolve(r, size))] for r in ranges)
        return RangesResult(parts, Generation(value.generation))

    def head(self, path: str) -> ObjectMetadata | None:
        value = self._values.get(path)
        if value is None:
            return None
        return ObjectMetadata(len(value.data), Generation(value.generation), value.last_modified)

    def get_into(self, path: str, out: memoryview, byte_range: ByteRange | None) -> int | None:
        value = self._values.get(path)
        if value is None:
            return None
        start, end = resolve(byte_range, len(value.data))
        n = end - start
        if n > len(out):
            raise ValueError(f"The output buffer holds {len(out)} bytes; the value has {n}.")
        out[:n] = memoryview(value.data)[start:end]
        return n

    # --- Writable

    def put(
        self, path: str, value: memoryview, *, if_match: Generation | None, if_none_match: bool
    ) -> PutResult:
        data = bytes(value)
        with self._lock:
            current = self._values.get(path)
            current_generation = None if current is None else Generation(current.generation)
            if if_none_match and current is not None:
                return PutResult(current_generation, applied=False)
            if if_match is not None and if_match != current_generation:
                return PutResult(current_generation, applied=False)
            generation = self._next_generation
            self._next_generation += 1
            self._values[path] = _Value(data, generation, time.time())
        return PutResult(Generation(generation))

    def delete(self, path: str, *, if_match: Generation | None) -> DeleteResult:
        with self._lock:
            current = self._values.get(path)
            if current is None:
                return DeleteResult(None, applied=False)
            current_generation = Generation(current.generation)
            if if_match is not None and if_match != current_generation:
                return DeleteResult(current_generation, applied=False)
            del self._values[path]
        return DeleteResult(current_generation, applied=True)

    def delete_dir(self, directory: str) -> None:
        with self._lock:
            for path in [p for p in self._values if strictly_within(p, directory)]:
                del self._values[path]

    # --- Listable

    def list(self, directory: str) -> Iterator[str]:
        with self._lock:
            paths = list(self._values)
        return (p for p in paths if strictly_within(p, directory))

    def list_dir(self, directory: str) -> Iterator[ListItem]:
        with self._lock:
            paths = list(self._values)
        return children(paths, directory)

    # --- async twins: memory never blocks, so they run inline

    async def get_async(self, path: str, byte_range: ByteRange | None) -> ReadResult | None:
        return self.get(path, byte_range)

    async def get_ranges_async(self, path: str, ranges: Sequence[ByteRange]) -> RangesResult | None:
        return self.get_ranges(path, ranges)

    async def head_async(self, path: str) -> ObjectMetadata | None:
        return self.head(path)

    async def get_into_async(
        self, path: str, out: memoryview, byte_range: ByteRange | None
    ) -> int | None:
        return self.get_into(path, out, byte_range)

    async def put_async(
        self, path: str, value: memoryview, *, if_match: Generation | None, if_none_match: bool
    ) -> PutResult:
        return self.put(path, value, if_match=if_match, if_none_match=if_none_match)

    async def delete_async(self, path: str, *, if_match: Generation | None) -> DeleteResult:
        return self.delete(path, if_match=if_match)

    async def delete_dir_async(self, directory: str) -> None:
        self.delete_dir(directory)

    async def list_async(self, directory: str) -> AsyncIterator[str]:
        for path in self.list(directory):
            yield path

    async def list_dir_async(self, directory: str) -> AsyncIterator[ListItem]:
        for item in self.list_dir(directory):
            yield item
