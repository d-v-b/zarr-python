"""An in-process backend over a dict."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, TypeVar

from zarr_storage._byte_range import resolve
from zarr_storage._types import (
    Copied,
    Deleted,
    Found,
    FoundRanges,
    Generation,
    Missing,
    ObjectMetadata,
    Unchanged,
    Written,
)
from zarr_storage.backends._common import IfMatch, always, children, if_absent, strictly_within

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence

    from zarr_storage._types import ByteRange, ListItem, Rejected

__all__ = ["MemoryBackend"]

R = TypeVar("R")


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

    def get(self, path: str, byte_range: ByteRange | None) -> Found | Missing:
        value = self._values.get(path)
        if value is None:
            return Missing(time.time())
        start, end = resolve(byte_range, len(value.data))
        return Found(memoryview(value.data)[start:end], Generation(value.generation), time.time())

    def get_if_changed(
        self, path: str, generation: Generation, byte_range: ByteRange | None
    ) -> Found | Missing | Unchanged:
        value = self._values.get(path)
        if value is not None and Generation(value.generation) == generation:
            return Unchanged(generation, time.time())
        return self.get(path, byte_range)

    def get_ranges(self, path: str, ranges: Sequence[ByteRange]) -> FoundRanges | Missing:
        value = self._values.get(path)
        if value is None:
            return Missing(time.time())
        view = memoryview(value.data)
        parts = tuple(view[slice(*resolve(r, len(value.data)))] for r in ranges)
        return FoundRanges(parts, Generation(value.generation), time.time())

    def head(self, path: str) -> ObjectMetadata | Missing:
        value = self._values.get(path)
        if value is None:
            return Missing(time.time())
        return ObjectMetadata(len(value.data), Generation(value.generation), value.last_modified)

    def get_into(
        self, path: str, out: memoryview, byte_range: ByteRange | None
    ) -> Copied | Missing:
        value = self._values.get(path)
        if value is None:
            return Missing(time.time())
        start, end = resolve(byte_range, len(value.data))
        n = end - start
        if n > len(out):
            raise ValueError(f"The output buffer holds {len(out)} bytes; the value has {n}.")
        out[:n] = memoryview(value.data)[start:end]
        return Copied(n, Generation(value.generation), time.time())

    # --- Writable

    def _put(
        self, path: str, value: memoryview, check: Callable[[Generation | None], R | None]
    ) -> Written | R:
        data = bytes(value)
        with self._lock:
            current = self._values.get(path)
            refusal = check(None if current is None else Generation(current.generation))
            if refusal is not None:
                return refusal
            generation = self._next_generation
            self._next_generation += 1
            self._values[path] = _Value(data, generation, time.time())
        return Written(Generation(generation))

    def put(self, path: str, value: memoryview) -> Written:
        return self._put(path, value, always)

    def put_if_absent(self, path: str, value: memoryview) -> Written | Rejected:
        return self._put(path, value, if_absent)

    def put_if_match(
        self, path: str, value: memoryview, generation: Generation
    ) -> Written | Rejected | Missing:
        return self._put(path, value, IfMatch(generation))

    def _delete(
        self, path: str, check: Callable[[Generation | None], R | None]
    ) -> Deleted | Missing | R:
        with self._lock:
            current = self._values.get(path)
            if current is None:
                return Missing(time.time())
            generation = Generation(current.generation)
            refusal = check(generation)
            if refusal is not None:
                return refusal
            del self._values[path]
        return Deleted(generation)

    def delete(self, path: str) -> Deleted | Missing:
        return self._delete(path, always)

    def delete_if_match(self, path: str, generation: Generation) -> Deleted | Rejected | Missing:
        return self._delete(path, IfMatch(generation))

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

    async def get_async(self, path: str, byte_range: ByteRange | None) -> Found | Missing:
        return self.get(path, byte_range)

    async def get_if_changed_async(
        self, path: str, generation: Generation, byte_range: ByteRange | None
    ) -> Found | Missing | Unchanged:
        return self.get_if_changed(path, generation, byte_range)

    async def get_ranges_async(
        self, path: str, ranges: Sequence[ByteRange]
    ) -> FoundRanges | Missing:
        return self.get_ranges(path, ranges)

    async def head_async(self, path: str) -> ObjectMetadata | Missing:
        return self.head(path)

    async def get_into_async(
        self, path: str, out: memoryview, byte_range: ByteRange | None
    ) -> Copied | Missing:
        return self.get_into(path, out, byte_range)

    async def put_async(self, path: str, value: memoryview) -> Written:
        return self.put(path, value)

    async def put_if_absent_async(self, path: str, value: memoryview) -> Written | Rejected:
        return self.put_if_absent(path, value)

    async def put_if_match_async(
        self, path: str, value: memoryview, generation: Generation
    ) -> Written | Rejected | Missing:
        return self.put_if_match(path, value, generation)

    async def delete_async(self, path: str) -> Deleted | Missing:
        return self.delete(path)

    async def delete_if_match_async(
        self, path: str, generation: Generation
    ) -> Deleted | Rejected | Missing:
        return self.delete_if_match(path, generation)

    async def delete_dir_async(self, directory: str) -> None:
        self.delete_dir(directory)

    async def list_async(self, directory: str) -> AsyncIterator[str]:
        for path in self.list(directory):
            yield path

    async def list_dir_async(self, directory: str) -> AsyncIterator[ListItem]:
        for item in self.list_dir(directory):
            yield item
