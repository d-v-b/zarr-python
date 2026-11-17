"""`Store` behavior that doesn't depend on a backend's own implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr_storage import (
    Copied,
    Found,
    FoundRanges,
    Generation,
    Missing,
    ObjectMetadata,
    OffsetRequest,
    RangeRequest,
    Store,
    SuffixRequest,
    Unchanged,
)
from zarr_storage._byte_range import resolve
from zarr_storage.backends import MemoryBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_storage import ByteRange


class _ReadOnlyBackend:
    """A backend with only the `Readable` capability, over a fixed dict."""

    def __init__(self, values: dict[str, bytes]) -> None:
        self._values = values

    @property
    def closed(self) -> bool:
        return False

    @property
    def writable(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def get(self, path: str, byte_range: ByteRange | None) -> Found | Missing:
        data = self._values.get(path)
        if data is None:
            return Missing()
        start, end = resolve(byte_range, len(data))
        return Found(memoryview(data)[start:end], Generation(0))

    def get_if_changed(
        self, path: str, generation: Generation, byte_range: ByteRange | None
    ) -> Found | Missing | Unchanged:
        raise NotImplementedError

    def get_ranges(self, path: str, ranges: Sequence[ByteRange]) -> FoundRanges | Missing:
        raise NotImplementedError

    def head(self, path: str) -> ObjectMetadata | Missing:
        data = self._values.get(path)
        return Missing() if data is None else ObjectMetadata(len(data), Generation(0), None)


class _DeviceBuffer:
    """Stands in for device memory (a GPU array): no buffer protocol, so host code can't view it."""

    def __init__(self, size: int) -> None:
        self._data = bytearray(size)

    def __len__(self) -> int:
        return len(self._data)

    def upload(self, data: memoryview) -> None:
        self._data[: len(data)] = data

    def download(self, n: int) -> bytes:
        return bytes(self._data[:n])


class _DeviceBackend(_ReadOnlyBackend):
    """A backend that reads into `_DeviceBuffer`s only, as a GPU direct storage backend would."""

    def get_into(
        self, path: str, out: _DeviceBuffer, byte_range: ByteRange | None
    ) -> Copied | Missing:
        found = self.get(path, byte_range)
        if isinstance(found, Missing):
            return found
        if len(found.value) > len(out):
            raise ValueError("The output buffer is too short.")
        out.upload(found.value)
        return Copied(len(found.value), found.generation)


@pytest.mark.parametrize(
    ("byte_range", "expected"),
    [(None, b"0123"), (RangeRequest(1, 3), b"12"), (SuffixRequest(1), b"3")],
)
def test_get_into_a_device_buffer(byte_range: ByteRange | None, expected: bytes) -> None:
    """`get_into` takes the buffer type the backend reads into, and applies the store's root."""
    store = Store(_DeviceBackend({"a/k": b"0123"})) / "a"
    out = _DeviceBuffer(4)
    assert store.get_into("k", out, byte_range=byte_range) == Copied(len(expected), Generation(0))
    assert out.download(len(expected)) == expected
    assert store.get_into("missing", out) == Missing()


def _static_buffer_checks(host: Store[MemoryBackend], device: Store[_DeviceBackend]) -> None:
    """Never run; pyright checks that each marked call is a type error."""
    host.get_into("k", bytearray(4))
    device.get_into("k", _DeviceBuffer(4))
    device.get_into("k", bytearray(4))  # pyright: ignore[reportArgumentType]
    host.get_into("k", _DeviceBuffer(4))  # pyright: ignore[reportArgumentType]
    Store(_ReadOnlyBackend({})).get_into("k", bytearray(4))  # pyright: ignore[reportAttributeAccessIssue]


def test_store_over_unwritable_backend_is_read_only() -> None:
    store = Store(_ReadOnlyBackend({}))
    assert store.read_only
    assert (store / "a").read_only


@pytest.mark.parametrize(
    "make",
    [
        lambda: RangeRequest(1.5, 3),  # pyright: ignore[reportArgumentType]
        lambda: RangeRequest(True, 3),
        lambda: OffsetRequest("1"),  # pyright: ignore[reportArgumentType]
        lambda: SuffixRequest(2.0),  # pyright: ignore[reportArgumentType]
    ],
    ids=["float", "bool", "str", "float-suffix"],
)
def test_byte_range_of_non_int_raises(make: object) -> None:
    with pytest.raises(TypeError, match="must be an int"):
        make()  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize(
    "make",
    [
        lambda: RangeRequest(-1, 3),
        lambda: RangeRequest(3, 1),
        lambda: OffsetRequest(-1),
        lambda: SuffixRequest(-1),
    ],
    ids=["negative-start", "end-before-start", "negative-offset", "negative-suffix"],
)
def test_byte_range_out_of_order_raises(make: object) -> None:
    with pytest.raises(ValueError, match="<=|>="):
        make()  # pyright: ignore[reportCallIssue]


def test_store_is_not_value_equal() -> None:
    backend = MemoryBackend()
    assert Store(backend) != Store(backend)
