"""`Store` behavior that doesn't depend on a backend's own implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr_storage import (
    Generation,
    ObjectMetadata,
    OffsetRequest,
    RangeRequest,
    RangesResult,
    ReadResult,
    Store,
    SuffixRequest,
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

    def get(self, path: str, byte_range: ByteRange | None) -> ReadResult | None:
        data = self._values.get(path)
        if data is None:
            return None
        start, end = resolve(byte_range, len(data))
        return ReadResult(memoryview(data)[start:end], Generation(0))

    def get_ranges(self, path: str, ranges: Sequence[ByteRange]) -> RangesResult | None:
        raise NotImplementedError

    def head(self, path: str) -> ObjectMetadata | None:
        data = self._values.get(path)
        return None if data is None else ObjectMetadata(len(data), Generation(0), None)


@pytest.mark.parametrize(
    ("byte_range", "expected"),
    [(None, b"0123"), (RangeRequest(1, 3), b"12"), (SuffixRequest(1), b"3")],
)
def test_get_into_without_backend_support(byte_range: ByteRange | None, expected: bytes) -> None:
    """A backend without `get_into` is read with `get` and copied into the buffer."""
    store = Store(_ReadOnlyBackend({"a/k": b"0123"})) / "a"
    out = bytearray(4)
    n = store.get_into("k", out, byte_range=byte_range)
    assert (n, bytes(out[: n or 0])) == (len(expected), expected)
    assert store.get_into("missing", out) is None


def test_get_into_without_backend_support_short_buffer_raises() -> None:
    store = Store(_ReadOnlyBackend({"k": b"0123"}))
    with pytest.raises(ValueError, match="holds"):
        store.get_into("k", bytearray(3))


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
