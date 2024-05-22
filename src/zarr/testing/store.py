from typing import Generic, TypeVar

import pytest

from zarr.abc.store import Store
from zarr.buffer import Buffer
from zarr.store.core import _normalize_interval_index
from zarr.testing.utils import assert_bytes_equal

S = TypeVar("S", bound=Store)


class StoreTests(Generic[S]):
    store_cls: type[S]

    def set(self, store: S, key: str, value: Buffer) -> None:
        """
        Insert key: value pairs into a store without using the store methods.
        """
        raise NotImplementedError

    def get(self, store: S, key: str) -> Buffer:
        """
        Retrieve values from a store without using the store methods.
        """

        raise NotImplementedError

    @pytest.fixture(scope="function")
    def store(self) -> Store:
        return self.store_cls()

    def test_store_type(self, store: S) -> None:
        assert isinstance(store, Store)
        assert isinstance(store, self.store_cls)

    def test_store_repr(self, store: S) -> None:
        raise NotImplementedError

    def test_store_supports_writes(self, store: S) -> None:
        raise NotImplementedError

    def test_store_supports_partial_writes(self, store: S) -> None:
        raise NotImplementedError

    def test_store_supports_listing(self, store: S) -> None:
        raise NotImplementedError

    @pytest.mark.parametrize("key", ["c/0", "foo/c/0.0", "foo/0/0"])
    @pytest.mark.parametrize("data", [b"\x01\x02\x03\x04", b""])
    @pytest.mark.parametrize("byte_range", (None, (0, None), (1, None), (1, 2), (None, 1)))
    async def test_get(
        self, store: S, key: str, data: bytes, byte_range: None | tuple[int | None, int | None]
    ) -> None:
        # insert values into the store
        data_buf = Buffer.from_bytes(data)
        self.set(store, key, data_buf)
        observed = await store.get(key, byte_range=byte_range)
        start, length = _normalize_interval_index(data_buf, interval=byte_range)
        expected = data_buf[start : start + length]
        assert_bytes_equal(observed, expected)

    @pytest.mark.parametrize("key", ["zarr.json", "c/0", "foo/c/0.0", "foo/0/0"])
    @pytest.mark.parametrize("data", [b"\x01\x02\x03\x04", b""])
    async def test_set(self, store: S, key: str, data: bytes) -> None:
        data_buf = Buffer.from_bytes(data)
        await store.set(key, data_buf)
        observed = self.get(store, key)
        assert_bytes_equal(observed, data_buf)

    @pytest.mark.parametrize(
        "key_ranges",
        (
            [],
            [("zarr.json", (0, 1))],
            [("c/0", (0, 1)), ("zarr.json", (0, 2))],
            [("c/0/0", (0, 1)), ("c/0/1", (0, 2)), ("c/0/2", (0, 3))],
        ),
    )
    async def test_get_partial_values(
        self, store: S, key_ranges: list[tuple[str, tuple[int, int]]]
    ) -> None:
        # put all of the data
        for key, _ in key_ranges:
            self.set(store, key, Buffer.from_bytes(bytes(key, encoding="utf-8")))

        # read back just part of it
        observed_maybe = await store.get_partial_values(key_ranges=key_ranges)

        observed: list[Buffer] = []
        expected: list[Buffer] = []

        for obs in observed_maybe:
            assert obs is not None
            observed.append(obs)

        for idx in range(len(observed)):
            key, byte_range = key_ranges[idx]
            result = await store.get(key, byte_range=byte_range)
            assert result is not None
            expected.append(result)

        assert all(
            obs.to_bytes() == exp.to_bytes() for obs, exp in zip(observed, expected, strict=True)
        )

    async def test_exists(self, store: S) -> None:
        assert not await store.exists("foo")
        await store.set("foo/zarr.json", Buffer.from_bytes(b"bar"))
        assert await store.exists("foo/zarr.json")

    async def test_delete(self, store: S) -> None:
        await store.set("foo/zarr.json", Buffer.from_bytes(b"bar"))
        assert await store.exists("foo/zarr.json")
        await store.delete("foo/zarr.json")
        assert not await store.exists("foo/zarr.json")

    async def test_list(self, store: S) -> None:
        assert [k async for k in store.list()] == []
        await store.set("foo/zarr.json", Buffer.from_bytes(b"bar"))
        keys = [k async for k in store.list()]
        assert keys == ["foo/zarr.json"], keys

        expected = ["foo/zarr.json"]
        for i in range(10):
            key = f"foo/c/{i}"
            expected.append(key)
            await store.set(
                f"foo/c/{i}", Buffer.from_bytes(i.to_bytes(length=3, byteorder="little"))
            )

    async def test_list_prefix(self, store: S) -> None:
        # TODO: we currently don't use list_prefix anywhere
        raise NotImplementedError

    async def test_list_dir(self, store: S) -> None:
        assert [k async for k in store.list_dir("")] == []
        assert [k async for k in store.list_dir("foo")] == []
        await store.set("foo/zarr.json", Buffer.from_bytes(b"bar"))
        await store.set("foo/c/1", Buffer.from_bytes(b"\x01"))

        keys = [k async for k in store.list_dir("foo")]
        assert set(keys) == set(["zarr.json", "c"]), keys

        keys = [k async for k in store.list_dir("foo/")]
        assert set(keys) == set(["zarr.json", "c"]), keys
