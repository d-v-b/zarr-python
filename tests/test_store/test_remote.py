from __future__ import annotations

import json

import fsspec
import pytest
from upath import UPath

import zarr.api.asynchronous
from zarr.core.buffer import Buffer, cpu, default_buffer_prototype
from zarr.core.sync import _collect_aiterator
from zarr.storage import RemoteStore
from zarr.testing.store import StoreTests

from ..conftest import endpoint_url, test_bucket_name


async def test_basic() -> None:
    store = RemoteStore.from_url(
        f"s3://{test_bucket_name}/foo/spam/",
        mode="w",
        storage_options={"endpoint_url": endpoint_url, "anon": False},
    )
    assert store.fs.asynchronous
    assert store.path == f"{test_bucket_name}/foo/spam"
    assert await _collect_aiterator(store.list()) == ()
    assert not await store.exists("foo")
    data = b"hello"
    await store.set("foo", cpu.Buffer.from_bytes(data))
    assert await store.exists("foo")
    assert (await store.get("foo", prototype=default_buffer_prototype())).to_bytes() == data
    out = await store.get_partial_values(
        prototype=default_buffer_prototype(), key_ranges=[("foo", (1, None))]
    )
    assert out[0].to_bytes() == data[1:]


class TestRemoteStoreS3(StoreTests[RemoteStore, cpu.Buffer]):
    store_cls = RemoteStore
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self, request) -> dict[str, str | bool]:
        fs, path = fsspec.url_to_fs(
            f"s3://{test_bucket_name}", endpoint_url=endpoint_url, anon=False, asynchronous=True
        )
        return {"fs": fs, "path": path, "mode": "r+"}

    @pytest.fixture
    def store(self, store_kwargs: dict[str, str | bool]) -> RemoteStore:
        return self.store_cls(**store_kwargs)

    async def get(self, store: RemoteStore, key: str) -> Buffer:
        #  make a new, synchronous instance of the filesystem because this test is run in sync code
        new_fs = fsspec.filesystem(
            "s3", endpoint_url=store.fs.endpoint_url, anon=store.fs.anon, asynchronous=False
        )
        return self.buffer_cls.from_bytes(new_fs.cat(f"{store.path}/{key}"))

    async def set(self, store: RemoteStore, key: str, value: Buffer) -> None:
        #  make a new, synchronous instance of the filesystem because this test is run in sync code
        new_fs = fsspec.filesystem(
            "s3", endpoint_url=store.fs.endpoint_url, anon=store.fs.anon, asynchronous=False
        )
        new_fs.write_bytes(f"{store.path}/{key}", value.to_bytes())

    def test_store_repr(self, store: RemoteStore) -> None:
        assert str(store) == "<RemoteStore(S3FileSystem, test)>"

    def test_store_supports_writes(self, store: RemoteStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: RemoteStore) -> None:
        assert not store.supports_partial_writes

    def test_store_supports_listing(self, store: RemoteStore) -> None:
        assert store.supports_listing

    async def test_remote_store_from_uri(self, store: RemoteStore):
        storage_options = {
            "endpoint_url": endpoint_url,
            "anon": False,
        }

        meta = {"attributes": {"key": "value"}, "zarr_format": 3, "node_type": "group"}

        await store.set(
            "zarr.json",
            self.buffer_cls.from_bytes(json.dumps(meta).encode()),
        )
        group = await zarr.api.asynchronous.open_group(
            store=f"s3://{test_bucket_name}", storage_options=storage_options
        )
        assert dict(group.attrs) == {"key": "value"}

        meta["attributes"]["key"] = "value-2"
        await store.set(
            "directory-2/zarr.json",
            self.buffer_cls.from_bytes(json.dumps(meta).encode()),
        )
        group = await zarr.api.asynchronous.open_group(
            store=f"s3://{test_bucket_name}/directory-2", storage_options=storage_options
        )
        assert dict(group.attrs) == {"key": "value-2"}

        meta["attributes"]["key"] = "value-3"
        await store.set(
            "directory-3/zarr.json",
            self.buffer_cls.from_bytes(json.dumps(meta).encode()),
        )
        group = await zarr.api.asynchronous.open_group(
            store=f"s3://{test_bucket_name}", path="directory-3", storage_options=storage_options
        )
        assert dict(group.attrs) == {"key": "value-3"}

    def test_from_upath(self) -> None:
        path = UPath(
            f"s3://{test_bucket_name}/foo/bar/",
            endpoint_url=endpoint_url,
            anon=False,
            asynchronous=True,
        )
        result = RemoteStore.from_upath(path)
        assert result.fs.endpoint_url == endpoint_url
        assert result.fs.asynchronous
        assert result.path == f"{test_bucket_name}/foo/bar"

    def test_init_raises_if_path_has_scheme(self, store_kwargs) -> None:
        # regression test for https://github.com/zarr-developers/zarr-python/issues/2342
        store_kwargs["path"] = "s3://" + store_kwargs["path"]
        with pytest.raises(
            ValueError, match="path argument to RemoteStore must not include scheme .*"
        ):
            self.store_cls(**store_kwargs)

    def test_init_warns_if_fs_asynchronous_is_false(self) -> None:
        fs, path = fsspec.url_to_fs(
            f"s3://{test_bucket_name}", endpoint_url=endpoint_url, anon=False, asynchronous=False
        )
        store_kwargs = {"fs": fs, "path": path, "mode": "r+"}
        with pytest.warns(UserWarning, match=r".* was not created with `asynchronous=True`.*"):
            self.store_cls(**store_kwargs)

    async def test_empty_nonexistent_path(self, store_kwargs) -> None:
        # regression test for https://github.com/zarr-developers/zarr-python/pull/2343
        store_kwargs["mode"] = "w-"
        store_kwargs["path"] += "/abc"
        store = await self.store_cls.open(**store_kwargs)
        assert await store.empty()
