from __future__ import annotations

import numpy as np
import pytest

import zarr
from zarr.core.buffer import cpu
from zarr.core.sync import sync
from zarr.storage import MemoryStore

pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.testclient import TestClient

from zarr.experimental.serve import serve_node, serve_store


@pytest.fixture
def memory_store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def group_with_arrays(memory_store: MemoryStore) -> zarr.Group:
    """Create a group containing a regular array and a sharded array."""
    root = zarr.open_group(memory_store, mode="w")
    zarr.create_array(root.store_path / "regular", shape=(4, 4), chunks=(2, 2), dtype="f8")
    zarr.create_array(
        root.store_path / "sharded",
        shape=(8, 8),
        chunks=(2, 2),
        shards=(4, 4),
        dtype="i4",
    )
    return root


class TestServeNodeDoesNotExposeNonZarrKeys:
    """serve_node must never expose keys that are not part of the zarr hierarchy."""

    def test_non_zarr_key_returns_404(
        self, memory_store: MemoryStore, group_with_arrays: zarr.Group
    ) -> None:
        # Plant a non-zarr file in the store.
        non_zarr_buf = cpu.buffer_prototype.buffer.from_bytes(b"secret data")
        sync(memory_store.set("secret.txt", non_zarr_buf))

        app = serve_node(group_with_arrays)
        client = TestClient(app)

        # The non-zarr key must not be accessible.
        response = client.get("/secret.txt")
        assert response.status_code == 404

    def test_non_zarr_key_nested_returns_404(
        self, memory_store: MemoryStore, group_with_arrays: zarr.Group
    ) -> None:
        # Plant a non-zarr file under a path that shares a prefix with a
        # real array, but is not a valid chunk or metadata key.
        non_zarr_buf = cpu.buffer_prototype.buffer.from_bytes(b"not a chunk")
        sync(memory_store.set("regular/notes.txt", non_zarr_buf))

        app = serve_node(group_with_arrays)
        client = TestClient(app)

        response = client.get("/regular/notes.txt")
        assert response.status_code == 404

    def test_valid_metadata_is_accessible(self, group_with_arrays: zarr.Group) -> None:
        app = serve_node(group_with_arrays)
        client = TestClient(app)

        # Root group metadata
        response = client.get("/zarr.json")
        assert response.status_code == 200

        # Array metadata
        response = client.get("/regular/zarr.json")
        assert response.status_code == 200

    def test_valid_chunk_is_accessible(self, group_with_arrays: zarr.Group) -> None:
        # Write some data so the chunk actually exists in the store.
        arr = group_with_arrays["regular"]
        arr[:] = np.ones((4, 4))

        app = serve_node(group_with_arrays)
        client = TestClient(app)

        # c/0/0 is a valid chunk key for a (4,4) array with (2,2) chunks.
        response = client.get("/regular/c/0/0")
        assert response.status_code == 200


class TestShardedArrayByteRangeReads:
    """Byte-range reads against a sharded array served via serve_node."""

    def test_range_read_returns_206(self, group_with_arrays: zarr.Group) -> None:
        arr = group_with_arrays["sharded"]
        arr[:] = np.arange(64, dtype="i4").reshape((8, 8))

        app = serve_node(group_with_arrays)
        client = TestClient(app)

        # c/0/0 is the first shard key for an (8,8) array with (4,4) shards.
        full_response = client.get("/sharded/c/0/0")
        assert full_response.status_code == 200
        full_body = full_response.content

        # Request the first 8 bytes.
        range_response = client.get("/sharded/c/0/0", headers={"Range": "bytes=0-7"})
        assert range_response.status_code == 206
        assert range_response.content == full_body[:8]

    def test_suffix_range_read(self, group_with_arrays: zarr.Group) -> None:
        arr = group_with_arrays["sharded"]
        arr[:] = np.arange(64, dtype="i4").reshape((8, 8))

        app = serve_node(group_with_arrays)
        client = TestClient(app)

        full_response = client.get("/sharded/c/0/0")
        full_body = full_response.content

        # Request the last 4 bytes.
        range_response = client.get("/sharded/c/0/0", headers={"Range": "bytes=-4"})
        assert range_response.status_code == 206
        assert range_response.content == full_body[-4:]

    def test_offset_range_read(self, group_with_arrays: zarr.Group) -> None:
        arr = group_with_arrays["sharded"]
        arr[:] = np.arange(64, dtype="i4").reshape((8, 8))

        app = serve_node(group_with_arrays)
        client = TestClient(app)

        full_response = client.get("/sharded/c/0/0")
        full_body = full_response.content

        # Request everything from byte 4 onward.
        range_response = client.get("/sharded/c/0/0", headers={"Range": "bytes=4-"})
        assert range_response.status_code == 206
        assert range_response.content == full_body[4:]


class TestWriteViaPut:
    """serve_store and serve_node can be configured to accept PUT writes."""

    def test_put_writes_to_store(self, memory_store: MemoryStore) -> None:
        app = serve_store(memory_store, methods={"GET", "PUT"})
        client = TestClient(app)

        payload = b"hello zarr"
        response = client.put("/some/key", content=payload)
        assert response.status_code == 204

        # Verify the data landed in the store.
        buf = sync(memory_store.get("some/key", cpu.buffer_prototype))
        assert buf is not None
        assert buf.to_bytes() == payload

    def test_put_then_get_roundtrip(self, memory_store: MemoryStore) -> None:
        app = serve_store(memory_store, methods={"GET", "PUT"})
        client = TestClient(app)

        payload = b"\x00\x01\x02\x03"
        client.put("/data/blob", content=payload)

        response = client.get("/data/blob")
        assert response.status_code == 200
        assert response.content == payload

    def test_put_rejected_when_not_configured(self, memory_store: MemoryStore) -> None:
        # Default methods is {"GET"} only.
        app = serve_store(memory_store)
        client = TestClient(app)

        response = client.put("/some/key", content=b"data")
        assert response.status_code == 405

    def test_put_on_node_validates_key(
        self, memory_store: MemoryStore, group_with_arrays: zarr.Group
    ) -> None:
        app = serve_node(group_with_arrays, methods={"GET", "PUT"})
        client = TestClient(app)

        # Writing to a non-zarr key should be rejected.
        response = client.put("/not_a_zarr_key.bin", content=b"data")
        assert response.status_code == 404
