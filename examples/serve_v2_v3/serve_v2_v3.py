# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "zarr[server]",
#   "httpx",
# ]
# ///
#
"""
Serve a Zarr v2 array over HTTP as a Zarr v3 array.

This example demonstrates how to build a read-only ``Store`` that translates
between Zarr formats on the fly.  A v2 array lives in a ``MemoryStore``; the
custom ``V2AsV3Store`` intercepts reads and translates metadata:

* Requests for ``zarr.json`` are answered by reading the v2 ``.zarray`` /
  ``.zattrs`` metadata and converting it to a v3 ``zarr.json`` document
  using the same ``_convert_array_metadata`` helper that powers the
  ``zarr migrate v3`` CLI command.

* Chunk keys are passed through unchanged because ``_convert_array_metadata``
  preserves the v2 chunk key encoding (``V2ChunkKeyEncoding``).  A v3
  client reads the encoding from ``zarr.json`` and naturally produces the
  same keys (e.g. ``0.0``) that the v2 store already contains.

* The v2 metadata files (``.zarray``, ``.zattrs``) are hidden so only
  v3 keys are visible.

The translated store is then served over HTTP with ``serve_store``.  A test
at the bottom opens the served data *as a v3 array* and verifies it can
read the values back.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

import zarr
from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, cpu
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.sync import sync
from zarr.metadata.migrate_v3 import _convert_array_metadata
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from zarr.core.buffer import BufferPrototype


# ---------------------------------------------------------------------------
# Custom store that presents v2 data as v3
# ---------------------------------------------------------------------------

# v2 metadata keys that should be hidden from v3 clients.
_HIDDEN_V2_KEYS = frozenset({ZARRAY_JSON, ZATTRS_JSON})


class V2AsV3Store(Store):
    """A read-only store that wraps an existing v2 store and presents it as v3.

    Metadata translation
    --------------------
    ``zarr.json``  ←  ``.zarray`` + ``.zattrs``  (converted via
    ``_convert_array_metadata`` from the CLI migration module)

    Chunk keys
    ----------
    Chunk keys are **not** translated.  The v3 metadata produced by
    ``_convert_array_metadata`` uses ``V2ChunkKeyEncoding`` with the
    same separator as the original v2 array, so chunk keys like ``0.0``
    are valid in both formats.

    Visibility
    ----------
    The v2 metadata files (``.zarray``, ``.zattrs``) are hidden from
    listing and ``get`` so that clients only see v3 keys.
    """

    supports_writes: bool = False
    supports_deletes: bool = False
    supports_listing: bool = True

    def __init__(self, v2_store: Store) -> None:
        super().__init__(read_only=True)
        self._v2 = v2_store

    def __eq__(self, other: object) -> bool:
        return isinstance(other, V2AsV3Store) and self._v2 == other._v2

    # -- metadata conversion -----------------------------------------------

    async def _build_zarr_json(self, prototype: BufferPrototype) -> Buffer | None:
        """Read v2 metadata from the wrapped store and return a v3
        ``zarr.json`` buffer."""
        zarray_buf = await self._v2.get(ZARRAY_JSON, prototype)
        if zarray_buf is None:
            return None

        zarray_dict = json.loads(zarray_buf.to_bytes())
        v2_meta = ArrayV2Metadata.from_dict(zarray_dict)

        # Merge in .zattrs if present.
        zattrs_buf = await self._v2.get(ZATTRS_JSON, prototype)
        if zattrs_buf is not None:
            attrs = json.loads(zattrs_buf.to_bytes())
            if attrs:
                v2_meta = v2_meta.update_attributes(attrs)

        # Reuse the same conversion the CLI uses.
        v3_meta = _convert_array_metadata(v2_meta)
        v3_json = json.dumps(v3_meta.to_dict()).encode()
        return prototype.buffer.from_bytes(v3_json)

    # -- Store ABC implementation ------------------------------------------

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if prototype is None:
            prototype = default_buffer_prototype()
        await self._ensure_open()

        # Synthesise zarr.json from v2 metadata.
        if key == ZARR_JSON:
            buf = await self._build_zarr_json(prototype)
            if buf is None or byte_range is None:
                return buf
            from zarr.storage._utils import _normalize_byte_range_index

            start, stop = _normalize_byte_range_index(buf, byte_range)
            return prototype.buffer.from_buffer(buf[start:stop])

        # Hide v2 metadata files.
        if key in _HIDDEN_V2_KEYS:
            return None

        # All other keys (chunk keys) pass through unchanged.
        return await self._v2.get(key, prototype, byte_range=byte_range)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [await self.get(k, prototype, br) for k, br in key_ranges]

    async def exists(self, key: str) -> bool:
        if key == ZARR_JSON:
            return await self._v2.exists(ZARRAY_JSON)
        if key in _HIDDEN_V2_KEYS:
            return False
        return await self._v2.exists(key)

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("V2AsV3Store is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("V2AsV3Store is read-only")

    async def list(self) -> AsyncIterator[str]:
        async for key in self._v2.list():
            if key == ZARRAY_JSON:
                yield ZARR_JSON
            elif key in _HIDDEN_V2_KEYS:
                continue
            else:
                yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix) :]
            if "/" not in remainder:
                yield key


# ---------------------------------------------------------------------------
# Demo / tests
# ---------------------------------------------------------------------------


def create_v2_array() -> tuple[MemoryStore, np.ndarray]:
    """Create a v2 array with some data and return the store + data."""
    store = MemoryStore()
    data = np.arange(16, dtype="float64").reshape(4, 4)
    arr = zarr.create_array(store, shape=data.shape, chunks=(2, 2), dtype=data.dtype, zarr_format=2)
    arr[:] = data
    return store, data


def test_metadata_translation() -> None:
    """The translated zarr.json should be valid v3 metadata."""
    v2_store, _ = create_v2_array()
    v3_store = V2AsV3Store(v2_store)

    buf = sync(v3_store.get(ZARR_JSON, cpu.buffer_prototype))
    assert buf is not None
    meta = json.loads(buf.to_bytes())

    assert meta["zarr_format"] == 3
    assert meta["node_type"] == "array"
    assert meta["shape"] == [4, 4]
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [2, 2]
    # The v2 chunk key encoding is preserved.
    assert meta["chunk_key_encoding"]["name"] == "v2"
    assert any(c["name"] in ("bytes", "zstd", "blosc") for c in meta["codecs"])
    print("  metadata translation: OK")
    print(f"  zarr.json:\n{json.dumps(meta, indent=2)}")


def test_chunk_passthrough() -> None:
    """Chunk keys should pass through unchanged (v2 encoding preserved)."""
    v2_store, _ = create_v2_array()
    v3_store = V2AsV3Store(v2_store)

    # The v2 store has chunk key "0.0"; the v3 store should serve the
    # same key since the metadata says V2ChunkKeyEncoding.
    v2_buf = sync(v2_store.get("0.0", cpu.buffer_prototype))
    v3_buf = sync(v3_store.get("0.0", cpu.buffer_prototype))
    assert v2_buf is not None
    assert v3_buf is not None
    assert v2_buf.to_bytes() == v3_buf.to_bytes()
    print("  chunk passthrough: OK")


def test_v2_metadata_hidden() -> None:
    """v2 metadata files should not be visible."""
    v2_store, _ = create_v2_array()
    v3_store = V2AsV3Store(v2_store)

    assert sync(v3_store.get(ZARRAY_JSON, cpu.buffer_prototype)) is None
    assert sync(v3_store.get(ZATTRS_JSON, cpu.buffer_prototype)) is None
    assert not sync(v3_store.exists(ZARRAY_JSON))
    assert not sync(v3_store.exists(ZATTRS_JSON))
    print("  v2 metadata hidden: OK")


def test_listing() -> None:
    """Store listing should show v3 keys only."""
    v2_store, _ = create_v2_array()
    v3_store = V2AsV3Store(v2_store)

    async def _list() -> list[str]:
        return [k async for k in v3_store.list()]

    keys = sync(_list())
    assert ZARR_JSON in keys
    assert ZARRAY_JSON not in keys
    assert ZATTRS_JSON not in keys
    # Chunk keys use v2 encoding (unchanged).
    assert "0.0" in keys
    print(f"  listing keys: {sorted(keys)}")


def test_serve_roundtrip() -> None:
    """Serve the translated store over HTTP and read it back as v3."""
    from starlette.testclient import TestClient

    from zarr.experimental.serve import serve_store

    v2_store, _data = create_v2_array()
    v3_store = V2AsV3Store(v2_store)

    app = serve_store(v3_store)
    client = TestClient(app)

    # Metadata should be valid v3 JSON.
    resp = client.get("/zarr.json")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    meta = resp.json()
    assert meta["zarr_format"] == 3

    # Chunks should be accessible via v2-style keys (as the metadata declares).
    resp = client.get("/0.0")
    assert resp.status_code == 200
    assert len(resp.content) > 0

    # v2 metadata files should NOT be accessible.
    resp = client.get("/.zarray")
    assert resp.status_code == 404
    resp = client.get("/.zattrs")
    assert resp.status_code == 404

    print("  HTTP round-trip: OK")


def test_open_as_v3_array() -> None:
    """Open the translated store as a v3 array and verify the data."""
    v2_store, data = create_v2_array()
    v3_store = V2AsV3Store(v2_store)

    arr = zarr.open_array(v3_store)
    assert arr.metadata.zarr_format == 3
    np.testing.assert_array_equal(arr[:], data)
    print("  open as v3 array: OK")
    print(f"  data:\n{arr[:]}")


if __name__ == "__main__":
    print("Creating v2 array and wrapping it with V2AsV3Store...\n")

    print("1. Metadata translation")
    test_metadata_translation()

    print("\n2. Chunk key passthrough")
    test_chunk_passthrough()

    print("\n3. v2 metadata hidden")
    test_v2_metadata_hidden()

    print("\n4. Store listing")
    test_listing()

    print("\n5. HTTP round-trip via serve_store")
    test_serve_roundtrip()

    print("\n6. Open as v3 array")
    test_open_as_v3_array()

    print("\nAll checks passed.")
