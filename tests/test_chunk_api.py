"""Tests for the chunk-based IO routines"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.experimental.chunk import (
    MissingChunk,
    get_chunk,
    get_decode_chunk,
    get_decode_chunk_region,
)
from zarr.storage import MemoryStore
from zarr.testing.store import LatencyStore


class TestFuture:
    """Tests for Future functionality."""

    def test_future_result_sync(self, memory_store, prototype):
        """Test that Future.result() works for synchronous access."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 123

        # Get future and call result() synchronously
        future = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        chunk_bytes = future.result()
        assert chunk_bytes is not None
        assert len(chunk_bytes) > 0

    def test_future_result_missing_chunk(self, memory_store, prototype):
        """Test that Future.result() returns MissingChunk for missing chunks."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )

        future = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(3, 3),
            prototype=prototype,
        )

        chunk_bytes = future.result()
        assert chunk_bytes is MissingChunk

    def test_future_decode_result_sync(self, memory_store, prototype):
        """Test that Future.result() works for decoded chunks."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 999

        future = get_decode_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        chunk_array = future.result()
        assert chunk_array is not None
        assert chunk_array.shape == (10, 10)
        np.testing.assert_array_equal(chunk_array.as_numpy_array(), 999)

    @pytest.mark.asyncio
    async def test_future_await(self, memory_store, prototype):
        """Test that Future can be awaited."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 456

        future = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        # Await the future
        chunk_bytes = await future
        assert chunk_bytes is not None
        assert len(chunk_bytes) > 0

    def test_future_done_property(self, memory_store, prototype):
        """Test that Future.done property works."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 789

        future = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        # Initially not done
        assert not future.done

        # After calling result(), it's done
        future.result()
        assert future.done

    def test_future_exception(self, memory_store, prototype):
        """Test that Future.exception() returns None when no exception."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 111

        future = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        exc = future.exception()
        assert exc is None

    @pytest.mark.asyncio
    async def test_future_multiple_awaits(self, memory_store, prototype):
        """Test that Future can be awaited multiple times."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 222

        future = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        # First await
        chunk1 = await future
        assert chunk1 is not None
        assert len(chunk1) > 0

        # Second await - should return cached result
        chunk2 = await future
        assert chunk2 is not None
        assert len(chunk2) > 0

        # Should be the same object (cached)
        assert chunk1 is chunk2

    def test_future_multiple_result_calls(self, memory_store, prototype):
        """Test that Future.result() can be called multiple times."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 333

        future = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        # First call
        chunk1 = future.result()
        assert chunk1 is not None
        assert len(chunk1) > 0

        # Second call - should return cached result
        chunk2 = future.result()
        assert chunk2 is not None
        assert len(chunk2) > 0

        # Should be the same object (cached)
        assert chunk1 is chunk2

    @pytest.mark.asyncio
    async def test_future_mixed_await_and_result(self, memory_store, prototype):
        """Test that Future works when mixing await and result() calls."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 444

        future = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        # First await
        chunk1 = await future
        assert chunk1 is not None

        # Then call result() - should return cached result
        chunk2 = future.result()
        assert chunk2 is not None

        # Should be the same object
        assert chunk1 is chunk2

    def test_future_result_then_await(self, memory_store, prototype):
        """Test calling result() first, then await."""
        import asyncio

        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 555

        future = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        # First call result()
        chunk1 = future.result()
        assert chunk1 is not None

        # Then await in a new async context
        async def check_await():
            chunk2 = await future
            assert chunk2 is not None
            return chunk2

        chunk2 = asyncio.run(check_await())

        # Should be the same object
        assert chunk1 is chunk2


class TestAsyncioGather:
    """Tests for using asyncio.gather() with Futures."""

    @pytest.mark.asyncio
    async def test_gather_multiple_chunks_async(self, memory_store, prototype):
        """Test using asyncio.gather() with multiple chunk futures."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        # Write different values to different chunks
        arr[:10, :10] = 1
        arr[:10, 10:20] = 2
        arr[10:20, :10] = 3

        # Create futures for multiple chunks
        future1 = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )
        future2 = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 1),
            prototype=prototype,
        )
        future3 = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(1, 0),
            prototype=prototype,
        )

        # Gather all futures using asyncio.gather
        chunks = await asyncio.gather(future1, future2, future3)

        # Should get list of 3 chunks (asyncio.gather returns list)
        assert isinstance(chunks, list)
        assert len(chunks) == 3
        assert all(chunk is not None for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_gather_multiple_chunks_sync(self, memory_store, prototype):
        """Test using asyncio.gather() with sync result() calls."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 10
        arr[:10, 10:20] = 20

        # Create futures for multiple chunks
        future1 = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )
        future2 = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 1),
            prototype=prototype,
        )

        # Can't use asyncio.gather in sync context, just get results individually
        chunk1 = future1.result()
        chunk2 = future2.result()

        # Should get 2 chunks
        assert chunk1 is not None
        assert chunk2 is not None
        assert len(chunk1) > 0
        assert len(chunk2) > 0

    @pytest.mark.asyncio
    async def test_gather_decoded_chunks(self, memory_store, prototype):
        """Test gathering multiple decoded chunk futures."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 100
        arr[:10, 10:20] = 200
        arr[10:20, :10] = 300

        # Create futures for decoded chunks
        future1 = get_decode_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )
        future2 = get_decode_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 1),
            prototype=prototype,
        )
        future3 = get_decode_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(1, 0),
            prototype=prototype,
        )

        # Gather all futures using asyncio.gather
        chunks = await asyncio.gather(future1, future2, future3)

        # Verify results
        assert len(chunks) == 3
        np.testing.assert_array_equal(chunks[0].as_numpy_array(), 100)
        np.testing.assert_array_equal(chunks[1].as_numpy_array(), 200)
        np.testing.assert_array_equal(chunks[2].as_numpy_array(), 300)

    @pytest.mark.asyncio
    async def test_gather_with_missing_chunks(self, memory_store, prototype):
        """Test gathering futures where some chunks are missing."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        # Only write to first chunk
        arr[:10, :10] = 42

        # Create futures for chunks (some missing)
        future1 = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )
        future2 = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(1, 1),  # Missing
            prototype=prototype,
        )
        future3 = get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(2, 2),  # Missing
            prototype=prototype,
        )

        # Gather all futures using asyncio.gather
        chunks = await asyncio.gather(future1, future2, future3)

        # First should have data, others should be MissingChunk
        assert chunks[0] is not MissingChunk
        assert len(chunks[0]) > 0
        assert chunks[1] is MissingChunk
        assert chunks[2] is MissingChunk

    @pytest.mark.asyncio
    async def test_gather_empty(self):
        """Test asyncio.gather with no futures."""
        # Should return empty list (asyncio.gather returns list)
        chunks = await asyncio.gather()
        assert chunks == []


class TestMissingChunk:
    """Tests for MissingChunk singleton."""

    def test_singleton(self):
        """Test that MissingChunk is a singleton."""
        from zarr.experimental.chunk import _MissingChunk

        instance1 = _MissingChunk()
        instance2 = _MissingChunk()
        assert instance1 is instance2
        assert instance1 is MissingChunk

    def test_falsy(self):
        """Test that MissingChunk is falsy."""
        assert not MissingChunk
        assert bool(MissingChunk) is False

    def test_repr(self):
        """Test MissingChunk repr."""
        assert repr(MissingChunk) == "MissingChunk"

    def test_identity_check(self):
        """Test that identity checks work."""
        result = MissingChunk
        assert result is MissingChunk


@pytest.fixture
async def memory_store():
    """Create a memory store for testing."""
    return await MemoryStore.open()


@pytest.fixture
def prototype():
    """Create a default buffer prototype for testing."""
    return default_buffer_prototype()


class TestGetChunk:
    """Tests for get_chunk function."""

    @pytest.mark.asyncio
    async def test_get_chunk_exists_v3(self, memory_store, prototype):
        """Test fetching a raw chunk that exists (V3 format)."""
        # Create a V3 array
        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 42

        # Fetch the raw chunk
        chunk_bytes = await get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        assert chunk_bytes is not None
        assert len(chunk_bytes) > 0

    @pytest.mark.asyncio
    async def test_get_chunk_exists_v2(self, memory_store, prototype):
        """Test fetching a raw chunk that exists (V2 format)."""
        # Create a V2 array
        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=2,
        )
        arr[:10, :10] = 42

        # Fetch the raw chunk
        chunk_bytes = await get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        assert chunk_bytes is not None
        assert len(chunk_bytes) > 0

    @pytest.mark.asyncio
    async def test_get_chunk_not_exists(self, memory_store, prototype):
        """Test fetching a chunk that doesn't exist returns MissingChunk."""
        # Create an array but don't write to chunk (5, 5)
        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )

        # Fetch a chunk that was never written
        chunk_bytes = await get_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(5, 5),
            prototype=prototype,
        )

        assert chunk_bytes is MissingChunk

    @pytest.mark.asyncio
    async def test_get_chunk_with_store_path(self, memory_store, prototype):
        """Test get_chunk works with StorePath."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 99

        # Use StorePath directly
        store_path = arr._async_array.store_path
        chunk_bytes = await get_chunk(
            metadata=arr._async_array.metadata,
            store=store_path,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        assert chunk_bytes is not None


class TestGetDecodeChunk:
    """Tests for get_decode_chunk function."""

    @pytest.mark.asyncio
    async def test_decode_chunk_v3(self, memory_store, prototype):
        """Test decoding a chunk (V3 format)."""
        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10] = 42

        # Decode the chunk
        chunk_array = await get_decode_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        assert chunk_array is not None
        assert chunk_array.shape == (10, 10)
        np.testing.assert_array_equal(chunk_array.as_numpy_array(), 42)

    @pytest.mark.asyncio
    async def test_decode_chunk_v2(self, memory_store, prototype):
        """Test decoding a chunk (V2 format)."""
        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=2,
        )
        arr[:10, :10] = 99

        # Decode the chunk
        chunk_array = await get_decode_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        assert chunk_array is not None
        assert chunk_array.shape == (10, 10)
        np.testing.assert_array_equal(chunk_array.as_numpy_array(), 99)

    @pytest.mark.asyncio
    async def test_decode_chunk_not_exists(self, memory_store, prototype):
        """Test decoding a non-existent chunk returns MissingChunk."""
        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )

        # Try to decode a chunk that was never written
        chunk_array = await get_decode_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(5, 5),
            prototype=prototype,
        )

        assert chunk_array is MissingChunk

    @pytest.mark.asyncio
    async def test_decode_chunk_with_compression(self, memory_store, prototype):
        """Test decoding a compressed chunk."""
        from zarr.codecs import BloscCodec, BytesCodec

        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="f8",
            store=memory_store,
            zarr_format=3,
            codecs=[BytesCodec(), BloscCodec()],
        )
        # Write some data with patterns
        data = np.arange(100, dtype="f8").reshape(10, 10)
        arr[:10, :10] = data

        # Decode the chunk
        chunk_array = await get_decode_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0),
            prototype=prototype,
        )

        assert chunk_array is not None
        np.testing.assert_array_equal(chunk_array.as_numpy_array(), data)

    @pytest.mark.asyncio
    async def test_decode_chunk_different_dtypes(self, prototype):
        """Test decoding chunks with different dtypes."""
        from zarr.storage import MemoryStore

        dtypes = ["i1", "i4", "f4", "f8", "bool"]

        for dtype in dtypes:
            # Create a fresh store for each dtype to avoid conflicts
            store = await MemoryStore.open()
            arr = zarr.create(
                shape=(50, 50),
                chunks=(10, 10),
                dtype=dtype,
                store=store,
                zarr_format=3,
            )

            if dtype == "bool":
                arr[:10, :10] = True
                expected = True
            else:
                arr[:10, :10] = 7
                expected = 7

            chunk_array = await get_decode_chunk(
                metadata=arr._async_array.metadata,
                store=store,
                chunk_index=(0, 0),
                prototype=prototype,
            )

            assert chunk_array is not None
            # dtype.name may vary (e.g., 'i1' vs 'int8'), so check dtype instead
            assert np.dtype(chunk_array.dtype) == np.dtype(dtype)
            np.testing.assert_array_equal(chunk_array.as_numpy_array(), expected)

    @pytest.mark.asyncio
    async def test_decode_chunk_1d(self, memory_store, prototype):
        """Test decoding a 1D chunk."""
        arr = zarr.create(
            shape=(100,),
            chunks=(10,),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10] = 33

        chunk_array = await get_decode_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0,),
            prototype=prototype,
        )

        assert chunk_array is not None
        assert chunk_array.shape == (10,)
        np.testing.assert_array_equal(chunk_array.as_numpy_array(), 33)

    @pytest.mark.asyncio
    async def test_decode_chunk_3d(self, memory_store, prototype):
        """Test decoding a 3D chunk."""
        arr = zarr.create(
            shape=(50, 50, 50),
            chunks=(10, 10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:10, :10, :10] = 77

        chunk_array = await get_decode_chunk(
            metadata=arr._async_array.metadata,
            store=memory_store,
            chunk_index=(0, 0, 0),
            prototype=prototype,
        )

        assert chunk_array is not None
        assert chunk_array.shape == (10, 10, 10)
        np.testing.assert_array_equal(chunk_array.as_numpy_array(), 77)


class TestGetDecodeChunkRegion:
    """Tests for get_decode_chunk_region function."""

    @pytest.mark.asyncio
    async def test_region_single_chunk(self, memory_store, prototype):
        """Test fetching a region that spans a single chunk."""
        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:] = 42

        chunks = []
        async for chunk_index, chunk_array in get_decode_chunk_region(
            metadata=arr._async_array.metadata,
            store=memory_store,
            region=((0, 10), (0, 10)),
            prototype=prototype,
        ):
            chunks.append((chunk_index, chunk_array.shape))

        # Should get exactly one chunk
        assert len(chunks) == 1
        assert chunks[0][0] == (0, 0)
        assert chunks[0][1] == (10, 10)

    @pytest.mark.asyncio
    async def test_region_multiple_chunks(self, memory_store, prototype):
        """Test fetching a region that spans multiple chunks."""
        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:] = 42

        chunks = []
        async for chunk_index, chunk_array in get_decode_chunk_region(
            metadata=arr._async_array.metadata,
            store=memory_store,
            region=((0, 20), (0, 30)),  # 2x3 chunks
            prototype=prototype,
        ):
            chunks.append((chunk_index, chunk_array.shape))

        # Should get 2 * 3 = 6 chunks
        assert len(chunks) == 6

        # Check we got the right chunk indices
        expected_indices = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        ]
        actual_indices = [chunk[0] for chunk in chunks]
        assert set(actual_indices) == set(expected_indices)

        # All chunks should be (10, 10)
        for _, shape in chunks:
            assert shape == (10, 10)

    @pytest.mark.asyncio
    async def test_region_partial_chunks(self, memory_store, prototype):
        """Test region that partially overlaps chunks."""
        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:] = 99

        chunks = []
        async for chunk_index, chunk_array in get_decode_chunk_region(
            metadata=arr._async_array.metadata,
            store=memory_store,
            region=((5, 15), (5, 15)),  # Spans 4 chunks partially
            prototype=prototype,
        ):
            chunks.append((chunk_index, chunk_array))

        # Should get chunks (0,0), (0,1), (1,0), (1,1)
        assert len(chunks) == 4
        expected_indices = {(0, 0), (0, 1), (1, 0), (1, 1)}
        actual_indices = {chunk[0] for chunk in chunks}
        assert actual_indices == expected_indices

    @pytest.mark.asyncio
    async def test_region_with_missing_chunks(self, memory_store, prototype):
        """Test region where some chunks don't exist."""
        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        # Only write to first chunk
        arr[:10, :10] = 42

        chunks = []
        async for chunk_index, chunk_array in get_decode_chunk_region(
            metadata=arr._async_array.metadata,
            store=memory_store,
            region=((0, 30), (0, 30)),  # 3x3 region
            prototype=prototype,
        ):
            chunks.append(chunk_index)

        # Should only get the one chunk that exists
        assert len(chunks) == 1
        assert chunks[0] == (0, 0)

    @pytest.mark.asyncio
    async def test_region_1d(self, memory_store, prototype):
        """Test region fetching for 1D array."""
        arr = zarr.create(
            shape=(100,),
            chunks=(10,),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:] = 88

        chunks = []
        async for chunk_index, chunk_array in get_decode_chunk_region(
            metadata=arr._async_array.metadata,
            store=memory_store,
            region=((0, 25),),  # First 25 elements (3 chunks)
            prototype=prototype,
        ):
            chunks.append(chunk_index)

        assert len(chunks) == 3
        assert chunks == [(0,), (1,), (2,)]

    @pytest.mark.asyncio
    async def test_region_3d(self, memory_store, prototype):
        """Test region fetching for 3D array."""
        arr = zarr.create(
            shape=(50, 50, 50),
            chunks=(10, 10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        arr[:] = 11

        chunks = []
        async for chunk_index, chunk_array in get_decode_chunk_region(
            metadata=arr._async_array.metadata,
            store=memory_store,
            region=((0, 20), (0, 20), (0, 20)),  # 2x2x2 chunks
            prototype=prototype,
        ):
            chunks.append(chunk_index)

        # Should get 2*2*2 = 8 chunks
        assert len(chunks) == 8

    @pytest.mark.asyncio
    async def test_region_preserves_data(self, memory_store, prototype):
        """Test that decoded chunks contain correct data."""
        arr = zarr.create(
            shape=(50, 50),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=3,
        )
        # Write different values to different chunks
        arr[:10, :10] = 1
        arr[:10, 10:20] = 2
        arr[10:20, :10] = 3
        arr[10:20, 10:20] = 4

        chunk_data = {}
        async for chunk_index, chunk_array in get_decode_chunk_region(
            metadata=arr._async_array.metadata,
            store=memory_store,
            region=((0, 20), (0, 20)),
            prototype=prototype,
        ):
            # Store the first value from each chunk
            chunk_data[chunk_index] = chunk_array.as_numpy_array()[0, 0].item()

        assert chunk_data[(0, 0)] == 1
        assert chunk_data[(0, 1)] == 2
        assert chunk_data[(1, 0)] == 3
        assert chunk_data[(1, 1)] == 4

    @pytest.mark.asyncio
    async def test_region_v2_array(self, memory_store, prototype):
        """Test region fetching works with V2 arrays."""
        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="i4",
            store=memory_store,
            zarr_format=2,
        )
        arr[:] = 55

        chunks = []
        async for chunk_index, chunk_array in get_decode_chunk_region(
            metadata=arr._async_array.metadata,
            store=memory_store,
            region=((0, 20), (0, 20)),
            prototype=prototype,
        ):
            chunks.append(chunk_index)
            np.testing.assert_array_equal(chunk_array.as_numpy_array(), 55)

        assert len(chunks) == 4  # 2x2 chunks

    @pytest.mark.asyncio
    async def test_region_concurrent_fetching(self, prototype):
        """Test that get_decode_chunk_region fetches chunks concurrently."""
        # Create store with latency
        base_store = await MemoryStore.open()
        latency = 0.1  # 100ms per get operation
        store = LatencyStore(base_store, get_latency=latency)

        # Create array and write data
        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="i4",
            store=store,
            zarr_format=3,
        )
        arr[:] = 42

        # Fetch a region spanning 2x3 = 6 chunks
        # If sequential: would take 6 * 0.1 = 0.6 seconds
        # If concurrent: should take ~0.1 seconds (single batch)
        start_time = time.perf_counter()

        chunks = []
        async for chunk_index, chunk_array in get_decode_chunk_region(
            metadata=arr._async_array.metadata,
            store=store,
            region=((0, 20), (0, 30)),  # 2x3 chunks
            prototype=prototype,
        ):
            chunks.append(chunk_index)

        elapsed = time.perf_counter() - start_time

        # Should have fetched 6 chunks
        assert len(chunks) == 6

        # If concurrent, elapsed time should be much less than sequential
        # Sequential would be 6 * 0.1 = 0.6s
        # Concurrent should be close to 0.1s (single round trip)
        # Allow some overhead, but should be < 0.3s (half of sequential time)
        assert elapsed < 0.35, (
            f"Took {elapsed:.2f}s, expected < 0.35s (likely sequential, not concurrent)"
        )

        # More precisely: should be closer to latency than to (num_chunks * latency)
        # If it's > 2.5 * latency, it's definitely sequential
        assert elapsed < 2.5 * latency, (
            f"Took {elapsed:.2f}s for {len(chunks)} chunks, not concurrent"
        )

    @pytest.mark.asyncio
    async def test_region_concurrent_task_count(self, prototype):
        """Test that get_decode_chunk_region schedules all fetches concurrently by tracking task count."""
        import asyncio

        # Track the maximum number of concurrent tasks
        max_concurrent_tasks = 0
        task_counts = []

        # Wrapper store that counts tasks during get operations
        class TaskCountingStore(MemoryStore):
            async def get(
                self,
                key: str,
                prototype: BufferPrototype,
                byte_range: tuple[int | None, int | None] | None = None,
            ) -> Buffer | None:
                nonlocal max_concurrent_tasks

                # Count how many tasks are currently running
                # We look for tasks that are doing store.get operations
                # by checking if they're in a waiting state
                all_tasks = asyncio.all_tasks()
                current_count = len(all_tasks)
                task_counts.append(current_count)
                max_concurrent_tasks = max(max_concurrent_tasks, current_count)

                # Small delay to ensure tasks overlap
                await asyncio.sleep(0.01)

                # Call parent get
                return await super().get(key, prototype, byte_range)

        # Create array with custom store
        store = await TaskCountingStore.open()
        arr = zarr.create(
            shape=(100, 100),
            chunks=(10, 10),
            dtype="i4",
            store=store,
            zarr_format=3,
        )
        arr[:] = 42

        # Fetch a region spanning 2x3 = 6 chunks
        chunks = []
        async for chunk_index, chunk_array in get_decode_chunk_region(
            metadata=arr._async_array.metadata,
            store=store,
            region=((0, 20), (0, 30)),  # 2x3 chunks
            prototype=prototype,
        ):
            chunks.append(chunk_index)

        # Should have fetched 6 chunks
        assert len(chunks) == 6

        # If concurrent, we should see multiple tasks running at once
        # With 6 chunks + 1 main test task, we expect max_concurrent_tasks to be 7
        # (all 6 chunk fetches scheduled concurrently)
        # Sequential execution would show max of 2 (main task + 1 fetch at a time)
        assert max_concurrent_tasks >= 6, (
            f"Expected at least 6 concurrent tasks for 6 chunks, got {max_concurrent_tasks}. "
            f"Task counts during execution: {task_counts}. "
            f"Sequential execution would show max of 2-3 tasks."
        )
