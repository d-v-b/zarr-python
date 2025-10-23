from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
    from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
    from zarr.storage import StorePath

__all__ = [
    "Future",
    "MissingChunk",
    "get_chunk",
    "get_decode_chunk",
    "get_decode_chunk_region",
]

T = TypeVar("T")


class Future(Generic[T]):
    """A Future represents the result of an asynchronous computation.

    This class is similar to tensorstore.Future and provides both synchronous and
    asynchronous access to the result of an async operation.

    The Future can be used in two ways:
    1. Asynchronously: ``result = await future``
    2. Synchronously: ``result = future.result()``

    The Future starts the async computation immediately when created, and caches
    the result so it can be safely awaited or accessed via result() multiple times.

    Since Future is awaitable, it can be used directly with asyncio.gather() for
    concurrent execution of multiple operations.

    Examples
    --------
    >>> import zarr
    >>> import asyncio
    >>> from zarr.experimental.chunk import get_chunk
    >>> from zarr.core.buffer import default_buffer_prototype
    >>>
    >>> arr = zarr.create(shape=(100, 100), chunks=(10, 10), dtype='i4')
    >>> arr[:] = 42
    >>>
    >>> # Single async usage
    >>> async def async_example():
    ...     future = get_chunk(
    ...         metadata=arr._async_array.metadata,
    ...         store=arr._async_array.store_path.store,
    ...         chunk_index=(0, 0),
    ...         prototype=default_buffer_prototype()
    ...     )
    ...     chunk_bytes = await future
    ...     # Can await multiple times
    ...     chunk_bytes_again = await future
    ...     return chunk_bytes
    >>>
    >>> # Multiple concurrent operations with asyncio.gather
    >>> async def concurrent_example():
    ...     future1 = get_chunk(..., chunk_index=(0, 0), ...)
    ...     future2 = get_chunk(..., chunk_index=(0, 1), ...)
    ...     future3 = get_chunk(..., chunk_index=(1, 0), ...)
    ...     # Use asyncio.gather for concurrent execution
    ...     chunks = await asyncio.gather(future1, future2, future3)
    ...     return chunks
    >>>
    >>> # Sync usage
    >>> future = get_chunk(
    ...     metadata=arr._async_array.metadata,
    ...     store=arr._async_array.store_path.store,
    ...     chunk_index=(0, 0),
    ...     prototype=default_buffer_prototype()
    ... )
    >>> chunk_bytes = future.result()  # Blocks until complete
    >>> chunk_bytes_again = future.result()  # Returns cached result
    """

    def __init__(self, coro: Awaitable[T]) -> None:
        """Initialize a Future with a coroutine.

        Parameters
        ----------
        coro : Awaitable[T]
            The coroutine to wrap.
        """
        import threading

        self._coro = coro
        self._result: T | None = None
        self._exception: BaseException | None = None
        self._done = False
        self._lock = threading.Lock()
        # Task will be created lazily when first accessed
        self._task: asyncio.Task[T] | None = None

    def _ensure_task(self) -> asyncio.Task[T]:
        """Ensure the task is created and running.

        Returns
        -------
        asyncio.Task[T]
            The running task.
        """
        if self._task is not None:
            return self._task

        # Try to get the current event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, we'll need to handle this in result()
            return None  # type: ignore[return-value]

        # Create task in the current event loop
        self._task = loop.create_task(self._coro)
        return self._task

    async def _get_result(self) -> T:
        """Internal async method to get the result with caching.

        Returns
        -------
        T
            The result of the computation.
        """
        with self._lock:
            if self._done:
                if self._exception is not None:
                    raise self._exception
                return self._result  # type: ignore[return-value]

        # Ensure task is created
        task = self._ensure_task()
        if task is None:
            # No event loop, shouldn't happen in async context
            raise RuntimeError("No event loop available")

        # Wait for task to complete
        try:
            result = await task
            with self._lock:
                self._result = result
                self._done = True
            return result
        except BaseException as e:
            with self._lock:
                self._exception = e
                self._done = True
            raise

    def __await__(self):
        """Make the Future awaitable.

        This can be called multiple times safely - the result is cached.
        """
        return self._get_result().__await__()

    def result(self, timeout: float | None = None) -> T:
        """Block until the result is available and return it.

        This method can be called multiple times safely - the result is cached.

        Parameters
        ----------
        timeout : float | None, optional
            Maximum time to wait in seconds. If None, wait indefinitely.

        Returns
        -------
        T
            The result of the computation.

        Raises
        ------
        TimeoutError
            If the timeout is reached before the result is available.
        Exception
            Any exception raised during the computation.
        """
        with self._lock:
            if self._done:
                if self._exception is not None:
                    raise self._exception
                return self._result  # type: ignore[return-value]

        # Run the coroutine using sync()
        from zarr.core.sync import sync

        try:
            result = sync(self._get_result(), timeout=timeout)
            return result
        except BaseException:
            # Exception was already stored by _get_result
            raise

    def exception(self, timeout: float | None = None) -> BaseException | None:
        """Return the exception raised by the computation, if any.

        Parameters
        ----------
        timeout : float | None, optional
            Maximum time to wait in seconds. If None, wait indefinitely.

        Returns
        -------
        BaseException | None
            The exception if one was raised, otherwise None.
        """
        if not self._done:
            try:
                self.result(timeout=timeout)
            except BaseException:
                pass
        return self._exception

    @property
    def done(self) -> bool:
        """Check if the Future is done.

        Returns
        -------
        bool
            True if the computation has completed (successfully or with an exception).
        """
        with self._lock:
            return self._done


class _MissingChunk:
    """Singleton representing a missing chunk.

    This is returned by chunk API functions when a requested chunk doesn't exist in storage.
    """

    _instance: _MissingChunk | None = None

    def __new__(cls) -> _MissingChunk:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MissingChunk"

    def __bool__(self) -> bool:
        """MissingChunk is falsy for easy truthiness checks."""
        return False


# Singleton instance
MissingChunk = _MissingChunk()


async def _get_chunk_async(
    *,
    metadata: ArrayV2Metadata | ArrayV3Metadata,
    store: Store | StorePath,
    chunk_index: tuple[int, ...],
    prototype: BufferPrototype,
) -> Buffer | _MissingChunk:
    """
    Fetch a raw (compressed) chunk from the store.

    Parameters
    ----------
    metadata : ArrayV2Metadata | ArrayV3Metadata
        The array metadata containing chunk encoding information.
    store : Store | StorePath
        The store to read from.
    chunk_index : tuple[int, ...]
        The coordinates of the chunk to fetch (e.g., (0, 0) for the first chunk).
    prototype : BufferPrototype
        The buffer prototype to use for reading.

    Returns
    -------
    Buffer | MissingChunk
        The raw chunk bytes, or MissingChunk if the chunk doesn't exist.

    Examples
    --------
    >>> import zarr
    >>> import asyncio
    >>> from zarr.experimental.chunk import get_chunk
    >>> from zarr.core.buffer import default_buffer_prototype
    >>>
    >>> async def example():
    ...     arr = zarr.create(shape=(100, 100), chunks=(10, 10), dtype='i4', store='data.zarr')
    ...     arr[:10, :10] = 42
    ...     chunk_bytes = await get_chunk(
    ...         metadata=arr._async_array.metadata,
    ...         store=arr._async_array.store_path.store,
    ...         chunk_index=(0, 0),
    ...         prototype=default_buffer_prototype()
    ...     )
    ...     return chunk_bytes
    >>> asyncio.run(example())  # doctest: +SKIP
    """
    from zarr.storage import StorePath as _StorePath

    # Normalize to StorePath
    if isinstance(store, _StorePath):
        store_path = store
    else:
        store_path = _StorePath(store, path="")

    # Encode the chunk key according to the metadata's encoding scheme
    chunk_key = metadata.encode_chunk_key(chunk_index)

    # Fetch the raw chunk bytes from storage
    chunk_bytes = await (store_path / chunk_key).get(prototype=prototype)

    if chunk_bytes is None:
        return MissingChunk
    return chunk_bytes


def get_chunk(
    *,
    metadata: ArrayV2Metadata | ArrayV3Metadata,
    store: Store | StorePath,
    chunk_index: tuple[int, ...],
    prototype: BufferPrototype,
) -> Future[Buffer | _MissingChunk]:
    """
    Fetch a raw (compressed) chunk from the store.

    Returns a Future that can be awaited or called with .result() for synchronous access.

    Parameters
    ----------
    metadata : ArrayV2Metadata | ArrayV3Metadata
        The array metadata containing chunk encoding information.
    store : Store | StorePath
        The store to read from.
    chunk_index : tuple[int, ...]
        The coordinates of the chunk to fetch (e.g., (0, 0) for the first chunk).
    prototype : BufferPrototype
        The buffer prototype to use for reading.

    Returns
    -------
    Future[Buffer | MissingChunk]
        A Future containing the raw chunk bytes, or MissingChunk if the chunk doesn't exist.

    Examples
    --------
    >>> import zarr
    >>> from zarr.experimental.chunk import get_chunk
    >>> from zarr.core.buffer import default_buffer_prototype
    >>>
    >>> arr = zarr.create(shape=(100, 100), chunks=(10, 10), dtype='i4', store='data.zarr')
    >>> arr[:10, :10] = 42
    >>>
    >>> # Async usage
    >>> async def async_example():
    ...     future = get_chunk(
    ...         metadata=arr._async_array.metadata,
    ...         store=arr._async_array.store_path.store,
    ...         chunk_index=(0, 0),
    ...         prototype=default_buffer_prototype()
    ...     )
    ...     chunk_bytes = await future
    ...     return chunk_bytes
    >>>
    >>> # Sync usage
    >>> future = get_chunk(
    ...     metadata=arr._async_array.metadata,
    ...     store=arr._async_array.store_path.store,
    ...     chunk_index=(0, 0),
    ...     prototype=default_buffer_prototype()
    ... )
    >>> chunk_bytes = future.result()  # doctest: +SKIP
    """
    return Future(
        _get_chunk_async(
            metadata=metadata,
            store=store,
            chunk_index=chunk_index,
            prototype=prototype,
        )
    )


async def _get_decode_chunk_async(
    *,
    metadata: ArrayV2Metadata | ArrayV3Metadata,
    store: Store | StorePath,
    chunk_index: tuple[int, ...],
    prototype: BufferPrototype,
) -> NDBuffer | _MissingChunk:
    """
    Fetch a chunk from the store, then decode it.

    Parameters
    ----------
    metadata : ArrayV2Metadata | ArrayV3Metadata
        The array metadata containing chunk encoding and codec information.
    store : Store | StorePath
        The store to read from.
    chunk_index : tuple[int, ...]
        The coordinates of the chunk to fetch (e.g., (0, 0) for the first chunk).
    prototype : BufferPrototype
        The buffer prototype to use for reading and creating arrays.

    Returns
    -------
    NDBuffer | MissingChunk
        The decoded chunk as an N-dimensional buffer, or MissingChunk if the chunk doesn't exist.

    Examples
    --------
    >>> import zarr
    >>> import asyncio
    >>> from zarr.experimental.chunk import get_decode_chunk
    >>> from zarr.core.buffer import default_buffer_prototype
    >>>
    >>> async def example():
    ...     arr = zarr.create(shape=(100, 100), chunks=(10, 10), dtype='i4', store='data.zarr')
    ...     arr[:10, :10] = 42
    ...     chunk_array = await get_decode_chunk(
    ...         metadata=arr._async_array.metadata,
    ...         store=arr._async_array.store_path.store,
    ...         chunk_index=(0, 0),
    ...         prototype=default_buffer_prototype()
    ...     )
    ...     return chunk_array
    >>> asyncio.run(example())  # doctest: +SKIP
    """
    from zarr.core.codec_pipeline import BatchedCodecPipeline
    from zarr.storage import StorePath as _StorePath

    # Normalize to StorePath
    if isinstance(store, _StorePath):
        store_path = store
    else:
        store_path = _StorePath(store, path="")

    # Get the raw chunk bytes
    chunk_key = metadata.encode_chunk_key(chunk_index)
    chunk_bytes = await (store_path / chunk_key).get(prototype=prototype)

    # If chunk doesn't exist, return MissingChunk
    if chunk_bytes is None:
        return MissingChunk

    # Create the codec pipeline from the metadata
    from zarr.core.metadata import ArrayV3Metadata

    if isinstance(metadata, ArrayV3Metadata):
        # V3 metadata
        codec_pipeline = BatchedCodecPipeline.from_codecs(metadata.codecs)
    else:
        # V2 metadata - use V2Codec wrapper
        from zarr.codecs._v2 import V2Codec

        v2_codec = V2Codec(filters=metadata.filters, compressor=metadata.compressor)
        codec_pipeline = BatchedCodecPipeline.from_codecs([v2_codec])

    # Create array spec for the chunk
    from zarr.core.array_spec import parse_array_config

    config = parse_array_config(None)
    chunk_spec = metadata.get_chunk_spec(chunk_index, config, prototype)

    # Decode the chunk
    decoded_chunks = await codec_pipeline.decode_batch([(chunk_bytes, chunk_spec)])
    decoded_chunk = list(decoded_chunks)[0]

    return decoded_chunk


def get_decode_chunk(
    *,
    metadata: ArrayV2Metadata | ArrayV3Metadata,
    store: Store | StorePath,
    chunk_index: tuple[int, ...],
    prototype: BufferPrototype,
) -> Future[NDBuffer | _MissingChunk]:
    """
    Fetch a chunk from the store, then decode it.

    Returns a Future that can be awaited or called with .result() for synchronous access.

    Parameters
    ----------
    metadata : ArrayV2Metadata | ArrayV3Metadata
        The array metadata containing chunk encoding and codec information.
    store : Store | StorePath
        The store to read from.
    chunk_index : tuple[int, ...]
        The coordinates of the chunk to fetch (e.g., (0, 0) for the first chunk).
    prototype : BufferPrototype
        The buffer prototype to use for reading and creating arrays.

    Returns
    -------
    Future[NDBuffer | MissingChunk]
        A Future containing the decoded chunk as an N-dimensional buffer,
        or MissingChunk if the chunk doesn't exist.

    Examples
    --------
    >>> import zarr
    >>> from zarr.experimental.chunk import get_decode_chunk
    >>> from zarr.core.buffer import default_buffer_prototype
    >>>
    >>> arr = zarr.create(shape=(100, 100), chunks=(10, 10), dtype='i4', store='data.zarr')
    >>> arr[:10, :10] = 42
    >>>
    >>> # Async usage
    >>> async def async_example():
    ...     future = get_decode_chunk(
    ...         metadata=arr._async_array.metadata,
    ...         store=arr._async_array.store_path.store,
    ...         chunk_index=(0, 0),
    ...         prototype=default_buffer_prototype()
    ...     )
    ...     chunk_array = await future
    ...     return chunk_array
    >>>
    >>> # Sync usage
    >>> future = get_decode_chunk(
    ...     metadata=arr._async_array.metadata,
    ...     store=arr._async_array.store_path.store,
    ...     chunk_index=(0, 0),
    ...     prototype=default_buffer_prototype()
    ... )
    >>> chunk_array = future.result()  # doctest: +SKIP
    """
    return Future(
        _get_decode_chunk_async(
            metadata=metadata,
            store=store,
            chunk_index=chunk_index,
            prototype=prototype,
        )
    )


async def get_decode_chunk_region(
    *,
    metadata: ArrayV2Metadata | ArrayV3Metadata,
    store: Store | StorePath,
    region: tuple[tuple[int, int], ...],
    prototype: BufferPrototype,
) -> AsyncGenerator[tuple[tuple[int, ...], NDBuffer], None]:
    """
    Stream decoded chunks from a bounding box defined in array coordinates.

    This function yields chunks that intersect with the specified region.
    Each yielded item is a tuple of (chunk_index, decoded_chunk_array).

    Parameters
    ----------
    metadata : ArrayV2Metadata | ArrayV3Metadata
        The array metadata containing chunk grid and codec information.
    store : Store | StorePath
        The store to read from.
    region : tuple[tuple[int, int], ...]
        The bounding box in array coordinates, specified as ((start0, end0), (start1, end1), ...).
        Each dimension specifies [start, end) with Python-style half-open intervals.
    prototype : BufferPrototype
        The buffer prototype to use for reading and creating arrays.

    Yields
    ------
    tuple[tuple[int, ...], NDBuffer]
        A tuple of (chunk_index, decoded_chunk) for each chunk that intersects the region.

    Examples
    --------
    >>> import zarr
    >>> import asyncio
    >>> from zarr.experimental.chunk import get_decode_chunk_region
    >>> from zarr.core.buffer import default_buffer_prototype
    >>>
    >>> async def example():
    ...     arr = zarr.create(shape=(100, 100), chunks=(10, 10), dtype='i4', store='data.zarr')
    ...     arr[:] = 42
    ...     chunks = []
    ...     async for chunk_index, chunk_array in get_decode_chunk_region(
    ...         metadata=arr._async_array.metadata,
    ...         store=arr._async_array.store_path.store,
    ...         region=((0, 20), (0, 30)),  # First 20 rows, first 30 columns
    ...         prototype=default_buffer_prototype()
    ...     ):
    ...         chunks.append((chunk_index, chunk_array.shape))
    ...     return chunks
    >>> asyncio.run(example())  # doctest: +SKIP
    [((0, 0), (10, 10)), ((0, 1), (10, 10)), ((0, 2), (10, 10)),
     ((1, 0), (10, 10)), ((1, 1), (10, 10)), ((1, 2), (10, 10))]
    """
    from zarr.core.indexing import _iter_grid

    # Get the chunk grid from metadata
    chunk_grid = metadata.chunk_grid

    # Calculate which chunks intersect with the region
    # Convert region to grid coordinates
    start_coords = tuple(start for start, _ in region)
    stop_coords = tuple(end for _, end in region)

    # Calculate the first and last chunk indices that intersect the region
    start_chunk_indices = tuple(
        start // chunk for start, chunk in zip(start_coords, chunk_grid.chunk_shape, strict=False)
    )
    # For end chunks, we need (end - 1) // chunk to handle exclusive end boundaries correctly
    # e.g., for end=15 with chunk=10, chunk index should be (15-1)//10 = 1
    end_chunk_indices = tuple(
        (end - 1) // chunk for end, chunk in zip(stop_coords, chunk_grid.chunk_shape, strict=False)
    )

    # Calculate the grid shape (number of chunks in each dimension)
    grid_shape = tuple(
        end_idx - start_idx + 1
        for start_idx, end_idx in zip(start_chunk_indices, end_chunk_indices, strict=False)
    )

    # Collect all chunk coordinates in the region
    chunk_coords_list = list(
        _iter_grid(
            grid_shape=grid_shape,
            origin=start_chunk_indices,
        )
    )

    # Fetch all chunks concurrently
    fetch_tasks = [
        _get_decode_chunk_async(
            metadata=metadata,
            store=store,
            chunk_index=coords,
            prototype=prototype,
        )
        for coords in chunk_coords_list
    ]
    decoded_chunks = await asyncio.gather(*fetch_tasks)

    # Yield results, skipping MissingChunk
    # MissingChunk is falsy, so this works with truthiness checks
    for coords, chunk in zip(chunk_coords_list, decoded_chunks, strict=False):
        if chunk:
            yield (coords, chunk)
