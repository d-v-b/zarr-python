# How Zarr-Python Encodes and Decodes Chunks

This document describes the complete read and write paths for chunk data in zarr-python, from the moment a user does `arr[:]` or `arr[:] = value` down to individual bytes hitting the store.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Abstractions](#key-abstractions)
3. [The Codec Type Hierarchy](#the-codec-type-hierarchy)
4. [The Codec Pipeline](#the-codec-pipeline)
5. [The Read Path (Decoding)](#the-read-path-decoding)
6. [The Write Path (Encoding)](#the-write-path-encoding)
7. [Sync vs Async Dispatch](#sync-vs-async-dispatch)
8. [Thread Pool for Parallel Codec Compute](#thread-pool-for-parallel-codec-compute)
9. [Partial Decode and Encode](#partial-decode-and-encode)
10. [The Sharding Codec](#the-sharding-codec)
11. [Concrete Codec Implementations](#concrete-codec-implementations)
12. [Metadata Resolution](#metadata-resolution)
13. [Configuration](#configuration)

---

## Overview

When a user reads or writes array data, the following layers are involved:

```
User code          arr[0:100]  /  arr[0:100] = data
                       |
Array              Array.__getitem__ / __setitem__
                       |
Indexer            Computes (chunk_coords, chunk_selection, out_selection) tuples
                       |
Codec Pipeline     BatchedCodecPipeline.read() / .write()
                       |
                  +----+-----+
                  |          |
             Sync path   Async path
                  |          |
Store IO       get_sync()  await get()
                  |          |
Codec chain    _decode_one / decode_batch
                  |          |
Scatter/Gather  out[sel] = chunk_array
```

Two execution strategies exist:

- **Sync path**: When all codecs implement `SupportsSyncCodec` and the store implements `SyncByteGetter`, the entire operation runs on the calling thread. No event loop involvement.
- **Async path**: When any codec lacks sync support or the store is remote, the pipeline uses `concurrent_map` with the async event loop for IO overlap.

The choice is made automatically by `Array._can_use_sync_path()`.

---

## Key Abstractions

### ArraySpec

`ArraySpec` (`src/zarr/core/array_spec.py`) carries metadata about a chunk as it flows through the codec chain:

| Field       | Type              | Description                                    |
|-------------|-------------------|------------------------------------------------|
| `shape`     | `tuple[int, ...]` | Shape of the chunk (may differ from array shape at edges) |
| `dtype`     | `ZDType`          | Data type wrapper                              |
| `fill_value`| `Any`             | Fill value for missing chunks                  |
| `config`    | `ArrayConfig`     | Runtime config (memory order, write_empty_chunks) |
| `prototype` | `BufferPrototype`  | Factory for creating Buffer/NDBuffer instances |

As data passes through each codec, `resolve_metadata()` transforms the ArraySpec to reflect changes the codec makes (e.g., TransposeCodec changes `shape`).

### Buffer and NDBuffer

- `Buffer` (`src/zarr/core/buffer/core.py`): A 1-D byte container. Wraps an `ArrayLike` (typically `numpy.ndarray` with dtype `uint8`). Used for serialized chunk bytes.
- `NDBuffer`: An N-D typed array container. Wraps an `NDArrayLike` (typically `numpy.ndarray`). Used for decoded chunk data.
- `BufferPrototype`: A named tuple with `.buffer` and `.nd_buffer` class factories, allowing GPU or other memory backends.

### ByteGetter / ByteSetter

Protocols defined in `src/zarr/abc/store.py` that represent access to a single key in the store:

```python
class ByteGetter(Protocol):
    async def get(self, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None: ...

class ByteSetter(Protocol):
    async def get(...) -> Buffer | None: ...
    async def set(self, value: Buffer) -> None: ...
    async def delete(self) -> None: ...
```

In practice, `StorePath` (the combination of a `Store` + a string path) implements both protocols. When you see `store_path / metadata.encode_chunk_key(chunk_coords)` in the array code, the result is a `StorePath` that acts as the `ByteGetter`/`ByteSetter` for that chunk.

### SyncByteGetter / SyncByteSetter

Optional sync counterparts:

```python
class SyncByteGetter(Protocol):
    def get_sync(self, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None: ...

class SyncByteSetter(SyncByteGetter, Protocol):
    def set_sync(self, value: Buffer) -> None: ...
    def delete_sync(self) -> None: ...
```

`MemoryStore` and `LocalStore` implement these. Remote stores (fsspec, obstore) do not. The sync path requires the store to implement `SyncByteGetter`.

---

## The Codec Type Hierarchy

All codecs inherit from `BaseCodec` (`src/zarr/abc/codec.py`). There are three codec categories:

### ArrayArrayCodec

Transforms an NDBuffer into another NDBuffer. The shape or layout may change, but the data stays as a typed array.

- **Example**: `TransposeCodec` reorders array dimensions.
- Input: `NDBuffer`, Output: `NDBuffer`
- Position in pipeline: first (before ArrayBytesCodec)

### ArrayBytesCodec

Converts between NDBuffer (typed array) and Buffer (raw bytes). Exactly one must be present in every pipeline.

- **Example**: `BytesCodec` handles endianness and serialization to/from raw bytes.
- Input: `NDBuffer`, Output: `Buffer` (encode) / reverse for decode
- Position in pipeline: middle (between ArrayArrayCodecs and BytesBytesCodecs)

### BytesBytesCodec

Transforms raw bytes into other raw bytes. Typically compression or checksums.

- **Examples**: `BloscCodec`, `GzipCodec`, `ZstdCodec`, `Crc32cCodec`
- Input: `Buffer`, Output: `Buffer`
- Position in pipeline: last (after ArrayBytesCodec)

### The `Codec` Union

```python
Codec = ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec
```

### SupportsSyncCodec Protocol

Codecs that implement synchronous encode/decode:

```python
class SupportsSyncCodec(Protocol):
    def _decode_sync(self, chunk_data: NDBuffer | Buffer, chunk_spec: ArraySpec) -> NDBuffer | Buffer: ...
    def _encode_sync(self, chunk_data: NDBuffer | Buffer, chunk_spec: ArraySpec) -> NDBuffer | Buffer | None: ...
```

All built-in codecs (BytesCodec, BloscCodec, GzipCodec, ZstdCodec, TransposeCodec, Crc32cCodec, ShardingCodec) implement this. Third-party codecs (e.g. numcodecs wrappers) may not.

---

## The Codec Pipeline

### codecs_from_list()

Before a pipeline is created, `codecs_from_list()` (`src/zarr/core/codec_pipeline.py:987`) validates and partitions the codec list into three groups:

```python
(array_array_codecs, array_bytes_codec, bytes_bytes_codecs) = codecs_from_list(codecs)
```

Validation rules:
- Exactly one `ArrayBytesCodec` is required.
- `ArrayArrayCodec`s must come before the `ArrayBytesCodec`.
- `BytesBytesCodec`s must come after the `ArrayBytesCodec`.
- No `ArrayArrayCodec` after `ArrayBytesCodec` or `BytesBytesCodec`.
- If a `ShardingCodec` is present with other codecs, a warning is issued (partial reads/writes are disabled).

### BatchedCodecPipeline

The sole `CodecPipeline` implementation (`src/zarr/core/codec_pipeline.py:128`). A frozen dataclass:

```python
@dataclass(frozen=True)
class BatchedCodecPipeline(CodecPipeline):
    array_array_codecs: tuple[ArrayArrayCodec, ...]
    array_bytes_codec: ArrayBytesCodec
    bytes_bytes_codecs: tuple[BytesBytesCodec, ...]
```

Key computed attribute:
- `_all_sync: bool` -- True when every codec in the pipeline implements `SupportsSyncCodec`. Computed once in `__post_init__`.

The pipeline is iterable: yields all codecs in order (aa, ab, bb).

---

## The Read Path (Decoding)

### Entry Point: Array.__getitem__

```
Array.__getitem__(selection)
  -> Array.get_basic_selection(selection)
    -> if _can_use_sync_path():
         _get_selection_sync(...)     # sync path
       else:
         sync(_get_selection(...))    # async path wrapped in sync()
```

### _get_selection_sync (sync fast path)

1. **Resolve metadata**: dtype, memory order, fields.
2. **Create output buffer**: `prototype.nd_buffer.empty(shape=indexer.shape, dtype=..., order=...)`.
3. **Call `codec_pipeline.read_sync(batch_info, out_buffer)`**.

### _get_selection (async path)

Same setup, but calls `await codec_pipeline.read(batch_info, out_buffer)`.

### batch_info Structure

Each entry in `batch_info` is a 5-tuple:

```python
(byte_getter, chunk_spec, chunk_selection, out_selection, is_complete_chunk)
```

- `byte_getter`: A `StorePath` (or `_ShardingByteGetter` inside sharding) for fetching chunk bytes.
- `chunk_spec`: `ArraySpec` describing this chunk's shape, dtype, fill value.
- `chunk_selection`: Slice within the chunk to read (e.g., `(slice(0, 100),)`).
- `out_selection`: Where in the output array to write this chunk's data.
- `is_complete_chunk`: Whether the entire chunk is being read (optimization hint).

### read_sync (fully synchronous, no event loop)

`BatchedCodecPipeline.read_sync()` (`codec_pipeline.py:814`):

```
Phase 1: IO — fetch all chunk bytes sequentially
  for each chunk:
    chunk_bytes = byte_getter.get_sync(prototype=chunk_spec.prototype)

Phase 2: Decode — run the codec chain for each chunk
  Optionally parallelize with ThreadPoolExecutor (see Thread Pool section)
  for each chunk:
    chunk_array = _decode_one(chunk_bytes, chunk_spec, aa_chain, ab_pair, bb_chain)

Phase 3: Scatter — write decoded data into the output buffer
  for each chunk:
    out[out_selection] = chunk_array[chunk_selection]
```

**Special case**: If `supports_partial_decode` is True (pipeline has only an `ArrayBytesCodec` that supports it, no aa or bb codecs), the pipeline calls `_decode_partial_sync()` instead, which fetches only the needed byte ranges.

### read_batch (async path)

`BatchedCodecPipeline.read_batch()` (`codec_pipeline.py:470`):

If `_all_sync` is True (sync codecs but async store), uses a streaming per-chunk approach:

```python
async def _read_chunk(byte_getter, chunk_spec, chunk_selection, out_selection):
    chunk_bytes = await byte_getter.get(prototype=chunk_spec.prototype)  # async IO
    chunk_array = self._decode_one(chunk_bytes, ...)                     # sync decode
    out[out_selection] = chunk_array[chunk_selection]                     # scatter

await concurrent_map(batch, _read_chunk, concurrency)
```

If `_all_sync` is False (async codecs), uses a phased approach:

```
Phase 1: Fetch all chunks concurrently via concurrent_map
Phase 2: Decode all chunks via decode_batch (async, layer-by-layer)
Phase 3: Scatter into output
```

### _decode_one (per-chunk sync decode)

`BatchedCodecPipeline._decode_one()` (`codec_pipeline.py:261`):

Runs the full codec chain in reverse for a single chunk:

```
Input: Buffer (compressed bytes from store)

1. BytesBytesCodecs (reversed):
   for bb_codec in reversed(bb_chain):
       bytes = bb_codec._decode_sync(bytes, spec)
   # e.g., GzipCodec decompresses, Crc32cCodec verifies checksum

2. ArrayBytesCodec:
   array = ab_codec._decode_sync(bytes, spec)
   # BytesCodec: reinterprets bytes as typed array, handles endianness

3. ArrayArrayCodecs (reversed):
   for aa_codec in reversed(aa_chain):
       array = aa_codec._decode_sync(array, spec)
   # e.g., TransposeCodec: un-transposes dimensions

Output: NDBuffer (decoded chunk data)
```

Returns `None` if input is `None` (missing chunk).

### decode_batch (async layer-by-layer decode)

`BatchedCodecPipeline.decode_batch()` (`codec_pipeline.py:355`):

Processes all chunks at each codec layer before moving to the next:

```
All chunks' bytes
  -> all BytesBytesCodecs (reversed, each processes all chunks)
  -> ArrayBytesCodec (processes all chunks)
  -> all ArrayArrayCodecs (reversed, each processes all chunks)
All chunks' arrays
```

Each codec's `.decode()` method calls `_batching_helper()` which uses `concurrent_map` to process chunks concurrently via the async API (`_decode_single()`).

---

## The Write Path (Encoding)

### Entry Point: Array.__setitem__

```
Array.__setitem__(selection, value)
  -> Array.set_basic_selection(selection, value)
    -> if _can_use_sync_path():
         _set_selection_sync(...)
       else:
         sync(_set_selection(...))
```

### write_sync (fully synchronous)

`BatchedCodecPipeline.write_sync()` (`codec_pipeline.py:913`):

```
Phase 1: IO — read existing chunk bytes for partial writes
  for each chunk:
    if not is_complete_chunk:
      existing_bytes = byte_setter.get_sync(prototype=chunk_spec.prototype)

Phase 2: Compute — decode existing, merge new data, encode
  Optionally parallelize with ThreadPoolExecutor
  for each chunk:
    result = _write_chunk_compute(existing_bytes, chunk_spec, ..., value, drop_axes)

Phase 3: IO — write encoded chunks to store
  for each chunk:
    if result is _DELETED:
      byte_setter.delete_sync()
    else:
      byte_setter.set_sync(result)
```

### _write_chunk_compute (per-chunk compute)

`BatchedCodecPipeline._write_chunk_compute()` (`codec_pipeline.py:871`):

```
1. If existing_bytes is not None (partial write):
   existing_array = _decode_one(existing_bytes, ...)

2. Merge:
   chunk_array = _merge_chunk_array(existing_array, value, ...)

3. Empty chunk optimization:
   If write_empty_chunks=False and chunk is all fill_value:
     return _DELETED

4. Encode:
   chunk_bytes = _encode_one(chunk_array, chunk_spec)
   If chunk_bytes is None: return _DELETED
   return chunk_bytes
```

### _merge_chunk_array

`BatchedCodecPipeline._merge_chunk_array()` (`codec_pipeline.py:577`):

Handles the merge of new data into an existing (or freshly allocated) chunk:

1. **Complete chunk optimization**: If `is_complete_chunk` and the value's shape matches the chunk shape, return the value directly (no copy needed).
2. **Create chunk from fill value**: If no existing data, create a new NDBuffer filled with the fill value.
3. **Copy existing**: If existing data, make a writable copy.
4. **Handle singleton dimensions**: If `drop_axes` is set (e.g., indexing `arr[0]` drops a dimension), re-expand the value with `np.newaxis`.
5. **Assign**: `chunk_array[chunk_selection] = chunk_value`.

### _encode_one (per-chunk sync encode)

`BatchedCodecPipeline._encode_one()` (`codec_pipeline.py:291`):

Runs the full codec chain forward:

```
Input: NDBuffer (chunk array data)

1. ArrayArrayCodecs (forward):
   for aa_codec in array_array_codecs:
       array = aa_codec._encode_sync(array, spec)
   # e.g., TransposeCodec: transpose dimensions

2. ArrayBytesCodec:
   bytes = ab_codec._encode_sync(array, spec)
   # BytesCodec: flatten array, handle endianness, reinterpret as bytes

3. BytesBytesCodecs (forward):
   for bb_codec in bytes_bytes_codecs:
       bytes = bb_codec._encode_sync(bytes, spec)
   # e.g., GzipCodec compresses, Crc32cCodec appends checksum

Output: Buffer (encoded bytes to write to store)
```

Returns `None` if any codec returns `None` along the way.

### The _DELETED Sentinel

```python
_DELETED = object()
```

Used in `write_sync` to distinguish "this chunk should be deleted from the store" from `None` (which `_encode_one` returns when encoding produces no output). In phase 3, if `encoded is _DELETED`, the pipeline calls `byte_setter.delete_sync()` instead of `byte_setter.set_sync()`.

### write_batch (async path)

Similar to read_batch, has two strategies:

**Sync codecs, async store** (`_all_sync=True`):
```python
async def _write_chunk(byte_setter, chunk_spec, ...):
    existing_bytes = await byte_setter.get(...)   # async IO
    chunk_bytes = self._write_chunk_compute(...)    # sync compute
    await byte_setter.set(chunk_bytes)              # async IO

await concurrent_map(batch, _write_chunk, concurrency)
```

**Async codecs** (`_all_sync=False`):
```
Phase 1: Fetch existing chunks concurrently
Phase 2: Decode -> merge -> encode (async codec API via _write_batch_compute)
Phase 3: Write encoded chunks concurrently
```

---

## Sync vs Async Dispatch

### The Decision: Array._can_use_sync_path()

```python
def _can_use_sync_path(self) -> bool:
    pipeline = self.async_array.codec_pipeline
    store = self.async_array.store_path.store
    return getattr(pipeline, "supports_sync_io", False) and isinstance(store, SyncByteGetter)
```

Two conditions:
1. `pipeline.supports_sync_io` is True (all codecs implement `SupportsSyncCodec`).
2. The store implements `SyncByteGetter`.

### The _all_sync Flag

Computed once in `BatchedCodecPipeline.__post_init__()`:

```python
object.__setattr__(
    self, "_all_sync",
    all(isinstance(c, SupportsSyncCodec) for c in self),
)
```

Iterating `self` yields all codecs (aa + ab + bb).

### Why Two Paths?

The sync path avoids:
- **Event loop overhead**: No `sync()` wrapper, no `asyncio.run()` or `loop.run_until_complete()`.
- **Task creation**: No coroutine objects, no `asyncio.Task` scheduling.
- **Thread hopping**: No `asyncio.to_thread()` for CPU-bound codec work.

For local/memory stores with built-in codecs, the sync path is significantly faster (measured at 30-50% improvement on benchmarks).

### Dispatch Flow

```
Array.__getitem__
  |
  +-- _can_use_sync_path() == True?
  |     |
  |     Yes: _get_selection_sync() -> codec_pipeline.read_sync()
  |     |
  |     No:  sync(_get_selection()) -> await codec_pipeline.read()
  |                                         |
  |                                    _all_sync?
  |                                    Yes: streaming per-chunk (sync decode, async IO)
  |                                    No:  phased (async decode + async IO)
```

---

## Thread Pool for Parallel Codec Compute

In the sync path, `read_sync` and `write_sync` can optionally dispatch codec work to a `ThreadPoolExecutor`. This helps when:
- There are multiple chunks to process.
- The codecs do real work (compression) that releases the GIL.
- The chunks are large enough to offset thread dispatch overhead.

### _choose_workers()

```python
_MIN_CHUNK_NBYTES_FOR_POOL = 100_000  # 100 KB

def _choose_workers(n_chunks: int, chunk_nbytes: int, codecs: Iterable[Codec]) -> int:
```

Returns the number of workers to use (0 = don't use pool):

1. If `threading.codec_workers.enabled` is False: return 0.
2. If `n_chunks < 2`: return `min_workers` (usually 0).
3. If no `BytesBytesCodec` in the pipeline and `min_workers == 0`: return 0.
   - BytesBytesCodec = compression/checksum. These release the GIL in C extensions.
   - Pipelines with only BytesCodec + TransposeCodec are too cheap to benefit.
4. If `chunk_nbytes < 100KB` and `min_workers == 0`: return 0.
   - Small chunks: dispatch overhead dominates codec work.
5. Otherwise: `max(min_workers, min(n_chunks, max_workers))`.

### _get_pool()

A module-level lazy singleton:

```python
_pool: ThreadPoolExecutor | None = None

def _get_pool() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _, _, max_workers = _get_codec_worker_config()
        _pool = ThreadPoolExecutor(max_workers=max_workers)
    return _pool
```

`ThreadPoolExecutor` creates threads lazily, so a pool with `max_workers=8` only spawns threads as tasks arrive.

### Usage in read_sync

```python
n_workers = _choose_workers(len(batch_info_list), chunk_nbytes, self)
if n_workers > 0:
    pool = _get_pool()
    chunk_arrays = list(pool.map(self._decode_one, chunk_bytes_list, ...))
else:
    chunk_arrays = [self._decode_one(...) for ...]
```

### Configuration

Via `zarr.config`:

```python
config.get("threading.codec_workers")
# Returns dict with:
#   "enabled": bool (default True)
#   "min": int (default 0)
#   "max": int or None (default None -> os.cpu_count())
```

---

## Partial Decode and Encode

### When Available

The pipeline supports partial decode/encode when:
1. There are no `ArrayArrayCodec`s (they can change slice shapes).
2. There are no `BytesBytesCodec`s (they change byte ranges).
3. The single `ArrayBytesCodec` implements the partial mixin.

```python
@property
def supports_partial_decode(self) -> bool:
    return (len(self.array_array_codecs) + len(self.bytes_bytes_codecs)) == 0 and isinstance(
        self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin
    )
```

### Partial Decode

`ArrayBytesCodecPartialDecodeMixin.decode_partial()`:
- Receives `(ByteGetter, SelectorTuple, ArraySpec)` tuples.
- Each codec determines which byte ranges to fetch based on the selection.
- Fetches only those ranges from the store, then decodes.

The primary user of this is the **ShardingCodec**, which can read individual inner chunks from a shard by fetching byte ranges.

### Partial Encode

`ArrayBytesCodecPartialEncodeMixin.encode_partial()`:
- Receives `(ByteSetter, NDBuffer, SelectorTuple, ArraySpec)` tuples.
- Reads existing chunk data, merges new data, encodes and writes back.

Again, primarily used by ShardingCodec.

---

## The Sharding Codec

The `ShardingCodec` (`src/zarr/codecs/sharding.py`) is the most complex codec. It is an `ArrayBytesCodec` that also implements both partial decode and encode mixins.

### Concept

A shard is a single store key containing multiple inner chunks plus an index. The array's "chunk" in the chunk grid becomes a "shard", and each shard is subdivided into smaller inner chunks.

```
Shard file layout (index_location=end):
  [inner_chunk_0_bytes][inner_chunk_1_bytes]...[inner_chunk_N_bytes][shard_index]

Shard file layout (index_location=start):
  [shard_index][inner_chunk_0_bytes][inner_chunk_1_bytes]...[inner_chunk_N_bytes]
```

### _ShardIndex

A NamedTuple wrapping a numpy array of shape `(*chunks_per_shard, 2)` with dtype uint64. Each entry is `(offset, length)`. Empty chunks have `(MAX_UINT_64, MAX_UINT_64)`.

### _ShardReader

Implements `ShardMapping` (a `Mapping[tuple[int, ...], Buffer | None]`). Wraps the full shard buffer + the decoded index. `__getitem__(chunk_coords)` returns the chunk bytes by slicing the shard buffer at the recorded offset/length.

### Inner Codec Pipeline

`ShardingCodec.codec_pipeline` creates a fresh `BatchedCodecPipeline` from `self.codecs` (the inner codecs). Inner chunks are decoded/encoded through this inner pipeline.

The inner pipeline interacts with `_ShardingByteGetter` / `_ShardingByteSetter`, which are backed by an in-memory dict (no real IO).

### Full Shard Decode (_decode_single / _decode_sync)

1. Parse shard bytes -> `_ShardReader` (decode the shard index).
2. Create output NDBuffer of shard shape.
3. Create a `BasicIndexer` over the shard shape with the inner chunk grid.
4. For each inner chunk: create a `_ShardingByteGetter` pointing to the shard dict.
5. Call `self.codec_pipeline.read(batch_info, out)` to decode all inner chunks.

### Partial Shard Decode (_decode_partial_single / _decode_partial_sync)

1. Determine which inner chunks are needed from the selection.
2. If all chunks needed: read entire shard.
3. Otherwise: read just the shard index (byte-range read), then read each needed inner chunk (byte-range reads).
4. Build a dict of chunk_coords -> Buffer.
5. Decode through the inner pipeline.

### Full Shard Encode (_encode_single / _encode_sync)

1. Create a `BasicIndexer` over the shard shape.
2. Create `_ShardingByteSetter`s backed by a dict (initialized with morton order keys).
3. Call `self.codec_pipeline.write(batch_info, shard_array)` to encode each inner chunk.
4. Assemble: iterate chunks in morton order, concatenate buffers, build shard index, prepend/append index bytes.

### Partial Shard Encode (_encode_partial_single / _encode_partial_sync)

1. Read existing shard (if any) into a `_ShardReader`.
2. Convert to a mutable dict.
3. Encode new data through the inner pipeline (which writes into the dict via `_ShardingByteSetter`).
4. Re-assemble the full shard from the updated dict.
5. Write the assembled shard back to the store.

### Shard Index Encoding/Decoding

The shard index itself is encoded/decoded through its own codec pipeline (`self.index_codecs`), typically `[BytesCodec(), Crc32cCodec()]`.

Sync versions (`_decode_shard_index_sync`, `_encode_shard_index_sync`) run the index codecs inline by manually calling `_decode_sync`/`_encode_sync` on each codec, without constructing a full pipeline object.

---

## Concrete Codec Implementations

### BytesCodec (ArrayBytesCodec)

**File**: `src/zarr/codecs/bytes.py`

The fundamental codec that every pipeline must include. Converts between NDBuffer and Buffer.

**Decode** (`_decode_sync`):
1. View the raw bytes as the target dtype (with correct endianness).
2. Wrap in NDBuffer via `prototype.nd_buffer.from_ndarray_like()`.
3. Reshape to chunk shape if needed.

**Encode** (`_encode_sync`):
1. If endianness doesn't match, `astype()` to the correct byte order.
2. Flatten (ravel) the array.
3. Reinterpret as bytes (`view(dtype="B")`).
4. Wrap in Buffer via `prototype.buffer.from_array_like()`.

Fixed size: Yes (output size == input size).

### BloscCodec (BytesBytesCodec)

**File**: `src/zarr/codecs/blosc.py`

High-performance compressor using the Blosc library (C extension, releases GIL).

**Decode**: `as_numpy_array_wrapper(self._blosc_codec.decode, chunk_bytes, prototype)`
**Encode**: `prototype.buffer.from_bytes(self._blosc_codec.encode(chunk_bytes.as_numpy_array()))`

The async versions (`_decode_single`, `_encode_single`) use `asyncio.to_thread()` to avoid blocking the event loop.

The `_blosc_codec` is a `@cached_property` creating a `numcodecs.Blosc` instance once.

`evolve_from_array_spec()` adjusts `typesize` and `shuffle` based on the array's dtype item size.

### GzipCodec (BytesBytesCodec)

**File**: `src/zarr/codecs/gzip.py`

Standard gzip compression via `numcodecs.GZip`.

**Decode/Encode**: Same pattern as Blosc: `as_numpy_array_wrapper` for decode, `as_numpy_array_wrapper` for encode.

The `_gzip_codec` is a `@cached_property` creating a `numcodecs.GZip(level)` instance once.

### ZstdCodec (BytesBytesCodec)

Similar pattern to Blosc and Gzip, using `numcodecs.Zstd`.

### TransposeCodec (ArrayArrayCodec)

**File**: `src/zarr/codecs/transpose.py`

Reorders array dimensions.

**Decode** (`_decode_sync`): `chunk_array.transpose(inverse_order)` where `inverse_order = np.argsort(self.order)`.
**Encode** (`_encode_sync`): `chunk_array.transpose(self.order)`.

`resolve_metadata()` permutes the chunk shape to reflect the transposed layout.

Fixed size: Yes. No data copying, just a view.

### Crc32cCodec (BytesBytesCodec)

**File**: `src/zarr/codecs/crc32c_.py`

Appends/verifies a CRC-32C checksum.

**Decode**: Verify the last 4 bytes match the CRC-32C of the preceding bytes. Strip the checksum. Raise `ValueError` on mismatch.
**Encode**: Compute CRC-32C of the data, append as 4 bytes.

Fixed size: Yes (output = input + 4 bytes).

---

## Metadata Resolution

As data flows through the codec chain, each codec can transform the `ArraySpec` via `resolve_metadata()`. This is used to propagate shape and dtype changes.

Example for a pipeline `[TransposeCodec(order=(1,0)), BytesCodec(), GzipCodec()]`:

```
Start:     ArraySpec(shape=(100, 200), dtype=float32)
  TransposeCodec.resolve_metadata -> ArraySpec(shape=(200, 100), dtype=float32)
  BytesCodec.resolve_metadata     -> ArraySpec(shape=(200, 100), dtype=float32)  [no change]
  GzipCodec.resolve_metadata      -> ArraySpec(shape=(200, 100), dtype=float32)  [no change]
```

The resolved specs are passed to each codec's decode/encode methods so they know the expected shape/dtype at their position in the chain.

In `_decode_one`, the chain is resolved once via `_resolve_metadata_chain()` and reused for all chunks (since all chunks in a batch share the same spec structure, differing only in edge-chunk shapes which are handled separately).

---

## Configuration

### Array-level config (ArrayConfig)

- `order`: Memory layout (`"C"` or `"F"`). Affects how decoded chunks are stored in memory.
- `write_empty_chunks`: If False, chunks that are entirely fill_value are deleted instead of written.

### Global config (zarr.config)

- `async.concurrency`: Max concurrent tasks for `concurrent_map` in the async path. Default: 10.
- `threading.codec_workers.enabled`: Enable/disable thread pool for sync codec compute. Default: True.
- `threading.codec_workers.min`: Minimum thread pool workers. Default: 0.
- `threading.codec_workers.max`: Maximum thread pool workers. Default: `os.cpu_count()`.

---

## Summary: Complete Read Path Example

For `arr[10:20]` on an array with chunks of size 100, BloscCodec compression, local store:

1. `Array.__getitem__((slice(10, 20),))` -> `Array.get_basic_selection`.
2. `_can_use_sync_path()` -> True (BloscCodec has `_decode_sync`, LocalStore has `get_sync`).
3. `BasicIndexer` computes: chunk 0, chunk_selection=`(slice(10,20),)`, out_selection=`(slice(0,10),)`.
4. `_get_selection_sync()` calls `codec_pipeline.read_sync(batch_info, out_buffer)`.
5. `read_sync`:
   - Phase 1: `store_path.get_sync(prototype)` -> reads `c/0` from disk -> `Buffer`.
   - Phase 2: `_choose_workers(1, 800, codecs)` -> 0 (single chunk). Run inline:
     - `_decode_one(chunk_bytes, spec, [], (BytesCodec, spec), [(BloscCodec, spec)])`:
       - `BloscCodec._decode_sync(bytes, spec)` -> decompresses -> `Buffer`.
       - `BytesCodec._decode_sync(bytes, spec)` -> reinterprets as float32 array, reshapes to (100,) -> `NDBuffer`.
   - Phase 3: `out[slice(0,10)] = chunk_array[slice(10,20)]`.
6. Return `out_buffer.as_ndarray_like()` -> numpy array of shape (10,).

## Summary: Complete Write Path Example

For `arr[10:20] = np.ones(10)` on the same array:

1. `Array.__setitem__((slice(10, 20),), np.ones(10))`.
2. `_can_use_sync_path()` -> True.
3. `BasicIndexer` computes: chunk 0, `is_complete_chunk=False` (partial write).
4. `_set_selection_sync()` calls `codec_pipeline.write_sync(batch_info, value_buffer)`.
5. `write_sync`:
   - Phase 1: `store_path.get_sync(prototype)` -> reads existing chunk 0 bytes.
   - Phase 2: `_write_chunk_compute`:
     - `_decode_one(existing_bytes, ...)` -> decompresses, reinterprets -> existing NDBuffer (100,).
     - `_merge_chunk_array`: copies existing, assigns `chunk_array[10:20] = value[0:10]`.
     - `_encode_one(merged_array, spec)`:
       - `BytesCodec._encode_sync` -> flatten, reinterpret as bytes.
       - `BloscCodec._encode_sync` -> compress.
     - Returns compressed `Buffer`.
   - Phase 3: `store_path.set_sync(encoded_bytes)` -> writes to disk.
