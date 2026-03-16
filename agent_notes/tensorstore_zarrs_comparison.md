# How Tensorstore and Zarrs Model Chunk Encoding/Decoding

A detailed comparison of how Google's Tensorstore (C++/Python) and zarrs (Rust) implement the Zarr v3 codec pipeline, contrasted with zarr-python's approach.

---

## Table of Contents

1. [Overview](#overview)
2. [Codec Type Hierarchy](#codec-type-hierarchy)
3. [Codec Pipeline Composition](#codec-pipeline-composition)
4. [Data Flow Through the Pipeline](#data-flow-through-the-pipeline)
5. [The Read Path](#the-read-path)
6. [The Write Path](#the-write-path)
7. [Sync vs Async Execution](#sync-vs-async-execution)
8. [Threading and Parallelism](#threading-and-parallelism)
9. [Sharding](#sharding)
10. [Partial Decoding and Encoding](#partial-decoding-and-encoding)
11. [Store Abstraction](#store-abstraction)
12. [Caching](#caching)
13. [Summary Comparison Table](#summary-comparison-table)

---

## Overview

All three libraries implement the same Zarr v3 specification, which defines a three-stage codec pipeline:

```
ArrayArrayCodecs -> ArrayBytesCodec -> BytesBytesCodecs
```

But they differ substantially in how they model this pipeline at the language and architecture level.

| Library | Language | Core pattern | Pipeline entity |
|---------|----------|-------------|-----------------|
| zarr-python | Python | Frozen dataclass | `BatchedCodecPipeline` |
| Tensorstore | C++ (pybind11 Python bindings) | Spec/PreparedState separation | `ZarrCodecChain` |
| zarrs | Rust | Trait objects behind `Arc` | `CodecChain` |

---

## Codec Type Hierarchy

### zarr-python

A single class hierarchy rooted in `BaseCodec`:

```
BaseCodec[CodecInput, CodecOutput]  (Generic ABC)
  ├── ArrayArrayCodec  (BaseCodec[NDBuffer, NDBuffer])
  ├── ArrayBytesCodec  (BaseCodec[NDBuffer, Buffer])
  └── BytesBytesCodec  (BaseCodec[Buffer, Buffer])

Codec = ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec

SupportsSyncCodec  (Protocol, opt-in for sync encode/decode)
```

A codec object is both its configuration and its execution logic. There is no separation between "what codec this is" and "how it runs."

### Tensorstore

A two-phase design separating specification from execution:

**Spec layer** (configuration/metadata):
```
ZarrCodecSpec  (abstract base)
  ├── ZarrArrayToArrayCodecSpec
  ├── ZarrArrayToBytesCodecSpec
  └── ZarrBytesToBytesCodecSpec
```

**Codec layer** (runtime), each with a nested `PreparedState`:
```
ZarrArrayToArrayCodec::PreparedState
  - EncodeArray() / DecodeArray()
  - Read(NextReader, ...) / Write(NextWriter, ...)   (callback-based)

ZarrArrayToBytesCodec::PreparedState
  - EncodeArray(SharedArrayView, riegeli::Writer&)
  - DecodeArray(span<Index>, riegeli::Reader&)

ZarrBytesToBytesCodec::PreparedState
  - GetEncodeWriter(riegeli::Writer&) -> riegeli::Writer  (decorator pattern)
  - GetDecodeReader(riegeli::Reader&) -> riegeli::Reader

ZarrShardingCodec::PreparedState  (extends ArrayToBytes)
  - GetSubChunkKvstore()
```

The `Resolve()` method transforms a Spec into a PreparedState. This enables:
- Merging partial specifications before finalization (user constraints + stored metadata).
- One-time computation of derived properties (encoded sizes, layouts).
- Immutable, thread-safe prepared state shared across concurrent operations.

### zarrs

A trait-based hierarchy:

```
CodecTraits  (root: name, configuration, partial_decoder_capability, partial_encoder_capability)
  ├── ArrayCodecTraits  (+ recommended_concurrency, partial_decode_granularity)
  │    ├── ArrayToArrayCodecTraits  (encode/decode ArrayBytes, encoded_shape, etc.)
  │    └── ArrayToBytesCodecTraits  (encode/decode ArrayBytes <-> raw bytes, decode_into)
  └── BytesToBytesCodecTraits  (encode/decode raw bytes, recommended_concurrency)
```

All codec objects are wrapped in `Arc<dyn XxxCodecTraits>` (trait objects behind reference-counted pointers). The `Codec` enum unifies all three:

```rust
enum Codec {
    ArrayToArray(Arc<dyn ArrayToArrayCodecTraits>),
    ArrayToBytes(Arc<dyn ArrayToBytesCodecTraits>),
    BytesToBytes(Arc<dyn BytesToBytesCodecTraits>),
}
```

Key distinction: every codec must declare its `PartialDecoderCapability` and `PartialEncoderCapability` upfront. These capabilities drive automatic cache insertion in the partial decoder chain (see [Partial Decoding](#partial-decoding-and-encoding)).

Every codec must also declare its `recommended_concurrency` -- a `Range<usize>` (min..max) indicating how much internal parallelism it can exploit. The codec chain uses this to distribute a concurrency budget across levels.

---

## Codec Pipeline Composition

### zarr-python: `BatchedCodecPipeline`

```python
@dataclass(frozen=True)
class BatchedCodecPipeline(CodecPipeline):
    array_array_codecs: tuple[ArrayArrayCodec, ...]
    array_bytes_codec: ArrayBytesCodec
    bytes_bytes_codecs: tuple[BytesBytesCodec, ...]
```

Created by `codecs_from_list()`, which validates ordering (aa before ab before bb), ensures exactly one ArrayBytesCodec, and warns if ShardingCodec is combined with other codecs.

The pipeline is a `CodecPipeline` subclass -- a separate abstraction from individual codecs.

### Tensorstore: `ZarrCodecChainSpec` -> `ZarrCodecChain`

```cpp
struct ZarrCodecChainSpec {
    vector<ZarrArrayToArrayCodecSpec> array_to_array;
    ZarrArrayToBytesCodecSpec         array_to_bytes;   // exactly 1
    vector<ZarrBytesToBytesCodecSpec>  bytes_to_bytes;
};
```

Key feature: `MergeFrom()` can combine two codec chain specs. This supports merging user constraints with stored metadata. If the array-to-bytes codec does not support the required inner order, `Resolve()` **automatically inserts a TransposeCodecSpec**.

After resolution, the chain becomes `ZarrCodecChain::PreparedState`, containing vectors of prepared states for each codec layer.

### zarrs: `CodecChain`

```rust
struct CodecChain {
    array_to_array: Vec<Arc<dyn ArrayToArrayCodecTraits>>,
    array_to_bytes: Arc<dyn ArrayToBytesCodecTraits>,
    bytes_to_bytes: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    cache_index: Option<usize>,   // Where to insert cache in partial decoder chain
}
```

Notably, `CodecChain` itself implements `ArrayToBytesCodecTraits`. This means a codec chain can be used wherever a single array-to-bytes codec is expected -- the sharding codec exploits this by embedding an inner `CodecChain` for its sub-chunks.

The `cache_index` is computed at construction time by scanning each codec's `partial_decoder_capability()` to find the optimal cache insertion point.

---

## Data Flow Through the Pipeline

### Encoding (array -> bytes)

| Library | Mechanism |
|---------|-----------|
| **zarr-python** | Materialized buffers passed between codecs. Each codec receives an `NDBuffer` or `Buffer`, returns a new `NDBuffer` or `Buffer`. |
| **Tensorstore** | Bytes-to-bytes codecs use **Riegeli streaming writers** (decorator pattern). Each codec wraps the previous writer. Data streams through without intermediate buffer allocation. Array-to-array and array-to-bytes codecs use materialized arrays. |
| **zarrs** | Materialized buffers (`ArrayBytes` / `ArrayBytesRaw`) passed between codecs, similar to zarr-python. |

### Decoding (bytes -> array)

| Library | Mechanism |
|---------|-----------|
| **zarr-python** | Materialized buffers in reverse. `_decode_one()` runs bb codecs in reverse, then ab codec, then aa codecs in reverse. |
| **Tensorstore** | Riegeli streaming readers for bb codecs (reverse order decorator chain), then ab codec reads from composed reader, then aa codecs in reverse. |
| **zarrs** | Materialized buffers in reverse, same pattern as zarr-python. |

### Per-chunk vs layer-by-layer

| Library | Sync path | Async path |
|---------|-----------|------------|
| **zarr-python** | Per-chunk: each chunk flows through the full codec chain via `_decode_one` / `_encode_one`. | Layer-by-layer: all chunks processed by one codec layer before moving to the next (`decode_batch`). |
| **Tensorstore** | Per-chunk always: the codec chain's `EncodeArray` / `DecodeArray` processes one chunk through all stages. | Same as sync (C++ futures handle concurrency). |
| **zarrs** | Per-chunk: `codecs.decode(encoded_bytes, ...)` processes one chunk. Parallelism across chunks is via rayon. | Per-chunk: same as sync but with `futures::buffered()` for IO concurrency. |

zarr-python's async layer-by-layer approach (`decode_batch`) is unique. It was designed to allow each codec's `.decode()` batch method to potentially exploit internal parallelism, but in practice all built-in codecs just call `concurrent_map` on `_decode_single`.

---

## The Read Path

### zarr-python

```
Array.__getitem__
  -> Indexer computes (chunk_coords, chunk_selection, out_selection) tuples
  -> codec_pipeline.read_sync() or await codec_pipeline.read()

read_sync phases:
  1. IO: fetch all chunk bytes sequentially via byte_getter.get_sync()
  2. Decode: _decode_one per chunk (optionally via ThreadPoolExecutor)
  3. Scatter: out[out_selection] = chunk_array[chunk_selection]
```

### Tensorstore

```
TensorStore::Read()
  -> IndexTransform decomposed against chunk grid (grid_partition)
  -> ZarrChunkCache lookup (LRU)
  -> KvStore.Read() for cache misses (with byte range, generation tracking)
  -> codec_state_->DecodeArray() via Riegeli reader chain
  -> Data assembly into output array via IndexTransform
```

Tensorstore integrates caching directly: a `ZarrChunkCache` (or `ZarrShardedChunkCache`) sits between the driver and the store, intercepting all chunk reads. Cache entries are tracked by `TimestampedStorageGeneration` for invalidation.

### zarrs

```
Array::retrieve_array_subset(subset)
  -> chunks_in_array_subset() identifies overlapping chunks
  -> concurrency_chunks_and_codec() balances parallelism budget
  -> iter_concurrent_limit! parallelizes over chunks via rayon
  -> Per chunk:
       storage.get(key) -> Option<encoded_bytes>
       codecs.decode(encoded_bytes, ...) -> ArrayBytes
       Write decoded bytes into output buffer (UnsafeCellSlice)
```

zarrs has a performance-oriented feature: `decode_into()` can decode directly into a preallocated output buffer, avoiding intermediate allocation and copy. It uses `UnsafeCellSlice` to allow concurrent writes to disjoint regions of the same buffer (safe because regions are guaranteed non-overlapping by construction).

### Fill value handling

All three handle missing chunks (store returns None) by filling with the array's fill value. zarr-python and zarrs check this at the pipeline level. Tensorstore's `fill_missing_data_reads` setting controls this behavior.

---

## The Write Path

### zarr-python

```
write_sync phases:
  1. IO: read existing chunk bytes for partial writes
  2. Compute: _write_chunk_compute per chunk
     - decode existing -> merge new data -> encode
     - Empty chunk optimization: if all fill_value, return _DELETED sentinel
  3. IO: write/delete encoded chunks sequentially
```

### Tensorstore

```
TensorStore::Write() -> WriteFutures (copy_future, commit_future)
  -> AsyncWriteArray accumulates writes via MaskedArray
  -> Writeback (on cache eviction or transaction commit):
       GetArrayForWriteback() merges accumulated writes
       codec_state_->EncodeArray()
       KvStore.Write()
```

Key differences:
- **Write coalescing**: Multiple writes to the same chunk accumulate in memory via `MaskedArray`. A single writeback encodes and stores the final state.
- **Two-stage futures**: `copy_future` resolves when data is buffered; `commit_future` resolves when durably persisted.
- **Transactions**: Multiple writes can be grouped atomically. `AtomicMultiPhaseMutation` ensures all sub-chunks within a shard commit together.
- **Optimistic concurrency control**: File driver uses advisory locks, GCS uses conditional writes based on generation, conflicts trigger automatic retry of just the conflicting chunks.

### zarrs

```
Array::store_array_subset(subset, data)
  -> Identify chunks: fully covered vs partially covered
  -> Fully covered chunks: store_chunk() directly (encode + store)
  -> Partially covered chunks: store_chunk_subset()
       If experimental_partial_encoding and codec supports it:
         partial_encoder.partial_encode(...)  (update in-place, e.g. shard inner chunk)
       Else:
         Read-modify-write: retrieve existing -> update subset -> store_chunk()
  -> All chunks parallelized via iter_concurrent_limit!
```

zarrs has an experimental `partial_encode` path that allows the sharding codec to update individual inner chunks without rewriting the entire shard. This is gated behind `CodecOptions.experimental_partial_encoding`.

### Empty chunk optimization

All three check whether a chunk is entirely fill values and skip writing (or erase the key) when `write_empty_chunks` / `store_empty_chunks` is false.

---

## Sync vs Async Execution

### zarr-python: Three execution modes

1. **Pure async**: `AsyncArray` methods, used by async callers.
2. **Sync via event loop bridge**: `Array` methods call `sync()` which runs the async implementation on an event loop.
3. **Fully sync bypass**: When `_can_use_sync_path()` is True (all codecs implement `SupportsSyncCodec` and store implements `SyncByteGetter`), the entire path runs on the calling thread with zero async overhead.

The sync bypass exists specifically to avoid the latency of `sync()` for the common case of local/memory stores with built-in codecs.

### Tensorstore: Async-first C++

Everything is async at the C++ level. Operations return `Future<T>`. Python bindings support both:
- `await ts_array.read()` (asyncio)
- `ts_array.read().result()` (blocking)

There is no "sync wrapper around async" pattern. The C++ executor and thread pool handle scheduling natively. `WriteFutures` provides two-stage futures (copy completion vs commit completion).

The `Batch` class groups deferred reads: multiple reads are held until the batch is submitted, allowing the driver to optimize IO (e.g., reading one shard index to service multiple sub-chunk reads, or reading an entire shard if the batch covers all its entries).

### zarrs: Independent sync and async implementations

One `Array` struct with two separate method sets:
- **Sync**: `retrieve_chunk()`, `store_chunk()` -- uses rayon for parallelism.
- **Async**: `async_retrieve_chunk()`, `async_store_chunk()` -- uses `futures::buffered()` for IO concurrency, rayon for CPU-bound codec work.

The sync and async paths have **completely independent implementations**. The sync path does not wrap the async path (unlike zarr-python). This avoids bridging overhead but means maintaining two parallel codebases.

An `AsyncToSyncBlockOn` adapter can wrap an async store for use with the sync API, providing rayon-based parallelism over an async store backend.

---

## Threading and Parallelism

### zarr-python

Two independent mechanisms:
- **`concurrent_map`** with asyncio semaphore (`async.concurrency` config): for concurrent IO in the async path.
- **`ThreadPoolExecutor`** (`threading.codec_workers` config): for parallel codec compute in the sync path. A lazy singleton pool created on first use. The `_choose_workers()` heuristic skips the pool for single chunks, cheap-only pipelines, or small chunks (<100KB).

The GIL limits true parallelism to C extensions that release it (blosc, gzip, zstd).

### Tensorstore

Unified C++ executor model:
- **`data_copy_concurrency`**: Limits CPU cores for data copying/encoding/decoding. Configurable per-context, can be `"shared"` (defaults to CPU count).
- Multiple chunks and IO operations run concurrently, bounded by the executor.
- Codec compute runs natively in C++ -- no GIL concern.

### zarrs

Sophisticated two-level parallelism:
- **Outer level**: chunks processed in parallel via rayon's `iter_concurrent_limit!`.
- **Inner level**: codec work within a single chunk uses remaining concurrency budget.

The split is computed by `concurrency_chunks_and_codec()`:

```
Given: concurrent_target (e.g. 16 threads), n_chunks, codec recommended_concurrency
Compute: chunk_concurrency × codec_concurrency ≈ concurrent_target
```

Each codec declares its `RecommendedConcurrency` as a `Range<usize>` (min..max). The codec chain aggregates these ranges and the split algorithm distributes the budget.

`CodecOptions.concurrent_target` flows down through the codec chain. Each level consumes some concurrency and passes the remainder to the next. For example, if `concurrent_target` is 16 and 4 chunks run in parallel, each chunk's inner codec chain receives a target of 4.

The `iter_concurrent_limit!` macro subdivides a rayon parallel iterator:
- Chunk size = `iterator_len / concurrent_limit`
- Items within each chunk execute sequentially
- Chunks execute in parallel across rayon's thread pool

---

## Sharding

### zarr-python: Sharding as a codec

`ShardingCodec` (`src/zarr/codecs/sharding.py`) is an `ArrayBytesCodec` that also implements `ArrayBytesCodecPartialDecodeMixin` and `ArrayBytesCodecPartialEncodeMixin`.

- Creates `_ShardingByteGetter` / `_ShardingByteSetter` objects backed by an in-memory dict.
- Inner codec pipeline processes sub-chunks through these dict-backed getters.
- No KvStore-level abstraction -- byte range logic is managed directly.
- Combining sharding with aa or bb codecs disables partial decode/encode.

### Tensorstore: Sharding as a virtual KvStore

The `zarr3_sharding_indexed` KvStore driver wraps a base KvStore and maps sub-chunk keys to byte ranges within the shard file:

- Any KvStore operation (read, write, list, delete) transparently works on sub-chunks.
- Cache layer, transaction layer, and batch layer compose naturally with sharding.
- `ShardIndexCache` reads the shard index via byte-range requests.
- `AtomicMultiPhaseMutation` ensures all sub-chunk mutations within a shard commit atomically.
- Nested sharding is supported: `sharding_height()` tracks nesting depth.
- Batch optimization: if a batch covers all entries in a shard, the entire shard is read in one IO operation.

Write amplification warning: without transactions, modifying a single sub-chunk rewrites the entire shard.

### zarrs: Sharding as array-to-bytes codec with partial encoding

`sharding_indexed` implements `ArrayToBytesCodecTraits`. The inner chunks are processed through an inner `CodecChain`.

Key features:
- `partial_decode_granularity()` returns the inner chunk shape, enabling efficient partial reads at sub-chunk granularity.
- If the store supports `supports_get_partial()`, partial shard reads fetch only the index + needed inner chunks via byte-range requests.
- Experimental `partial_encode`: can update individual inner chunks without rewriting the entire shard (requires `ReadableWritableStorage`).
- `ArrayShardedExt` trait provides `is_sharded()`, `inner_chunk_shape()`, `inner_chunk_grid()`.

---

## Partial Decoding and Encoding

### zarr-python

Limited to the pipeline level via `ArrayBytesCodecPartialDecodeMixin`:
- `supports_partial_decode` is True only when there are no aa or bb codecs -- just a bare ab codec with the mixin.
- In practice, only the sharding codec implements partial decode (reading individual inner chunks via byte ranges).
- No automatic caching in the partial decoder chain.

### Tensorstore

Partial reads at the KvStore level via `OptionalByteRangeRequest`:
- Used by the sharding KvStore adapter to read shard indices and individual sub-chunks.
- Non-sharded arrays always fetch and decode full chunks.
- Batch groups deferred reads for cross-chunk optimization.

### zarrs: Partial decoder chain with automatic caching

zarrs has the most elaborate partial decode system:

**Trait hierarchy**:
- `BytesPartialDecoderTraits`: byte-range partial decode on raw bytes.
- `ArrayPartialDecoderTraits`: subset-based partial decode on arrays. Uses an `Indexer` trait (can be `ArraySubset` or other patterns).

**Chain construction**:
When `partial_decoder()` is called on `CodecChain`:
1. Start with a `StoragePartialDecoder` (wraps store + key).
2. Chain each bb codec's `partial_decoder()` in reverse.
3. Insert `BytesPartialDecoderCache` at the computed `cache_index` if needed.
4. Chain the ab codec's `partial_decoder()`.
5. Insert `ArrayPartialDecoderCache` if needed.
6. Chain each aa codec's `partial_decoder()` in reverse.

**Automatic cache insertion**:
At `CodecChain` construction, `cache_index` is computed by scanning each codec's `PartialDecoderCapability`:
```rust
struct PartialDecoderCapability {
    partial_read: bool,   // Can read only part of input?
    partial_decode: bool,  // Can decode only part of data?
}
```
A cache is inserted after the last codec where `partial_decode` is false. This avoids redundant full-decode operations.

**`decode_into`**: zarrs supports decoding directly into a preallocated output buffer, avoiding intermediate allocation. This uses `UnsafeCellSlice` for concurrent writes to disjoint buffer regions.

**Partial encoding** (experimental):
- `ArrayPartialEncoderTraits` extends `ArrayPartialDecoderTraits`.
- Enables the sharding codec to update individual inner chunks without full shard rewrite.
- Gated behind `experimental_partial_encoding` in `CodecOptions`.

---

## Store Abstraction

### zarr-python

Three-layer abstraction:
1. **`Store`** (ABC): `get`, `set`, `delete`, `list`, `exists`. Implementations: `LocalStore`, `MemoryStore`, `FsspecStore`, `ObstoreStore`, `ZipStore`.
2. **`StorePath`**: `Store` + path prefix. Acts as `ByteGetter` / `ByteSetter`.
3. **`ByteGetter` / `ByteSetter`** (Protocols): lightweight per-chunk handles passed to codec pipeline.

Optional sync protocols: `SyncByteGetter` / `SyncByteSetter` for stores that support synchronous IO.

No generation tracking, no transactions, no batch operations.

### Tensorstore: KvStore

`KvStore` = `DriverPtr` + `path` + `transaction`:
- `Read(key, ReadOptions)` -> `Future<ReadResult>` with `TimestampedStorageGeneration`.
- `Write(key, value, WriteOptions)` -> `Future<TimestampedStorageGeneration>`.
- `ReadOptions` supports byte range, generation-based conditional reads, staleness bounds, batch grouping.
- Drivers: `file`, `gcs`, `s3`, `http`, `memory`, plus adapters (`zarr3_sharding_indexed`, `ocdbt`, `zip`).
- Built-in optimistic concurrency control via generation tracking.

### zarrs

Trait-based design with explicit capability declarations:

```rust
trait ReadableStorageTraits {
    fn get(&self, key) -> Result<MaybeBytes>;
    fn get_partial(&self, key, byte_range) -> Result<MaybeBytes>;
    fn get_partial_many(&self, key, byte_ranges) -> Result<...>;
    fn supports_get_partial(&self) -> bool;
}

trait WritableStorageTraits {
    fn set(&self, key, value) -> Result<()>;
    fn set_partial(&self, key, offset, value) -> Result<()>;
    fn supports_set_partial(&self) -> bool;
    fn erase(&self, key) -> Result<()>;
}
```

Composite traits: `ReadableWritableStorageTraits`, `ReadableListableStorageTraits`, etc.

Async variants mirror these exactly with `async fn` methods.

Stores declare capabilities (`supports_get_partial()`, `supports_set_partial()`) that the sharding codec queries at runtime to decide between partial and full shard operations.

Store implementations: `MemoryStore`, `FilesystemStore`, `HTTPStore`, `ZipStore` (sync); `object_store` wrapper, `OpenDAL` wrapper, `Icechunk` (async).

---

## Caching

### zarr-python

No built-in chunk cache. Every read/write goes directly to the store.

### Tensorstore

Built-in LRU chunk cache (`KvsBackedChunkCache`):
- Configurable size limits via `cache_pool.total_bytes_limit`.
- Generation tracking (`TimestampedStorageGeneration`) for cache invalidation.
- Write-back support: dirty entries are flushed on eviction or transaction commit.
- `recheck_cached_data` controls staleness bounds.
- Separate `ZarrChunkCache` and `ZarrShardedChunkCache` for non-sharded and sharded arrays.

### zarrs

No built-in chunk cache at the array level. However, the partial decoder chain has automatic caching:
- `BytesPartialDecoderCache`: caches fully decoded bytes after decompression.
- `ArrayPartialDecoderCache`: caches the fully decoded array.
- Cache placement is computed at `CodecChain` construction time based on codec capabilities.

These caches are per-decoder-chain instances, not a shared LRU.

---

## Summary Comparison Table

| Aspect | zarr-python | Tensorstore | zarrs |
|--------|-------------|-------------|-------|
| **Language** | Python | C++ (pybind11) | Rust |
| **Codec model** | Single hierarchy (ABC) | Spec/PreparedState separation | Trait objects behind Arc |
| **Pipeline entity** | `BatchedCodecPipeline` (separate from codecs) | `ZarrCodecChain` (separate) | `CodecChain` (itself an ArrayToBytesCodec) |
| **BB codec data flow** | Materialized `Buffer` objects | Riegeli streaming writers/readers (zero-copy) | Materialized `ArrayBytesRaw` |
| **Chunk processing** | Per-chunk (sync) or layer-by-layer (async) | Per-chunk always | Per-chunk always |
| **Sync/async** | 3 modes: pure async, sync-via-bridge, fully-sync bypass | Async-first C++ with blocking `.result()` | Independent sync (rayon) and async (futures) implementations |
| **Parallelism** | asyncio semaphore + ThreadPoolExecutor | Unified C++ executor (`data_copy_concurrency`) | Two-level rayon (`chunk_concurrency × codec_concurrency`) |
| **Concurrency budget** | Two independent configs | Single `data_copy_concurrency` resource | `concurrent_target` flows down chain |
| **Thread pool decision** | Heuristic: BytesBytesCodec present + chunk > 100KB | Always uses executor | Always uses rayon, balanced by `recommended_concurrency` |
| **Chunk cache** | None | Built-in LRU with generation tracking, write-back | None (but partial decoder chain has per-instance caches) |
| **Sharding model** | Codec with in-memory dict | Virtual KvStore adapter | Codec with partial encode support |
| **Sharding composability** | Limited (disables partial decode if combined with aa/bb) | Full (KvStore adapter composes with cache, transactions, batches) | Good (partial decode granularity, experimental partial encode) |
| **Partial decode** | Codec-level opt-in, no caching | KvStore byte-range reads + batch optimization | Full partial decoder chain with automatic cache insertion |
| **Partial encode** | Not supported | Via shard KvStore adapter | Experimental, per-inner-chunk updates |
| **Transactions** | None | Built-in with atomic multi-phase commit | Via Icechunk store (external) |
| **Concurrency control** | None (user responsibility) | OCC via generation tracking, advisory locks, conditional writes | None (user responsibility) |
| **Write coalescing** | None (immediate read-modify-write) | MaskedArray accumulates writes, single writeback | None (immediate read-modify-write) |
| **decode_into** | No (intermediate buffer + copy) | No explicit API (but C++ avoids copies internally) | Yes (`UnsafeCellSlice` for zero-copy into output) |
| **Store capabilities** | Implicit (Protocol-based) | Rich (`ReadOptions` with byte range, generation, batch) | Explicit (`supports_get_partial()`, `supports_set_partial()`) |
| **GIL impact** | Limits parallelism to C extensions that release it | No GIL (C++) | No GIL (Rust) |
| **Codec registration** | Registry pattern (`register_pipeline`) | JSON-based codec lookup | `inventory` compile-time + runtime registry |
| **Auto transpose insertion** | No | Yes (if ab codec doesn't support required order) | No |
| **Nested sharding** | No | Yes (`sharding_height()`) | No |
| **Batch read optimization** | No | Yes (`Batch` groups deferred reads) | No |

---

## Key Architectural Insights

### Tensorstore's streaming codec composition is unique

Tensorstore's use of Riegeli writers/readers for bytes-to-bytes codecs means compressed data streams through the pipeline without intermediate buffer allocation. This is a significant performance advantage for large chunks where compression/decompression dominates, as it avoids allocating and filling temporary buffers between codec stages.

zarr-python and zarrs both use materialized buffers between codec stages, meaning each bytes-to-bytes codec allocates a new buffer for its output.

### zarrs' concurrency budget model is the most sophisticated

zarrs' approach of flowing a `concurrent_target` through the codec chain, with each level consuming some concurrency and passing the remainder, is more nuanced than zarr-python's binary "use thread pool or not" decision or Tensorstore's single `data_copy_concurrency` limit. It allows the system to automatically balance chunk-level parallelism against codec-internal parallelism.

### Tensorstore's KvStore-as-sharding is the most composable

By modeling sharding as a KvStore adapter rather than a codec, Tensorstore gets caching, transactions, batch optimization, and listing "for free" -- they all compose naturally with the KvStore interface. zarr-python and zarrs model sharding as a codec, which means each of these features must be specifically implemented within the sharding codec.

### zarrs' automatic partial decoder caching is unique

Neither zarr-python nor Tensorstore automatically insert caches in the partial decoder chain based on codec capabilities. zarrs computes the optimal cache insertion point at construction time, avoiding redundant decompression when the same chunk is partially decoded multiple times.

### zarr-python's layer-by-layer async decode is unique

zarr-python's `decode_batch` processes all chunks through one codec layer before moving to the next. This was designed to allow codecs to exploit batch-level optimizations, but in practice the built-in codecs don't do this. Tensorstore and zarrs always process one chunk through all stages before moving to the next.

### Tensorstore has the richest store semantics

Tensorstore's KvStore supports generation tracking, conditional operations (optimistic concurrency), batch grouping, staleness bounds, and transactions. zarr-python's Store and zarrs' storage traits are simpler, focusing on basic CRUD operations with optional byte-range support. Concurrency control and transactions are left to external mechanisms (or not supported at all).
