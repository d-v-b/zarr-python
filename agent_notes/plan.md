# Implementation Plan: Improving zarr-python Chunk Encoding/Decoding

Based on analysis of Tensorstore (C++) and zarrs (Rust), this plan identifies concrete improvements to zarr-python's codec pipeline. The changes are organized into tiers by impact and risk.

---

## Design Principles

These principles guide every change in this plan:

1. **Eliminate repeated work** -- compute things once, cache the result.
2. **Reduce abstraction overhead on hot paths** -- fewer objects, fewer indirections per chunk.
3. **Strict IO/compute separation** -- codecs never do IO. All IO happens in a preparation phase; all codec compute operates on in-memory data. Only the outermost pipeline boundary crosses into IO; inner levels (e.g., inner sharding) are always pure compute on pre-fetched bytes.
4. **Make the pipeline composable** -- the sharding codec holds a `CodecChain` directly and calls `decode_chunk`/`encode_chunk` on it, rather than creating a full `BatchedCodecPipeline` for inner codec processing.
5. **Avoid nested thread pool deadlocks** -- a thread pool worker must never submit blocking work to the same pool. (Follows naturally from #3: if compute never does IO, compute can safely run on the thread pool.)
6. **Fewer classes, more reusable abstractions** -- the current codebase has many single-use classes (e.g., `_ShardingByteGetter`, `_ShardingByteSetter`, `ArrayBytesCodecPartialDecodeMixin`, `ArrayBytesCodecPartialEncodeMixin`) and parallel protocol hierarchies (`ByteGetter`/`SyncByteGetter`/`ByteSetter`/`SyncByteSetter`) that make the code hard for new contributors to navigate. Prefer a small number of general-purpose types (like `dict[ChunkCoords, Buffer | None]`) over bespoke wrapper classes. Every new class should earn its place by being used in multiple contexts, not just one.

---

## Tier 1: Low-risk fixes with clear performance wins

### 1.1 Cache `ShardingCodec.codec_pipeline`

**Problem**: `ShardingCodec.codec_pipeline` is a `@property` that creates a new `BatchedCodecPipeline` every time it's accessed. This runs `codecs_from_list()`, allocates a frozen dataclass, and re-computes `_all_sync` on every shard decode/encode. For an array with 1000 shards, that's 1000+ unnecessary pipeline constructions.

**Fix**: Change from `@property` to `@cached_property` (or compute in `__init__` like the existing `_get_index_chunk_spec` / `_get_chunks_per_shard` pattern).

**Complications**: `ShardingCodec` is a frozen dataclass, so `@cached_property` won't work directly. Use the same `lru_cache` pattern already used for `_get_index_chunk_spec` and `_get_chunks_per_shard` (line 371-372 of sharding.py): assign an instance-local `lru_cache`-wrapped version in `__init__`. Since `codec_pipeline` takes no arguments, this simplifies to computing it once and storing via `object.__setattr__`.

**suggestion**:

We should create this object when we call `__init__`.

**Files**: `src/zarr/codecs/sharding.py`

### 1.2 Cache shard index codec chain

**Problem**: `_decode_shard_index_sync` and `_encode_shard_index_sync` both call `codecs_from_list(list(self.index_codecs))` every time. `_shard_index_size` creates a throwaway pipeline via `get_pipeline_class().from_codecs(self.index_codecs)` every time.

**Fix**: Compute the index codec classification `(aa_codecs, ab_codec, bb_codecs)` once in `__init__` and store it. Similarly, cache the index codec pipeline (or eliminate it in favor of the pre-classified codecs). Pre-resolve the metadata chain for the index codecs once.

**Files**: `src/zarr/codecs/sharding.py`

### 1.3 Pass resolved metadata chain to `_write_chunk_compute`

**Problem**: `_write_chunk_compute` calls `self._resolve_metadata_chain(chunk_spec)` per-chunk when decoding existing bytes during partial writes (line 884). This is unnecessary: the chain is the same for all chunks in a batch. `read_sync` already pre-computes the chain once (line 837).

**Fix**: Pre-compute the chain once in `write_sync` and pass it to `_write_chunk_compute` as an additional parameter.

**Files**: `src/zarr/core/codec_pipeline.py`

### 1.4 Cache `TransposeCodec` inverse order

**Problem**: `TransposeCodec._decode_sync` calls `np.argsort(self.order)` per chunk (line 99 of transpose.py). Since `order` is immutable (frozen dataclass), the inverse order is always the same.

**Fix**: Compute `inverse_order` once in `__init__` (or as a `cached_property`) and store it.

**Files**: `src/zarr/codecs/transpose.py`

---

## Tier 2: Moderate-risk structural improvements

### 2.1 Prevent nested thread pool deadlock

**Problem**: When the outer pipeline dispatches shard decoding to the thread pool, each worker thread enters `ShardingCodec._decode_sync`, which calls the inner pipeline's `read_sync`. If the inner pipeline also tries to use the same thread pool (via `_choose_workers` + `_get_pool`), outer workers block on `pool.map()` while holding pool threads that inner tasks need. This is a classic nested deadlock.

**Fix**: Add a mechanism to prevent inner pipelines from using the thread pool when they're already running on a pool thread. Options:

- **Option A (simple)**: Add a `threading.local()` flag `_in_pool_worker = False`. Set it to `True` before dispatching to the pool, check it in `_choose_workers`. If `True`, return 0 (don't use pool).
- **Option B (explicit)**: Add an `allow_pool: bool = True` parameter to `read_sync` / `write_sync`. The sharding codec's `_decode_sync` / `_encode_sync` pass `allow_pool=False` when calling the inner pipeline.

Option B is more explicit and doesn't rely on thread-local state, but changes the method signature. Option A is less invasive.

**Files**: `src/zarr/core/codec_pipeline.py`, `src/zarr/codecs/sharding.py`

### 2.2 Fuse fetch-decode-scatter in `read_sync`

**Problem**: `read_sync` currently has three separate phases: (1) fetch all bytes, (2) decode all, (3) scatter all. This means all decoded chunk arrays (N full-size NDBuffers) are held in memory simultaneously between phases 2 and 3. For a 1GB array with 1000 chunks of 1MB each, that's 1GB of decoded data in memory before scatter begins. The async path with `_all_sync=True` already fuses these into a per-chunk streaming pipeline.

**Fix**: When the thread pool is not used (`n_workers == 0`), fuse phases 2 and 3 into a single loop: decode each chunk and immediately scatter it into the output buffer, then discard the decoded chunk. This matches the async `_read_chunk` pattern.

When the thread pool *is* used, we still need phases because `pool.map` returns results in order. But we could use an approach where the main thread scatters results as they become available rather than waiting for all to complete.

For a first pass, the simple improvement is: fuse decode+scatter in the non-threaded case.

**Files**: `src/zarr/core/codec_pipeline.py`

### 2.3 Fuse fetch-compute-store in `write_sync`

**Problem**: Same phased memory issue as 2.2 but for writes. All encoded chunks are held in memory between phase 2 and phase 3.

**Fix**: Same approach: when the thread pool is not used, fuse phases 2 and 3 into a single loop that encodes each chunk and immediately writes it to the store.

**Files**: `src/zarr/core/codec_pipeline.py`

---

## Tier 3: Larger architectural improvements (informed by Tensorstore/zarrs)

### 3.1 Extract `CodecChain` and eliminate `_ShardingByteGetter`/`_ShardingByteSetter`

**Problem**: zarr-python's `BatchedCodecPipeline` is a separate abstraction from codecs. It cannot be used where a codec is expected. zarrs' `CodecChain` implements `ArrayToBytesCodecTraits`, making it composable: the sharding codec's inner pipeline is just another `CodecChain`.

Currently, the sharding codec creates a full `BatchedCodecPipeline` for its inner codec chain, pulling in all the batching/threading/concurrent_map machinery for what is essentially a simple sequential codec chain operating on in-memory data.

To feed bytes into the inner pipeline, the sharding codec wraps an in-memory `dict[tuple[int, ...], Buffer | None]` in adapter classes that satisfy the `ByteGetter`/`ByteSetter` protocols:

- `_ShardingByteGetter` (sharding.py lines 85-117): A frozen dataclass holding `shard_dict` and `chunk_coords`. Its `get`/`get_sync` methods are just `self.shard_dict.get(self.chunk_coords)` — a dict lookup pretending to be store IO.
- `_ShardingByteSetter` (sharding.py lines 120-138): Extends `_ShardingByteGetter` with `set`/`delete`/`set_sync`/`delete_sync` — all plain dict mutations.

#### Design principle: strict IO/compute separation via a preparation phase

The `decode_chunk(bytes, spec)` vs `decode(byte_getter, spec, selection)` split reveals a deeper tension: `decode_chunk` is pure compute (bytes in, array out), while `decode` mixes IO (fetching bytes via a `ByteGetter`) with compute (running the codec chain). Putting both on `CodecChain` conflates two concerns.

If we take the separation seriously, **no codec should ever do IO**. A codec chain is a pure function from bytes to arrays (or vice versa). All IO — including the byte-range reads that sharding needs — must happen *before* any codec runs.

#### The central abstraction: `deserialize` / `serialize`

The key insight is that an `ArrayBytesCodec` defines a **serialization format** for a `dict[ChunkCoords, Buffer | None]` — a key-value mapping from chunk coordinates to encoded chunk bytes.

- **`deserialize`**: takes a storage blob (`Buffer | None`) and unpacks it into `dict[ChunkCoords, Buffer | None]`.
- **`serialize`**: takes a `ChunkMapping` (`dict[ChunkCoords, Buffer | None]`) and returns a `dict[int, Buffer]` — a mapping from byte offset to data. This represents the write operations needed to commit the serialized data: a dense write is `{0: full_blob}`, a sparse write is `{offset_a: data_a, offset_b: data_b, ...}`. `None` entries in the input represent missing/empty chunks and are recorded accordingly (e.g., `offset=-1` in a shard index).

These are inverses of each other (up to offset information). For a plain (non-sharded) codec, both are trivial: the dict has one entry at key `(0,)`, and the storage blob *is* the chunk bytes. For sharding, `deserialize` decodes the shard index and slices out per-chunk buffers; `serialize` concatenates chunks and builds a new index. When inner chunks are uncompressed (fixed size), `serialize` can return targeted byte ranges instead of a full blob, enabling sparse shard writes.

```python
# Type alias for the chunk dict — used everywhere
ChunkMapping = dict[ChunkCoords, Buffer | None]

class ArrayBytesCodec:
    def deserialize(self, raw: Buffer | None) -> ChunkMapping:
        """Unpack stored bytes into a chunk dict. Pure compute."""
        # Default: one chunk at (0,)
        return {(0,): raw}

    def serialize(self, chunks: ChunkMapping) -> dict[int, Buffer]:
        """Pack a chunk dict into byte ranges for writing. Pure compute.
        Returns {offset: data} — dense write is {0: blob}, sparse is multiple entries.
        None values are omitted (missing/empty chunks)."""
        # Default: one chunk, full overwrite
        assert len(chunks) == 1
        value = next(iter(chunks.values()))
        assert value is not None
        return {0: value}

class ShardingCodec(ArrayBytesCodec):
    def deserialize(self, raw: Buffer | None) -> ChunkMapping:
        """Decode shard index, slice blob into per-chunk buffers.
        Returns an entry for every inner chunk coord; missing chunks have None values."""
        if raw is None:
            return {coords: None for coords in all_chunk_coords}
        shard_index = self._index_chain.decode_chunk(raw[-index_size:], index_spec)
        chunks = {}
        for coords in all_chunk_coords:
            offset, length = shard_index[coords]
            if offset != -1:  # chunk exists
                chunks[coords] = raw[offset:offset+length]
            else:
                chunks[coords] = None
        return chunks

    def serialize(self, chunks: ChunkMapping) -> dict[int, Buffer]:
        """Pack chunks into byte ranges for writing. Pure compute.
        None values are recorded as empty (offset=-1) in the index.
        Dense (compressed inner chunks): returns {0: full_shard_blob}.
        Sparse (uncompressed, fixed-size inner chunks): returns targeted
        byte ranges for only the modified chunks + updated index."""
        if self._has_fixed_size_inner_chunks:
            # Sparse: only write modified chunks at their known offsets
            ranges = {}
            for coords, data in chunks.items():
                if data is not None:
                    offset = self._chunk_offset(coords)  # deterministic
                    ranges[offset] = data
            # Always write updated index at end
            ranges[self._index_offset] = self._index_chain.encode_chunk(
                self._build_index(chunks), index_spec)
            return ranges
        else:
            # Dense: lay out all non-None chunks sequentially, build fresh index
            # Encode index via self._index_chain.encode_chunk(...)
            blob = chunk_data + encoded_index
            return {0: blob}
```

Examples:
```python
# Plain chunk — trivial round-trip
codec.deserialize(chunk_bytes)  # → {(0,): chunk_bytes}
codec.serialize({(0,): chunk_bytes})  # → {0: chunk_bytes}

# Shard (compressed) — dense write
codec.deserialize(shard_blob)  # → {(0,0): b_00, (0,1): b_01, ..., (3,3): b_33}
codec.serialize({(0,0): b_00, ..., (1,2): new_b_12, ..., (3,3): b_33})  # → {0: new_shard_blob}

# Shard (uncompressed, fixed-size) — sparse write
codec.serialize({(1,2): new_b_12})  # → {offset_12: new_b_12, index_offset: new_index}
```

A plain chunk is just a degenerate shard with one inner chunk at key `(0,)`. This makes every type and operation uniform across sharded and non-sharded codec chains.

**Dense vs sparse shard writes:** With compressed inner chunks, the encoded size of a new chunk is unpredictable, so there's no way to splice it into the existing shard blob without shifting all subsequent offsets. In this case, `serialize` returns `{0: full_shard_blob}` — a dense write that rebuilds the blob from scratch. This is what the current code does too (`_encode_shard_dict_sync`).

However, when inner chunks have no bytes-to-bytes codecs (no compression), their encoded sizes are fixed and deterministic. Each chunk occupies a known offset in the shard. In this case, `serialize` can return targeted byte ranges — e.g., `{offset_of_chunk_12: new_bytes, offset_of_index: new_index}` — enabling **sparse shard writes** that only touch modified chunks. The pipeline's commit phase handles both cases uniformly by iterating the `dict[int, Buffer]`.

#### `prepare_read` and `prepare_write`: IO + deserialize

The pipeline needs to cross the IO boundary to get the storage blob, then deserialize it. `prepare_read` and `prepare_write` both do the same thing: **fetch bytes from the store, then call `deserialize`** to produce a `ChunkMapping`. The only difference is which chunks end up in the mapping:

- **`prepare_read`** (partial read): may fetch only the shard index + the needed inner chunks, returning a partial `ChunkMapping` with only the entries the selection requires.
- **`prepare_write`**: for compressed inner chunks, must fetch the *entire* existing shard and deserialize *all* inner chunks, because `serialize` needs the full dict to rebuild the shard blob. For uncompressed (fixed-size) inner chunks, only needs to fetch the chunks being overwritten (or nothing at all for complete-chunk writes), since `serialize` can target individual byte ranges.

Both return a `ChunkMapping` (`dict[ChunkCoords, Buffer | None]`). There are no separate `PreparedRead` / `PreparedWrite` types — just a dict.

```python
class ArrayBytesCodec:
    def prepare_read(
        self, byte_getter: ByteGetter, chunk_spec: ArraySpec, selection: SelectorTuple | None
    ) -> ChunkMapping:
        """IO + deserialize. Fetch bytes from store, unpack into chunk dict."""
        raw = byte_getter.get_sync(prototype=chunk_spec.prototype)
        return self.deserialize(raw)  # → {(0,): raw}

    def prepare_write(
        self, byte_getter: ByteGetter, chunk_spec: ArraySpec, selection: SelectorTuple | None
    ) -> ChunkMapping:
        """IO + deserialize. Same as prepare_read — fetch existing, unpack."""
        raw = byte_getter.get_sync(prototype=chunk_spec.prototype)
        return self.deserialize(raw)  # → {(0,): raw}
```

For the default (non-sharding) case, `prepare_read` and `prepare_write` are identical. For sharding, `prepare_read` can be smarter:

```python
class ShardingCodec(ArrayBytesCodec):
    def prepare_read(self, byte_getter, chunk_spec, selection=None) -> ChunkMapping:
        """Partial read: fetch only the shard index + needed chunks."""
        index_bytes = byte_getter.get_sync(byte_range=SuffixByteRequest(...))
        shard_index = self._index_chain.decode_chunk(index_bytes, index_spec)
        chunks = {}
        for coords in needed_chunks(shard_index, selection):
            offset, length = shard_index[coords]
            chunk_bytes = byte_getter.get_sync(byte_range=RangeByteRequest(offset, length))
            chunks[coords] = chunk_bytes
        return chunks

    def prepare_write(self, byte_getter, chunk_spec, selection=None) -> ChunkMapping:
        """Full read: fetch entire shard, deserialize ALL inner chunks."""
        raw = byte_getter.get_sync(prototype=chunk_spec.prototype)
        return self.deserialize(raw)  # all inner chunks
```

**How the pipeline uses these:**

```python
ab_codec = codec_chain.array_bytes_codec

# Read path — uniform for all codecs
chunk_dict: ChunkMapping = ab_codec.prepare_read(byte_getter, chunk_spec, selection)
for coords, chunk_bytes in chunk_dict.items():
    chunk_array = codec_chain.decode_chunk(chunk_bytes, chunk_spec)
    chunk_sel, out_sel = chunk_grid.get_selections(coords, selection)
    out[out_sel] = chunk_array[chunk_sel]

# Write path — prepare/compute/serialize/commit
chunk_dict: ChunkMapping = ab_codec.prepare_write(byte_getter, chunk_spec, selection)
for coords, existing_bytes in chunk_dict.items():
    if not overlaps(coords, selection):
        continue  # leave existing encoded bytes (or None) untouched
    existing_array = codec_chain.decode_chunk(existing_bytes, chunk_spec)
    chunk_sel, out_sel = chunk_grid.get_selections(coords, selection)
    merged = merge(existing_array, value, chunk_sel, out_sel)
    chunk_dict[coords] = codec_chain.encode_chunk(merged, chunk_spec)
write_ops: dict[int, Buffer] = ab_codec.serialize(chunk_dict)
for offset, data in write_ops.items():
    byte_setter.set_sync(data, byte_range=RangeByteRequest(offset, len(data)))
```

The read and write paths are structurally identical: fetch + deserialize → iterate chunks doing codec compute → (writes only: serialize + commit). No branching, no codec-specific types, no subclass hierarchy.

Note on the write path: for compressed shards, the pipeline iterates the *full* `chunk_dict` but only re-encodes entries that overlap the selection. Entries that don't overlap are left as-is (already-encoded bytes from the existing shard). `serialize` receives the full dict and returns `{0: full_blob}`. For uncompressed shards, `prepare_write` fetches only the touched chunks, the pipeline re-encodes those, and `serialize` returns targeted byte ranges. In both cases, the pipeline's commit loop iterates the `dict[int, Buffer]` uniformly.

**`is_complete_chunk` is eliminated.** Currently, the pipeline carries an `is_complete_chunk` boolean per chunk to decide whether to fetch existing bytes before writing. In the new design, this decision moves into `prepare_write`: for a complete-chunk write (where the selection covers the entire chunk), `prepare_write` returns `None` for that chunk's bytes (no fetch needed). The pipeline sees `None`, calls `decode_chunk(None)` → `None`, and `merge(None, value, ...)` just uses the new value directly. For a partial write, `prepare_write` fetches the existing bytes and returns them. The pipeline doesn't need to know *why* the bytes are `None` vs present — the logic is uniform either way.

**`decode_chunk` is the only compute method the pipeline calls for decoding.** It works for all cases:
- Pipeline iterating a `ChunkMapping` (outer level, after IO)
- Inner sharding decoding in-memory bytes (no `ChunkMapping` involved — `decode_chunk` is called directly with bytes from the shard blob)

This leads to a phased design:

1. **Preparation phase (IO + deserialize)**: The pipeline calls `prepare_read` or `prepare_write`, which fetches bytes from the store and calls `deserialize` to produce a `ChunkMapping`. This may include small amounts of compute necessary to determine IO requirements (e.g., decoding a shard index to learn which byte ranges to fetch), but heavy codec compute (decompression) stays in the execution phase.
2. **Execution phase (compute)**: The pipeline iterates the `ChunkMapping`, calling `decode_chunk` for each entry. No `ByteGetter`, no store access, no IO of any kind.
3. **For writes only — serialize + commit (compute + IO)**: After the pipeline re-encodes modified chunks, it calls `serialize` to produce a `dict[int, Buffer]` of write operations. The pipeline then commits by iterating the dict and doing byte-range writes. A dense result (`{0: blob}`) means a full overwrite; a sparse result means targeted byte-range writes for individual chunks.

This applies uniformly to all codec chains:

| Codec chain | Preparation (IO + deserialize) | Execution (compute) |
|---|---|---|
| `BytesCodec → BloscCodec` | Fetch full chunk bytes → `{(0,): raw}` | Decompress + reinterpret |
| `ShardingCodec` (full read) | Fetch entire shard blob → deserialize all inner chunks | Decode each inner chunk via inner chain |
| `ShardingCodec` (partial read) | Fetch shard index + needed chunk byte-ranges → partial `ChunkMapping` | Decode each fetched chunk via inner chain |
| `ShardingCodec` (write, compressed) | Fetch entire shard blob → deserialize ALL inner chunks | Decode/merge/encode touched chunks; serialize → `{0: full_blob}` (dense commit) |
| `ShardingCodec` (write, uncompressed) | Fetch only touched inner chunks (byte-range reads) | Decode/merge/encode touched chunks; serialize → `{offset: data, ...}` (sparse commit) |
| Nested sharding (2 levels) | Fetch outer shard blob → deserialize outer (one IO) | Decode outer index, slice to get inner shard blob (in-memory), deserialize inner (in-memory), decode chunk |

The key insight: **only the outermost level requires IO.** For nested sharding, the outer pipeline fetches the outer shard blob. From there, everything is in-memory: `deserialize` extracts inner shard bytes, the inner sharding codec's `deserialize` extracts chunk bytes, and `decode_chunk` decompresses. No inner codec ever touches a store.

For partial reads, the same principle holds but `prepare_read` is smarter: it does byte-range reads to fetch the shard index, uses it to determine which byte ranges contain the needed inner chunks, fetches those too — all IO. Then it returns a partial `ChunkMapping`. The pipeline doesn't know or care whether the dict is partial or full.

**How this changes `_decode_partial_sync`:**

Currently, `ShardingCodec._decode_partial_sync` receives a `ByteGetter` and interleaves IO and compute. In the new design, this method's IO responsibilities move into `prepare_read`:

```python
# prepare_read (IO + deserialize):
#   1. Fetch shard index bytes (byte-range read)
#   2. Decode index (compute)
#   3. Use index to determine needed chunk byte ranges
#   4. Fetch those byte ranges (more IO)
#   5. Return ChunkMapping with fetched chunks

# Pipeline (compute phase):
#   1. Iterate ChunkMapping
#   2. decode_chunk each chunk
#   3. Scatter into output
```

In practice, `prepare_read` / `prepare_write` / `deserialize` / `serialize` all live on `ArrayBytesCodec` (the individual codec). `CodecChain` does **not** delegate these — it is a pure-compute abstraction. The pipeline accesses IO methods directly on the underlying `ArrayBytesCodec`:

```python
# Pipeline accesses IO methods via the codec chain's ab_codec:
ab_codec = codec_chain.array_bytes_codec
chunk_dict = ab_codec.prepare_read(byte_getter, chunk_spec, selection)
# ... iterate ChunkMapping, decode_chunk/encode_chunk (via codec_chain) ...
write_ops = ab_codec.serialize(chunk_dict)  # dict[int, Buffer]
for offset, data in write_ops.items():
    byte_setter.set_sync(data, byte_range=RangeByteRequest(offset, len(data)))
```

Each codec type defines its own IO requirements and serialization format. The pipeline orchestrates the phases — it calls the `ArrayBytesCodec`'s IO methods directly, iterates the `ChunkMapping`, calls `CodecChain.decode_chunk`/`encode_chunk` for compute, and (for writes) calls `serialize` to get a `dict[int, Buffer]` then commits by iterating the byte-range writes. This keeps `CodecChain` as a pure-compute abstraction while making the IO/compute boundary visible at the type level.

**This generalizes.** Every codec chain has a preparation phase, even if it's trivial (fetch full bytes, deserialize to `{(0,): raw}`). The preparation phase is the *only* place IO happens (aside from the final write commit). `CodecChain.decode_chunk` and `encode_chunk` are pure compute; `ArrayBytesCodec.serialize` and `deserialize` are also pure compute.

**Implications for the layering:**

```
┌─────────────────────────────────────────────────────────┐
│  Array._get_selection_sync / _set_selection_sync        │  (user-facing)
│    ↓ produces batch_info: [(StorePath, spec, sel, ...)] │
├─────────────────────────────────────────────────────────┤
│  BatchedCodecPipeline.read_sync / write_sync            │  (orchestration)
│    Phase 1: prepare — call ab_codec.prepare_read()      │
│             or prepare_write() (IO + deserialize)       │
│    Phase 2: iterate ChunkMapping, call decode_chunk()     │
│             for each inner chunk (pure compute, pool)   │
│    Phase 3: scatter into output (reads) or              │
│             encode + update dict (writes)               │
│    Phase 4 (writes only): ab_codec.serialize() (compute) │
│             → iterate dict[int, Buffer], byte-range      │
│               writes via byte_setter (IO)                │
├─────────────────────────────────────────────────────────┤
│  CodecChain                                              │
│    decode_chunk / encode_chunk — pure compute                │
│    resolve_metadata_chain — cached chain resolution      │
│    array_bytes_codec — exposes the ab_codec for IO       │
├─────────────────────────────────────────────────────────┤
│  Individual codecs                                       │
│    ArrayBytesCodec.deserialize() — pure compute          │
│      (ShardingCodec: decode index + slice blob;          │
│       default: {(0,): raw})                              │
│    ArrayBytesCodec.serialize() → dict[int, Buffer]       │
│      (ShardingCodec: dense {0: blob} or sparse ranges;   │
│       default: {0: chunk_bytes})                         │
│    ArrayBytesCodec.prepare_read() — IO + deserialize     │
│    ArrayBytesCodec.prepare_write() — IO + deserialize    │
│    ArrayArrayCodec / BytesBytesCodec — pure compute      │
│      (_decode_sync / _encode_sync)                       │
└─────────────────────────────────────────────────────────┘
```

`CodecChain` has no IO methods — it is purely compute (`decode_chunk`, `encode_chunk`, `resolve_metadata_chain`). The pipeline-facing methods (`prepare_read`, `prepare_write`, `serialize`) live on `ArrayBytesCodec`, and the pipeline accesses them directly via `codec_chain.array_bytes_codec`. Of these, `prepare_read` and `prepare_write` do IO; `serialize` is pure compute. This makes the IO/compute boundary visible at the type level: `CodecChain` is guaranteed to be pure compute, and IO orchestration belongs to the pipeline.

**Pros of this approach:**

1. **No codec ever does IO in the compute path.** `decode_chunk`, `encode_chunk`, `deserialize`, and `serialize` are always pure compute. Period.
2. **Uniform model.** Every codec chain — simple, sharding, nested sharding — follows the same prepare/execute/serialize pattern. No special cases, no subclass hierarchies for prepared state.
3. **Thread pool safety.** The prepare phase (IO) runs on the main thread. The execute phase (compute) runs on the thread pool. No risk of a pool worker doing IO or deadlocking.
4. **Testability.** `decode_chunk` can be tested with in-memory bytes. `deserialize` and `serialize` can be tested as a round-trip. `prepare_read` can be tested with a mock `ByteGetter`. They're independently testable.
5. **Nested sharding is naturally handled.** The outer `prepare_read` fetches the shard blob. The inner sharding codec's `decode_chunk` operates on the pre-fetched blob — it calls `deserialize` internally but never does IO.
6. **`serialize` / `deserialize` are conceptual inverses.** Both operate on `ChunkMapping`: `deserialize` unpacks stored bytes into it, `serialize` packs it into write operations (`dict[int, Buffer]`). The extra offset information in the return type enables sparse writes without changing the pipeline's logic.

**Cost:**

One dict allocation per outer chunk. For the common case (simple codec chain), the dict has one entry. For sharding, it has one entry per inner chunk. The overhead is minimal compared to the IO and codec compute.

This design means `supports_partial_decode`/`supports_partial_encode` become implementation details of `prepare_read`/`prepare_write`. The pipeline doesn't branch — it always calls `prepare_read`, iterates the `ChunkMapping`, calls `decode_chunk` for each. The codec chain internally decides whether `prepare_read` fetches the full chunk or does byte-range reads. The `ArrayBytesCodecPartialDecodeMixin` and `ArrayBytesCodecPartialEncodeMixin` can be removed or reduced to internal markers that `prepare_read` checks.

These adapter classes exist solely because `read_sync`/`write_sync` require `ByteGetter`/`ByteSetter`. They are used in 8+ locations across `_decode_sync`, `_decode_single`, `_decode_partial_sync`, `_decode_partial_single`, `_encode_sync`, `_encode_single`, `_encode_partial_sync`, and `_encode_partial_single`. Each creates one `_ShardingByteGetter`/`_ShardingByteSetter` per inner chunk — hundreds of throwaway dataclass instances per shard.

**Fix**: Extract a lightweight `CodecChain` class that encapsulates `(aa_codecs, ab_codec, bb_codecs)` and provides:
- `decode_chunk(bytes, spec) -> NDBuffer` — pure compute, no IO (resolves metadata chain internally, cached)
- `encode_chunk(array, spec) -> Buffer` — pure compute, no IO
- `resolve_metadata_chain(spec)` — one-time chain resolution (cached)
- `array_bytes_codec` — attribute exposing the `ArrayBytesCodec`, which the pipeline uses directly for `prepare_read` (IO), `prepare_write` (IO), and `serialize` (pure compute)

`CodecChain` has zero IO methods. The IO/compute boundary is visible at the type level: `CodecChain` = pure compute, `ArrayBytesCodec` = owns its IO and serialization format.

`BatchedCodecPipeline` would wrap a `CodecChain` and add batching, threading, and scatter/gather logic on top. The pipeline always follows the same pattern: call `codec_chain.array_bytes_codec.prepare_read()`/`prepare_write()` (IO + deserialize), then iterate the `ChunkMapping` calling `codec_chain.decode_chunk` for each inner chunk (compute, dispatchable to thread pool), then scatter (reads) or encode + `codec_chain.array_bytes_codec.serialize()` → iterate `dict[int, Buffer]` byte-range writes (writes). IO is the pipeline's responsibility; `CodecChain` stays pure compute.

The sharding codec would use `CodecChain` directly for its inner codec processing. At the inner level, `decode_chunk` is called directly with in-memory bytes — no `prepare_read` needed because the bytes are already available from slicing the shard blob.

This is the most impactful architectural change. It:
- Eliminates the pipeline creation overhead inside sharding (1.1 becomes unnecessary).
- Eliminates the nested thread pool risk (2.1 becomes unnecessary).
- **Eliminates `_ShardingByteGetter` and `_ShardingByteSetter` entirely** — inner codecs never need IO adapters.
- **Enforces strict IO/compute separation** — no codec ever does IO in the compute path.
- Makes the code more testable (codec chains can be tested in isolation).
- Aligns with zarrs' composable design.

**Assessment against principle #6 (fewer classes, more reusable abstractions):**

Classes/protocols eliminated (6):
- `_ShardingByteGetter` (single-use: dict pretending to be a `ByteGetter`)
- `_ShardingByteSetter` (single-use: dict pretending to be a `ByteSetter`)
- `ArrayBytesCodecPartialDecodeMixin` (single-use: only `ShardingCodec` uses it)
- `ArrayBytesCodecPartialEncodeMixin` (single-use: only `ShardingCodec` uses it)
- `SyncByteGetter` (redundant: merged into `ByteGetter` by item 3.2)
- `SyncByteSetter` (redundant: merged into `ByteSetter` by item 3.2)

Classes introduced (1):
- `CodecChain` — used in 3 contexts: `BatchedCodecPipeline.codec_chain`, `ShardingCodec._inner_chain`, `ShardingCodec._index_chain`. Earns its place by being genuinely multi-use.

Type aliases introduced (1):
- `ChunkMapping = dict[ChunkCoords, Buffer | None]` — not a class, just a name for a plain `dict`. This is the universal currency between IO and compute phases: returned by `prepare_read`/`prepare_write`/`deserialize`, consumed by `serialize` and the pipeline's compute loop. No bespoke wrapper class needed.

Other simplifications:
- Pipeline branching on `supports_partial_decode`/`supports_partial_encode` disappears. The pipeline always calls the same sequence (`prepare_read` → iterate `ChunkMapping` → `decode_chunk`), with no `isinstance` checks.
- The `is_complete_chunk` boolean is eliminated from `batch_info` tuples. `prepare_write` absorbs this decision: it returns `None` for chunks that don't need fetching.
- Store protocols go from 4 to 2 (item 3.2).
- `BatchedCodecPipeline` becomes a thin orchestration layer over `CodecChain`, instead of reimplementing codec chain logic.

Net: **5 fewer classes, 2 fewer protocols, zero new single-use types.** The central data type is `dict`, not a wrapper class.

**Resolved:** `CodecChain` has zero IO methods. The pipeline accesses `prepare_read`/`prepare_write`/`serialize` via `codec_chain.array_bytes_codec` directly.

#### Walkthroughs

All walkthroughs use the same setup: a sharded array where the outer chunk grid maps `(5, 10)` to shard `(0, 0)`. Inside the shard, the indexer maps to inner chunk `(1, 2)`, with a single-element selection within that chunk. The outer codec chain is `ShardingCodec` (with inner chain `BytesCodec → BloscCodec`).

##### Reading a single subchunk from a shard: `arr[5, 10]`

**Current code (4 calls deep, fake ByteGetters, IO/compute interleaved):**

```
1. _get_selection_sync
     batch_info = [(StorePath("data/c/0/0"), shard_spec, chunk_sel, out_sel, False)]

2. BatchedCodecPipeline.read_sync
     sees supports_partial_decode=True (ShardingCodec has the mixin)
     calls: ShardingCodec._decode_partial_sync(StorePath, selection, shard_spec)

3. ShardingCodec._decode_partial_sync        ← CODEC DOES IO (violation!)
     IO: byte_getter.get_sync(byte_range=SuffixByteRequest(...))  → shard index bytes
     IO: decode shard index (inline codec compute)
     IO: byte_getter.get_sync(byte_range=RangeByteRequest(start, end))  → inner chunk bytes
     builds: shard_dict = {(1,2): chunk_bytes}
     creates: _ShardingByteGetter(shard_dict, (1,2))  ← fake ByteGetter wrapping a dict
     calls: self.codec_pipeline.read_sync(  ← creates a NEW BatchedCodecPipeline
                [(_ShardingByteGetter, chunk_spec, chunk_sel, out_sel, ...)], out)

4. Inner BatchedCodecPipeline.read_sync
     Phase 1 IO: _ShardingByteGetter.get_sync()  → dict lookup, returns chunk_bytes
     Phase 2 compute: _decode_chunk(chunk_bytes, chunk_spec)  → BloscCodec decompress
     Phase 3 scatter: out[out_sel] = chunk_array[chunk_sel]
```

Problems: step 3 has the codec doing IO (byte-range reads), which violates IO/compute separation. It creates a throwaway `BatchedCodecPipeline` and `_ShardingByteGetter` that fakes being a store. Step 4 goes through the full phased `read_sync` machinery for what is really just a dict lookup + decompress.

**Proposed design (prepare/execute, strict IO/compute separation):**

```
1. _get_selection_sync
     batch_info = [(StorePath("data/c/0/0"), shard_spec, chunk_sel, out_sel)]

2. BatchedCodecPipeline.read_sync
     ab_codec = codec_chain.array_bytes_codec  # ShardingCodec
     Phase 1 (IO + deserialize):
       chunk_dict = ab_codec.prepare_read(StorePath, shard_spec, selection)
       → ShardingCodec.prepare_read:
           IO: byte_getter.get_sync(byte_range=SuffixByteRequest(...))  → shard index bytes
           compute: self._index_chain.decode_chunk(index_bytes, index_spec)  → decoded index
           IO: byte_getter.get_sync(byte_range=RangeByteRequest(start, end))  → chunk bytes
           returns: {(1,2): chunk_bytes}  ← partial ChunkMapping (only needed chunks)

     Phase 2 (compute, can go to thread pool):
       for coords, chunk_bytes in chunk_dict.items():  ← pipeline does the loop
           chunk_array = codec_chain.decode_chunk(chunk_bytes, chunk_spec)  ← pure compute
           chunk_sel, out_sel = chunk_grid.get_selections(coords, selection)
           out[out_sel] = chunk_array[chunk_sel]
```

IO happens only in Phase 1, and it's clearly separated. Phase 2 is pure compute — the pipeline iterates the `ChunkMapping` and calls `decode_chunk` for each entry. No inner pipeline, no `_ShardingByteGetter`, no fake stores. The shard's chunk grid maps inner chunk coords `(1,2)` to output selections.

##### Writing a single subchunk in a shard: `arr[5, 10] = 42`

This is a partial write — we're updating one element, not the entire shard. The existing shard must be read, the relevant inner chunk decoded, the new value merged in, the chunk re-encoded, and the shard written back.

**Current code (5 calls deep, IO/compute interleaved throughout):**

```
1. _set_selection_sync
     batch_info = [(StorePath("data/c/0/0"), shard_spec, chunk_sel, out_sel, False)]
     calls: codec_pipeline.write_sync(batch_info, value_buffer)

2. BatchedCodecPipeline.write_sync
     sees supports_partial_encode=True (ShardingCodec has the mixin)
     calls: ShardingCodec._encode_partial_sync(StorePath, value, chunk_selection, shard_spec)

3. ShardingCodec._encode_partial_sync          ← CODEC DOES IO (violation!)
     IO: _load_full_shard_maybe_sync(byte_setter)  → fetches ENTIRE existing shard
     compute: decode shard index, slice blob into per-chunk dict
     shard_dict = {(0,0): bytes, (0,1): bytes, ..., (1,2): bytes, ...}  ← ALL chunks
     builds indexer for selection → [(chunk_coords=(1,2), chunk_sel, out_sel, ...)]
     creates: _ShardingByteSetter(shard_dict, (1,2))  ← fake ByteSetter wrapping dict
     calls: self.codec_pipeline.write_sync(      ← creates a NEW BatchedCodecPipeline
                [(_ShardingByteSetter, chunk_spec, chunk_sel, out_sel, False)], value)

4. Inner BatchedCodecPipeline.write_sync
     is_complete_chunk=False, so needs existing bytes for merge:
     Phase 1 IO: _ShardingByteSetter.get_sync()  → dict lookup, returns existing chunk bytes
     Phase 2 compute: _write_chunk_compute:
       - _decode_chunk(existing_bytes, chunk_spec)  → decompress existing chunk
       - _merge_chunk_array(existing_array, value, ...)  → overwrite the one element
       - _encode_chunk(merged_array, chunk_spec)  → recompress
     Phase 3 IO: _ShardingByteSetter.set_sync(encoded_bytes)  → dict[(1,2)] = encoded_bytes

5. Back in ShardingCodec._encode_partial_sync    ← MORE IO
     compute: _encode_shard_dict_sync(shard_dict)  → assemble all chunks + index into shard blob
     IO: byte_setter.set_sync(shard_blob)  → write entire shard back to store
```

Problems:
- Step 3 creates a throwaway `BatchedCodecPipeline` and a `_ShardingByteSetter` that wraps a dict as a fake store.
- Step 4 goes through the full phased `write_sync` — the "IO" is dict lookups and dict assignments.
- The codec interleaves IO (fetch shard, write shard) with compute (decode, merge, encode) across steps 3-5.
- Fetching the entire shard and rewriting it is unavoidable (compressed inner chunks have unpredictable sizes), but the current code buries this behind layers of fake store abstractions instead of making it explicit.

**Proposed design (prepare/compute/serialize/commit, strict IO/compute separation):**

```
1. _set_selection_sync
     batch_info = [(StorePath("data/c/0/0"), shard_spec, chunk_sel, out_sel)]
     calls: codec_pipeline.write_sync(batch_info, value_buffer)

2. BatchedCodecPipeline.write_sync
     ab_codec = codec_chain.array_bytes_codec  # ShardingCodec
     Phase 1 (IO + deserialize):
       chunk_dict = ab_codec.prepare_write(StorePath, shard_spec, selection)
       → ShardingCodec.prepare_write:
           IO: byte_getter.get_sync(...)  → fetches entire existing shard blob
           compute: self.deserialize(shard_blob)  → all inner chunks
           returns: {(0,0): b_00, (0,1): b_01, ..., (1,2): b_12, ..., (3,3): b_33}

     Phase 2 (compute — can go to thread pool):
       for coords, existing_bytes in chunk_dict.items():
           if overlaps(coords, selection):  ← only re-encode touched chunks
               existing_array = codec_chain.decode_chunk(existing_bytes, chunk_spec)
               merged = merge(existing_array, value, selection)
               chunk_dict[coords] = codec_chain.encode_chunk(merged, chunk_spec)
           # else: leave existing encoded bytes untouched

     Phase 3 (serialize — pure compute):
       write_ops = ab_codec.serialize(chunk_dict)
         → ShardingCodec.serialize (compressed): {0: full_shard_blob}
         → ShardingCodec.serialize (uncompressed): {offset_12: new_b_12, idx_off: new_index}
         (for a plain chunk: {0: chunk_bytes})

     Phase 4 (IO — commit):
       for offset, data in write_ops.items():
           byte_setter.set_sync(data, byte_range=RangeByteRequest(offset, len(data)))
```

Key improvements:
- **No fake stores.** No `_ShardingByteSetter`, no dict pretending to be a store.
- **No inner pipeline.** Inner chunk decode/merge/encode uses `decode_chunk` and `encode_chunk` directly.
- **Strict IO/compute separation.** IO happens only in Phase 1 (prepare) and Phase 4 (commit). Phases 2-3 are pure compute, safe for the thread pool.
- **`serialize` is the inverse of `deserialize`.** Both are pure compute on `ArrayBytesCodec`. The pipeline doesn't need to know anything about shard structure — it calls `ab_codec.prepare_write()` (IO + deserialize), modifies entries in the dict, calls `ab_codec.serialize()` (pure compute), and commits by iterating the returned `dict[int, Buffer]`.
- **Writes have four phases**: prepare/deserialize (IO) → compute (decode/merge/encode touched chunks) → serialize (pure compute) → commit (IO).
- **Dense vs sparse commit is data-driven.** `serialize` returns `dict[int, Buffer]`. For compressed shards, this is `{0: full_blob}` (dense). For uncompressed shards with fixed-size inner chunks, this is `{offset_a: data_a, ...}` (sparse — only modified chunks + updated index). The pipeline's commit loop handles both uniformly.

##### Nested sharding (shard within a shard)

This case reveals why the prepare/execute split is the right design. Consider an array with two levels of sharding: outer shards contain inner shards which contain chunks.

**Read path:**

```
1. BatchedCodecPipeline.read_sync
     outer_ab = outer_chain.array_bytes_codec  # OuterShardingCodec
     Phase 1 (IO + deserialize):
       chunk_dict = outer_ab.prepare_read(StorePath, outer_spec, selection)
       → OuterShardingCodec.prepare_read:
           IO: fetch outer shard index (byte-range read)
           IO: fetch the relevant inner shard byte ranges
           returns: {(inner_shard_coords): inner_shard_bytes, ...}

     Phase 2 (compute — ALL in-memory, no IO):
       for coords, inner_shard_bytes in chunk_dict.items():  ← pipeline iterates
           chunk_array = outer_chain.decode_chunk(inner_shard_bytes, outer_spec)
             → OuterShardingCodec._decode_sync:
                 compute: self.deserialize(inner_shard_bytes)  → inner chunk dict (in-memory)
                 → for each inner chunk: inner_chain.decode_chunk(chunk_bytes, chunk_spec) → decompress
           scatter chunk_array into output using outer chunk grid
```

Only Phase 1 does IO. The inner sharding codec never needs `prepare_read` — it receives pre-fetched bytes via `decode_chunk` and calls `deserialize` purely in memory to unpack inner chunks. This is the key insight: **only the outermost level crosses the IO boundary; all inner levels are pure compute on in-memory data.** `deserialize` is the same function whether called from `prepare_read` (after IO) or from `_decode_sync` (on in-memory bytes).

**Sketch**:

```python
@dataclass(frozen=True)
class CodecChain:
    """Lightweight codec chain: aa -> ab -> bb. No threading, no batching.
    Pure compute only — no IO methods. The pipeline accesses IO methods
    (prepare_read, prepare_write) via codec_chain.array_bytes_codec directly."""
    array_array_codecs: tuple[ArrayArrayCodec, ...]
    array_bytes_codec: ArrayBytesCodec
    bytes_bytes_codecs: tuple[BytesBytesCodec, ...]

    def decode_chunk(self, chunk_bytes: Buffer | None,
                   chunk_spec: ArraySpec) -> NDBuffer | None:
        """Pure compute: bytes → array. No IO.
        Resolves the metadata chain internally (cached after first call)."""
        # Current _decode_chunk logic — accepts bytes directly, no ByteGetter

    def encode_chunk(self, chunk_array: NDBuffer | None,
                   chunk_spec: ArraySpec) -> Buffer | None:
        """Pure compute: array → bytes. No IO."""
        # Current _encode_chunk logic — returns bytes directly, no ByteSetter

    def resolve_metadata_chain(self, chunk_spec: ArraySpec) -> ResolvedChain:
        # Current _resolve_metadata_chain logic

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec]) -> Self:
        aa, ab, bb = codecs_from_list(codecs)
        return cls(aa, ab, bb)

@dataclass(frozen=True)
class BatchedCodecPipeline(CodecPipeline):
    codec_chain: CodecChain  # Replaces the three separate tuples
    # ... batching, threading, scatter/gather on top
    #
    # Pipeline orchestrates IO + compute phases:
    #   ab_codec = self.codec_chain.array_bytes_codec
    #
    # read_sync:  ab_codec.prepare_read() (IO + deserialize) → iterate ChunkMapping,
    #             codec_chain.decode_chunk() each (compute) → scatter
    # write_sync: ab_codec.prepare_write() (IO + deserialize) → iterate ChunkMapping,
    #             codec_chain.decode_chunk/encode_chunk() touched chunks (compute)
    #             → ab_codec.serialize() → dict[int, Buffer] (compute)
    #             → iterate byte-range writes via byte_setter (IO)
    # No branching on supports_partial_decode — prepare_read handles it internally.
```

The sharding codec would then hold:
```python
@dataclass(frozen=True)
class ShardingCodec(ArrayBytesCodec, ...):
    chunk_shape: tuple[int, ...]
    codecs: tuple[Codec, ...]
    index_codecs: tuple[Codec, ...]
    # Computed in __init__:
    _inner_chain: CodecChain = field(init=False)
    _index_chain: CodecChain = field(init=False)

    def deserialize(self, raw: Buffer | None) -> ChunkMapping:
        """Pure compute: decode shard index, slice blob into per-chunk buffers."""
        if raw is None:
            return {coords: None for coords in all_chunk_coords}
        shard_index = self._index_chain.decode_chunk(raw[-index_size:], index_spec)
        chunks = {}
        for coords in all_chunk_coords:
            offset, length = shard_index[coords]
            chunks[coords] = raw[offset:offset+length] if offset != -1 else None
        return chunks

    def serialize(self, chunks: ChunkMapping) -> dict[int, Buffer]:
        """Pure compute: pack chunks into byte ranges for writing.
        Dense (compressed): returns {0: full_shard_blob}.
        Sparse (uncompressed, fixed-size): returns targeted byte ranges."""
        # See earlier sketch for dense vs sparse logic

    def prepare_read(self, byte_getter, chunk_spec, selection=None) -> ChunkMapping:
        """IO: fetch shard index + needed inner chunk bytes (partial read).
        Falls back to full fetch + deserialize when selection is None."""
        if selection is None:
            raw = byte_getter.get_sync(prototype=chunk_spec.prototype)
            return self.deserialize(raw)
        index_bytes = byte_getter.get_sync(byte_range=SuffixByteRequest(...))
        shard_index = self._index_chain.decode_chunk(index_bytes, index_spec)
        chunks = {}
        for chunk_coords in needed_chunks(shard_index, selection):
            offset, length = shard_index[chunk_coords]
            if offset == -1:  # chunk missing from shard
                chunks[chunk_coords] = None
            else:
                chunks[chunk_coords] = byte_getter.get_sync(
                    byte_range=RangeByteRequest(offset, length))
        return chunks

    def prepare_write(self, byte_getter, chunk_spec, selection=None) -> ChunkMapping:
        """IO + deserialize. Strategy depends on inner chunk compressibility:
        - Compressed: fetch entire shard, deserialize ALL inner chunks
          (serialize will rebuild the full blob).
        - Uncompressed (fixed-size): fetch only the touched chunks
          (serialize will write targeted byte ranges)."""
        if self._has_fixed_size_inner_chunks:
            # Sparse: only fetch the chunks we're overwriting
            return self.prepare_read(byte_getter, chunk_spec, selection)
        else:
            # Dense: must fetch everything — serialize rebuilds the full blob
            raw = byte_getter.get_sync(prototype=chunk_spec.prototype)
            return self.deserialize(raw)

    def _decode_sync(self, shard_bytes, chunk_spec):
        """Pure compute: full shard decode. No IO — bytes already in memory.
        Uses deserialize to unpack into chunk dict, then decode_chunk each.
        Returns a single NDBuffer covering the full shard shape."""
        out = NDBuffer.create(shape=shard_shape, dtype=chunk_spec.dtype,
                              fill_value=chunk_spec.fill_value)
        chunk_dict = self.deserialize(shard_bytes)
        for chunk_coords, chunk_bytes in chunk_dict.items():
            chunk_array = self._inner_chain.decode_chunk(chunk_bytes, chunk_spec)
            if chunk_array is not None:
                out[chunk_selection(chunk_coords)] = chunk_array
        return out
```

And the full shard sync decode path simplifies from:

```python
# Current: wrap dict in fake ByteGetter, feed to full pipeline
self.codec_pipeline.read_sync(
    [(_ShardingByteGetter(shard_dict, chunk_coords), chunk_spec,
      chunk_selection, out_selection, is_complete_shard)
     for chunk_coords, chunk_selection, out_selection, is_complete_shard in indexer],
    out,
)
```

To:

```python
# New: deserialize shard blob, decode each chunk directly, scatter into output
chunk_dict = self.deserialize(shard_bytes)
for chunk_coords, chunk_selection, out_selection in indexer:
    chunk_bytes = chunk_dict.get(chunk_coords)
    chunk_array = self._inner_chain.decode_chunk(chunk_bytes, chunk_spec)
    if chunk_array is not None:
        out[out_selection] = chunk_array[chunk_selection]
    else:
        out[out_selection] = fill_value_or_default(chunk_spec)
```

**Impact on store protocols**: `ByteGetter` and `ByteSetter` remain as-is for actual store IO — they are used only in `prepare_read`/`prepare_write`. No codec's compute methods (`_decode_sync`, `_encode_sync`, `decode_chunk`, `encode_chunk`) ever touch a `ByteGetter`. The `_ShardingByteGetter`/`_ShardingByteSetter` adapters are eliminated entirely.

**Files**: `src/zarr/core/codec_pipeline.py`, `src/zarr/codecs/sharding.py`, `src/zarr/abc/codec.py`

### 3.2 Simplify store protocols: merge `ByteGetter`/`SyncByteGetter` and `ByteSetter`/`SyncByteSetter`

**Problem**: The codec pipeline currently consumes four separate store protocols:

- `ByteGetter` (async): `get(prototype, byte_range) -> Buffer | None`
- `SyncByteGetter` (sync): `get_sync(prototype, byte_range) -> Buffer | None`
- `ByteSetter` (async): `get()`, `set()`, `delete()`, `set_if_not_exists()`
- `SyncByteSetter` (sync): `get_sync()`, `set_sync()`, `delete_sync()`

This means every store-like object must provide two parallel sets of methods. `StorePath` (the real object passed to the pipeline) implements all four, with the sync variants doing `self.store.get_sync(self.path, ...)` via `# type: ignore[attr-defined]` casts (lines 252-262 of `_common.py`). The `_ShardingByteGetter`/`_ShardingByteSetter` adapters also implement both async and sync versions of every method, even though the async versions are identical to the sync ones (dict lookups).

The root cause is that `read_sync`/`write_sync` use duck-typed `Any` for the byte getter/setter parameter (see `read_sync` signature at line 816: `batch_info: Iterable[tuple[Any, ...]]`), relying on the caller to pass an object with `get_sync`/`set_sync`/`delete_sync`. Meanwhile the async `read`/`write` methods require `ByteGetter`/`ByteSetter`.

**Fix**: After 3.1 eliminates the sharding codec's need for `ByteGetter`/`ByteSetter`, the only consumer of these protocols is the outer `BatchedCodecPipeline`. Merge the sync and async protocols:

```python
@runtime_checkable
class ByteGetter(Protocol):
    async def get(self, prototype, byte_range=None) -> Buffer | None: ...
    def get_sync(self, prototype, byte_range=None) -> Buffer | None: ...

@runtime_checkable
class ByteSetter(ByteGetter, Protocol):
    async def set(self, value: Buffer, byte_range: ByteRequest | None = None) -> None: ...
    async def delete(self) -> None: ...
    def set_sync(self, value: Buffer, byte_range: ByteRequest | None = None) -> None: ...
    def delete_sync(self) -> None: ...
```

This eliminates the separate `SyncByteGetter`/`SyncByteSetter` protocols entirely. `StorePath` already satisfies this merged protocol. The `_can_use_sync_path` check in `array.py` (line 1998) currently checks `isinstance(store, SyncByteGetter)` — it would instead check `hasattr(store, 'get_sync')` or we add a `supports_sync` property to `ByteGetter`.

**Dependency**: Enabled by 3.1. Once the sharding codec no longer creates `_ShardingByteGetter`/`_ShardingByteSetter` objects, the only implementer of `ByteGetter`/`ByteSetter` is `StorePath`. This makes unifying the sync/async protocols safe — there's only one class to update.

**Files**: `src/zarr/abc/store.py`, `src/zarr/storage/_common.py`, `src/zarr/core/array.py`

### 3.3 Adopt zarrs' concurrency budget model

**Problem**: zarr-python has two independent parallelism controls: `async.concurrency` (asyncio semaphore) and `threading.codec_workers` (thread pool config). There's no coordination between them, and no concept of a "concurrency budget" that flows through the codec chain.

zarrs' approach: `CodecOptions.concurrent_target` flows down through the chain. Each level consumes some concurrency and passes the remainder to inner levels. This naturally prevents the nested deadlock problem and provides better utilization.

**Fix**: Add an optional `concurrent_target: int` parameter to `read_sync` / `write_sync`. When not provided, default to `os.cpu_count()`. In the thread pool dispatch:
- Outer level uses `min(n_chunks, concurrent_target)` workers.
- Inner level (e.g., sharding) receives `concurrent_target // outer_workers` as its budget.

This is a more principled version of fix 2.1 and aligns with zarrs' proven approach.

**Note**: This could be implemented as a later evolution of 3.1, since the `CodecChain` / `BatchedCodecPipeline` split provides a natural place to thread the budget through.

**Files**: `src/zarr/core/codec_pipeline.py`, `src/zarr/codecs/sharding.py`

### 3.4 Automatic partial decoder caching (zarrs pattern)

**Problem**: zarr-python has no caching in the partial decoder path. When the sharding codec does partial reads, decompressed data is not cached. If the same shard is accessed multiple times (e.g., iterating over rows that span the same shard), the shard index and inner chunks are re-fetched and re-decompressed each time.

zarrs automatically inserts caches (`BytesPartialDecoderCache`, `ArrayPartialDecoderCache`) in the partial decoder chain based on each codec's declared `PartialDecoderCapability`.

**Fix**: This is a significant feature addition. At minimum:
1. Add `partial_decoder_capability` to the codec protocol (or as a property on `BaseCodec`).
2. In the pipeline, when constructing the partial decode path, insert a cache after the last codec that cannot partially decode (typically: after the bytes-to-bytes compression codec).
3. The cache holds the fully decompressed bytes/array for a chunk, keyed by chunk coordinates.

This is the highest-effort item in the plan and may warrant its own design document. It should be pursued after the Tier 1 and Tier 2 items are complete.

**Files**: `src/zarr/abc/codec.py`, `src/zarr/core/codec_pipeline.py`, new file for cache implementation

---

## Implementation Order

```
Tier 1 (do first, each is independent):
  1.1 Cache ShardingCodec.codec_pipeline
  1.2 Cache shard index codec chain
  1.3 Pass resolved metadata chain to _write_chunk_compute
  1.4 Cache TransposeCodec inverse order

Tier 2 (do next):
  2.1 Prevent nested thread pool deadlock
  2.2 Fuse fetch-decode-scatter in read_sync
  2.3 Fuse fetch-compute-store in write_sync

Tier 3 (larger, do after Tier 1-2 are merged):
  3.1 Extract CodecChain + eliminate _ShardingByteGetter  (subsumes 1.1, 1.2, 2.1; partially addresses 2.2, 2.3)
  3.2 Simplify store protocols                             (depends on 3.1)
  3.3 Concurrency budget model                             (builds on 3.1)
  3.4 Automatic partial decoder caching                    (independent, highest effort)
```

Tier 1 items should each be a separate PR. Tier 2 items can be combined into 1-2 PRs. Tier 3 items are each a standalone project.

**Note on ordering**: Items 1.1 and 1.2 are good quick wins that can be merged immediately. However, they become unnecessary once 3.1 lands (since the sharding codec will hold `CodecChain` instances directly instead of re-creating pipelines). If 3.1 is pursued soon, it may be more efficient to skip 1.1/1.2 and go straight to 3.1. Items 1.3 and 1.4 remain independently valuable regardless of 3.1. Items 2.2 and 2.3 (fuse fetch-decode-scatter, fuse fetch-compute-store) are partially addressed by 3.1's prepare/execute structure, which naturally separates IO and compute phases — but the memory optimization (don't hold all decoded chunks simultaneously) remains independently valuable for the non-sharding case.

---

## What We're NOT Doing (and Why)

### Streaming codec composition (Tensorstore's Riegeli pattern)

Tensorstore uses Riegeli streaming writers/readers for bytes-to-bytes codecs, enabling zero-copy data flow through compression stages. This avoids intermediate buffer allocation between codec stages.

**Why not**: zarr-python's buffer allocation between codec stages is already efficient (the research showed ~1-2 allocations per decode, mostly unavoidable compression output buffers). The dominant cost is the compression/decompression itself, not the buffer allocation. Adopting streaming composition would require a fundamental redesign of all codec interfaces for marginal gain. The C libraries (blosc, gzip, zstd) already allocate their own output buffers internally regardless of how we compose them.

### Write coalescing / MaskedArray (Tensorstore pattern)

Tensorstore accumulates multiple writes to the same chunk in memory via `MaskedArray`, flushing once on writeback.

**Why not**: This requires a chunk cache with write-back semantics, transaction support, and generation tracking -- a massive increase in complexity. zarr-python's "immediate write-through" model is simpler and matches user expectations. Users who need write coalescing can batch their writes at the application level.

### Store-level transactions and generation tracking (Tensorstore pattern)

**Why not**: These are store-level features orthogonal to codec pipeline improvements. They could be added independently of any codec pipeline changes.

**Note**: Store abstraction *simplifications* are in scope — removing `_ShardingByteGetter`/`_ShardingByteSetter` (item 3.1) and merging the four separate `ByteGetter`/`SyncByteGetter`/`ByteSetter`/`SyncByteSetter` protocols (item 3.2). What's out of scope is adding new store-level complexity (transactions, generation counters, etc.).

### `decode_into` / zero-copy scatter (zarrs pattern)

zarrs decodes directly into preallocated output buffers using `UnsafeCellSlice`.

**Why not**: Python/numpy does not provide a safe equivalent of Rust's `UnsafeCellSlice`. numpy slice assignment (`out[sel] = chunk_array[sel]`) already avoids unnecessary copies. The scatter phase is not a bottleneck -- the codec compute and IO dominate.

### Layer-by-layer async batch optimization

zarr-python's `decode_batch` processes all chunks through one codec layer before the next. This is unique among the three libraries and was designed to allow codecs to exploit batch-level optimizations.

**Why not change it**: The async batch path is only used when `_all_sync` is False (custom/numcodecs codecs). For built-in codecs, the per-chunk sync path is used. Changing the async path has limited impact, and some third-party codecs may genuinely benefit from batch processing (e.g., GPU codecs that can decode multiple chunks in a single kernel launch).

---

## Comparison: Proposed Approach vs Tensorstore vs zarrs

This section compares our proposed architecture (prepare/execute with `CodecChain`) against the designs used by Tensorstore (C++) and zarrs (Rust). See `agent_notes/tensorstore_zarrs_comparison.md` for the full research.

### IO/compute separation

| | Tensorstore | zarrs | Proposed |
|---|---|---|---|
| **IO boundary** | KvStore layer (below codecs) | Store traits (below codecs) | `prepare_read`/`prepare_write` (on `ArrayBytesCodec`, called by pipeline) |
| **Codecs do IO?** | No — codecs receive bytes via Riegeli readers, never touch KvStore | No — codecs receive `ArrayBytes`, never touch store | No — `decode_chunk`/`encode_chunk` are pure compute |
| **Sharding IO** | KvStore adapter does all IO; codec chain receives bytes | Codec does IO in partial decode path (via `ReadableStorageTraits`) | `prepare_read` does all IO; `decode_chunk` is pure compute |
| **Separation enforced by** | Type system: codecs take `riegeli::Reader&`, not `KvStore` | Convention: sync codecs take `ArrayBytes`, but partial decoders take store reference | Convention: `prepare_read`/`prepare_write` take `ByteGetter`; `decode_chunk`/`encode_chunk`/`deserialize`/`serialize` take in-memory data |

**Our approach is closest to Tensorstore's.** Both enforce that the codec compute path never touches the store. The difference is that Tensorstore achieves this by making sharding a KvStore adapter (IO layer), while we achieve it by splitting the codec into IO methods (`prepare_read`/`prepare_write`) and pure compute (`decode_chunk`, `encode_chunk`, `deserialize`, `serialize`). zarrs is less strict: its partial decoder chain passes store references into codecs, so the sharding codec does IO internally during partial decode.

### Sharding model

| | Tensorstore | zarrs | Proposed |
|---|---|---|---|
| **Abstraction** | Virtual KvStore adapter | Codec with partial encode/decode | Codec with prepare/execute split |
| **Inner chunks accessed via** | KvStore `Read(key)` on the virtual adapter | `CodecChain.decode()` with store reference | `CodecChain.decode_chunk()` with pre-fetched bytes |
| **Shard-as-store?** | Yes — full KvStore semantics (read, write, list, delete) | No — codec with inner `CodecChain` | No — codec with inner `CodecChain` |
| **Nested sharding** | Supported (`sharding_height()` tracks depth) | Supported: `CodecChain` implements `ArrayToBytesCodecTraits`, so sharding's inner `CodecChain` can itself contain sharding | Naturally supported: inner sharding is pure compute on pre-fetched bytes |
| **Partial shard write** | Via KvStore adapter (rewrites shard) | Experimental `partial_encode` (updates inner chunk in-place) | Compressed: `prepare_write` fetches entire shard; `serialize` → `{0: blob}` (dense). Uncompressed: `prepare_write` fetches only touched chunks; `serialize` → targeted byte ranges (sparse). Pipeline commit loop handles both. |

**Our approach takes the codec-based model (like zarrs) but achieves the IO separation of Tensorstore.** Tensorstore's KvStore adapter is more composable in C++ (caching, transactions, and batching compose for free), but in Python the overhead of protocol dispatch at each virtual store layer would be measurable. Our approach avoids that overhead while still achieving strict IO/compute separation.

All three implementations support nested sharding. Tensorstore uses explicit `sharding_height()` tracking and recursive KvStore wrapping. zarrs achieves it through composability: `CodecChain` implements `ArrayToBytesCodecTraits`, so a sharding codec's inner `CodecChain` can itself contain another sharding codec — no special nesting code needed, it falls out of the type system. Our approach is similar in spirit to zarrs': the outer `prepare_read` fetches the shard blob, and the inner sharding codec's `_decode_sync` operates on in-memory bytes — nested sharding works naturally without special support.

### Composability

| | Tensorstore | zarrs | Proposed |
|---|---|---|---|
| **Pipeline = codec?** | No — `ZarrCodecChain` is separate from codecs | Yes — `CodecChain` implements `ArrayToBytesCodecTraits` | No — `CodecChain` is separate from codecs, but sharding holds `CodecChain` directly (see note below) |
| **Sharding inner pipeline** | KvStore adapter (not a codec chain) | `CodecChain` (is itself a codec) | `CodecChain` (used directly, no pipeline overhead) |

zarrs has the cleanest composability: `CodecChain` is itself an `ArrayToBytesCodec`, so the sharding codec's inner pipeline is just another codec. We achieve a similar result — the sharding codec holds a `CodecChain` and calls `decode_chunk`/`encode_chunk` directly — but `CodecChain` is not itself a codec (it doesn't implement `ArrayBytesCodec`). This is a deliberate choice: making `CodecChain` a codec would mean it needs `_decode_sync`/`_encode_sync` methods, which would conflate the chain with individual codecs.

### Concurrency model

| | Tensorstore | zarrs | Proposed |
|---|---|---|---|
| **Parallelism** | Unified C++ executor | Two-level rayon (chunks × codec) | ThreadPoolExecutor for codec compute |
| **Concurrency budget** | Single `data_copy_concurrency` | `concurrent_target` flows through chain | Planned (item 3.3): adopt zarrs' model |
| **Thread pool + sharding** | No issue (C++ executor) | Budget flows to inner level | No issue: inner codecs are pure compute, no pool needed |

Our prepare/execute split solves the nested thread pool deadlock problem that zarr-python currently has. Because inner codec compute (`decode_chunk`) never does IO, it's safe to run on the thread pool without risk of deadlock. The concurrency budget model (item 3.3) would further refine this by flowing a budget through the chain, zarrs-style.

### Partial decode/encode

| | Tensorstore | zarrs | Proposed |
|---|---|---|---|
| **Partial decode** | KvStore byte-range reads + batch optimization | Partial decoder chain with automatic cache insertion | `prepare_read` handles byte-range IO; pipeline unaware of full-vs-partial |
| **Pipeline branching** | No branching — KvStore adapter handles it | No branching — `partial_decoder()` returns appropriate chain | No branching — `prepare_read` decides internally, pipeline always calls prepare then execute |
| **Automatic caching** | No (relies on LRU chunk cache) | Yes — `BytesPartialDecoderCache` / `ArrayPartialDecoderCache` inserted automatically | Planned (item 3.4) |

All three approaches eliminate pipeline-level branching on `supports_partial_decode`, but by different mechanisms. Tensorstore hides it in the KvStore layer. zarrs builds a different decoder chain at construction time. We hide it in `prepare_read`: the same pipeline code always calls `prepare_read` then `decode_chunk`, and `prepare_read` internally decides whether to fetch the full chunk or do byte-range reads.

### Summary

Our proposed architecture is a pragmatic middle ground:

- **Like Tensorstore**: strict IO/compute separation — codecs never do IO in the compute path.
- **Like zarrs**: codec-based sharding model with inner `CodecChain`, avoiding the overhead of virtual store layers.
- **Unlike both**: the prepare/execute split is explicit in the API — not hidden in a KvStore adapter (Tensorstore) or implicit in how the partial decoder chain is constructed (zarrs).
- **Unlike Tensorstore**: no streaming codec composition (Riegeli), no write coalescing, no transactions. These are out of scope.
- **Unlike zarrs**: no `decode_into` (Python/numpy limitation), no compile-time codec registration. Automatic partial decoder caching is planned as a future item.

The main trade-off is that `prepare_read` lives on `ArrayBytesCodec`, which means the individual codec has a method that does IO. However, `CodecChain` itself has zero IO methods — it is purely compute. The IO/compute boundary is visible at the type level: calling `decode_chunk`/`encode_chunk` on a `CodecChain` is guaranteed to be pure compute, while the pipeline explicitly calls `codec_chain.array_bytes_codec.prepare_read(...)` when it needs IO. This is more practical in Python than Tensorstore's approach (where codecs physically cannot access the store via the C++ type system), while still making the separation explicit enough to prevent accidental IO in compute paths.
