# The Chunk-as-Layered-Array Redesign

## The core idea

A stored chunk is not "data that a pipeline processes." A chunk **is** an array-like object, constructed from layers. Each layer corresponds to one array-returning codec. The chunk knows its own shape, dtype, and how to materialize itself.

We split this into two concepts:

- **`ChunkTransform`**: A codec chain bound to an `ArraySpec`. Pure compute — knows how to encode/decode but does not own data or store references. Has a `layers` attribute (one entry per `ArrayArrayCodec`), `shape`/`dtype` (the post-AA-transform representation), and `encode_chunk()`/`decode_chunk()` methods.

- **`Chunk`** (future): Combines a `ChunkTransform` with either explicit bytes or a store reference. Supports `read()`/`read_async()` (IO + decode) and `write(array)`/`write_async(array)` (encode + IO).

```python
# ChunkTransform — pure compute, no IO (implemented)
transform = ChunkTransform(codecs, chunk_spec)
transform.shape          # post-AA-transform shape
transform.dtype          # post-AA-transform dtype
transform.layers         # ((aa_codec, input_spec), ...)
decoded = transform.decode_chunk(raw_bytes)   # Buffer → NDBuffer
encoded = transform.encode_chunk(nd_buffer)   # NDBuffer → Buffer

# Chunk — owns data reference (future)
chunk.read()             # → NDBuffer  (sync, does IO if needed)
chunk.read_async()       # → NDBuffer  (async, does IO)
chunk.write(array)       # → None      (sync, does IO)
chunk.write_async(array) # → None      (async, does IO)
```

The key shift: **the chunk is the unit of abstraction, not the pipeline.** The pipeline becomes a thin dispatcher that constructs chunks and coordinates IO.

---

## `ChunkTransform` structure

```python
@dataclass(slots=True)
class ChunkTransform:
    codecs: tuple[Codec, ...]
    chunk_spec: ArraySpec

    # Computed in __post_init__:
    layers: tuple[tuple[ArrayArrayCodec, ArraySpec], ...]  # (codec, input_spec) per AA codec
    _ab_codec: ArrayBytesCodec
    _ab_spec: ArraySpec       # spec after all AA codecs (input to AB codec)
    _bb_codecs: tuple[BytesBytesCodec, ...]
    _all_sync: bool

    @property
    def shape(self) -> tuple[int, ...]:
        return self._ab_spec.shape  # post-AA-transform

    @property
    def dtype(self) -> ZDType:
        return self._ab_spec.dtype  # post-AA-transform
```

Key design decisions:
- **Not frozen** — we don't mutate or hash these, so `frozen=True` adds unnecessary `object.__setattr__` ceremony.
- **No `_bb_spec`** — `BytesBytesCodec.resolve_metadata` and `ArrayBytesCodec.resolve_metadata` return the spec unchanged (default `BaseCodec` implementation), so BB codecs use the same `_ab_spec`.
- **`layers`** — each element is `(ArrayArrayCodec, input_spec)`. For a chain with `TransposeCodec → BytesCodec → GzipCodec` on a `(3,4)` array, `layers` has one entry: `(TransposeCodec, ArraySpec(shape=(3,4)))`. The `ChunkTransform.shape` is then `(4, 3)`.

---

## Comparison with zarrs and tensorstore

### zarrs (Rust)

zarrs has a `CodecChain` struct that implements `ArrayToBytesCodecTraits` — meaning a codec chain is itself a codec. This is compositional: sharding nests a `CodecChain` as its inner array-to-bytes codec. zarrs tracks per-layer representations explicitly via `get_representations()`, building vectors of `(shape, dtype, fill_value)` at each layer boundary. zarrs also computes an optimal cache insertion point (`cache_index`) at construction time, and builds partial decoder chains by composing `BytesPartialDecoderTraits` / `ArrayPartialDecoderTraits` objects in reverse.

**Our `ChunkTransform` does the zarrs `CodecChain` thing** — eagerly resolving per-layer specs in `__post_init__` and exposing them via `layers`. Where zarrs goes further is (a) the compositionality (codec chain IS a codec), and (b) the partial decoder chain with automatic cache placement.

### tensorstore (C++)

tensorstore uses a three-phase model: Spec → Resolved Codec → PreparedState (bound to shape). The key architectural insight is the **kvstore adapter pattern for sharding**: the sharding codec doesn't process data through a nested pipeline — it creates a *new kvstore* that presents sub-chunks as individually addressable keys. The inner codec chain then operates against this adapted kvstore as if reading independent chunks, completely unaware it's inside a shard.

tensorstore also has an explicit three-level chunk grid (write_chunk / read_chunk / codec_chunk) that makes the shard/chunk/sub-chunk hierarchy first-class.

**Our layered model takes a different path than tensorstore's kvstore adapter.** Rather than creating virtual stores, we make the chunk itself hierarchical: a sharded chunk's array-to-bytes layer is itself a container of inner `ChunkTransform` objects. This is closer to zarrs' compositional approach.

### Where our approach differs from both

Neither zarrs nor tensorstore separates the "codec chain bound to a spec" (`ChunkTransform`) from the "chunk with data" (`Chunk`). In zarrs, `CodecChain` is a stateless codec you call `encode()`/`decode()` on with data + metadata arguments. In tensorstore, `PreparedState` is bound to a shape but still operates on data passed in.

Our two-level split means:
- `ChunkTransform` is the pure-compute object (comparable to zarrs' `CodecChain` or tensorstore's `PreparedState`)
- `Chunk` (future) **encapsulates both the transform AND the storage location**. When you call `chunk.read_async()`, the chunk knows where to fetch bytes from, how to decode them, and what shape/dtype the result will have.

---

## What changes concretely

### Current architecture

```
AsyncArray._get_selection()
  → builds list of (ByteGetter, ArraySpec, chunk_sel, out_sel, is_complete)
  → passes to CodecPipeline.read(batch_info, out)
    → pipeline resolves metadata, orchestrates IO + codec compute
```

The pipeline is the god object. It knows about IO, codec ordering, metadata resolution, thread pools, sharding inner chunks, partial decode, etc.

### Proposed architecture

```
AsyncArray._get_selection()
  → builds list of Chunk objects (each knows its store path, transform, spec)
  → for each chunk: result = chunk.read() or chunk.read_async()
  → scatters results into output buffer
```

The pipeline shrinks to a **dispatcher** that:
1. Constructs `Chunk` objects from `ChunkTransform` + store paths
2. Decides sync vs async execution strategy
3. Manages parallelism (thread pool / concurrent_map)
4. Scatters results into the output buffer

### How sharding uses ChunkTransform

`ShardingCodec` constructs a `ChunkTransform` from its inner codecs and uses `decode_chunk`/`encode_chunk` directly — no pipeline re-entry:

```python
class ShardingCodec(ArrayBytesCodec):
    def _decode_single_sync(self, shard_bytes, shard_spec):
        shard_dict = self._parse_shard_index(shard_bytes)
        inner_transform = ChunkTransform(codecs=self.codecs, chunk_spec=self._inner_spec(shard_spec))
        out = create_output_buffer(shard_spec)
        for coords, chunk_sel, out_sel in self._inner_indexer(shard_spec):
            raw = shard_dict.get(coords)
            decoded = inner_transform.decode_chunk(raw)
            if decoded is not None:
                out[out_sel] = decoded[chunk_sel]
            else:
                out[out_sel] = fill_value
        return out
```

No inner `BatchedCodecPipeline`. No `_ShardingByteGetter`/`_ShardingByteSetter`. No pipeline re-entry. The sharding codec is self-contained.

### What simplifies

#### 1. `batch_info` tuples disappear

Currently, `batch_info` is a list of 5-tuples threaded through multiple functions. With chunks owning their transform and spec, the `ArraySpec` moves inside the object.

#### 2. `prepare_read` / `prepare_write` / `finalize_write` collapse

With `ChunkTransform` owning pure-compute encode/decode, the pipeline only needs to coordinate IO and scattering. The `PreparedWrite` data structure becomes an internal detail of `ShardingCodec`.

#### 3. Sharding becomes composition, not special-casing

The pipeline no longer needs `inner_chain = ab_codec.inner_chunk or chunk`. The sharding codec constructs inner `ChunkTransform` objects internally.

#### 4. `_codecs_with_resolved_metadata_batched` disappears

`ChunkTransform` eagerly resolves specs in `__post_init__`. No need to re-resolve per batch.

#### 5. `ArraySpec` threading reduces dramatically

With `ChunkTransform` owning its spec, external code accesses `transform.shape`, `transform.dtype`, `transform.chunk_spec` when needed.

### What stays the same

- **Codec interface**: `_decode_sync` / `_encode_sync` / `resolve_metadata` don't change
- **`SupportsSyncCodec` protocol**: Still needed to detect sync-capable codecs
- **Thread pool infrastructure**: Still used for parallel codec compute
- **`concurrent_map`**: Still used for async IO overlap
- **Store interface**: `ByteGetter` / `ByteSetter` / `StorePath` unchanged

---

## Implementation plan

### PR 5 (current): `ChunkTransform` with `layers`, `shape`, `dtype`

On branch `perf/codec-chain`. `CodecChain` → `ChunkTransform` with:
- `layers: tuple[tuple[ArrayArrayCodec, ArraySpec], ...]` — one entry per AA codec
- `shape` / `dtype` properties — post-AA-transform representation
- No `_bb_spec` (BB codecs use `_ab_spec`)
- Not frozen (simplifies construction)
- `encode_chunk()` / `decode_chunk()` — pure compute, sync only

Pre-commit clean, all tests pass.

### PR 6: Remove `chunk_spec` parameter redundancy

**Goal**: Any function that receives a `ChunkTransform` no longer takes a separate `chunk_spec` parameter.

Files:
- `src/zarr/core/codec_pipeline.py`
  - `_write_chunk_compute_default`: remove `chunk_spec` param, use `transform.chunk_spec`
  - `_read_chunk` closure in `read_batch`: remove `chunk_spec` from closure args
  - `_write_chunk` closure in `write_batch`: remove `chunk_spec` from closure args
  - `read_sync` / `write_sync` non-threaded paths: use `transform.chunk_spec`
  - `write_sync` threaded path `pool.map`: remove `chunk_spec` list arg

- `src/zarr/abc/codec.py`
  - `prepare_read_sync`: remove `chunk_spec` param, use `transform.chunk_spec`
  - `prepare_read`: remove `chunk_spec` param, use `transform.chunk_spec`

### PR 7: Sharding uses `ChunkTransform` directly (no pipeline re-entry)

**Goal**: `ShardingCodec` constructs inner `ChunkTransform` objects and uses `decode_chunk()`/`encode_chunk()` directly, eliminating pipeline re-entry and the `inner_chunk` protocol.

Files:
- `src/zarr/codecs/sharding.py`
  - `ShardingCodec._decode_single` / `_encode_single`: Build inner `ChunkTransform` from `self.codecs` + inner spec, use `transform.decode_chunk()` / `encode_chunk()`
  - Remove `self.codec_pipeline` property — no more inner `BatchedCodecPipeline`
  - Add `SupportsSyncCodec` to `ShardingCodec`

- `src/zarr/abc/codec.py`
  - Remove `inner_chunk` property from `ArrayBytesCodec`

- `src/zarr/core/codec_pipeline.py`
  - Remove `inner_chain = ab_codec.inner_chunk or chunk` pattern
  - Simplify write path without inner-chunk awareness

### PR 8: Write mode flag (replace vs read-modify-write)

**Goal**: Replace the current `is_complete_chunk` boolean with a clearer write mode.

#### Why not SliceCodec

We considered a `SliceCodec` (an `ArrayArrayCodec` that trims/pads edge chunks), but rejected it: the regular chunk grid always stores full-size chunks on disk, and the codec chain should only know about transforming data — not about where a chunk sits within the larger array.

#### What changes

- **Replace mode**: The caller is overwriting the entire chunk. No need to read existing data. The pipeline calls `transform.encode_chunk(value)` directly.
- **Read-modify-write mode**: The caller is updating a sub-region. The pipeline reads existing bytes, decodes, merges new data into the decoded array, re-encodes, and writes back.

### PR 9 (future): Simplify `batch_info` and pipeline interface

**Goal**: The pipeline receives `ChunkTransform` objects (or `Chunk` objects) instead of 5-tuples.

### PR 10 (future): Selection propagation through layers

**Goal**: Partial decode — read a selection without materializing the full chunk.

- Add `propagate_selection(selection) -> selection` to `ArrayArrayCodec`
- `ChunkTransform.decode_partial(bytes, selection) -> NDBuffer`: propagates selection backward through `layers`

---

## Risk assessment

### Low risk
- PR 5 (done) and PR 6 (chunk_spec removal) are incremental, backwards-compatible internal refactors

### Medium risk
- PR 7 (sharding refactor) touches the most complex codec. Must carefully handle nested sharding, partial read/write, and index codec pipeline.

### Low risk
- PR 8 (write mode flag) is a naming/clarity refactor of existing logic.

### Low-medium risk
- PR 9 (batch_info simplification) is an internal API change.

### Speculative
- PR 10 (selection propagation) is genuinely new functionality.

---

## Key insight: what the layered model buys us

The deepest value is the **elimination of the pipeline as orchestrator of codec internals**. Today:

1. Pipeline knows about sharding (via `inner_chunk`, `PreparedWrite.chunk_dict`)
2. Pipeline resolves metadata that codecs already know about
3. Pipeline manages IO interleaving that only matters for specific codecs

With `ChunkTransform` + future `Chunk`:

1. Each codec handles its own complexity internally
2. Metadata resolution happens once, at `ChunkTransform` construction
3. IO strategy is chosen at the dispatch level, not woven through codec execution

The pipeline becomes what it should be: **a scheduler**, not an interpreter.
