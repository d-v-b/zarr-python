# Merge plan: `perf/smarter-codecs` → `main`

## Overview

The `perf/smarter-codecs` branch contains ~3,430 lines of changes across 27 files.
The changes achieve 5–22x speedups on sharded read/write benchmarks.
This document proposes a sequence of self-contained PRs that land these changes incrementally without breaking existing workflows.

## Guiding principles

1. **Each PR must leave `main` green.** All existing tests pass, pre-commit passes.
2. **Additive first.** New APIs, protocols, and classes are added alongside existing ones. Nothing is removed or replaced until a later PR wires things together.
3. **Opt-in where possible.** The sync fast path and `set_range` optimizations can be gated behind a config flag (e.g. `zarr.config["codec_pipeline.sync_fast_path"]`) so they can be merged without changing default behavior, then enabled by default once stable.
4. **Small surface area per PR.** Each PR touches a focused set of files so reviewers can reason about correctness locally.

---

## Branch chain

Each branch builds linearly on the previous one. Store sync methods originate in `perf/store-sync` and flow forward.

```
main
 └── perf/sync-codec-protocol   (PR 1)
      └── perf/store-sync        (PR 1 + PR 2 partial + PR 3)
           └── perf/codec-chain  (completes PR 2: ChunkTransform)
                └── perf/chunkrequest  (PR 4 + PR 5 + PR 6 partial)
```

Milestone branches (not part of the chain, for reference only):
- `perf/prepared-write` — snapshot after PR 4 work (needs rebase onto `perf/codec-chain`)
- `perf/sharding-chunk-transform` — snapshot after PR 6 partial (needs rebase onto `perf/codec-chain`)

---

## Dependency graph

```
PR1  SupportsSyncCodec + _encode_sync/_decode_sync on all codecs
 │
PR2  ChunkTransform (pure-compute codec wrapper)
 │
PR3  Store layer: set_range + sync methods
 │
PR4  PreparedWrite + prepare/finalize pattern on ArrayBytesCodec
 │
PR5  BatchedCodecPipeline refactor (uses ChunkTransform, read_sync/write_sync)
 │
PR6  ShardingCodec refactor (prepare_write, finalize_write, set_range, dense layout)
 │
PR7  Vectorized shard encoding (_encode_vectorized, _encode_vectorized_sparse)
 │
PR8  Sync fast path in Array (opt-in config flag)
```

PRs 1–3 are independent of each other and can be developed/reviewed in parallel.
PRs 4–8 are sequential — each builds on the previous.

---

## PR 1: `SupportsSyncCodec` protocol + sync codec methods

**Branch**: `perf/sync-codec-protocol` (1 commit: `2b64daa6`)
**Status**: Done

### Motivation
Establishes a runtime-checkable protocol so the pipeline can detect which codecs support synchronous operation.
This is purely additive — existing async paths are unchanged.

### Files
| File | Changes |
|---|---|
| `src/zarr/abc/codec.py` | Add `SupportsSyncCodec` protocol (runtime-checkable, declares `_encode_sync` / `_decode_sync`) |
| `src/zarr/codecs/bytes.py` | Add `_encode_sync`, `_decode_sync`; async methods delegate to sync |
| `src/zarr/codecs/blosc.py` | Same pattern |
| `src/zarr/codecs/gzip.py` | Same pattern |
| `src/zarr/codecs/zstd.py` | Same pattern; fix `is_fixed_size` from `True` → `False` |
| `src/zarr/codecs/crc32c_.py` | Same pattern |
| `src/zarr/codecs/transpose.py` | Same pattern |
| `src/zarr/codecs/vlen_utf8.py` | Same pattern |
| `tests/test_sync_codec_pipeline.py` | Protocol conformance tests, encode/decode round-trips |

### Risk
None. Purely additive. Async paths unchanged. All codecs still work exactly as before.

### Review checklist
- [ ] Every codec that implements `_encode_sync` also satisfies `isinstance(codec, SupportsSyncCodec)`
- [ ] Async methods delegate to sync (no duplicated logic)
- [ ] `zstd.is_fixed_size` is now `False`

---

## PR 2: `ChunkTransform` dataclass

**Branch**: `perf/codec-chain` (incremental commit: `71a780b1`)
**Status**: Done
**Note**: The intermediate steps (CodecChain → refactor → separate specs) are in `perf/store-sync` commits `cd4efb0a`, `41b7a6ad`, `5a2a884e`. The final ChunkTransform rename is in `perf/codec-chain`.

### Motivation
A codec chain bound to an `ArraySpec`. Pure compute — knows how to encode/decode but does not own data or store references. Provides `encode_chunk` / `decode_chunk` methods, a `layers` attribute (one entry per `ArrayArrayCodec`), and `shape`/`dtype` properties reflecting the post-AA-transform representation. No IO, no threading.

### Files
| File | Changes |
|---|---|
| `src/zarr/core/codec_pipeline.py` | Add `ChunkTransform` class (~110 lines). `layers`, `shape`, `dtype`, `encode_chunk()`, `decode_chunk()`, `all_sync`. |
| `tests/test_sync_codec_pipeline.py` | ChunkTransform construction, layers/shape/dtype tests, round-trip encode/decode tests |

### Dependency
PR 1 (uses `SupportsSyncCodec` to compute `_all_sync`).

### Risk
None. `ChunkTransform` is a new class — nothing references it yet. `BatchedCodecPipeline` is unchanged.

### Review checklist
- [ ] `_all_sync` is True only when all codecs in the chain implement `SupportsSyncCodec`
- [ ] `encode_chunk`/`decode_chunk` match the existing pipeline's encode/decode behavior
- [ ] `layers` has one entry per `ArrayArrayCodec`, each paired with its input `ArraySpec`
- [ ] `shape`/`dtype` reflect the post-AA-transform representation (input to `ArrayBytesCodec`)

---

## PR 3: Store layer — `set_range` + sync methods

**Branch**: `perf/store-sync` (incremental commit: `4e262b17`)
**Status**: Done

### Motivation
Adds the building blocks for byte-range writes and synchronous store IO:
- `set_range(key, value, start)` — write bytes at an offset within an existing blob
- `get_sync`, `set_sync`, `delete_sync`, `set_range_sync` — synchronous IO methods

These are additive — the default `set_range` raises `NotImplementedError`, and sync methods sit alongside existing async methods.

### Files
| File | Changes |
|---|---|
| `src/zarr/abc/store.py` | `SupportsGetSync`, `SupportsSetSync`, `SupportsSetRangeSync`, `SupportsDeleteSync`, `SupportsSyncStore` runtime-checkable protocols |
| `src/zarr/storage/_common.py` | `StorePath.get_sync()`, `set_sync()`, `set_range_sync()`, `delete_sync()` |
| `src/zarr/storage/_memory.py` | `MemoryStore`: `get_sync()`, `set_sync()`, `delete_sync()`, `set_range_sync()`, `_set_range_impl()` |
| `src/zarr/storage/_local.py` | `LocalStore`: `_put_range()` helper, `_ensure_open_sync()`, `get_sync()`, `set_sync()`, `delete_sync()`, `set_range_sync()` |
| `src/zarr/testing/store.py` | `StoreTests`: `_require_get_sync`, `_require_set_sync`, etc.; tests for `get_sync`, `set_sync`, `delete_sync`, `set_range_sync` |

### Risk
Low. Sync methods are new — nothing calls them yet.

### Review checklist
- [ ] `_put_range` on LocalStore uses seek+write (no atomic rename — intentional for partial writes)
- [ ] `_set_range_impl` on MemoryStore uses numpy buffer slice assignment
- [ ] Sync methods on `StorePath` delegate to store sync methods with `type: ignore[attr-defined]`
- [ ] `_ensure_open_sync` on LocalStore mirrors the async `_open` logic

---

## PR 4: `PreparedWrite` + prepare/finalize pattern on `ArrayBytesCodec`

**Branch**: `perf/chunkrequest` (incremental commits: `9b949849`, `8c33b349`, `1193d9ca`)
**Milestone branch**: `perf/prepared-write` (needs rebase onto `perf/codec-chain`)
**Status**: Done (code exists in `perf/chunkrequest`)

### Motivation
Introduces the `PreparedWrite` dataclass and the prepare/finalize write protocol. This is the architectural pattern that lets `ShardingCodec` control both the read (prepare) and write (finalize) phases while the pipeline handles the compute in between.

Default implementations on `ArrayBytesCodec` preserve existing behavior — only `ShardingCodec` will override (in PR 6).

### Files
| File | Changes |
|---|---|
| `src/zarr/abc/codec.py` | `PreparedWrite` dataclass; `ArrayBytesCodec` methods: `prepare_read_sync()`, `prepare_write_sync()`, `finalize_write_sync()`, async variants, `serialize()`, `deserialize()`, `inner_codec_chain` property, `_is_complete_selection()` helper |

### Dependency
PR 2 (`PreparedWrite.inner_codec_chain` is typed as `Any` to avoid circular import, but conceptually carries a `ChunkTransform`).

### Risk
Low. All new methods have default implementations that match current behavior. No callers yet.

### Design note
`inner_codec_chain` is typed as `Any` (with a comment `# ChunkTransform`) to avoid circular imports between `abc/codec.py` and `core/codec_pipeline.py`. This is an acceptable trade-off; the type is checked at runtime by the pipeline.

### Review checklist
- [ ] Default `prepare_write_sync` returns a `PreparedWrite` with sensible defaults
- [ ] Default `finalize_write_sync` does serialize + set (same as current behavior)
- [ ] `_is_complete_selection` correctly detects full-chunk writes
- [ ] No circular imports

---

## PR 5: `BatchedCodecPipeline` refactor — `ChunkTransform` integration, `read_sync`/`write_sync`

**Branch**: `perf/chunkrequest` (incremental commits: `13b52fd5`, `56dbe612`, `9473b2af`)
**Status**: Done (code exists in `perf/chunkrequest`)

### Motivation
Adds `ReadChunkRequest` / `WriteChunkRequest` dataclasses (replacing anonymous tuples in `batch_info`), wires `ChunkTransform` into `BatchedCodecPipeline`, and adds synchronous `read_sync`/`write_sync` methods that bypass the async event loop.

This PR also adds thread pool management (`_choose_workers`, `_get_pool`, `_mark_pool_worker`) with the `threading.codec_workers` config key.

### Opt-in strategy
The `supports_sync_io` property is added but nothing calls it yet (Array integration comes in PR 8). The sync paths exist but are dormant.

### Files
| File | Changes |
|---|---|
| `src/zarr/core/codec_pipeline.py` | `ReadChunkRequest`, `WriteChunkRequest` dataclasses; `get_chunk_transform()` method; `read_sync()`, `write_sync()`, `supports_sync_io` property; thread pool management (`_choose_workers`, `_get_pool`, `_mark_pool_worker`); `_scatter` helper |
| `src/zarr/abc/codec.py` | `get_chunk_transform()` abstract method on `CodecPipeline`; `supports_sync_io`, `read_sync`, `write_sync` non-abstract defaults on `CodecPipeline` |
| `src/zarr/core/array.py` | Updated to use `ReadChunkRequest` / `WriteChunkRequest` with `codec_pipeline.get_chunk_transform()` |
| `src/zarr/core/config.py` | Add `threading.codec_workers` config key |
| `tests/test_sync_codec_pipeline.py` | `TestSyncPipeline` (write_sync/read_sync round-trips), `TestChooseWorkers` |
| `tests/test_config.py` | Config key test, `MockCodecPipeline` updated for `WriteChunkRequest` |

### Dependency
PR 2 (ChunkTransform), PR 3 (store sync methods — `read_sync`/`write_sync` call `get_sync`/`set_sync`/`delete_sync`), PR 4 (prepare/finalize pattern — `write_sync` calls `prepare_write_sync` and `finalize_write_sync`).

### Risk
Medium. This is the largest pipeline change. Key concerns:
- Thread pool deadlock prevention (thread-local flag `_mark_pool_worker` / `_is_pool_worker`)
- `_choose_workers` heuristic must not over-thread
- `ReadChunkRequest` / `WriteChunkRequest` must be backwards-compatible with existing `batch_info` consumers

### Review checklist
- [ ] `_choose_workers(0, ...)` returns 0 when there are no `BytesBytesCodec`s
- [ ] Thread-local deadlock prevention works (nested `_get_pool` calls don't create inner pools)
- [ ] Existing async `write_batch`/`read_batch` paths unchanged
- [ ] `get_chunk_transform()` on `BatchedCodecPipeline` correctly constructs `ChunkTransform` from its codecs

---

## PR 6: `ShardingCodec` refactor — prepare/finalize, `set_range`, dense layout

**Branch**: `perf/chunkrequest` (incremental commit: `b1c245e4`)
**Milestone branch**: `perf/sharding-chunk-transform` (needs rebase onto `perf/codec-chain`)
**Status**: Partial — sharding wired to ChunkTransform, but dense layout / set_range / vectorized encoding not yet implemented

### Motivation
This is the core optimization PR. `ShardingCodec` overrides `prepare_write_sync`/`finalize_write_sync` to:
1. Detect complete-shard writes and skip per-inner-chunk iteration
2. Use byte-range reads for partial fixed-size shard updates (only fetch affected inner chunks)
3. Use `set_range` for partial fixed-size shard writes (write only modified chunks + index)
4. Fall back gracefully when `set_range` raises `NotImplementedError`

Also introduces dense shard layout helpers (`_morton_rank_map`, `_chunk_byte_offset`, `_build_dense_shard_index`, `_build_dense_shard_blob`) and removes the old `_ShardingByteGetter`/`_ShardingByteSetter` classes.

### Files
| File | Changes |
|---|---|
| `src/zarr/codecs/sharding.py` | Override `prepare_write_sync`/`finalize_write_sync`/async variants; `_prepare_write_partial_fixed_sync`; dense layout helpers; `_inner_codecs_fixed_size`, `_inner_chunk_byte_length`; remove `_ShardingByteGetter`/`_ShardingByteSetter`; remove `ArrayBytesCodecPartialEncodeMixin` inheritance |
| `tests/test_codecs/test_sharding.py` | `test_sharding_subchunk_writes_are_independent` (verifies `set_range` usage via `LoggingStore`); `test_nested_sharding_subchunk_writes_are_independent` |

### Dependency
PR 3 (`set_range`), PR 4 (`PreparedWrite`, `prepare_write_sync`, `finalize_write_sync`), PR 5 (`ChunkTransform.encode_chunk` used in inner-chunk encoding).

### Risk
High. This is the most complex change. Key concerns:
- Correct offset computation in dense layout (morton order vs C-order indexing)
- `_ShardIndex.offsets_and_lengths` uses chunk-coordinate indexing, NOT morton-rank indexing — this was a source of bugs during development
- `index_location=start` case: must not add offset to `MAX_UINT_64` empty entries
- Fallback path when `set_range` is not supported must produce identical results

### Review checklist
- [ ] `_chunk_byte_offset` computes correct byte positions for morton-ordered dense layout
- [ ] `_build_dense_shard_index` uses `set_chunk_slice(chunk_coords, ...)` not ravel-based indexing
- [ ] `finalize_write_sync` gracefully falls back to full serialize+set when store raises `NotImplementedError` for `set_range`
- [ ] `_prepare_write_partial_fixed_sync` skips reading inner chunks that will be fully overwritten
- [ ] `index_location=start` offset adjustment filters out `MAX_UINT_64` entries
- [ ] Nested sharding test validates data integrity

---

## PR 7: Vectorized shard encoding

**Branch**: No branch yet
**Status**: Not started

### Motivation
For full-shard writes with many small fixed-size inner chunks (e.g. 32^3 = 32,768 chunks), the per-chunk Python loop is the bottleneck. This PR adds `_encode_vectorized` and `_encode_vectorized_sparse` which use numpy reshape/transpose/fancy indexing to reorder the entire shard from C-order to morton order in one shot — no per-chunk Python calls.

This is the change responsible for the 10–22x speedups on the write benchmarks.

### Files
| File | Changes |
|---|---|
| `src/zarr/codecs/sharding.py` | `_encode_sync()` dispatches to `_encode_vectorized` for fixed-size codecs; `_encode_vectorized()` (all chunks present); `_encode_vectorized_sparse()` (some chunks are fill value); `_ShardIndex.get_chunk_slices_vectorized()`, `_ShardReader.to_dict_vectorized()` |

### Dependency
PR 6 (dense layout helpers, `_inner_codecs_fixed_size`, `_inner_chunk_byte_length`).

### Risk
Medium. The vectorized path is a complex numpy operation:
- Reshape to `(cps[0], cs[0], cps[1], cs[1], ...)`, transpose to separate grid axes from chunk-interior axes
- Fancy indexing to reorder from C-order to morton order
- Must handle arbitrary dtypes and endianness
- `_encode_vectorized_sparse`: must correctly identify fill-value chunks and build sparse index

### Review checklist
- [ ] Reshape dimensions match `chunks_per_shard` × `chunk_shape` interleaved
- [ ] Transpose puts grid axes first, chunk-interior axes second
- [ ] Morton reordering uses `_morton_rank_map` correctly (coords→rank mapping)
- [ ] Endianness swap happens at shard level before chunking (avoids per-chunk swap)
- [ ] Sparse variant correctly identifies fill-value chunks via numpy `all_equal` reduction
- [ ] Round-trip: write vectorized → read back → data matches

---

## PR 8: Sync fast path in `Array` (opt-in)

**Branch**: No branch yet
**Status**: Not started

### Motivation
Wires everything together: `Array`'s get/set selection methods check `_can_use_sync_path()` and call the pipeline's `read_sync`/`write_sync` directly, bypassing `sync()` wrappers and the event loop.

### Opt-in strategy
Gate behind `zarr.config["codec_pipeline.sync_fast_path"]` (default: `False` initially, flipped to `True` once stable). When disabled, the existing async-wrapped path is used — zero behavior change.

### Files
| File | Changes |
|---|---|
| `src/zarr/core/array.py` | `_can_use_sync_path()` method; `_get_selection_sync()`, `_set_selection_sync()` methods; all 10 `get_*`/`set_*` methods check sync path first |
| `src/zarr/core/config.py` | Add `codec_pipeline.sync_fast_path` config key (default: `False`) |
| `tests/test_array.py` | Tests that verify sync path produces same results as async path |

### Dependency
PR 5 (`read_sync`/`write_sync` on pipeline, `supports_sync_io`), PR 3 (store sync methods).

### Risk
Low with the config flag. The sync path is only activated when:
1. Config flag is `True`
2. All codecs in the pipeline implement `SupportsSyncCodec`
3. The store has `get_sync` method

If any condition is false, the existing async path is used.

### Review checklist
- [ ] `_can_use_sync_path()` checks config flag, pipeline capability, and store capability
- [ ] All 10 selection methods (get_basic, get_orthogonal, set_basic, etc.) have the sync check
- [ ] When config flag is `False`, behavior is identical to current `main`
- [ ] Performance benchmarks confirm speedup when enabled

---

## Branch status summary

| Branch | PRs covered | Rebased onto chain? | Tests pass? |
|---|---|---|---|
| `perf/sync-codec-protocol` | PR 1 | Yes (root) | Yes |
| `perf/store-sync` | PR 1, PR 2 (partial), PR 3 | Yes | Yes |
| `perf/codec-chain` | PR 1, PR 2, PR 3 | Yes | Yes |
| `perf/prepared-write` | PR 1–4 | No (has merge commit, needs rebase) | — |
| `perf/sharding-chunk-transform` | PR 1–4, PR 6 (partial) | No (has merge commit, needs rebase) | — |
| `perf/chunkrequest` | PR 1–6 (partial) | Yes (rebased onto `perf/codec-chain`) | Yes |

### Milestone branches needing rebase

`perf/prepared-write` and `perf/sharding-chunk-transform` diverged from the old lineage (before `perf/codec-chain` was rebased onto `perf/store-sync`). They still carry a merge commit (`a7164801`). Their content is fully contained within `perf/chunkrequest`, so they serve as reference snapshots only. If they need to be used independently, they should be rebased:

```bash
# To rebase perf/prepared-write onto the chain:
git rebase --onto perf/codec-chain 5a2a884e perf/prepared-write
# (then remove duplicate store sync methods if any)

# To rebase perf/sharding-chunk-transform:
git rebase --onto perf/codec-chain 5a2a884e perf/sharding-chunk-transform
```

---

## Test changes

Some test changes span multiple PRs. Here's the mapping:

| Test file | PR |
|---|---|
| `tests/test_sync_codec_pipeline.py` (new) | PRs 1, 2, 5 (grows with each) |
| `tests/test_codecs/test_sharding.py` (additions) | PR 6 |
| `tests/test_array.py` (modifications) | PR 8 |
| `tests/test_indexing.py` (modifications) | PR 6 (CountingDict.set signature fix) |
| `tests/test_config.py` (additions) | PRs 5, 8 |
| `tests/package_with_entrypoint/__init__.py` | PR 5 (config key update) |

---

## Open questions

1. **Config flag naming**: `zarr.config["codec_pipeline.sync_fast_path"]` vs `zarr.config["threading.sync_fast_path"]` vs something else?

2. **`MemoryStore.set()` breaking change**: The branch removes the undocumented `byte_range` parameter from `MemoryStore.set()`. Should we deprecate first or just remove it? (It was undocumented and not part of the `Store` ABC.)

3. **Thread pool config**: The branch adds `threading.codec_workers`. Should this be exposed as a public config key from the start, or kept internal?

4. **Vectorized encoding scope**: PR 7 is only beneficial for fixed-size inner codecs (no compression). Should we document this limitation explicitly, or is it self-evident from the code?

5. **Removal of `_ShardingByteGetter`/`_ShardingByteSetter`**: These are replaced by the prepare/finalize pattern. Should we deprecate them first (they're internal/private classes)?

6. **`ArrayBytesCodecPartialEncodeMixin` removal**: `ShardingCodec` no longer inherits from this. The mixin may still be used by third-party codecs. Should we keep it available?
