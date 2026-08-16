# Zarrista-native region I/O

**Status:** Approved for implementation planning

**Date:** 2026-08-16

**Base:** `origin/zarrs-bindings` at `3ff999dab`

**Working branch:** `codex/coordinate-aware-codec-io-zarrista`

## Context

The zarrista binding branch already places coordinate-aware I/O at the array engine
boundary. `Region` describes an ndim-preserving, step-one rectangular selection, and the
zarrista engine reads it with `retrieve_array_subset`.

Writes currently stop short of that design. The pinned zarrista revision does not expose
`store_array_subset`, so `ZarristaEngine` decomposes every region into chunks in Python,
retrieves partially covered chunks, patches NumPy arrays, converts them to
`zarrista.ArrayBytes`, and stores chunks individually. The synchronous and asynchronous
engines duplicate this algorithm.

Current zarrista main exposes both synchronous and asynchronous
`store_array_subset(selection, data, **codec_options)`. It performs partial-chunk
read-modify-write and codec-aware work in Rust through zarrs. This is the API this branch
should target.

## Decision

Use zarrista's array-level selection API as the coordinate-aware codec boundary:

- retain `Region` as the zarr-python engine selection type;
- continue mapping reads to `retrieve_array_subset`;
- map writes directly to `store_array_subset`;
- remove the Python chunk-overlap and read-modify-write implementation; and
- leave the existing Python codec interfaces unchanged in this increment.

This deliberately does not introduce CuTe expressions into codec signatures. For the
contiguous, ndim-preserving selections supported by `Region`, zarrista already owns the
selection-to-chunk, shard, and codec translation. Recreating that translation as a
zarr-python codec protocol would add a second abstraction that a future zarrista-backed
implementation could not consume directly.

## Goals and success criteria

The change is successful when it delivers all three outcomes below.

### New functionality

- Region writes use zarrista's native array-subset operation for both regular and sharded
  arrays.
- A selection may span multiple chunks, inner shard chunks, and edge chunks without a
  Python implementation of those storage details.

### Simpler code

- Synchronous and asynchronous writes each contain one selection conversion and one
  zarrista API call.
- The Python chunk-overlap helper, partial-chunk patching loops, and explicit
  `ArrayBytes` construction are deleted.
- No new public selection or codec abstraction is introduced.

### Faster code

- A multi-chunk write crosses the Python/Rust boundary once instead of once or more per
  touched chunk.
- Chunk and shard concurrency is controlled inside zarrista/zarrs rather than serialized
  by Python loops.
- Benchmarks show no material regression for a single-chunk write and improvement for at
  least the multi-chunk or sharded partial-write workloads.

## Detailed design

### Dependency target

Update the exact zarrista Git revision from
`95e47ad4c414c5920f0cf15550f923039641da8e` to
`92d26b65b90e9715d5c658c71b9216449f25ae64`, the inspected zarrista main revision on
2026-08-16, and regenerate the lock file.

The dependency remains pinned to a commit because these bindings are pre-release and the
adapter depends on a specific Python API. Any compatibility edits required by changes
between those revisions are in scope, but only where needed by the existing engine.

In particular, the current API exposes array storage as `storage`, replacing the older
`store` attribute used by `with_metadata`. The metadata-rebinding path must use the new
attribute and retain the existing storage object.

### Selection mapping

`Region` remains the canonical engine representation. Each pair of `start` and
`end_exclusive` coordinates maps to a Python slice with unit step:

```python
tuple(slice(start, stop) for start, stop in zip(region.start, region.end_exclusive))
```

This mapping exactly represents the current engine contract and the selection subset
accepted by zarrista. Integer indexing, stepped slices, dimension insertion, and fancy
indexing remain the responsibility of higher array-indexing layers.

### Reads

Read behavior is unchanged conceptually:

```python
decoded = array.retrieve_array_subset(selection)
```

and, for the asynchronous engine:

```python
decoded = await array.retrieve_array_subset(selection)
```

The existing decoded-tensor-to-`NDBuffer` conversion remains in place. This increment is
not a redesign of returned buffer ownership or device support.

### Writes

The synchronous write becomes equivalent to:

```python
data = np.asarray(value.as_ndarray_like(), order="C")
array.store_array_subset(selection, data)
```

The asynchronous write performs the same input normalization and awaits the corresponding
zarrista call.

Using `np.asarray(..., order="C")` preserves 0-D inputs while ensuring non-scalar inputs
are C-contiguous. (`np.ascontiguousarray` promotes a 0-D input to shape `(1,)`, which does
not match a scalar destination.) Current zarrista accepts a `DataInput` backed by DLPack,
`ArrayBytes`, or a buffer, but broadening zarr-python's device or zero-copy behavior is not
part of this change. A copy may still occur during input normalization.

Zarrista validates the selection, shape, dtype, and contiguity at its boundary. Its errors
continue to propagate through the engine in the same manner as other zarrista operations;
zarr-python does not duplicate those checks per chunk.

### Deleted implementation

Remove code that becomes redundant, including:

- the `_chunks_overlapping` helper;
- imports used only by chunk enumeration;
- synchronous and asynchronous per-chunk read-modify-write loops;
- explicit construction of `zarrista.ArrayBytes` for each chunk; and
- local calculations of chunk bounds, intersections, and source/destination slices.

No compatibility shim should retain the old write algorithm. The pinned dependency makes
`store_array_subset` part of the engine's required backend contract.

### Codec options

Zarrista accepts options such as checksum validation, empty-chunk storage, concurrency
targets, and experimental partial encoding. This increment uses zarrista defaults, as the
current engine API has no codec-options parameter. Exposing options through the engine can
be evaluated separately without coupling it to the removal of the Python write loop.

## Verification

### Differential correctness tests

Tests compare zarrista-engine behavior with the existing zarr-python engine and/or a
fully materialized expected NumPy array. At minimum they cover:

- a partial write contained in one regular chunk;
- a write spanning multiple regular chunks;
- a partial write to a sharded array that crosses inner-chunk boundaries;
- a sharded write that crosses shard boundaries;
- writes intersecting edge chunks where stored chunk shapes differ from the nominal
  shape;
- preservation of values immediately outside the selected region;
- synchronous and asynchronous engine paths; and
- metadata rebinding through `with_metadata` while retaining storage.

The normal-case combinations should be exercised together where practical, with separate
tests for distinct error conditions in accordance with repository testing guidance.

Existing zarrista engine, array indexing, codec, and v2/v3 compatibility tests must remain
green. Unsupported variable-length and masked decoded values remain unsupported unless the
dependency update requires a mechanical compatibility adjustment.

### Performance benchmarks

Record a baseline on the old pinned revision before replacing it, then rerun the same
workloads on the new implementation. Benchmark:

- a full single-chunk write;
- a partial single-chunk write;
- a partial write spanning many regular chunks; and
- a partial write spanning inner chunks and shards.

Use identical array shapes, selections, stores, codecs, warmup, and repetition counts.
Report median wall-clock time and dispersion. Where the test harness permits, also record
the count of Python-to-zarrista write calls and store requests. The structural expectation
is one `store_array_subset` call per engine write; store request counts may still vary with
partial updates and codec capabilities.

A single-chunk result within normal benchmark noise is acceptable. Multi-chunk or sharded
partial writes should improve materially; otherwise profile the Rust call and input-copy
cost before claiming a performance win.

## Compatibility and rollout

`Region`, the engine methods, and user-facing array indexing remain source-compatible. The
change affects only the optional zarrista engine and its pinned dependency. The Python
engine and public codec pipeline are unchanged.

Because the zarrista revision changes several binding names, run import and construction
smoke tests before the broader suite. If an unavoidable upstream incompatibility is found,
adapt the private engine wrapper rather than expanding zarr-python's public API.

The implementation can be reverted by restoring the old pin and write implementation;
there is no metadata migration or on-disk format change.

## Non-goals

- Changing array-to-array, array-to-bytes, or bytes-to-bytes codec signatures.
- Adding a CuTe layout-expression implementation.
- Expanding `Region` to arbitrary or stepped selections.
- Changing the default zarr-python engine or codec pipeline.
- Adding GPU/device-buffer support or promising zero-copy writes.
- Exposing every zarrista `CodecOptions` field through public APIs.

## Follow-up decisions

After correctness and benchmark data are available, separately evaluate:

- whether the zarrista engine should become more prominent or the default for supported
  configurations;
- whether codec options need an engine-level configuration surface;
- whether direct DLPack or buffer inputs can avoid the NumPy normalization copy; and
- whether any remaining Python sharding special cases can move behind the same array-level
  backend boundary.

Those decisions should be based on observed workloads and zarrista API stability, not on a
parallel codec-selection algebra introduced in advance.
