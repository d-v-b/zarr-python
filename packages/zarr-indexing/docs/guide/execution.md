# Experimental execution consumers

This follow-up implements a private, opt-in execution experiment in
`zarr_indexing._execution`. It does not replace Zarr's default indexers or change
the public transform algebra. Start with [From a selection to chunk
operations](selection-flow.md) for coordinate spaces, partitioning, and the paired
source/result correspondence.

## Preparation and lowering

`execute_selection(selection, shape, grids, mode=...)` interprets literal
coordinates and prepares an `ExecutionPlan`. Basic, orthogonal, and vectorized
modes choose the selection mapping. Orthogonal scalar integers are applied as a
basic selection first, dropping their result axes before the remaining orthogonal
selection. This differs intentionally from the low-level transform `oindex`,
which retains scalar axes. The resulting selectors already have the reduced
rank; they do not need the legacy indexer's later `drop_axes` squeeze.
Negative scalar coordinates still raise: this frontend does not normalize
NumPy-style negative indices. Vectorized mixed-scalar semantics remain those of
the transform algebra; this is not a universal Zarr Indexer replacement.

`execute_transform(transform, grids)` accepts an existing immutable mapping.
Both entry points share affine-axis planning with declarative tables. Large
basic axes remain implicit; small axes cache at most 128 pieces. Sorted arrays
use a structural regular-grid capability, and connected-component tables lower
directly where supported. Other transforms use declarative projections.

Iterating a plan lowers it for the NumPy consumer. Each row has `chunk_coords`,
`chunk_selection`, `out_selection`, and `is_complete_chunk`. For a read:

```text
result[out_selection] = decoded_chunk[chunk_selection]
```

`plan.lower("numpy").operations()` also provides `value_shape` and
`selector_kind`, distinguishing basic selectors from paired broadcast coordinate
arrays. `plan.lower("shard")` adapts to the current shard indexer. That adaptation
can expand compact selections into flat coordinates; passing compact plans
through the shard boundary remains future work.

## Ownership and writes

`ownership="snapshot"` is the default. `ownership="borrow"` permits borrowing
caller arrays, which must remain unchanged throughout every use of the plan and
its iterators. It does not guarantee zero copies on every planning path.

Writers prepare with `access="write"`. By default, repeated destinations raise;
`conflicts="last"` explicitly retains the last value in row-major request order
before dispatch. A write plan that discards overwritten values is not a read
plan. Reads preserve repeated positions.

Completeness refers to the **valid data extent**, matching declarative coverage
and Zarr's existing indexers. A full boundary write can skip its read: Zarr's
merge helper fills the rest of a declared codec buffer when the selected slab
is smaller. A singleton selected once is full regardless of stride magnitude.
The execution adapter emits a true flag only for layouts the current codec's
merge shortcut can safely consume; reverse and general coordinate layouts stay
conservative. Coverage alone does not prove the values are in codec-buffer order.

Preparation validates supported bounds and write policies, but does not provide
transactional I/O or protect against mutation of borrowed arrays.

## Verification boundary

`tests/test_indexing_execution.py` lives in the package suite, which CI runs in
the repo-root environment with Zarr installed. It skips only when Zarr itself is
unavailable. Both codec pipelines run against v2, v3, and sharded layouts,
including orthogonal scalar removal. Boundary-write tests count storage reads,
so marking a full boundary cell partial cannot silently add read-modify-write.
The package runtime still has no dependency on Zarr.
