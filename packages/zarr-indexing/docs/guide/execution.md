# Partition rows for a codec pipeline

`zarr_indexing._execution` is a private, opt-in adapter that walks a
[`GridPartition`](../api/grid.md) in the shape Zarr's codec pipeline consumes.
It does not replace Zarr's default indexers or change the transform algebra.
Start with [From a selection to chunk operations](selection-flow.md) for
coordinate spaces and partitioning.

## From a selection to rows

`execute_selection(selection, shape, grids, mode=...)` reads a NumPy-dialect
selection the way `LazyArray` does: positions count from the end when negative,
and in the orthogonal and vectorized modes scalar integers are applied first so
their axes drop, as NumPy and Zarr's indexers do. It builds the transform and
hands it to `execute_transform(transform, grids)`, which factors it with
`plan_chunks(...).partition()` and wraps the partition as an `ExecutionPlan`.

An `ExecutionPlan` has Zarr's `Indexer` surface: `shape`, an empty `drop_axes`,
and iteration yielding one row per touched chunk, in the partition's row-major
order, with `chunk_coords`, `chunk_selection`, `out_selection` and
`is_complete_chunk`. For a read:

```text
result[out_selection] = decoded_chunk[chunk_selection]
```

and a write scatters `value[out_selection]` into `chunk[chunk_selection]`. The
same plan serves both; nothing about it depends on the direction.

## Two row layouts

Both sides of that assignment must produce values of one shape. Rows read
straight off the partition tables in one of two layouts:

- When every table is a `StridedSet` and the request axes appear in storage
  order, chunk selectors are integers and ascending slices, and out selectors
  are slices. A reversed request reads its chunks ascending and puts the
  reversal in the out slice, which is what lets a fully covered chunk stay
  eligible for the pipeline's whole-chunk write shortcut. `is_complete_chunk`
  is true when every table row covers its chunk's data extent exactly once.
- Otherwise every selector is an integer array shaped along one slot of a
  common layout: one slot per request axis, with the broadcast axes of a
  connected index-array component sharing the slot of its first. All selectors
  are advanced, so NumPy's placement rules for mixed indexing never apply.
  These rows are never marked complete.

Index arrays in the selection are snapshotted when the transform is built, so a
plan cannot change under its caller. Rows are materialized per table, so
planning is linear in the touched chunks per axis, like the partition itself.

## Verification

`tests/test_execution_properties.py` assembles the rows of arbitrary selections
over arbitrary grids out of a chunked NumPy reference and checks the result
against NumPy, that a plan walks the same rows twice, that `is_complete_chunk`
proves whole-chunk coverage, and that scattering through the rows writes
exactly the selected cells. `tests/test_indexing_execution.py` drives both
codec pipelines with plans in place of Zarr's indexers, over v2, v3 and sharded
layouts, and counts storage reads to show that complete chunk writes, forward
and reversed, skip the read-modify-write. It skips only when Zarr itself is
unavailable; the package runtime has no dependency on Zarr.
