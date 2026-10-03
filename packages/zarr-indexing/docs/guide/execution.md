# Partition rows for a codec pipeline

`zarr_indexing._indexer` is a private, opt-in adapter: a
[`GridPartition`](../api/chunk_resolution.md) speaking Zarr's `Indexer`
protocol. It does not replace Zarr's default indexers or change the transform
algebra. Start with [From a selection to chunk operations](selection-flow.md)
for coordinate spaces and partitioning.

## From a selection to rows

`PartitionIndexer.from_selection(selection, shape, grids, mode=...)` reads a
NumPy-dialect selection exactly as `LazyArray` does, through the shared
`select_positional`: positions count from the end when negative, and in the
orthogonal and vectorized modes scalar integers are applied first so their axes
drop, as NumPy and Zarr's indexers do. It builds the transform and hands it to
`PartitionIndexer.from_transform(transform, grids)`, which factors it with
`plan_chunks(...).partition()`.

A `PartitionIndexer` has Zarr's `Indexer` surface: `shape`, an empty
`drop_axes`, and iteration yielding one row per touched chunk, in the
partition's row-major order, with `chunk_coords`, `chunk_selection`,
`out_selection` and `is_complete_chunk`. For a read:

```text
result[out_selection] = decoded_chunk[chunk_selection]
```

and a write scatters `value[out_selection]` into `chunk[chunk_selection]`.
Index arrays are snapshotted by `IndexTransform` when the transform is built,
so an indexer cannot change under its caller. Rows are read straight off the
partition tables, so building them is linear in the touched chunks per axis,
like the partition itself. A request axis that no storage dimension reads must
have extent one (a newaxis): a wider one would scatter one chunk cell to several
positions, and the indexer refuses it.

## Two row layouts

Both sides of the assignment above must produce values of one shape. Rows
come in one of two layouts, chosen per partition, so that this holds by
construction and never through NumPy's placement rules for mixed basic and
advanced indices:

- **Slices**, when every table is a `StridedSet` with a nonzero stride and the
  request axes appear in storage order. Chunk selectors are integers and
  ascending slices, out selectors are slices. A reversed request reads its
  chunks ascending and puts the reversal in the out slice. `is_complete_chunk`
  is true only when the chunk selectors walk the chunk's data extent in order,
  because Zarr's whole-chunk write shortcut takes `value[out_selection]` as the
  chunk without applying `chunk_selection`; the ascending emission is what keeps
  a fully covered reversed chunk eligible for it.
- **Coordinates**, otherwise. Every selector is an integer array shaped along
  one slot of the partition's synthetic layout (one slot per connected
  index-array component, then one per remaining request axis). All selectors
  are advanced, so both sides take the broadcast shape. These rows are never
  marked complete.

## Verification

`tests/test_execution_properties.py` assembles the rows of arbitrary selections
over arbitrary grids out of a chunked NumPy reference and checks the result
against NumPy, that an indexer walks the same rows twice, that a complete row's
chunk selectors are the identity walk of the chunk, and that scattering through
the rows writes exactly the selected cells. `tests/test_indexing_execution.py`
drives both codec pipelines with a `PartitionIndexer` in place of Zarr's
indexers, over v2, v3 and sharded layouts, and counts storage reads to show
that complete chunk writes, forward and reversed, skip the read-modify-write.
It skips only when Zarr itself is unavailable; the package runtime has no
dependency on Zarr.
