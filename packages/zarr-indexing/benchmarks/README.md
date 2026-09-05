# Chunk planning performance investigation

Measured on 2026-09-05 using Python 3.12 and the repository's
`test.py3.12-minimal` Hatch environment. The baseline is PR #4310 at
`5bb5278`; the comparison includes the fixes in this branch. Timings are
medians of nine runs, with baseline and changed versions run sequentially.
Peak allocations are measured separately with `tracemalloc`. They exclude
source data and transform construction. These are planning measurements,
not storage throughput measurements; small timing differences are noise.

## Reproduce

From the repository root, with `zarr-indexing` installed in the environment:

```sh
hatch run test.py3.12-minimal:python packages/zarr-indexing/benchmarks/chunk_planning.py --repeats 9
```

To compare checkouts, run the same script and interpreter with each checkout's
`packages/zarr-indexing/src` on `PYTHONPATH`. Both require installed package
metadata for the version lookup in `zarr_indexing.__init__`.

## Changes retained

| Operation | PR baseline | Changed | Peak allocation before → after |
| --- | ---: | ---: | ---: |
| Walk 10,000 basic chunks | 21.41 ms | 21.43 ms | 0.047 → 0.047 MiB |
| Walk 1M sorted points in 10 chunks | 3.91 ms | 3.80 ms | 22.90 → 22.90 MiB |
| Walk 1,000 correlated points | 9.06 ms | 7.57 ms | 0.141 → 0.141 MiB |
| Plan 1,000 correlated points × a 1,000-element residual slice | 1.08 ms | 0.069 ms | 15.32 → 0.073 MiB |
| Access `.local[run]` for 1,000 table rows | 12.31 ms | 0.221 ms | 0.153 → 0.00022 MiB |
| Enumerate 1M chunk coordinates | 8.33 ms (one array) | 6.02 ms (1,024-row batches) | 53.41 → 0.080 MiB |

Local-column timing measures repeated access to an already-built table; the
changed version retains the computed column after its first access. The memory
saving on subsequent calls trades against retaining one coordinate column for
the lifetime of the table. It does not make arbitrary gathers constant-memory.

Correlated projections now keep residual slices as affine maps and broadcast
singletons as singletons. The scatter adapter preserves the synthetic domain's
axis order with compact coordinate selectors when needed. This removes a copy
per requested cell that was unnecessary for describing placement.

The coordinate-batch API does not build projections or the whole Cartesian
product. The existing all-coordinates API retains its vectorized fast path;
batched traversal also supports a table product larger than `np.intp`.

Affine diagonal iteration is compatible again. Chunk discovery walks each
shared input axis to its next grid boundary; a prefix tree merges connected
groups in lexicographic output-coordinate order. Its candidate storage scales
with the sum of the connected-group rows rather than their Cartesian product.
The public per-output-axis `partition()` representation still rejects diagonals.

## TensorStore model and next experiments

TensorStore partitions the bipartite graph of input dimensions and grid
dimensions into connected sets. Its affine sets remain implicit and its index
array sets retain compact input-coordinate arrays per component. See
[grid_partition.h](https://github.com/google/tensorstore/blob/master/tensorstore/internal/grid_partition.h)
and
[grid_partition_impl.h](https://github.com/google/tensorstore/blob/master/tensorstore/internal/grid_partition_impl.h).

1. **Multiple independent index-array components are the largest remaining opportunity.**
   For `(u, v) -> (a[u], b[u], c[v])` with 1,000 values on each input axis,
   the source index arrays occupy 24 KB. The current single joint table expands
   to 1M points: about 25 ms and 122 MiB peak. A probe that partitions the two
   independent components separately uses about 0.053 ms and 0.072 MiB.
   The probe deliberately does not assemble their projections, so these numbers
   demonstrate the avoidable planning expansion, not a complete new planner.
   Production adoption needs a component-based row representation, agreement on
   the public `joint` accessor, and scatter/oracle tests for products of components.

2. **Keep regular strided sets implicit until columns are requested.**
   TensorStore's strided set stores the input dimension and dependent grid
   dimensions, not one record per touched chunk. A range/batch interface could
   reduce first-result latency and memory for very long axes while retaining
   materialized columns for vectorized consumers. The present batch API bounds
   Cartesian expansion, but per-axis tables are still eager.

3. **Benchmark a real codec consumer before changing Zarr's indexers.**
   Zarr's `BasicIndexer` and `OrthogonalIndexer` already form products of
   per-axis projections; its coordinate indexer already groups points by chunk.
   The gain from this package requires consuming its tables, not adding a pair
   of transform objects to every existing codec call. The integration benchmark
   should compare selector creation, decoded-buffer copies, and store calls for
   ordinary and sharded arrays, including boundary chunks and partial writes.
   Preserve clipped data extents separately from codec buffer extents, and
   skip reads only for proven complete writes. This branch does not change
   Zarr's codec or shard execution paths.
