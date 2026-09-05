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

The follow-up implements connected index-array components in production and
preserves singleton axes during vectorized compilation. The API now exposes
`joint_sets`; `row_shape` lists `sets` followed by all component tables.
This is an intentional pre-1.0 API change, with no singular `joint` accessor.

Compared with the previous commit (`59c0041c4`), again using nine sequential
runs and the same interpreter:

| Operation | Previous | Components | Peak before → after |
| --- | ---: | ---: | ---: |
| Partition `(u,v) -> (a[u],b[u],c[v])`, 1,000 positions per axis | 19.38 ms | 0.070 ms | 122.07 → 0.081 MiB |
| Complete projection walk, same request | 24.33 ms | 0.094 ms | 122.07 → 0.098 MiB |
| Compile that request through `vindex` | 1.62 ms | 0.022 ms | 22.91 → 0.049 MiB |
| Walk 1,000 ordinary correlated points | 7.54 ms | 8.47 ms | 0.141 → 0.142 MiB |

The production partition result is close to the independent-component probe
(0.056 ms / 0.072 MiB), while also constructing the public partition and usable
paired projections. These gains apply when dependencies are independent;
a genuinely connected multidimensional index block still requires storing
its selected points. The initial generalized projection assembler added about 12% in
this small single-component walk benchmark; the follow-up below removes that
regression. Basic/orthogonal paths retain
their existing implementations. End-to-end NumPy/basic-reader scatter tests
verify values and placement; the timing table does not measure storage I/O.

### Single-component follow-up

A direct iterator now handles requests whose input axes and output dimensions
all belong to a single index-array component. It avoids per-chunk product
bookkeeping and combines chunk-coordinate and map construction in one loop.
Residual slices, unread axes, constants, and multiple components continue to
use the general assembler.

Three sequential comparison rounds, each with 31 timing repetitions per
operation, compared `59c0041c4` (before components), `9e8a2986c` (components),
and this follow-up. Values below are the median of the three run medians;
traced peak memory is measured separately. No test jobs ran during these rounds.

| Operation | Before components | Components | Direct iterator |
| --- | ---: | ---: | ---: |
| Ordinary correlated walk | 7.71 ms | 8.54 ms | 7.25 ms |
| Correlated slab walk | 0.071 ms | 0.074 ms | 0.072 ms |
| Independent-component partition | 24.35 ms | 0.067 ms | 0.065 ms |
| Independent-component walk | 29.17 ms | 0.091 ms | 0.091 ms |
| Independent-component `vindex` compilation | 1.968 ms | 0.021 ms | 0.021 ms |

The ordinary walk's three medians were 7.18–7.26 ms, versus 7.67–7.86 ms
before components and 8.48–8.56 ms with components. The direct iterator removes
the observed slowdown and is about 6% faster than the original baseline in
this case. Its traced peak remains 0.142 MiB; independent-component partition
and walk peaks remain 0.081 and 0.098 MiB. The slab uses the unchanged general
assembler; its small timing differences do not establish a speedup.

Remaining experiments:

1. **Keep regular strided sets implicit until columns are requested.**
   TensorStore's strided set stores the input dimension and dependent grid
   dimensions, not one record per touched chunk. A range/batch interface could
   reduce first-result latency and memory for very long axes while retaining
   materialized columns for vectorized consumers. The present batch API bounds
   Cartesian expansion, but per-axis tables are still eager.

2. **Benchmark a real codec consumer before changing Zarr's indexers.**
   Zarr's `BasicIndexer` and `OrthogonalIndexer` already form products of
   per-axis projections; its coordinate indexer already groups points by chunk.
   The gain from this package requires consuming its tables, not adding a pair
   of transform objects to every existing codec call. The integration benchmark
   should compare selector creation, decoded-buffer copies, and store calls for
   ordinary and sharded arrays, including boundary chunks and partial writes.
   Preserve clipped data extents separately from codec buffer extents, and
   skip reads only for proven complete writes. This branch does not change
   Zarr's codec or shard execution paths.

## Immediate selector execution prototype

`_execution.py` is an internal, opt-in prototype. `execute_selection` takes
literal-coordinate selections, a storage shape, and dimension grids;
`execute_transform` lowers an existing immutable transform. Both return a
reusable `ExecutionPlan` with a zero-origin output shape and rows containing
chunk coordinates, chunk selectors, output selectors, and a complete-codec-buffer
flag. Zarr's default indexers and the existing declarative API are unchanged.

Basic selections use the existing normalization and slice-resolution helpers,
then generate selectors directly. Dense sorted native-`intp` coordinates use a
monotonicity check and `searchsorted` on internal chunk boundaries. Neither path
constructs transform pairs for each chunk. Other selections currently lower the
existing projections to broadcast NumPy coordinates; their additional conversion
cost is included below. Immediate setup for this fallback is lazy, so compare
complete walks rather than constructor times alone.

The immediate sorted path borrows its array: it must remain unchanged for every
use of the plan and any outstanding iterator. Immutable declarative transforms
retain their snapshot semantics. The prototype shares transform semantics,
including literal bounds; it is not a replacement for Zarr's user-facing NumPy
normalization. Unread request axes longer than one are explicitly rejected until
repeated-write ordering has a defined policy.

### Measurements against the existing Zarr indexers

Run from the root Hatch environment with this checkout's `src` and
`packages/zarr-indexing/src` on `PYTHONPATH` (and package metadata installed):

```sh
hatch run test.py3.12-minimal:python packages/zarr-indexing/benchmarks/execution.py
```

The benchmark checks chunk-coordinate equality and declarative selected-cell
counts. It runs three rounds of 31 repetitions, reversing operation order in the
middle round. Each row below reports the median of the three timing medians.
Input selections and grids are prepared before measurement; selection compilation,
planning, and streaming consumption are timed together. Peak allocations are
measured separately. These measurements exclude data reads, writes, codec work,
and the optional sharding adapter.

Measured on 2026-09-05, Python 3.12 / NumPy 2.0, against the existing Zarr indexers
in the PR checkout (`src/zarr/core/indexing.py` is unchanged by this branch):

| Complete walk | Zarr | Declarative projections | Immediate selectors |
| --- | ---: | ---: | ---: |
| Basic slice, 10,000 chunks | 7.991 ms | 21.794 ms | 7.415 ms |
| Sorted orthogonal selection, 1M points / 10 chunks | 5.019 ms | 5.008 ms | 0.332 ms |
| Sorted coordinate selection, 1M points / 10 chunks | 0.350 ms | 4.759 ms | 0.331 ms |
| Correlated points, 100 chunks | 0.269 ms | 0.909 ms | 1.282 ms |
| Sparse correlated points, 900 chunks | 5.366 ms | 7.326 ms | 10.671 ms |
| Independent components, one chunk / 1M points | 12.181 ms | 0.128 ms | 0.141 ms |
| Independent components, 100 chunks / 1M points | 27.879 ms | 1.474 ms | 2.062 ms |

| Peak allocation during complete walk | Zarr | Declarative projections | Immediate selectors |
| --- | ---: | ---: | ---: |
| Basic slice | 0.049 MiB | 0.048 MiB | 0.034 MiB |
| Sorted coordinate selection | 1.529 MiB | 30.529 MiB | 1.528 MiB |
| Independent components, 100 chunks | 83.950 MiB | 0.107 MiB | 0.107 MiB |

Basic walking now meets the baseline, and sorted coordinate walking and allocation
are close to Zarr's specialized path. Basic constructor time is still higher:
6.1 µs versus 2.5 µs, with both near 0.9 KiB peak allocation. The sorted constructor
is 0.121 ms versus Zarr's 0.130 ms. Constructor and first-result latency deserve
separate attention for tiny requests. Basic multidimensional iteration still
uses `itertools.product`, which retains per-axis selector pieces; it is not yet
a fully implicit Cartesian walk.

The generic fallback is slower than the existing projection walk because it
also materializes execution selectors. In particular, ordinary correlated
selections remain substantially slower than Zarr. This prototype should not be
installed as a universal replacement. Direct lowering of component-table rows
is the next step for removing that overhead while keeping the independent-
component gains.

### Codec verification and the sharding boundary

`tests/test_indexing_execution.py` exercises real reads and writes through both
BatchedCodecPipeline and FusedCodecPipeline, for v2, v3, and sharded v3 arrays.
It checks basic slices, integer-axis drops, reverse slices, sorted coordinates,
and independent components, including partial boundary writes and preservation
of untouched values. Unit tests also cover irregular grids, duplicates, empty
selections, singleton newaxes, bounds errors, and immutable snapshots.

The current sharding codec's internal indexer rejects negative slices and returns
flat coordinate results. `for_shard_indexer` adapts those selections per shard;
positive basic selectors pass through. This can materialize the selected points
within one shard, so the compact planning gains above must not be read as sharded
I/O speedups. A future sharding consumer should accept the compact execution
representation directly. The prototype only marks complete writes when direct
basic selectors cover the declared codec buffer, not merely the clipped data
extent of a boundary chunk.
