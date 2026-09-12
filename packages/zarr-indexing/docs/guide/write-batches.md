# Conflict-free write batches

Data often arrives in chunks that do not align with the destination chunks.
Disjoint array slices can still race: both writes may read, modify, and replace
the same destination chunk. `plan_rechunk` keeps each incoming source chunk as
one task and groups tasks into batches with disjoint destination write units.
It plans the work; your code reads buffers, submits tasks, and writes data.

## Source chunks stay intact

A length-12 source arriving in chunks of 3 and targeting chunks of 4 produces:

| Source task | Global values | Destination units | Batch |
| --- | --- | --- | --- |
| 0 | `[0:3]` | `{0}` | 0 |
| 1 | `[3:6]` | `{0, 1}` | 1 |
| 2 | `[6:9]` | `{1, 2}` | 0 |
| 3 | `[9:12]` | `{2}` | 1 |

Each source chunk is one task even when it touches several destination units.
Tasks 0 and 2 run concurrently, then tasks 1 and 3. Reordering is valid because
their logical destination elements are disjoint. Chunk boundaries still require
read-modify-write, so all four tasks cannot safely run at once.

```text
source chunks + global domain
       -> one piece per intersecting source chunk
       -> destination transform per piece
       -> touched destination write-unit coordinates
       -> batches with no shared write unit
       -> caller executes each batch, then waits for completion
```

A `RechunkPiece` has a source chunk coordinate, slices relative to that decoded
source chunk, and a destination transform. The transform is an identity over
the piece's global domain; its bounds give the destination slices. A subdomain
clips source selections without changing chunk identity. Irregular grids,
nonzero origins, empty domains, and scalar arrays are supported.

The example uses already-arrived NumPy chunks and a NumPy destination to expose
the task boundary. Replace `target[selection] = values` with the destination's
write operation. Real Zarr chunked and sharded writes are covered by the package's
codec integration tests.

```python
--8<-- "snippets/write_batches.py:copy"
```

## What makes a batch safe

Supply the grid of **independently writable storage units**. For sharded Zarr,
that is generally the shard grid rather than the inner codec-chunk grid. When
metadata or an enclosing object couples otherwise separate chunks, use the
coarser transaction unit, or provide external coordination.

Complete all destination reads and writes in batch *n* before starting any
destination read-modify-write for batch *n + 1*. Merely submitting futures is
not a barrier. Prefetching destination snapshots across batches can lose updates
even with a correct schedule. Independent source data may be prefetched.

The source and destination must not alias during a copy. The schedule knows
write conflicts, not read-after-write dependencies between source and target.
It does not coordinate unrelated writers, retries still running after a failed
batch, metadata changes, or mutations to the supplied grids. Each task must also
perform its own writes safely; the schedule coordinates between tasks.

## Scheduling existing transformed tasks

`plan_write_batches(writes, destination_grid)` accepts a finite iterable of
request-to-destination `IndexTransform`s. Task IDs are their zero-based input
positions. Keep your buffers or futures separately and use these IDs to find
them. Every task appears once, including an empty write; an empty iterable
produces no batches. Duplicate coordinates within one task occupy a unit once
for scheduling, but assignment semantics within that task remain your concern.

The default `order="preserve"` retains input order between tasks that touch any
of the same destination units. Unrelated tasks can move earlier; this is not a
guarantee about arbitrary side effects. `order="reorder"` explicitly permits
reordering and uses deterministic first-fit coloring. For overlapping logical
writes, that can change the final values. `plan_rechunk` uses reorder because
its identity-copy pieces are logically disjoint.

Footprints come from the existing chunk planner, preserving its supported
transform semantics. Affine diagonals and mixed affine/index-array maps sharing
an input axis are unsupported by the factored planner and raise. Nonempty out-of-bounds transforms
raise before a schedule is returned. See the [scheduling API](../api/scheduling.md).

## Costs and limits

Preparation is eager. The returned schedule contains immutable tuples of task
IDs; a rechunk plan additionally retains piece metadata. It holds no array data.
It does not enforce a byte-memory budget, split oversized source chunks, choose
temporary storage, or guarantee the fewest batches.

Preserve mode tracks the last batch using each destination unit. Reorder mode
tracks occupied colors and the first free color per unit; this avoids repeatedly
scanning a hot unit's full history. It still uses a heuristic with no linear
worst-case runtime claim. No pairwise conflict graph is built. Planning memory
includes the current footprint, chunk-planner intermediates, task IDs, and
per-unit state. Reorder state scales with task/unit memberships; large footprints
and many source tasks can still consume substantial memory.

`n_write_units` counts distinct destination units and `n_memberships` sums the
units touched per task. They are structural diagnostics, not exact byte-I/O or
peak-memory predictions. To limit execution memory, independently cap concurrent
tasks inside a batch. Splitting a batch into smaller sequential sub-batches
preserves safety; combining batches does not necessarily do so.

A row-to-column transfer exposes the limitation: every row task touches every
column write unit, so rows must run serially. No recoloring can create parallelism
while preserving those tasks. Destination-owned assembly or an intermediate
layout changes the task boundaries and is often the better strategy.

## Comparison with existing planners

These systems solve related but different problems:

| Planner | Work it chooses | Main constraint | Execution consequence |
| --- | --- | --- | --- |
| This utility | Batches of existing source tasks | Disjoint destination write units per batch | Repeated destination RMW and serialization are possible |
| Rechunker | Consolidated read/write regions, optionally via an intermediate layout | Configured worker memory and reduced transfer overhead | Can change task boundaries and use temporary storage |
| Dask task rechunking | Split/merge transfer stages between chunk layouts | Graph growth and block-size limits | Constructs a transfer graph rather than storage-write batches |

[Rechunker's algorithm](https://rechunker.readthedocs.io/en/stable/algorithm.html)
selects consolidated read and write chunks and uses an intermediate array when
needed. Its [planner implementation](https://github.com/pangeo-data/rechunker/blob/v0.5.4/rechunker/algorithm.py)
also includes multistage planning. It is a better fit when transfer layout and
temporary storage can be chosen under a memory budget.

[Dask's task planner](https://docs.dask.org/en/stable/_modules/dask/array/rechunk.html)
chooses intermediate layouts using graph-size and block-size controls. Dask also
has a peer-to-peer rechunk path; the comparison script exercises only its task
planner, not that distributed execution path.

[Xarray's `to_zarr`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_zarr.html)
checks chunk alignment for parallel safety and provides alignment options.
This scheduler offers a different tradeoff for fixed incoming tasks: keep their
boundaries and serialize conflicts. It does not disable or replace Xarray's
checks automatically.

`benchmarks/write_scheduling.py` runs each planner separately, recording native
batch/task/membership counts, Dask's transfer layouts, or Rechunker's named
read/intermediate/write block shapes. Those three shapes do not imply three
execution stages. External
planners can run in their own environments, avoiding dependency conflicts.
Planning times describe different returned products and are not interchangeable
performance scores or end-to-end copy measurements.
