# From a selection to chunk operations

A selection describes values in an array. A chunk operation identifies a chunk,
which values to select inside its decoded buffer, and where those values belong
in the result. This page follows that conversion through the current planner and
explains where the proposed sharding integration would fit.

The public declarative API is `plan_chunks(transform, grids)`. The internal
`_execution` module is an opt-in experiment that produces selectors for existing
codec consumers. Zarr's default indexers have not been replaced.

## The flow

```text
selection + source domain + indexing dialect
                    |
          normalize / compose
                    |
       request-to-storage mapping
                    |  + caller-selected chunk grid
                    v
       factor and partition the request
                    |
          +---------+-----------------+
          |                           |
  declarative projections       execution lowering
  chunk coordinates             chunk coordinates
  chunk_transform               chunk_selection
  cell_transform                out_selection
  coverage                      selected-value shape / coverage flag
          |                           |
          +------------+--------------+
                       |
             consumer fetches, decodes,
             gathers/scatters, encodes
```

This is a semantic flow, not a requirement to allocate every intermediate object.
For a basic slice, `execute_selection` prepares affine axes directly. For sorted
one-dimensional coordinates, it locates runs without creating a transform pair
for every chunk. Both use the same selection semantics as the transform path.

## 1. Interpret the selection

The indexing boundary chooses the dialect before chunk planning begins:

- A positional boundary, such as `LazyArray`, interprets NumPy-style negative
  indices and slice clipping against the current view.
- `IndexTransform` and the experimental `execute_selection` frontend use literal
  coordinates. Bounds are addresses; they do not wrap or clip like NumPy bounds.
- Basic, orthogonal, and vectorized indexing specify different mappings.
  Orthogonal arrays form an outer product; vectorized arrays select paired,
  broadcast coordinates.

Normalization resolves omitted dimensions and slice bounds and checks the
selection. Composing another selection onto a view produces a mapping directly
to the original source, rather than reading the intermediate view first. See
[Indexing patterns](patterns.md) for the dialect examples.

An `IndexTransform` represents this mapping with a request domain and one output
map per storage dimension: a constant, an affine function of a request axis, or
an index-array lookup. An integer selection can therefore remove a result axis
while retaining a constant coordinate for that storage dimension.

## 2. Keep the coordinate spaces distinct

| Space | Meaning | Example for `a[1:7:2, 2:6]` |
| --- | --- | --- |
| Global storage | Addresses in the source array | Rows `1, 3, 5`; columns `2, 3, 4, 5` |
| Chunk grid | Which grid cell contains an address | With `(4, 4)` chunks, address `(5, 4)` belongs to chunk `(1, 1)` |
| Chunk local | Address relative to that chunk's origin | `(5, 4) - (4, 4) = (1, 0)` |
| Request domain | Coordinates accepted by the request transform | May have a nonzero origin under literal slice semantics |
| Result buffer | Zero-based positions in the materialized result | Source `(5, 4)` goes to result `(2, 2)` |

For each storage dimension, the grid supplies `index_to_chunk`, `chunk_offset`,
and `chunk_size`. Uniform grids can use integer division; irregular grids use
their actual boundaries. The planner consumes this grid protocol rather than
assuming that all chunks have the same size.

A declarative `cell_transform` returns **request-domain coordinates**, not
necessarily zero-based buffer positions. A consumer placing values into a NumPy
result subtracts the request domain origin. Execution `out_selection` already
uses zero-based buffer positions.

## 3. Partition without expanding the entire request

Planning groups selected values by the chunks that contain them, retaining the
mapping back to their original request positions.

For affine selections, the shared `axis_runs` kernel finds each run that falls
inside one chunk. A run records the chunk coordinate, chunk origin, local start,
number of selected values, and starting request position. For a rectangular
request, the multidimensional chunk operations are products of these axis runs.
Declarative `StridedSet` tables materialize per-axis rows; basic execution keeps
large axes implicit and caches only small axes.

For index arrays, planning groups coordinates by chunk while retaining their
request positions. Arrays that share request axes must be grouped together:
those coordinates are correlated. Independent connected components stay
separate until chunk operations are visited. This avoids expanding an outer
product into one coordinate tuple per requested value during preparation.

`ChunkPlan.partition()` memoizes the factored `GridPartition`. Its tables include
strided sets, indexed sets, and joint sets for connected index-array components.
Plan iteration builds paired projections from those tables. Affine diagonals
use a separate intersection path because the per-storage-axis table form cannot
represent them; see [the current limitations](../design-notes.md#current-scope).

An empty request produces no chunk operations. Repeated coordinates remain
repeated request positions for reads: the selection is an ordered mapping, not
a mathematical set of source points.

## 4. Describe each chunk's contribution

The declarative representation is a `ChunkProjection`:

- `chunk_coords` identifies the grid cell, and `chunk_domain` gives its global
  storage bounds.
- `chunk_transform` maps a shared synthetic domain to chunk-local coordinates.
- `cell_transform` maps the same synthetic domain to request-domain coordinates.

The shared domain pairs each source value with its destination. Its coordinates
are neither automatically chunk-local nor automatically result positions.
For every point `p` in that domain, the defining relationship is:

```text
request_transform(cell_transform(p))
    == chunk_origin + chunk_transform(p)
```

A consumer can evaluate these transforms, consume the factored tables directly,
or lower them to its own selection vocabulary.

The experimental execution path emits `ExecutionChunk` rows. For a NumPy read,
their meaning is:

```text
result[row.out_selection] = decoded_chunk[row.chunk_selection]
```

`chunk_coords` tells the consumer which decoded chunk to obtain. For writes,
values flow in the opposite direction, using a plan prepared with `access="write"`.
`plan.lower("numpy").operations()` additionally exposes `value_shape` and
`selector_kind`: basic selectors use slice/integer semantics, while paired
coordinate arrays broadcast together. Consumers must preserve that distinction.

### A strided rectangle crossing four chunks

For an `(8, 10)` source with `(4, 4)` chunks, `a[1:7:2, 2:6]` has result shape
`(3, 4)`. The following slices describe the four contributions; slice stops are
shown clipped to each chunk's extent, so equivalent generated stops may differ.

| Chunk coordinates | Chunk origin | Chunk-local selection | Result selection |
| --- | --- | --- | --- |
| `(0, 0)` | `(0, 0)` | `[1:4:2, 2:4]` | `[0:2, 0:2]` |
| `(0, 1)` | `(0, 4)` | `[1:4:2, 0:2]` | `[0:2, 2:4]` |
| `(1, 0)` | `(4, 0)` | `[1:2:2, 2:4]` | `[2:3, 0:2]` |
| `(1, 1)` | `(4, 4)` | `[1:2:2, 0:2]` | `[2:3, 2:4]` |

This executable example checks both the touched chunks and the assembled values.
The array slicing used to obtain each decoded chunk stands in for storage and
codec work:

```python
--8<-- "snippets/selection_flow.py:basic"
```

### A gather with repeated, out-of-order coordinates

For `a[[6, 1, 6, 4]]` and chunk size `4`, grouping by chunk must preserve these
pairs:

| Chunk | Chunk-local coordinates | Result positions |
| --- | --- | --- |
| `(0,)` | `[1]` | `[1]` |
| `(1,)` | `[2, 2, 0]` | `[0, 2, 3]` |

Chunk `(1,)` is fetched once, but its local value `2` contributes to two result
positions. Chunk visitation order need not be result order; `out_selection`
restores the requested arrangement.

```python
--8<-- "snippets/selection_flow.py:gather"
```

## 5. Apply ownership, write, and coverage contracts

Execution defaults to snapshot ownership. Explicit borrowing permits retaining
caller arrays and requires them to remain unchanged throughout the plan's use.
A reader may repeat a source coordinate. A writer defaults to rejecting repeated
destinations; `conflicts="last"` explicitly keeps the final value in row-major
request order before dispatch. Grouping by chunk must not change that policy.

Coverage is relative to a grid and a consumer. Declarative coverage describes
the grid cell's data domain. The execution `is_complete_chunk` flag conservatively
proves coverage of the declared codec buffer, which may extend beyond the valid
data in a boundary chunk. Neither "this chunk was touched" nor "every chunk was
touched" proves that a write can skip reading existing contents.

Planning does not fetch bytes, decode buffers, choose concurrency, or guarantee
transactional writes. Those responsibilities begin at the consumer boundary.
Retaining every emitted row also costs memory even when the planner is compact.

## 6. Cross the sharding boundary

A sharded array has two relevant grids: outer shards and inner codec chunks.
The same logical request can be partitioned at either level, but their chunk
coordinates and origins differ.

**Current implementation:** the experimental shard lowering adapts selectors to
what the existing sharding indexer accepts. Some compact selections become flat
paired coordinate arrays. The sharding codec then calls `get_indexer` against
its inner grid, collects touched inner chunks for byte retrieval, and dispatches
inner chunk selections. This can expand and re-plan a selection.

**Proposed integration, not implemented:** pass the structured request mapping
into sharding, partition it against the inner grid, and lower selectors only
when the inner consumer needs them. Conceptually:

```text
request -> global storage
        -> selected outer shard + request-to-shard-local mapping
        -> selected inner chunk + request-to-inner-local mapping
        -> inner chunk selection + destination in the result
```

Touched inner chunk coordinates can support shard-index lookup and byte-range
coalescing without retaining every expanded value selector. Direct scattering
into the final result would additionally require the surrounding pipeline to
pass the output buffer and destination mapping through the shard boundary.
These are execution-interface changes; they do not require an on-disk format
change. Missing chunks, fill values, duplicate writes, and partial-write coverage
must retain their existing meaning across both partitioning levels.

For further details, see [Integration boundaries](integrations.md) and
[Prepared execution and explicit consumer lowering](../design-notes.md#prepared-execution-and-explicit-consumer-lowering).
