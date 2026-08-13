# Lazy Indexing with asyncio

This example demonstrates using `zarr_indexing.LazyArray` as a chunk *planner*
while an async I/O layer — here `zarr.AsyncArray` — performs every read. The
package deliberately contains no scheduler; `parts()` exposes the partition
structure and this example shows `asyncio.gather` driving it.

The example shows how to:

- Lower each box-shaped partition to a backend-native request with
  `part.source_selection` — a tuple of integers and slices in the wrapped
  array's own coordinates — and fetch all partitions concurrently, assembling
  each block with `out[part.out_selection] = await source.getitem(part.source_selection)`
- Build a decoded-chunk cache keyed on `part.base_coords`: fetch each touched
  grid cell (`projection.chunk_domain`) once, then serve every overlapping view
  from the cache with `part.chunk_local_selection`
- Fall back for query partitions (`oindex`/`vindex`/mask selections), whose
  gathers have no single-slab spelling: `source_selection` raises `ValueError`
  and the part resolves through its own `part.view.result()` instead

The async side only needs one method — `async def getitem(selection)` accepting
a basic selection — so the same loop drives an HTTP range endpoint or any other
async source. Note that a reversing view (`lazy[::-1]`) lowers to a
negative-step slice, which Zarr's basic selections reject; inspect the slice
steps if your backend only walks forward.

## Running the Example

The script declares its dependencies inline
([PEP 723](https://peps.python.org/pep-0723/)), so the easiest way to run it is
with [uv](https://docs.astral.sh/uv/), which installs them automatically:

```bash
cd packages/zarr-indexing
uv run --with-editable . examples/lazy_indexing_asyncio/lazy_indexing_asyncio.py
```

Alternatively, run it with plain Python, in which case you must first install
`zarr`, `zarr-indexing`, `numpy`, and `pytest` yourself:

```bash
cd packages/zarr-indexing
python examples/lazy_indexing_asyncio/lazy_indexing_asyncio.py
```
