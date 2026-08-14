# Lazy Indexing with asyncio

This example demonstrates using `zarr_indexing.LazyArray` as a chunk *planner*
while an async I/O layer — here `zarr.AsyncArray` — performs every read. The
package deliberately contains no scheduler; `parts()` exposes the partition
structure and this example shows `asyncio.gather` driving it.

The example shows how to:

- Lower each compatible box-shaped partition to a backend-native request with
  `part.source_selection` — a tuple of integers and slices in the wrapped
  array's own coordinates — and fetch all partitions concurrently, assembling
  each block with `out[part.out_selection] = await source.getitem(part.source_selection)`
- Normalize the two NumPy basic selectors that `zarr.AsyncArray.getitem` does
  not accept: a newaxis (`None`) and a negative-step slice. Those parts fetch
  their ascending cover through Zarr and apply the residual transform in
  memory.
- Build a decoded-chunk cache keyed on each touched grid cell's global origin:
  fetch the cell (`projection.chunk_domain`) once, then serve every overlapping
  view from the cache with `part.chunk_local_selection`
- Fall back for query partitions (`oindex`/`vindex`/mask selections), whose
  gathers have no single-slab spelling: `source_selection` raises
  `NoBasicSelectionError`, so the adapter fetches their ascending cover and
  applies the residual gather in memory

The async side only needs one method — `async def getitem(selection)` accepting
ascending basic slices — so the same adapter drives an HTTP range endpoint or
any other async source with Zarr's selection dialect. A backend with a wider
dialect can take the direct `source_selection` path for more parts.

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
