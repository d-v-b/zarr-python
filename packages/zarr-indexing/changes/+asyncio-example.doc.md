Added an asyncio integration example (`examples/lazy_indexing_asyncio/`),
peer to the Dask example: `parts()` driven by `asyncio.gather` against
`zarr.AsyncArray`, using `Partition.source_selection` for per-part fetches, a
decoded-chunk cache keyed on each cell's `chunk_domain` origin and placed with
`chunk_local_selection`, and the `ValueError` fallback for query parts. The
integrations guide gained a matching "Consumer-owned I/O" section.
