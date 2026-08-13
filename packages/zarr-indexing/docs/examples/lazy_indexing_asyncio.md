--8<-- "lazy_indexing_asyncio/README.md"

`LazyArray` owns the indexing-derived plan — which grid cells a view touches,
what to request from each, and where each block lands — while the consumer's
event loop owns concurrency and I/O. The projection pair travels with each
partition, so a cache keyed on `base_coords` and sliced with
`chunk_local_selection` needs no coordinate arithmetic of its own.

## Source Code

```python
--8<-- "lazy_indexing_asyncio/lazy_indexing_asyncio.py"
```
