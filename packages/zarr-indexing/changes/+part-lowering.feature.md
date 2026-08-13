Box selections and partitions now lower to backend-native basic selections,
for consumers that plan reads here but fetch through their own I/O layer (an
async store, an HTTP range endpoint):

- `IndexTransform.as_basic_selection()` converts a box transform to a tuple of
  integers and slices such that `source[selection]` reads exactly the
  transform's cells, at exactly its domain shape. A collapsed
  single-coordinate gather keeps its singleton axis through a length-1 slice;
  queries, broadcasts, and transposed or repeated axes raise `ValueError`
  instead of guessing a slab.
- `Partition.source_selection` is that lowering of a part's global read, so an
  async consumer's whole loop is
  `out[part.out_selection] = await source.getitem(part.source_selection)`.
- `Partition.chunk_local_selection` is the same read relative to
  `projection.chunk_domain`'s origin, for decoded-chunk caches. The domain
  names its cell in the source's own coordinates even for a view partitioning
  a window of it, so the cached cell is the same read either way;
  `base_coords` counts cells of the partitioned base, so it keys such a cache
  only together with the grid that produced it.

`ChainedIndexingStateMachine` gained an invariant that runs both documented
assembly loops literally under plain NumPy semantics, with no reader involved.
