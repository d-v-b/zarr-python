Resolving a partition's view on its own (`part.view.result()`) now hands the
reader the same `ReadContext` the parent's partitioned `result()` passes: the
paired `ChunkProjection` rides along instead of arriving as `projection=None`.
A custom reader keyed on `projection.chunk_coords` — a decoded-chunk cache —
now behaves identically on both paths, which the dask example's
task-per-partition pattern relies on. `with_reader` and repartitioning keep
the pairing; a further `.lazy` selection describes a different read and drops
it. So does a further partitioning of a part, whose cells are counted in the
part's own box and would name the wrong chunk of the source — such a read
pairs with no projection, as it did before, rather than with a misleading one.

Reusing a plan (`view.result(parts=view.parts())`) on an unpartitioned view
now reads through the same context as the plain call, instead of through a
freshly synthesized whole-base projection.
