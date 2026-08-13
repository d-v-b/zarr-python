Resolving a partition's view on its own (`part.view.result()`) now hands the
reader the same `ReadContext` the parent's partitioned `result()` passes: the
paired `ChunkProjection` rides along instead of arriving as `projection=None`.
A custom reader keyed on `projection.chunk_coords` — a decoded-chunk cache —
now behaves identically on both paths, which the dask example's
task-per-partition pattern relies on. `with_reader` and repartitioning keep
the pairing; a further `.lazy` selection describes a different read and drops
it.
