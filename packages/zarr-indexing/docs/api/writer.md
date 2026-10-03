---
title: Writers
---

# Writers

`LazyArray.write(values)` and assignment through a view are implemented by
`write_into`, which writes through a transform using only basic integer/slice
assignment on the source. Independent affine selections become one basic
assignment. Other selections are scattered against the source's write grid:
each touched cell is read once, updated in memory, and written back, so
storage round trips are bounded by touched cells rather than selected
elements. NumPy sources receive one fancy assignment instead. A source that
advertises no grid is one cell covering the whole array, so it is read and
written back once, over the selection's bounding box.

::: zarr_indexing.writer
