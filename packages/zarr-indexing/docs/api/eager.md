---
title: Eager adapter
---

# Eager adapter

`LazyArray` indexing returns views. `dask.array.from_array` reads its blocks
through NumPy conversion, which a bare view supports, but it also infers a
block prototype by indexing the input with empty slices, and a view's empty
slice is another view; reductions that call NumPy methods on that prototype
then fail. `EagerArrayAdapter` returns materialized arrays from indexing, read
through the view's reader and partitioning, so the prototype is an `ndarray`,
while the view itself keeps lazy indexing.

::: zarr_indexing.eager
