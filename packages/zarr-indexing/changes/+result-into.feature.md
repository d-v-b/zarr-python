Added `LazyArray.result_into(out, *, parts=None)`: the non-allocating form of
`result()`. The caller's writable buffer is validated against the view's shape
and dtype, filled in place — every cell written exactly once — and returned. A
view into a larger array qualifies, so a part's block can land directly in its
final slot, and a `numpy.ma` masked buffer keeps a masked source's mask.
`result()` itself is unchanged and always allocates.
