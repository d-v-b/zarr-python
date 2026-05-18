# Lazy Indexing

Zarr arrays expose a deferred-indexing API via the `lazy` accessor. Indexing
into `arr.lazy[...]` composes a selection without performing any I/O; calling
`.result()` (or `np.asarray()`) materializes the view as a numpy array.

```python
import zarr

arr = zarr.create_array(
    store="data.zarr",
    shape=(1000, 1000),
    dtype="int32",
    chunks=(100, 100),
)

# Indexing arr.lazy composes a selection without I/O.
view = arr.lazy[100:200]

# Further indexing still does no I/O — selections compose.
sub = view[10:50, 5:50]

# Call .result() to materialize as a numpy array.
data = sub.result()           # equivalent to arr[110:150, 5:50]
```

## Orthogonal and vectorized indexing

The lazy view supports the same three indexing modes as `Array` itself:

- `arr.lazy[selection]` — basic indexing (slices, ints, ellipsis).
- `arr.lazy.oindex[selection]` — orthogonal indexing (one selector per axis).
- `arr.lazy.vindex[selection]` — vectorized indexing (correlated point selection).

Each returns another `_LazyArray` with the selection composed onto its
internal transform; no data is read until `.result()` is called.

```python
import numpy as np

# Orthogonal: pick rows 1, 3, 5 and all columns.
rows = arr.lazy.oindex[np.array([1, 3, 5]), :].result()

# Vectorized: pick the (1, 4), (3, 6), (5, 8) points.
points = arr.lazy.vindex[np.array([1, 3, 5]), np.array([4, 6, 8])].result()
```

## NumPy interoperability

`_LazyArray` implements NumPy's `__array__` protocol, so `np.asarray(view)`
and `np.array(view)` materialize it. This enables interop with matplotlib,
dask, scikit-learn, and other libraries that call `np.asarray()` internally:

```python
import matplotlib.pyplot as plt

plt.imshow(arr.lazy[100:200, 100:200])    # materialized via __array__
```

The `copy=False` argument to `np.array` is not honored: the lazy view always
materializes by allocating a new array via the eager indexing path. Passing
`copy=False` raises `ValueError`, matching the NumPy 2.0 protocol contract.

## Writes

Writing via the lazy accessor materializes immediately:

```python
arr.lazy[5, 10] = -1                # written through to arr
arr.lazy.oindex[np.array([1, 3]), :] = 99  # also immediate
```

Lazy *writes* are not deferred. The lazy view is for deferred *reads*; writes
are pass-through to the underlying array's eager set methods. If you want
batched writes, build the selection and value separately and call them
eagerly.

## API status

This API is in early development. The underlying machinery
(`zarr.core._lazy`, `zarr.core._transforms`) is private and may change
without notice. The user-visible `Array.lazy` property is the only public
surface, and may also evolve based on early-adopter feedback.
