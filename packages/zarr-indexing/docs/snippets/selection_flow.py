"""Follow a selection into declarative projections and executable chunk rows."""

import numpy as np

from zarr_indexing import IndexTransform, plan_chunks
from zarr_indexing._execution import execute_selection
from zarr_indexing.grid import dimension_grids_from_chunks

# --8<-- [start:basic]
source = np.arange(80).reshape(8, 10)
selection = (slice(1, 7, 2), slice(2, 6))
grids = dimension_grids_from_chunks((4, 4), source.shape)
transform = IndexTransform.from_shape(source.shape)[selection]
projections = plan_chunks(transform, grids)
execution = execute_selection(selection, source.shape, grids)

assert execution.shape == (3, 4)
assert [p.chunk_coords for p in projections] == [(0, 0), (0, 1), (1, 0), (1, 1)]

result = np.empty(execution.shape, dtype=source.dtype)
for row in execution:
    # In a codec, the storage layer supplies this decoded chunk.
    bounds = tuple(
        slice(grid.chunk_offset(c), grid.chunk_offset(c) + grid.data_size(c))
        for grid, c in zip(grids, row.chunk_coords, strict=True)
    )
    chunk = source[bounds]
    result[row.out_selection] = chunk[row.chunk_selection]

np.testing.assert_array_equal(result, source[selection])
assert result.tolist() == [[12, 13, 14, 15], [32, 33, 34, 35], [52, 53, 54, 55]]
# --8<-- [end:basic]

# Every declarative pair expresses the same global-to-request correspondence.
for projection in projections:
    domain = projection.chunk_transform.domain
    for position in np.ndindex(domain.shape):
        point = tuple(a + b for a, b in zip(domain.origin, position, strict=True))
        local = projection.chunk_transform.apply(point)
        request = projection.cell_transform.apply(point)
        global_point = tuple(
            grid.chunk_offset(c) + x
            for grid, c, x in zip(grids, projection.chunk_coords, local, strict=True)
        )
        assert transform.apply(request) == global_point

# --8<-- [start:gather]
coordinates = np.array([6, 1, 6, 4])
gather = execute_selection(
    coordinates, (8,), dimension_grids_from_chunks((4,), (8,)), mode="vectorized"
)
rows = {row.chunk_coords: row for row in gather}
np.testing.assert_array_equal(rows[(0,)].chunk_selection[0], [1])
np.testing.assert_array_equal(rows[(0,)].out_selection[0], [1])
np.testing.assert_array_equal(rows[(1,)].chunk_selection[0], [2, 2, 0])
np.testing.assert_array_equal(rows[(1,)].out_selection[0], [0, 2, 3])

values = np.arange(8)
gathered = np.empty(gather.shape, dtype=values.dtype)
for row in gather:
    start = row.chunk_coords[0] * 4
    gathered[row.out_selection] = values[start : start + 4][row.chunk_selection]
np.testing.assert_array_equal(gathered, values[coordinates])
# --8<-- [end:gather]
