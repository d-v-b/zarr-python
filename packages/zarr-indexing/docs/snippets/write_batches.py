"""Schedule already chunked data without changing the incoming chunk tasks."""

# --8<-- [start:copy]
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from zarr_indexing import IndexDomain, plan_rechunk
from zarr_indexing.grid import dimension_grids_from_chunks

source_chunks = {i: np.arange(3 * i, 3 * i + 3) for i in range(4)}
target = np.full(12, -1)
plan = plan_rechunk(
    IndexDomain.from_shape((12,)),
    dimension_grids_from_chunks((3,), (12,)),
    dimension_grids_from_chunks((4,), (12,)),
)
assert plan.schedule.batches == ((0, 2), (1, 3))


def copy_piece(task: int) -> None:
    piece = plan.pieces[task]
    values = source_chunks[piece.source_chunk[0]][piece.source_selection]
    domain = piece.destination.domain
    selection = tuple(
        slice(lo, hi) for lo, hi in zip(domain.origin, domain.exclusive_max, strict=True)
    )
    target[selection] = values


with ThreadPoolExecutor(max_workers=4) as pool:
    for batch in plan.schedule.batches:
        # Consume results and propagate errors before the next batch starts.
        list(pool.map(copy_piece, batch))

np.testing.assert_array_equal(target, np.arange(12))
# --8<-- [end:copy]
