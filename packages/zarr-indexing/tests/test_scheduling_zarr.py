"""Optional real codec integration; collected by the package CI suite."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from zarr_indexing import IndexDomain, plan_rechunk
from zarr_indexing.grid import dimension_grids_from_chunks

zarr = pytest.importorskip("zarr")


@pytest.mark.parametrize("pipeline", ["BatchedCodecPipeline", "FusedCodecPipeline"])
@pytest.mark.parametrize("layout", ["v2", "v3", "sharded"])
def test_concurrent_copy(pipeline: str, layout: str) -> None:
    shape = (9, 10)
    source_grid = dimension_grids_from_chunks((3, 4), shape)
    original = np.arange(90).reshape(shape)
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline." + pipeline}):
        target = zarr.create_array(
            {},
            shape=shape,
            chunks=(4, 3),
            dtype="int64",
            fill_value=-1,
            zarr_format=2 if layout == "v2" else 3,
            shards=(8, 6) if layout == "sharded" else None,
        )
        # Shards, not inner codec chunks, are the write isolation units.
        destination_grid = dimension_grids_from_chunks(target.shards or target.chunks, shape)
        domain = IndexDomain((1, 2), (8, 9))
        plan = plan_rechunk(domain, source_grid, destination_grid)

        def copy_piece(task: int) -> None:
            piece = plan.pieces[task]
            source_bounds = tuple(
                slice(g.chunk_offset(c), g.chunk_offset(c) + g.data_size(c))
                for g, c in zip(source_grid, piece.source_chunk, strict=True)
            )
            data = original[source_bounds][piece.source_selection]
            destination = tuple(
                slice(lo, hi)
                for lo, hi in zip(
                    piece.destination.domain.origin,
                    piece.destination.domain.exclusive_max,
                    strict=True,
                )
            )
            target[destination] = data

        with ThreadPoolExecutor(max_workers=4) as pool:
            for batch in plan.schedule.batches:
                # Consume the futures: submission alone is not a batch barrier.
                list(pool.map(copy_piece, batch))
        expected = np.full(shape, -1)
        expected[1:8, 2:9] = original[1:8, 2:9]
        np.testing.assert_array_equal(target[:], expected)
