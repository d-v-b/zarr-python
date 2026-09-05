"""Compare PR planner with existing Zarr indexers; no storage I/O.

Grid construction and input selection allocation are excluded for both sides.
Selection compilation/indexer construction and complete streaming walks are included.
Run with the worktree src and packages/zarr-indexing/src on PYTHONPATH in Hatch.
"""

from __future__ import annotations

import json
import math
import statistics
import time
import tracemalloc
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr.core.indexing as zi
from zarr.core.chunk_grids import ChunkGrid

from zarr_indexing import IndexTransform, plan_chunks
from zarr_indexing._execution import execute_selection
from zarr_indexing.grid import dimension_grids_from_chunks

if TYPE_CHECKING:
    from collections.abc import Callable


def consume(iterator: Any) -> int:
    return sum(1 for _ in iterator)


def measure(op: Callable[[], Any], repeats: int = 31) -> dict[str, float]:
    op()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        op()
        samples.append((time.perf_counter() - start) * 1000)
    tracemalloc.start()
    result = op()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return {"ms": statistics.median(samples), "peak_mib": peak / 2**20}


def main() -> None:
    i = np.arange(1000)
    cases = [
        ("basic_10000_chunks", (1000, 1000), (10, 10), (slice(None), slice(None)), "basic"),
        (
            "sorted_1M_points_10_chunks",
            (1_000_000,),
            (100_000,),
            (np.arange(1_000_000),),
            "orthogonal",
        ),
        (
            "sorted_coordinate_1M_points_10_chunks",
            (1_000_000,),
            (100_000,),
            (np.arange(1_000_000),),
            "coordinate",
        ),
        ("correlated_dense", (1000, 1000), (10, 10), (i, i), "coordinate"),
        ("correlated_sparse", (10000, 10000), (10, 10), (i * 9, i * 9), "coordinate"),
        (
            "independent_one_chunk",
            (1000,) * 3,
            (1000,) * 3,
            (i[:, None], i[:, None], i[None, :]),
            "coordinate",
        ),
        (
            "independent_100_chunks",
            (1000,) * 3,
            (100,) * 3,
            (i[:, None], i[:, None], i[None, :]),
            "coordinate",
        ),
    ]
    results = {
        name: compare_case(shape, chunks, selection, mode)
        for name, shape, chunks, selection, mode in cases
    }
    print(json.dumps(results, indent=2))


def compare_case(
    shape: tuple[int, ...], chunks: tuple[int, ...], selection: Any, mode: str
) -> dict[str, Any]:
    zg = ChunkGrid.from_sizes(shape, chunks)
    pg = dimension_grids_from_chunks(chunks, shape)
    cls = {
        "basic": zi.BasicIndexer,
        "orthogonal": zi.OrthogonalIndexer,
        "coordinate": zi.CoordinateIndexer,
    }[mode]

    def baseline() -> Any:
        return cls(selection, shape, zg)

    def new() -> Any:
        base = IndexTransform.from_shape(shape)
        transform = (
            base[selection]
            if mode == "basic"
            else (base.oindex[selection] if mode == "orthogonal" else base.vindex[selection])
        )
        return plan_chunks(transform, pg).partition()

    old_coords = [tuple(p.chunk_coords) for p in baseline()]
    partition = new()
    new_coords = [tuple(p.chunk_coords) for p in partition]
    assert old_coords == new_coords
    expected_size = (
        math.prod(shape)
        if mode == "basic"
        else (
            selection[0].size
            if mode == "orthogonal"
            else math.prod(np.broadcast_shapes(*(s.shape for s in selection)))
        )
    )
    assert sum(math.prod(p.cell_transform.domain.shape) for p in partition) == expected_size
    # Alternate evaluation order across rounds to reduce temporal bias.
    rounds = []

    def immediate() -> Any:
        return execute_selection(
            selection,
            shape,
            pg,
            mode={"basic": "basic", "orthogonal": "orthogonal", "coordinate": "vectorized"}[mode],
        )

    assert [tuple(p.chunk_coords) for p in immediate()] == old_coords
    operations = {
        "zarr_setup": baseline,
        "new_setup": new,
        "zarr_walk": lambda: consume(baseline()),
        "new_walk": lambda: consume(new()),
        "immediate_setup": immediate,
        "immediate_walk": lambda: consume(immediate()),
    }
    for round_id in range(3):
        names = list(operations)
        if round_id % 2:
            names.reverse()
        rounds.append({key: measure(operations[key]) for key in names})
    return {
        "chunks": len(old_coords),
        "elements": expected_size,
        "metrics": {
            key: {
                "ms": statistics.median(r[key]["ms"] for r in rounds),
                "peak_mib": max(r[key]["peak_mib"] for r in rounds),
                "round_ms": [r[key]["ms"] for r in rounds],
            }
            for key in operations
        },
    }


if __name__ == "__main__":
    main()
