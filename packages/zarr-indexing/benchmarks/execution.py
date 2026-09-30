"""Compare partition rows with Zarr's indexers on the same selections; no storage I/O.

Grid construction and input selection allocation are excluded for both sides.
Indexer construction and complete row walks are included. Run from this
package directory with the in-repo zarr overlaid:

    uv run --with-editable ../.. --group test python benchmarks/execution.py
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

from zarr_indexing._execution import execute_selection

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
            "vectorized",
        ),
        ("correlated_dense", (1000, 1000), (10, 10), (i, i), "vectorized"),
        ("correlated_sparse", (10000, 10000), (10, 10), (i * 9, i * 9), "vectorized"),
        (
            "independent_one_chunk",
            (1000,) * 3,
            (1000,) * 3,
            (i[:, None], i[:, None], i[None, :]),
            "vectorized",
        ),
        (
            "independent_100_chunks",
            (1000,) * 3,
            (100,) * 3,
            (i[:, None], i[:, None], i[None, :]),
            "vectorized",
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
    cls = {
        "basic": zi.BasicIndexer,
        "orthogonal": zi.OrthogonalIndexer,
        "vectorized": zi.CoordinateIndexer,
    }[mode]

    def baseline() -> Any:
        return cls(selection, shape, zg)

    def rows() -> Any:
        return execute_selection(selection, shape, zg._dimensions, mode=mode)

    old_coords = [tuple(p.chunk_coords) for p in baseline()]
    if [tuple(p.chunk_coords) for p in rows()] != old_coords:
        raise RuntimeError("partition rows visit chunks in a different order from the zarr indexer")
    expected_size = (
        math.prod(shape)
        if mode == "basic"
        else (
            selection[0].size
            if mode == "orthogonal"
            else math.prod(np.broadcast_shapes(*(s.shape for s in selection)))
        )
    )
    operations = {
        "zarr_setup": baseline,
        "rows_setup": rows,
        "zarr_walk": lambda: consume(baseline()),
        "rows_walk": lambda: consume(rows()),
        "zarr_retained": lambda: list(baseline()),
        "rows_retained": lambda: list(rows()),
    }
    # Alternate evaluation order across rounds to reduce temporal bias.
    rounds = []
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
