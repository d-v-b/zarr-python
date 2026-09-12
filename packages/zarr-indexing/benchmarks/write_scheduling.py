"""Structural planner comparison; separate planner runs may use separate environments.

No data is copied. Native returns write batches; Dask and Rechunker return
transfer stages with different execution/memory contracts. Times are not
end-to-end rechunk throughput. --planner rechunker requires Rechunker separately.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import statistics
import time
import tracemalloc
from typing import Any

CASES = [
    ("aligned", (120, 120), (12, 12), (12, 12)),
    ("misaligned", (120, 120), (12, 12), (16, 16)),
    ("row_to_column", (128, 128), (1, 128), (128, 1)),
    ("shard_units", (120, 120), (12, 12), (60, 60)),
    *[(f"hot_unit_{n}", (n,), (1,), (n,)) for n in (100, 1000, 10000)],
]


def split(shape: tuple[int, ...], chunks: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(min(c, n - i) for i in range(0, n, c)) for n, c in zip(shape, chunks, strict=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", choices=("native", "dask", "rechunker"), default="native")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    results: dict[str, Any] = {}
    for name, shape, source, target in CASES:
        max_mem = 4 * 8 * max(math.prod(source), math.prod(target))

        def operation(
            shape: tuple[int, ...] = shape,
            source: tuple[int, ...] = source,
            target: tuple[int, ...] = target,
            max_mem: int = max_mem,
        ) -> Any:
            if args.planner == "native":
                from zarr_indexing import IndexDomain, plan_rechunk
                from zarr_indexing.grid import dimension_grids_from_chunks

                return plan_rechunk(
                    IndexDomain.from_shape(shape),
                    dimension_grids_from_chunks(source, shape),
                    dimension_grids_from_chunks(target, shape),
                )
            if args.planner == "dask":
                module = importlib.import_module("dask.array.rechunk")
                return module.plan_rechunk(
                    split(shape, source), split(shape, target), itemsize=8, block_size_limit=max_mem
                )
            from rechunker.algorithm import rechunking_plan

            return rechunking_plan(shape, source, target, itemsize=8, max_mem=max_mem)

        operation()  # imports and warmup excluded
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            plan = operation()
            samples.append((time.perf_counter() - start) * 1000)
        tracemalloc.start()
        allocated = operation()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del allocated
        row: dict[str, Any] = {
            "shape": shape,
            "source": source,
            "destination_units": target,
            "median_ms": statistics.median(samples),
            "peak_mib": peak / 2**20,
        }
        if args.planner == "native":
            row.update(
                tasks=len(plan.pieces),
                batches=len(plan.schedule.batches),
                max_parallel_tasks=max(map(len, plan.schedule.batches), default=0),
                write_units=plan.schedule.n_write_units,
                task_unit_memberships=plan.schedule.n_memberships,
            )
        elif args.planner == "dask":
            row.update(max_mem=max_mem, transfer_stages=plan)
        else:
            read_chunks, intermediate_chunks, write_chunks = plan
            row.update(
                max_mem=max_mem,
                read_chunks=read_chunks,
                intermediate_chunks=intermediate_chunks,
                write_chunks=write_chunks,
            )
        results[name] = row
    package = "zarr-indexing" if args.planner == "native" else args.planner
    print(
        json.dumps(
            {
                "planner": args.planner,
                "version": importlib.metadata.version(package),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
