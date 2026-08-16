# Zarrista-native Region I/O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace zarr-python's Python-side zarrista chunk write loop with zarrista's native synchronous and asynchronous `store_array_subset` operations, with regular/sharded correctness coverage and measured performance.

**Architecture:** `Region` remains the engine boundary and maps directly to step-one slices. The zarrista adapter normalizes each input to a contiguous NumPy array and makes one backend call; zarrista/zarrs owns chunk intersection, sharding, read-modify-write, and concurrency. The public Python codec pipeline is unchanged.

**Tech Stack:** Python 3.12, NumPy, pytest/pytest-asyncio, pytest-benchmark, zarrista at Git revision `92d26b65b90e9715d5c658c71b9216449f25ae64`, uv lock resolution.

**Spec:** `docs/superpowers/specs/2026-08-16-zarrista-native-region-io-design.md`

## Global Constraints

- Preserve the public `Region`, engine methods, array indexing APIs, and on-disk format.
- Do not change public codec signatures or introduce CuTe expressions.
- Keep `np.ascontiguousarray(value.as_ndarray_like())`; zero-copy and device inputs are out of scope.
- Pin zarrista exactly to `92d26b65b90e9715d5c658c71b9216449f25ae64`.
- Keep all unrelated untracked workspace files untouched and unstaged.
- Use conventional commits with `Assisted-by: Codex:GPT-5`.

## File map

- `src/zarr/zarrista/_engine.py`: translate `Region` and delegate sync/async reads and writes to zarrista; update metadata rebinding for the current zarrista API.
- `tests/zarrista/test_engine.py`: protect the one-call write boundary and exercise regular, sharded, edge, sync, async, and metadata-rebinding behavior.
- `tests/engine/test_differential.py`: compare sharded writes through default and zarrista engines with a hand-derived NumPy result.
- `tests/benchmarks/test_zarrista.py`: reproducible single-chunk, multi-chunk, and sharded regional-write workloads.
- `pyproject.toml`: update the exact zarrista dependency revision.
- `uv.lock`: record the resolved current zarrista source and transitive dependency changes.

---

### Task 1: Add the performance baseline and backend-contract test

**Files:**
- Create: `tests/benchmarks/test_zarrista.py`
- Modify: `tests/zarrista/test_engine.py`

**Interfaces:**
- Consumes: `ZarristaEngine.write_selection(selection: Region, value: NDBuffer, *, prototype: BufferPrototype) -> None`.
- Produces: a regression test requiring the backend operation `store_array_subset(selection: tuple[slice, ...], data: numpy.ndarray) -> None`, plus repeatable benchmark workloads.

- [ ] **Step 1: Add a benchmark that runs against the old implementation**

Create `tests/benchmarks/test_zarrista.py` with three cases whose setup is outside the timed function:

```python
from __future__ import annotations

from operator import setitem
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("zarrista")

import zarr
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture


CASES = [
    pytest.param((512, 512), (128, 128), None, np.s_[160:224, 160:224], id="single-partial"),
    pytest.param((512, 512), (128, 128), None, np.s_[32:480, 32:480], id="multi-chunk"),
    pytest.param((512, 512), (64, 64), (256, 256), np.s_[96:416, 96:416], id="sharded"),
]


@pytest.mark.parametrize("shape,chunks,shards,selection", CASES)
def test_zarrista_region_write(
    tmp_path: Path,
    benchmark: BenchmarkFixture,
    shape: tuple[int, int],
    chunks: tuple[int, int],
    shards: tuple[int, int] | None,
    selection: tuple[slice, slice],
) -> None:
    zarr.create_array(
        LocalStore(tmp_path), shape=shape, chunks=chunks, shards=shards, dtype="float32"
    )
    array = zarr.open_array(LocalStore(tmp_path), engine="zarrista")
    value_shape = tuple(index.stop - index.start for index in selection)
    value = np.ones(value_shape, dtype="float32")

    benchmark(setitem, array, selection, value)
```

- [ ] **Step 2: Capture the old-revision baseline**

Run:

```bash
uv run --group zarrista pytest --benchmark-enable \
  --benchmark-json=/private/tmp/zarrista-region-old.json \
  tests/benchmarks/test_zarrista.py
```

Expected: three benchmark results using zarrista revision
`95e47ad4c414c5920f0cf15550f923039641da8e`. Retain the JSON outside the repository for
the final comparison.

- [ ] **Step 3: Write the failing synchronous backend-contract test**

Add a small fake array to `tests/zarrista/test_engine.py`. It deliberately implements the
new array-subset contract and none of the old chunk methods:

```python
class SubsetWriteArray:
    def __init__(self) -> None:
        self.writes: list[tuple[tuple[slice, ...], np.ndarray[Any, Any]]] = []

    def store_array_subset(
        self, selection: tuple[slice, ...], data: np.ndarray[Any, Any]
    ) -> None:
        self.writes.append((selection, data.copy()))
```

Add imports for `Region`, `default_buffer_prototype`, and `ZarristaEngine`, then exercise
the fake with exact inputs:

```python
def test_zarrista_write_uses_array_subset_contract() -> None:
    backend = SubsetWriteArray()
    engine = ZarristaEngine(backend)
    prototype = default_buffer_prototype()
    value_np = np.arange(6, dtype="int32").reshape(3, 2).T
    assert not value_np.flags.c_contiguous
    value = prototype.nd_buffer.from_numpy_array(value_np)

    engine.write_selection(
        Region(start=(1, 2), end_exclusive=(3, 5)), value, prototype=prototype
    )

    assert len(backend.writes) == 1
    selection, written = backend.writes[0]
    assert selection == (slice(1, 3), slice(2, 5))
    np.testing.assert_array_equal(written, [[0, 2, 4], [1, 3, 5]])
    assert written.flags.c_contiguous
```

- [ ] **Step 4: Run the contract test and verify RED**

Run:

```bash
uv run --group zarrista pytest \
  tests/zarrista/test_engine.py::test_zarrista_write_uses_array_subset_contract -q
```

Expected: FAIL because the existing writer calls the absent `chunk_shape` method. This is
the intended failure: the wrapper still requires the obsolete per-chunk backend contract.

- [ ] **Step 5: Commit the tests and recorded benchmark harness**

Stage only the two files and commit with:

```text
test: cover zarrista native region writes

Assisted-by: Codex:GPT-5
```

### Task 2: Update zarrista and delegate region writes

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/zarr/zarrista/_engine.py`

**Interfaces:**
- Consumes: zarrista `Array.store_array_subset(selection, data)` and `AsyncArray.store_array_subset(selection, data)` at the pinned revision.
- Produces: synchronous and asynchronous `write_selection` methods requiring only the array-level subset operation; `with_metadata` retains `self._arr.storage`.

- [ ] **Step 1: Update and lock the dependency**

Change the zarrista revision in `pyproject.toml` to
`92d26b65b90e9715d5c658c71b9216449f25ae64`, then run:

```bash
uv lock --upgrade-package zarrista
uv sync --group zarrista
```

Verify the installed revision/API with:

```bash
uv run --group zarrista python -c \
  "import zarrista; assert hasattr(zarrista.Array, 'store_array_subset')"
```

- [ ] **Step 2: Implement the minimal synchronous delegation**

In `ZarristaEngine.write_selection`, retain the existing contiguous normalization and
replace the entire chunk loop with:

```python
self._arr.store_array_subset(_region_to_selection(selection), value_np)
```

Remove the local `import zarrista` from the method. Update `with_metadata` to pass
`self._arr.storage` to `Array.from_metadata`.

- [ ] **Step 3: Implement the minimal asynchronous delegation**

In `ZarristaAsyncEngine.write_selection`, retain `_ensure_arr` and contiguous normalization,
then replace the entire chunk loop with:

```python
await arr.store_array_subset(_region_to_selection(selection), value_np)
```

Remove the local `import zarrista` from the method.

- [ ] **Step 4: Delete redundant helpers and imports**

Delete `_chunks_overlapping` and the `itertools` import. Confirm there are no remaining
references to `retrieve_chunk`, `store_chunk`, `chunk_shape`, `chunk_subset`, or
`zarrista.ArrayBytes` in `_engine.py`.

- [ ] **Step 5: Run the contract test and verify GREEN**

Run the Task 1 contract test. Expected: PASS with exactly one recorded subset write and the
literal selection/data assertions satisfied.

- [ ] **Step 6: Run focused compatibility tests**

Run:

```bash
uv run --group zarrista pytest tests/zarrista/test_engine.py tests/zarrista/test_translate.py -q
uv run --group zarrista mypy src/zarr/zarrista tests/zarrista
```

Expected: all tests pass and mypy reports no errors. Adapt only private zarrista wrapper
code if the new binding has another required mechanical rename.

- [ ] **Step 7: Commit the implementation**

Stage only `pyproject.toml`, `uv.lock`, and `_engine.py`, then commit with:

```text
feat: delegate region writes to zarrista

Assisted-by: Codex:GPT-5
```

### Task 3: Add regular and sharded differential coverage

**Files:**
- Modify: `tests/zarrista/test_engine.py`
- Modify: `tests/engine/test_differential.py`

**Interfaces:**
- Consumes: public sync/async array indexing and the zarrista engine implementation from Task 2.
- Produces: regression coverage for regular multi-chunk, edge-chunk, inner-shard, cross-shard, async, and metadata-rebinding writes.

- [ ] **Step 1: Add a combined synchronous regular/sharded write test**

Parameterize literal cases containing shape, chunks, optional shards, selection, and value.
Include `(10, 9)/(3, 4)/None` with `np.s_[2:10, 1:9]` and
`(12, 12)/(3, 3)/(6, 6)` with `np.s_[2:10, 2:11]`. Initialize each array with
`np.arange`, write a separately constructed negative-valued array, mutate an independent
NumPy expected array, and assert the entire stored array equals expected. Comparing the
whole array proves neighbors and edge chunks are preserved.

- [ ] **Step 2: Extend the differential test to sharded writes**

Add `test_sharded_writes` beside `test_sharded_reads`. Parameterize over `ENGINES`, create a
sharded `(10, 9)` array with inner chunks `(3, 4)` and shards `(6, 8)`, seed it from a
literal `arange` array, write `np.s_[2:9, 3:9]`, and compare the whole result with an
independently mutated NumPy copy.

- [ ] **Step 3: Extend the asynchronous integration test**

In `test_zarrista_async_engine_read_write_combinations`, add a write crossing multiple
chunks and compare the entire async readback with an independently mutated expected array.
Keep the existing one-cell neighbor assertion.

- [ ] **Step 4: Add metadata-rebinding coverage**

Exercise `ZarristaEngine.with_metadata` using a real current-zarrista array. Assert a read
through the rebound engine returns the same stored values, proving `.storage` and `.path`
were retained. Do not assert the private property name directly.

- [ ] **Step 5: Run all new and existing differential tests**

Run:

```bash
uv run --group zarrista pytest tests/zarrista/test_engine.py tests/engine/test_differential.py -q
```

Expected: all default- and zarrista-engine cases pass.

- [ ] **Step 6: Perform the mutation check**

Temporarily replace `_region_to_selection(selection)` in the sync writer with full slices
and verify the contract or integration test fails; restore the implementation and rerun the
focused suite to green. This proves the tests catch an incorrect backend selection.

- [ ] **Step 7: Commit the differential coverage**

Stage only both test files and commit with:

```text
test: exercise zarrista sharded region writes

Assisted-by: Codex:GPT-5
```

### Task 4: Benchmark, verify, and document the outcome

**Files:**
- Modify only if verification reveals a scoped defect in files already listed above.

**Interfaces:**
- Consumes: old benchmark JSON from Task 1 and the completed implementation.
- Produces: measured comparison and final verification evidence.

- [ ] **Step 1: Capture the new-revision benchmark**

Run:

```bash
uv run --group zarrista pytest --benchmark-enable \
  --benchmark-json=/private/tmp/zarrista-region-new.json \
  tests/benchmarks/test_zarrista.py
```

- [ ] **Step 2: Compare benchmark medians**

Run a Python command in the same dependency environment that loads both JSON files and
prints each benchmark name, old median, new median, and `old / new` speed ratio. Report the
actual results without claiming a win for ratios within normal benchmark dispersion.

- [ ] **Step 3: Run focused and broader verification**

Run:

```bash
uv run --group zarrista pytest tests/zarrista tests/engine -q
uv run --group zarrista pytest tests/test_array.py tests/test_indexing.py -q
uv run --group zarrista mypy src/zarr/zarrista tests/zarrista tests/engine
uv run --group dev ruff check src/zarr/zarrista tests/zarrista tests/engine tests/benchmarks/test_zarrista.py
uv run --group dev ruff format --check src/zarr/zarrista tests/zarrista tests/engine tests/benchmarks/test_zarrista.py
```

Expected: every command exits zero, with no new warnings or errors.

- [ ] **Step 4: Review the final diff against the specification**

Confirm that the final diff contains one dependency update, the wrapper simplification,
tests, and the benchmark only. Confirm `rg` finds none of the deleted chunk operations in
`src/zarr/zarrista/_engine.py`, and confirm unrelated untracked files remain unstaged.

- [ ] **Step 5: Commit any final benchmark/test formatting changes**

If Task 4 changed tracked files, stage only those scoped files and commit with:

```text
test: benchmark zarrista region writes

Assisted-by: Codex:GPT-5
```

If no files changed, do not create an empty commit.
