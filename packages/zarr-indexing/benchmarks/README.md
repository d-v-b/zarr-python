# Chunk planning benchmarks

Run from the repository root, with `zarr-indexing` installed in the selected
Hatch environment:

```sh
hatch run test.py3.12-minimal:python packages/zarr-indexing/benchmarks/chunk_planning.py --repeats 9
```

`chunk_planning.py` measures construction, projection traversal, local-column
access, and chunk-coordinate enumeration. See the script for workloads and output fields.

Use the same interpreter, dependencies, inputs, and script revision when
comparing checkouts. Put the intended checkout's package source on `PYTHONPATH`;
installed package metadata is also required for version lookup. Record both git
revisions and the interpreter/dependency versions alongside saved output.

Compare complete operations, not only constructor times: plans can defer work
until iteration. Measure allocation separately from elapsed time. Distinguish
streaming consumption from retaining all projections or coordinates, and report
whether input arrays and transform construction are included. Repeated local
column access measures cache reuse, which trades allocation against retained
memory. Bounded coordinate batches avoid constructing the full coordinate array.

These scripts measure planning rather than codec or storage throughput. Repeat
measurements with alternating operation order before interpreting small timing
differences. Preserve raw benchmark output as an experiment artifact rather than
accumulating successive result tables in this README.

## Experimental execution

The execution follow-up adds two scripts:

```sh
hatch run test.py3.12-minimal:python packages/zarr-indexing/benchmarks/execution.py
hatch run test.py3.12-minimal:python packages/zarr-indexing/benchmarks/execution_io.py
```

`execution.py` compares Zarr indexers, declarative projections, and execution
selectors, including retained rows, borrowed inputs, snapshots, and shard
lowering. `execution_io.py` verifies and measures MemoryStore reads and writes
through real codec pipelines. Neither establishes filesystem or cloud throughput.
The prototype remains opt-in; planning wins alone do not justify replacing the
existing indexers.
