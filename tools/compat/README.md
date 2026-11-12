# Compatibility differential

Checks that this checkout behaves as a baseline zarr does wherever existing code can
look: the synchronous `Array` API, `AsyncArray` used as a dataclass (fields, `replace`,
`asdict`, equality, `object.__setattr__`), copies, pickles carried between the two
versions in both directions, `AsyncArray` subclasses of several shapes, and arrays whose
metadata documents another writer stored (chunk sizes zarr reads only by repairing them).

```bash
hatch run test.py3.12-optional:python tools/compat/compat.py --baseline "$(git merge-base HEAD upstream/main)" --seeds 2000
```

`--baseline` takes any commit, branch or tag: the `main` a branch holds, as above, or a
release such as `v3.4.2`. The run takes about half a minute for 2000 seeds, prints the differences by class
(see `diff.py`) and leaves every run's steps in its work directory (`--work`).

- `compat_scenarios.py` holds the scenarios. Random programs (one per seed) create or open
  an array, then apply a seeded sequence of writes, reads, resizes, appends, attribute
  updates, reopens and copies through several handles, recording what each step returns,
  raises and warns, and the store's contents. The sequence depends only on the seed, so
  both versions perform the same operations.
- `run.py` runs them under one source tree; `diff.py` compares two runs.

A difference is a change a caller can observe, which is not always a break: read it, and
say in the change's notes why each one is intended. Steps labeled `internal:` look at
private state and are reported apart. The scenarios cover existing surfaces only; the
explicit-IO API (`zarr.ArrayIO`) is new, so its tests are the regular suite.
