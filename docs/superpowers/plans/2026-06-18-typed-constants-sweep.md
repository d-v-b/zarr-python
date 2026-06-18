# Typed Constants Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For every enumerable string `Literal` in zarr-python, expose a paired runtime `Final` tuple of its members, sourcing the spec-string pairs from the `zarr-metadata` package and eliminating ill-typed `get_args()` calls from `src/`.

**Architecture:** Adopt the jorenham/`optype` + zarr-metadata convention: a `Literal` alias (singular name) and a `Final` tuple of the same members (plural/SCREAMING name), **written independently**, tied by a `set(CONST) == set(get_args(Literal))` drift test. `Literal[*CONST]` unpacking is rejected by mypy (verified) and `get_args()` leaks `tuple[Any, ...]` under `strict = true`, so neither is used in `src/`. Spec strings (codec/dtype/grid/chunk-key names, blosc/endian/sharding enums) live in `zarr-metadata` and are imported by zarr-python; non-spec operational enums (access modes, cast-value behavior) stay local but adopt the same paired form.

**Tech Stack:** Python 3.12 (zarr-python floor), 3.11 (zarr-metadata floor), `typing.Final`/`Literal`, mypy `strict = true`, pytest, `uv`.

## Global Constraints

- Always run tooling via `uv run` (e.g. `uv run pytest`, `uv run mypy`). Verbatim user rule.
- Docs/docstrings use single-backtick markdown (mkdocs), never RST double-backtick. Verbatim user rule.
- Never use `Literal[*CONST]` unpacking — mypy `strict` rejects it (`Parameter 1 of Literal[...] is invalid [valid-type]`). Verified in this session.
- `get_args()` is permitted **only in tests** (drift guards, `pytest.parametrize`), never in `src/`.
- zarr-metadata floor is `>=3.11`: any new zarr-metadata code must avoid 3.12-only syntax (no PEP 695 `type X = ...` statement unless the file already uses it — check the file first; the existing pairs use `Name = Literal[...]` assignment form, follow that).
- Convention for every pair: type alias `SomeName` (singular), constant `SOME_NAME: Final = (...)` (plural). Each pair gets exactly one drift test.
- Both packages run mypy `strict`; every task must end green under `uv run mypy`.

---

## File Structure

**zarr-metadata (source of truth for spec strings):**
- `packages/zarr-metadata/src/zarr_metadata/v3/codec/sharding_indexed.py` — add `SUBCHUNK_WRITE_ORDER` constant if `SubchunkWriteOrder` is added here (see Task 7 decision).
- `packages/zarr-metadata/src/zarr_metadata/v3/data_type/numpy_timedelta64.py` — add `DATETIME_UNIT: Final` next to existing `DateTimeUnit` (currently has Literal but no constant tuple).
- `packages/zarr-metadata/src/zarr_metadata/__init__.py` — promote the constant+Literal pairs zarr-python consumes into the public API (`__all__`).
- `packages/zarr-metadata/tests/` — drift tests for any newly added constants.

**zarr-python (consumer):**
- `pyproject.toml` — add `zarr-metadata` path dependency.
- `src/zarr/codecs/bytes.py`, `blosc.py`, `sharding.py` — replace local spec pairs with re-exports from zarr-metadata.
- `src/zarr/core/dtype/npy/common.py`, `core/dtype/common.py` — `DateTimeUnit`/`DATETIME_UNIT` from zarr-metadata; keep dtype-internal pairs local.
- `src/zarr/core/common.py`, `core/chunk_key_encodings.py` — keep `MemoryOrder`, `SeparatorLiteral`, `AccessModeLiteral` local (see decisions); ensure each has a paired constant.
- `src/zarr/storage/_zip.py`, `codecs/cast_value.py` — add paired constants for `ZipStoreAccessModeLiteral`, `RoundingMode`, `OutOfRangeMode`.
- `src/zarr/core/dtype/npy/time.py` — replace `get_args(DateTimeUnit)` with `DATETIME_UNIT`.
- `tests/test_common.py`, `tests/test_metadata/`, etc. — drift tests.

## Decisions locked before implementation (deviations from raw inventory)

- **`MemoryOrder` stays LOCAL.** It is runtime array memory layout (C/F contiguity), *not* the v2-metadata `order` field. Aliasing it to `zarr_metadata.v2.array.ArrayOrderV2` would couple a runtime concept to a v2-spec type and break if the spec type ever diverges. Keep `MemoryOrder` in `core/common.py`; add `MEMORY_ORDER: Final = ("C", "F")` locally if a runtime tuple is needed (only if a consumer exists — none currently, so **skip** per "coverage follows consumption").
- **`SeparatorLiteral` stays LOCAL** unless order is normalized. zarr-python uses `Literal[".", "/"]`; zarr-metadata uses `Literal["/", "."]`. Same members, but to import the zarr-metadata type the zarr-python annotations must tolerate the reordered Literal (they do — Literal member order is not semantically significant to type compatibility, only to `get_args` ordering). Migration is *optional* and low-value; **keep local** to avoid touching 6 sites for cosmetics.
- **`SubchunkWriteOrder`** has no zarr-metadata equivalent and is a zarr-python operational detail (not in the sharding spec). **Keep local**, already paired.
- **Pure dtype-internal pairs** (`EndiannessStr`, `NumpyEndiannessStr`, `SpecialFloatStrings`, `ObjectCodecID`) have no zarr-metadata home and are already paired. **Keep local**, no change beyond a drift test.
- **Migrate to zarr-metadata imports:** `EndianLiteral`→`Endian`, `BloscShuffleLiteral`→`BloscShuffle`, `BloscCnameLiteral`→`BloscCName`, `IndexLocation`→`IndexLocation`, `DateTimeUnit`→`DateTimeUnit`. These have identical members and a true spec home.

---

### Task 1: Add zarr-metadata as a dependency of zarr-python

**Files:**
- Modify: `pyproject.toml` (zarr-python root — `[project] dependencies` and `[tool.uv.sources]`)

**Interfaces:**
- Produces: `import zarr_metadata` succeeds in the zarr-python env. Later tasks rely on this.

- [ ] **Step 1: Inspect current dependency + uv source config**

Run: `uv run python -c "import tomllib,sys; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['dependencies']); print(d.get('tool',{}).get('uv',{}).get('sources'))"`
Expected: prints the deps list (no `zarr-metadata`) and existing uv sources, if any.

- [ ] **Step 2: Add the dependency and path source**

Add `"zarr-metadata"` to `[project] dependencies` in `pyproject.toml`, and under `[tool.uv.sources]` add:

```toml
zarr-metadata = { path = "packages/zarr-metadata", editable = true }
```

(If `[tool.uv.sources]` does not exist, create it. Match the existing TOML quoting/indent style in the file.)

- [ ] **Step 3: Sync and verify import**

Run: `uv sync && uv run python -c "import zarr_metadata; print(zarr_metadata.__file__)"`
Expected: prints a path under `packages/zarr-metadata/src/zarr_metadata/__init__.py`.

- [ ] **Step 4: Verify nothing broke**

Run: `uv run python -c "import zarr; print(zarr.__version__)"`
Expected: prints a version, no ImportError.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: depend on zarr-metadata as a path dependency"
```

---

### Task 2: Add missing constants in zarr-metadata + promote consumed pairs to public API

**Files:**
- Modify: `packages/zarr-metadata/src/zarr_metadata/v3/data_type/numpy_timedelta64.py` (add `DATETIME_UNIT`)
- Modify: `packages/zarr-metadata/src/zarr_metadata/__init__.py` (export consumed pairs)
- Test: `packages/zarr-metadata/tests/test_public_api.py` (new) and a drift test

**Interfaces:**
- Consumes: existing `DateTimeUnit`, `Endian`/`ENDIAN`, `BloscShuffle`/`BLOSC_SHUFFLE`, `BloscCName`/`BLOSC_CNAME`, `IndexLocation`/`INDEX_LOCATION`.
- Produces: public top-level names importable as `from zarr_metadata import Endian, ENDIAN, BloscShuffle, BLOSC_SHUFFLE, BloscCName, BLOSC_CNAME, IndexLocation, INDEX_LOCATION, DateTimeUnit, DATETIME_UNIT`.

- [ ] **Step 1: Write the failing drift + export test**

Create `packages/zarr-metadata/tests/test_public_api.py`:

```python
from typing import get_args

import zarr_metadata as zm


def test_consumed_pairs_are_public() -> None:
    for name in (
        "Endian", "ENDIAN",
        "BloscShuffle", "BLOSC_SHUFFLE",
        "BloscCName", "BLOSC_CNAME",
        "IndexLocation", "INDEX_LOCATION",
        "DateTimeUnit", "DATETIME_UNIT",
    ):
        assert name in zm.__all__, f"{name} missing from public API"
        assert hasattr(zm, name)


def test_constant_matches_literal() -> None:
    assert set(zm.ENDIAN) == set(get_args(zm.Endian))
    assert set(zm.BLOSC_SHUFFLE) == set(get_args(zm.BloscShuffle))
    assert set(zm.BLOSC_CNAME) == set(get_args(zm.BloscCName))
    assert set(zm.INDEX_LOCATION) == set(get_args(zm.IndexLocation))
    assert set(zm.DATETIME_UNIT) == set(get_args(zm.DateTimeUnit))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/zarr-metadata/tests/test_public_api.py -v`
Expected: FAIL — `DATETIME_UNIT` absent and names not in `__all__`.

- [ ] **Step 3: Add `DATETIME_UNIT` constant**

In `packages/zarr-metadata/src/zarr_metadata/v3/data_type/numpy_timedelta64.py`, directly after the existing `DateTimeUnit = Literal[...]` block, add (copy the same 15 members verbatim, in the same order as the Literal):

```python
DATETIME_UNIT: Final = (
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as", "generic",
)
"""Runtime tuple of the permitted `numpy.timedelta64`/`numpy.datetime64` unit strings."""
```

Add `"DATETIME_UNIT"` to that module's `__all__` if it has one.

- [ ] **Step 4: Promote pairs to top-level `__init__`**

In `packages/zarr-metadata/src/zarr_metadata/__init__.py`, add imports and `__all__` entries:

```python
from zarr_metadata.v3.codec.bytes import ENDIAN, Endian
from zarr_metadata.v3.codec.blosc import BLOSC_CNAME, BLOSC_SHUFFLE, BloscCName, BloscShuffle
from zarr_metadata.v3.codec.sharding_indexed import INDEX_LOCATION, IndexLocation
from zarr_metadata.v3.data_type.numpy_timedelta64 import DATETIME_UNIT, DateTimeUnit
```

Add `"ENDIAN"`, `"Endian"`, `"BLOSC_CNAME"`, `"BLOSC_SHUFFLE"`, `"BloscCName"`, `"BloscShuffle"`, `"INDEX_LOCATION"`, `"IndexLocation"`, `"DATETIME_UNIT"`, `"DateTimeUnit"` to `__all__` (keep it sorted to match the file's existing alphabetized style).

- [ ] **Step 5: Run tests + mypy**

Run: `uv run pytest packages/zarr-metadata/tests/test_public_api.py -v && uv run mypy packages/zarr-metadata`
Expected: PASS, no mypy errors.

- [ ] **Step 6: Commit**

```bash
git add packages/zarr-metadata
git commit -m "feat(zarr-metadata): add DATETIME_UNIT and promote consumed constant pairs to public API"
```

---

### Task 3: Re-export Endian pair in zarr-python from zarr-metadata

**Files:**
- Modify: `src/zarr/codecs/bytes.py:21-24`
- Test: `tests/test_codecs/test_bytes.py`

**Interfaces:**
- Consumes: `from zarr_metadata import Endian, ENDIAN` (Task 2).
- Produces: `zarr.codecs.bytes.EndianLiteral` and `zarr.codecs.bytes.ENDIAN` remain importable with identical members (back-compat alias).

- [ ] **Step 1: Write the failing equivalence test**

Add to `tests/test_codecs/test_bytes.py`:

```python
def test_endian_pair_sourced_from_zarr_metadata() -> None:
    from typing import get_args

    import zarr_metadata as zm
    from zarr.codecs.bytes import ENDIAN, EndianLiteral

    assert ENDIAN == zm.ENDIAN
    assert set(ENDIAN) == set(get_args(EndianLiteral))
    assert get_args(EndianLiteral) == get_args(zm.Endian)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_codecs/test_bytes.py::test_endian_pair_sourced_from_zarr_metadata -v`
Expected: FAIL — local `EndianLiteral` not yet identical to `zm.Endian` (it is independently defined; identity check on `get_args` may pass by value, but import wiring not in place — if it passes already, still proceed to make the source single).

- [ ] **Step 3: Replace local definitions with re-export**

In `src/zarr/codecs/bytes.py`, replace lines 21-24 (the `EndianLiteral = Literal["little", "big"]` and `ENDIAN: Final = ("little", "big")` block) with:

```python
from zarr_metadata import ENDIAN, Endian as EndianLiteral
```

(Keep the name `EndianLiteral` so existing `src/` and test imports at `core/buffer/core.py:24` etc. are unchanged. `ENDIAN` keeps its name.)

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_codecs/test_bytes.py -v && uv run mypy src/zarr/codecs/bytes.py src/zarr/core/buffer/core.py`
Expected: PASS, no mypy errors.

- [ ] **Step 5: Commit**

```bash
git add src/zarr/codecs/bytes.py tests/test_codecs/test_bytes.py
git commit -m "refactor(codecs): source bytes Endian pair from zarr-metadata"
```

---

### Task 4: Re-export Blosc pairs in zarr-python from zarr-metadata

**Files:**
- Modify: `src/zarr/codecs/blosc.py:24-32`
- Test: `tests/test_codecs/test_blosc.py`

**Interfaces:**
- Consumes: `from zarr_metadata import BloscShuffle, BLOSC_SHUFFLE, BloscCName, BLOSC_CNAME` (Task 2).
- Produces: `zarr.codecs.blosc.BloscShuffleLiteral`, `BLOSC_SHUFFLE`, `BloscCnameLiteral`, `BLOSC_CNAME` importable with identical members.

- [ ] **Step 1: Write the failing equivalence test**

Add to `tests/test_codecs/test_blosc.py`:

```python
def test_blosc_pairs_sourced_from_zarr_metadata() -> None:
    from typing import get_args

    import zarr_metadata as zm
    from zarr.codecs.blosc import BLOSC_CNAME, BLOSC_SHUFFLE, BloscCnameLiteral, BloscShuffleLiteral

    assert BLOSC_SHUFFLE == zm.BLOSC_SHUFFLE
    assert BLOSC_CNAME == zm.BLOSC_CNAME
    assert set(BLOSC_SHUFFLE) == set(get_args(BloscShuffleLiteral))
    assert set(BLOSC_CNAME) == set(get_args(BloscCnameLiteral))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_codecs/test_blosc.py::test_blosc_pairs_sourced_from_zarr_metadata -v`
Expected: FAIL or, if values already match, proceed to single-source them.

- [ ] **Step 3: Replace local definitions with re-exports**

In `src/zarr/codecs/blosc.py`, replace lines 24-32 (the `BloscShuffleLiteral`, `BLOSC_SHUFFLE`, `BloscCnameLiteral`, `BLOSC_CNAME` block) with:

```python
from zarr_metadata import (
    BLOSC_CNAME,
    BLOSC_SHUFFLE,
    BloscCName as BloscCnameLiteral,
    BloscShuffle as BloscShuffleLiteral,
)
```

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_codecs/test_blosc.py -v && uv run mypy src/zarr/codecs/blosc.py`
Expected: PASS, no mypy errors.

- [ ] **Step 5: Commit**

```bash
git add src/zarr/codecs/blosc.py tests/test_codecs/test_blosc.py
git commit -m "refactor(codecs): source blosc cname/shuffle pairs from zarr-metadata"
```

---

### Task 5: Re-export IndexLocation pair in zarr-python from zarr-metadata

**Files:**
- Modify: `src/zarr/codecs/sharding.py:76-79`
- Test: `tests/test_codecs/test_sharding.py`

**Interfaces:**
- Consumes: `from zarr_metadata import IndexLocation, INDEX_LOCATION` (Task 2).
- Produces: `zarr.codecs.sharding.IndexLocation`, `INDEX_LOCATION` unchanged names/members. `SubchunkWriteOrder`/`SUBCHUNK_WRITE_ORDER` remain locally defined (no change).

- [ ] **Step 1: Write the failing equivalence test**

Add to `tests/test_codecs/test_sharding.py`:

```python
def test_index_location_pair_sourced_from_zarr_metadata() -> None:
    from typing import get_args

    import zarr_metadata as zm
    from zarr.codecs.sharding import INDEX_LOCATION, IndexLocation

    assert INDEX_LOCATION == zm.INDEX_LOCATION
    assert set(INDEX_LOCATION) == set(get_args(IndexLocation))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_codecs/test_sharding.py::test_index_location_pair_sourced_from_zarr_metadata -v`
Expected: FAIL or values-match; proceed.

- [ ] **Step 3: Replace local IndexLocation/INDEX_LOCATION with re-export**

In `src/zarr/codecs/sharding.py`, replace lines 76-79 (the `IndexLocation = Literal["start", "end"]` and `INDEX_LOCATION: Final = ("start", "end")` block) with:

```python
from zarr_metadata import INDEX_LOCATION, IndexLocation
```

Leave the `SubchunkWriteOrder` / `SUBCHUNK_WRITE_ORDER` block (lines 91-97) untouched.

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_codecs/test_sharding.py -v && uv run mypy src/zarr/codecs/sharding.py src/zarr/core/array.py`
Expected: PASS, no mypy errors.

- [ ] **Step 5: Commit**

```bash
git add src/zarr/codecs/sharding.py tests/test_codecs/test_sharding.py
git commit -m "refactor(codecs): source sharding IndexLocation pair from zarr-metadata"
```

---

### Task 6: Source DateTimeUnit/DATETIME_UNIT from zarr-metadata and remove get_args from src

**Files:**
- Modify: `src/zarr/core/dtype/npy/common.py:36-55` (replace pair with re-export)
- Modify: `src/zarr/core/dtype/npy/time.py:234-235` (replace `get_args(DateTimeUnit)` with `DATETIME_UNIT`)
- Test: `tests/test_dtype/test_npy/test_time.py`

**Interfaces:**
- Consumes: `from zarr_metadata import DateTimeUnit, DATETIME_UNIT` (Task 2).
- Produces: `zarr.core.dtype.npy.common.DateTimeUnit`, `DATETIME_UNIT` unchanged names/members; `time.py` validation uses `DATETIME_UNIT` (a correctly-typed `tuple`), not `get_args`.

- [ ] **Step 1: Write the failing test (validation + no get_args in src)**

Add to `tests/test_dtype/test_npy/test_time.py`:

```python
def test_datetime_unit_pair_sourced_and_validates() -> None:
    from typing import get_args

    import zarr_metadata as zm
    from zarr.core.dtype.npy.common import DATETIME_UNIT, DateTimeUnit

    assert DATETIME_UNIT == zm.DATETIME_UNIT
    assert set(DATETIME_UNIT) == set(get_args(DateTimeUnit))


def test_time_module_has_no_get_args() -> None:
    import pathlib

    import zarr.core.dtype.npy.time as time_mod

    src = pathlib.Path(time_mod.__file__).read_text()
    assert "get_args(" not in src, "src/ must not call get_args; use DATETIME_UNIT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dtype/test_npy/test_time.py::test_time_module_has_no_get_args -v`
Expected: FAIL — `time.py` still calls `get_args(DateTimeUnit)` at lines 234-235.

- [ ] **Step 3: Replace the pair with a re-export**

In `src/zarr/core/dtype/npy/common.py`, replace lines 36-55 (the `DateTimeUnit = Literal[...]` and `DATETIME_UNIT: Final = (...)` blocks) with:

```python
from zarr_metadata import DATETIME_UNIT, DateTimeUnit
```

(Place this import with the other top-of-module imports; remove the now-unused `Literal`/`Final` only if nothing else in the file uses them — check first.)

- [ ] **Step 4: Replace get_args with the constant in time.py**

In `src/zarr/core/dtype/npy/time.py`, change lines 234-235 from:

```python
        if self.unit not in get_args(DateTimeUnit):
            raise ValueError(f"unit must be one of {get_args(DateTimeUnit)}, got {self.unit!r}.")
```

to:

```python
        if self.unit not in DATETIME_UNIT:
            raise ValueError(f"unit must be one of {DATETIME_UNIT}, got {self.unit!r}.")
```

Update the import in `time.py` so `DATETIME_UNIT` is imported (from `zarr.core.dtype.npy.common`) and remove the now-unused `get_args` import if nothing else uses it.

- [ ] **Step 5: Run tests + mypy**

Run: `uv run pytest tests/test_dtype/test_npy/test_time.py -v && uv run mypy src/zarr/core/dtype/npy/time.py src/zarr/core/dtype/npy/common.py`
Expected: PASS, no mypy errors.

- [ ] **Step 6: Commit**

```bash
git add src/zarr/core/dtype/npy/common.py src/zarr/core/dtype/npy/time.py tests/test_dtype/test_npy/test_time.py
git commit -m "refactor(dtype): source DateTimeUnit from zarr-metadata, drop get_args in time validation"
```

---

### Task 7: Add paired constants for the local operational Literals that lack them

**Files:**
- Modify: `src/zarr/storage/_zip.py:23` (add `ZIP_STORE_ACCESS_MODE`)
- Modify: `src/zarr/codecs/cast_value.py:36-44` (add `ROUNDING_MODE`, `OUT_OF_RANGE_MODE`)
- Test: `tests/test_store/test_zip.py`, `tests/test_codecs/test_cast_value.py` (or nearest existing test module for each — verify path exists first)

**Interfaces:**
- Produces: `zarr.storage._zip.ZIP_STORE_ACCESS_MODE: Final`, `zarr.codecs.cast_value.ROUNDING_MODE: Final`, `zarr.codecs.cast_value.OUT_OF_RANGE_MODE: Final`, each a tuple matching its Literal.

- [ ] **Step 1: Write the failing drift tests**

Add to the relevant existing test module for the zip store (verify with `ls tests/test_store/`):

```python
def test_zip_store_access_mode_pair() -> None:
    from typing import get_args

    from zarr.storage._zip import ZIP_STORE_ACCESS_MODE, ZipStoreAccessModeLiteral

    assert set(ZIP_STORE_ACCESS_MODE) == set(get_args(ZipStoreAccessModeLiteral))
```

Add to the cast_value test module (verify path with `ls tests/test_codecs/ | grep cast`):

```python
def test_cast_value_mode_pairs() -> None:
    from typing import get_args

    from zarr.codecs.cast_value import OUT_OF_RANGE_MODE, OutOfRangeMode, ROUNDING_MODE, RoundingMode

    assert set(ROUNDING_MODE) == set(get_args(RoundingMode))
    assert set(OUT_OF_RANGE_MODE) == set(get_args(OutOfRangeMode))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store -k zip_store_access_mode_pair tests/test_codecs -k cast_value_mode_pairs -v`
Expected: FAIL — `ZIP_STORE_ACCESS_MODE`, `ROUNDING_MODE`, `OUT_OF_RANGE_MODE` do not exist.

- [ ] **Step 3: Add the paired constants**

In `src/zarr/storage/_zip.py`, directly after `ZipStoreAccessModeLiteral = Literal["r", "w", "a"]` (line 23), add:

```python
ZIP_STORE_ACCESS_MODE: Final = ("r", "w", "a")
```

(Add `Final` to the `typing` import if absent.)

In `src/zarr/codecs/cast_value.py`, after the `RoundingMode` Literal block (ends line 42) add:

```python
ROUNDING_MODE: Final = (
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
)
```

and after `OutOfRangeMode = Literal["clamp", "wrap"]` (line 44) add:

```python
OUT_OF_RANGE_MODE: Final = ("clamp", "wrap")
```

(Add `Final` to the `typing` import if absent.)

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/test_store -k zip_store_access_mode_pair tests/test_codecs -k cast_value_mode_pairs -v && uv run mypy src/zarr/storage/_zip.py src/zarr/codecs/cast_value.py`
Expected: PASS, no mypy errors.

- [ ] **Step 5: Commit**

```bash
git add src/zarr/storage/_zip.py src/zarr/codecs/cast_value.py tests/test_store tests/test_codecs
git commit -m "feat: add paired runtime constants for zip-store and cast-value Literals"
```

---

### Task 8: Add drift tests for the local dtype-internal pairs (no migration)

**Files:**
- Test: `tests/test_dtype/test_npy/test_common.py`, `tests/test_dtype/test_common.py` (verify paths)

**Interfaces:**
- Consumes: existing local pairs `EndiannessStr`/`ENDIANNESS_STR`, `NumpyEndiannessStr`/`NUMPY_ENDIANNESS_STR`, `SpecialFloatStrings`/`SPECIAL_FLOAT_STRINGS`, `ObjectCodecID`/`OBJECT_CODEC_IDS`.
- Produces: drift guards so these pairs cannot silently diverge.

- [ ] **Step 1: Write the failing drift test**

Add to `tests/test_dtype/test_npy/test_common.py`:

```python
def test_local_dtype_constant_literal_pairs() -> None:
    from typing import get_args

    from zarr.core.dtype.common import (
        ENDIANNESS_STR,
        EndiannessStr,
        OBJECT_CODEC_IDS,
        ObjectCodecID,
        SPECIAL_FLOAT_STRINGS,
        SpecialFloatStrings,
    )
    from zarr.core.dtype.npy.common import NUMPY_ENDIANNESS_STR, NumpyEndiannessStr

    assert set(ENDIANNESS_STR) == set(get_args(EndiannessStr))
    assert set(NUMPY_ENDIANNESS_STR) == set(get_args(NumpyEndiannessStr))
    assert set(SPECIAL_FLOAT_STRINGS) == set(get_args(SpecialFloatStrings))
    assert set(OBJECT_CODEC_IDS) == set(get_args(ObjectCodecID))
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_dtype/test_npy/test_common.py::test_local_dtype_constant_literal_pairs -v`
Expected: PASS immediately (pairs already aligned) — this test is a regression guard, not a behavior change. If any assertion fails, the pair is already out of sync and must be corrected in the same task.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dtype
git commit -m "test(dtype): guard local constant/Literal pairs against drift"
```

---

### Task 9: Full-suite verification + src-wide get_args audit

**Files:**
- Test: whole suite, both packages.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: green build; proof that `src/` contains no `get_args` calls.

- [ ] **Step 1: Assert no get_args anywhere in src**

Run: `grep -rn "get_args(" src/ packages/zarr-metadata/src/ || echo "NONE FOUND"`
Expected: `NONE FOUND` (all remaining `get_args` live under `tests/`).
If any remain, replace each with its paired constant before proceeding.

- [ ] **Step 2: Run mypy on both packages**

Run: `uv run mypy src tests && uv run mypy packages/zarr-metadata`
Expected: `Success: no issues found` for both.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -q`
Expected: all pass (note any pre-existing unrelated failures by comparing to a clean `main` run if needed).

- [ ] **Step 4: Run zarr-metadata's own tests**

Run: `uv run pytest packages/zarr-metadata -q`
Expected: all pass.

- [ ] **Step 5: Commit any final fixups**

```bash
git add -A
git commit -m "test: verify typed-constants sweep across both packages" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- "Expose a runtime constant for enumerable Literals" → Tasks 3–8 cover every enumerable string Literal; non-enumerable/no-consumer cases (`MemoryOrder`, `SeparatorLiteral`) explicitly deferred with rationale in Decisions.
- "Source spec strings from zarr-metadata" → Tasks 1–6.
- "Eliminate ill-typed get_args from src" → Task 6 (the only src call site) + Task 9 audit.
- "TypedDicts get no key-constants" → out of scope by decision (metadata-doc key checks already derive from `__annotations__`); not a task, intentionally.

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to" — each code step shows the exact code. Test modules with unverified paths are flagged with an explicit "verify path first" instruction (Tasks 7, 8).

**Type consistency:** Re-export aliases preserve the original zarr-python names (`EndianLiteral`, `BloscCnameLiteral`, `IndexLocation`, `DateTimeUnit`) so no downstream import site changes. Constant names preserved (`ENDIAN`, `BLOSC_CNAME`, `INDEX_LOCATION`, `DATETIME_UNIT`). New constants follow the plural/SCREAMING convention (`ZIP_STORE_ACCESS_MODE`, `ROUNDING_MODE`, `OUT_OF_RANGE_MODE`).

**Open risk flagged for executor:** Task 3/4/5/6 re-exports assume zarr-python annotations that currently reference the local Literal accept the zarr-metadata Literal identically. Members are identical in every migrated case (verified in inventory), so this holds; if mypy reports a Literal-identity error at a use site, fall back to keeping the local Literal and importing only the constant.
