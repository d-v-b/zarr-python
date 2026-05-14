# Deprecate `zarr.create` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deprecate the top-level `create` function (sync and async), removable in 3.3.0, while giving users of the array-creation convenience functions advance notice that their keyword arguments will change.

**Architecture:** Split `zarr.api.asynchronous.create` into a `@deprecated` public wrapper and a private `_create_array_compat` engine. Convenience functions and the sync `create` route through the private engine, so only direct `create` calls warn. Convenience functions gain a conditional `ZarrDeprecationWarning` that fires only when a legacy-only kwarg is passed. Existing internal usages of `create` migrate to `create_array`; tests that exist solely to exercise `create`'s legacy behavior are deleted.

**Tech Stack:** Python, `typing_extensions.deprecated`, `zarr.errors.ZarrDeprecationWarning`, pytest, towncrier.

---

## Reference: spec

Full design: `docs/superpowers/specs/2026-05-14-deprecate-create-design.md`. Read it before starting.

## Reference: legacy-only kwargs

Keyword arguments accepted by `create` but **not** by `create_array`. Passing any of these to a convenience function triggers the kwargs-change warning, and any test that passes one of these to `create` is "create-specific" (delete, not migrate):

```
compressor, synchronizer, cache_metadata, cache_attrs, read_only,
object_codec, dimension_separator, write_empty_chunks, meta_array,
chunk_shape, codecs, path, chunk_store
```

## Reference: pytest config gotcha

`pyproject.toml` sets `filterwarnings = ["error", ...]`. Any `ZarrDeprecationWarning` emitted during a test that does not expect it becomes a **test failure**. This is why incidental `create` usages must be migrated and create-specific tests deleted.

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/zarr/api/asynchronous.py` | async API surface | rename `create` body → `_create_array_compat`; add `@deprecated` `create` wrapper; add `_LEGACY_CREATE_KWARGS` + `_warn_legacy_create_kwargs`; repoint internal callers; wire kwargs warning into the 5 direct convenience fns |
| `src/zarr/api/synchronous.py` | sync API surface | `@deprecated` on `create`; docstring admonition; repoint to `_create_array_compat` |
| `src/zarr/core/array.py` | docstring examples | migrate `zarr.create` / `zarr.api.asynchronous.create` examples to `create_array` |
| `src/zarr/experimental/cache_store.py` | docstring example | migrate `zarr.create` example to `create_array` |
| `tests/test_deprecation.py` (new) | tests for the new deprecation + kwargs warning behavior | create |
| `tests/test_api.py` | API tests | delete create-specific tests; migrate incidental usages |
| `tests/test_array.py` | array tests | delete create-specific tests; migrate incidental usages |
| `tests/test_v2.py` | v2 tests | delete create-specific tests (all use v2 legacy kwargs) |
| `tests/test_attributes.py`, `tests/test_indexing.py`, `tests/test_store/test_core.py`, `tests/test_store/test_logging.py`, `tests/test_regression/scripts/v2.18.py` | misc tests | migrate / delete per call site |
| `changes/3970.removal.md` (new) | changelog fragment | create |

---

## Task 1: Extract `_create_array_compat` private engine

**Files:**
- Modify: `src/zarr/api/asynchronous.py:849-1065` (the `async def create` definition)

- [ ] **Step 1: Rename the function**

In `src/zarr/api/asynchronous.py`, rename `async def create(` (line 849) to `async def _create_array_compat(`. Leave the entire signature and body **unchanged** otherwise. Replace the existing `"""Create an array.` docstring's first line with a short internal note, keeping the Parameters section intact (it is still useful internal documentation):

Change the docstring opening from:
```python
    """Create an array.

    Parameters
```
to:
```python
    """Create an array (private engine for the deprecated public ``create``).

    This carries the full legacy ``create`` signature. Public ``create`` (sync
    and async) and the array-creation convenience functions forward here.

    Parameters
```

- [ ] **Step 2: Run the test suite for this module to confirm it still imports**

Run: `uv run python -c "import zarr.api.asynchronous"`
Expected: no output, exit 0 (the rename alone breaks nothing because callers are updated in Task 2; this just confirms no syntax error).

- [ ] **Step 3: Commit**

```bash
git add src/zarr/api/asynchronous.py
git commit -m "refactor: rename async create body to _create_array_compat"
```

---

## Task 2: Repoint internal callers to `_create_array_compat`

**Files:**
- Modify: `src/zarr/api/asynchronous.py` — lines 619, 1087, 1137, 1178, 1244, 1293 (internal `create` call sites)

After Task 1, `create` no longer exists (it is `_create_array_compat`). Update every internal caller. These line numbers are from the original file; find them by the surrounding code shown.

- [ ] **Step 1: Repoint `_save_array_like` (was line 619)**

Find `z = await create(**kwargs)` and change to:
```python
    z = await _create_array_compat(**kwargs)
```

- [ ] **Step 2: Repoint `empty` (was line 1087)**

Find `return await create(shape=shape, **kwargs)` inside `async def empty` and change to:
```python
    return await _create_array_compat(shape=shape, **kwargs)
```

- [ ] **Step 3: Repoint `full` (was line 1137)**

Find `return await create(shape=shape, fill_value=fill_value, **kwargs)` inside `async def full` and change to:
```python
    return await _create_array_compat(shape=shape, fill_value=fill_value, **kwargs)
```

- [ ] **Step 4: Repoint `ones` (was line 1178)**

Find `return await create(shape=shape, fill_value=1, **kwargs)` inside `async def ones` and change to:
```python
    return await _create_array_compat(shape=shape, fill_value=1, **kwargs)
```

- [ ] **Step 5: Repoint `open_array` (was line 1244)**

Inside `async def open_array`, find:
```python
            return await create(
                store=store_path,
                zarr_format=_zarr_format,
                overwrite=overwrite,
                **kwargs,
            )
```
and change `create` to `_create_array_compat`:
```python
            return await _create_array_compat(
                store=store_path,
                zarr_format=_zarr_format,
                overwrite=overwrite,
                **kwargs,
            )
```

- [ ] **Step 6: Repoint `zeros` (was line 1293)**

Find `return await create(shape=shape, fill_value=0, **kwargs)` inside `async def zeros` and change to:
```python
    return await _create_array_compat(shape=shape, fill_value=0, **kwargs)
```

- [ ] **Step 7: Verify no `create(` calls remain that should be `_create_array_compat`**

Run: `grep -nE "[^_a-zA-Z.]create\(|await create\(" src/zarr/api/asynchronous.py`
Expected: no output (every internal call now uses `_create_array_compat`; the public `create` wrapper does not exist yet — it is added in Task 3).

- [ ] **Step 8: Commit**

```bash
git add src/zarr/api/asynchronous.py
git commit -m "refactor: repoint async internal callers to _create_array_compat"
```

---

## Task 3: Add the deprecated public async `create` wrapper

**Files:**
- Modify: `src/zarr/api/asynchronous.py` — insert immediately above `async def _create_array_compat`

- [ ] **Step 1: Add the wrapper**

Immediately **above** `async def _create_array_compat(` (the function from Task 1), insert a new `@deprecated` public `create`. It keeps the full explicit signature (copied verbatim from `_create_array_compat`'s signature) so IDE help and type-checkers see the real parameters, and forwards every argument by name.

```python
@deprecated(
    "Use zarr.create_array instead.",
    category=ZarrDeprecationWarning,
)
async def create(
    shape: tuple[int, ...] | int,
    *,  # Note: this is a change from v2
    chunks: tuple[int, ...] | int | bool | None = None,
    dtype: ZDTypeLike | None = None,
    compressor: CompressorLike = "auto",
    fill_value: Any | None = DEFAULT_FILL_VALUE,
    order: MemoryOrder | None = None,
    store: StoreLike | None = None,
    synchronizer: Any | None = None,
    overwrite: bool = False,
    path: PathLike | None = None,
    chunk_store: StoreLike | None = None,
    filters: Iterable[dict[str, JSON] | Numcodec] | None = None,
    cache_metadata: bool | None = None,
    cache_attrs: bool | None = None,
    read_only: bool | None = None,
    object_codec: Codec | None = None,
    dimension_separator: Literal[".", "/"] | None = None,
    write_empty_chunks: bool | None = None,
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,
    attributes: dict[str, JSON] | None = None,
    chunk_shape: tuple[int, ...] | int | None = None,
    chunk_key_encoding: (
        ChunkKeyEncoding
        | tuple[Literal["default"], Literal[".", "/"]]
        | tuple[Literal["v2"], Literal[".", "/"]]
        | None
    ) = None,
    codecs: Iterable[Codec | dict[str, JSON]] | None = None,
    dimension_names: DimensionNamesLike = None,
    storage_options: dict[str, Any] | None = None,
    config: ArrayConfigLike | None = None,
    **kwargs: Any,
) -> AnyAsyncArray:
    """Create an array.

    !!! warning "Deprecated"
        `zarr.create` is deprecated since v3.2 and will be removed in v3.3.0.
        Use [`zarr.create_array`][] instead.

    See [`zarr.api.asynchronous._create_array_compat`][] for the full parameter
    documentation.
    """
    return await _create_array_compat(
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        fill_value=fill_value,
        order=order,
        store=store,
        synchronizer=synchronizer,
        overwrite=overwrite,
        path=path,
        chunk_store=chunk_store,
        filters=filters,
        cache_metadata=cache_metadata,
        cache_attrs=cache_attrs,
        read_only=read_only,
        object_codec=object_codec,
        dimension_separator=dimension_separator,
        write_empty_chunks=write_empty_chunks,
        zarr_format=zarr_format,
        meta_array=meta_array,
        attributes=attributes,
        chunk_shape=chunk_shape,
        chunk_key_encoding=chunk_key_encoding,
        codecs=codecs,
        dimension_names=dimension_names,
        storage_options=storage_options,
        config=config,
        **kwargs,
    )
```

- [ ] **Step 2: Verify it imports and warns**

Run:
```bash
uv run python -c "
import asyncio, warnings
import zarr.api.asynchronous as a
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    asyncio.run(a.create(shape=(4,), store={}))
print('warnings:', [x.category.__name__ for x in w])
assert any(x.category.__name__ == 'ZarrDeprecationWarning' for x in w), w
print('OK')
"
```
Expected: prints a list containing `ZarrDeprecationWarning` and `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/zarr/api/asynchronous.py
git commit -m "feat: deprecate zarr.api.asynchronous.create"
```

---

## Task 4: Add the legacy-kwargs warning helper and wire it into the direct convenience functions

**Files:**
- Modify: `src/zarr/api/asynchronous.py` — add helper near top-level constants; edit the 5 *direct* convenience functions: `empty`, `full`, `ones`, `zeros`, `open_array`.

The other convenience functions are covered transitively and need no edit: `open_like` → `open_array`; `empty_like` → `empty`; `full_like` → `full`; `ones_like` → `ones`; `zeros_like` → `zeros`. Wiring the helper into only the 5 direct callers therefore covers all 10 public convenience functions with no double warning.

- [ ] **Step 1: Add the constant and helper**

In `src/zarr/api/asynchronous.py`, after the imports block (before the first function definition), add:

```python
#: Keyword arguments accepted by the deprecated ``create`` but not by
#: ``create_array``. The array-creation convenience functions still accept
#: these in 3.2.x but will drop them in 3.3.0.
_LEGACY_CREATE_KWARGS: frozenset[str] = frozenset(
    {
        "compressor",
        "synchronizer",
        "cache_metadata",
        "cache_attrs",
        "read_only",
        "object_codec",
        "dimension_separator",
        "write_empty_chunks",
        "meta_array",
        "chunk_shape",
        "codecs",
        "path",
        "chunk_store",
    }
)


def _warn_legacy_create_kwargs(kwargs: dict[str, Any]) -> None:
    """Warn if ``kwargs`` contains a keyword argument that ``create_array``
    does not accept. The array-creation convenience functions will change
    their keyword arguments to match ``create_array`` in zarr-python 3.3.0.
    """
    legacy = sorted(_LEGACY_CREATE_KWARGS & kwargs.keys())
    if legacy:
        warnings.warn(
            ZarrDeprecationWarning(
                f"The keyword argument(s) {legacy!r} will not be accepted in "
                f"zarr-python 3.3.0. The array-creation convenience functions "
                f"will change their keyword arguments to match `zarr.create_array`. "
                f"Migrate to `zarr.create_array`."
            ),
            stacklevel=3,
        )
```

`stacklevel=3` points the warning at the user's call site: user → convenience fn → `_warn_legacy_create_kwargs` → `warnings.warn`.

- [ ] **Step 2: Wire into `empty`**

In `async def empty`, the body is currently:
```python
    return await _create_array_compat(shape=shape, **kwargs)
```
Change to:
```python
    _warn_legacy_create_kwargs(kwargs)
    return await _create_array_compat(shape=shape, **kwargs)
```

- [ ] **Step 3: Wire into `full`**

In `async def full`, change:
```python
    return await _create_array_compat(shape=shape, fill_value=fill_value, **kwargs)
```
to:
```python
    _warn_legacy_create_kwargs(kwargs)
    return await _create_array_compat(shape=shape, fill_value=fill_value, **kwargs)
```

- [ ] **Step 4: Wire into `ones`**

In `async def ones`, change:
```python
    return await _create_array_compat(shape=shape, fill_value=1, **kwargs)
```
to:
```python
    _warn_legacy_create_kwargs(kwargs)
    return await _create_array_compat(shape=shape, fill_value=1, **kwargs)
```

- [ ] **Step 5: Wire into `zeros`**

In `async def zeros`, change:
```python
    return await _create_array_compat(shape=shape, fill_value=0, **kwargs)
```
to:
```python
    _warn_legacy_create_kwargs(kwargs)
    return await _create_array_compat(shape=shape, fill_value=0, **kwargs)
```

- [ ] **Step 6: Wire into `open_array`**

In `async def open_array`, find the line `mode = kwargs.pop("mode", None)`. Insert the check **before** it (so the check sees `mode` is not a legacy kwarg anyway, and the check runs once per call regardless of which branch is taken):
```python
    _warn_legacy_create_kwargs(kwargs)
    mode = kwargs.pop("mode", None)
```

- [ ] **Step 7: Verify warning fires only on legacy kwargs**

Run:
```bash
uv run python -c "
import asyncio, warnings
import zarr.api.asynchronous as a

def calls(coro_factory):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        asyncio.run(coro_factory())
    return [x.category.__name__ for x in w]

# clean kwargs -> no ZarrDeprecationWarning
clean = calls(lambda: a.zeros(shape=(4,), store={}, chunks=(2,)))
assert 'ZarrDeprecationWarning' not in clean, clean

# legacy kwarg -> ZarrDeprecationWarning
legacy = calls(lambda: a.zeros(shape=(4,), store={}, compressor=None))
assert 'ZarrDeprecationWarning' in legacy, legacy
print('OK')
"
```
Expected: prints `OK`.

- [ ] **Step 8: Commit**

```bash
git add src/zarr/api/asynchronous.py
git commit -m "feat: warn when convenience functions get legacy create kwargs"
```

---

## Task 5: Deprecate the sync `create`

**Files:**
- Modify: `src/zarr/api/synchronous.py:602-796` (the `def create` definition)

- [ ] **Step 1: Add the decorator and docstring admonition**

In `src/zarr/api/synchronous.py`, find `def create(` at line 602. Add the `@deprecated` decorator directly above it:
```python
@deprecated(
    "Use zarr.create_array instead.",
    category=ZarrDeprecationWarning,
)
def create(
```

Then add the admonition to the docstring. Find the docstring opening:
```python
) -> AnyArray:
    """Create an array.

    Parameters
```
and change to:
```python
) -> AnyArray:
    """Create an array.

    !!! warning "Deprecated"
        `zarr.create` is deprecated since v3.2 and will be removed in v3.3.0.
        Use [`zarr.create_array`][] instead.

    Parameters
```

- [ ] **Step 2: Repoint the call to the private engine**

In the `create` body, find:
```python
    return Array(
        sync(
            async_api.create(
```
and change `async_api.create` to `async_api._create_array_compat`:
```python
    return Array(
        sync(
            async_api._create_array_compat(
```
(The rest of the call — all the `shape=shape, ...` lines — is unchanged.)

- [ ] **Step 3: Verify it warns exactly once**

Run:
```bash
uv run python -c "
import warnings
import zarr
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    zarr.create(shape=(4,), store={})
dep = [x for x in w if x.category.__name__ == 'ZarrDeprecationWarning']
print('ZarrDeprecationWarning count:', len(dep))
assert len(dep) == 1, [str(x.message) for x in dep]
assert 'create_array' in str(dep[0].message)
print('OK')
"
```
Expected: prints `ZarrDeprecationWarning count: 1` and `OK`. (Exactly one — proves no double warning from the sync→async path.)

- [ ] **Step 4: Commit**

```bash
git add src/zarr/api/synchronous.py
git commit -m "feat: deprecate zarr.create"
```

---

## Task 6: Tests for the new deprecation behavior

**Files:**
- Create: `tests/test_deprecation.py`

- [ ] **Step 1: Write the test file**

Create `tests/test_deprecation.py` with the following content:

```python
"""Tests for the deprecation of ``zarr.create`` and the advance-notice warning
emitted by the array-creation convenience functions when given legacy kwargs.
"""

from __future__ import annotations

import warnings

import pytest

import zarr
import zarr.api.asynchronous
from zarr.errors import ZarrDeprecationWarning


def test_sync_create_deprecated() -> None:
    """zarr.create emits exactly one ZarrDeprecationWarning pointing at create_array."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        zarr.create(shape=(4,), store={})
    deprecations = [w for w in record if issubclass(w.category, ZarrDeprecationWarning)]
    assert len(deprecations) == 1
    assert "create_array" in str(deprecations[0].message)


async def test_async_create_deprecated() -> None:
    """zarr.api.asynchronous.create emits exactly one ZarrDeprecationWarning."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        await zarr.api.asynchronous.create(shape=(4,), store={})
    deprecations = [w for w in record if issubclass(w.category, ZarrDeprecationWarning)]
    assert len(deprecations) == 1
    assert "create_array" in str(deprecations[0].message)


@pytest.mark.parametrize("name", ["empty", "zeros", "ones", "full", "open_array"])
def test_convenience_fns_no_create_deprecation_on_clean_kwargs(name: str) -> None:
    """Convenience functions must not emit ZarrDeprecationWarning for clean kwargs."""
    fn = getattr(zarr.api.synchronous, name)
    common = {"store": {}, "shape": (4,), "chunks": (2,)}
    if name == "full":
        common["fill_value"] = 0
    if name == "open_array":
        # open_array creates on miss; supply a mode that allows creation
        common["mode"] = "w"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrDeprecationWarning)
        fn(**common)  # must not raise


@pytest.mark.parametrize("name", ["empty", "zeros", "ones", "full"])
def test_convenience_fns_warn_on_legacy_kwarg(name: str) -> None:
    """Convenience functions warn when passed a legacy-only kwarg (compressor)."""
    fn = getattr(zarr.api.synchronous, name)
    common = {"store": {}, "shape": (4,), "chunks": (2,), "compressor": None}
    if name == "full":
        common["fill_value"] = 0
    with pytest.warns(ZarrDeprecationWarning, match="create_array"):
        fn(**common)


def test_zeros_like_warns_on_legacy_kwarg() -> None:
    """A *_like variant warns transitively (zeros_like -> zeros)."""
    base = zarr.create_array(store={}, shape=(4,), chunks=(2,), dtype="i4")
    with pytest.warns(ZarrDeprecationWarning, match="create_array"):
        zarr.zeros_like(base, store={}, compressor=None)
```

- [ ] **Step 2: Run the new tests**

Run: `uv run pytest tests/test_deprecation.py -v`
Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_deprecation.py
git commit -m "test: cover create deprecation and legacy-kwarg warnings"
```

---

## Task 7: Delete create-specific tests in `tests/test_api.py`

These tests exist solely to exercise `create` or its legacy-only kwargs. They cannot be migrated to `create_array` without losing their purpose, and `create` is being removed in 3.3.0.

**Files:**
- Modify: `tests/test_api.py`

- [ ] **Step 1: Delete `test_create`**

Delete the entire `def test_create(memory_store: Store) -> None:` function (starts at line 57). It tests `create` directly, including the legacy `chunk_shape` kwarg.

- [ ] **Step 2: Delete `test_unimplemented_kwarg_warnings`**

Delete the entire `def test_unimplemented_kwarg_warnings(kwarg_name: str) -> None:` function (and its `@pytest.mark.parametrize` decorator, around line 1554). It tests `create`'s warnings for `synchronizer`, `chunk_store`, `cache_attrs`, `meta_array` — all legacy-only kwargs.

- [ ] **Step 3: Delete `test_v2_without_compressor` and `test_v2_with_v3_compressor`**

Delete both functions (around lines 1438 and 1448). Both call `zarr.create(..., compressor=...)` — `compressor` is a legacy-only kwarg; the equivalent `create_array` parameter is `compressors`, and `create_array` already has its own coverage for compressor handling.

- [ ] **Step 4: Run test_api.py to find remaining create-related failures**

Run: `uv run pytest tests/test_api.py -x -q`
Expected: FAIL — remaining incidental `create` usages still emit `ZarrDeprecationWarning` (handled in Task 8). Note which test fails first.

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py
git commit -m "test: remove create-specific tests from test_api"
```

---

## Task 8: Migrate incidental `create` usages in `tests/test_api.py`

The remaining `create` call sites in `test_api.py` use `create` only as a convenient way to make an array. Migrate each to `zarr.create_array` (or to `zarr.open`/`open_array` where the test is about opening).

**Files:**
- Modify: `tests/test_api.py`

- [ ] **Step 1: Migrate the `write_empty_chunks` RuntimeWarning test (around line 205)**

This test (`test_write_empty_chunks_warns` or similar — the block containing `zarr.create(shape=(10,), dtype="uint8", write_empty_chunks=...)`) tests that `write_empty_chunks` passed to a creation function triggers a `RuntimeWarning`. `write_empty_chunks` is a legacy kwarg. Since `create_array` does not accept `write_empty_chunks`, and the convenience functions now warn `ZarrDeprecationWarning` for it, **this test's `create` arm is create-specific — delete just the `zarr.create(...)` arm** (the `with pytest.warns(RuntimeWarning, ...)` block that wraps `zarr.create`). If the surrounding test function has another arm that still exercises `write_empty_chunks` via `config`, keep that. If deleting the arm leaves the function empty, delete the whole function.

Inspect first:
```bash
sed -n '180,215p' tests/test_api.py
```
Then delete only the `zarr.create`-based `with pytest.warns(RuntimeWarning ...)` block.

- [ ] **Step 2: Migrate `test_open_array_respects_write_empty_chunks_config` (around line 218)**

Find:
```python
    _ = zarr.create(
        store=store,
        path="test_array",
        shape=(10,),
        chunks=(5,),
        dtype="f8",
        fill_value=0.0,
        zarr_format=zarr_format,
    )
```
Replace with (`path` → `name`):
```python
    _ = zarr.create_array(
        store=store,
        name="test_array",
        shape=(10,),
        chunks=(5,),
        dtype="f8",
        fill_value=0.0,
        zarr_format=zarr_format,
    )
```

- [ ] **Step 3: Migrate the `node.path` test (around line 250)**

Find:
```python
        node = create(store=memory_store, path=path, shape=(2,))
```
Replace with (`path` → `name`, add a `dtype` since `create_array` does not default it the same way; `create` defaulted dtype to float64 — keep that):
```python
        node = create_array(store=memory_store, name=path, shape=(2,), dtype="float64")
```

- [ ] **Step 4: Run test_api.py again**

Run: `uv run pytest tests/test_api.py -q`
Expected: PASS. If a test still fails on a `ZarrDeprecationWarning`, inspect that line — there may be a `create` usage missed by the audit; migrate it the same way (incidental → `create_array` with `path`→`name`, or delete if it passes a legacy kwarg).

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py
git commit -m "test: migrate incidental create usages in test_api to create_array"
```

---

## Task 9: Delete create-specific tests and migrate incidental usages in `tests/test_array.py`

**Files:**
- Modify: `tests/test_array.py`

- [ ] **Step 1: Delete `test_create_invalid_v2_arguments` and `test_invalid_v3_arguments`**

Both (around lines 1438 and 1468) parametrize over `**kwargs` and exist to test `create`'s argument validation. Delete both functions including their `@staticmethod` and `@pytest.mark.parametrize` decorators. Inspect the surrounding class first:
```bash
sed -n '1420,1490p' tests/test_array.py
```
If they are the only members of a test class that becomes empty, delete the empty class too.

- [ ] **Step 2: Migrate the `codecs=` usages (lines 463, 475)**

Line 463:
```python
    arr = zarr.create(shape=(100,), chunks=(10,), dtype="i4", codecs=[BytesCodec()])
```
`codecs` is legacy; `create_array`'s equivalent for an explicit byte codec is `serializer`. Replace with:
```python
    arr = zarr.create_array(store={}, shape=(100,), chunks=(10,), dtype="i4", serializer=BytesCodec())
```

Line 475:
```python
    arr = await zarr.api.asynchronous.create(
        shape=(100,), chunks=(10,), dtype="i4", codecs=[BytesCodec()]
    )
```
Replace with:
```python
    arr = await zarr.api.asynchronous.create_array(
        store={}, shape=(100,), chunks=(10,), dtype="i4", serializer=BytesCodec()
    )
```
**Note:** the test at line 475 (`test_nbytes_stored` async variant) asserts exact byte counts (`502`, `702`, `902`). `create_array` may produce different metadata sizes than `create`. After migrating, run the test; if the byte-count asserts fail, update the expected numbers to the actual values produced (the test comment already calls itself "a fragile test"). Do the same for the sync variant if it has byte-count asserts.

- [ ] **Step 3: Migrate line 154 (async create, group-path creation)**

```python
    await zarr.api.asynchronous.create(
        shape=(2, 2), store=store, path="a/b/c/d", zarr_format=zarr_format
    )
```
Replace (`path` → `name`):
```python
    await zarr.api.asynchronous.create_array(
        shape=(2, 2), store=store, name="a/b/c/d", zarr_format=zarr_format, dtype="float64"
    )
```

- [ ] **Step 4: Migrate the plain incidental usages (lines 672, 718, 794, 832, 859, 888, 911, 932)**

Each of these is `zarr.create(shape=..., chunks=..., dtype=..., fill_value=..., store=store, zarr_format=zarr_format[, overwrite=True])`. Every kwarg used is accepted by `create_array` unchanged. For each, replace `zarr.create(` with `zarr.create_array(`. The `store=store` argument is already present in all of them, so no positional-store fix is needed. Example — line 672:
```python
    z = zarr.create(
        shape=105, chunks=10, dtype="i4", fill_value=0, store=store, zarr_format=zarr_format
    )
```
becomes:
```python
    z = zarr.create_array(
        shape=105, chunks=10, dtype="i4", fill_value=0, store=store, zarr_format=zarr_format
    )
```
Apply the identical `create` → `create_array` rename at lines 718, 794, 832, 859, 888, 911, 932.

- [ ] **Step 5: Migrate the `dimension_separator` / `chunk_key_encoding` usages (lines 1719, 1723)**

Line 1719 uses `dimension_separator="/"` (legacy). Line 1723 uses `chunk_key_encoding=("default", ".")` (accepted by `create_array`). Inspect the full `if/else`:
```bash
sed -n '1710,1735p' tests/test_array.py
```
The `dimension_separator="/"` arm: `create_array` expresses this via `chunk_key_encoding`. Replace `dimension_separator="/"` with `chunk_key_encoding=("v2", "/")` for a v2 array, or `chunk_key_encoding=("default", "/")` for v3 — match `src_format`. Since this branch is selected by `src_format`, use:
```python
        src = zarr.create_array(
            store=store,
            shape=(50, 50),
            chunks=(10, 10),
            dtype="float64",
            zarr_format=src_format,
            chunk_key_encoding=("v2", "/") if src_format == 2 else ("default", "/"),
        )
```
And the `else` arm (line 1723) → `zarr.create_array(...)` with `chunk_key_encoding=("default", ".")` unchanged, adding `store=store` (already present) and `dtype`. Confirm the exact original kwargs from the `sed` output and preserve `shape`/`chunks`; add `dtype="float64"` if the original relied on `create`'s default dtype.

- [ ] **Step 6: Migrate line 1770**

```python
    src = zarr.create(
        (100, 10),
        chunks=src_chunks,
        dtype=src_dtype,
        store=store,
        fill_value=src_fill_value,
        attributes=src_attributes,
    )
```
`(100, 10)` is the positional `shape`. `create_array`'s first positional is `store`, so `shape` must be passed as a keyword. Replace:
```python
    src = zarr.create_array(
        store=store,
        shape=(100, 10),
        chunks=src_chunks,
        dtype=src_dtype,
        fill_value=src_fill_value,
        attributes=src_attributes,
    )
```

- [ ] **Step 7: Run test_array.py**

Run: `uv run pytest tests/test_array.py -q`
Expected: PASS. If the `nbytes_stored` byte-count asserts fail (Step 2 note), update expected values to the actual output and re-run. If any other `ZarrDeprecationWarning` failure appears, inspect and migrate that line.

- [ ] **Step 8: Commit**

```bash
git add tests/test_array.py
git commit -m "test: migrate test_array off deprecated create"
```

---

## Task 10: Delete create-specific tests in `tests/test_v2.py`

Every `zarr.create` call in `test_v2.py` passes v2-era legacy kwargs (`compressor`, `filters` with numcodecs configs) and exists to test v2 array creation through the legacy `create` path. `create_array` has its own v2 coverage. These are create-specific.

**Files:**
- Modify: `tests/test_v2.py`

- [ ] **Step 1: Inspect each call site and its enclosing test**

Run:
```bash
sed -n '45,60p;110,135p;235,250p;295,310p' tests/test_v2.py
```
Call sites: lines 51, 116, 131, 240, 300.

- [ ] **Step 2: Decide per test**

For each enclosing test function:
- If the test's **sole purpose** is to exercise v2 creation via legacy `create` kwargs (`compressor=`, `filters=[...config...]`), and `create_array` has equivalent coverage elsewhere — **delete the test function**.
- If the test asserts something about the *resulting array's behavior* (read/write round-trip, fill_value handling) that is not v2-creation-specific — **migrate** it: `zarr.create(...)` → `zarr.create_array(...)`, translating `compressor=X` → `compressors=X` and keeping `filters=` (both `create` and `create_array` accept `filters`), `zarr_format=2`, `dtype`, `shape`, `chunks`, `fill_value`, `store` unchanged.

Apply judgment per function. The lines 240 and 300 sites (`za = zarr.create(shape=(3,), store=array_path, chunks=(2,), fill_value=fill_value, zarr_format=2, dtype=a.dtype)`) use **no legacy kwargs** — these are incidental; migrate them by renaming `zarr.create` → `zarr.create_array`. The lines 51, 116, 131 sites use `compressor=`/`filters=[Delta(...).get_config()]` — if the test is purely about v2 codec round-tripping, delete; otherwise migrate translating `compressor`→`compressors`.

- [ ] **Step 3: Run test_v2.py**

Run: `uv run pytest tests/test_v2.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_v2.py
git commit -m "test: migrate/remove create usages in test_v2"
```

---

## Task 11: Migrate remaining test files

**Files:**
- Modify: `tests/test_attributes.py`, `tests/test_indexing.py`, `tests/test_store/test_core.py`, `tests/test_store/test_logging.py`, `tests/test_regression/scripts/v2.18.py`

- [ ] **Step 1: `tests/test_attributes.py` (lines 44, 58, 69)**

All three are `z = zarr.create(10, store=store, overwrite=True)`. `10` is positional `shape`; `create_array` needs `shape` as a keyword and a `dtype`. Replace each with:
```python
    z = zarr.create_array(store=store, shape=10, dtype="float64", overwrite=True)
```

- [ ] **Step 2: `tests/test_indexing.py` (lines 1986, 2000)**

Line 1986 uses `codecs=[BytesCodec(), BloscCodec()]` (legacy). Inspect:
```bash
sed -n '1980,2002p' tests/test_indexing.py
```
Translate `codecs=[BytesCodec(), BloscCodec()]` → `serializer=BytesCodec(), compressors=BloscCodec()`. `shape` is positional → keyword. Replace:
```python
    arr = zarr.create_array(
        store=store,
        shape=shape,
        chunks=chunks,
        dtype=np.int16,
        fill_value=fill_value,
        serializer=zarr.codecs.BytesCodec(),
        compressors=zarr.codecs.BloscCodec(),
    )
```
Line 2000: `a = zarr.create((10, 10), chunks=chunks)` — no store, no dtype. `create_array` requires `store`. Replace:
```python
    a = zarr.create_array(store={}, shape=(10, 10), chunks=chunks, dtype="float64")
```

- [ ] **Step 3: `tests/test_store/test_core.py` (lines 279, 286)**

Both: `zarr.create((100,), store=..., zarr_format=2, path="a")`. `(100,)` positional shape → keyword; `path` → `name`. Replace line 279:
```python
    zarr.create_array((100,), store=store, zarr_format=2, name="a", dtype="float64")
```
Wait — `create_array`'s first positional is `store`. Use all keywords:
```python
    zarr.create_array(store=store, shape=(100,), zarr_format=2, name="a", dtype="float64")
```
And line 286 likewise with `store=zip_store`.

- [ ] **Step 4: `tests/test_store/test_logging.py` (line 158)**

`arr = zarr.create(shape=(10,), store=wrapped, overwrite=True)`. No legacy kwargs. Replace:
```python
    arr = zarr.create_array(shape=(10,), store=wrapped, overwrite=True, dtype="float64")
```

- [ ] **Step 5: `tests/test_regression/scripts/v2.18.py` (line 32)**

This script calls `zarr.create` with `compressor=`, `dimension_separator=`, `order=`, `filters=` — heavy legacy v2 kwargs. Inspect the whole script:
```bash
sed -n '1,60p' tests/test_regression/scripts/v2.18.py
```
This is a regression script that reproduces a v2.18 array for cross-version comparison. It legitimately needs the legacy v2 creation surface. **Two options — pick based on the script's role:**
- If the script must keep using `create` to faithfully reproduce v2.18 output: wrap the `zarr.create(...)` call in `warnings.catch_warnings()` + `warnings.simplefilter("ignore", ZarrDeprecationWarning)` and add a comment explaining why. Add `import warnings` if absent.
- If `create_array` can faithfully reproduce the same array: migrate, translating `compressor`→`compressors`, `dimension_separator="/"`→`chunk_key_encoding=("v2", <sep>)`.

Default to the **first option** (suppress the warning locally) — regression scripts deliberately exercise old behavior, and the warning suppression is the honest expression of "yes, we know, this is intentional."

- [ ] **Step 6: Run the affected tests**

Run:
```bash
uv run pytest tests/test_attributes.py tests/test_indexing.py tests/test_store/test_core.py tests/test_store/test_logging.py -q
```
Expected: PASS. The regression script under `tests/test_regression/scripts/` is run by the regression test harness — run `uv run pytest tests/test_regression/ -q` if that harness exists and is not network-gated; otherwise note it for the final full-suite run.

- [ ] **Step 7: Commit**

```bash
git add tests/test_attributes.py tests/test_indexing.py tests/test_store/test_core.py tests/test_store/test_logging.py tests/test_regression/scripts/v2.18.py
git commit -m "test: migrate remaining test files off deprecated create"
```

---

## Task 12: Migrate docstring examples in source files

`tests/test_docs.py` executes code examples found in `docs/` **and** in `src/zarr/`. Under `filterwarnings = ["error"]`, a `zarr.create` call in a docstring example fails the doc test.

**Files:**
- Modify: `src/zarr/core/array.py` (lines ~1202, ~1240, ~1734, ~2280, ~3912)
- Modify: `src/zarr/experimental/cache_store.py` (line ~84)

- [ ] **Step 1: `src/zarr/core/array.py` line ~1202 (`nchunks_initialized` example)**

```python
            arr = await zarr.api.asynchronous.create(shape=(10,), chunks=(1,))
```
Replace:
```python
            arr = await zarr.api.asynchronous.create_array(store={}, shape=(10,), chunks=(1,), dtype="float64")
```

- [ ] **Step 2: `src/zarr/core/array.py` line ~1240 (`_nshards_initialized` example)**

```python
            arr = await zarr.api.asynchronous.create(shape=(10,), chunks=(2,))
```
Replace:
```python
            arr = await zarr.api.asynchronous.create_array(store={}, shape=(10,), chunks=(2,), dtype="float64")
```

- [ ] **Step 3: `src/zarr/core/array.py` line ~1734 (`info` example)**

```python
        >>> arr = await zarr.api.asynchronous.create(
        ...     path="array", shape=(3, 4, 5), chunks=(2, 2, 2))
        ... )
```
Replace (`path` → `name`, add `store`/`dtype`; also fix the stray `)`):
```python
        >>> arr = await zarr.api.asynchronous.create_array(
        ...     store={}, name="array", shape=(3, 4, 5), chunks=(2, 2, 2), dtype="float64")
```

- [ ] **Step 4: `src/zarr/core/array.py` line ~2280**

```python
        >>> arr = await zarr.create(shape=(10,), chunks=(2,))
```
Replace:
```python
        >>> arr = await zarr.api.asynchronous.create_array(store={}, shape=(10,), chunks=(2,), dtype="float64")
```

- [ ] **Step 5: `src/zarr/core/array.py` line ~3912**

```python
        >>> arr = zarr.create(shape=(10,), chunks=(2,), dtype="float32")
```
Replace:
```python
        >>> arr = zarr.create_array(store={}, shape=(10,), chunks=(2,), dtype="float32")
```

- [ ] **Step 6: `src/zarr/experimental/cache_store.py` line ~84**

```python
    array = zarr.create(shape=(100,), store=cached_store)
```
Replace:
```python
    array = zarr.create_array(shape=(100,), store=cached_store, dtype="float64")
```

- [ ] **Step 7: Run the doc example tests**

Run: `uv run pytest tests/test_docs.py -q`
Expected: PASS. If a migrated example's printed output (e.g. the `arr.info` block) differs because `create_array` defaults differ from `create` (e.g. zarr format, chunk key encoding), update the expected output block in the docstring to match the actual output.

- [ ] **Step 8: Commit**

```bash
git add src/zarr/core/array.py src/zarr/experimental/cache_store.py
git commit -m "docs: migrate create examples to create_array"
```

---

## Task 13: Changelog fragment

**Files:**
- Create: `changes/3970.removal.md`

> **Note:** `3970` is a placeholder PR number. If the actual PR number is known, use it; otherwise rename the file to match the PR number before merge. towncrier `removal` type matches the existing `changes/3963.removal.md`.

- [ ] **Step 1: Write the fragment**

Create `changes/3970.removal.md`:
```
The top-level ``create`` function (``zarr.create``,
``zarr.api.synchronous.create``, and ``zarr.api.asynchronous.create``) is now
deprecated and will be removed in 3.3.0. Use ``zarr.create_array`` instead.

Relatedly, the array-creation convenience functions (``zarr.zeros``,
``zarr.ones``, ``zarr.empty``, ``zarr.full``, ``zarr.open_array``, and their
``*_like`` variants) will change their keyword arguments in 3.3.0 to match
``zarr.create_array``. Passing a keyword argument that ``create_array`` does
not accept (for example ``compressor``, ``dimension_separator``, or
``chunk_shape``) now emits a ``ZarrDeprecationWarning``.
```

- [ ] **Step 2: Verify towncrier accepts the fragment**

Run: `uv run towncrier build --draft --version 3.3.0`
Expected: the draft output includes the new fragment text under a removals/deprecations heading, no errors.

- [ ] **Step 3: Commit**

```bash
git add changes/3970.removal.md
git commit -m "docs: changelog fragment for create deprecation"
```

---

## Task 14: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full fast test suite**

Run: `uv run pytest tests/test_api.py tests/test_array.py tests/test_v2.py tests/test_attributes.py tests/test_indexing.py tests/test_store/ tests/test_deprecation.py tests/test_docs.py -q`
Expected: all PASS, no `ZarrDeprecationWarning`-as-error failures.

- [ ] **Step 2: Grep for any surviving un-migrated `create` calls**

Run:
```bash
grep -rnE "(^|[^_a-zA-Z.])create\(|zarr\.create\(|async_api\.create\(|asynchronous\.create\(" tests/ src/ --include="*.py" | grep -vE "create_array|create_group|create_hierarchy|_create_array_compat|_create|def create"
```
Expected: the only remaining hits are: the `@deprecated` public `create` definition in `src/zarr/api/asynchronous.py` and `src/zarr/api/synchronous.py`, the new `tests/test_deprecation.py` (which intentionally calls `create`), and any regression script where the warning is intentionally suppressed (Task 11 Step 5). No incidental usages.

- [ ] **Step 3: Run type checking**

Run: `uv run mypy src/zarr/api/asynchronous.py src/zarr/api/synchronous.py`
Expected: no new errors. (`@deprecated` is understood by mypy; the `_create_array_compat` forward should type-check since the wrapper's signature matches.)

- [ ] **Step 4: Run linting**

Run: `uv run ruff check src/zarr/api/ tests/test_deprecation.py && uv run ruff format --check src/zarr/api/ tests/test_deprecation.py`
Expected: no errors.

- [ ] **Step 5: Run the broader suite**

Run: `uv run pytest -q -x` (or the project's standard test command)
Expected: all PASS. Investigate any failure — likely a missed `create` usage in a test file not covered by the audit; migrate it the same way (incidental → `create_array`, create-specific → delete).

- [ ] **Step 6: Final commit (if any fixes were needed in Step 5)**

```bash
git add -A
git commit -m "test: migrate remaining create usages found in full suite"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1-3 = async split + deprecation (spec §1). Task 4 = kwargs helper + wiring (spec §2-3). Task 5 = sync deprecation (spec §1). Task 6 = new tests (spec Testing §1-5). Tasks 7-12 = migrating existing usages (spec Testing §6, expanded per user decisions). Task 13 = changelog (spec §4). Task 14 = full verification (spec Testing §6).
- **User decisions baked in:** async = `@deprecated` wrapper + private `_create_array_compat` engine; legacy-kwarg warning fires only on legacy kwargs; all 10 convenience fns covered (5 direct + 5 transitive); incidental usages migrate to `create_array`; create-specific tests are deleted.
- **Known fragile points flagged inline:** `nbytes_stored` byte-count asserts (Task 9 Step 2), docstring printed-output blocks (Task 12 Step 7), regression script intent (Task 11 Step 5). Each task tells the engineer to inspect and adjust expected values rather than guess.
