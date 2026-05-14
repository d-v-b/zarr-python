# Deprecate `zarr.create` and warn about upcoming kwargs changes

**Date:** 2026-05-14
**Branch:** `deprecate-create`
**Target release:** 3.3.0 (next minor)

## Goal

1. Deprecate the top-level `create` function (both the synchronous
   `zarr.create` / `zarr.api.synchronous.create` and the asynchronous
   `zarr.api.asynchronous.create`). It will be removed in 3.3.0. Users should
   migrate to `create_array`.
2. The array-creation convenience functions that currently route through
   `create` (`empty`, `full`, `ones`, `zeros`, `open_array`, `open_like`, and
   the `*_like` variants `empty_like`, `full_like`, `ones_like`,
   `zeros_like`) are **not** being deprecated, but their keyword-argument
   contract will change in 3.3.0 to match `create_array`. Give users advance
   notice by emitting a `ZarrDeprecationWarning` — but **only** when a caller
   actually passes a legacy keyword argument that `create_array` does not
   accept.

## Background

`zarr.api.synchronous.create` ([synchronous.py:602]) is a thin wrapper that
calls `zarr.api.asynchronous.create` ([asynchronous.py:849]) via `sync()`.

The async `create` is also the shared engine for 10 public convenience
functions in `asynchronous.py`:

- Direct callers: `empty` (1087), `full` (1137), `ones` (1178),
  `zeros` (1293), `open_array` (1244), `open_array`'s create-on-miss path,
  and `open_like` → `open_array`.
- `*_like` variants delegate: `empty_like` → `empty`, `full_like` → `full`,
  `ones_like` → `ones`, `zeros_like` → `zeros`, `open_like` → `open_array`.
- `_save_array_like` internal helper also calls `create` (619) — internal,
  not user-facing, left as-is.

`create` carries a v2-era signature with legacy keyword arguments;
`create_array` has a cleaner v3-oriented signature. They are **not**
drop-in compatible, which is why the convenience functions cannot simply be
re-pointed at `create_array` in this change — instead they keep calling
`create` for now and warn about the future kwarg change.

The codebase already has an established deprecation pattern:
`typing_extensions.deprecated` + `category=ZarrDeprecationWarning` (see
`zarr.tree` at [synchronous.py:352]) plus a `!!! warning "Deprecated"`
docstring admonition. `ZarrDeprecationWarning` lives in `zarr.errors` and
subclasses `DeprecationWarning`.

## Legacy-only keyword arguments

These are parameters accepted by `create` but **not** by `create_array`.
Passing any of them to a convenience function triggers the kwargs-change
warning:

```
compressor, synchronizer, cache_metadata, cache_attrs, read_only,
object_codec, dimension_separator, write_empty_chunks, meta_array,
chunk_shape, codecs, path, chunk_store
```

Notes on the mapping (for the warning message and user migration):

- `compressor` → `compressors`
- `chunk_shape` → `chunks`
- `path` → `name`
- `codecs` → `filters` / `compressors` / `serializer`
- `synchronizer`, `cache_metadata`, `cache_attrs`, `read_only`,
  `object_codec`, `dimension_separator`, `write_empty_chunks`,
  `meta_array`, `chunk_store` — no direct `create_array` equivalent

This list is derived by diffing the two signatures and will be defined once
as a module-level constant so it stays in one place.

## Design

### 1. Deprecate `create` (sync and async)

**`zarr.api.asynchronous.create`** — add the decorator:

```python
@deprecated(
    "Use zarr.create_array instead.",
    category=ZarrDeprecationWarning,
)
async def create(...):
```

`typing_extensions.deprecated` emits a runtime warning of the given category
when the function is called (verified), and also marks it deprecated for
type checkers.

**`zarr.api.synchronous.create`** — add the same decorator. Both modules
already import `deprecated` (from `typing_extensions`) and
`ZarrDeprecationWarning` (from `zarr.errors`), so no new imports are needed.

Add a `!!! warning "Deprecated"` admonition to both docstrings, right under
the `"""Create an array.` summary line, following the `zarr.tree` precedent:

```
!!! warning "Deprecated"
    `zarr.create` is deprecated since v3.2 and will be removed in v3.3.0.
    Use [`zarr.create_array`][] instead.
```

**Double-warning concern:** sync `create` calls async `create`, so a direct
`zarr.create(...)` call would emit the warning twice (once per layer). This
is acceptable and consistent with how the two-layer API behaves elsewhere;
both warnings point at the same replacement. We will not add suppression
plumbing for this. (Confirmed acceptable during brainstorming.)

### 2. Kwargs-change warning helper

Add a single private helper in `zarr.api.asynchronous` (next to the existing
`_warn_write_empty_chunks_kwarg` usage pattern):

```python
_LEGACY_CREATE_KWARGS: frozenset[str] = frozenset({...})  # the 13 names above

def _warn_legacy_create_kwargs(kwargs: dict[str, Any]) -> None:
    legacy = _LEGACY_CREATE_KWARGS & kwargs.keys()
    if legacy:
        warnings.warn(
            ZarrDeprecationWarning(
                f"The keyword argument(s) {sorted(legacy)!r} will not be "
                f"accepted in zarr-python 3.3.0. The array-creation "
                f"convenience functions will change their keyword arguments "
                f"to match `zarr.create_array`. Migrate to `zarr.create_array`."
            ),
            stacklevel=...,
        )
```

The helper is the **only** place this logic lives; each convenience function
calls it.

### 3. Wire the helper into the 10 convenience functions

For each of `empty`, `full`, `ones`, `zeros`, `open_array`, `open_like`,
`empty_like`, `full_like`, `ones_like`, `zeros_like`: call
`_warn_legacy_create_kwargs(kwargs)` on the incoming `**kwargs` before
delegating.

- The 6 direct callers check their own `kwargs`.
- The 4 `*_like` variants build `like_kwargs` and currently delegate to a
  direct caller. To avoid a double warning (once in the `_like` fn, once in
  the delegate), the `*_like` functions do **not** call the helper
  themselves — they delegate to the direct caller, which performs the
  check. Verify by tracing: `empty_like` → `empty`, `full_like` → `full`,
  `ones_like` → `ones`, `zeros_like` → `zeros`, `open_like` → `open_array`.
  All 10 are therefore covered with the helper wired into only the 6 direct
  callers.

  **Decision:** wire the helper into the 6 direct callers only; the 4
  `*_like` variants are covered transitively. This satisfies "all 10
  convenience fns" with no double-warning.

- `open_array` has two internal `create` call sites (the explicit
  create-on-miss branch); the kwargs check happens once at function entry,
  not per call site.

These functions continue to call `create` internally — their behavior is
unchanged in 3.2.x except for the new conditional warning.

### 4. Changelog fragment

Add `changes/<PR>.removal.md` (towncrier `removal` type, matching
`3963.removal.md`). Content covers both: `create` deprecation and the
upcoming convenience-function kwargs change. PR number filled in at PR time;
until then use a placeholder filename and note it.

## Testing

New tests in the existing API test module(s):

1. **`create` deprecation** — `pytest.warns(ZarrDeprecationWarning)` around
   `zarr.create(...)` and `await zarr.api.asynchronous.create(...)`. Assert
   the message mentions `create_array`.
2. **Convenience fn warns on legacy kwarg** — for a representative legacy
   kwarg (e.g. `compressor`), assert `empty`, `zeros`, `ones`, `full`,
   `open_array` each emit `ZarrDeprecationWarning` when it is passed.
3. **Convenience fn silent on clean kwargs** — assert no
   `ZarrDeprecationWarning` is emitted when only `create_array`-compatible
   kwargs (e.g. `shape`, `dtype`, `chunks`) are passed. Use
   `warnings.catch_warnings` + `simplefilter("error", ZarrDeprecationWarning)`
   or `pytest.warns(None)`-style assertion.
4. **`*_like` coverage** — assert at least one `*_like` function (e.g.
   `zeros_like`) warns when given a legacy kwarg, confirming transitive
   coverage.
5. **Existing test suite** — run the full `test_api` modules to confirm no
   regressions from the new warnings (existing tests that call `create` or
   the convenience functions with legacy kwargs may need
   `pytest.warns`/filter updates; fix those as found).

Run with `uv run pytest`.

## Out of scope

- Actually changing the convenience functions' signatures to match
  `create_array` — that is the 3.3.0 work this change announces.
- Removing `create` — also 3.3.0.
- Touching `_save_array_like` or other internal-only `create` callers.
- Any change to `create_array` itself.

## Files touched

- `src/zarr/api/synchronous.py` — decorate `create`, docstring admonition.
- `src/zarr/api/asynchronous.py` — decorate `create`, docstring admonition,
  add `_LEGACY_CREATE_KWARGS` + `_warn_legacy_create_kwargs`, wire into the
  6 direct convenience functions.
- `changes/<PR>.removal.md` — changelog fragment.
- Test module(s) under `tests/` covering the API surface.
