# ArrayIO internal-first: split plan (draft)

**Release N** ships the binding layer as private internals, which `AsyncArray` is built
on. **Release N+1** makes it public, designed against zarrs/zarrista, together with
`Context` scopes.

This is a plan only. Nothing here is implemented yet.

## Why

Of the 32 fix commits on #419, and nearly all of #420's churn, most addressed the new
public surface: binding subclasses, the `with_*` derivations, the builders, `verify`,
and Context scopes. The parts that fix things for users today need none of that:

- the stored-document cache, which replaces the `_stored_document` mark;
- `AsyncArray` running on one binding.

Shipping only those removes most of the review surface. It also gives the binding a
release of real use through every `AsyncArray` caller before its API is frozen.

## What `AsyncArray` uses from the binding today

From #419's `array.py` (its tip), `AsyncArray` and `Array` use only these binding
members:

- selection IO: `getitem`, `_get_selection`, `_set_selection`;
- state: `metadata_legacy`, `store_path`, `config`, `_source`, `_pipeline_class`,
  `_chunk_grid`, `_resolved_codec_pipeline`;
- writing metadata: `store_metadata`;
- making and changing bindings: `_bind`, `_detached`, `_rebound_like`;
- pickling the cache entry: `_carried_cache_entry`, `_restore_cache_entry`.

Nothing else is called from outside the binding classes.

## Release N: one internal-only PR, cut from #419

**Keep:**

- `AsyncArray` wrapping one binding (`_io`): the `_BindingField` descriptors,
  copy-on-write (`_own_binding`), `_rebound_like` (which handles `AsyncArray` subclass
  `__init__`s), and the field-state pickles that load across versions in both
  directions.
- `zarr/core/metadata/stored.py` (the per-store cache), the `repair.py` changes, and
  the cache hooks in `group.py` and `api/asynchronous.py` (`observe_read`,
  `copies_from_documents`), plus `_store_repaired_document`.
- `tools/compat/`, the compatibility differential against a baseline (main, v3.4.2).
  It is the breakage gate for this PR.
- The tests that assert `AsyncArray`/`Array` behavior, the cache, repairs and pickles.
  Many of them build through `ArrayIO` for convenience; rewrite those to the private
  names or to the legacy API.

**Hide:**

- Rename `AsyncArrayIO` to a private name (for example `_ArrayBinding`). Remove
  `ArrayIO`/`AsyncArrayIO` from `zarr/__init__.py` and `docs/api/zarr/array.md`.
- Remove the `AsyncArray.io` and `Array.io` properties; `_io` already exists.

**Park for N+1** (on a branch, not deleted):

- The sync `ArrayIO` class, `build_v2`/`build_v3`, `open`, `verify`, `erase_*`,
  `with_metadata`/`with_store`/`with_config`, and the binding's `to_array`, `__eq__`
  and `__hash__`.
- The module-level `build_array_metadata_v2`/`v3`, unless the legacy create path
  adopts them.
- Binding subclass support beyond what `AsyncArray` needs: `binding_class` in pickles,
  and `_bind` documented as a hook.
- The zarr-metadata model as a binding source: the `.metadata` model property,
  `_array_metadata_model`, `_checked_model`, `_model_of_legacy`,
  `_bound_metadata_dict` and `_differing_fields`. Without a model source the
  first-resolution lock has nothing to guard, so it goes too.
- The `zarr-metadata` dependency and the `typing-extensions` 4.16 floor. Nothing in
  release N needs them: `stored.py` imports zarr-metadata for type annotations only.
- The tests of that surface, and #419's changelog fragment. Release N gets a short
  fragment for what users see: a repair that moves chunks is recorded per store,
  survives pickling, and covers consolidated groups.

**Gate:**

- the full suite;
- `tools/compat` against main and v3.4.2 (2000 seeds);
- one breakage review.

The new-API and graft review lenses have nothing to review in this PR.

## Release N+1: the public API

Rebase the parked public surface onto release N, then #420 (Context scopes) onto
that. Design it against zarrs/zarrista, using what release N taught.
`proposals/context-scopes-parked.md` lists #420's open findings. The property tests
in `tests/test_context_properties.py` are the entry gate. Extend them before fixing
anything.

## How the fork PRs map

| Now | Becomes |
|---|---|
| #419 (58 commits) | A new internal-only PR, squashed to one or two commits; the public remainder parked on a branch |
| #420 (Context) | Parked, rebased onto the parked public branch later |
| Fork #420 head 7d55d8eba (force push denied) | Moot until N+1 |

## Decisions for the maintainer

1. Drop `zarr-metadata` from release N? Recommended: yes. Nothing internal needs it,
   and dropping it removes the conda-forge wait and the `typing-extensions` floor bump.
2. Do `build_array_metadata_v2`/`v3` stay public in release N, or wait for N+1?
3. Squash #419's history for the internal PR, or keep its fix commits?

## Measure before committing to it

- Size: how much of #419's ~1,900-line `array.py` diff the internal PR keeps.
  `AsyncArrayIO` is about 800 lines and `ArrayIO` about 170 on #419's tip; the kept
  part is an estimate until the cut is made.
- Whether any internal path still reaches a parked member (the call list above came
  from a grep of #419's tip).
