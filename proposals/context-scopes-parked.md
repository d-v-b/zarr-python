# Context scopes: parked findings

This branch (`claude/arrayio-context`, fork PR #420) adds per-binding `Context` scopes
over data types, codecs and the codec pipeline. It is parked for the release that makes
the binding layer (`ArrayIO`/`AsyncArrayIO`) public: the first release ships that layer as
private internals, where a binding takes no scope of its own.

Pick it up by rebasing onto that work, then fixing the findings below. Each one has a
strict-xfail test in `tests/test_context_properties.py`. Fixing a finding makes its
test XPASS and fail, so remove that test's marker in the same commit.

## How the code is organized

Every route that resolves a name under a context goes through one resolver per kind:

- codecs: `_resolve_codec` (JSON to instance) and `_codec_in` (an instance held to a
  scope: refused for a builder, resolved from its JSON for metadata already resolved),
  in `src/zarr/core/metadata/v3.py`;
- data types: `get_data_type_from_json` (JSON to instance) and `_data_type_in`, in
  `src/zarr/core/dtype/__init__.py`;
- metadata entering or leaving a scope: `_resolved_again` in `src/zarr/core/array.py`.

`tests/test_context_properties.py` pins two invariants: every route reaches the same
outcome for a document and a context, and what a context declares stores and reopens
under it with the same classes and data. They replace the probe scripts of 25 review
rounds, which lived outside the repository and were lost. Extend these tests before
fixing a finding, not after.

## Findings

From the review of the single-resolver refactor (`bdb1807c4`).

| | Finding | Test |
|---|---|---|
| M1 | A codec instance already of the scope's class skips the read-back check (`_codec_in` keeps it), so the builder, entering and `with_context` admit what document routes refuse, and the stored array fails to reopen. | `test_instance_routes_resolve_alike` |
| L1 | The codec read-back checks only that `from_dict` of the written JSON does not raise, not which class it builds. | `test_a_codec_whose_json_reads_back_as_another_class_is_refused` |
| L2 | Leaving a scope (`with_context(None)`) re-raises a scope class's `TypeError` bare, where opening the stored document with no context raises `MetadataValidationError`. | `test_leaving_a_scope_reads_as_opening_with_no_context` |
| L3 | `parse_dtype`'s JSON branch catches `ValueError`, which swallows the data type read-back error; the native-dtype fallback then reports "Multiple data type wrappers found". | `test_parse_dtype_reports_a_data_type_scope_that_cannot_read_back` |
| L4 | The branch predates upstream #4506 (short-hand codec names such as `"codecs": ["bytes"]`). On rebase, the expansion belongs in `_resolve_codec`, not `parse_codecs`. | rebase |
| L5 | `_resolve_codec`'s docstring calls it "the one place a codec name is resolved", but `zarr.registry._resolve_codec` (same name, no context) still resolves names for `create_array`/`init_array`. Behavior is right; rename one or reword. | none |

M1 and L1 share a fix: check the finished metadata once, at the end of every route,
by reading its JSON back under the same context and comparing classes. That guards
the invariant itself instead of each path that can break it, and it replaces the
read-back inside `_resolve_codec`.

## Decisions waiting on the maintainer

- `dataclasses.replace(array, config=...)` drops the array's context; `with_config`
  keeps it (roborev 1501, open).
- Make `v3_unstable_dtype_warning` public, so a third-party data type that warns is
  silenced on a scope's re-read like zarr's own (only warnings sent through it are).

## Known and accepted

- A rebind that switches the pipeline class builds the new pipeline on first use, not
  at once: building it eagerly raised inside `__setstate__` and could report a
  committed resize as failed.
- The first-resolution lock is not reset after `fork` (held only for one
  compare-and-assign; `stored.py`'s `_CREATING` lock already behaves the same).
- Scope by pattern (a resolver callable on `Context`, as zarrs' `match_name`) can be
  added later without breaking anything.
- Pre-existing on main: `BytesCodec.from_dict` refuses the registered alias `endian`.
