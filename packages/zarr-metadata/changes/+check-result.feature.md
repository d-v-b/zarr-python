Added `check_*` entry points in `zarr_metadata.rules` returning a
discriminated `Valid[T] | Invalid`, for callers who want a document and
its problems in one value.

`validate_*` returns a problem collection whose emptiness the type
checker cannot connect to the document's validity; `parse_*` moves the
failure into control flow. `check_*` closes the gap: the two arms differ
in a `Literal[bool]` discriminant, so narrowing on `result.valid` gives
the checker `result.document` in one branch and `result.problems` in the
other, and reading the wrong one is a static error rather than a runtime
`None`.

```python
result = check_array_metadata_v3(loaded)
if result.valid:
    store(result.document)      # typed ZarrV3ArrayMetadataJSON
else:
    report(result.problems)     # non-empty tuple of problems
```

Offered alongside the `validate_*` / `is_*` / `parse_*` trios, not in
place of them: the trios mirror the model layer's grammar name-for-name,
and a reader who knows the structural functions should not need a second
shape to get the composition-aware judgment. Prefer `check_*` when the
document is the point, `validate_*` when collecting problems across many
documents, and `parse_*` at a trust boundary where invalid input should
abort.
