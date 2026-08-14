**Breaking:** every `validate_*` function across `zarr_metadata.model` and
`zarr_metadata.rules` now returns `tuple[ValidationProblem, ...]` instead of
`list[ValidationProblem]`, and `MetadataValidationError.problems` is a tuple.
A validation report is a finished record; handing back a mutable list invited
callers to edit it, and an immutable return is the package's default for
JSON-array-shaped data everywhere else. Code that only iterates, indexes, or
compares length is unaffected; code that called `.append()` on a returned
report, or compared against a list literal, must adapt (`list(problems)` if a
mutable copy is genuinely wanted). `MetadataValidationError` still accepts any
sequence of problems when constructed.

Emptiness is now tested explicitly (`len(problems) == 0`) rather than by
truthiness throughout the package.
