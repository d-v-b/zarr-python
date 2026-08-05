Added `zarr_metadata.builder`: incremental, validated construction of v3
array metadata documents over the plain JSON TypedDict shapes.
`ZarrV3ArrayMetadataBuilder` accumulates a
`ZarrV3ArrayMetadataJSONPartial` and hands out evolved copies:
`evolve(**kwargs)` is the single fully-typed setter (unknown keys and
wrong value types are static errors), `evolve_extension` sets extension
fields, `without` unsets keys (absence is UNSET; a stored `None` is JSON
`null`, and the two never convert), per-field properties answer
`T | UNSET`, `build()` returns the validated complete document, and
`to_partial_json()` returns the fragment under an honest partial type —
there is no `build_unchecked`.

Semantic rules live as data (`Rule`, keyed by the document keys they
depend on) and fire eagerly after every change, whenever their
dependencies are all present — so field order is unconstrained, coupled
fields are checked as soon as they coexist, and a conflict names both
fields plus the batch-`evolve` escape hatch. The initial rule set checks
fill_value against the data type (with range checks for ints and the
spec's special float spellings), codec pipeline kind ordering via the new
`TypeIs` guards, `dimension_names` length, and `regular` chunk grid
dimensionality. Rules never reject what they cannot interpret: unknown
data types, codecs, and configurations pass through (extension openness).
All problems from a check are reported together in one
`MetadataValidationError`.
