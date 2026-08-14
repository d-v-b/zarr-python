Added `zarr_metadata.builder`: incremental, validated construction of v3
array metadata documents over the plain JSON TypedDict shapes.
`ZarrV3ArrayMetadataBuilder` accumulates a
`ZarrV3ArrayMetadataJSONPartial` and returns evolved copies.
`evolve(**kwargs)` types standard fields; PEP 728 checkers also accept
extension fields, while `evolve_extension` works across checkers.
`without` removes keys, properties return `T | UNSET`, `build()` returns
a complete validated document, and `to_partial_json()` returns the
current fragment. Inputs are copied and JSON arrays normalize to tuples.

Composition rules fire after each change once their dependencies are
present. They check fill values, known entity shapes, codec pipelines,
dimension counts, chunk grids, transpose codecs, and sharding. Unknown
entity names are left unjudged; known names must use their canonical
shape. A failing update raises one `MetadataValidationError` containing
all problems found.
