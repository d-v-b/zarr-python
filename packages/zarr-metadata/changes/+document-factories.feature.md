Added `create_*` factories in `zarr_metadata.builder` — one per public
document TypedDict (`create_zarr_v3_array_metadata_json`,
`create_zarr_v3_group_metadata_json`, `create_zarr_v3_consolidated_metadata_json`,
`create_zarr_v2_array_metadata_json`, `create_zarr_v2_group_metadata_json`,
`create_zarr_v2_z_array_json`, `create_zarr_v2_z_group_json`,
`create_zarr_v2_consolidated_metadata_json`), each taking
`**kwargs: Unpack[<TypedDict>]`. Unpacking the total TypedDict makes a
missing required key a **static** error at the call site; at runtime each
factory deep-copies its inputs, materializes JSON arrays as tuples, runs
the structural validator and (for v3 arrays) the semantic rules, and
raises one `MetadataValidationError` carrying every problem. The open v3
array/group factories take an `extensions=` mapping for extension fields
(the hatch for type checkers without PEP 728 support); extension names
that shadow standard keys are rejected.

This encodes a package rule — every public *document* TypedDict gets a
factory — enforced by a drift test against the `DOCUMENT_FACTORIES`
registry, so a new document type cannot ship without one. The rule is
deliberately scoped to documents: entity TypedDicts (codec objects,
configurations, ...) are constructed with TypedDict constructor syntax,
which already enforces their shape statically, and no semantic rules
apply to an entity in isolation.

Prefer the factories for one-shot construction (the common case);
`ZarrV3ArrayMetadataBuilder` remains for staged assembly across program
points, where its eager rule firing and cross-call conflict attribution
apply.
