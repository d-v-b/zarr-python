Added `create_*` factories in `zarr_metadata.builder` — one per public
document TypedDict (`create_zarr_v3_array_metadata_json`,
`create_zarr_v3_group_metadata_json`, `create_zarr_v3_consolidated_metadata_json`,
`create_zarr_v2_array_metadata_json`, `create_zarr_v2_group_metadata_json`,
`create_zarr_v2_z_array_json`, `create_zarr_v2_z_group_json`,
`create_zarr_v2_consolidated_metadata_json`), each taking
`**kwargs: Unpack[<TypedDict>]`. Each factory copies and normalizes its
input, runs structural and composition validation, and raises one
`MetadataValidationError` containing all problems. The strict on-disk
`.zarray`/`.zgroup` factories reject `attributes` at runtime, and the v2
consolidated factory checks the `.zmetadata` envelope. The open v3
array/group factories take an `extensions=` mapping for extension fields
(for type checkers without PEP 728 support) and reject names that shadow
standard fields.

`DOCUMENT_FACTORIES` and a drift test ensure every public document
TypedDict has a factory.

Prefer the factories for one-shot construction (the common case);
`ZarrV3ArrayMetadataBuilder` remains for staged assembly across program
points, where its eager rule firing and cross-call conflict attribution
apply.
