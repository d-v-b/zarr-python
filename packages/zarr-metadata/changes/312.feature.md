Added `zarr_metadata.msgspec`, an optional msgspec integration module: field
types over the core metadata models plus a `dec_hook` (and a `make_dec_hook`
composer for applications with hooks of their own) that route raw documents
through the models' strict `from_json` parser, so the models can be used as
field types in `msgspec.Struct` classes and with `msgspec.json.decode` /
`msgspec.convert`. Invalid documents surface as `msgspec.ValidationError`
with the loc-annotated problem messages. msgspec stays an optional
dependency of `zarr-metadata`; the module requires msgspec 0.19 or newer.
