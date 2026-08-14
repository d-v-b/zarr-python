Added `zarr_metadata.v3.codec.kind`: codec kind classification. One
branded union per spec pipeline kind over the concrete codec types
(`ArrayArrayCodecMetadata`, `ArrayBytesCodecMetadata`,
`BytesBytesCodecMetadata`, plus `KnownCodecMetadata` and the paired
`*_CODEC_NAMES` constants). Shape-exact `TypeIs` guards narrow canonical
codec metadata to those unions. `codec_kind_of_name` classifies known
names without validating object shape and returns `None` for unknown
names. All names are re-exported from `zarr_metadata.v3.codec`.
