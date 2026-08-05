Added `zarr_metadata.v3.codec.kind`: codec kind classification. One
branded union per spec pipeline kind over the concrete codec types
(`ArrayArrayCodecMetadata`, `ArrayBytesCodecMetadata`,
`BytesBytesCodecMetadata`, plus `KnownCodecMetadata` and the paired
`*_CODEC_NAMES` constants), and `TypeIs` guards
(`is_array_array_codec`, `is_array_bytes_codec`, `is_bytes_bytes_codec`,
`is_known_codec`) that classify a codec entry by its `name` and narrow it
to the matching union. Unknown codec names answer `False` to every guard,
so extension codecs classify as "unknown kind" rather than being
misassigned. All names are re-exported from `zarr_metadata.v3.codec`.
