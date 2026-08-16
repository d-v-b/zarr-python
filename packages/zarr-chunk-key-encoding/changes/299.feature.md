Initial release of `zarr-chunk-key-encoding`: the `ChunkKeyEncoding` abstract
base class, the spec-defined `DefaultChunkKeyEncoding` and
`V2ChunkKeyEncoding`, a name-keyed registry with entry-point discovery
(modeled on the `zarrs` plugin registry), and JSON round-tripping
(`chunk_key_encoding_from_json`, `parse_chunk_key_encoding`) built on the
`zarr-metadata` types.
