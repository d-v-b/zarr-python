# API Reference

The public API is re-exported from the top-level `zarr_chunk_key_encoding`
namespace; the per-module pages document where each symbol lives.

| Module | Contents |
| --- | --- |
| [`abc`](abc.md) | The [`ChunkKeyEncoding`][zarr_chunk_key_encoding.abc.ChunkKeyEncoding] abstract base class and JSON input alias |
| [`default`](default.md) | The v3 `default` encoding |
| [`v2`](v2.md) | The v2-compatibility encoding |
| [`bounded`](bounded.md) | [`BoundedChunkKeyEncoding`][zarr_chunk_key_encoding.bounded.BoundedChunkKeyEncoding] — an encoding restricted to a known chunk grid |
| [`registry`](registry.md) | Registration, lookup, and JSON/lenient construction |
| [`support`](support.md) | [`ChunkKeyEncodingSupport`][zarr_chunk_key_encoding.support.ChunkKeyEncodingSupport] — core / extension / custom, for gating |
| [`separator`](separator.md) | The separator literal type and validator |
| [`errors`](errors.md) | The exception hierarchy |
