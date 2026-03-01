# Serve a Zarr v2 Array as v3 over HTTP

This example demonstrates how to build a custom read-only `Store` that
translates Zarr v2 data into v3 format on the fly, and serve it over HTTP
using `zarr.experimental.serve.serve_store`.

The example shows how to:

- Implement a custom `Store` subclass (`V2AsV3Store`) that wraps an
  existing v2 store
- Translate v2 metadata (`.zarray` + `.zattrs`) to v3 `zarr.json` using
  the same `_convert_array_metadata` helper that powers `zarr migrate v3`
- Pass chunk keys through unchanged (the converted metadata preserves
  `V2ChunkKeyEncoding`, so keys like `0.0` work in both formats)
- Serve the translated store over HTTP so that any v3-compatible client
  can read v2 data without knowing the original format

## Running the Example

```bash
python examples/serve_v2_v3/serve_v2_v3.py
```

Or run with uv:

```bash
uv run examples/serve_v2_v3/serve_v2_v3.py
```
