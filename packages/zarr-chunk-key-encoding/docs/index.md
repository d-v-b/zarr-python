# zarr-chunk-key-encoding

Chunk key encodings for Zarr version 3 arrays.

A chunk key encoding maps the grid index of a chunk — a tuple of
non-negative integers identifying a cell in the chunk grid — to the string
key under which that chunk is stored, and (where well-defined) back again.
The [Zarr v3 core spec](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding)
defines two encodings, both provided here:

- **`default`** — `(1, 23)` ↦ `c/1/23` (separator configurable to `.`)
- **`v2`** — `(1, 23)` ↦ `1.23` (separator configurable to `/`), reproducing
  the chunk layout of Zarr v2 stores

## Quickstart

```python
>>> from zarr_chunk_key_encoding import chunk_key_encoding_from_json
>>> encoding = chunk_key_encoding_from_json({"name": "default", "configuration": {"separator": "/"}})
>>> encoding.encode((1, 23))
'c/1/23'
>>> encoding.decode("c/1/23")
(1, 23)
>>> encoding.to_json()
{'name': 'default', 'configuration': {'separator': '/'}}
```

## Design

The package is modeled on how the [zarrs](https://docs.rs/zarrs_chunk_key_encoding)
Rust implementation factors chunk key encodings into a small standalone crate
with a name-keyed plugin registry:

- Encodings are identified by their registered `name` and constructed from
  Zarr v3 JSON metadata via the registry
  ([`chunk_key_encoding_from_json`][zarr_chunk_key_encoding.registry.chunk_key_encoding_from_json]).
- Third-party encodings register imperatively
  ([`register_chunk_key_encoding`][zarr_chunk_key_encoding.registry.register_chunk_key_encoding])
  or declaratively via the `zarr_chunk_key_encoding` entry point group, and
  registrations can be reversed
  ([`unregister_chunk_key_encoding`][zarr_chunk_key_encoding.registry.unregister_chunk_key_encoding]).
- [`encode`][zarr_chunk_key_encoding.abc.ChunkKeyEncoding.encode] is
  required; [`decode`][zarr_chunk_key_encoding.abc.ChunkKeyEncoding.decode]
  is optional, since an encoding need not be injective.

Where decoding is defined, it is a *strict* inverse of encoding: malformed
or non-canonical keys (`c/01`, `c/-1`, a wrong prefix or separator) raise
[`ChunkKeyDecodeError`][zarr_chunk_key_encoding.errors.ChunkKeyDecodeError]
instead of being silently normalized, and `encode` validates that its input
coordinates are non-negative integers (anything implementing `__index__`,
so NumPy integers work).

JSON shapes are typed by
[zarr-metadata](https://zarr-metadata.readthedocs.io/); this package
supplies the runtime behavior for those types. It does not import `zarr`.

## Installation

```bash
pip install zarr-chunk-key-encoding
```
