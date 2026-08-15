# zarr-chunk-key-encoding

Chunk key encodings for Zarr version 3 arrays.

Documentation: <https://zarr-chunk-key-encoding.readthedocs.io/>

A chunk key encoding maps the grid index of a chunk — a tuple of
non-negative integers — to the string key under which that chunk is stored,
and (where well-defined) back again. This package provides:

- `ChunkKeyEncoding` — the abstract base class
- `DefaultChunkKeyEncoding`, `V2ChunkKeyEncoding` — the two encodings defined
  by the [Zarr v3 core spec](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding)
- `BoundedChunkKeyEncoding` (via `ChunkKeyEncoding.bind`) — an encoding
  restricted to a known chunk grid, whose finite key set supports membership
  testing, iteration, and `len`
- a name-keyed registry with entry-point discovery, modeled on the plugin
  registry of the [zarrs](https://docs.rs/zarrs_chunk_key_encoding) Rust
  implementation, so third-party encodings are first-class
- `chunk_key_encoding_from_json` / `parse_chunk_key_encoding` — construct
  encodings from JSON metadata or looser user input

JSON shapes are typed by
[zarr-metadata](https://zarr-metadata.readthedocs.io/); this package supplies
the runtime behavior for those types. It does not import `zarr`. It is
developed in the
[zarr-python](https://github.com/zarr-developers/zarr-python) repository and
intended to be consumed by `zarr` and other Zarr tooling.

```python
>>> from zarr_chunk_key_encoding import chunk_key_encoding_from_json
>>> encoding = chunk_key_encoding_from_json({"name": "default", "configuration": {"separator": "/"}})
>>> encoding.encode((1, 23))
'c/1/23'
>>> encoding.decode("c/1/23")
(1, 23)
```

Design notes, relative to the chunk key encoding code inside `zarr`:

- `decode` is a first-class part of the interface (optional for encodings
  that are not injective), and is a strict inverse of `encode`: malformed or
  non-canonical keys (`"c/01"`, `"c/-1"`, a wrong prefix) raise
  `ChunkKeyDecodeError` instead of being silently normalized.
- `encode` validates its input: chunk coordinates must be non-negative
  integers (anything implementing `__index__`, so NumPy integers work), and
  invalid coordinates raise `InvalidChunkCoordsError` rather than producing
  an unusable key.
- All errors derive from `ChunkKeyEncodingError`.
- Following zarrs, registrations can be reversed
  (`unregister_chunk_key_encoding`), and JSON metadata is resolved through
  the registry by `name`, so third-party encodings round-trip through
  `chunk_key_encoding_from_json` exactly like the built-ins.

## Installation

```bash
pip install zarr-chunk-key-encoding
```

## Binding to a chunk grid

A plain encoding maps *any* tuple of non-negative integers to a key. When an
array's chunk grid shape is known, `bind` restricts the domain to the grid's
valid indices, making the key set a first-class finite collection:

```python
>>> from zarr_chunk_key_encoding import DefaultChunkKeyEncoding
>>> bounded = DefaultChunkKeyEncoding().bind((2, 3))
>>> bounded.encode((1, 2))
'c/1/2'
>>> "c/1/2" in bounded
True
>>> "c/2/0" in bounded  # out of bounds
False
>>> "c/01/2" in bounded  # non-canonical spelling
False
>>> len(bounded)
6
```

Membership is the full store-key check — grammar, rank, bounds, and
canonical spelling — in one test, which is exactly what a server validating
candidate chunk keys against an array needs. Bounded `decode` is also a
*total* inverse of `encode`: with the grid rank known, the `v2` encoding's
rank-zero ambiguity (`"0"` is the key for both `()` and `(0,)`) disappears.
For sharded arrays, bind the shard grid shape, since shards are the unit of
storage.

## Registering a custom encoding

Subclass `ChunkKeyEncoding` and register it, either imperatively:

```python
from zarr_chunk_key_encoding import ChunkKeyEncoding, register_chunk_key_encoding


class MyEncoding(ChunkKeyEncoding):
    name = "my_encoding"
    ...


register_chunk_key_encoding(MyEncoding)
```

or declaratively from another package, via the `zarr_chunk_key_encoding`
entry point group:

```toml
[project.entry-points.zarr_chunk_key_encoding]
my_encoding = "my_package:MyEncoding"
```

Entry points are loaded lazily, the first time a name lookup would otherwise
fail.

## Developing

Package-scoped development commands live in the [`justfile`](./justfile)
(requires [just](https://github.com/casey/just)):

```
just test        # run the test suite (extra args go to pytest)
just lint        # ruff, same invocation as CI
just typecheck   # pyright, same invocation as CI
just docs-check  # strict build of the docs site
just check       # all of the above
just docs-serve  # serve the docs site locally
```

Run them from this directory, or from anywhere in the repository as
`just packages/zarr-chunk-key-encoding/<recipe>`.

The test recipe layers the sibling `packages/zarr-metadata` in as an
editable overlay, so changes to the two packages can be developed in
lockstep; the published package depends on the released `zarr-metadata`
distribution.

## License

MIT
