# zarr-chunk-key-encoding

Chunk key encodings for Zarr version 3 arrays.

Documentation: <https://zarr-chunk-key-encoding.readthedocs.io/>

A chunk key encoding maps the grid index of a chunk — a tuple of
non-negative integers — to the string key under which that chunk is stored,
and (where well-defined) back again. This package provides:

- `ChunkKeyEncoding` — the abstract base class
- `DefaultChunkKeyEncoding`, `V2ChunkKeyEncoding` — the two encodings defined
  by the [Zarr v3 core spec](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding)
- `BoundedChunkKeyEncoding` (via `ChunkKeyEncoding.to_bounded`) — an encoding
  restricted to a known chunk grid, whose finite key set supports membership
  testing, iteration, and `len`
- `chunk_key_encoding_from_json` / `parse_chunk_key_encoding` — construct
  encodings from JSON metadata or looser user input

Everything public is importable from the top-level `zarr_chunk_key_encoding`
namespace, and its `__all__` is the whole API. Every module beneath it is
named with a leading underscore: submodule paths are implementation detail
and may be reorganized without a major version.

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
- `encode` returns `ChunkKey`, a `typing.NewType` brand over `str` — the
  static analogue of the validated `StoreKey` newtype in `zarrs` — so
  key-consuming code can require proof that a string came out of an encoding.
  `decode` deliberately accepts plain `str`, since its job is judging
  untrusted input.
- All errors derive from `ChunkKeyEncodingError`.
- JSON metadata is resolved by `name` against the closed set of
  spec-defined encodings; see below.

## Installation

```bash
pip install zarr-chunk-key-encoding
```

## Binding to a chunk grid

A plain encoding maps *any* tuple of non-negative integers to a key. When an
array's chunk grid shape is known, `to_bounded` restricts the domain to the grid's
valid indices, making the key set a first-class finite collection:

```python
>>> from zarr_chunk_key_encoding import DefaultChunkKeyEncoding
>>> bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
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

A bound encoding is not a Zarr v3 metadata extension — it has no `name`, and
an array's `chunk_key_encoding` field cannot hold one — but it does have a
JSON form of its own, so a consumer can persist or transmit the bound object
as a unit:

```python
>>> bounded.to_json()
{'grid_shape': [2, 3], 'chunk_key_encoding': {'name': 'default', 'configuration': {'separator': '/'}}}
>>> BoundedChunkKeyEncoding.from_json(bounded.to_json()) == bounded
True
```

## Extensibility

Chunk key encoding is a Zarr v3 *extension point*, so the set of encodings is
open-ended in principle. This package covers the closed set the core spec
defines, and deliberately provides no registration API and no entry point
group:

```python
>>> from zarr_chunk_key_encoding import CHUNK_KEY_ENCODINGS
>>> sorted(CHUNK_KEY_ENCODINGS)
['default', 'v2']
```

The machinery third-party encodings need — registration, entry-point
discovery, and a notion of which encodings are spec-defined versus
registered in
[zarr-extensions](https://github.com/zarr-developers/zarr-extensions) versus
purely local — is identical for all five v3 extension points. The `zarrs`
Rust implementation factors exactly that into a shared
[zarrs_plugin](https://docs.rs/zarrs_plugin) crate that each extension-point
crate depends on, and the same layer belongs in one shared Python package
rather than reinvented here. Note too that `zarr` already scans a
`zarr.chunk_key_encoding` entry point group; a second, competing group is a
commitment worth not making early, since a group name is a compatibility
ratchet the moment a third party publishes against it.

Subclassing `ChunkKeyEncoding` works today, and
`chunk_key_encoding_from_json`'s signature is the one an open set would use,
so growing into a registry later is an additive change.

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
