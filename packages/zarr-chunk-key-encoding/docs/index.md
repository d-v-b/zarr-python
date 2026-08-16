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

When the chunk grid shape is known,
[`bind`][zarr_chunk_key_encoding.ChunkKeyEncoding.bind] restricts the
encoding to the grid's valid indices. The resulting
[`BoundedChunkKeyEncoding`][zarr_chunk_key_encoding.BoundedChunkKeyEncoding]
treats the valid key set as a finite collection — membership testing checks
grammar, rank, bounds, and canonical spelling in one test — and its `decode`
is a total inverse of `encode`:

```python
>>> bounded = encoding.bind((2, 3))
>>> "c/1/2" in bounded
True
>>> "c/2/0" in bounded  # out of bounds
False
>>> len(bounded)
6
```

Chunk key encoding is a Zarr v3 *extension point*, so the set of encodings
is open-ended in principle. This package covers the closed set the core spec
defines, and deliberately provides no registration API and no entry point
group:

```python
>>> from zarr_chunk_key_encoding import CHUNK_KEY_ENCODINGS
>>> sorted(CHUNK_KEY_ENCODINGS)
['default', 'v2']
```

The machinery third-party encodings need is identical for all five v3
extension points, so it belongs in one shared package rather than reinvented
here — see [Extensibility](#extensibility) below.

## Design

The package is modeled on how the [zarrs](https://docs.rs/zarrs_chunk_key_encoding)
Rust implementation factors chunk key encodings into a small standalone crate:

- Encodings are identified by the `name` in their v3 metadata and constructed
  from it
  ([`chunk_key_encoding_from_json`][zarr_chunk_key_encoding.chunk_key_encoding_from_json]).
- [`encode`][zarr_chunk_key_encoding.ChunkKeyEncoding.encode] is
  required; [`decode`][zarr_chunk_key_encoding.ChunkKeyEncoding.decode]
  is optional, since an encoding need not be injective.

Where decoding is defined, it is a *strict* inverse of encoding: malformed
or non-canonical keys (`c/01`, `c/-1`, a wrong prefix or separator) raise
[`ChunkKeyDecodeError`][zarr_chunk_key_encoding.ChunkKeyDecodeError]
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

## Extensibility

The `zarrs` Rust implementation puts its plugin registry in a separate
[zarrs_plugin](https://docs.rs/zarrs_plugin) crate that every
extension-point crate depends on, because registration, entry-point
discovery, and the distinction between spec-defined, `zarr-extensions`-registered,
and purely local encodings are the same problem for all five v3 extension
points. The equivalent Python layer belongs in one shared package, so this
one does not provide it.

That distinction matters more here than at most extension points:
`must_understand: false` is not supported for chunk key encodings, so a
reader that meets an unknown one must fail rather than ignore it, which
makes deciding up front which encodings you accept the only graceful option.

`zarr` also already scans a `zarr.chunk_key_encoding` entry point group. A
second, competing group is a commitment worth not making early, since a
group name is a compatibility ratchet the moment a third party publishes
against it.

Subclassing
[`ChunkKeyEncoding`][zarr_chunk_key_encoding.ChunkKeyEncoding] works
today, and `chunk_key_encoding_from_json`'s signature is the one an open set
would use, so growing into a registry later is an additive change.
