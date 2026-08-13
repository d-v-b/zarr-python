# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "zarr-http-server @ git+https://github.com/zarr-developers/zarr-python.git@main#subdirectory=packages/zarr-http-server",
#   # 2026.5.2 is the release whose zarr stores speak zarr format 3 and whose
#   # `imread` accepts `return_as`; earlier versions produce a format 2 store
#   # that is not a `zarr.abc.store.Store` at all.
#   "tifffile>=2026.5.2",
#   "fsspec[http]",
#   "httpx",
# ]
# ///
"""
Serve a TIFF file over HTTP as a Zarr hierarchy.

`tifffile` can present a TIFF as a zarr store: `imread(path,
return_as="zarr")` returns a `ZarrTiffStore`, a read-only
`zarr.abc.store.Store` whose keys are the `zarr.json` documents and chunks a
zarr client expects. Chunks are the TIFF's own tiles, read from the file --
and decoded -- only when a key is requested.

Because it is an ordinary `Store`, `zarr-http-server` can serve it directly.
Nothing is converted or copied: the TIFF on disk stays the only copy of the
data, and clients on the other end of the HTTP connection see a normal zarr
hierarchy.

This example writes a small pyramidal OME-TIFF, serves it, and reads it back
both as raw HTTP responses and through a zarr client.
"""

import json
import tempfile
from pathlib import Path

import httpx
import numpy as np
import tifffile
import zarr

from zarr_http_server import serve_background, store_app

# -- write a pyramidal OME-TIFF ---------------------------------------------
# Any TIFF works; a pyramid is used here because it makes the store a *group*
# of arrays -- one per resolution level -- rather than a single array, which
# is the more interesting thing to serve.
tmpdir = tempfile.TemporaryDirectory()
path = Path(tmpdir.name) / "pyramid.ome.tif"

rng = np.random.default_rng(0)
full = rng.integers(0, 255, size=(512, 512), dtype="uint8")

with tifffile.TiffWriter(path, ome=True) as tif:
    # `subifds=2` reserves slots for the two reduced-resolution levels, which
    # are then written with `subfiletype=1` to mark them as such.
    tif.write(full, tile=(128, 128), photometric="minisblack", subifds=2, compression="zlib")
    tif.write(full[::2, ::2], tile=(128, 128), photometric="minisblack", subfiletype=1)
    tif.write(full[::4, ::4], tile=(128, 128), photometric="minisblack", subfiletype=1)

print(f"wrote {path.name}: {path.stat().st_size} bytes")

# -- open it as a zarr store ------------------------------------------------
store = tifffile.imread(path, return_as="zarr")
print(f"store: {type(store).__name__}, read_only={store.read_only}")

# -- serve it ---------------------------------------------------------------
# `store_app` serves every key in the store, which here means the whole TIFF
# and nothing else -- the store's key space is exactly this one file. And the
# store reports `read_only=True`, so writes are refused by the store itself,
# not only by the default read-only method set.
with serve_background(store_app(store), host="127.0.0.1") as server:
    # -- the root is an OME-NGFF multiscales group ---------------------------
    root = httpx.get(f"{server.url}/zarr.json", timeout=30).json()
    assert root["node_type"] == "group"
    datasets = root["attributes"]["ome"]["multiscales"][0]["datasets"]
    print(f"\nresolution levels: {[d['path'] for d in datasets]}")

    # -- a level's metadata, and one of its chunks --------------------------
    level0 = httpx.get(f"{server.url}/0/zarr.json", timeout=30).json()
    print("\n0/zarr.json:")
    print(json.dumps(level0, indent=2))

    chunk = httpx.get(f"{server.url}/0/c/0/0", timeout=30)
    assert chunk.status_code == 200
    # 128 * 128 uint8: the tile arrives decoded. `ZarrTiffStore` decompresses
    # each TIFF tile as it reads it and hands zarr plain bytes, which is why
    # the metadata above lists no codec beyond `bytes`. The compression saved
    # space on disk, not on the wire.
    print(f"\nchunk 0/c/0/0: {len(chunk.content)} bytes")

    # -- read it back with a zarr client ------------------------------------
    # This needs `fsspec[http]`. To the client this is an ordinary remote zarr
    # group; that it is backed by a TIFF is not visible from here.
    remote = zarr.open_group(server.url, mode="r")
    print("\nover HTTP, through zarr:")
    # The level names come from the group's own multiscales metadata rather
    # than from `remote.arrays()`: HTTP exposes no directory listing, so a
    # client reading over HTTP can only open keys it already knows about.
    for dataset in datasets:
        array = remote[dataset["path"]]
        print(f"  {dataset['path']}: shape={array.shape} chunks={array.chunks} dtype={array.dtype}")

    # The pixels survive the round trip: TIFF tile -> HTTP -> zarr chunk.
    assert np.array_equal(remote["0"][:], full)
    assert np.array_equal(remote["2"][:], full[::4, ::4])
    print("\nremote data matches the array that was written")

tmpdir.cleanup()
