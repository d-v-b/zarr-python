"""An explicit-IO layer for Zarr arrays.

`AsyncArrayIO` and `ArrayIO` separate declaring an array from storing it.
Constructing either class performs no IO: an instance binds an array metadata
document to a location in a store, and every read or write of that store is an
explicit method call. In particular, metadata is stored only when
`store_metadata` is called, so chunks can be written before metadata — a
pattern the eager creation API (`zarr.create_array`) cannot express. This
mirrors the array API of `zarrs`, where storing, retrieving and erasing
metadata and chunks are each explicit operations.

The classes live in `zarr.core.array`, where `AsyncArray` wraps an
`AsyncArrayIO` and delegates its metadata and selection IO to it; this module
is their public, experimental home.
"""

from zarr.core.array import ArrayIO, AsyncArrayIO

__all__ = ["ArrayIO", "AsyncArrayIO"]
