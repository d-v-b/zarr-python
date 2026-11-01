"""What zarr knows of the array metadata documents a store holds: a cache over the store.

Whether an array's stored document needs a repair that moves chunks (see
`zarr.core.metadata.repair`) is a fact about the store, not about metadata read from
it. zarr records it here, per store object, from the array documents it reads and
stores, and asks here before it writes chunks. Every binding of one store object
shares what any of them learned, and metadata stays a value that copies and
re-resolves without carrying store state.

The cache observes; it never serves a read. A path it knows nothing about is read
from the store when the answer is needed.
"""

from __future__ import annotations

import copy
import weakref
from functools import partial
from typing import TYPE_CHECKING, Final

from zarr.core._json import buffer_to_json_object
from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON
from zarr.core.metadata.repair import repair_array_document

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr.abc.store import Store
    from zarr.core.buffer import Buffer
    from zarr.core.common import JSON, ZarrFormat
    from zarr.core.metadata.repair import ArrayDocument


def moves_chunks(document: ArrayDocument, zarr_format: ZarrFormat) -> bool:
    """Whether reading `document` needs a repair that moves chunks."""
    _, readings = repair_array_document(document, zarr_format)
    return any(reading.moves_chunks for reading in readings)


class StoredDocuments:
    """The array metadata documents zarr has read from or stored to one store.

    Each array document zarr has seen, by its path and Zarr format (a path may hold a
    document of each), maps to the document as it was stored when reading it needs a
    repair that moves chunks, and to `None` when it needs none or there is no
    document; a document zarr has not seen is unknown. Copies of array documents in
    consolidated metadata are recorded apart, by the array's path and format, only
    when reading them needs such a repair, so a group write can store them as they
    were stored. A document zarr reads to open an array is recorded only once it has
    been read as metadata, so one it could not open is unknown; one a chunk write reads
    is recorded as read.
    """

    def __init__(self) -> None:
        self._arrays: dict[tuple[str, ZarrFormat], ArrayDocument | None] = {}
        self._copies: dict[tuple[str, ZarrFormat], ArrayDocument] = {}

    def knows(self, path: str, zarr_format: ZarrFormat) -> bool:
        """Whether zarr has read or stored the array document of `zarr_format` at `path`."""
        return (path, zarr_format) in self._arrays

    def needing_repair(self, path: str, zarr_format: ZarrFormat) -> ArrayDocument | None:
        """The document of `zarr_format` stored at `path` as it was stored, if reading it
        needs a repair that moves chunks; `None` if it needs none, there is none, or it
        is unknown."""
        return self._arrays.get((path, zarr_format))

    def copy_needing_repair(self, path: str, zarr_format: ZarrFormat) -> ArrayDocument | None:
        """The consolidated copy of the array document of `zarr_format` at `path` as it
        was stored, if reading it needs a repair that moves chunks."""
        return self._copies.get((path, zarr_format))

    def restore(self, path: str, zarr_format: ZarrFormat, document: ArrayDocument | None) -> None:
        """Record `document` as `needing_repair(path, zarr_format)` returned it from
        another store object's cache (a binding carries it when pickled), unless this
        cache already knows the document: what it learned itself is newer."""
        self._arrays.setdefault((path, zarr_format), document)

    def observe(self, path: str, document: ArrayDocument | None, zarr_format: ZarrFormat) -> None:
        """Record that the array document of `zarr_format` at `path` is `document`
        (`None`: none)."""
        if document is not None and moves_chunks(document, zarr_format):
            self._arrays[path, zarr_format] = copy.deepcopy(document)
        else:
            self._arrays[path, zarr_format] = None

    def observe_written(self, path: str, documents: Mapping[str, Buffer | None]) -> None:
        """Record the metadata documents just stored under `path`, by key (`None`:
        deleted), as `store_documents` takes them."""
        if ZARR_JSON in documents:
            zarr_json = documents[ZARR_JSON]
            document = None if zarr_json is None else buffer_to_json_object(zarr_json)
            if document is not None and document.get("node_type") != "array":
                document = None
            self.observe(path, document, 3)
        if ZARRAY_JSON in documents:
            zarray = documents[ZARRAY_JSON]
            if zarray is None:
                self.observe(path, None, 2)
            else:
                zattrs = documents.get(ZATTRS_JSON)
                v2_document: dict[str, JSON] = buffer_to_json_object(zarray)
                v2_document["attributes"] = {} if zattrs is None else buffer_to_json_object(zattrs)
                self.observe(path, v2_document, 2)

    def observe_consolidated(
        self, group_path: str, members: Mapping[str, JSON], zarr_format: ZarrFormat
    ) -> None:
        """Record the member documents of consolidated metadata read from the group at
        `group_path`, by their paths relative to it."""
        prefix = f"{group_path}/" if group_path else ""
        for key, member in members.items():
            if not isinstance(member, dict) or member.get("node_type", "array") != "array":
                continue
            if zarr_format == 2 and "shape" not in member:
                continue  # a v2 group: it has no `shape`
            path = f"{prefix}{key}"
            if moves_chunks(member, zarr_format):
                self._copies[path, zarr_format] = copy.deepcopy(member)
            else:
                self._copies.pop((path, zarr_format), None)


def _forget(key: int, ref: weakref.ref[Store]) -> None:
    entry = _CACHES.get(key)
    if entry is not None and entry[0] is ref:
        del _CACHES[key]


_CACHES: Final[dict[int, tuple[weakref.ref[Store], StoredDocuments]]] = {}
"""The cache of each live store object, by its `id`: stores define `__eq__`, so they
are not hashable, and two equal stores are not one store object."""


def stored_documents(store: Store) -> StoredDocuments:
    """The cache of `store`, shared by every binding of this store object; it is dropped
    with the store."""
    key = id(store)
    entry = _CACHES.get(key)
    if entry is not None and entry[0]() is store:
        return entry[1]
    documents = StoredDocuments()
    _CACHES[key] = (weakref.ref(store, partial(_forget, key)), documents)
    return documents
