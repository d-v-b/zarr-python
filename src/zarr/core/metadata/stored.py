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
import threading
import weakref
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, cast

from zarr.core._json import buffer_to_json_object
from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON
from zarr.core.metadata.repair import may_move_chunks, repair_array_document

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from zarr_metadata import ZarrV2ArrayMetadata, ZarrV3ArrayMetadata

    from zarr.abc.store import Store
    from zarr.core.buffer import Buffer
    from zarr.core.common import JSON, ZarrFormat
    from zarr.core.metadata import ArrayMetadata
    from zarr.core.metadata.repair import ArrayDocument

    type WrittenArrayMetadata = ArrayMetadata | ZarrV2ArrayMetadata | ZarrV3ArrayMetadata


type CarriedEntry = (
    tuple[Literal["document"], ArrayDocument | None] | tuple[Literal["copy"], ArrayDocument]
)
"""What a cache records of one array document, carried in a pickle: the document's
entry, or else a consolidated copy's."""


def moves_chunks(document: ArrayDocument, zarr_format: ZarrFormat) -> bool:
    """Whether reading `document` needs a repair that moves chunks."""
    if not may_move_chunks(document, zarr_format):
        return False
    _, readings = repair_array_document(document, zarr_format)
    return any(reading.moves_chunks for reading in readings)


class StoredDocuments:
    """The array metadata documents zarr has read from or stored to one store.

    Each array document zarr has seen, by its path and Zarr format (a path may hold a
    document of each), maps to the document as it was stored when reading it needs a
    repair that moves chunks. A document that needs none (or its absence) is recorded,
    as `None`, only where it replaces such a document or a consolidated copy that
    needs one; otherwise it is unknown, which means the same, so the cache holds only
    what stands for a repair. Copies of array documents in
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
        """Whether this cache records the array document of `zarr_format` at `path`:
        one that needs a repair that moves chunks, or a valid one that replaced such
        a document or stands against a consolidated copy that needs one. A valid
        document read with nothing to replace is not recorded, so whether it is
        known depends on what was read first; either way it means no repair."""
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
        cache already knows the document: it keeps what it learned itself."""
        self._arrays.setdefault((path, zarr_format), document)

    def restore_copy(self, path: str, zarr_format: ZarrFormat, document: ArrayDocument) -> None:
        """Record a consolidated copy as `copy_needing_repair` returned it from another
        store object's cache, unless this cache already knows the array's document."""
        if not self.knows(path, zarr_format):
            self._copies.setdefault((path, zarr_format), document)

    def observe(self, path: str, document: ArrayDocument | None, zarr_format: ZarrFormat) -> None:
        """Record that the array document of `zarr_format` at `path` is `document`
        (`None`: none)."""
        key = (path, zarr_format)
        if document is not None and moves_chunks(document, zarr_format):
            self._arrays[key] = copy.deepcopy(document)
        elif key in self._arrays or key in self._copies:
            self._arrays[key] = None

    def entries(self, paths: Iterable[str], zarr_format: ZarrFormat) -> dict[str, CarriedEntry]:
        """What this cache records of the array documents at `paths`, to carry to
        another store object's cache (see `restore_entries`)."""
        carried: dict[str, CarriedEntry] = {}
        for path in paths:
            if self.knows(path, zarr_format):
                carried[path] = ("document", self.needing_repair(path, zarr_format))
            elif (copy_entry := self.copy_needing_repair(path, zarr_format)) is not None:
                carried[path] = ("copy", copy_entry)
        return carried

    def restore_entries(self, entries: Mapping[str, CarriedEntry], zarr_format: ZarrFormat) -> None:
        """Record what `entries` carried, keeping what this cache already knows."""
        for path, entry in entries.items():
            if entry[0] == "document":
                self.restore(path, zarr_format, entry[1])
            else:
                self.restore_copy(path, zarr_format, entry[1])

    def observe_written(
        self, path: str, documents: Mapping[str, Buffer | None], written: WrittenArrayMetadata
    ) -> None:
        """Record the array metadata documents of `written` just stored under `path`, by
        key (`None`: deleted), as `store_documents` takes them. They are parsed only
        when this cache knows the path or `written` might move chunks, so a write of
        valid metadata to a path the cache knows nothing about costs no parse."""
        key = (path, written.zarr_format)
        if key not in self._arrays and key not in self._copies and not _may_move(written):
            return
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


def _may_move(written: WrittenArrayMetadata) -> bool:
    """`may_move_chunks` for the document `written` encodes, read from its chunk sizes."""
    from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

    if isinstance(written, ArrayV2Metadata):
        return may_move_chunks({"chunks": list(written.chunks)}, 2)
    if isinstance(written, ArrayV3Metadata):
        return may_move_chunks(
            cast("ArrayDocument", {"chunk_grid": written.chunk_grid.to_dict()}), 3
        )
    if written.zarr_format == 2:
        return may_move_chunks({"chunks": list(written.chunks)}, 2)
    return may_move_chunks(cast("ArrayDocument", {"chunk_grid": written.chunk_grid.json}), 3)


def _forget(key: int, ref: weakref.ref[Store]) -> None:
    entry = _CACHES.get(key)
    if entry is not None and entry[0] is ref:
        del _CACHES[key]


_CREATING: Final = threading.Lock()
"""Held while a cache is created, so two threads first touching one store object share one."""

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
    with _CREATING:
        entry = _CACHES.get(key)
        if entry is not None and entry[0]() is store:
            return entry[1]
        documents = StoredDocuments()
        _CACHES[key] = (weakref.ref(store, partial(_forget, key)), documents)
        return documents
