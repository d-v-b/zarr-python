"""Helpers shared by backends that hold a flat set of keys."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_storage._keys import is_within, relativize
from zarr_storage._types import Key, Prefix

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from zarr_storage._types import ListItem

__all__ = ["children", "strictly_within"]


def strictly_within(path: str, directory: str) -> bool:
    """Whether `path` lies under `directory` (normalized) and is not `directory` itself."""
    return path != directory and is_within(path, directory)


def children(paths: Iterable[str], directory: str) -> Iterator[ListItem]:
    """The direct children of `directory` among a flat set of key paths.

    A path directly inside `directory` is a `Key`; the first segment of a deeper path is a
    `Prefix`, reported once. A name can be both.
    """
    seen_prefixes: set[str] = set()
    for path in paths:
        if not strictly_within(path, directory):
            continue
        name, sep, _ = relativize(path, directory).partition("/")
        if not sep:
            yield Key(name)
        elif name not in seen_prefixes:
            seen_prefixes.add(name)
            yield Prefix(name)
