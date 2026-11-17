"""Helpers shared by backends: preconditions, and listing a flat set of keys.

A precondition is a function of a key's current generation (`None` if the key is absent) that
returns the outcome that refuses the operation, or `None` to let it proceed. A backend writes each
operation once, parameterized by its precondition; the return type follows from the precondition's,
so `put` (with `always`) returns only `Written`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zarr_storage._keys import is_within, relativize
from zarr_storage._types import Key, Missing, Prefix, Rejected

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from zarr_storage._types import Generation, ListItem

__all__ = ["IfMatch", "always", "children", "if_absent", "strictly_within"]


def always(current: Generation | None) -> None:
    """The precondition of an unconditional operation: it always proceeds."""


def if_absent(current: Generation | None) -> Rejected | None:
    """Proceed only if the key holds no value."""
    return None if current is None else Rejected(current)


@dataclass(frozen=True, slots=True)
class IfMatch:
    """Proceed only if the key's value has `generation`."""

    generation: Generation

    def __call__(self, current: Generation | None) -> Rejected | Missing | None:
        if current is None:
            return Missing()
        if current != self.generation:
            return Rejected(current)
        return None


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
