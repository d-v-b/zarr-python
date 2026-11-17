"""A Hypothesis state machine that checks a store against a dict model.

Each step reads or writes through a view at a random root, so roots, views and the key grammar are
exercised together with the operations. Half the time a step targets a key the model holds, so
deletes and conditional writes meet present keys, not just absent ones. Generations that a write or
delete retired are kept and replayed as preconditions: a backend must never hand one out again.
After every step the machine checks that listing, reading and metadata agree with the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from zarr_storage._errors import KeyConflictError
from zarr_storage._keys import join
from zarr_storage._types import (
    Deleted,
    Found,
    Generation,
    Key,
    Missing,
    ObjectMetadata,
    Prefix,
    Rejected,
    Unchanged,
    Written,
)
from zarr_storage.testing._api import delete_if_match

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr_storage._types import ListItem
    from zarr_storage.testing._api import StoreAPI

__all__ = ["Capabilities", "StoreMachine"]

SEGMENTS = ["a", "b", "ab"]
ROOT_CHOICES = ["", "a", "a/b", "b"]
KEYS = st.lists(st.sampled_from(SEGMENTS), min_size=1, max_size=3).map("/".join)
ROOTS = st.sampled_from(ROOT_CHOICES)
DIRECTORIES = st.lists(st.sampled_from(SEGMENTS), max_size=2).map("/".join)
VALUES = st.binary(max_size=6)


@dataclass(frozen=True)
class Capabilities:
    """What the backend under test supports; the machine expects refusals for the rest."""

    conditional_put: bool
    conditional_delete: bool
    holds_key_and_directory: bool
    leaves_empty_directories: bool
    """Whether `list_dir` may report a `Prefix` for a directory a delete left empty."""
    unique_generations: bool
    """Whether the backend never hands out a generation twice, even after a delete."""


def _under(path: str, directory: str) -> bool:
    return directory == "" or path.startswith(f"{directory}/")


def _relative(path: str, root: str) -> str:
    return path.removeprefix(f"{root}/") if root else path


class StoreMachine(RuleBasedStateMachine):
    """Operations on one store, compared with a dict from key to value and generation."""

    def __init__(self, make_store: Callable[[], StoreAPI], capabilities: Capabilities) -> None:
        super().__init__()
        self._make_store = make_store
        self._capabilities = capabilities
        self.values: dict[str, bytes] = {}
        self.generations: dict[str, Generation] = {}
        self.retired: list[Generation] = []

    @initialize()
    def open_store(self) -> None:
        self.store = self._make_store()

    # --- model helpers

    def _target(self, data: st.DataObject) -> tuple[str, str]:
        """A `(root, key)` to operate on: half the time a key the model holds."""
        if self.values and data.draw(st.booleans(), label="existing key"):
            path = data.draw(st.sampled_from(sorted(self.values)), label="path")
            roots = [r for r in ROOT_CHOICES if r == "" or path.startswith(f"{r}/")]
            root = data.draw(st.sampled_from(roots), label="root")
            return root, _relative(path, root)
        return data.draw(ROOTS, label="root"), data.draw(KEYS, label="key")

    def _conflicts(self, path: str) -> bool:
        """Whether writing `path` collides with a stored key, on a backend that can't hold both."""
        if self._capabilities.holds_key_and_directory:
            return False
        return any(_under(path, k) or _under(k, path) for k in self.values if k != path)

    def _expect_conflict(self, root: str, key: str, value: bytes) -> None:
        """A write to a path that collides with a stored key raises, whatever its preconditions."""
        try:
            (self.store / root).put_if_match(key, value, Generation(object()))
        except KeyConflictError:
            return
        raise AssertionError(f"put({join(root, key)!r}) should have raised KeyConflictError")

    def _refusal(self, path: str) -> Rejected | Missing:
        """What a conditional write or delete whose precondition fails returns for `path`."""
        current = self.generations.get(path)
        return Missing() if current is None else Rejected(current)

    def _check_refused(self, result: object, path: str) -> None:
        expected = self._refusal(path)
        if isinstance(expected, Missing):
            assert isinstance(result, Missing), result
        else:
            assert result == expected

    def _record(self, path: str, value: bytes, generation: Generation) -> None:
        if self._capabilities.unique_generations:
            assert generation not in self.retired, "a retired generation was handed out again"
        assert generation != self.generations.get(path)
        if path in self.generations:
            self.retired.append(self.generations[path])
        self.values[path] = value
        self.generations[path] = generation

    def _forget(self, path: str) -> None:
        self.retired.append(self.generations.pop(path))
        del self.values[path]

    # --- writes

    @rule(data=st.data(), value=VALUES)
    def put(self, data: st.DataObject, value: bytes) -> None:
        root, key = self._target(data)
        path = join(root, key)
        if self._conflicts(path):
            try:
                (self.store / root).put(key, value)
            except KeyConflictError:
                return
            raise AssertionError(f"put({path!r}) should have raised KeyConflictError")
        result = (self.store / root).put(key, value)
        self._record(path, value, result.generation)

    @rule(data=st.data(), value=VALUES)
    def put_if_absent(self, data: st.DataObject, value: bytes) -> None:
        root, key = self._target(data)
        path = join(root, key)
        if self._conflicts(path):
            try:
                (self.store / root).put_if_absent(key, value)
            except KeyConflictError:
                return
            raise AssertionError(f"put_if_absent({path!r}) should have raised KeyConflictError")
        result = (self.store / root).put_if_absent(key, value)
        if path in self.values:
            assert result == Rejected(self.generations[path])
        else:
            assert isinstance(result, Written)
            self._record(path, value, result.generation)

    @precondition(lambda self: self._capabilities.conditional_put)
    @rule(data=st.data(), value=VALUES, stale=st.booleans())
    def put_if_match(self, data: st.DataObject, value: bytes, stale: bool) -> None:
        root, key = self._target(data)
        path = join(root, key)
        if self._conflicts(path):
            self._expect_conflict(root, key, value)
            return
        current = self.generations.get(path)
        expected = Generation(object()) if stale or current is None else current
        result = (self.store / root).put_if_match(key, value, expected)
        if not stale and current is not None:
            assert isinstance(result, Written)
            self._record(path, value, result.generation)
        else:
            self._check_refused(result, path)

    @precondition(
        lambda self: (
            self._capabilities.conditional_put
            and self._capabilities.unique_generations
            and len(self.retired) > 0
        )
    )
    @rule(data=st.data(), value=VALUES)
    def put_if_match_retired(self, data: st.DataObject, value: bytes) -> None:
        """A generation a write or delete retired never matches again."""
        root, key = self._target(data)
        if self._conflicts(join(root, key)):
            self._expect_conflict(root, key, value)
            return
        retired = data.draw(st.sampled_from(self.retired), label="retired generation")
        result = (self.store / root).put_if_match(key, value, retired)
        self._check_refused(result, join(root, key))

    @rule(data=st.data())
    def delete(self, data: st.DataObject) -> None:
        root, key = self._target(data)
        path = join(root, key)
        result = (self.store / root).delete(key)
        if path in self.values:
            assert result == Deleted(self.generations[path])
            self._forget(path)
        else:
            assert isinstance(result, Missing)

    @precondition(lambda self: self._capabilities.conditional_delete)
    @rule(data=st.data(), stale=st.booleans())
    def delete_if_match(self, data: st.DataObject, stale: bool) -> None:
        root, key = self._target(data)
        path = join(root, key)
        current = self.generations.get(path)
        expected = Generation(object()) if stale or current is None else current
        result = delete_if_match(self.store / root, key, expected)
        if not stale and current is not None:
            assert result == Deleted(current)
            self._forget(path)
        else:
            self._check_refused(result, path)

    @precondition(
        lambda self: (
            self._capabilities.conditional_delete
            and self._capabilities.unique_generations
            and len(self.retired) > 0
        )
    )
    @rule(data=st.data())
    def delete_if_match_retired(self, data: st.DataObject) -> None:
        root, key = self._target(data)
        retired = data.draw(st.sampled_from(self.retired), label="retired generation")
        result = delete_if_match(self.store / root, key, retired)
        self._check_refused(result, join(root, key))

    @rule(root=ROOTS, directory=DIRECTORIES)
    def delete_dir(self, root: str, directory: str) -> None:
        (self.store / root).delete_dir(directory)
        target = join(root, directory)
        for path in [p for p in self.values if p != target and _under(p, target)]:
            self._forget(path)

    # --- reads, checked against the model

    @rule(root=ROOTS, directory=DIRECTORIES)
    def list_view(self, root: str, directory: str) -> None:
        view = self.store / root
        under = join(root, directory)
        expected = {_relative(p, root) for p in self.values if p != under and _under(p, under)}
        listed = list(view.list(directory))
        assert len(listed) == len(set(listed)), "list reported a key twice"
        assert set(listed) == expected

    @rule(root=ROOTS, directory=DIRECTORIES)
    def list_dir_view(self, root: str, directory: str) -> None:
        under = join(root, directory)
        expected: set[ListItem] = set()
        for path in self.values:
            if path == under or not _under(path, under):
                continue
            name, sep, _ = _relative(path, under).partition("/")
            expected.add(Prefix(name) if sep else Key(name))
        listed = list((self.store / root).list_dir(directory))
        assert len(listed) == len(set(listed)), "list_dir reported a child twice"
        found = set(listed)
        if self._capabilities.leaves_empty_directories:
            found -= {item for item in found - expected if isinstance(item, Prefix)}
        assert found == expected

    @invariant()
    def reads_match_the_model(self) -> None:
        if not hasattr(self, "store"):
            return
        assert set(self.store.list()) == set(self.values)
        for path, value in self.values.items():
            generation = self.generations[path]
            result = self.store.get(path)
            assert isinstance(result, Found)
            assert (bytes(result.value), result.generation) == (value, generation)
            meta = self.store.head(path)
            assert isinstance(meta, ObjectMetadata)
            assert (meta.size, meta.generation) == (len(value), generation)
            unchanged = self.store.get_if_changed(path, generation)
            assert isinstance(unchanged, Unchanged)
            assert unchanged.generation == generation

    def teardown(self) -> None:
        if hasattr(self, "store"):
            self.store.close()
