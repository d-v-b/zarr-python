from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

from zarr_storage import Store
from zarr_storage.backends import MemoryBackend
from zarr_storage.testing import (
    ConditionalDeleteSuite,
    ConditionalPutSuite,
    ConditionalRaceSuite,
    KeyAndDirectorySuite,
    PickleSuite,
    StoreSuite,
    UniqueGenerationsSuite,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestMemoryBackend(
    StoreSuite,
    UniqueGenerationsSuite,
    ConditionalPutSuite,
    ConditionalDeleteSuite,
    ConditionalRaceSuite,
    KeyAndDirectorySuite,
    PickleSuite,
):
    def make_store(self, directory: Path) -> Store[MemoryBackend]:
        return Store(MemoryBackend())


def test_pickled_copy_is_independent() -> None:
    """A pickled memory store carries a copy of its values; writes to either don't reach the other."""
    store = Store(MemoryBackend())
    store.put("a", b"1")
    copy = pickle.loads(pickle.dumps(store))
    copy.put("b", b"2")
    store.put("c", b"3")
    assert set(store.list()) == {"a", "c"}
    assert set(copy.list()) == {"a", "b"}
