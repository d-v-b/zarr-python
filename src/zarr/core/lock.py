from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Literal
import os
from collections import defaultdict
from threading import Lock
from typing import Protocol

import fasteners


class Synchronizer(Protocol):
    """Base class for synchronizers."""

    def __getitem__(self, item: str) -> Lock: ...


class ThreadSynchronizer(Synchronizer):
    """Provides synchronization using thread locks."""

    mutex: Lock
    locks: defaultdict[str, Lock]

    def __init__(self) -> None:
        self.mutex = Lock()
        self.locks = defaultdict(Lock)

    def __getitem__(self, item: str) -> Lock:
        with self.mutex:
            return self.locks[item]

    def __getstate__(self) -> Literal[True]:
        return True

    def __setstate__(self, *args: Any) -> None:
        # reinitialize from scratch
        self.__init__()  # type: ignore[misc]


class ProcessSynchronizer(Synchronizer):
    """Provides synchronization using file locks via the
    `fasteners <https://fasteners.readthedocs.io/en/latest/api/inter_process/>`_
    package.

    Parameters
    ----------
    path : string
        Path to a directory on a file system that is shared by all processes.
        N.B., this should be a *different* path to where you store the array.

    """

    path: str

    def __init__(self, path: str) -> None:
        self.path = path

    def __getitem__(self, item: str) -> fasteners.InterprocessLock:
        path = os.path.join(self.path, item)
        lock = fasteners.InterProcessLock(path)
        return lock

    # pickling and unpickling should be handled automatically
