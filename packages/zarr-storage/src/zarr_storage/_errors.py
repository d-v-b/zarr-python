"""The errors a store raises. A missing key is not one of them: reads return `Missing`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "BatchError",
    "InvalidByteRangeError",
    "InvalidKeyError",
    "KeyConflictError",
    "ReadOnlyError",
    "StorageError",
    "StoreClosedError",
    "UnsupportedOperationError",
]


class StorageError(Exception):
    """Base class for every error this package raises."""


class InvalidKeyError(StorageError, ValueError):
    """A key or directory does not follow the key grammar, or names something the backend reserves."""


class InvalidByteRangeError(StorageError, ValueError):
    """A byte range starts past the end of the value it reads."""


class KeyConflictError(StorageError):
    """A key collides with a directory of the same name on a backend that can't hold both.

    Object stores and in-memory stores hold `a` and `a/b` side by side; a filesystem can't.
    """


class ReadOnlyError(StorageError):
    """A write was attempted through a read-only store."""


class StoreClosedError(StorageError):
    """An operation was attempted after the store's backend was closed."""


class UnsupportedOperationError(StorageError, NotImplementedError):
    """The store's backend does not implement the requested operation."""


class BatchError(ExceptionGroup[Exception], StorageError):
    """Some operations of a batch (`get_many`, `put_many`, ...) raised.

    After the first failure a batch starts no new operations; the ones already running finish. The
    group holds every exception those operations raised. `completed` pairs each request that
    finished with its result, and `remaining` holds the requests that failed or never started: a
    write among them may or may not have landed.
    """

    completed: tuple[tuple[object, object], ...]
    remaining: tuple[object, ...]

    def __new__(
        cls,
        message: str,
        exceptions: Sequence[Exception],
        *,
        completed: tuple[tuple[object, object], ...],
        remaining: tuple[object, ...],
    ) -> Self:
        self = super().__new__(cls, message, exceptions)
        self.completed = completed
        self.remaining = remaining
        return self

    def __init__(
        self,
        message: str,
        exceptions: Sequence[Exception],
        *,
        completed: tuple[tuple[object, object], ...],
        remaining: tuple[object, ...],
    ) -> None:
        super().__init__(message, exceptions)

    def derive(self, excs: Sequence[Exception]) -> BatchError:  # pyright: ignore[reportIncompatibleMethodOverride] -- a batch only ever holds `Exception`s
        """The same batch error, holding `excs`; `except*` builds its subgroups with this."""
        return BatchError(self.message, excs, completed=self.completed, remaining=self.remaining)
