"""Lazy view over a zarr Array.

`_LazyArray` is the runtime object returned by `Array.lazy[...]`. It holds
a reference to an underlying array and an `IndexTransform` describing the
deferred selection. Indexing into a `_LazyArray` composes a new transform
without I/O; `.result()` materializes by computing a selection from the
transform and calling the underlying array's existing eager indexing.

Private package: not part of the public zarr API. The leading underscore
in the module name signals this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from zarr.core._transforms import IndexTransform
    from zarr.core.array import Array


@dataclass(frozen=True, slots=True)
class _LazyArray:
    """A lazy view over a zarr Array.

    Indexing into a `_LazyArray` returns a new `_LazyArray` with a composed
    transform; calling `.result()` materializes the view through the
    underlying array's eager indexing path.
    """

    _array: Array[Any]
    _transform: IndexTransform

    @property
    def shape(self) -> tuple[int, ...]:
        return self._transform.domain.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._array.dtype

    @property
    def ndim(self) -> int:
        return self._transform.input_rank

    def __repr__(self) -> str:
        path = getattr(self._array, "path", "<array>")
        return (
            f"<_LazyArray {path} shape={self.shape} dtype={self.dtype} "
            f"domain={self._transform.selection_repr}>"
        )
