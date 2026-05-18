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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from zarr.core._transforms.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr.core._transforms.transform import selection_to_transform

if TYPE_CHECKING:
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

    def __getitem__(self, selection: Any) -> _LazyArray:
        """Compose a basic-indexing selection onto the underlying transform.
        Returns a new _LazyArray; no I/O is performed.

        For orthogonal-style array indexing or vectorized indexing, use
        `lazy.oindex[...]` / `lazy.vindex[...]` (helpers added in a follow-up).
        """
        new_transform = selection_to_transform(selection, self._transform, "basic")
        return _LazyArray(_array=self._array, _transform=new_transform)

    def result(self) -> Any:
        """Materialize the lazy view by dispatching through the underlying
        array's eager indexing path.

        Currently supports basic mode only. Orthogonal and vectorized modes
        come in a follow-up commit and currently raise NotImplementedError
        with a clear message naming the unsupported mode.

        Return type is `Any` rather than `np.ndarray` because basic indexing
        with an integer scalar returns a numpy scalar, not an array.
        """
        selection, mode = transform_to_selection(self._transform)
        if mode == "basic":
            return self._array[selection]
        raise NotImplementedError(
            f"_LazyArray.result() does not yet support mode={mode!r}; "
            "oindex/vindex come in a follow-up commit"
        )


# Type alias for the selection that we hand to eager indexing.
_Selection = tuple[Any, ...]  # tuple of int | slice | np.ndarray; loose at runtime
_Mode = Literal["basic", "orthogonal", "vectorized"]


def _classify_mode(t: IndexTransform) -> _Mode:
    """Determine the indexing mode from the structure of t.output.

    Rules:
    - No ArrayMaps → "basic".
    - One or more ArrayMaps with disjoint input_dimensions → "orthogonal".
    - Two or more ArrayMaps sharing at least one input_dim → "vectorized".
    """
    array_maps = [m for m in t.output if isinstance(m, ArrayMap)]
    if len(array_maps) == 0:
        return "basic"
    seen: set[int] = set()
    for m in array_maps:
        dims = set(m.input_dimensions)
        if len(seen & dims) > 0:
            return "vectorized"
        seen.update(dims)
    return "orthogonal"


def transform_to_selection(t: IndexTransform) -> tuple[_Selection, _Mode]:
    """Convert an IndexTransform back into a selection tuple and indexing mode.

    The returned (selection, mode) tuple can be dispatched through zarr's
    existing eager indexing API:
    - mode="basic"      → array[selection]
    - mode="orthogonal" → array.oindex[selection]
    - mode="vectorized" → array.vindex[selection]

    Constraints relied on by this function:
    - `DimensionMap.stride` is positive (enforced by `IndexTransform.__post_init__`).
      Negative-stride DimensionMaps would produce ambiguous `slice(start, stop, -k)`
      because Python interprets the resulting negative `stop` as from-the-end.
    - For the orthogonal path, each `ArrayMap.index_array` must be 1-D (zarr's
      `oindex` expects one 1-D selector per axis). Multi-dimensional ArrayMaps,
      which are legal at the `IndexTransform` level, are rejected here.
    """
    mode = _classify_mode(t)
    entries: list[Any] = []
    for m in t.output:
        if isinstance(m, ConstantMap):
            entries.append(m.offset)
        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            lo = t.domain.inclusive_min[d]
            hi = t.domain.exclusive_max[d]
            start = m.offset + m.stride * lo
            stop = m.offset + m.stride * hi
            if m.stride == 1:
                entries.append(slice(start, stop))
            else:
                entries.append(slice(start, stop, m.stride))
        else:  # ArrayMap
            assert isinstance(m, ArrayMap)
            if mode == "orthogonal" and m.index_array.ndim > 1:
                raise NotImplementedError(
                    "transform_to_selection: multi-dimensional ArrayMap in "
                    "orthogonal mode is not supported; zarr's oindex requires "
                    "1-D per-axis selectors"
                )
            coords = m.offset + m.stride * m.index_array
            entries.append(coords.astype(np.intp))
    return tuple(entries), mode
