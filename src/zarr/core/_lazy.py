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
    import numpy.typing as npt

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
        `lazy.oindex[...]` / `lazy.vindex[...]`.
        """
        new_transform = selection_to_transform(selection, self._transform, "basic")
        return _LazyArray(_array=self._array, _transform=new_transform)

    def __setitem__(self, selection: Any, value: Any) -> None:
        """Materializing write: composes the selection onto the current
        transform, recovers a (selection, mode) tuple, and writes through
        the underlying array's eager set_* methods.

        Lazy writes are NOT deferred; the value is written immediately. This
        matches the design: the lazy view is for deferred *reads*. Use
        `lazy.oindex[sel] = value` / `lazy.vindex[sel] = value` for
        orthogonal / vectorized writes (helpers below).
        """
        new_transform = selection_to_transform(selection, self._transform, "basic")
        sel, mode = transform_to_selection(new_transform)
        if mode == "basic":
            self._array[sel] = value
        elif mode == "orthogonal":
            self._array.oindex[sel] = value
        else:  # vectorized
            self._array.vindex[sel] = value

    @property
    def oindex(self) -> _LazyOIndex:
        """Helper for orthogonal lazy indexing: `lazy.oindex[selection]`."""
        return _LazyOIndex(_parent=self)

    @property
    def vindex(self) -> _LazyVIndex:
        """Helper for vectorized lazy indexing: `lazy.vindex[selection]`."""
        return _LazyVIndex(_parent=self)

    def result(self) -> Any:
        """Materialize the lazy view by dispatching through the underlying
        array's eager indexing path.

        Dispatch is governed by the transform's structure (see
        `transform_to_selection`): basic → `array[selection]`, orthogonal →
        `array.oindex[selection]`, vectorized → `array.vindex[selection]`.

        Return type is `Any` rather than `np.ndarray` because basic indexing
        with an integer scalar returns a numpy scalar, not an array.
        """
        selection, mode = transform_to_selection(self._transform)
        if mode == "basic":
            return self._array[selection]
        if mode == "orthogonal":
            return self._array.oindex[selection]
        # mode == "vectorized"
        return self._array.vindex[selection]

    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """NumPy interop: `np.asarray(lazy)` and `np.array(lazy)` materialize
        the lazy view by calling `.result()`.

        Honors the NumPy 2.0 `__array__` `copy` parameter contract:
        - `copy=None` (default) or `copy=True`: materialize freely.
        - `copy=False`: raise `ValueError`, because materialization itself
          allocates a new array — there is no zero-copy path through the
          eager indexing dispatch we use under the hood.
        """
        if copy is False:
            raise ValueError(
                "_LazyArray cannot satisfy copy=False; materialization always "
                "allocates a new array via the eager indexing path"
            )
        return np.asarray(self.result(), dtype=dtype)


@dataclass(frozen=True, slots=True)
class _LazyOIndex:
    """Helper for orthogonal lazy indexing: `lazy.oindex[selection]`.

    Composes the selection onto the parent _LazyArray's transform in
    orthogonal mode; returns a new _LazyArray. Setitem writes through
    the underlying array's `oindex[sel] = value`.
    """

    _parent: _LazyArray

    def __getitem__(self, selection: Any) -> _LazyArray:
        new_transform = selection_to_transform(selection, self._parent._transform, "orthogonal")
        return _LazyArray(_array=self._parent._array, _transform=new_transform)

    def __setitem__(self, selection: Any, value: Any) -> None:
        """Materializing orthogonal write: `lazy.oindex[sel] = value` writes
        through `underlying.oindex[recovered_sel] = value`."""
        new_transform = selection_to_transform(selection, self._parent._transform, "orthogonal")
        sel, _mode = transform_to_selection(new_transform)
        self._parent._array.oindex[sel] = value


@dataclass(frozen=True, slots=True)
class _LazyVIndex:
    """Helper for vectorized lazy indexing: `lazy.vindex[selection]`.

    Composes the selection onto the parent _LazyArray's transform in
    vectorized mode; returns a new _LazyArray. Setitem writes through
    the underlying array's `vindex[sel] = value`.
    """

    _parent: _LazyArray

    def __getitem__(self, selection: Any) -> _LazyArray:
        new_transform = selection_to_transform(selection, self._parent._transform, "vectorized")
        return _LazyArray(_array=self._parent._array, _transform=new_transform)

    def __setitem__(self, selection: Any, value: Any) -> None:
        """Materializing vectorized write: `lazy.vindex[sel] = value` writes
        through `underlying.vindex[recovered_sel] = value`."""
        new_transform = selection_to_transform(selection, self._parent._transform, "vectorized")
        sel, _mode = transform_to_selection(new_transform)
        self._parent._array.vindex[sel] = value


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
