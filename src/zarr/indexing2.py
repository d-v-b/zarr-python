from __future__ import annotations

import itertools
import math
import numbers
import operator
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from zarr.common import ChunkCoords, SliceSelection
from zarr.v2.errors import (
    ArrayIndexError,
    BoundsCheckError,
    NegativeStepError,
    VindexInvalidSelectionError,
    err_too_many_indices,
)

if TYPE_CHECKING:
    from zarr.array import Array
    from zarr.buffer import NDArrayLike
    from zarr.chunk_grids import ChunkGrid

Selector = int | slice | npt.NDArray[np.bool_]
Selection = tuple[Selector, ...]
Fields = str | list | tuple


@runtime_checkable
class Indexer(Protocol):
    shape: ChunkCoords
    drop_axes: ChunkCoords | None

    def __iter__(self) -> Iterator[ChunkProjection]: ...


def is_integer(x: Selector) -> bool:
    """True if x is an integer (both pure Python or NumPy).

    Note that Python's bool is considered an integer too.
    """
    return isinstance(x, numbers.Integral)


def is_integer_list(x: Selector) -> bool:
    """True if x is a list of integers.

    This function assumes ie *does not check* that all elements of the list
    have the same type. Mixed type lists will result in other errors that will
    bubble up anyway.
    """
    return isinstance(x, list) and len(x) > 0 and is_integer(x[0])


def is_integer_array(x: Selector, ndim: int | None = None) -> bool:
    t = not np.isscalar(x) and hasattr(x, "shape") and hasattr(x, "dtype") and x.dtype.kind in "ui"
    if ndim is not None:
        t = t and len(x.shape) == ndim
    return t


def is_bool_array(x: Selector, ndim: int | None = None) -> bool:
    t = hasattr(x, "shape") and hasattr(x, "dtype") and x.dtype == bool
    if ndim is not None:
        t = t and len(x.shape) == ndim
    return t


def is_scalar(value: Selector, dtype: npt.DTypeLike) -> bool:
    if np.isscalar(value):
        return True
    if hasattr(value, "shape") and value.shape == ():
        return True
    if isinstance(value, tuple) and dtype.names and len(value) == len(dtype.names):
        return True
    return False


def is_pure_fancy_indexing(selection: Selection, ndim: int) -> bool:
    """Check whether a selection contains only scalars or integer array-likes.

    Parameters
    ----------
    selection : tuple, slice, or scalar
        A valid selection value for indexing into arrays.

    Returns
    -------
    is_pure : bool
        True if the selection is a pure fancy indexing expression (ie not mixed
        with boolean or slices).
    """
    if ndim == 1:
        if is_integer_list(selection) or is_integer_array(selection):
            return True
        # if not, we go through the normal path below, because a 1-tuple
        # of integers is also allowed.
    no_slicing = (
        isinstance(selection, tuple)
        and len(selection) == ndim
        and not (any(isinstance(elem, slice) or elem is Ellipsis for elem in selection))
    )
    return (
        no_slicing
        and all(
            is_integer(elem) or is_integer_list(elem) or is_integer_array(elem)
            for elem in selection
        )
        and any(is_integer_list(elem) or is_integer_array(elem) for elem in selection)
    )


def is_pure_orthogonal_indexing(selection: Selection, ndim: int) -> bool:
    if not ndim:
        return False

    # Case 1: Selection is a single iterable of integers
    if is_integer_list(selection) or is_integer_array(selection, ndim=1):
        return True

    # Case two: selection contains either zero or one integer iterables.
    # All other selection elements are slices or integers
    return (
        isinstance(selection, tuple)
        and len(selection) == ndim
        and sum(is_integer_list(elem) or is_integer_array(elem) for elem in selection) <= 1
        and all(
            is_integer_list(elem) or is_integer_array(elem) or isinstance(elem, int | slice)
            for elem in selection
        )
    )


def get_chunk_shape(chunk_grid: ChunkGrid) -> ChunkCoords:
    from zarr.chunk_grids import RegularChunkGrid

    assert isinstance(
        chunk_grid, RegularChunkGrid
    ), "Only regular chunk grid is supported, currently."
    return chunk_grid.chunk_shape


def normalize_integer_selection(dim_sel: int, dim_len: int) -> int:
    # normalize type to int
    dim_sel = int(dim_sel)

    # handle wraparound
    if dim_sel < 0:
        dim_sel = dim_len + dim_sel

    # handle out of bounds
    if dim_sel >= dim_len or dim_sel < 0:
        raise BoundsCheckError(dim_len)

    return dim_sel


class ChunkDimProjection(NamedTuple):
    """A mapping from chunk to output array for a single dimension.

    Parameters
    ----------
    dim_chunk_ix
        Index of chunk.
    dim_chunk_sel
        Selection of items from chunk array.
    dim_out_sel
        Selection of items in target (output) array.

    """

    dim_chunk_ix: int
    dim_chunk_sel: slice
    dim_out_sel: slice


@dataclass(frozen=True)
class IntDimIndexer:
    dim_sel: int
    dim_len: int
    dim_chunk_len: int
    nitems: int = 1

    def __init__(self, dim_sel: int, dim_len: int, dim_chunk_len: int):
        object.__setattr__(self, "dim_sel", normalize_integer_selection(dim_sel, dim_len))
        object.__setattr__(self, "dim_len", dim_len)
        object.__setattr__(self, "dim_chunk_len", dim_chunk_len)

    def __iter__(self) -> Iterator[ChunkDimProjection]:
        dim_chunk_ix = self.dim_sel // self.dim_chunk_len
        dim_offset = dim_chunk_ix * self.dim_chunk_len
        dim_chunk_sel = self.dim_sel - dim_offset
        dim_out_sel = None
        yield ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


def ceildiv(a: int, b: int) -> int:
    return math.ceil(a / b)


@dataclass(frozen=True)
class SliceDimIndexer:
    dim_len: int
    dim_chunk_len: int
    nitems: int
    nchunks: int

    start: int
    stop: int
    step: int

    def __init__(self, dim_sel: slice, dim_len: int, dim_chunk_len: int):
        # normalize
        start, stop, step = dim_sel.indices(dim_len)
        if step < 1:
            raise NegativeStepError

        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "step", step)

        object.__setattr__(self, "dim_len", dim_len)
        object.__setattr__(self, "dim_chunk_len", dim_chunk_len)
        object.__setattr__(self, "nitems", max(0, ceildiv((stop - start), step)))
        object.__setattr__(self, "nchunks", ceildiv(dim_len, dim_chunk_len))

    def __iter__(self) -> Iterator[ChunkDimProjection]:
        # figure out the range of chunks we need to visit
        dim_chunk_ix_from = self.start // self.dim_chunk_len
        dim_chunk_ix_to = ceildiv(self.stop, self.dim_chunk_len)

        # iterate over chunks in range
        for dim_chunk_ix in range(dim_chunk_ix_from, dim_chunk_ix_to):
            # compute offsets for chunk within overall array
            dim_offset = dim_chunk_ix * self.dim_chunk_len
            dim_limit = min(self.dim_len, (dim_chunk_ix + 1) * self.dim_chunk_len)

            # determine chunk length, accounting for trailing chunk
            dim_chunk_len = dim_limit - dim_offset

            if self.start < dim_offset:
                # selection starts before current chunk
                dim_chunk_sel_start = 0
                remainder = (dim_offset - self.start) % self.step
                if remainder:
                    dim_chunk_sel_start += self.step - remainder
                # compute number of previous items, provides offset into output array
                dim_out_offset = ceildiv((dim_offset - self.start), self.step)

            else:
                # selection starts within current chunk
                dim_chunk_sel_start = self.start - dim_offset
                dim_out_offset = 0

            if self.stop > dim_limit:
                # selection ends after current chunk
                dim_chunk_sel_stop = dim_chunk_len

            else:
                # selection ends within current chunk
                dim_chunk_sel_stop = self.stop - dim_offset

            dim_chunk_sel = slice(dim_chunk_sel_start, dim_chunk_sel_stop, self.step)
            dim_chunk_nitems = ceildiv((dim_chunk_sel_stop - dim_chunk_sel_start), self.step)

            # If there are no elements on the selection within this chunk, then skip
            if dim_chunk_nitems == 0:
                continue

            dim_out_sel = slice(dim_out_offset, dim_out_offset + dim_chunk_nitems)

            yield ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


def check_selection_length(selection: Selection, shape: ChunkCoords):
    if len(selection) > len(shape):
        err_too_many_indices(selection, shape)


def replace_ellipsis(selection: Selection, shape: ChunkCoords):
    selection = ensure_tuple(selection)

    # count number of ellipsis present
    n_ellipsis = sum(1 for i in selection if i is Ellipsis)

    if n_ellipsis > 1:
        # more than 1 is an error
        raise IndexError("an index can only have a single ellipsis ('...')")

    elif n_ellipsis == 1:
        # locate the ellipsis, count how many items to left and right
        n_items_l = selection.index(Ellipsis)  # items to left of ellipsis
        n_items_r = len(selection) - (n_items_l + 1)  # items to right of ellipsis
        n_items = len(selection) - 1  # all non-ellipsis items

        if n_items >= len(shape):
            # ellipsis does nothing, just remove it
            selection = tuple(i for i in selection if i != Ellipsis)

        else:
            # replace ellipsis with as many slices are needed for number of dims
            new_item = selection[:n_items_l] + ((slice(None),) * (len(shape) - n_items))
            if n_items_r:
                new_item += selection[-n_items_r:]
            selection = new_item

    # fill out selection if not completely specified
    if len(selection) < len(shape):
        selection += (slice(None),) * (len(shape) - len(selection))

    # check selection not too long
    check_selection_length(selection, shape)

    return selection


def replace_lists(selection: Selection) -> Selection:
    return tuple(
        np.asarray(dim_sel) if isinstance(dim_sel, list) else dim_sel for dim_sel in selection
    )


def ensure_tuple(v: Selector | tuple[Selector]) -> tuple[Selector, ...]:
    if not isinstance(v, tuple):
        v = (v,)
    return v


class ChunkProjection(NamedTuple):
    """A mapping of items from chunk to output array. Can be used to extract items from the
    chunk array for loading into an output array. Can also be used to extract items from a
    value array for setting/updating in a chunk array.

    Parameters
    ----------
    chunk_coords
        Indices of chunk.
    chunk_selection
        Selection of items from chunk array.
    out_selection
        Selection of items in target (output) array.

    """

    chunk_coords: ChunkCoords
    chunk_selection: SliceSelection
    out_selection: SliceSelection


def is_slice(s: slice) -> bool:
    return isinstance(s, slice)


def is_contiguous_slice(s: slice) -> bool:
    return is_slice(s) and (s.step is None or s.step == 1)


def is_positive_slice(s: slice) -> bool:
    return is_slice(s) and (s.step is None or s.step >= 1)


def is_contiguous_selection(selection: Selection) -> bool:
    selection = ensure_tuple(selection)
    return all((is_integer_array(s) or is_contiguous_slice(s) or s == Ellipsis) for s in selection)


def is_basic_selection(selection: Selection) -> bool:
    selection = ensure_tuple(selection)
    return all(is_integer(s) or is_positive_slice(s) for s in selection)


@dataclass(frozen=True)
class BasicIndexer(Indexer):
    dim_indexers: list[IntDimIndexer | SliceDimIndexer]
    shape: ChunkCoords
    drop_axes: None

    def __init__(
        self,
        selection: Selection,
        shape: ChunkCoords,
        chunk_grid: ChunkGrid,
    ):
        chunk_shape = get_chunk_shape(chunk_grid)
        # handle ellipsis
        selection = replace_ellipsis(selection, shape)

        # setup per-dimension indexers
        dim_indexers = []
        for dim_sel, dim_len, dim_chunk_len in zip(selection, shape, chunk_shape, strict=True):
            if is_integer(dim_sel):
                dim_indexer = IntDimIndexer(dim_sel, dim_len, dim_chunk_len)

            elif is_slice(dim_sel):
                dim_indexer = SliceDimIndexer(dim_sel, dim_len, dim_chunk_len)

            else:
                raise IndexError(
                    "unsupported selection item for basic indexing; "
                    f"expected integer or slice, got {type(dim_sel)!r}"
                )

            dim_indexers.append(dim_indexer)

        object.__setattr__(self, "dim_indexers", dim_indexers)
        object.__setattr__(
            self,
            "shape",
            tuple(s.nitems for s in self.dim_indexers if not isinstance(s, IntDimIndexer)),
        )
        object.__setattr__(self, "drop_axes", None)

    def __iter__(self) -> Iterator[ChunkProjection]:
        for dim_projections in itertools.product(*self.dim_indexers):
            chunk_coords = tuple(p.dim_chunk_ix for p in dim_projections)
            chunk_selection = tuple(p.dim_chunk_sel for p in dim_projections)
            out_selection = tuple(
                p.dim_out_sel for p in dim_projections if p.dim_out_sel is not None
            )

            yield ChunkProjection(chunk_coords, chunk_selection, out_selection)


@dataclass(frozen=True)
class BoolArrayDimIndexer:
    dim_sel: npt.NDArray[np.bool_]
    dim_len: int
    dim_chunk_len: int
    nchunks: int

    chunk_nitems: npt.NDArray[Any]
    chunk_nitems_cumsum: npt.NDArray[Any]
    nitems: int
    dim_chunk_ixs: int

    def __init__(self, dim_sel: npt.NDArray[np.bool_], dim_len: int, dim_chunk_len: int):
        # check number of dimensions
        if not is_bool_array(dim_sel, 1):
            raise IndexError("Boolean arrays in an orthogonal selection must be 1-dimensional only")

        # check shape
        if dim_sel.shape[0] != dim_len:
            raise IndexError(
                f"Boolean array has the wrong length for dimension; expected {dim_len}, got {dim_sel.shape[0]}"
            )

        # precompute number of selected items for each chunk
        nchunks = ceildiv(dim_len, dim_chunk_len)
        chunk_nitems = np.zeros(nchunks, dtype="i8")
        for dim_chunk_ix in range(nchunks):
            dim_offset = dim_chunk_ix * dim_chunk_len
            chunk_nitems[dim_chunk_ix] = np.count_nonzero(
                dim_sel[dim_offset : dim_offset + dim_chunk_len]
            )
        chunk_nitems_cumsum = np.cumsum(chunk_nitems)
        nitems = chunk_nitems_cumsum[-1]
        dim_chunk_ixs = np.nonzero(chunk_nitems)[0]

        # store attributes
        object.__setattr__(self, "dim_sel", dim_sel)
        object.__setattr__(self, "dim_len", dim_len)
        object.__setattr__(self, "dim_chunk_len", dim_chunk_len)
        object.__setattr__(self, "nchunks", nchunks)
        object.__setattr__(self, "chunk_nitems", chunk_nitems)
        object.__setattr__(self, "chunk_nitems_cumsum", chunk_nitems_cumsum)
        object.__setattr__(self, "nitems", nitems)
        object.__setattr__(self, "dim_chunk_ixs", dim_chunk_ixs)

    def __iter__(self) -> Iterator[ChunkDimProjection]:
        # iterate over chunks with at least one item
        for dim_chunk_ix in self.dim_chunk_ixs:
            # find region in chunk
            dim_offset = dim_chunk_ix * self.dim_chunk_len
            dim_chunk_sel = self.dim_sel[dim_offset : dim_offset + self.dim_chunk_len]

            # pad out if final chunk
            if dim_chunk_sel.shape[0] < self.dim_chunk_len:
                tmp = np.zeros(self.dim_chunk_len, dtype=bool)
                tmp[: dim_chunk_sel.shape[0]] = dim_chunk_sel
                dim_chunk_sel = tmp

            # find region in output
            if dim_chunk_ix == 0:
                start = 0
            else:
                start = self.chunk_nitems_cumsum[dim_chunk_ix - 1]
            stop = self.chunk_nitems_cumsum[dim_chunk_ix]
            dim_out_sel = slice(start, stop)

            yield ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


class Order(Enum):
    UNKNOWN = 0
    INCREASING = 1
    DECREASING = 2
    UNORDERED = 3

    @staticmethod
    def check(a: npt.NDArray[Any]) -> Order:
        diff = np.diff(a)
        diff_positive = diff >= 0
        n_diff_positive = np.count_nonzero(diff_positive)
        all_increasing = n_diff_positive == len(diff_positive)
        any_increasing = n_diff_positive > 0
        if all_increasing:
            order = Order.INCREASING
        elif any_increasing:
            order = Order.UNORDERED
        else:
            order = Order.DECREASING
        return order


def wraparound_indices(x, dim_len):
    loc_neg = x < 0
    if np.any(loc_neg):
        x[loc_neg] = x[loc_neg] + dim_len


def boundscheck_indices(x, dim_len):
    if np.any(x < 0) or np.any(x >= dim_len):
        raise BoundsCheckError(dim_len)


@dataclass(frozen=True)
class IntArrayDimIndexer:
    """Integer array selection against a single dimension."""

    dim_len: int
    dim_chunk_len: int
    nchunks: int
    nitems: int
    order: Order
    dim_sel: int
    dim_out_sel: int
    chunk_nitems: int
    dim_chunk_ixs: npt.NDArray[np.intp]
    chunk_nitems_cumsum: npt.NDArray[np.intp]

    def __init__(
        self,
        dim_sel: int,
        dim_len: int,
        dim_chunk_len: int,
        wraparound=True,
        boundscheck=True,
        order=Order.UNKNOWN,
    ):
        # ensure 1d array
        dim_sel = np.asanyarray(dim_sel)
        if not is_integer_array(dim_sel, 1):
            raise IndexError("integer arrays in an orthogonal selection must be 1-dimensional only")

        nitems = len(dim_sel)
        nchunks = ceildiv(dim_len, dim_chunk_len)

        # handle wraparound
        if wraparound:
            wraparound_indices(dim_sel, dim_len)

        # handle out of bounds
        if boundscheck:
            boundscheck_indices(dim_sel, dim_len)

        # determine which chunk is needed for each selection item
        # note: for dense integer selections, the division operation here is the
        # bottleneck
        dim_sel_chunk = dim_sel // dim_chunk_len

        # determine order of indices
        if order == Order.UNKNOWN:
            order = Order.check(dim_sel)
        order = Order(order)

        if order == Order.INCREASING:
            dim_sel = dim_sel
            dim_out_sel = None
        elif order == Order.DECREASING:
            dim_sel = dim_sel[::-1]
            # TODO should be possible to do this without creating an arange
            dim_out_sel = np.arange(nitems - 1, -1, -1)
        else:
            # sort indices to group by chunk
            dim_out_sel = np.argsort(dim_sel_chunk)
            dim_sel = np.take(dim_sel, dim_out_sel)

        # precompute number of selected items for each chunk
        chunk_nitems = np.bincount(dim_sel_chunk, minlength=nchunks)

        # find chunks that we need to visit
        dim_chunk_ixs = np.nonzero(chunk_nitems)[0]

        # compute offsets into the output array
        chunk_nitems_cumsum = np.cumsum(chunk_nitems)

        # store attributes
        object.__setattr__(self, "dim_len", dim_len)
        object.__setattr__(self, "dim_chunk_len", dim_chunk_len)
        object.__setattr__(self, "nchunks", nchunks)
        object.__setattr__(self, "nitems", nitems)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "dim_sel", dim_sel)
        object.__setattr__(self, "dim_out_sel", dim_out_sel)
        object.__setattr__(self, "chunk_nitems", chunk_nitems)
        object.__setattr__(self, "dim_chunk_ixs", dim_chunk_ixs)
        object.__setattr__(self, "chunk_nitems_cumsum", chunk_nitems_cumsum)

    def __iter__(self) -> Iterator[ChunkDimProjection]:
        for dim_chunk_ix in self.dim_chunk_ixs:
            # find region in output
            if dim_chunk_ix == 0:
                start = 0
            else:
                start = self.chunk_nitems_cumsum[dim_chunk_ix - 1]
            stop = self.chunk_nitems_cumsum[dim_chunk_ix]
            if self.order == Order.INCREASING:
                dim_out_sel = slice(start, stop)
            else:
                dim_out_sel = self.dim_out_sel[start:stop]

            # find region in chunk
            dim_offset = dim_chunk_ix * self.dim_chunk_len
            dim_chunk_sel = self.dim_sel[start:stop] - dim_offset

            yield ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


def slice_to_range(s: slice, l: int):  # noqa: E741
    return range(*s.indices(l))


def ix_(selection: Selection, shape: ChunkCoords):
    """Convert an orthogonal selection to a numpy advanced (fancy) selection, like numpy.ix_
    but with support for slices and single ints."""

    # normalisation
    selection = replace_ellipsis(selection, shape)

    # replace slice and int as these are not supported by numpy.ix_
    selection = [
        slice_to_range(dim_sel, dim_len)
        if isinstance(dim_sel, slice)
        else [dim_sel]
        if is_integer(dim_sel)
        else dim_sel
        for dim_sel, dim_len in zip(selection, shape, strict=True)
    ]

    # now get numpy to convert to a coordinate selection
    selection = np.ix_(*selection)

    return selection


def oindex(a: npt.NDArray[Any], selection: Selection):
    """Implementation of orthogonal indexing with slices and ints."""
    selection = replace_ellipsis(selection, a.shape)
    drop_axes = tuple(i for i, s in enumerate(selection) if is_integer(s))
    selection = ix_(selection, a.shape)
    result = a[selection]
    if drop_axes:
        result = result.squeeze(axis=drop_axes)
    return result


def oindex_set(a: npt.NDArray[Any], selection: Selection, value):
    selection = replace_ellipsis(selection, a.shape)
    drop_axes = tuple(i for i, s in enumerate(selection) if is_integer(s))
    selection = ix_(selection, a.shape)
    if not np.isscalar(value) and drop_axes:
        value = np.asanyarray(value)
        value_selection = [slice(None)] * len(a.shape)
        for i in drop_axes:
            value_selection[i] = np.newaxis
        value_selection = tuple(value_selection)
        value = value[value_selection]
    a[selection] = value


@dataclass(frozen=True)
class OrthogonalIndexer(Indexer):
    dim_indexers: list[IntDimIndexer | SliceDimIndexer | IntArrayDimIndexer | BoolArrayDimIndexer]
    shape: ChunkCoords
    chunk_shape: ChunkCoords
    is_advanced: bool
    drop_axes: tuple[int, ...] | None

    def __init__(self, selection: Selection, shape: ChunkCoords, chunk_grid: ChunkGrid):
        chunk_shape = get_chunk_shape(chunk_grid)

        # handle ellipsis
        selection = replace_ellipsis(selection, shape)

        # normalize list to array
        selection = replace_lists(selection)

        # setup per-dimension indexers
        dim_indexers = []
        for dim_sel, dim_len, dim_chunk_len in zip(selection, shape, chunk_shape, strict=True):
            if is_integer(dim_sel):
                dim_indexer = IntDimIndexer(dim_sel, dim_len, dim_chunk_len)

            elif isinstance(dim_sel, slice):
                dim_indexer = SliceDimIndexer(dim_sel, dim_len, dim_chunk_len)

            elif is_integer_array(dim_sel):
                dim_indexer = IntArrayDimIndexer(dim_sel, dim_len, dim_chunk_len)

            elif is_bool_array(dim_sel):
                dim_indexer = BoolArrayDimIndexer(dim_sel, dim_len, dim_chunk_len)

            else:
                raise IndexError(
                    "unsupported selection item for orthogonal indexing; "
                    "expected integer, slice, integer array or Boolean "
                    f"array, got {type(dim_sel)!r}"
                )

            dim_indexers.append(dim_indexer)

        dim_indexers = dim_indexers
        shape = tuple(s.nitems for s in dim_indexers if not isinstance(s, IntDimIndexer))
        chunk_shape = chunk_shape
        is_advanced = not is_basic_selection(selection)
        if is_advanced:
            drop_axes = tuple(
                i
                for i, dim_indexer in enumerate(dim_indexers)
                if isinstance(dim_indexer, IntDimIndexer)
            )
        else:
            drop_axes = None

        object.__setattr__(self, "dim_indexers", dim_indexers)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "chunk_shape", chunk_shape)
        object.__setattr__(self, "is_advanced", is_advanced)
        object.__setattr__(self, "drop_axes", drop_axes)

    def __iter__(self) -> Iterator[ChunkProjection]:
        for dim_projections in itertools.product(*self.dim_indexers):
            chunk_coords = tuple(p.dim_chunk_ix for p in dim_projections)
            chunk_selection = tuple(p.dim_chunk_sel for p in dim_projections)
            out_selection = tuple(
                p.dim_out_sel for p in dim_projections if p.dim_out_sel is not None
            )

            # handle advanced indexing arrays orthogonally
            if self.is_advanced:
                # N.B., numpy doesn't support orthogonal indexing directly as yet,
                # so need to work around via np.ix_. Also np.ix_ does not support a
                # mixture of arrays and slices or integers, so need to convert slices
                # and integers into ranges.
                chunk_selection = ix_(chunk_selection, self.chunk_shape)

                # special case for non-monotonic indices
                if not is_basic_selection(out_selection):
                    out_selection = ix_(out_selection, self.shape)

            yield ChunkProjection(chunk_coords, chunk_selection, out_selection)


@dataclass(frozen=True)
class OIndex:
    array: Array

    def __getitem__(self, selection: Selection) -> NDArrayLike:
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        return self.array.get_orthogonal_selection(selection, fields=fields)

    def __setitem__(self, selection: Selection, value: NDArrayLike) -> None:
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        return self.array.set_orthogonal_selection(selection, value, fields=fields)


@dataclass(frozen=True)
class BlockIndexer(Indexer):
    dim_indexers: list[SliceDimIndexer]
    shape: ChunkCoords
    drop_axes: None

    def __init__(self, selection, shape: ChunkCoords, chunk_grid: ChunkGrid):
        chunk_shape = get_chunk_shape(chunk_grid)

        # handle ellipsis
        selection = replace_ellipsis(selection, shape)

        # normalize list to array
        selection = replace_lists(selection)

        # setup per-dimension indexers
        dim_indexers = []
        for dim_sel, dim_len, dim_chunk_size in zip(selection, shape, chunk_shape, strict=True):
            dim_numchunks = int(np.ceil(dim_len / dim_chunk_size))

            if is_integer(dim_sel):
                if dim_sel < 0:
                    dim_sel = dim_numchunks + dim_sel

                start = dim_sel * dim_chunk_size
                stop = start + dim_chunk_size
                slice_ = slice(start, stop)

            elif is_slice(dim_sel):
                start = dim_sel.start if dim_sel.start is not None else 0
                stop = dim_sel.stop if dim_sel.stop is not None else dim_numchunks

                if dim_sel.step not in {1, None}:
                    raise IndexError(
                        "unsupported selection item for block indexing; "
                        f"expected integer or slice with step=1, got {type(dim_sel)!r}"
                    )

                # Can't reuse wraparound_indices because it expects a numpy array
                # We have integers here.
                if start < 0:
                    start = dim_numchunks + start
                if stop < 0:
                    stop = dim_numchunks + stop

                start = start * dim_chunk_size
                stop = stop * dim_chunk_size
                slice_ = slice(start, stop)

            else:
                raise IndexError(
                    "unsupported selection item for block indexing; "
                    f"expected integer or slice, got {type(dim_sel)!r}"
                )

            dim_indexer = SliceDimIndexer(slice_, dim_len, dim_chunk_size)
            dim_indexers.append(dim_indexer)

            if start >= dim_len or start < 0:
                raise BoundsCheckError(dim_len)

        dim_indexers = dim_indexers
        shape = tuple(s.nitems for s in dim_indexers)

        object.__setattr__(self, "dim_indexers", dim_indexers)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "drop_axes", None)

    def __iter__(self) -> Iterator[ChunkProjection]:
        for dim_projections in itertools.product(*self.dim_indexers):
            chunk_coords = tuple(p.dim_chunk_ix for p in dim_projections)
            chunk_selection = tuple(p.dim_chunk_sel for p in dim_projections)
            out_selection = tuple(
                p.dim_out_sel for p in dim_projections if p.dim_out_sel is not None
            )

            yield ChunkProjection(chunk_coords, chunk_selection, out_selection)


@dataclass(frozen=True)
class BlockIndex:
    array: Array

    def __getitem__(self, selection: Selection) -> NDArrayLike:
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        return self.array.get_block_selection(selection, fields=fields)

    def __setitem__(self, selection: Selection, value: NDArrayLike) -> None:
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        return self.array.set_block_selection(selection, value, fields=fields)


def is_coordinate_selection(selection: Selection, shape: ChunkCoords) -> bool:
    return (len(selection) == len(shape)) and all(
        is_integer(dim_sel) or is_integer_array(dim_sel) for dim_sel in selection
    )


def is_mask_selection(selection: Selection, shape: ChunkCoords) -> bool:
    return len(selection) == 1 and is_bool_array(selection[0]) and selection[0].shape == shape


@dataclass(frozen=True)
class CoordinateIndexer(Indexer):
    sel_shape: ChunkCoords
    selection: Selection
    sel_sort: npt.NDArray[np.intp] | None
    chunk_nitems_cumsum: npt.NDArray[np.intp]
    chunk_rixs: npt.NDArray[np.intp]
    chunk_mixs: tuple[npt.NDArray[np.intp], ...]
    shape: ChunkCoords
    chunk_shape: ChunkCoords
    drop_axes: None

    def __init__(self, selection: Selection, shape: ChunkCoords, chunk_grid: ChunkGrid):
        chunk_shape = get_chunk_shape(chunk_grid)

        if shape == ():
            cdata_shape = (1,)
        else:
            cdata_shape = tuple(math.ceil(s / c) for s, c in zip(shape, chunk_shape, strict=True))
        nchunks = reduce(operator.mul, cdata_shape, 1)

        # some initial normalization
        selection = ensure_tuple(selection)
        selection = tuple([i] if is_integer(i) else i for i in selection)
        selection = replace_lists(selection)

        # validation
        if not is_coordinate_selection(selection, shape):
            raise IndexError(
                "invalid coordinate selection; expected one integer "
                "(coordinate) array per dimension of the target array, "
                f"got {selection!r}"
            )

        # handle wraparound, boundscheck
        for dim_sel, dim_len in zip(selection, shape, strict=True):
            # handle wraparound
            wraparound_indices(dim_sel, dim_len)

            # handle out of bounds
            boundscheck_indices(dim_sel, dim_len)

        # compute chunk index for each point in the selection
        chunks_multi_index = tuple(
            dim_sel // dim_chunk_len
            for (dim_sel, dim_chunk_len) in zip(selection, chunk_shape, strict=True)
        )

        # broadcast selection - this will raise error if array dimensions don't match
        selection = np.broadcast_arrays(*selection)
        chunks_multi_index = np.broadcast_arrays(*chunks_multi_index)

        # remember shape of selection, because we will flatten indices for processing
        sel_shape = selection[0].shape if selection[0].shape else (1,)

        # flatten selection
        selection = [dim_sel.reshape(-1) for dim_sel in selection]
        chunks_multi_index = [dim_chunks.reshape(-1) for dim_chunks in chunks_multi_index]

        # ravel chunk indices
        chunks_raveled_indices = np.ravel_multi_index(chunks_multi_index, dims=cdata_shape)

        # group points by chunk
        if np.any(np.diff(chunks_raveled_indices) < 0):
            # optimisation, only sort if needed
            sel_sort = np.argsort(chunks_raveled_indices)
            selection = tuple(dim_sel[sel_sort] for dim_sel in selection)
        else:
            sel_sort = None

        shape = selection[0].shape if selection[0].shape else (1,)

        # precompute number of selected items for each chunk
        chunk_nitems = np.bincount(chunks_raveled_indices, minlength=nchunks)
        chunk_nitems_cumsum = np.cumsum(chunk_nitems)
        # locate the chunks we need to process
        chunk_rixs = np.nonzero(chunk_nitems)[0]

        # unravel chunk indices
        chunk_mixs = np.unravel_index(chunk_rixs, cdata_shape)

        object.__setattr__(self, "sel_shape", sel_shape)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "sel_sort", sel_sort)
        object.__setattr__(self, "chunk_nitems_cumsum", chunk_nitems_cumsum)
        object.__setattr__(self, "chunk_rixs", chunk_rixs)
        object.__setattr__(self, "chunk_mixs", chunk_mixs)
        object.__setattr__(self, "chunk_shape", chunk_shape)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "drop_axes", None)

    def __iter__(self) -> Iterator[ChunkProjection]:
        # iterate over chunks
        for i, chunk_rix in enumerate(self.chunk_rixs):
            chunk_coords = tuple(m[i] for m in self.chunk_mixs)
            if chunk_rix == 0:
                start = 0
            else:
                start = self.chunk_nitems_cumsum[chunk_rix - 1]
            stop = self.chunk_nitems_cumsum[chunk_rix]
            if self.sel_sort is None:
                out_selection = slice(start, stop)
            else:
                out_selection = self.sel_sort[start:stop]

            chunk_offsets = tuple(
                dim_chunk_ix * dim_chunk_len
                for dim_chunk_ix, dim_chunk_len in zip(chunk_coords, self.chunk_shape, strict=True)
            )
            chunk_selection = tuple(
                dim_sel[start:stop] - dim_chunk_offset
                for (dim_sel, dim_chunk_offset) in zip(self.selection, chunk_offsets, strict=True)
            )

            yield ChunkProjection(chunk_coords, chunk_selection, out_selection)


@dataclass(frozen=True)
class MaskIndexer(CoordinateIndexer):
    def __init__(self, selection, shape: ChunkCoords, chunk_grid: ChunkGrid):
        # some initial normalization
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)

        # validation
        if not is_mask_selection(selection, shape):
            raise IndexError(
                "invalid mask selection; expected one Boolean (mask)"
                f"array with the same shape as the target array, got {selection!r}"
            )

        # convert to indices
        selection = np.nonzero(selection[0])

        # delegate the rest to superclass
        super().__init__(selection, shape, chunk_grid)


@dataclass(frozen=True)
class VIndex:
    array: Array

    def __getitem__(self, selection: Selection) -> NDArrayLike:
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        if is_coordinate_selection(selection, self.array.shape):
            return self.array.get_coordinate_selection(selection, fields=fields)
        elif is_mask_selection(selection, self.array.shape):
            return self.array.get_mask_selection(selection, fields=fields)
        else:
            raise VindexInvalidSelectionError(selection)

    def __setitem__(self, selection: Selection, value: NDArrayLike) -> None:
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        if is_coordinate_selection(selection, self.array.shape):
            self.array.set_coordinate_selection(selection, value, fields=fields)
        elif is_mask_selection(selection, self.array.shape):
            self.array.set_mask_selection(selection, value, fields=fields)
        else:
            raise VindexInvalidSelectionError(selection)


def check_fields(fields: Fields, dtype: npt.DTypeLike) -> npt.DTypeLike:
    # early out
    if fields is None:
        return dtype
    # check type
    if not isinstance(fields, Fields):
        raise IndexError(
            f"'fields' argument must be a string or list of strings; found {type(fields)!r}"
        )
    if fields:
        if dtype.names is None:
            raise IndexError("invalid 'fields' argument, array does not have any fields")
        try:
            if isinstance(fields, str):
                # single field selection
                out_dtype = dtype[fields]
            else:
                # multiple field selection
                out_dtype = np.dtype([(f, dtype[f]) for f in fields])
        except KeyError as e:
            raise IndexError(f"invalid 'fields' argument, field not found: {e!r}") from e
        else:
            return out_dtype
    else:
        return dtype


def check_no_multi_fields(fields: Fields) -> Fields:
    if isinstance(fields, list):
        if len(fields) == 1:
            return fields[0]
        elif len(fields) > 1:
            raise IndexError("multiple fields are not supported for this operation")
    return fields


def pop_fields(selection: Selection) -> tuple[Fields | None, Selection]:
    if isinstance(selection, str):
        # single field selection
        fields = selection
        selection = ()
    elif not isinstance(selection, tuple):
        # single selection item, no fields
        fields = None
        # leave selection as-is
    else:
        # multiple items, split fields from selection items
        fields = [f for f in selection if isinstance(f, str)]
        fields = fields[0] if len(fields) == 1 else fields
        selection = tuple(s for s in selection if not isinstance(s, str))
        selection = selection[0] if len(selection) == 1 else selection
    return fields, selection


def make_slice_selection(selection: Selection) -> list[int | slice]:
    ls = []
    for dim_selection in selection:
        if is_integer(dim_selection):
            ls.append(slice(int(dim_selection), int(dim_selection) + 1, 1))
        elif isinstance(dim_selection, np.ndarray):
            if len(dim_selection) == 1:
                ls.append(slice(int(dim_selection[0]), int(dim_selection[0]) + 1, 1))
            else:
                raise ArrayIndexError
        else:
            ls.append(dim_selection)
    return ls