"""Output index maps — three ordered mappings to integer coordinates.

An output index map describes how input cells address one dimension of
the output space. Its coordinates form an **ordered, duplicate-preserving sequence**
aligned with the input domain, never a mathematical set. Three representations
cover the cases that arise in practice:

- `ConstantMap(offset=5)` — every request cell maps to coordinate `5`
- `DimensionMap(input_dimension=0, offset=3, stride=2)` over input `[0, 5)`
  — the ordered arithmetic progression `[3, 5, 7, 9, 11]`
- `ArrayMap(index_array=[5, 1, 1])` — the explicit sequence `[5, 1, 1]`,
  preserving both order and the repeated coordinate

Every output map participates in two operations defined on `IndexTransform`,
which provides the input-domain context these maps lack:

- **intersect** — retain mapped cells whose coordinates lie within a range
  (e.g., a chunk), without changing their order or multiplicity.
  Restricting `[3, 5, 5, 9]` to `[4, 8)` produces `[5, 5]`.
- **translate** — shift every coordinate by a constant (e.g., make chunk-local).
  Translating `[5, 5, 7]` by `-4` produces `[1, 1, 3]`.

These two operations are the foundation of chunk resolution: for each chunk,
intersect the map with the chunk's range, then translate to chunk-local
coordinates.

The three types exist because they trade off generality for efficiency:

- `ConstantMap`: O(1) storage, O(1) intersection
- `DimensionMap`: O(1) storage, O(1) intersection (analytical)
- `ArrayMap`: O(n) storage, O(n) intersection (must scan the array)

Collapsing everything to `ArrayMap` would be correct but wasteful — a
billion-element slice would materialize a billion coordinates just to group
them by chunk, when `DimensionMap` does it with three integers.

An `ArrayMap` also carries an `IndexRange`: the closed interval its raw values
are *declared* to lie in (ndsel's `index_array_bounds`, TensorStore's
`index_range`). The declaration is retained, not enforced at construction; a
value outside it fails when it is used to address storage. See `IndexRange`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from zarr_indexing._affine import checked_affine
from zarr_indexing.errors import BoundsCheckError

if TYPE_CHECKING:
    import numpy.typing as npt

    from zarr_indexing.json import IndexValueJSON, OutputIndexMapJSON


def _array_map_dependency_axes(index_array: np.ndarray[Any, Any]) -> tuple[int, ...]:
    """Return the input axes on which a normalized index array varies.

    Normalized `ArrayMap` index arrays carry the full input rank of their
    enclosing transform: an axis the array varies over has its full size, while
    an axis the array is independent of is a singleton (size 1). The dependency
    axes are therefore exactly the axes of size 2 or more. This is structural:
    values are not inspected for constant or repeated coordinates. An array
    with shape `(0, 2)` is empty but still reports axis 1; the zero-size axis
    itself is not reported.
    """
    return tuple(axis for axis, size in enumerate(index_array.shape) if size > 1)


@dataclass(frozen=True, slots=True)
class IndexRange:
    """The closed interval an `ArrayMap`'s raw index values are declared to lie in.

    This is TensorStore's `index_range` on an index-array output map
    ([`output_index_map.h#L72-L73`](https://github.com/google/tensorstore/blob/5c6997751f4b4855de72d6e7f696c572aae3a4e5/tensorstore/index_space/output_index_map.h#L72-L73)),
    which ndsel spells `index_array_bounds` (spec section 4.2). It bounds the
    values of `index_array` themselves, before `offset` and `stride` are
    applied, and it is infinite on both sides by default, so `IndexRange()` is
    the unconstrained range every selection-built map carries.

    A range is a **declaration, not a proof**. An `ArrayMap` may hold values
    outside its range, and loading such a document succeeds; the value fails
    when it is *used* to address storage. This copies TensorStore, whose JSON
    binder parses the interval without looking at the array
    ([`json.cc#L200-L215`](https://github.com/google/tensorstore/blob/5c6997751f4b4855de72d6e7f696c572aae3a4e5/tensorstore/index_space/json.cc#L200-L215))
    and whose point evaluation checks each value it reads
    ([`transform_rep.cc#L471-L497`](https://github.com/google/tensorstore/blob/5c6997751f4b4855de72d6e7f696c572aae3a4e5/tensorstore/index_space/internal/transform_rep.cc#L471-L497)),
    as do its array iteration
    ([`iterate.cc#L232-L239`](https://github.com/google/tensorstore/blob/5c6997751f4b4855de72d6e7f696c572aae3a4e5/tensorstore/index_space/internal/iterate.cc#L232-L239))
    and the collapse of a one-element map to a constant
    ([`transform_rep.cc#L526-L531`](https://github.com/google/tensorstore/blob/5c6997751f4b4855de72d6e7f696c572aae3a4e5/tensorstore/index_space/internal/transform_rep.cc#L526-L531)).
    `ArrayMap.checked_index_array` is that gate here.

    Both bounds are inclusive, as on the wire. The lower bound is an integer or
    `"-inf"` and the upper an integer or `"+inf"`: a sentinel on the other side,
    or a lower bound above the upper, names no closed interval, and the
    constructor rejects it, as TensorStore's `IndexInterval::Closed` does at
    parse (`(10, 0) do not specify a valid closed index interval`).

    Examples
    --------
    >>> IndexRange().contains([-(10**18), 10**18])
    True
    >>> IndexRange(0, 9).contains([7, 3, 5]), IndexRange(0, 9).contains([7, 30])
    (True, False)
    >>> IndexRange(0, "+inf").to_json()
    [0, '+inf']
    """

    inclusive_min: int | Literal["-inf"] = "-inf"
    """The smallest value a raw index may take; `"-inf"` for no lower bound."""

    inclusive_max: int | Literal["+inf"] = "+inf"
    """The largest value a raw index may take; `"+inf"` for no upper bound."""

    def __post_init__(self) -> None:
        for name, value, sentinel in (
            ("inclusive_min", self.inclusive_min, "-inf"),
            ("inclusive_max", self.inclusive_max, "+inf"),
        ):
            if isinstance(value, (bool, np.bool_)) or not (
                value == sentinel or isinstance(value, (int, np.integer))
            ):
                raise ValueError(f"{name} must be an integer or {sentinel!r}, got {value!r}")
            if isinstance(value, np.integer):
                object.__setattr__(self, name, int(value))
        lo, hi = self.inclusive_min, self.inclusive_max
        if isinstance(lo, int) and isinstance(hi, int) and lo > hi:
            raise ValueError(
                f"[{lo}, {hi}] is not a closed interval: inclusive_min > inclusive_max"
            )

    @property
    def is_unbounded(self) -> bool:
        """`True` when neither side constrains anything: the default range."""
        return self.inclusive_min == "-inf" and self.inclusive_max == "+inf"

    def contains(self, values: npt.ArrayLike) -> bool:
        """Whether every value lies in the range; vacuously true of no values."""
        return self._outlier(np.asarray(values)) is None

    def check(self, values: npt.ArrayLike, *, output_dimension: int) -> None:
        """Raise `BoundsCheckError` unless every value lies in the range.

        `output_dimension` names the map the values came from in the error,
        which also names the offending value and the range.
        """
        outlier = self._outlier(np.asarray(values))
        if outlier is not None:
            raise BoundsCheckError(
                f"index {outlier} on output dimension {output_dimension} is outside "
                f"index_array_bounds {self.to_json()}"
            )

    def _outlier(self, values: np.ndarray[Any, Any]) -> int | None:
        """One value outside the range, or `None`; extrema decide in one scan.

        Python integers hold the extrema, so integer limits compare exactly.
        """
        if values.size == 0 or self.is_unbounded:
            return None
        low, high = int(values.min()), int(values.max())
        if self.inclusive_min != "-inf" and low < self.inclusive_min:
            return low
        if self.inclusive_max != "+inf" and high > self.inclusive_max:
            return high
        return None

    def to_json(self) -> list[IndexValueJSON]:
        """The wire spelling: `[inclusive_min, inclusive_max]`."""
        return [self.inclusive_min, self.inclusive_max]

    @classmethod
    def from_json(cls, bounds: Any, where: str = "output") -> IndexRange:
        """Lower a wire `index_array_bounds` value, validating its syntax.

        The message layer's `validate_index_array_bounds` checks the shape,
        the `index-value` grammar and the extended-integer order of the pair;
        the constructor then rejects a sentinel on the wrong side, the one
        well-ordered pair that is not a closed interval. Both failures are
        `NdselError("invalid_json")` (order errors are `bounds_out_of_order`).
        Array values are not consulted here or anywhere at load.

        Examples
        --------
        >>> IndexRange.from_json([0, 9])
        IndexRange(inclusive_min=0, inclusive_max=9)
        >>> IndexRange.from_json(["-inf", "+inf"]).is_unbounded
        True
        """
        from zarr_indexing.messages import NdselError, validate_index_array_bounds

        lo, hi = validate_index_array_bounds(bounds, where)
        try:
            # The validator's `int | str` is narrowed at runtime by the constructor.
            return cls(cast("Any", lo), cast("Any", hi))
        except ValueError as exc:
            raise NdselError("invalid_json", f"{where}.index_array_bounds: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ConstantMap:
    """A constant output-coordinate mapping.

    Every input cell maps to `offset`. Arises from integer indexing (e.g.,
    `arr[5]` fixes one dimension to coordinate 5).

    Examples
    --------
    Every input cell maps to the same output coordinate, like broadcasting
    coordinate 5 with `np.broadcast_to(5, (3,))`. Repeated fancy indices can
    also describe these coordinates, using an explicit list:

    >>> from zarr_indexing.domain import IndexDomain
    >>> from zarr_indexing.transform import IndexTransform
    >>> domain = IndexDomain.from_shape((3,))
    >>> t = IndexTransform(domain=domain, output=(ConstantMap(offset=5),))
    >>> t.apply((0,)), t.apply((1,)), t.apply((2,))
    ((5,), (5,), (5,))
    """

    offset: int = 0
    """The fixed output coordinate every input cell maps to."""

    def to_json(self) -> OutputIndexMapJSON:
        """Convert to the canonical wire form: the bare `constant` map.

        Examples
        --------
        >>> ConstantMap(5).to_json()
        {'offset': 5}
        """
        return {"offset": self.offset}


@dataclass(frozen=True, slots=True)
class DimensionMap:
    """An ordered affine mapping to output coordinates.

    Maps each input coordinate `i` to `offset + stride * i`, where the input
    range comes from the enclosing `IndexTransform`'s domain. Arises from slice
    indexing (e.g., `arr[2:10:3]` gives offset=2, stride=3).

    Examples
    --------
    The slice `arr[2:11:3]` reads coordinates `2, 5, 8` — the rule
    `offset + stride * i` with `offset=2`, `stride=3`:

    >>> m = DimensionMap(input_dimension=0, offset=2, stride=3)
    >>> [m.offset + m.stride * i for i in range(3)]
    [2, 5, 8]
    >>> np.arange(11)[2:11:3].tolist()
    [2, 5, 8]
    """

    input_dimension: int
    """The input (domain) dimension whose coordinate this map reads."""

    offset: int = 0
    """The output coordinate that input coordinate `0` maps to."""

    stride: int = 1
    """The output-coordinate step per unit input step; negative walks backward, zero repeats `offset`."""

    def to_json(self) -> OutputIndexMapJSON:
        """Convert to the canonical wire form: the `single_input_dimension` map.

        Examples
        --------
        >>> DimensionMap(input_dimension=1, offset=0, stride=2).to_json()
        {'offset': 0, 'stride': 2, 'input_dimension': 1}
        """
        return {
            "offset": self.offset,
            "stride": self.stride,
            "input_dimension": self.input_dimension,
        }


@dataclass(frozen=True, slots=True)
class ArrayMap:
    """An explicit ordered, duplicate-preserving coordinate mapping.

    Maps each input position `i` to `offset + stride * index_array[i]`.
    Index-array order and repeated entries are semantic and remain present in
    the result. Arises from fancy indexing (e.g., `arr[[5, 1, 1]]` or boolean
    masks).

    A map used in a transform must have its **full input rank**:
    `index_array` has the enclosing domain's rank, sized
    fully on the axes it varies over and singleton (size 1) elsewhere. The
    shape is the single source of truth for what the map depends on — its
    **dependency axes** are exactly its axes of size greater than one (see
    `_array_map_dependency_axes`) — and it distinguishes the two
    flavors of multi-array fancy indexing:

    - **orthogonal** (`oindex`): each array varies along a single, *distinct*
      axis (all others singleton); the result is their outer product.
    - **vectorized** (`vindex`): the arrays are correlated and share the same
      non-singleton (broadcast) axes; the result is a pointwise scatter.

    A map holding exactly one coordinate carries no shape to read a dependency
    from, and none is needed: it is the `ConstantMap` it equals, and the
    selection layer builds that instead (see `array_map_or_constant`). A
    hand-built all-singleton `ArrayMap` is still a valid value; resolution
    classifies it with the correlated maps and reads it pointwise.

    `index_range` declares the closed interval the raw values lie in. It is
    retained, not enforced here: a map may hold a value outside its range, and
    the value fails at the use that reads it (see `IndexRange` and
    `checked_index_array`). The range travels unchanged through the affine
    adjustment, reindexing, slicing and composition.

    Examples
    --------
    The fancy selection `arr[[5, 1, 1]]` reads coordinate 5, then 1, then 1
    — order and the duplicate preserved, exactly as NumPy fancy indexing:

    >>> m = ArrayMap(index_array=np.array([5, 1, 1]))
    >>> [m.offset + m.stride * c for c in m.index_array.tolist()]
    [5, 1, 1]
    >>> np.arange(10)[[5, 1, 1]].tolist()
    [5, 1, 1]

    A declared range does not reject a value at construction; the value is
    refused when it is read to address storage:

    >>> m = ArrayMap(index_array=np.array([5, 1, 40]), index_range=IndexRange(0, 9))
    >>> m.checked_index_array(output_dimension=0)
    Traceback (most recent call last):
        ...
    zarr_indexing.errors.BoundsCheckError: index 40 on output dimension 0 is outside index_array_bounds [0, 9]
    """

    index_array: npt.NDArray[np.integer[Any]]
    """Explicit coordinates at the enclosing transform's full input rank; order and
    duplicates are semantic. Its non-singleton axes are the map's dependency axes."""

    offset: int = 0
    """Constant term of the affine adjustment: the output coordinate is `offset + stride * index_array[i]`."""

    stride: int = 1
    """Multiplier applied to each `index_array` value before `offset` is added."""

    index_range: IndexRange = IndexRange()
    """The closed interval the raw `index_array` values are declared to lie in;
    unbounded by default. Retained and checked at use, never at construction."""

    def __post_init__(self) -> None:
        """Own an immutable snapshot of the integer index coordinates.

        The snapshot is backed by immutable bytes, so callers cannot modify it
        or re-enable its WRITEABLE flag. Changes to the supplied array do not
        change the map's coordinates or hash. This freezes the coordinate
        mapping, not the source values read through it. The values are not
        compared with `index_range`: that is a declaration checked at use."""
        # Immutable bytes are the ultimate owner so callers cannot re-enable
        # the WRITEABLE flag, as they can on a read-only array that owns its
        # allocation. `asarray` also accepts the NumPy scalars that reach here
        # after indexing an array down to one element.
        array = np.asarray(self.index_array)
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"index_array must have an integer dtype, got {array.dtype}")
        if not isinstance(self.index_range, IndexRange):  # pyright: ignore[reportUnnecessaryIsInstance] - untyped callers pass wire lists
            raise TypeError(f"index_range must be an IndexRange, got {self.index_range!r}")
        normalized = checked_affine(0, 1, array)
        frozen = np.frombuffer(normalized.tobytes(), dtype=np.intp).reshape(normalized.shape)
        object.__setattr__(self, "index_array", frozen)

    def __reduce__(self) -> tuple[object, tuple[object, int, int, IndexRange]]:
        """Reconstruct through `__init__`, preserving the ownership invariant."""
        return (
            type(self),
            (self.index_array, self.offset, self.stride, self.index_range),
        )

    def _with_affine(self, offset: int, stride: int) -> ArrayMap:
        """Return a map with a different affine adjustment.

        Share the immutable index array and the range while replacing the
        offset and stride. This preserves coordinate ownership without copying
        the array."""
        new = object.__new__(ArrayMap)
        object.__setattr__(new, "index_array", self.index_array)
        object.__setattr__(new, "offset", offset)
        object.__setattr__(new, "stride", stride)
        object.__setattr__(new, "index_range", self.index_range)
        return new

    def __eq__(self, other: object) -> bool:
        """Compare offset, stride, range, array shape, and index values.

        Return a scalar boolean for another ArrayMap and NotImplemented for
        other types."""
        if not isinstance(other, ArrayMap):
            return NotImplemented
        return (
            self.offset == other.offset
            and self.stride == other.stride
            and self.index_range == other.index_range
            and self.index_array.shape == other.index_array.shape
            and bool(np.array_equal(self.index_array, other.index_array))
        )

    def __hash__(self) -> int:
        """Hash the offset, stride, range, array shape, and index bytes.

        The immutable coordinate snapshot keeps the hash stable, and equal
        maps have equal hashes."""
        return hash(
            (
                self.offset,
                self.stride,
                self.index_range,
                self.index_array.shape,
                self.index_array.tobytes(),
            )
        )

    def checked_index_array(self, output_dimension: int) -> npt.NDArray[np.integer[Any]]:
        """The index array, after every value is checked against `index_range`.

        This is the gate through which index-array values reach storage.
        Intersection, chunk planning, composition through the array and the
        readers that lower a transform to array operations all read the array
        here, so a map holding a value outside its declared range fails at
        that use, not at construction or load, as TensorStore's does (see
        `IndexRange`). Point evaluation gathers only the values a point needs
        and checks those through `index_range.check`. `output_dimension` names
        the map in the `BoundsCheckError`.

        Examples
        --------
        >>> ArrayMap(np.array([3, 1]), index_range=IndexRange(0, 4)).checked_index_array(0)
        array([3, 1])
        """
        self.index_range.check(self.index_array, output_dimension=output_dimension)
        return self.index_array

    def with_index_array(
        self, index_array: npt.ArrayLike, *, output_dimension: int
    ) -> ArrayMap | ConstantMap:
        """This map over a reindexed selection of its own coordinates.

        `offset`, `stride` and `index_range` carry over: basic indexing and
        composition pick among the map's coordinates without changing what
        those coordinates are declared to satisfy, as in TensorStore
        ([`compose_transforms.cc#L244`](https://github.com/google/tensorstore/blob/5c6997751f4b4855de72d6e7f696c572aae3a4e5/tensorstore/index_space/internal/compose_transforms.cc#L244)).
        A result holding exactly one coordinate is the `ConstantMap` it equals;
        building that constant reads the coordinate, so the coordinate is
        checked against the range first, as TensorStore's collapse does
        ([`transform_rep.cc#L526-L531`](https://github.com/google/tensorstore/blob/5c6997751f4b4855de72d6e7f696c572aae3a4e5/tensorstore/index_space/internal/transform_rep.cc#L526-L531)).
        An empty result stays an `ArrayMap`, as in `array_map_or_constant`.

        Examples
        --------
        >>> m = ArrayMap(np.array([5, 1, 40]), offset=100, index_range=IndexRange(0, 9))
        >>> m.with_index_array(np.array([1, 5]), output_dimension=0)
        ArrayMap(index_array=array([1, 5]), offset=100, stride=1, index_range=IndexRange(inclusive_min=0, inclusive_max=9))
        >>> m.with_index_array(np.array([5]), output_dimension=0)
        ConstantMap(offset=105)
        >>> m.with_index_array(np.array([40]), output_dimension=0)
        Traceback (most recent call last):
            ...
        zarr_indexing.errors.BoundsCheckError: index 40 on output dimension 0 is outside index_array_bounds [0, 9]
        """
        arr = np.asarray(index_array)
        if arr.size == 1:
            self.index_range.check(arr, output_dimension=output_dimension)
            value = int(arr.reshape(-1)[0])
            return ConstantMap(offset=checked_affine(self.offset, self.stride, value))
        return ArrayMap(
            index_array=arr, offset=self.offset, stride=self.stride, index_range=self.index_range
        )

    @property
    def dependency_axes(self) -> tuple[int, ...]:
        """Structural dependency axes: axes of size greater than one.

        Axes of size greater than one are reported, regardless of coordinate
        values or a zero-size axis elsewhere. Whether the whole transform is
        orthogonal also depends on how other maps use these axes; a single
        map's shape does not establish independence.

        Examples
        --------
        >>> ArrayMap(index_array=np.array([[4, 0, 2]])).dependency_axes
        (1,)
        >>> ArrayMap(index_array=np.array([[1, 2], [3, 4]])).dependency_axes
        (0, 1)
        """
        return _array_map_dependency_axes(self.index_array)

    @property
    def dependent_axis(self) -> int | None:
        """Return the single input axis an orthogonal `ArrayMap` varies over.

        This is the array's one non-singleton axis, read from the shape — the
        single source of truth for what a map depends on. The selection layer
        collapses a single-coordinate map to a `ConstantMap`
        (`array_map_or_constant`), so a non-empty map built by this package always
        has at least one dependency axis.

        Returns
        -------
        int or None
            The axis the map varies over, or `None` when it varies over no input
            axis of size greater than one, such as an all-singleton map. `None`
            is a valid result, not an error; such maps resolve through the
            pointwise (general) path.

        Raises
        ------
        ValueError
            If the map varies over more than one axis, which makes it correlated
            rather than orthogonal.

        Examples
        --------
        An `oindex` selection on axis 1 of a rank-2 transform stores its
        coordinates full-sized on axis 1 and singleton on axis 0, so the
        dependency axis is read straight off the shape:

        >>> m = ArrayMap(index_array=np.array([[4, 0, 2]]))
        >>> m.index_array.shape
        (1, 3)
        >>> m.dependent_axis
        1
        """
        dep = self.dependency_axes
        if len(dep) == 1:
            return dep[0]
        if len(dep) == 0:
            return None
        raise ValueError(
            f"orthogonal ArrayMap must vary over exactly one axis; got dependency axes {dep}"
        )

    def to_json(self) -> OutputIndexMapJSON:
        """Convert to the canonical wire form, collapsing a degenerate map.

        A map holding exactly one coordinate, or none at all, is emitted as a
        `constant` map — see the module note on the wire format in
        [`zarr_indexing.json`][zarr_indexing.json]. Both are degenerate: the
        first selects one coordinate whatever the input, and the second names
        no cell and can only be empty because an input dimension is, so the
        emptiness travels in the domain instead.

        `index_array_bounds` is the stored `index_range`, always present, as
        ndsel's canonical form requires (spec section 4.3); TensorStore omits
        a range its values already satisfy, which the spec notes as its
        minimal encoding. A one-coordinate map whose coordinate lies outside
        its range is not collapsed: the constant it would become is not the
        map, which fails at use, so the array and its range are emitted, which
        is also what TensorStore emits for it.

        Examples
        --------
        >>> ArrayMap(np.array([[4], [1], [1]])).to_json()["index_array"]
        [[4], [1], [1]]
        >>> ArrayMap(np.array([7])).to_json()  # degenerate: one coordinate
        {'offset': 7}
        >>> ArrayMap(np.array([4, 1]), index_range=IndexRange(0, 9)).to_json()["index_array_bounds"]
        [0, 9]
        """
        if self.index_array.size == 1 and self.index_range.contains(self.index_array):
            value = int(self.index_array.reshape(-1)[0])
            return {"offset": self.offset + self.stride * value}
        if self.index_array.size == 0:
            return {"offset": 0}
        return {
            "offset": self.offset,
            "stride": self.stride,
            "index_array": self.index_array.tolist(),
            "index_array_bounds": self.index_range.to_json(),
        }


def output_index_map_from_json(data: OutputIndexMapJSON) -> OutputIndexMap:
    """Construct the output map a canonical wire form names.

    The wire form is structurally discriminated: the presence of `index_array`
    selects an array map, `input_dimension` selects a dimension map, and
    neither selects a constant map.

    `index_array_bounds` is lowered to the map's `index_range` as given; its
    syntax is validated, its relation to the array's values is not. A value
    outside the range loads, and fails at the use that reads it (see
    `IndexRange`).

    Examples
    --------
    >>> output_index_map_from_json({"offset": 5})
    ConstantMap(offset=5)
    >>> output_index_map_from_json({"offset": 0, "stride": 2, "input_dimension": 1})
    DimensionMap(input_dimension=1, offset=0, stride=2)
    >>> output_index_map_from_json({"index_array": [0, 40], "index_array_bounds": [0, 9]}).index_range
    IndexRange(inclusive_min=0, inclusive_max=9)
    """
    from zarr_indexing._wire import lower_index_array

    if "index_array" in data:
        array = lower_index_array(data["index_array"], "index_array")
        return ArrayMap(
            index_array=array,
            offset=data.get("offset", 0),
            stride=data.get("stride", 1),
            index_range=IndexRange.from_json(data.get("index_array_bounds", ["-inf", "+inf"])),
        )
    if "input_dimension" in data:
        return DimensionMap(
            input_dimension=data["input_dimension"],
            offset=data.get("offset", 0),
            stride=data.get("stride", 1),
        )
    return ConstantMap(offset=data.get("offset", 0))


def array_map_or_constant(
    index_array: npt.NDArray[np.integer[Any]],
    offset: int = 0,
    stride: int = 1,
) -> ArrayMap | ConstantMap:
    """An `ArrayMap`, collapsed to the `ConstantMap` it equals when it can be.

    An index array holding exactly one coordinate maps every input cell to the
    same place; representing it as a lookup table would leave a map whose shape
    names no dependency axis, the one form the shape-derived classifier cannot
    read. The selection and composition layers build their array maps through
    this helper so that a non-empty `ArrayMap` always varies over at least one
    axis. An empty array stays an `ArrayMap`: it maps no cell at all, and the
    emptiness lives in the domain that accompanies it.
    """
    arr = np.asarray(index_array)
    if arr.size == 1:
        return ConstantMap(offset=checked_affine(offset, stride, int(arr.reshape(-1)[0])))
    return ArrayMap(index_array=arr, offset=offset, stride=stride)


OutputIndexMap = ConstantMap | DimensionMap | ArrayMap
