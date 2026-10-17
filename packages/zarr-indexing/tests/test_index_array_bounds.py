"""`index_array_bounds` is retained as `ArrayMap.index_range` and checked at use.

The behaviour is TensorStore's, established against tensorstore 0.1.85 and its
source at the commit `IndexRange` cites: a document whose `index_array` holds a
value outside finite `index_array_bounds` loads; the value fails with an
out-of-range error when a point that reads it is evaluated, when the array is
iterated to address storage, or when a one-element map collapses to a
constant; a selection that avoids the value never trips over it; and the
range travels unchanged through slicing, translation and affine composition.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from zarr_indexing import (
    ArrayMap,
    BoundsCheckError,
    IndexDomain,
    IndexRange,
    IndexTransform,
    NdselError,
    ReadContext,
    basic_reader,
    dimension_grids_from_chunks,
    normalize_ndsel,
    numpy_reader,
    output_index_map_from_json,
    plan_chunks,
    unit_step_reader,
)
from zarr_indexing.writer import write_into

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr_indexing.json import IndexTransformJSON, OutputIndexMapJSON

INTP_MIN = int(np.iinfo(np.intp).min)
INTP_MAX = int(np.iinfo(np.intp).max)


def body_for(values: Any, bounds: Any, offset: int = 0, stride: int = 1) -> IndexTransformJSON:
    output: OutputIndexMapJSON = {"index_array": values, "offset": offset, "stride": stride}
    if bounds is not None:
        output["index_array_bounds"] = bounds
    return {"input_shape": list(np.asarray(values).shape), "output": [output]}


def load(values: Any, bounds: Any, entry: str, offset: int = 0, stride: int = 1) -> Any:
    """Load through either engine entry point, as the eager check once did."""
    body = body_for(values, bounds, offset, stride)
    if entry == "output_map":
        return output_index_map_from_json(body["output"][0])
    return IndexTransform.from_json(body)


def outlier_transform() -> IndexTransform:
    """The probe's transform: values `[0, 5, 20]`, declared range `[0, 10]`."""
    return IndexTransform.from_json(body_for([0, 5, 20], [0, 10]))


OUTLIER_MESSAGE = r"index 20 on output dimension 0 is outside index_array_bounds \[0, 10\]"


# --------------------------------------------------------------------------- #
# IndexRange
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("lo", "hi", "values", "inside"),
    [
        ("-inf", "+inf", [INTP_MIN, INTP_MAX], True),
        ("-inf", 4, [-3, 0, 4], True),
        ("-inf", 4, [5], False),
        (-3, "+inf", [-3, 0, 4], True),
        (-3, "+inf", [-4], False),
        (-3, 4, [-3, 0, 4], True),
        (-3, 4, [-3, 0, 5], False),
        (2, 2, [2, 2], True),
        (2, 2, [2, 3], False),
        (0, 9, [], True),
        (INTP_MIN, INTP_MIN, [INTP_MIN], True),
        (INTP_MIN + 1, "+inf", [INTP_MIN], False),
        ("-inf", INTP_MAX - 1, [INTP_MAX], False),
        (0, 9, [[7, 3], [5, 9]], True),
        (0, 9, [[7, 3], [5, 10]], False),
    ],
)
def test_index_range_contains(lo: Any, hi: Any, values: Any, inside: bool) -> None:
    index_range = IndexRange(lo, hi)
    assert index_range.contains(np.asarray(values, dtype=np.intp)) is inside
    assert index_range.is_unbounded is (lo == "-inf" and hi == "+inf")
    assert index_range.to_json() == [lo, hi]
    assert IndexRange.from_json([lo, hi]) == index_range
    if inside:
        index_range.check(np.asarray(values, dtype=np.intp), output_dimension=3)
    else:
        with pytest.raises(BoundsCheckError, match=r"on output dimension 3 is outside"):
            index_range.check(np.asarray(values, dtype=np.intp), output_dimension=3)


def test_index_range_rejects_reversed_finite_bounds() -> None:
    with pytest.raises(ValueError, match="not a closed interval"):
        IndexRange(5, 4)


def test_index_range_rejects_positive_infinity_as_lower_bound() -> None:
    with pytest.raises(ValueError, match="inclusive_min must be an integer or '-inf'"):
        IndexRange("+inf", "+inf")


def test_index_range_rejects_negative_infinity_as_upper_bound() -> None:
    with pytest.raises(ValueError, match="inclusive_max must be an integer or '\\+inf'"):
        IndexRange("-inf", "-inf")


@pytest.mark.parametrize("bad", [True, 1.5, "1", None])
def test_index_range_rejects_non_integer_bound(bad: Any) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        IndexRange(bad, 3)


def test_index_range_from_json_rejects_a_sentinel_on_the_wrong_side() -> None:
    """Ordered in the extended integers, yet no closed interval: TensorStore
    rejects `(+inf, +inf)` at parse, and so does lowering."""
    for bounds in (["+inf", "+inf"], ["-inf", "-inf"]):
        with pytest.raises(NdselError, match="index_array_bounds") as exc:
            IndexRange.from_json(bounds)
        assert exc.value.reason == "invalid_json"


@pytest.mark.parametrize("bounds", [[], [0], [0, 1, 2], [False, 2], [0.5, 2], ["bad", 2], 3])
def test_index_range_from_json_rejects_malformed_bounds(bounds: Any) -> None:
    with pytest.raises(NdselError) as exc:
        IndexRange.from_json(bounds)
    assert exc.value.reason == "invalid_json"


def test_index_range_from_json_rejects_reversed_bounds() -> None:
    with pytest.raises(NdselError) as exc:
        IndexRange.from_json([2, 0])
    assert exc.value.reason == "bounds_out_of_order"


def test_array_map_rejects_a_range_that_is_not_an_index_range() -> None:
    with pytest.raises(TypeError, match="index_range must be an IndexRange"):
        ArrayMap(np.array([1, 2]), index_range=[0, 9])


# --------------------------------------------------------------------------- #
# Loading retains the bounds and never looks at the values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("entry", ["transform", "output_map"])
@pytest.mark.parametrize(("offset", "stride"), [(0, 1), (10, -2), (7, 0)])
@pytest.mark.parametrize(
    ("values", "bounds"),
    [
        ([-3, 0, 4], [-3, 4]),
        ([-3, 0, 4], ["-inf", 4]),
        ([-3, 0, 4], [-3, "+inf"]),
        ([-3, 0, 4], ["-inf", "+inf"]),
        ([-3, 0, 4], None),
        ([[0, 2], [2, 0]], [0, 2]),
        ([], [2, 2]),
        # Outside the declared range: loads all the same.
        ([-1, 0], [0, 2]),
        ([0, 3], [0, 2]),
        ([-1, 5], [0, "+inf"]),
        ([3, 1], ["-inf", 2]),
        ([[0, 1], [1, 3]], [0, 2]),
        ([INTP_MIN, INTP_MAX], [INTP_MIN + 1, INTP_MAX - 1]),
    ],
)
def test_load_retains_bounds_and_round_trips(
    values: Any, bounds: Any, entry: str, offset: int, stride: int
) -> None:
    loaded = load(values, bounds, entry, offset, stride)
    expected_range = IndexRange.from_json(bounds) if bounds is not None else IndexRange()
    array_map = loaded.output[0] if entry == "transform" else loaded
    assert isinstance(array_map, ArrayMap)
    assert array_map.index_range == expected_range
    np.testing.assert_array_equal(array_map.index_array.reshape(-1), np.asarray(values).reshape(-1))

    emitted = loaded.to_json()
    map_json = emitted["output"][0] if entry == "transform" else emitted
    restored = (
        IndexTransform.from_json(emitted)
        if entry == "transform"
        else output_index_map_from_json(map_json)
    )
    if np.asarray(values).size == 0:
        # An empty map collapses to a constant on the wire (the emptiness
        # travels in the domain), so it round-trips by behaviour, not identity.
        assert map_json == {"offset": 0}
    else:
        assert map_json["index_array_bounds"] == expected_range.to_json()
        assert restored == loaded
    assert restored.to_json() == emitted


@pytest.mark.parametrize("entry", ["transform", "output_map"])
def test_singleton_outside_its_range_is_not_collapsed_to_a_constant(entry: str) -> None:
    """A constant `70` would be a valid map; the loaded one is not, so
    serialization keeps the array and its range, as TensorStore emits it."""
    inside = load([7], [0, 10], entry)
    outside = load([70], [0, 10], entry)
    inside_json = inside.to_json()["output"][0] if entry == "transform" else inside.to_json()
    outside_json = outside.to_json()["output"][0] if entry == "transform" else outside.to_json()
    assert inside_json == {"offset": 7}
    assert outside_json == {
        "offset": 0,
        "stride": 1,
        "index_array": [70],
        "index_array_bounds": [0, 10],
    }
    reloaded = (
        IndexTransform.from_json(outside.to_json())
        if entry == "transform"
        else output_index_map_from_json(outside_json)
    )
    assert reloaded == outside


def test_normalize_is_idempotent_with_finite_bounds() -> None:
    message = {
        "kind": "transform",
        "input_shape": [3],
        "output": [{"index_array": [0, 5, 20], "index_array_bounds": [0, 10]}],
    }
    canonical = normalize_ndsel(message)
    assert canonical["output"][0]["index_array_bounds"] == [0, 10]
    assert normalize_ndsel({**canonical, "kind": "transform"}) == canonical
    assert IndexTransform.from_json(canonical).to_json() == canonical


# --------------------------------------------------------------------------- #
# The value fails at the use that reads it
# --------------------------------------------------------------------------- #


def test_apply_checks_only_the_values_it_gathers() -> None:
    transform = outlier_transform()
    assert transform.apply((0,)) == (0,)
    assert transform.apply((1,)) == (5,)
    with pytest.raises(BoundsCheckError, match=OUTLIER_MESSAGE):
        transform.apply((2,))
    np.testing.assert_array_equal(transform.apply_many([[0], [1]]), [[0], [5]])
    with pytest.raises(BoundsCheckError, match=OUTLIER_MESSAGE):
        transform.apply_many([[0], [1], [2]])


def test_bounds_apply_to_raw_values_before_offset_and_stride() -> None:
    """`100 + 2 * 20 = 140` is a fine coordinate; the raw `20` is the outlier."""
    transform = IndexTransform.from_json(body_for([0, 5, 20], [0, 10], offset=100, stride=2))
    assert transform.apply((1,)) == (110,)
    with pytest.raises(BoundsCheckError, match=OUTLIER_MESSAGE):
        transform.apply((2,))


def _intersect(transform: IndexTransform) -> Any:
    return transform.intersect(IndexDomain.from_shape((100,)))


def _plan(transform: IndexTransform) -> Any:
    return list(plan_chunks(transform, dimension_grids_from_chunks((4,), shape=(100,))))


def _partition(transform: IndexTransform) -> Any:
    return plan_chunks(transform, dimension_grids_from_chunks((4,), shape=(100,))).partition()


def _read(reader: Any) -> Callable[[IndexTransform], Any]:
    def read(transform: IndexTransform) -> Any:
        out = np.empty(transform.domain.shape, dtype=np.intp)
        reader.read_into(np.arange(100), ReadContext(transform), out)
        return out

    return read


def _write(transform: IndexTransform) -> Any:
    source = np.zeros(100, dtype=np.intp)
    write_into(source, transform, 1)
    return source


def _gather_through(transform: IndexTransform) -> Any:
    """The map's values address another index array (composition)."""
    inner = IndexTransform(IndexDomain.from_shape((100,)), (ArrayMap(np.arange(100) * 3),))
    return transform.compose(inner)


USES: dict[str, Callable[[IndexTransform], Any]] = {
    "intersect": _intersect,
    "plan_chunks": _plan,
    "partition": _partition,
    "basic_reader": _read(basic_reader),
    "numpy_reader": _read(numpy_reader),
    "unit_step_reader": _read(unit_step_reader),
    "write_into": _write,
    "gather_through": _gather_through,
}


@pytest.mark.parametrize("use", list(USES), ids=list(USES))
def test_uses_that_read_every_value_raise(use: str) -> None:
    with pytest.raises(BoundsCheckError, match=OUTLIER_MESSAGE):
        USES[use](outlier_transform())


@pytest.mark.parametrize("use", list(USES), ids=list(USES))
def test_uses_ignore_a_range_every_value_satisfies(use: str) -> None:
    bounded = IndexTransform.from_json(body_for([0, 5, 20], [0, 20]))
    unbounded = IndexTransform.from_json(body_for([0, 5, 20], None))
    bounded_result = USES[use](bounded)
    unbounded_result = USES[use](unbounded)
    if isinstance(bounded_result, np.ndarray):
        np.testing.assert_array_equal(bounded_result, unbounded_result)
    elif use == "gather_through":
        assert [bounded_result.apply((i,)) for i in range(3)] == [
            unbounded_result.apply((i,)) for i in range(3)
        ]
    elif use == "intersect":
        assert bounded_result[0].domain == unbounded_result[0].domain
    elif use == "partition":
        assert bounded_result.chunk_coords().tolist() == unbounded_result.chunk_coords().tolist()
    else:
        assert [p.chunk_coords for p in bounded_result] == [
            p.chunk_coords for p in unbounded_result
        ]


def test_correlated_maps_are_checked_where_the_block_is_read() -> None:
    body: IndexTransformJSON = {
        "input_shape": [3],
        "output": [
            {"index_array": [0, 1, 2], "index_array_bounds": [0, 9]},
            {"index_array": [4, 40, 2], "index_array_bounds": [0, 9]},
        ],
    }
    transform = IndexTransform.from_json(body)
    assert transform.apply((0,)) == (0, 4)
    message = r"index 40 on output dimension 1 is outside index_array_bounds \[0, 9\]"
    with pytest.raises(BoundsCheckError, match=message):
        transform.apply((1,))
    with pytest.raises(BoundsCheckError, match=message):
        transform.intersect(IndexDomain.from_shape((10, 10)))
    with pytest.raises(BoundsCheckError, match=message):
        list(plan_chunks(transform, dimension_grids_from_chunks((2, 2), shape=(10, 10))))
    out = np.empty((3,), dtype=np.intp)
    with pytest.raises(BoundsCheckError, match=message):
        numpy_reader.read_into(np.arange(100).reshape(10, 10), ReadContext(transform), out)


# --------------------------------------------------------------------------- #
# The range travels; a selection that avoids the value never meets it
# --------------------------------------------------------------------------- #


def test_slicing_away_the_outlier_leaves_a_usable_transform() -> None:
    narrowed = outlier_transform()[0:2]
    array_map = narrowed.output[0]
    assert isinstance(array_map, ArrayMap)
    assert array_map.index_range == IndexRange(0, 10)
    assert narrowed.to_json()["output"][0]["index_array_bounds"] == [0, 10]
    assert [narrowed.apply((i,)) for i in range(2)] == [(0,), (5,)]
    for use in USES.values():
        use(narrowed)


def test_integer_index_collapses_an_inlier_and_refuses_the_outlier() -> None:
    transform = outlier_transform()
    assert transform[1].to_json()["output"] == [{"offset": 5}]
    with pytest.raises(BoundsCheckError, match=OUTLIER_MESSAGE):
        transform[2]
    with pytest.raises(BoundsCheckError, match=OUTLIER_MESSAGE):
        transform[2:3]


def test_fancy_selection_over_an_array_map_keeps_its_range() -> None:
    transform = outlier_transform()
    avoided = transform.oindex[[1, 0]]
    assert [avoided.apply((i,)) for i in range(2)] == [(5,), (0,)]
    kept = transform.oindex[[0, 2]]
    kept_map = kept.output[0]
    assert isinstance(kept_map, ArrayMap)
    assert kept_map.index_range == IndexRange(0, 10)
    assert kept.apply((0,)) == (0,)
    with pytest.raises(BoundsCheckError, match=OUTLIER_MESSAGE):
        kept.apply((1,))
    with pytest.raises(BoundsCheckError, match=OUTLIER_MESSAGE):
        transform.oindex[[2]]


@pytest.mark.parametrize(
    "operation",
    [
        lambda t: t.translate((3,)),
        lambda t: t.translate_domain_by((3,)),
        lambda t: t.translate_domain_to((-1,)),
        lambda t: t.compose(IndexTransform.from_shape((100,))[1::2]),
    ],
    ids=["translate", "translate_domain_by", "translate_domain_to", "compose_affine"],
)
def test_affine_operations_keep_the_range(
    operation: Callable[[IndexTransform], IndexTransform],
) -> None:
    moved = operation(outlier_transform())
    array_map = moved.output[0]
    assert isinstance(array_map, ArrayMap)
    assert array_map.index_range == IndexRange(0, 10)
    assert moved.to_json()["output"][0]["index_array_bounds"] == [0, 10]
    origin = moved.domain.inclusive_min[0]
    moved.apply((origin,))
    with pytest.raises(BoundsCheckError, match=OUTLIER_MESSAGE):
        moved.apply((origin + 2,))


def test_composition_defers_a_value_inside_the_next_domain_but_outside_its_range() -> None:
    """Probe 2(b): composing an affine map onto the array proves the values
    lie in the domain, and leaves the map's own declaration to fail at use."""
    first = IndexTransform(
        IndexDomain.from_shape((3,)), (ArrayMap(np.array([0, 1, 2]), index_range=IndexRange(0, 1)),)
    )
    composed = first.compose(IndexTransform.from_shape((3,)))
    assert composed.apply((1,)) == (1,)
    with pytest.raises(BoundsCheckError, match=r"index 2 on output dimension 0"):
        composed.apply((2,))
    # Probe 2(a): gathering through the same map reads its values.
    with pytest.raises(BoundsCheckError, match=r"index 2 on output dimension 0"):
        first.compose(
            IndexTransform(IndexDomain.from_shape((3,)), (ArrayMap(np.array([7, 8, 9])),))
        )


def test_equality_hash_and_pickle_include_the_range() -> None:
    values = np.array([1, 3, 5])
    plain = ArrayMap(values)
    bounded = ArrayMap(values, index_range=IndexRange(0, 9))
    assert plain != bounded
    assert bounded == ArrayMap(values, index_range=IndexRange(0, 9))
    assert hash(bounded) == hash(ArrayMap(values, index_range=IndexRange(0, 9)))
    assert pickle.loads(pickle.dumps(bounded)) == bounded
    assert bounded._with_affine(2, 3).index_range == IndexRange(0, 9)


def test_selection_built_maps_are_unbounded() -> None:
    transform = IndexTransform.from_shape((10, 10)).vindex[np.array([1, 2]), np.array([3, 4])]
    for output_map in transform.output:
        assert isinstance(output_map, ArrayMap)
        assert output_map.index_range == IndexRange()
        assert output_map.to_json()["index_array_bounds"] == ["-inf", "+inf"]


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #


@st.composite
def bounded_index_arrays(draw: st.DrawFn) -> tuple[np.ndarray[Any, np.dtype[np.intp]], Any, Any]:
    """An index array of rank 1 or 2 with a well-formed `index_array_bounds`.

    Half the draws pick bounds that contain every value and half pick bounds
    that exclude at least one, so both branches of every property are reached
    (see the `event` calls).
    """
    rank = draw(st.integers(min_value=1, max_value=2))
    shape = tuple(draw(st.integers(min_value=1, max_value=4)) for _ in range(rank))
    flat = draw(
        st.lists(
            st.integers(min_value=-20, max_value=20),
            min_size=int(np.prod(shape)),
            max_size=int(np.prod(shape)),
        )
    )
    values = np.asarray(flat, dtype=np.intp).reshape(shape)
    low, high = int(values.min()), int(values.max())
    contain = draw(st.booleans())
    if contain:
        lo = draw(st.one_of(st.just("-inf"), st.integers(min_value=-30, max_value=low)))
        hi = draw(st.one_of(st.just("+inf"), st.integers(min_value=high, max_value=30)))
    else:
        side = draw(st.sampled_from(["below", "above"]))
        if side == "below":
            lo = draw(st.integers(min_value=low + 1, max_value=30))
            hi = draw(st.one_of(st.just("+inf"), st.integers(min_value=lo, max_value=40)))
        else:
            hi = draw(st.integers(min_value=-30, max_value=high - 1))
            lo = draw(st.one_of(st.just("-inf"), st.integers(min_value=-40, max_value=hi)))
    event(f"rank {rank}")
    event("bounds contain every value" if contain else "bounds exclude a value")
    infinite_sides = sum(isinstance(bound, str) for bound in (lo, hi))
    event(("finite", "one-sided", "unbounded")[infinite_sides])
    event("single element" if values.size == 1 else "several elements")
    return values, lo, hi


@settings(max_examples=300)
@given(bounded_index_arrays())
def test_loading_never_rejects_on_values_and_round_trips(
    case: tuple[np.ndarray[Any, np.dtype[np.intp]], Any, Any],
) -> None:
    values, lo, hi = case
    body = normalize_ndsel(
        {
            "kind": "transform",
            "input_shape": list(values.shape),
            "output": [{"index_array": values.tolist(), "index_array_bounds": [lo, hi]}],
        }
    )
    assert normalize_ndsel({**body, "kind": "transform"}) == body
    transform = IndexTransform.from_json(body)
    array_map = transform.output[0]
    assert isinstance(array_map, ArrayMap)
    assert array_map.index_range == IndexRange(lo, hi)
    np.testing.assert_array_equal(array_map.index_array, values)

    emitted = transform.to_json()
    reloaded = IndexTransform.from_json(emitted)
    if values.size == 1 and array_map.index_range.contains(values):
        # An in-range one-element map is the constant it collapses to on the
        # wire; it round-trips by behaviour rather than identity.
        assert emitted["output"][0] == {"offset": int(values.reshape(-1)[0])}
        origin = tuple(0 for _ in values.shape)
        assert reloaded.apply(origin) == transform.apply(origin)
    else:
        assert emitted == body
        assert reloaded == transform


@settings(max_examples=300)
@given(bounded_index_arrays())
def test_uses_raise_exactly_when_a_read_value_is_outside(
    case: tuple[np.ndarray[Any, np.dtype[np.intp]], Any, Any],
) -> None:
    values, lo, hi = case
    index_range = IndexRange(lo, hi)
    transform = IndexTransform(
        IndexDomain.from_shape(values.shape), (ArrayMap(values, index_range=index_range),)
    )
    points = np.array(list(np.ndindex(values.shape)), dtype=np.intp).reshape(-1, values.ndim)

    for point in points:
        value = int(values[tuple(point)])
        if index_range.contains([value]):
            assert transform.apply(tuple(point)) == (value,)
        else:
            with pytest.raises(BoundsCheckError, match=rf"index {value} on output dimension 0"):
                transform.apply(tuple(point))

    grids = dimension_grids_from_chunks((8,), shape=(64,))
    shifted = transform.translate((32,))  # every storage coordinate lands in [12, 52]
    if index_range.contains(values):
        np.testing.assert_array_equal(shifted.apply_many(points).reshape(values.shape), values + 32)
        assert shifted.intersect(IndexDomain.from_shape((64,))) is not None
        assert len(list(plan_chunks(shifted, grids))) > 0
    else:
        with pytest.raises(BoundsCheckError):
            shifted.apply_many(points)
        with pytest.raises(BoundsCheckError):
            shifted.intersect(IndexDomain.from_shape((64,)))
        with pytest.raises(BoundsCheckError):
            list(plan_chunks(shifted, grids))
