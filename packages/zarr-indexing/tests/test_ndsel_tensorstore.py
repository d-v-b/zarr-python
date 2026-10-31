"""Cross-check canonical ndsel bodies against a real TensorStore.

Canonical ndsel bodies use TensorStore's `IndexTransform` field vocabulary,
but the consumers have different validation constraints. This test loads a
handful of finite-bound canonical bodies supported by both implementations into
`tensorstore.IndexTransform(json=...)` and confirms that TensorStore's own
`to_json()` re-loads, through our engine layer, into an equivalent transform.

Skipped when tensorstore is not installed. Run it explicitly with:

    hatch run test.py3.12-optional:pytest \
        packages/zarr-indexing/tests/test_ndsel_tensorstore.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_indexing.errors import BoundsCheckError
from zarr_indexing.messages import normalize_ndsel
from zarr_indexing.transform import IndexTransform

ts = pytest.importorskip("tensorstore")


def _canonical_transforms() -> list[IndexTransform]:
    base = IndexTransform.from_shape((10, 20))
    return [
        base,  # identity
        base[2:8:2, :],  # strided DimensionMap + identity
        base[3, :],  # integer index -> ConstantMap + DimensionMap
        base.oindex[np.array([1, 5, 9]), :],  # orthogonal index_array
        IndexTransform.from_shape((10, 20, 30)).vindex[
            np.array([1, 3]), np.array([2, 4]), :
        ],  # correlated index_arrays + residual slice
    ]


@pytest.mark.parametrize("transform", _canonical_transforms())
def test_body_loads_in_tensorstore_and_round_trips(transform: IndexTransform) -> None:
    body = transform.to_json()

    # (1) The canonical body loads directly as a TensorStore IndexTransform.
    ts_transform = ts.IndexTransform(json=body)

    # (2) TensorStore's own JSON re-loads, through our engine, to an equivalent
    #     transform. Comparing via our canonical form normalizes away
    #     representational choices (index_array_bounds, default omissions) that
    #     both sides make differently but that denote the same selection.
    ts_json = ts_transform.to_json()
    reloaded = IndexTransform.from_json(ts_json)
    assert reloaded.to_json() == transform.to_json()


@pytest.mark.parametrize(
    ("values", "bounds", "outside"),
    [
        ([0, 5, 20], [0, 10], {2}),
        ([0, -3, 2], [0, "+inf"], {1}),
        ([5, 6, 5], [5, 5], {1}),
        ([0, 5, 2], [-100, 100], set()),
    ],
    ids=["above", "below-one-sided", "degenerate", "all-inside"],
)
def test_index_array_bounds_are_checked_at_use_as_in_tensorstore(
    values: list[int], bounds: list[object], outside: set[int]
) -> None:
    """Both engines load an array whose values leave its bounds, agree point by
    point, and TensorStore's own encoding reloads here with the range intact.

    The point evaluations that TensorStore refuses with `OUT_OF_RANGE` are the
    ones this engine refuses with `BoundsCheckError`; the rest agree."""
    body = normalize_ndsel(
        {
            "kind": "transform",
            "input_shape": [len(values)],
            "output": [{"index_array": values, "index_array_bounds": bounds}],
        }
    )
    ours = IndexTransform.from_json(body)
    theirs = ts.IndexTransform(json=body)
    for point in range(len(values)):
        if point in outside:
            with pytest.raises(BoundsCheckError, match="index_array_bounds"):
                ours.apply((point,))
            with pytest.raises(ValueError, match="OUT_OF_RANGE"):
                theirs([point])
        else:
            assert ours.apply((point,)) == tuple(theirs([point]))
    # TensorStore's minimal encoding drops a range its values satisfy and keeps
    # one they leave (spec section 2.3); what it keeps reloads here intact, and
    # what it drops reloads as the unbounded default.
    kept = (
        body["output"][0]
        if outside
        else {**body["output"][0], "index_array_bounds": ["-inf", "+inf"]}
    )
    assert IndexTransform.from_json(theirs.to_json()).to_json() == {**body, "output": [kept]}
    # Slicing the outlier away leaves a transform both engines evaluate.
    if outside:
        keep = [p for p in range(len(values)) if p not in outside]
        first, last = keep[0], keep[-1] + 1
        assert [ours[first:last].apply((p,)) for p in keep] == [
            tuple(theirs[ts.d[0][first:last]]([p])) for p in keep
        ]
