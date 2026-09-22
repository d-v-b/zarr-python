"""The first layer of reading: a value refined to JSON, or the reasons it is not."""

from __future__ import annotations

import math
from collections import OrderedDict

import pytest

from zarr_metadata.model import MetadataValidationError, ValidationProblem
from zarr_metadata.model._validation import refine_json


@pytest.mark.parametrize(
    ("value", "refined"),
    [
        (1, 1),
        (2.5, 2.5),
        (True, True),
        ("x", "x"),
        (None, None),
        ([1, [2, 3]], (1, (2, 3))),
        ((1, 2), (1, 2)),
        ({"a": [1], "b": {"c": None}}, {"a": (1,), "b": {"c": None}}),
        (OrderedDict(k=[0]), {"k": (0,)}),
    ],
    ids=["int", "float", "bool", "str", "null", "nested-lists", "tuple", "object", "any-mapping"],
)
def test_json_is_refined_to_tuples_and_dicts(value: object, refined: object) -> None:
    # One walk normalizes and judges; what comes back is what every later
    # layer takes, and nothing later normalizes again.
    assert refine_json(value) == (refined, ())


@pytest.mark.parametrize(
    ("value", "loc", "kind"),
    [
        ({"a": object()}, ("a",), "invalid_type"),
        ([1, [2, {3: 4}]], (1, 1), "invalid_type"),
        ({"x": math.inf}, ("x",), "invalid_value"),
        ({"x": [math.nan]}, ("x", 0), "invalid_value"),
        (b"bytes", (), "invalid_type"),
    ],
    ids=["not-json-leaf", "non-string-key", "infinite", "nan-in-array", "bytes"],
)
def test_error_a_value_that_is_not_json_is_none_with_the_leaf_located(
    value: object, loc: tuple[str | int, ...], kind: str
) -> None:
    refined, problems = refine_json(value)
    assert refined is None
    assert [(problem.loc, problem.kind) for problem in problems] == [(loc, kind)]


def test_error_every_leaf_that_is_not_json_is_reported() -> None:
    refined, problems = refine_json({"a": object(), "b": [1, object()]})
    assert refined is None
    assert [problem.loc for problem in problems] == [("a",), ("b", 1)]


def test_error_the_error_refuses_what_is_not_a_problem() -> None:
    # `problem()` in the entity layer returns a one-element tuple; a list
    # of those passes the type at the call and fails far away otherwise.
    with pytest.raises(TypeError, match="collect with `extend`, not `append`"):
        MetadataValidationError([(ValidationProblem(("a",), "bad a", "invalid_value"),)])  # pyright: ignore[reportArgumentType]
