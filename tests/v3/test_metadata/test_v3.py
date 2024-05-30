from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr.metadata.v3 import parse_dimension_names, parse_zarr_format

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


# todo: test
def test_datatype_enum(): ...


# todo: test
# this will almost certainly be a collection of tests
def test_array_metadata(): ...


@pytest.mark.parametrize("data", [None, ("a", "b", "c"), ["a", "a", "a"]])
def parse_dimension_names_valid(data: Sequence[str] | None) -> None:
    assert parse_dimension_names(data) == data


@pytest.mark.parametrize("data", [(), [1, 2, "a"], {"foo": 10}])
def parse_dimension_names_invalid(data: Any) -> None:
    with pytest.raises(TypeError, match="Expected either None or iterable of str,"):
        parse_dimension_names(data)


# todo: test
def test_parse_attributes() -> None: ...


def test_parse_zarr_format_valid() -> None:
    assert parse_zarr_format(3) == 3


@pytest.mark.parametrize("data", [None, 1, 2, 4, 5, "3"])
def test_parse_zarr_format_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match=f"Invalid value. Expected 3. Got {data}"):
        parse_zarr_format(data)
