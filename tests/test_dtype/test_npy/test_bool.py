from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.test_dtype.test_wrapper import BaseTestZDType
from zarr.core.dtype.npy.bool import Bool

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


class TestBool(BaseTestZDType):
    test_cls = Bool

    valid_dtype = (np.dtype(np.bool_),)
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype(np.uint16),
    )
    valid_json_v2 = ({"name": "|b1", "object_codec_id": None},)
    valid_json_v3 = ("bool",)
    invalid_json_v2 = (
        {"name": "|b1", "object_codec_id": "vlen-utf8"},
        {"name": "bool", "object_codec_id": None},
        {"name": "|f8", "object_codec_id": None},
    )
    invalid_json_v3 = (
        "|b1",
        "|f8",
        {"name": "bool", "configuration": {"endianness": "little"}},
    )

    scalar_v2_params = ((Bool(), True), (Bool(), False))
    scalar_v3_params = ((Bool(), True), (Bool(), False))

    cast_value_params = (
        (Bool(), "true", np.True_),
        (Bool(), True, np.True_),
        (Bool(), False, np.False_),
        (Bool(), np.True_, np.True_),
        (Bool(), np.False_, np.False_),
    )
    invalid_scalar_params = (None,)
    item_size_params = (Bool(),)


@pytest.mark.parametrize(
    ("data", "zarr_format", "expected"),
    [
        (True, 3, np.True_),
        (False, 3, np.False_),
        (True, 2, np.True_),
        (False, 2, np.False_),
        # zarr 2.0 and 2.1 stored the fill value of a bool array as given, 0 by default
        (0, 2, np.False_),
        (1, 2, np.True_),
    ],
)
def test_from_json_scalar(data: JSON, zarr_format: ZarrFormat, expected: np.bool_) -> None:
    """
    A JSON boolean is read as that boolean, and so, in Zarr format 2, are the ints 0 and 1.
    """
    result = Bool().from_json_scalar(data, zarr_format=zarr_format)
    assert type(result) is np.bool_
    assert result == expected


@pytest.mark.parametrize(
    ("data", "zarr_format"),
    [
        ("false", 3),
        ("true", 3),
        ("false", 2),
        (None, 3),
        (0, 3),
        (1, 3),
        (2, 2),
        (-1, 2),
        (1.0, 2),
        (0.0, 3),
        ([], 3),
        ({}, 2),
    ],
)
def test_from_json_scalar_invalid(data: JSON, zarr_format: ZarrFormat) -> None:
    """
    Any other JSON value is rejected, rather than read by its truthiness.
    """
    with pytest.raises(TypeError, match="Expected a boolean"):
        Bool().from_json_scalar(data, zarr_format=zarr_format)
