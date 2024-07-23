from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from typing_extensions import Self


def get_order(data: Any) -> Literal["C", "F"]:
    if hasattr(data, "order"):
        if data.order == "C":
            return "C"
        elif data.order == "F":
            return "F"
    elif hasattr(data, "flags"):
        if data.flags["C_CONTIGUOUS"]:
            return "C"
        if data.flags["F_CONTIGUOUS"]:
            return "F"
        else:
            raise ValueError
    raise ValueError


def get_fill_value(data: Any) -> Any:
    if hasattr(data, "fill_value"):
        return data.fill_value
    raise ValueError


@dataclass
class ArrayModel:
    """
    Model an array with a fill value
    """

    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    order: Literal["C", "F"]
    fill_value: Any

    @classmethod
    def from_array(
        cls,
        array: Any,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype[Any] | None = None,
        order: Literal["C", "F"] | None = None,
        fill_value: Any | None = None,
    ) -> Self:
        if shape is None:
            shape_out = array.shape
        else:
            shape_out = shape

        if dtype is None:
            dtype_out = array.dtype
        else:
            dtype_out = dtype

        if order is None:
            order_out = get_order(array)
        else:
            order_out = order

        if fill_value is None:
            fill_value_out = get_fill_value(fill_value)
        else:
            fill_value_out = fill_value

        return cls(shape=shape_out, dtype=dtype_out, order=order_out, fill_value=fill_value_out)

    @property
    def ndim(self) -> int:
        return len(self.shape)
