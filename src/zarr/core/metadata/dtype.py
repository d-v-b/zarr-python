from __future__ import annotations

from typing import Literal

Endianness = Literal["little", "big"]


class Numpyable:
    def to_numpy(self, dtype: str) -> type:
        raise NotImplementedError


class DataType:
    byte_size: int | None
    endianness: Endianness | None
    name: str
