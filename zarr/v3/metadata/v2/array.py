from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
import json
from typing import Any, Dict, List, Literal, Optional, Union

from zarr.v3.types import Attributes, ChunkCoords


@dataclass(frozen=True)
class ArrayMetadata:
    shape: ChunkCoords
    chunks: ChunkCoords
    dtype: np.dtype
    fill_value: Union[None, int, float] = 0
    order: Literal["C", "F"] = "C"
    filters: Optional[List[Dict[str, Any]]] = None
    dimension_separator: Literal[".", "/"] = "."
    compressor: Optional[Dict[str, Any]] = None
    zarr_format: Literal[2] = field(default=2, init=False)
    attributes: Attributes = field(default_factory=dict)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to_bytes(self) -> bytes:
        def _json_convert(o):
            if isinstance(o, np.dtype):
                if o.fields is None:
                    return o.str
                else:
                    return o.descr
            raise TypeError

        return json.dumps(asdict(self), default=_json_convert).encode()

    @classmethod
    def from_json(cls, zarr_json: Any) -> ArrayMetadata:
        return make_cattr().structure(zarr_json, cls)
