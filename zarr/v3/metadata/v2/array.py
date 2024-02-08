from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING
import numpy as np
import json
from zarr.v3.common import parse_shape

if TYPE_CHECKING:   
    from typing import Any, Dict, List, Literal, Optional, Union, Tuple, Iterable
    from zarr.v3.abc.codec import Codec
    import numpy.typing as npt

from zarr.v3.types import Attributes, ChunkCoords

# todo: ensure consistency with chunks
def parse_chunks(data: Iterable[int]) -> Tuple[int]:
    return tuple(map(int, data))

def parse_dtype(data: npt.DTypeLike) -> np.dtype:
    return np.dtype(data)

def parse_order(data: Literal["C", "F"]) -> Literal["C", "F"]:
    if data in ("C", "F"):
        return data
    
    msg = f'Expected one of ("C", "F") got {data}'
    raise ValueError(msg)

# todo: actual validation
def parse_filters(data: Optional[List[Codec]]) -> Optional[List[Codec]]:
    if data is not None:
        is_codecs = list(map(lambda v: isinstance(v, Codec), data))
        if not all(is_codecs):
            raise ValueError('Not all of elements in data are instances of `Codec`')
    return data

# todo: actual validation
def parse_compressor(data: Optional[Codec]) -> Optional[Codec]:
    return data

def parse_fill_value(data: Any) -> Any:
    return data

def parse_dimension_separator(data: Optional[Literal[".", "/"]]) -> Literal[".", "/"]:
    if data is None:
        return "/"
    if data not in (".", "/"):
        msg = f'Invalid dimension_separator. Expected one of (`None`, "/", or "."), got {type(data)}'
        raise ValueError(msg)

def parse_shape_chunks(shape: Tuple[int, ...], chunks: Tuple[int, ...]) -> Tuple[Tuple[int,...], Tuple[int, ...]]
    if len(shape) != len(chunks):
        msg = f'Length of chunks ({len(chunks)}) does not match length of shape ({len(shape)}).'
        raise ValueError(msg)
    return shape, chunks

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


    def __init__(
            self, 
            shape: Iterable[int], 
            chunks: Iterable[int], 
            dtype: npt.DtypeLike, 
            fill_value: Any, 
            order: Literal["C", "F"], 
            filters: Optional[List[Codec]], 
            dimension_separator: Optional[Literal["/", "."]],
            compressor: Optional[Codec]):

        shape_parsed = parse_shape(shape)
        chunks_parsed = parse_chunks(chunks)
        shape_parsed, chunks_parsed = parse_shape_chunks(shape, chunks)
        dtype_parsed = parse_dtype(dtype)
        fill_value_parsed = parse_fill_value(fill_value)
        order_parsed = parse_order(order)
        filters_parsed = parse_filters(filters)
        dimension_separator_parsed = parse_dimension_separator(dimension_separator)
        compressor_parsed = parse_compressor(compressor)

        object.__setattr__(self, 'shape', shape_parsed)
        object.__setattr__(self, 'chunks', chunks_parsed)
        object.__setattr__(self, 'dtype', dtype_parsed)
        object.__setattr__(self, 'fill_value', fill_value_parsed)
        object.__setattr__(self, 'order', order_parsed)
        object.__setattr__(self, 'filters', filters_parsed)
        object.__setattr__(self, 'dimension_separator', dimension_separator_parsed)
        object.__setattr__(self, 'compressor', compressor_parsed)
        

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
    def from_dict(cls, data: Dict[str, Any]) -> ArrayMetadata:
        return cls()
