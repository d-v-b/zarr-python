from __future__ import annotations
from typing import Dict, List, Tuple, Union


BytesLike = Union[bytes, bytearray, memoryview]
ChunkCoords = Tuple[int, ...]
SliceSelection = Tuple[slice, ...]
Selection = Union[slice, SliceSelection]
JSON = Union[str, None, int, float, Dict[str, "JSON"], List["JSON"]]
Attributes = Dict[str, JSON]
