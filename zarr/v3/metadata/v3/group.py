from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, asdict, field

import json

if TYPE_CHECKING:
    from zarr.v3.types import JSON, Attributes
    from typing import Any, Dict, Literal, Optional

def parse_format(data: Any) -> Literal[3]:
    if data != 3:
        msg = f'Expected 3, got {data}'
        raise ValueError(msg)
    return data

def parse_node_type(data: Any) -> Literal["group"]:
    if data != "group":
        msg = f"Expected 'group', got {data}"
        raise ValueError(msg)
    return data

@dataclass(frozen=True)
class GroupMetadata:
    attributes: Optional[Attributes] = field(default_factory=dict)
    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["group"] = field(default="group", init=False)

    def __init__(self, attributes: Dict[str, JSON]):
        object.__setattr__(self, "attributes", attributes)

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroupMetadata:
        format_parsed = parse_format(data["zarr_format"])
        node_type_parsed = parse_node_type(data["node_type"])
        attributes = data.get("attributes", None)
        return cls(attributes=attributes)
