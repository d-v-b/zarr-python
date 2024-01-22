from __future__ import annotations
from dataclasses import dataclass, asdict, field

import json
from typing import Any, Dict, Literal
from zarr.v3.types import Attributes


@dataclass(frozen=True)
class GroupMetadata:
    attributes: Attributes = field(default_factory=dict)
    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["group"] = field(default="group", init=False)

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_json(cls, zarr_json: Any) -> GroupMetadata:
        return make_cattr().structure(zarr_json, GroupMetadata)
