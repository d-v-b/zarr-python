from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr.core.common import JSON


def parse_attributes(data: Mapping[str, Any] | None) -> dict[str, JSON]:
    if data is None:
        return {}

    return dict(data)
