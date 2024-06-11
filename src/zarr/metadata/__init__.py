from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any

import numpy as np

def jsonify_dtype(
    o: np.dtype[Any],
) -> str | list[tuple[str, str] | tuple[str, str, tuple[int, ...]]]:
    """
    JSON serialization for a numpy dtype
    """
    if isinstance(o, np.dtype):
        if o.fields is None:
            return o.str
        else:
            return o.descr
    raise TypeError