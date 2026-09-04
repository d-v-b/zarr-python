"""Executable examples remain complete, current, and self-contained."""

from __future__ import annotations

import runpy
from pathlib import Path


def test_build_v3_array_example() -> None:
    example = Path(__file__).parents[1] / "examples" / "build_v3_array.py"
    runpy.run_path(str(example), run_name="__main__")
