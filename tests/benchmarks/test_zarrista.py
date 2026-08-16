"""Benchmarks for zarrista-backed regional writes."""

from __future__ import annotations

from operator import setitem
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("zarrista")

import zarr
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture


CASES = [
    pytest.param(
        (512, 512),
        (128, 128),
        None,
        np.s_[160:224, 160:224],
        id="single-partial",
    ),
    pytest.param(
        (512, 512),
        (128, 128),
        None,
        np.s_[32:480, 32:480],
        id="multi-chunk",
    ),
    pytest.param(
        (512, 512),
        (64, 64),
        (256, 256),
        np.s_[96:416, 96:416],
        id="sharded",
    ),
]


@pytest.mark.parametrize(("shape", "chunks", "shards", "selection"), CASES)
def test_zarrista_region_write(
    tmp_path: Path,
    benchmark: BenchmarkFixture,
    shape: tuple[int, int],
    chunks: tuple[int, int],
    shards: tuple[int, int] | None,
    selection: tuple[slice, slice],
) -> None:
    zarr.create_array(
        LocalStore(tmp_path), shape=shape, chunks=chunks, shards=shards, dtype="float32"
    )
    array = zarr.open_array(LocalStore(tmp_path), engine="zarrista")
    value_shape = tuple(index.stop - index.start for index in selection)
    value = np.ones(value_shape, dtype="float32")

    benchmark(setitem, array, selection, value)
