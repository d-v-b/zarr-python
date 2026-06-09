"""
Benchmarks for end-to-end read/write performance of Zarr
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.benchmarks.common import Layout

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_benchmark.fixture import BenchmarkFixture

    from zarr.abc.store import Store

from operator import getitem, setitem
from typing import Any, Literal

import pytest

from zarr import create_array
from zarr.core.config import config as zarr_config
from zarr.testing.store import LatencyStore

CompressorName = Literal["gzip"] | None


def _compressor(name: CompressorName, zarr_format: Literal[2, 3]) -> Any:
    """Resolve a compressor for the given format.

    The v3 GzipCodec and the numcodecs (v2) Gzip are not interchangeable —
    create_array rejects each in the other format — so the gzip spelling must
    depend on zarr_format.
    """
    if name is None:
        return None
    if name == "gzip":
        if zarr_format == 2:
            import numcodecs

            return numcodecs.GZip(level=1)
        return {"name": "gzip", "configuration": {"level": 1}}
    raise AssertionError(name)


layouts: tuple[Layout, ...] = (
    # No shards, just 1000 chunks
    Layout(shape=(1_000_000,), chunks=(1000,), shards=None),
    # 1:1 chunk:shard shape, should measure overhead of sharding
    Layout(shape=(1_000_000,), chunks=(1000,), shards=(1000,)),
    # One shard with all the chunks, should measure overhead of handling inner shard chunks
    Layout(shape=(1_000_000,), chunks=(100,), shards=(10000 * 100,)),
)

_PIPELINE_PATHS = {
    "batched": "zarr.core.codec_pipeline.BatchedCodecPipeline",
    "fused": "zarr.core.codec_pipeline.FusedCodecPipeline",
}

_LATENCY_VALUES = (0.0, 0.001, 0.05, 0.2)


@pytest.fixture(params=_LATENCY_VALUES, ids=lambda v: f"latency={v}")
def latency(request: pytest.FixtureRequest) -> float:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def bench_store(store: Store, latency: float, request: pytest.FixtureRequest) -> Store:
    """Wraps the underlying store in LatencyStore when latency > 0.

    Local-store cases skip nonzero latency — synthetic latency on top of
    a real LocalStore is double-counting; latency simulation only applies
    to the in-process memory store.
    """
    callspec = getattr(request.node, "callspec", None)
    store_kind = callspec.params.get("store", "memory") if callspec is not None else "memory"
    if latency > 0:
        if store_kind == "local":
            pytest.skip("latency injection only applies to in-memory store")
        return LatencyStore(store, get_latency=latency, set_latency=latency)
    return store


@pytest.fixture(params=["batched", "fused"])
def pipeline(request: pytest.FixtureRequest) -> Iterator[str]:
    """Set ``codec_pipeline.path`` for the duration of the benchmark.

    Yields the pipeline name so each parametrize cell has a distinct
    benchmark id.
    """
    name = request.param
    with zarr_config.set({"codec_pipeline.path": _PIPELINE_PATHS[name]}):
        yield name


@pytest.fixture(params=[2, 3], ids=lambda v: f"v{v}")
def zarr_format(request: pytest.FixtureRequest, layout: Layout) -> Literal[2, 3]:
    """Zarr format axis. v2 uses the V2Codec path (filters + compressor) rather
    than the v3 array->array / array->bytes / bytes->bytes chain, so the codec
    pipeline behaves differently — worth measuring on both pipelines. Sharding
    is v3-only, so skip v2 cells that request shards.
    """
    fmt = request.param
    if fmt == 2 and layout.shards is not None:
        pytest.skip("zarr v2 does not support sharding")
    return fmt  # type: ignore[no-any-return]


@pytest.mark.parametrize("compression_name", [None, "gzip"])
@pytest.mark.parametrize("layout", layouts, ids=str)
@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
def test_write_array(
    bench_store: Store,
    layout: Layout,
    compression_name: CompressorName,
    pipeline: str,
    zarr_format: Literal[2, 3],
    benchmark: BenchmarkFixture,
) -> None:
    """
    Test the time required to fill an array with a single value
    """
    arr = create_array(
        bench_store,
        dtype="uint8",
        shape=layout.shape,
        chunks=layout.chunks,
        shards=layout.shards,
        compressors=_compressor(compression_name, zarr_format),
        fill_value=0,
        zarr_format=zarr_format,
    )

    benchmark(setitem, arr, Ellipsis, 1)


@pytest.mark.parametrize("compression_name", [None, "gzip"])
@pytest.mark.parametrize("layout", layouts, ids=str)
@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
def test_read_array(
    bench_store: Store,
    layout: Layout,
    compression_name: CompressorName,
    pipeline: str,
    zarr_format: Literal[2, 3],
    benchmark: BenchmarkFixture,
) -> None:
    """
    Test the time required to fill an array with a single value
    """
    arr = create_array(
        bench_store,
        dtype="uint8",
        shape=layout.shape,
        chunks=layout.chunks,
        shards=layout.shards,
        compressors=_compressor(compression_name, zarr_format),
        fill_value=0,
        zarr_format=zarr_format,
    )
    arr[:] = 1
    benchmark(getitem, arr, Ellipsis)
