from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("hypothesis")

import hypothesis.strategies as st
from hypothesis import given

import zarr
import zarr.api.asynchronous
import zarr.storage._local
from zarr.abc.codec import ArrayArrayCodec
from zarr.codecs import BytesCodec, CastValue, GzipCodec, TransposeCodec
from zarr.core.array import _get_chunk_spec
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.codec_pipeline import codecs_from_list
from zarr.core.config import config as zarr_config
from zarr.core.indexing import BasicIndexer
from zarr.storage import LocalStore, MemoryStore

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from zarr.abc.codec import Codec
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer


@pytest.fixture(autouse=True)
def _enable_rectilinear_chunks() -> Generator[None]:
    """Enable rectilinear chunks for all tests in this module."""
    with zarr_config.set({"array.rectilinear_chunks": True}):
        yield


pipeline_paths = [
    "zarr.core.codec_pipeline.BatchedCodecPipeline",
    "zarr.core.codec_pipeline.FusedCodecPipeline",
]


@pytest.fixture(params=pipeline_paths, ids=["batched", "sync"])
def pipeline_class(request: pytest.FixtureRequest) -> Generator[str]:
    """Temporarily set the codec pipeline class for the test."""
    path = request.param
    with zarr_config.set({"codec_pipeline.path": path}):
        yield path


# ---------------------------------------------------------------------------
# GetResult status tests (low-level pipeline API)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("write_slice", "read_slice", "expected_statuses"),
    [
        (slice(None), slice(None), ("present", "present", "present")),
        (slice(0, 2), slice(None), ("present", "missing", "missing")),
        (None, slice(None), ("missing", "missing", "missing")),
    ],
)
async def test_read_returns_get_results(
    pipeline_class: str,
    write_slice: slice | None,
    read_slice: slice,
    expected_statuses: tuple[str, ...],
) -> None:
    """CodecPipeline.read returns GetResult with correct statuses."""
    store = MemoryStore()
    arr = zarr.open_array(store, mode="w", shape=(6,), chunks=(2,), dtype="int64", fill_value=-1)

    if write_slice is not None:
        arr[write_slice] = 0

    async_arr = arr._async_array
    pipeline = async_arr.codec_pipeline
    metadata = async_arr.metadata

    prototype = default_buffer_prototype()
    config = async_arr.config
    indexer = BasicIndexer(
        read_slice,
        shape=metadata.shape,
        chunk_grid=async_arr._chunk_grid,
    )

    out_buffer = prototype.nd_buffer.empty(
        shape=indexer.shape,
        dtype=metadata.dtype.to_native_dtype(),
        order=config.order,
    )

    results = await pipeline.read(
        [
            (
                async_arr.store_path / metadata.encode_chunk_key(chunk_coords),
                _get_chunk_spec(metadata, async_arr._chunk_grid, chunk_coords, config, prototype),
                chunk_selection,
                out_selection,
                is_complete_chunk,
            )
            for chunk_coords, chunk_selection, out_selection, is_complete_chunk in indexer
        ],
        out_buffer,
        drop_axes=indexer.drop_axes,
    )

    assert len(results) == len(expected_statuses)
    for result, expected_status in zip(results, expected_statuses, strict=True):
        assert result["status"] == expected_status


# ---------------------------------------------------------------------------
# write_empty_chunks / read_missing_chunks config tests
# ---------------------------------------------------------------------------


async def test_write_empty_chunks_false_no_store(pipeline_class: str) -> None:
    """With write_empty_chunks=False, fill_value-only chunks should not be stored."""
    store: dict[str, Any] = {}
    arr = zarr.create_array(
        store=store,
        shape=(20,),
        dtype="float64",
        chunks=(10,),
        shards=None,
        compressors=None,
        fill_value=0.0,
        config={"write_empty_chunks": False},
    )
    arr[:] = 0.0  # all fill_value

    # Chunks should NOT be persisted
    assert "c/0" not in store
    assert "c/1" not in store

    # But reading should still return fill values
    np.testing.assert_array_equal(arr[:], np.zeros(20, dtype="float64"))


try:
    import cast_value_rs  # noqa: F401

    _HAS_CAST_VALUE_RS = True
except ModuleNotFoundError:
    _HAS_CAST_VALUE_RS = False

requires_cast_value_rs = pytest.mark.skipif(
    not _HAS_CAST_VALUE_RS, reason="cast-value-rs not installed"
)


@requires_cast_value_rs
@pytest.mark.parametrize(
    ("source_dtype", "target_dtype"),
    [
        # Source is single-byte (no endianness); target is multi-byte (has endianness).
        # Without the fix, BytesCodec.evolve_from_array_spec sees the source dtype,
        # strips its `endian` to None, and then chokes when the chunk_spec dtype
        # gets transformed to the multi-byte target before bytes-decoding.
        ("int8", "int16"),
        ("uint8", "int32"),
        ("int8", "float32"),
        # Source is multi-byte; target is single-byte (the reverse direction also
        # exercises the spec-threading logic).
        ("int16", "int8"),
    ],
)
def test_codec_pipeline_threads_dtype_through_evolve(source_dtype: str, target_dtype: str) -> None:
    """Regression for #3937: each codec must be evolved against the spec it
    will see at runtime, not the original array spec. cast_value transforms
    the dtype between AA codecs and the array->bytes serializer."""
    arr = zarr.create_array(
        store={},
        shape=(4,),
        chunks=(4,),
        dtype=source_dtype,
        fill_value=0,
        filters=[CastValue(data_type=target_dtype)],
        serializer=BytesCodec(endian="little"),
        compressors=[],
        zarr_format=3,
        overwrite=True,
    )
    arr[:] = np.asarray([0, 1, 2, 3], dtype=source_dtype)
    np.testing.assert_array_equal(arr[:], np.asarray([0, 1, 2, 3], dtype=source_dtype))


def test_evolve_threads_spec_preserving_serializer_endian(pipeline_class: str) -> None:
    """Regression for #3937, dependency-free variant.

    `evolve_from_array_spec` must thread the spec FORWARD through the codec chain:
    each codec is evolved against the spec produced by the previous one, not the
    original array spec. An array->array codec that widens the dtype from a
    single-byte type (no endianness) to a multi-byte type means the BytesCodec
    serializer must be evolved against the *widened* dtype — otherwise it sees
    the single-byte source, strips its `endian` to None, and later fails to
    decode the multi-byte data.

    The original regression test for this needs `cast_value_rs` (so it only runs
    in the optional-deps CI job). This variant uses a minimal dtype-widening AA
    codec stub, so it runs everywhere and on both pipelines via `pipeline_class`.
    """
    from dataclasses import dataclass

    from zarr.abc.codec import ArrayArrayCodec
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.dtype import get_data_type_from_native_dtype
    from zarr.registry import get_pipeline_class

    @dataclass(frozen=True)
    class _WidenToInt16(ArrayArrayCodec):
        """Test-only AA codec: reports the encoded dtype as int16 (no real encode)."""

        is_fixed_size = True

        def to_dict(self) -> dict[str, Any]:
            return {"name": "_widen_to_int16"}

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> _WidenToInt16:
            return cls()

        def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
            from dataclasses import replace

            return replace(chunk_spec, dtype=get_data_type_from_native_dtype(np.dtype("int16")))

        def compute_encoded_size(self, input_byte_length: int, _spec: ArraySpec) -> int:
            return input_byte_length

        async def _decode_single(self, chunk_array: Any, chunk_spec: ArraySpec) -> Any:
            return chunk_array  # pragma: no cover

        async def _encode_single(self, chunk_array: Any, chunk_spec: ArraySpec) -> Any:
            return chunk_array  # pragma: no cover

    zdtype = get_data_type_from_native_dtype(np.dtype("int8"))  # single-byte source
    spec = ArraySpec(
        shape=(4,),
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=False),
        prototype=default_buffer_prototype(),
    )

    from zarr.core.codec_pipeline import BatchedCodecPipeline, FusedCodecPipeline

    pipeline = get_pipeline_class().from_codecs((_WidenToInt16(), BytesCodec(endian="little")))
    evolved = pipeline.evolve_from_array_spec(spec)
    # Both concrete pipelines expose `array_bytes_codec`; narrow off the ABC.
    assert isinstance(evolved, (BatchedCodecPipeline, FusedCodecPipeline))
    serializer = evolved.array_bytes_codec

    # The serializer must keep its little-endian setting: it is evolved against
    # the widened (int16) dtype, not the single-byte source.
    assert isinstance(serializer, BytesCodec)
    assert serializer.endian is not None, (
        "BytesCodec serializer lost its `endian` — evolve_from_array_spec did not "
        "thread the dtype-widening AA codec's spec into the serializer"
    )


# Property-based check of codecs_from_list ordering validation.
#
# Valid codec orderings are exactly: (ArrayArrayCodec)* (ArrayBytesCodec)
# (BytesBytesCodec)*. codecs_from_list walks adjacent pairs and must raise
# TypeError the moment a codec appears in a structurally invalid position --
# notably, a BytesBytesCodec immediately following an ArrayArrayCodec with no
# ArrayBytesCodec in between (which previously built an error message but never
# raised it, falling through to an unrelated ValueError instead).
_AA = "AA"  # ArrayArrayCodec   -> TransposeCodec
_AB = "AB"  # ArrayBytesCodec   -> BytesCodec
_BB = "BB"  # BytesBytesCodec   -> GzipCodec

_CODEC_FACTORY: dict[str, Callable[[], Codec]] = {
    _AA: lambda: TransposeCodec(order=(0, 1)),
    _AB: BytesCodec,
    _BB: GzipCodec,
}


def _expected_codec_order_outcome(labels: list[str]) -> str:
    """Independently predict codecs_from_list's outcome: 'TypeError',
    'ValueError' or 'ok', mirroring its left-to-right scan and the order in
    which it checks ordering violations (TypeError) vs. the ArrayBytes-count
    constraints (ValueError)."""
    prev = None
    seen_array_bytes = False
    for cur in labels:
        if cur == _AA:
            if prev in (_AB, _BB):
                return "TypeError"
        elif cur == _AB:
            if prev == _BB:
                return "TypeError"
            if seen_array_bytes:
                return "ValueError"  # two ArrayBytesCodecs
            seen_array_bytes = True
        else:  # _BB
            if prev == _AA:
                return "TypeError"
        prev = cur
    if not seen_array_bytes:
        return "ValueError"  # Required ArrayBytesCodec was not found
    return "ok"


@given(labels=st.lists(st.sampled_from([_AA, _AB, _BB]), min_size=1, max_size=5))
def test_codecs_from_list_outcome_matches_order_rules(labels: list[str]) -> None:
    codecs = [_CODEC_FACTORY[label]() for label in labels]
    expected = _expected_codec_order_outcome(labels)
    if expected == "TypeError":
        with pytest.raises(TypeError):
            codecs_from_list(codecs)
    elif expected == "ValueError":
        with pytest.raises(ValueError):
            codecs_from_list(codecs)
    else:
        # Valid ordering: must classify without raising.
        aa, _ab, bb = codecs_from_list(codecs)
        assert labels.count(_AA) == len(aa)
        assert labels.count(_BB) == len(bb)


# ---------------------------------------------------------------------------
# A failed write stops changing the store before it raises
# ---------------------------------------------------------------------------

_SENTINEL = -1.0


@dataclass
class _WriteGate:
    """Holds a chunk's store write open until the test has seen a sibling chunk fail."""

    write_started: threading.Event = field(default_factory=threading.Event)
    encode_failing: threading.Event = field(default_factory=threading.Event)
    release_writes: threading.Event = field(default_factory=threading.Event)
    writes_in_flight: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True)
class _FailsOnSentinelCodec(ArrayArrayCodec):
    """Raises on chunks holding the sentinel, once another chunk's write is in flight."""

    gate: _WriteGate
    is_fixed_size = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": "test-fails-on-sentinel", "configuration": {}}

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length

    def _decode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        return chunk_array

    def _encode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        if np.any(chunk_array.as_numpy_array() == _SENTINEL):
            self.gate.write_started.wait(timeout=30)
            self.gate.encode_failing.set()
            raise ValueError("sentinel chunk")
        return chunk_array

    async def _decode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        return chunk_array

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        return await asyncio.to_thread(self._encode_sync, chunk_array, chunk_spec)


def _assign_expecting_failure(arr: zarr.Array[Any], data: Any, raised: list[BaseException]) -> None:
    try:
        arr[:] = data
    except ValueError as e:
        raised.append(e)


def _gate_local_store_writes(monkeypatch: pytest.MonkeyPatch, gate: _WriteGate) -> None:
    put = zarr.storage._local._put

    def gated_put(*args: Any, **kwargs: Any) -> int | None:
        with gate.lock:
            gate.writes_in_flight += 1
        gate.write_started.set()
        gate.release_writes.wait(timeout=30)
        try:
            return put(*args, **kwargs)
        finally:
            with gate.lock:
                gate.writes_in_flight -= 1

    monkeypatch.setattr(zarr.storage._local, "_put", gated_put)


def test_failed_write_stops_changing_the_store_before_raising(
    pipeline_class: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chunk 0 fails to encode while chunk 1 is mid-write: the write must not raise until
    chunk 1 is done. `LocalStore` writes run in threads, which cancellation cannot stop.

    The `join` timeout only gives a regression time to show. It cannot fail a correct
    implementation, which is blocked on `release_writes` for as long as the test likes.
    """
    gate = _WriteGate()
    data = np.arange(8, dtype="float64")
    data[0] = _SENTINEL
    raised: list[BaseException] = []
    with zarr_config.set({"async.concurrency": 2, "codec_pipeline.max_workers": 2}):
        arr = zarr.create_array(
            LocalStore(tmp_path),
            shape=(8,),
            chunks=(1,),
            dtype="float64",
            filters=[_FailsOnSentinelCodec(gate)],
            compressors=None,
            fill_value=0.0,
        )
        _gate_local_store_writes(monkeypatch, gate)
        writer = threading.Thread(target=_assign_expecting_failure, args=(arr, data, raised))
        writer.start()
        try:
            assert gate.encode_failing.wait(timeout=30)
            writer.join(timeout=0.1)
            returned_while_a_write_was_in_flight = not writer.is_alive()
        finally:
            gate.release_writes.set()
            writer.join(timeout=30)
        writes_in_flight_on_return = gate.writes_in_flight

    assert not returned_while_a_write_was_in_flight
    assert writes_in_flight_on_return == 0
    assert len(raised) == 1


async def test_cancelled_write_stops_changing_the_store_before_raising(
    pipeline_class: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling a write holds the `CancelledError` back until the write threads are done."""
    gate = _WriteGate()
    with zarr_config.set({"async.concurrency": 2, "codec_pipeline.max_workers": 2}):
        arr = await zarr.api.asynchronous.create_array(
            LocalStore(tmp_path), shape=(8,), chunks=(1,), dtype="float64", compressors=None
        )
        _gate_local_store_writes(monkeypatch, gate)
        task = asyncio.ensure_future(arr.setitem(slice(None), np.arange(8, dtype="float64")))
        assert await asyncio.to_thread(gate.write_started.wait, 30)
        task.cancel()
        await asyncio.sleep(0)
        returned_while_a_write_was_in_flight = task.done()

        gate.release_writes.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert not returned_while_a_write_was_in_flight

    assert gate.writes_in_flight == 0
