"""The fast path reads what the Python walk reads: one reading, whichever did the structural walk.

Every v3 array fixture, the benchmark documents, and a few documents
the fast path must hand back -- an unclaimed name, raw bits, a shard
holding a shard, a struct, keys in an unusual order -- are read both
ways, and the readings compared member by member: each field, the
pipeline, the problems, the document the model holds and the bytes it
writes. Skipped where pydantic-core is not installed, since then there
is one reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.benchmarks.read_documents import V3_ARRAY
from zarr_metadata.model import ZarrV3ArrayMetadata, read_array_metadata_v3
from zarr_metadata.v3 import _fast
from zarr_metadata.v3.definition import CORE, CORE_AND_EXTENSIONS, Context

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.skipif(not _fast.AVAILABLE, reason="pydantic-core is not installed")

FIXTURES = sorted((Path(__file__).parent / "array").glob("*.json"))

SHARD: dict[str, Any] = {
    "name": "sharding_indexed",
    "configuration": {
        "index_codecs": ["crc32c", {"name": "bytes", "configuration": {"endian": "little"}}],
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, "crc32c"],
        "chunk_shape": [4, 4],
    },
}
_SHARD_IN_SHARD = json.loads(json.dumps(SHARD))
_SHARD_IN_SHARD["configuration"]["codecs"] = [
    {**SHARD, "configuration": {**SHARD["configuration"], "chunk_shape": [2, 2]}}
]

DOCUMENTS: dict[str, dict[str, Any]] = {
    "benchmark": V3_ARRAY,
    "unclaimed names": {
        **V3_ARRAY,
        "data_type": {"name": "zzz.unknown", "configuration": {"x": [1, {"y": None}]}},
        "codecs": [{"name": "bytes"}, "a.compressor"],
        "storage_transformers": [{"name": "a.transformer", "configuration": {}}],
    },
    "raw bits": {**V3_ARRAY, "data_type": "r16", "fill_value": [0, 1]},
    "shard in shard": {
        **V3_ARRAY,
        "shape": [16, 16],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [8, 8]}},
        "codecs": [_SHARD_IN_SHARD],
    },
    "struct": {
        **V3_ARRAY,
        "data_type": {
            "name": "struct",
            "configuration": {
                "fields": [
                    {"data_type": "int8", "name": "a"},
                    {
                        "name": "b",
                        "data_type": {
                            "name": "struct",
                            "configuration": {"fields": [{"name": "c", "data_type": "float32"}]},
                        },
                    },
                ]
            },
        },
        "fill_value": {"a": 1, "b": {"c": 0.5}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    },
    "unusual order": dict(
        reversed(
            list(
                {
                    **V3_ARRAY,
                    "chunk_grid": {"configuration": {"chunk_shape": [10, 10]}, "name": "regular"},
                    "codecs": [
                        {
                            "configuration": {"checksum": False, "level": 0},
                            "name": "zstd",
                            "must_understand": True,
                        },
                        {"name": "bytes", "configuration": {"endian": "little"}},
                    ][::-1],
                    "attributes": {"b": [1, 2.5, "x", None, {"nan": float("nan")}], "a": {}},
                    "zz.extension": {"must_understand": False, "any": [1]},
                }.items()
            )
        )
    ),
    "must_understand false at a point": {
        **V3_ARRAY,
        "chunk_key_encoding": {"name": "default", "must_understand": False},
    },
    "a problem the rules find": {
        **V3_ARRAY,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {
                "name": "blosc",
                "configuration": {
                    "cname": "lz4",
                    "clevel": 5,
                    "shuffle": "shuffle",
                    "blocksize": 0,
                },
            },
        ],
    },
}
for path in FIXTURES:
    DOCUMENTS[path.name] = json.loads(path.read_text())


def _cases() -> Iterator[tuple[str, Context, dict[str, Any]]]:
    for name, document in DOCUMENTS.items():
        for scope_name, scope in (("extensions", CORE_AND_EXTENSIONS), ("core", CORE)):
            yield f"{name} / {scope_name}", scope, document


def _reading(document: object, context: Context) -> dict[str, object]:
    """Everything a reading says, as plain values that compare."""
    reading = read_array_metadata_v3(document, context=context)
    record: dict[str, object] = {
        "problems": [(p.loc, p.kind, p.message, p.input, dict(p.ctx)) for p in reading.problems],
        "fields": [(loc, type(field).__name__, field) for loc, field in reading.fields()],
        "pipeline": [
            (stage.codec, stage.incoming, dict(stage.inner)) for stage in reading.pipeline
        ],
        "chunk": reading.chunk,
    }
    if reading.metadata is not None:
        model = reading.metadata
        # Not `to_json` itself: an attribute may hold NaN, which is never
        # equal to itself; the bytes say what the document holds.
        record["document order"] = _orders(model.to_json())
        record["bytes"] = model.to_key_value()
        record["key"] = model._model_key
        record["claims"] = dict(model.claims)
    return record


def _orders(value: object) -> object:
    """The keys of each object in `value`, in order, where it sits."""
    if isinstance(value, dict):
        return [list(value), {key: _orders(item) for key, item in value.items()}]
    if isinstance(value, (list, tuple)):
        return [_orders(item) for item in value]
    return None


@pytest.mark.parametrize(
    ("case", "context", "document"),
    [(c, s, d) for c, s, d in _cases()],
    ids=[c for c, _, _ in _cases()],
)
def test_fast_path_reads_as_the_python_walk_reads(
    case: str, context: Context, document: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = _reading(document, context)
    monkeypatch.setattr(_fast, "ENABLED", False)
    walked = _reading(document, context)
    assert fast == walked


def test_the_fast_path_is_taken_for_a_document_with_no_problem() -> None:
    reader = _fast.reader_for(CORE_AND_EXTENSIONS)
    assert reader is not None
    assert reader.read(json.loads(json.dumps(V3_ARRAY))) is not None
    assert reader.read({**V3_ARRAY, "shape": [-1]}) is None
    assert reader.read("not a document") is None


def test_an_equal_scope_shares_the_compiled_reader() -> None:
    again = Context.of(*CORE_AND_EXTENSIONS.definitions())
    assert _fast.reader_for(again) is _fast.reader_for(CORE_AND_EXTENSIONS)


def test_a_model_reads_alike_whichever_path_built_it(monkeypatch: pytest.MonkeyPatch) -> None:
    document = json.loads(json.dumps(V3_ARRAY))
    fast = ZarrV3ArrayMetadata.from_json(document)
    monkeypatch.setattr(_fast, "ENABLED", False)
    walked = ZarrV3ArrayMetadata.from_json(document)
    assert fast == walked
    assert hash(fast) == hash(walked)
    assert fast.to_key_value() == walked.to_key_value()
