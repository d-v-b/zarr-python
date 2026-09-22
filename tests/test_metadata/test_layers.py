"""zarr-python reading metadata through zarr-metadata's layers: the consumer proof."""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import zarr
from zarr.codecs import BloscCodec
from zarr.core.metadata import _layers
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.errors import MetadataValidationError, ZarrUserWarning

LEGACY = Path(__file__).parent / "legacy_stores"

BASE: dict[str, Any] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [4],
    "data_type": "int16",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
    "chunk_key_encoding": {"name": "default"},
    "fill_value": 0,
    "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    "attributes": {},
}


@pytest.fixture
def enforcing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_layers, "MODE", "enforce")


@pytest.mark.parametrize(
    ("changes", "rewritten"),
    [
        (
            {
                "shape": [4, 20],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [2, [5, 10, 5]]},
                },
            },
            {
                "chunk_grid": {
                    "name": "rectilinear",
                    "configuration": {"kind": "inline", "chunk_shapes": [2, [5, 10, 5]]},
                }
            },
        ),
        (
            {
                "shape": [0],
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [0]}},
            },
            {"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1]}}},
        ),
        (
            {
                "shape": [0, 3],
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [False, 3]}},
            },
            {"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 3]}}},
        ),
        ({"fill_value": "7"}, {"fill_value": 7}),
        ({"data_type": "float32", "fill_value": "2.5"}, {"fill_value": 2.5}),
        # Not deviations: left for the layers to judge as written.
        (
            {
                "shape": [3],
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [0]}},
            },
            {},
        ),
        ({"data_type": "float32", "fill_value": "NaN"}, {}),
        ({"data_type": "float32", "fill_value": "0x7fc00000"}, {}),
        ({"fill_value": "seven"}, {}),
        ({}, {}),
    ],
    ids=[
        "mixed-regular-grid",
        "zero-edge-on-empty-axis",
        "false-edge-on-empty-axis",
        "integer-string-fill",
        "float-string-fill",
        "zero-edge-on-a-non-empty-axis",
        "nan-spelling",
        "hex-spelling",
        "not-a-number",
        "spec-document",
    ],
)
def test_rewrite_legacy_v3(changes: dict[str, Any], rewritten: dict[str, Any]) -> None:
    # JSON to JSON, before anything is judged: each rewrite recognizes one
    # spelling a writer produced and the spec does not allow, and nothing else.
    document = {**BASE, **changes}
    out, notes = _layers.rewrite_legacy_v3(document)
    assert out == {**document, **rewritten}
    assert len(notes) == (0 if rewritten == {} else 1)


@pytest.mark.parametrize(
    ("store", "expected"),
    [
        ("mixed_321.zarr", np.arange(80, dtype="uint8").reshape(4, 20)),
        ("zero_3010_zero.zarr", np.zeros((0,), dtype="int16")),
        ("zero_3010_false.zarr", np.zeros((0,), dtype="int16")),
        ("zero_316_zero.zarr", np.zeros((0,), dtype="int16")),
    ],
)
def test_legacy_stores_read_through_the_rewrite(
    enforcing: None, store: str, expected: np.ndarray[Any, Any], tmp_path: Path
) -> None:
    # Written by zarr 3.2.1 (a mixed chunk spec) and by 3.0.10 and 3.1.6 (a
    # zero-length axis). Without the rewrite, none of them opens.
    path = tmp_path / store
    shutil.copytree(LEGACY / store, path)
    with zarr.config.set({"array.rectilinear_chunks": True}):
        with pytest.warns(ZarrUserWarning):
            array = zarr.open_array(path, mode="r")
        np.testing.assert_array_equal(array[:], expected)


def test_the_resolved_pipeline_evolves_zarr_pythons_codecs() -> None:
    # What `ArrayV3Metadata.__init__` finds by threading an ArraySpec
    # through each codec's `resolve_metadata`, read off the pipeline.
    arrays = [
        zarr.create_array({}, shape=(8, 6), chunks=(4, 3), shards=(8, 6), dtype="float32"),
        zarr.create_array({}, shape=(8,), chunks=(4,), dtype="uint16", compressors=BloscCodec()),
        zarr.create_array({}, shape=(8,), chunks=(4,), dtype="U5"),
    ]
    for array in arrays:
        assert isinstance(array.metadata, ArrayV3Metadata)
        document = json.loads(
            array.metadata.to_buffer_dict(zarr.core.buffer.default_buffer_prototype())[
                "zarr.json"
            ].to_bytes()
        )
        reading = _layers.read_array_v3(document)
        assert reading.refined is not None
        assert reading.problems == ()
        codecs, declined = _layers.codecs_from_pipeline(
            reading.refined, array.metadata.shape, array.metadata.fill_value
        )
        assert declined is None
        assert codecs == array.metadata.codecs


def test_error_the_layers_refuse_before_zarr_python_parses(enforcing: None) -> None:
    # A multi-byte type through a `bytes` codec without an endianness.
    document = {**BASE, "codecs": [{"name": "bytes"}]}
    with pytest.raises(
        MetadataValidationError, match=r"codecs\.0\.configuration\.endian: endian is required"
    ):
        ArrayV3Metadata.from_dict(document)


def test_error_a_field_zarr_python_does_not_understand(enforcing: None) -> None:
    document = {**BASE, "provenance": {"tool": "x"}}
    with pytest.raises(
        MetadataValidationError, match="provenance: a field zarr-python does not understand"
    ):
        ArrayV3Metadata.from_dict(document)


def test_error_a_zarr_python_data_type_judges_its_own_width(enforcing: None) -> None:
    # zarr-python's own data types are entities in the scope it reads in.
    document = {
        **BASE,
        "data_type": {"name": "fixed_length_utf32", "configuration": {"length_bytes": 6}},
        "fill_value": "",
    }
    with pytest.raises(MetadataValidationError, match="expected a positive multiple of 4, got 6"):
        ArrayV3Metadata.from_dict(document)


def test_rewrite_legacy_v2_group() -> None:
    # A .zgroup allows no other keys; NCZarr writes `_nczarr_*` members,
    # and zarr-python 3 writes `consolidated_metadata` into nested entries.
    document = {
        "zarr_format": 2,
        "attributes": {},
        "_nczarr_superblock": {"version": "2.0.0"},
        "consolidated_metadata": {},
    }
    out, notes = _layers.rewrite_legacy_v2_group(document)
    assert out == {"zarr_format": 2, "attributes": {}}
    assert len(notes) == 2
    assert _layers.read_node_problems("v2-group", out) == ()


def test_zarr_python_writes_nested_zgroup_entries_the_spec_disallows() -> None:
    # Recorded, not fixed here: the writer of a v2 `.zmetadata` puts a
    # `consolidated_metadata` member in each nested group's `.zgroup` entry.
    store: dict[str, Any] = {}
    root = zarr.open_group(store, mode="w", zarr_format=2)
    root.create_group("a")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ZarrUserWarning)
        zarr.consolidate_metadata(store, zarr_format=2)
    written = json.loads(store[".zmetadata"].to_bytes())["metadata"]["a/.zgroup"]
    assert "consolidated_metadata" in written
