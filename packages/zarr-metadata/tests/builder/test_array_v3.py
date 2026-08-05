"""Tests for `ZarrV3ArrayMetadataBuilder`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from zarr_metadata.builder import ZarrV3ArrayMetadataBuilder
from zarr_metadata.model import UNSET, MetadataValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

# A structurally- and semantically-valid document, assembled below in
# different evolve() orders.
COMPLETE: dict[str, Any] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": (4, 4),
    "data_type": "uint8",
    "fill_value": 0,
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
    "chunk_key_encoding": "default",
    "codecs": ("bytes",),
}


def _steps(*chunks: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return chunks


# (evolve-call payloads applied in order, expected build() output). Every
# entry must build successfully; error paths get their own tests below.
CASES: dict[str, tuple[Sequence[dict[str, Any]], dict[str, Any]]] = {
    "one-call": (_steps(COMPLETE), COMPLETE),
    "field-at-a-time": (_steps(*({k: v} for k, v in COMPLETE.items())), COMPLETE),
    "fill-before-dtype": (
        _steps(
            {"fill_value": "NaN"},
            {"data_type": "float32"},
            {k: v for k, v in COMPLETE.items() if k not in ("fill_value", "data_type")},
        ),
        {**COMPLETE, "fill_value": "NaN", "data_type": "float32"},
    ),
    "conflict-escape-by-pair": (
        # uint8/0 established, then both members of the couple change at once.
        _steps(COMPLETE, {"data_type": "float64", "fill_value": "Infinity"}),
        {**COMPLETE, "data_type": "float64", "fill_value": "Infinity"},
    ),
    "field-replacement": (
        _steps(COMPLETE, {"shape": (8, 8)}, {"chunk_grid": COMPLETE["chunk_grid"]}),
        {**COMPLETE, "shape": (8, 8)},
    ),
    "with-optionals": (
        _steps(
            COMPLETE,
            {
                "attributes": {"unit": "kelvin", "nothing": None},
                "dimension_names": ("y", None),
                "storage_transformers": (),
            },
        ),
        {
            **COMPLETE,
            "attributes": {"unit": "kelvin", "nothing": None},
            "dimension_names": ("y", None),
            "storage_transformers": (),
        },
    ),
    "full-pipeline": (
        _steps(
            COMPLETE,
            {
                "codecs": (
                    {"name": "transpose", "configuration": {"order": (1, 0)}},
                    "bytes",
                    {"name": "gzip", "configuration": {"level": 5}},
                    "crc32c",
                )
            },
        ),
        {
            **COMPLETE,
            "codecs": (
                {"name": "transpose", "configuration": {"order": (1, 0)}},
                "bytes",
                {"name": "gzip", "configuration": {"level": 5}},
                "crc32c",
            ),
        },
    ),
    "unknown-codec-passes": (
        # An unclassifiable codec imposes no ordering constraint and may be
        # the pipeline's array->bytes stage.
        _steps(COMPLETE, {"codecs": ({"name": "lightspeed"},)}),
        {**COMPLETE, "codecs": ({"name": "lightspeed"},)},
    ),
    "unknown-dtype-accepts-any-fill": (
        _steps(COMPLETE, {"data_type": {"name": "bfloat16"}, "fill_value": "whatever"}),
        {**COMPLETE, "data_type": {"name": "bfloat16"}, "fill_value": "whatever"},
    ),
    "null-fill-value-is-a-value": (
        # JSON null is a stored value, not absence; builds only for a dtype
        # whose fill values this package does not judge.
        _steps(COMPLETE, {"data_type": {"name": "unknowable"}, "fill_value": None}),
        {**COMPLETE, "data_type": {"name": "unknowable"}, "fill_value": None},
    ),
}


@pytest.mark.parametrize(("steps", "expected"), CASES.values(), ids=list(CASES))
def test_build(steps: Sequence[dict[str, Any]], expected: dict[str, Any]) -> None:
    builder = ZarrV3ArrayMetadataBuilder()
    for step in steps:
        builder = builder.evolve(**step)
    assert builder.build() == expected


def test_extension_fields() -> None:
    builder = ZarrV3ArrayMetadataBuilder(COMPLETE).evolve_extension(
        "my_ext", {"must_understand": False, "level": 3}
    )
    assert builder.extension_fields == {"my_ext": {"must_understand": False, "level": 3}}
    built = builder.build()
    assert built["my_ext"] == {"must_understand": False, "level": 3}


def test_properties_unset_vs_value() -> None:
    empty = ZarrV3ArrayMetadataBuilder()
    assert empty.shape is UNSET
    assert empty.fill_value is UNSET
    assert empty.dimension_names is UNSET
    full = ZarrV3ArrayMetadataBuilder(COMPLETE)
    assert full.shape == (4, 4)
    assert full.zarr_format == 3
    assert full.codecs == ("bytes",)
    # optional-but-absent stays UNSET even on a buildable document
    assert full.attributes is UNSET


def test_without_unsets() -> None:
    builder = ZarrV3ArrayMetadataBuilder(COMPLETE).evolve(dimension_names=("y", "x"))
    assert builder.without("dimension_names").dimension_names is UNSET
    # removing an absent key is a no-op
    assert builder.without("attributes") == builder
    # a removed required key is UNSET, not null
    assert builder.without("fill_value").fill_value is UNSET


def test_immutability() -> None:
    source: dict[str, Any] = dict(COMPLETE)
    builder = ZarrV3ArrayMetadataBuilder(source)
    source["shape"] = (9,)  # the builder copied on ingest
    evolved = builder.evolve(shape=(8, 8))
    assert builder.shape == (4, 4)  # evolve did not mutate its receiver
    assert evolved.shape == (8, 8)
    built = evolved.build()
    built["attributes"] = {"corrupted": True}  # outputs are isolated copies
    assert evolved.attributes is UNSET
    partial = builder.to_partial_json()
    assert partial == builder.to_partial_json()
    partial["shape"] = (1,)
    assert builder.shape == (4, 4)


def test_to_partial_json_always_succeeds() -> None:
    fragment = ZarrV3ArrayMetadataBuilder().evolve(shape=(2,), fill_value=None)
    # incomplete and unbuildable, but honestly serializable — and the stored
    # null survives (key omission is never decided by value inspection)
    assert fragment.to_partial_json() == {"shape": (2,), "fill_value": None}
    with pytest.raises(MetadataValidationError):
        fragment.build()


# -- error cases, one test per failure mode ---------------------------------


def test_error_fill_dtype_conflict_names_both_fields() -> None:
    builder = ZarrV3ArrayMetadataBuilder().evolve(fill_value="NaN")
    with pytest.raises(MetadataValidationError) as info:
        builder.evolve(data_type="uint8")
    (problem,) = info.value.problems
    assert problem.loc == ("fill_value",)
    assert problem.kind == "invalid_value"
    # the conflict names the field just set AND the one set earlier, and
    # points at the batch-evolve escape hatch
    assert "data_type set in this call" in problem.message
    assert "fill_value set earlier" in problem.message
    assert "evolve()" in problem.message


def test_error_fill_out_of_range() -> None:
    with pytest.raises(MetadataValidationError, match=r"\[0, 255\]"):
        ZarrV3ArrayMetadataBuilder().evolve(data_type="uint8", fill_value=300)


def test_error_codec_order() -> None:
    with pytest.raises(MetadataValidationError, match="may not follow"):
        ZarrV3ArrayMetadataBuilder().evolve(
            codecs=("bytes", {"name": "transpose", "configuration": {"order": (0, 1)}})
        )


def test_error_two_array_bytes_codecs() -> None:
    with pytest.raises(MetadataValidationError, match="exactly one"):
        ZarrV3ArrayMetadataBuilder().evolve(codecs=("bytes", "bytes"))


def test_error_no_array_bytes_codec() -> None:
    with pytest.raises(MetadataValidationError, match="no array->bytes codec"):
        ZarrV3ArrayMetadataBuilder().evolve(codecs=("crc32c",))


def test_error_dimension_names_length() -> None:
    with pytest.raises(MetadataValidationError, match="2 entries.*3 dimensions"):
        ZarrV3ArrayMetadataBuilder().evolve(shape=(1, 2, 3), dimension_names=("a", "b"))


def test_error_regular_grid_dimensions() -> None:
    with pytest.raises(MetadataValidationError, match="chunk_shape has 1"):
        ZarrV3ArrayMetadataBuilder().evolve(
            shape=(4, 4),
            chunk_grid={"name": "regular", "configuration": {"chunk_shape": (2,)}},
        )


def test_error_build_incomplete_reports_every_missing_key() -> None:
    with pytest.raises(MetadataValidationError) as info:
        ZarrV3ArrayMetadataBuilder().evolve(shape=(2, 2)).build()
    missing = {p.loc[0] for p in info.value.problems if p.kind == "missing_key"}
    # every absent required key is reported at once, not one per attempt
    assert {"zarr_format", "node_type", "data_type", "fill_value"} <= missing


def test_error_build_reports_structural_problems() -> None:
    # The eager evolve/constructor pass runs semantic rules only; the
    # structural pass belongs to build(). An untyped caller smuggling in a
    # structurally-invalid value is caught there.
    builder = ZarrV3ArrayMetadataBuilder({**COMPLETE, "zarr_format": 2})
    with pytest.raises(MetadataValidationError) as info:
        builder.build()
    assert any(p.loc == ("zarr_format",) for p in info.value.problems)


def test_error_constructor_validates() -> None:
    with pytest.raises(MetadataValidationError, match="fill_value invalid"):
        ZarrV3ArrayMetadataBuilder({"data_type": "uint8", "fill_value": "NaN"})


def test_error_evolve_extension_rejects_standard_key() -> None:
    with pytest.raises(MetadataValidationError, match="standard v3 array metadata key"):
        ZarrV3ArrayMetadataBuilder().evolve_extension("shape", (1, 2))
