"""Tests for `ZarrV3ArrayMetadataBuilder`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr_metadata.builder import ZarrV3ArrayMetadataBuilder
from zarr_metadata.model import UNSET, MetadataValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

# A structurally- and semantically-valid document, assembled below in
# different with_fields() orders.
COMPLETE: dict[str, object] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": (4, 4),
    "data_type": "uint8",
    "fill_value": 0,
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
    "chunk_key_encoding": "default",
    "codecs": ("bytes",),
}

LITTLE_ENDIAN_BYTES: dict[str, object] = {
    "name": "bytes",
    "configuration": {"endian": "little"},
}


def _steps(*chunks: dict[str, object]) -> tuple[dict[str, object], ...]:
    return chunks


# (with_fields-call payloads applied in order, expected build() output). Every
# entry must build successfully; error paths get their own tests below.
CASES: dict[str, tuple[Sequence[dict[str, object]], dict[str, object]]] = {
    "one-call": (_steps(COMPLETE), COMPLETE),
    "field-at-a-time": (_steps(*({k: v} for k, v in COMPLETE.items())), COMPLETE),
    "fill-before-dtype": (
        _steps(
            {"fill_value": "NaN"},
            {"data_type": "float32"},
            {
                **{
                    k: v
                    for k, v in COMPLETE.items()
                    if k not in ("fill_value", "data_type", "codecs")
                },
                "codecs": (LITTLE_ENDIAN_BYTES,),
            },
        ),
        {
            **COMPLETE,
            "fill_value": "NaN",
            "data_type": "float32",
            "codecs": (LITTLE_ENDIAN_BYTES,),
        },
    ),
    "conflict-escape-by-pair": (
        # uint8/0 established, then both members of the couple change at once.
        _steps(
            COMPLETE,
            {
                "data_type": "float64",
                "fill_value": "Infinity",
                "codecs": (LITTLE_ENDIAN_BYTES,),
            },
        ),
        {
            **COMPLETE,
            "data_type": "float64",
            "fill_value": "Infinity",
            "codecs": (LITTLE_ENDIAN_BYTES,),
        },
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
    "json-loads-input-normalizes-to-tuples": (
        # Arrays arriving as lists (straight from json.loads) are
        # materialized as tuples at ingestion, so the built document is
        # spelling-identical to its tuple-spelled twin.
        _steps(
            {
                **COMPLETE,
                "shape": [4, 4],
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2, 2]}},
                "codecs": ["bytes"],
            }
        ),
        COMPLETE,
    ),
    "bare-spellings-where-permitted": (
        # scale_offset/bytes/crc32c have no required configuration, so
        # their bare short-hand spellings are canonical and pass the
        # spelling rule.
        _steps(COMPLETE, {"codecs": ("scale_offset", "bytes", "crc32c")}),
        {**COMPLETE, "codecs": ("scale_offset", "bytes", "crc32c")},
    ),
}


@pytest.mark.parametrize(("steps", "expected"), CASES.values(), ids=list(CASES))
def test_build(steps: Sequence[dict[str, object]], expected: dict[str, object]) -> None:
    builder = ZarrV3ArrayMetadataBuilder()
    for step in steps:
        builder = builder.with_fields(**step)
    assert builder.build() == expected


def test_extension_fields() -> None:
    builder = ZarrV3ArrayMetadataBuilder(COMPLETE).with_extension(
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
    builder = ZarrV3ArrayMetadataBuilder(COMPLETE).with_fields(dimension_names=("y", "x"))
    assert builder.without_fields("dimension_names").dimension_names is UNSET
    # removing an absent key is a no-op
    assert builder.without_fields("attributes") == builder
    # a removed required key is UNSET, not null
    assert builder.without_fields("fill_value").fill_value is UNSET


def test_immutability() -> None:
    source: dict[str, object] = dict(COMPLETE)
    builder = ZarrV3ArrayMetadataBuilder(source)
    source["shape"] = (9,)  # the builder copied on ingest
    evolved = builder.with_fields(shape=(8, 8))
    assert builder.shape == (4, 4)  # with_fields did not mutate its receiver
    assert evolved.shape == (8, 8)
    built = evolved.build()
    built["attributes"] = {"corrupted": True}  # outputs are isolated copies
    assert evolved.attributes is UNSET
    partial = builder.to_partial_json()
    assert partial == builder.to_partial_json()
    partial["shape"] = (1,)
    assert builder.shape == (4, 4)


def test_runtime_list_input_cannot_corrupt_builder() -> None:
    # Regression: `shape`/`dimension_names` used to hand out the internal
    # object, so a list smuggled past the type checker could be mutated in
    # place, corrupting the builder behind the eager rules' back.
    builder = ZarrV3ArrayMetadataBuilder().with_fields(shape=[2, 2], dimension_names=["a", "b"])  # type: ignore[arg-type]
    assert builder.shape == (2, 2)  # normalized to a tuple: no .append to abuse
    assert builder.dimension_names == ("a", "b")
    # and equality is spelling-insensitive as a consequence
    assert builder == ZarrV3ArrayMetadataBuilder().with_fields(
        shape=(2, 2), dimension_names=("a", "b")
    )


def test_to_partial_json_always_succeeds() -> None:
    fragment = ZarrV3ArrayMetadataBuilder().with_fields(shape=(2,), fill_value=None)
    # incomplete and unbuildable, but honestly serializable — and the stored
    # null survives (key omission is never decided by value inspection)
    assert fragment.to_partial_json() == {"shape": (2,), "fill_value": None}
    with pytest.raises(MetadataValidationError):
        fragment.build()


# -- error cases, one test per failure mode ---------------------------------


def test_error_fill_dtype_conflict_names_both_fields() -> None:
    builder = ZarrV3ArrayMetadataBuilder().with_fields(fill_value="NaN")
    with pytest.raises(MetadataValidationError) as info:
        builder.with_fields(data_type="uint8")
    (problem,) = info.value.problems
    assert problem.loc == ("fill_value",)
    assert problem.kind == "invalid_value"
    # the conflict names the field just set AND the one set earlier, and
    # points at the batch-with_fields escape hatch
    assert "data_type set in this call" in problem.message
    assert "fill_value set earlier" in problem.message
    assert "with_fields()" in problem.message


def test_error_fill_out_of_range() -> None:
    with pytest.raises(MetadataValidationError, match=r"\[0, 255\]"):
        ZarrV3ArrayMetadataBuilder().with_fields(data_type="uint8", fill_value=300)


def test_error_codec_order() -> None:
    with pytest.raises(MetadataValidationError, match="may not follow"):
        ZarrV3ArrayMetadataBuilder().with_fields(
            codecs=("bytes", {"name": "transpose", "configuration": {"order": (0, 1)}})
        )


def test_error_two_array_bytes_codecs() -> None:
    with pytest.raises(MetadataValidationError, match="exactly one"):
        ZarrV3ArrayMetadataBuilder().with_fields(codecs=("bytes", "bytes"))


def test_error_no_array_bytes_codec() -> None:
    with pytest.raises(MetadataValidationError, match="no array->bytes codec"):
        ZarrV3ArrayMetadataBuilder().with_fields(codecs=("crc32c",))


def test_error_dimension_names_length() -> None:
    with pytest.raises(MetadataValidationError, match="2 entries.*3 dimensions"):
        ZarrV3ArrayMetadataBuilder().with_fields(shape=(1, 2, 3), dimension_names=("a", "b"))


def test_error_regular_grid_dimensions() -> None:
    with pytest.raises(MetadataValidationError, match="chunk_shape has 1"):
        ZarrV3ArrayMetadataBuilder().with_fields(
            shape=(4, 4),
            chunk_grid={"name": "regular", "configuration": {"chunk_shape": (2,)}},
        )


def test_error_bare_spelling_of_config_required_codec() -> None:
    # Regression: bare "transpose" used to pass as an unknown extension,
    # suppressing both the spelling check and the exactly-one-array->bytes
    # count — build() would emit a pipeline with no array->bytes codec.
    with pytest.raises(MetadataValidationError) as info:
        ZarrV3ArrayMetadataBuilder().with_fields(codecs=("transpose",))
    messages = [p.message for p in info.value.problems]
    assert any("no bare short-hand form" in m for m in messages)
    assert any("no array->bytes codec" in m for m in messages)


def test_error_known_codec_missing_configuration_key() -> None:
    with pytest.raises(MetadataValidationError) as info:
        ZarrV3ArrayMetadataBuilder().with_fields(
            codecs=("bytes", {"name": "gzip", "configuration": {}})
        )
    (problem,) = info.value.problems
    assert problem.loc == ("codecs", 1, "configuration", "level")
    assert problem.kind == "missing_key"


def test_error_known_codec_missing_configuration_object() -> None:
    with pytest.raises(MetadataValidationError) as info:
        ZarrV3ArrayMetadataBuilder().with_fields(codecs=({"name": "transpose"}, "bytes"))
    (problem,) = info.value.problems
    assert problem.loc == ("codecs", 0, "configuration")
    assert problem.kind == "missing_key"


def test_error_known_codec_bad_configuration_literal() -> None:
    # Known-name configurations are held to their full canonical shapes,
    # value types included — not just key presence.
    with pytest.raises(MetadataValidationError) as info:
        ZarrV3ArrayMetadataBuilder().with_fields(
            codecs=({"name": "bytes", "configuration": {"endian": "middle"}},)
        )
    (problem,) = info.value.problems
    assert problem.loc == ("codecs", 0, "configuration", "endian")
    assert problem.kind == "invalid_value"


def test_error_known_codec_bad_configuration_value_type() -> None:
    with pytest.raises(MetadataValidationError) as info:
        ZarrV3ArrayMetadataBuilder().with_fields(
            codecs=("bytes", {"name": "gzip", "configuration": {"level": "high"}})
        )
    (problem,) = info.value.problems
    assert problem.loc == ("codecs", 1, "configuration", "level")
    assert problem.kind == "invalid_type"


def test_error_known_codec_unexpected_configuration_key() -> None:
    with pytest.raises(MetadataValidationError, match="unexpected key"):
        ZarrV3ArrayMetadataBuilder().with_fields(
            codecs=("bytes", {"name": "gzip", "configuration": {"level": 1, "speed": "max"}})
        )


def test_error_bare_chunk_grid_spelling() -> None:
    with pytest.raises(MetadataValidationError, match="no bare short-hand form"):
        ZarrV3ArrayMetadataBuilder().with_fields(chunk_grid="regular")


def test_error_chunk_grid_missing_configuration_object() -> None:
    with pytest.raises(MetadataValidationError, match="requires a 'configuration' object"):
        ZarrV3ArrayMetadataBuilder().with_fields(chunk_grid={"name": "regular"})


def test_error_chunk_grid_missing_configuration_key() -> None:
    with pytest.raises(MetadataValidationError) as info:
        ZarrV3ArrayMetadataBuilder().with_fields(
            chunk_grid={"name": "regular", "configuration": {}}
        )
    (problem,) = info.value.problems
    assert problem.loc == ("chunk_grid", "configuration", "chunk_shape")
    assert problem.kind == "missing_key"


def test_spelling_verdicts_agree_across_model_normalization() -> None:
    # Regression: the model layer collapses empty-config codecs to bare
    # names, and bare "gzip" used to classify as an unknown extension —
    # so a document the builder rejected round-tripped through the model
    # into one the rules accepted. Both spellings must now be rejected.
    from zarr_metadata.model import ZarrV3ArrayMetadata
    from zarr_metadata.rules import ZARR_V3_ARRAY_RULES, run_rules

    doc = {
        **COMPLETE,
        "codecs": ({"name": "gzip", "configuration": {}}, {"name": "bytes", "configuration": {}}),
    }
    with pytest.raises(MetadataValidationError):
        ZarrV3ArrayMetadataBuilder(doc)
    normalized = ZarrV3ArrayMetadata.from_json(doc).to_json()
    assert normalized["codecs"] == ("gzip", "bytes")  # the collapsed spelling
    assert run_rules(ZARR_V3_ARRAY_RULES, normalized)  # still rejected


def test_error_build_incomplete_reports_every_missing_key() -> None:
    with pytest.raises(MetadataValidationError) as info:
        ZarrV3ArrayMetadataBuilder().with_fields(shape=(2, 2)).build()
    missing = {p.loc[0] for p in info.value.problems if p.kind == "missing_key"}
    # every absent required key is reported at once, not one per attempt
    assert {"zarr_format", "node_type", "data_type", "fill_value"} <= missing


def test_error_build_reports_structural_problems() -> None:
    # The eager with_fields/constructor pass runs semantic rules only; the
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
        ZarrV3ArrayMetadataBuilder().with_extension("shape", (1, 2))
