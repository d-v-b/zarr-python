"""Tests for the v3 array and group composition rules added with the
rules-layer promotion: chunk grid values/geometry, transpose orders,
sharding pipelines/geometry, and consolidated-entry recursion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr_metadata.model import MetadataValidationError
from zarr_metadata.rules import validate_array_metadata_v3, validate_group_metadata_v3

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata import ZarrV3ArrayMetadataJSON

BASE: ZarrV3ArrayMetadataJSON = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": (4, 4),
    "data_type": "uint8",
    "fill_value": 0,
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
    "chunk_key_encoding": "default",
    "codecs": ("bytes",),
}


def _shard(**overrides: object) -> Mapping[str, object]:
    """A sharding codec entry; overrides may be deliberately malformed."""
    configuration: dict[str, object] = {
        "chunk_shape": (2, 2),
        "codecs": ("bytes",),
        "index_codecs": ("bytes", "crc32c"),
    }
    configuration.update(overrides)
    return {"name": "sharding_indexed", "configuration": configuration}


# Documents that must be fully valid: the rules judge geometry and values
# without rejecting legitimate spellings of the same constructs.
VALID_CASES: dict[str, Mapping[str, object]] = {
    "regular": BASE,
    "rectilinear-explicit-sums": {
        **BASE,
        "chunk_grid": {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": ((2, 2), (1, 3))},
        },
    },
    "rectilinear-rle-and-uniform": {
        **BASE,
        "chunk_grid": {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": (((2, 2),), 4)},
        },
    },
    "transpose": {
        **BASE,
        "codecs": ({"name": "transpose", "configuration": {"order": (1, 0)}}, "bytes"),
    },
    "sharding": {**BASE, "codecs": (_shard(),)},
    "nested-sharding": {
        **BASE,
        "codecs": (
            _shard(
                codecs=(
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": (1, 2),
                            "codecs": ("bytes",),
                            "index_codecs": ("bytes",),
                        },
                    },
                )
            ),
        ),
    },
    "unknown-grid-passes": {
        **BASE,
        "chunk_grid": {"name": "hilbert", "configuration": {"level": 3}},
    },
    "unknown-codec-inconclusive": {**BASE, "codecs": ({"name": "zfpy"}, "bytes")},
}


@pytest.mark.parametrize("doc", VALID_CASES.values(), ids=list(VALID_CASES))
def test_valid_documents(doc: Mapping[str, object]) -> None:
    assert validate_array_metadata_v3(doc) == ()


def _sole_problem(doc: Mapping[str, object]) -> tuple[tuple[str | int, ...], str]:
    problems = validate_array_metadata_v3(doc)
    assert len(problems) == 1, [p.message for p in problems]
    return problems[0].loc, problems[0].message


def test_error_regular_chunk_extent_zero() -> None:
    loc, message = _sole_problem(
        {**BASE, "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (0, 2)}}}
    )
    assert loc == ("chunk_grid", "configuration", "chunk_shape", 0)
    assert "positive chunk extent" in message


def test_error_rectilinear_rank_mismatch() -> None:
    loc, message = _sole_problem(
        {
            **BASE,
            "chunk_grid": {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": ((2, 2),)},
            },
        }
    )
    assert loc == ("chunk_grid", "configuration", "chunk_shapes")
    assert "2 dimensions" in message


def test_error_rectilinear_sum_mismatch() -> None:
    loc, message = _sole_problem(
        {
            **BASE,
            "chunk_grid": {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": ((3, 3), (2, 2))},
            },
        }
    )
    assert loc == ("chunk_grid", "configuration", "chunk_shapes", 0)
    assert "sum to 6" in message


def test_error_rectilinear_nonpositive_rle() -> None:
    problems = validate_array_metadata_v3(
        {
            **BASE,
            "chunk_grid": {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": (((0, 4),), 4)},
            },
        }
    )
    assert any("positive [size, count] pair" in p.message for p in problems)


def test_error_transpose_not_a_permutation() -> None:
    loc, message = _sole_problem(
        {**BASE, "codecs": ({"name": "transpose", "configuration": {"order": (5, 5)}}, "bytes")}
    )
    assert loc == ("codecs", 0, "configuration", "order")
    assert "permutation" in message


def test_error_transpose_rank_mismatch() -> None:
    loc, message = _sole_problem(
        {**BASE, "codecs": ({"name": "transpose", "configuration": {"order": (2, 0, 1)}}, "bytes")}
    )
    assert loc == ("codecs", 0, "configuration", "order")
    assert "shape has 2 dimensions" in message


def test_error_sharding_inner_pipeline_order() -> None:
    loc, _ = _sole_problem({**BASE, "codecs": (_shard(codecs=("crc32c", "bytes")),)})
    assert loc == ("codecs", 0, "configuration", "codecs", 1)


def test_error_sharding_inner_no_array_bytes() -> None:
    loc, message = _sole_problem({**BASE, "codecs": (_shard(codecs=("crc32c",)),)})
    assert loc == ("codecs", 0, "configuration", "codecs")
    assert "no array->bytes codec" in message


def test_error_sharding_index_codecs_no_array_bytes() -> None:
    loc, message = _sole_problem({**BASE, "codecs": (_shard(index_codecs=("crc32c",)),)})
    assert loc == ("codecs", 0, "configuration", "index_codecs")
    assert "no array->bytes codec" in message


def test_error_sharding_rank_mismatch() -> None:
    loc, message = _sole_problem({**BASE, "codecs": (_shard(chunk_shape=(2,)),)})
    assert loc == ("codecs", 0, "configuration", "chunk_shape")
    assert "enclosing chunk has 2 dimensions" in message


def test_error_sharding_not_divisible() -> None:
    loc, message = _sole_problem(
        {
            **BASE,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4, 4)}},
            "codecs": (_shard(chunk_shape=(3, 2)),),
        }
    )
    assert loc == ("codecs", 0, "configuration", "chunk_shape", 0)
    assert "does not evenly divide" in message


def test_error_nested_sharding_not_divisible() -> None:
    inner: Mapping[str, object] = {
        "name": "sharding_indexed",
        "configuration": {"chunk_shape": (2, 3), "codecs": ("bytes",), "index_codecs": ("bytes",)},
    }
    loc, message = _sole_problem(
        {
            **BASE,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4, 4)}},
            "codecs": (_shard(codecs=(inner,)),),
        }
    )
    assert loc == ("codecs", 0, "configuration", "codecs", 0, "configuration", "chunk_shape", 1)
    assert "does not evenly divide" in message


def test_error_sharding_inner_chunk_extent_zero() -> None:
    problems = validate_array_metadata_v3({**BASE, "codecs": (_shard(chunk_shape=(0, 2)),)})
    assert any(
        p.loc == ("codecs", 0, "configuration", "chunk_shape", 0)
        and "positive chunk extent" in p.message
        for p in problems
    )


def test_error_consolidated_child_violates_array_rules() -> None:
    doc: Mapping[str, object] = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {"a": {**BASE, "fill_value": 300}},
        },
    }
    problems = validate_group_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [
        (("consolidated_metadata", "metadata", "a", "fill_value"), "invalid_value")
    ]


def test_error_consolidated_nested_group_recursion() -> None:
    child_group: Mapping[str, object] = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {"b": {**BASE, "fill_value": 300}},
        },
    }
    doc: Mapping[str, object] = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {"g": child_group},
        },
    }
    problems = validate_group_metadata_v3(doc)
    assert [p.loc for p in problems] == [
        (
            "consolidated_metadata",
            "metadata",
            "g",
            "consolidated_metadata",
            "metadata",
            "b",
            "fill_value",
        )
    ]


def test_error_group_parse_raises() -> None:
    from zarr_metadata.rules import parse_group_metadata_v3

    with pytest.raises(MetadataValidationError, match="fill_value invalid"):
        parse_group_metadata_v3(
            {
                "zarr_format": 3,
                "node_type": "group",
                "consolidated_metadata": {
                    "kind": "inline",
                    "must_understand": False,
                    "metadata": {"a": {**BASE, "fill_value": 300}},
                },
            }
        )


# -- unknown configuration members --------------------------------------------
#
# The v3 spec does not say whether an extension's `configuration` is closed
# (zarr-developers/zarr-specs#270, open since 2023). This package takes the
# strict reading, matching most registered extension schemas and most other
# implementations — but reports it as its own `unknown_key` kind, and never
# lets it mask a real finding about the same entity.


def test_unknown_configuration_member_has_its_own_kind() -> None:
    doc = {
        **BASE,
        "codecs": ({"name": "bytes", "configuration": {"endian": "little", "hint": 1}},),
    }
    problems = validate_array_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("codecs", 0, "configuration"), "unknown_key")]


def test_unknown_member_does_not_mask_a_codec_rule() -> None:
    # Regression: an unrecognized member used to make the whole entity
    # uninterpretable, silently suppressing every other rule about it — so a
    # cosmetic extra key hid a genuine permutation error.
    doc = {
        **BASE,
        "codecs": ({"name": "transpose", "configuration": {"order": (5, 5), "hint": 1}}, "bytes"),
    }
    kinds = {(p.loc, p.kind) for p in validate_array_metadata_v3(doc)}
    assert (("codecs", 0, "configuration"), "unknown_key") in kinds
    assert (("codecs", 0, "configuration", "order"), "invalid_value") in kinds


def test_unknown_member_does_not_mask_a_chunk_grid_rule() -> None:
    doc = {
        **BASE,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2,), "hint": 1}},
    }
    kinds = {(p.loc, p.kind) for p in validate_array_metadata_v3(doc)}
    assert (("chunk_grid", "configuration"), "unknown_key") in kinds
    assert (("chunk_grid", "configuration", "chunk_shape"), "invalid_value") in kinds


def test_unknown_member_survives_a_round_trip() -> None:
    # Whatever the strict validator says, the package must never silently
    # drop a member it does not model: a writer that knows more than we do
    # must get its bytes back. (zarr-python's own chunk-grid path is lossy
    # here; this asserts we are not.)
    import json

    from zarr_metadata.model import ZarrV3ArrayMetadata

    raw = {
        **BASE,
        "shape": [4, 4],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2, 2]}},
        "codecs": [
            {
                "name": "blosc",
                "configuration": {
                    "cname": "zstd",
                    "clevel": 5,
                    "shuffle": "shuffle",
                    "blocksize": 0,
                    "numThreads": 4,
                },
            },
            "bytes",
        ],
    }
    model = ZarrV3ArrayMetadata.from_json(json.loads(json.dumps(raw)))
    emitted = model.to_json()
    codec = emitted["codecs"][0]
    assert codec["configuration"]["numThreads"] == 4
