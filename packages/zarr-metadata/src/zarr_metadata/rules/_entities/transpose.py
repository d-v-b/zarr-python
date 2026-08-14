"""Composition rules for the `transpose` codec."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._registry import entity_rule
from zarr_metadata.v3._extension_points import CODECS
from zarr_metadata.v3.codec.transpose import TRANSPOSE_CODEC_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

_ARRAY_V3 = "zarr_v3_array"


@entity_rule(_ARRAY_V3, CODECS, TRANSPOSE_CODEC_NAME)
def order_is_a_permutation(
    configuration: Mapping[str, object], document: Mapping[str, object]
) -> tuple[ValidationProblem, ...]:
    """`order` must be a permutation of its own indices.

    Checked without reference to `shape`, so it holds at any pipeline
    depth — including inside a shard, where the enclosing rank is the
    inner chunk's rather than the array's.
    """
    order = cast("tuple[int, ...]", configuration["order"])
    if sorted(order) == list(range(len(order))):
        return ()
    return (
        ValidationProblem(
            ("order",),
            f"expected a permutation of 0..{len(order) - 1}, got {order!r}",
            "invalid_value",
        ),
    )


@entity_rule(_ARRAY_V3, CODECS, TRANSPOSE_CODEC_NAME, requires=frozenset({"shape"}))
def order_matches_rank(
    configuration: Mapping[str, object], document: Mapping[str, object]
) -> tuple[ValidationProblem, ...]:
    """A top-level transpose permutes the array's dimensions, so ranks agree."""
    shape = document["shape"]
    if not isinstance(shape, (list, tuple)):
        return ()
    extents = cast("tuple[object, ...]", shape)
    order = cast("tuple[int, ...]", configuration["order"])
    if len(order) == len(extents):
        return ()
    return (
        ValidationProblem(
            ("order",),
            f"order has {len(order)} entries but shape has {len(extents)} dimensions",
            "invalid_value",
        ),
    )
