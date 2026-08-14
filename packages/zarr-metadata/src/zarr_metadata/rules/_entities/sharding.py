"""Composition rules for the `sharding_indexed` codec.

Sharding is the one entity whose configuration contains whole pipelines
and its own geometry, so its rules recurse: the inner `codecs` and
`index_codecs` are judged by the same pipeline checks that judge the
document's top-level `codecs`, at every nesting depth.

Inner pipelines are judged against a *synthetic document* whose `shape`
is the shard's inner chunk shape. Within a shard the array being encoded
is the inner chunk, so a `transpose` there permutes the inner chunk's
dimensions — a rank rule written against "the shape" is then correct at
any depth without knowing how deep it is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._pipeline import pipeline_order_problems, shape_problems
from zarr_metadata.rules._registry import entity_configuration, entity_rule, run_entity_rules
from zarr_metadata.v3._extension_points import CHUNK_GRID, CODECS
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID_NAME
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEXED_CODEC_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

_ARRAY_V3 = "zarr_v3_array"


@entity_rule(_ARRAY_V3, CODECS, SHARDING_INDEXED_CODEC_NAME)
def inner_chunk_extents_are_positive(
    configuration: Mapping[str, object], document: Mapping[str, object]
) -> tuple[ValidationProblem, ...]:
    chunk_shape = cast("tuple[int, ...]", configuration["chunk_shape"])
    return tuple(
        ValidationProblem(
            ("chunk_shape", position),
            f"expected a positive chunk extent, got {extent}",
            "invalid_value",
        )
        for position, extent in enumerate(chunk_shape)
        if extent < 1
    )


@entity_rule(_ARRAY_V3, CODECS, SHARDING_INDEXED_CODEC_NAME)
def inner_pipelines_are_pipelines(
    configuration: Mapping[str, object], document: Mapping[str, object]
) -> tuple[ValidationProblem, ...]:
    """`codecs` and `index_codecs` obey the pipeline rules, recursively.

    The same ordering and shape judgments the top-level pipeline gets,
    plus the entity rules of whatever codecs appear inside — so a
    violation caught at depth zero cannot hide at depth one.
    """
    # Within a shard the array being encoded is the inner chunk, tiled by a
    # regular grid of that same shape: one synthetic document makes every
    # rule that reads `shape` or `chunk_grid` correct at any nesting depth,
    # including this rule itself when shards nest.
    inner_shape = configuration["chunk_shape"]
    inner_document: Mapping[str, object] = {
        "shape": inner_shape,
        "chunk_grid": {
            "name": REGULAR_CHUNK_GRID_NAME,
            "configuration": {"chunk_shape": inner_shape},
        },
    }
    problems: list[ValidationProblem] = []
    for key in ("codecs", "index_codecs"):
        entries = configuration[key]
        if not isinstance(entries, (list, tuple)):
            continue
        sequence = cast("tuple[object, ...]", entries)
        problems.extend(pipeline_order_problems(sequence, (key,)))
        problems.extend(shape_problems(sequence, (key,)))
        for index, entry in enumerate(sequence):
            problems.extend(run_entity_rules(CODECS, entry, inner_document, (key, index)))
    return tuple(problems)


@entity_rule(_ARRAY_V3, CODECS, SHARDING_INDEXED_CODEC_NAME, requires=frozenset({"chunk_grid"}))
def inner_chunks_tile_the_enclosing_chunk(
    configuration: Mapping[str, object], document: Mapping[str, object]
) -> tuple[ValidationProblem, ...]:
    """The inner chunk must rank-match and evenly divide its enclosing chunk.

    The enclosing chunk is the `regular` grid's chunk shape at the top
    level. Nested shards are handled by `inner_pipelines_are_pipelines`,
    which re-enters this rule with the enclosing shard's inner chunk as
    the grid — each sharding level encloses the next.
    """
    outer = _enclosing_chunk_shape(document)
    if outer is None:
        return ()
    inner = cast("tuple[int, ...]", configuration["chunk_shape"])
    if len(inner) != len(outer):
        return (
            ValidationProblem(
                ("chunk_shape",),
                f"chunk_shape has {len(inner)} entries but the enclosing chunk has "
                f"{len(outer)} dimensions",
                "invalid_value",
            ),
        )
    return tuple(
        ValidationProblem(
            ("chunk_shape", position),
            f"inner chunk extent {inner_extent} does not evenly divide the "
            f"enclosing chunk extent {outer_extent}",
            "invalid_value",
        )
        for position, (outer_extent, inner_extent) in enumerate(zip(outer, inner, strict=True))
        if inner_extent >= 1 and outer_extent % inner_extent != 0
    )


def _enclosing_chunk_shape(document: Mapping[str, object]) -> tuple[int, ...] | None:
    """The chunk shape enclosing this codec, or None if not determinable.

    At the top level that is the `regular` grid's `chunk_shape`; inside a
    shard the synthetic document carries the enclosing inner chunk as a
    regular grid of the same shape. Non-positive extents decline — the
    values rule owns that complaint.
    """
    from zarr_metadata.v3._shape import entity_name

    grid = document.get("chunk_grid")
    if entity_name(grid) != REGULAR_CHUNK_GRID_NAME:
        return None
    configuration = entity_configuration(CHUNK_GRID, grid)
    if configuration is None:
        return None
    extents = configuration.get("chunk_shape")
    if not isinstance(extents, tuple):
        return None
    values = cast("tuple[object, ...]", extents)
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in values):
        return None
    return cast("tuple[int, ...]", values)
