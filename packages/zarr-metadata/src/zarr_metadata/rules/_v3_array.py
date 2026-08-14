"""Composition rules for v3 array metadata documents.

The model layer's validators check JSON *structure* and never interpret
extension points. The rules here are the complementary *composition*
layer: cross-field consistency checks (fill value vs. data type, codec
pipeline kind ordering, dimension counts) that only make sense once the
structure is trusted.

Extension openness: rules never reject what they cannot interpret. An
unknown data type name accepts any fill value here (its own validator is
whoever understands it), an unknown codec has unknown kind, and unknown
configuration contents pass through untouched. Openness is for genuinely
unknown names only, though: a codec or chunk-grid name this package
defines is held to its full canonical shape (via
`zarr_metadata.v3._shape`), and a known codec ranks as its pipeline kind
in every spelling — otherwise a misspelled known name would masquerade
as an unknown extension and silently escape both the shape and the
ordering checks.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._engine import (
    Rule,
    as_sequence,
    as_string_mapping,
    prefixed,
)
from zarr_metadata.v3._shape import (
    blocking_problems,
    entity_name,
    validate_known_chunk_grid_metadata,
    validate_known_codec_metadata,
)
from zarr_metadata.v3.chunk_grid.rectilinear import RECTILINEAR_CHUNK_GRID_NAME
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID_NAME
from zarr_metadata.v3.codec.kind import codec_kind_of_name
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEXED_CODEC_NAME
from zarr_metadata.v3.codec.transpose import TRANSPOSE_CODEC_NAME
from zarr_metadata.v3.data_type.bytes import base64_bytes
from zarr_metadata.v3.data_type.float16 import hex_float16
from zarr_metadata.v3.data_type.float32 import hex_float32
from zarr_metadata.v3.data_type.float64 import hex_float64
from zarr_metadata.v3.data_type.raw import raw_bytes_dtype_name

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from zarr_metadata.v3.codec.kind import CodecKind


# ---------------------------------------------------------------------------
# fill_value vs. data_type
# ---------------------------------------------------------------------------

_INT_RANGES: Final[dict[str, tuple[int, int]]] = {
    "int8": (-(2**7), 2**7 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
    "uint8": (0, 2**8 - 1),
    "uint16": (0, 2**16 - 1),
    "uint32": (0, 2**32 - 1),
    "uint64": (0, 2**64 - 1),
}

_FLOAT_HEX_VALIDATORS: Final[dict[str, Callable[[str], object]]] = {
    "float16": hex_float16,
    "float32": hex_float32,
    "float64": hex_float64,
}

_COMPLEX_COMPONENT_TYPES: Final[dict[str, str]] = {
    "complex64": "float32",
    "complex128": "float64",
}

_RAW_BYTES_NAME_RE: Final = re.compile(r"^r(\d+)$")

_FLOAT_SPECIALS: Final = frozenset({"NaN", "Infinity", "-Infinity"})


def _is_int(value: object) -> bool:
    # bool is an int subtype but is never a valid integer fill value.
    return isinstance(value, int) and not isinstance(value, bool)


def _check_float_fill(value: object, dtype_name: str) -> str | None:
    if _is_int(value) or isinstance(value, float):
        return None
    if isinstance(value, str):
        if value in _FLOAT_SPECIALS:
            return None
        try:
            _FLOAT_HEX_VALIDATORS[dtype_name](value)
        except ValueError:
            return (
                f"expected a number, one of 'NaN'/'Infinity'/'-Infinity', or a "
                f"{dtype_name} hex string, got {value!r}"
            )
        return None
    return f"expected a number or string, got {value!r}"


def _check_byte_sequence(value: object, expected_len: int | None) -> str | None:
    items = as_sequence(value)
    if items is None:
        return f"expected an array of byte values, got {value!r}"
    if expected_len is not None and len(items) != expected_len:
        return f"expected {expected_len} byte values, got {len(items)}"
    for item in items:
        if not _is_int(item) or not 0 <= cast(int, item) <= 255:
            return f"expected integers in [0, 255], got {item!r}"
    return None


def _check_fill_for_dtype(dtype_name: str, value: object) -> str | None:
    """Why `value` is not a valid fill value for `dtype_name`, or None.

    Unknown data type names accept anything (extension openness).
    """
    if dtype_name == "bool":
        return None if isinstance(value, bool) else f"expected a boolean, got {value!r}"
    if dtype_name in _INT_RANGES:
        low, high = _INT_RANGES[dtype_name]
        if not _is_int(value):
            return f"expected an integer, got {value!r}"
        if not low <= cast(int, value) <= high:
            return f"expected an integer in [{low}, {high}], got {value!r}"
        return None
    if dtype_name in _FLOAT_HEX_VALIDATORS:
        return _check_float_fill(value, dtype_name)
    if dtype_name in _COMPLEX_COMPONENT_TYPES:
        component = _COMPLEX_COMPONENT_TYPES[dtype_name]
        pair = as_sequence(value)
        if pair is None or len(pair) != 2:
            return f"expected a [real, imag] pair, got {value!r}"
        for part in pair:
            reason = _check_float_fill(part, component)
            if reason is not None:
                return f"invalid component: {reason}"
        return None
    if dtype_name == "string":
        return None if isinstance(value, str) else f"expected a string, got {value!r}"
    if dtype_name == "bytes":
        if isinstance(value, str):
            try:
                base64_bytes(value)
            except ValueError:
                return f"expected standard-alphabet base64, got {value!r}"
            return None
        return _check_byte_sequence(value, None)
    if dtype_name in ("numpy.datetime64", "numpy.timedelta64"):
        if value == "NaT" or _is_int(value):
            return None
        return f"expected an integer or 'NaT', got {value!r}"
    if dtype_name == "struct":
        if isinstance(value, Mapping):
            return None
        return f"expected an object of per-field fill values, got {value!r}"
    if _RAW_BYTES_NAME_RE.fullmatch(dtype_name) is not None:
        try:
            raw_bytes_dtype_name(dtype_name)
        except ValueError:
            return None  # malformed r<N> name: _check_data_type_spelling reports it
        return _check_byte_sequence(value, int(dtype_name[1:]) // 8)
    return None  # unknown data type: its fill values are not ours to judge


def _dtype_name(data_type: object) -> str | None:
    if isinstance(data_type, str):
        return data_type
    mapping = as_string_mapping(data_type)
    if mapping is not None:
        name = mapping.get("name")
        if isinstance(name, str):
            return name
    return None  # structurally invalid; the structural validator reports it


def _check_data_type_spelling(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """Misspellings of data type families this package defines.

    An `r<N>` name whose bit count is not a positive multiple of 8 is a
    misspelling of the known raw-bytes family, not an unknown extension:
    treating it as unknown would let the misspelling masquerade as an
    extension and escape judgment entirely (the same anti-masquerade
    reasoning as the codec spelling checks). Genuinely unknown names pass
    untouched.
    """
    name = _dtype_name(document["data_type"])
    if name is None or _RAW_BYTES_NAME_RE.fullmatch(name) is None:
        return ()
    try:
        raw_bytes_dtype_name(name)
    except ValueError as error:
        return (ValidationProblem(("data_type",), str(error), "invalid_value"),)
    return ()


def _check_fill_matches_dtype(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    dtype_name = _dtype_name(document["data_type"])
    if dtype_name is None:
        return ()
    reason = _check_fill_for_dtype(dtype_name, document["fill_value"])
    if reason is None:
        return ()
    return (
        ValidationProblem(
            ("fill_value",),
            f"fill_value invalid for data_type {dtype_name!r}: {reason}",
            "invalid_value",
        ),
    )


# ---------------------------------------------------------------------------
# codec pipeline kind ordering
# ---------------------------------------------------------------------------

_KIND_RANK: Final = {"array_array": 0, "array_bytes": 1, "bytes_bytes": 2}


def _codec_kind(codec: object) -> CodecKind | None:
    """The pipeline kind of `codec`, classified by name alone.

    Spelling-insensitive on purpose: a known codec in an invalid spelling
    (bare `"transpose"`, config-less `"gzip"` object) still ranks as its
    kind, so two spellings of the same pipeline always get the same
    ordering verdict and a misspelled known codec is never mistaken for an
    unknown extension (which would suppress the exactly-one-`array->bytes`
    count). Spelling validity itself is `_check_codec_spellings`' job.
    """
    name = entity_name(codec)
    if name is None:
        return None
    return codec_kind_of_name(name)


def _codec_label(codec: object) -> str:
    if isinstance(codec, str):
        return repr(codec)
    mapping = as_string_mapping(codec)
    if mapping is not None:
        return repr(mapping.get("name"))
    return repr(codec)


def _pipeline_order_problems(
    entries: Sequence[object], loc: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    """The spec pipeline shape: `array->array`* `array->bytes` `bytes->bytes`*.

    Codecs are classified by name across every spelling (see
    `_codec_kind`). Codecs of genuinely unknown name are skipped: they
    impose no ordering constraint, and their presence makes the
    exactly-one-`array->bytes` count inconclusive (an unknown codec might
    be the pipeline's `array->bytes` stage), so that check only fires when
    every codec is classified.
    """
    problems: list[ValidationProblem] = []
    kinds = [_codec_kind(codec) for codec in entries]
    max_rank_seen = -1
    array_bytes_count = 0
    for index, (codec, kind) in enumerate(zip(entries, kinds, strict=True)):
        if kind is None:
            continue
        rank = _KIND_RANK[kind]
        if rank < max_rank_seen:
            problems.append(
                ValidationProblem(
                    (*loc, index),
                    f"{kind.replace('_', '->')} codec {_codec_label(codec)} may not "
                    "follow a later-stage codec in the pipeline",
                    "invalid_value",
                )
            )
        max_rank_seen = max(max_rank_seen, rank)
        if kind == "array_bytes":
            array_bytes_count += 1
            if array_bytes_count > 1:
                problems.append(
                    ValidationProblem(
                        (*loc, index),
                        f"extra array->bytes codec {_codec_label(codec)}: a pipeline "
                        "has exactly one",
                        "invalid_value",
                    )
                )
    if array_bytes_count == 0 and all(kind is not None for kind in kinds):
        problems.append(
            ValidationProblem(
                loc,
                "codec pipeline has no array->bytes codec",
                "invalid_value",
            )
        )
    return tuple(problems)


def _check_codec_pipeline_order(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    entries = as_sequence(document["codecs"])
    if entries is None:
        return ()
    return _pipeline_order_problems(entries, ("codecs",))


# ---------------------------------------------------------------------------
# known-name spellings
# ---------------------------------------------------------------------------


def _spelling_problems(
    entries: Sequence[object], loc: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    """Shape problems for every known-name codec entry in `entries`.

    Delegates to the type-level validators in `zarr_metadata.v3._shape`,
    so known names are held to their full canonical shapes — spelling
    form, key sets, and configuration value types. Unknown names pass
    untouched (extension openness), and entries without an interpretable
    name decline in favor of the structural validator.
    """
    problems: list[ValidationProblem] = []
    for index, codec in enumerate(entries):
        found = validate_known_codec_metadata(codec)
        # None is "not a known codec" (unjudged); () is "known and valid".
        if found is not None:
            problems.extend(prefixed((*loc, index), found))
    return tuple(problems)


def _check_codec_spellings(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    entries = as_sequence(document["codecs"])
    if entries is None:
        return ()
    return _spelling_problems(entries, ("codecs",))


def _check_chunk_grid_spelling(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    found = validate_known_chunk_grid_metadata(document["chunk_grid"])
    # None is "not a known grid" (unjudged); () is "known and valid".
    if found is None:
        return ()
    return prefixed(("chunk_grid",), found)


# ---------------------------------------------------------------------------
# dimension counts
# ---------------------------------------------------------------------------


def _check_dimension_names_length(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    shape = as_sequence(document["shape"])
    names = as_sequence(document["dimension_names"])
    if shape is None or names is None:
        return ()
    if len(names) == len(shape):
        return ()
    return (
        ValidationProblem(
            ("dimension_names",),
            f"dimension_names has {len(names)} entries but shape has {len(shape)} dimensions",
            "invalid_value",
        ),
    )


# ---------------------------------------------------------------------------
# chunk grid geometry and values
# ---------------------------------------------------------------------------


def _valid_grid_configuration(grid: object) -> tuple[str, Mapping[str, object]] | None:
    """`grid`'s (name, configuration) if its modelled fields are usable.

    Declines (None) for unknown grids and for known grids whose shape is
    broken in a way that makes the fields uninterpretable — the spelling
    rule owns that complaint. An `unknown_key` does not decline: a member
    we do not model says nothing about the ones we do, and treating it as
    fatal would let a cosmetic extra key silently suppress the geometry
    checks below.
    """
    verdict = validate_known_chunk_grid_metadata(grid)
    if verdict is None or len(blocking_problems(verdict)) != 0:
        return None
    mapping = as_string_mapping(grid)
    if mapping is None:
        return None
    name = mapping.get("name")
    configuration = as_string_mapping(mapping.get("configuration"))
    if not isinstance(name, str) or configuration is None:
        return None
    return name, configuration


def _expanded_dim_extent(spec: Sequence[object]) -> int | None:
    """The total extent an explicit rectilinear dim spec covers, or None.

    Entries are chunk sizes or `[size, count]` run-length pairs. Answers
    None when any entry is non-positive (the values rule owns that
    complaint, and a sum over bad entries would be noise).
    """
    total = 0
    for item in spec:
        if _is_int(item) and cast(int, item) >= 1:
            total += cast(int, item)
        elif isinstance(item, tuple):
            size, count = cast("tuple[object, object]", item)
            if not (_is_int(size) and _is_int(count)):
                return None
            if cast(int, size) < 1 or cast(int, count) < 1:
                return None
            total += cast(int, size) * cast(int, count)
        else:
            return None
    return total


def _check_chunk_grid_values(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """Chunk extents must be positive, in both known grids.

    The shape validators enforce the value *types* (integers, RLE pairs);
    positivity is a value judgment and lives here.
    """
    valid = _valid_grid_configuration(document["chunk_grid"])
    if valid is None:
        return ()
    name, configuration = valid
    problems: list[ValidationProblem] = []
    if name == REGULAR_CHUNK_GRID_NAME:
        chunk_shape = cast("tuple[int, ...]", configuration["chunk_shape"])
        loc = ("chunk_grid", "configuration", "chunk_shape")
        problems.extend(
            ValidationProblem(
                (*loc, position), f"expected a positive chunk extent, got {extent}", "invalid_value"
            )
            for position, extent in enumerate(chunk_shape)
            if extent < 1
        )
    elif name == RECTILINEAR_CHUNK_GRID_NAME:
        chunk_shapes = cast("tuple[object, ...]", configuration["chunk_shapes"])
        for dim, spec in enumerate(chunk_shapes):
            loc = ("chunk_grid", "configuration", "chunk_shapes", dim)
            if _is_int(spec):
                if cast(int, spec) < 1:
                    problems.append(
                        ValidationProblem(
                            loc, f"expected a positive chunk extent, got {spec}", "invalid_value"
                        )
                    )
                continue
            for position, item in enumerate(cast("tuple[object, ...]", spec)):
                if _is_int(item) and cast(int, item) < 1:
                    problems.append(
                        ValidationProblem(
                            (*loc, position),
                            f"expected a positive chunk extent, got {item}",
                            "invalid_value",
                        )
                    )
                elif isinstance(item, tuple):
                    size, count = cast("tuple[int, int]", item)
                    if size < 1 or count < 1:
                        problems.append(
                            ValidationProblem(
                                (*loc, position),
                                f"expected a positive [size, count] pair, got {item!r}",
                                "invalid_value",
                            )
                        )
    return tuple(problems)


def _check_chunk_grid_geometry(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """Known chunk grids must agree with `shape` on dimensionality.

    `regular` grids must chunk every array dimension; `rectilinear` grids
    additionally pin each explicitly-listed dimension's chunk sizes to sum
    to that dimension's extent (bare-integer dim specs are uniform
    shorthand and impose no sum constraint). Unknown grids pass through.
    """
    shape = as_sequence(document["shape"])
    valid = _valid_grid_configuration(document["chunk_grid"])
    if shape is None or valid is None:
        return ()
    name, configuration = valid
    if name == REGULAR_CHUNK_GRID_NAME:
        chunk_shape = cast("tuple[int, ...]", configuration["chunk_shape"])
        if len(chunk_shape) == len(shape):
            return ()
        return (
            ValidationProblem(
                ("chunk_grid", "configuration", "chunk_shape"),
                f"chunk_shape has {len(chunk_shape)} entries but shape has {len(shape)} dimensions",
                "invalid_value",
            ),
        )
    if name == RECTILINEAR_CHUNK_GRID_NAME:
        chunk_shapes = cast("tuple[object, ...]", configuration["chunk_shapes"])
        if len(chunk_shapes) != len(shape):
            return (
                ValidationProblem(
                    ("chunk_grid", "configuration", "chunk_shapes"),
                    f"chunk_shapes has {len(chunk_shapes)} entries but shape has "
                    f"{len(shape)} dimensions",
                    "invalid_value",
                ),
            )
        problems: list[ValidationProblem] = []
        for dim, (spec, extent) in enumerate(zip(chunk_shapes, shape, strict=True)):
            if not _is_int(extent) or _is_int(spec) or not isinstance(spec, tuple):
                continue
            total = _expanded_dim_extent(cast("tuple[object, ...]", spec))
            if total is not None and total != extent:
                problems.append(
                    ValidationProblem(
                        ("chunk_grid", "configuration", "chunk_shapes", dim),
                        f"chunk sizes sum to {total} but shape[{dim}] is {extent}",
                        "invalid_value",
                    )
                )
        return tuple(problems)
    return ()


# ---------------------------------------------------------------------------
# codec configurations against the rest of the document
# ---------------------------------------------------------------------------


def _named_configurations(
    entries: Sequence[object], name: str
) -> Iterator[tuple[int, Mapping[str, object]]]:
    """(index, configuration) for each shape-valid codec named `name`.

    Shape-invalid entries decline — the spelling rule owns their
    complaints, and value judgments over a malformed configuration would
    be noise on top of them.
    """
    for index, codec in enumerate(entries):
        if entity_name(codec) != name:
            continue
        verdict = validate_known_codec_metadata(codec)
        if verdict is None or len(blocking_problems(verdict)) != 0 or isinstance(codec, str):
            continue
        mapping = as_string_mapping(codec)
        if mapping is None:
            continue
        configuration = as_string_mapping(mapping.get("configuration"))
        if configuration is not None:
            yield index, configuration


def _transpose_order_problems(
    entries: Sequence[object], loc: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    problems: list[ValidationProblem] = []
    for index, configuration in _named_configurations(entries, TRANSPOSE_CODEC_NAME):
        order = cast("tuple[int, ...]", configuration["order"])
        if sorted(order) != list(range(len(order))):
            problems.append(
                ValidationProblem(
                    (*loc, index, "configuration", "order"),
                    f"expected a permutation of 0..{len(order) - 1}, got {order!r}",
                    "invalid_value",
                )
            )
    return tuple(problems)


def _check_transpose_orders(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """A transpose `order` must be a permutation of its own indices."""
    entries = as_sequence(document["codecs"])
    if entries is None:
        return ()
    return _transpose_order_problems(entries, ("codecs",))


def _check_transpose_rank(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """A top-level transpose `order` must have one entry per dimension of shape."""
    shape = as_sequence(document["shape"])
    entries = as_sequence(document["codecs"])
    if shape is None or entries is None:
        return ()
    problems: list[ValidationProblem] = []
    for index, configuration in _named_configurations(entries, TRANSPOSE_CODEC_NAME):
        order = cast("tuple[int, ...]", configuration["order"])
        if len(order) != len(shape):
            problems.append(
                ValidationProblem(
                    ("codecs", index, "configuration", "order"),
                    f"order has {len(order)} entries but shape has {len(shape)} dimensions",
                    "invalid_value",
                )
            )
    return tuple(problems)


# ---------------------------------------------------------------------------
# sharding: inner pipelines and geometry
# ---------------------------------------------------------------------------


def _inner_pipeline_problems(
    entries: Sequence[object], loc: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    """Pipeline judgments inside shape-valid sharding codecs, recursively.

    A sharding codec's `codecs` and `index_codecs` are pipelines like any
    other: the same kind-ordering, known-shape, and transpose-permutation
    checks that judge the top-level pipeline judge them, at every nesting
    depth. Inner chunk extents must also be positive.
    """
    problems: list[ValidationProblem] = []
    for index, configuration in _named_configurations(entries, SHARDING_INDEXED_CODEC_NAME):
        chunk_shape = cast("tuple[int, ...]", configuration["chunk_shape"])
        problems.extend(
            ValidationProblem(
                (*loc, index, "configuration", "chunk_shape", position),
                f"expected a positive chunk extent, got {extent}",
                "invalid_value",
            )
            for position, extent in enumerate(chunk_shape)
            if extent < 1
        )
        for key in ("codecs", "index_codecs"):
            inner = cast("tuple[object, ...]", configuration[key])
            inner_loc = (*loc, index, "configuration", key)
            problems.extend(_pipeline_order_problems(inner, inner_loc))
            problems.extend(_spelling_problems(inner, inner_loc))
            problems.extend(_transpose_order_problems(inner, inner_loc))
            problems.extend(_inner_pipeline_problems(inner, inner_loc))
    return tuple(problems)


def _check_sharding_pipelines(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    entries = as_sequence(document["codecs"])
    if entries is None:
        return ()
    return _inner_pipeline_problems(entries, ("codecs",))


def _sharding_geometry_problems(
    entries: Sequence[object], outer: Sequence[int], loc: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    problems: list[ValidationProblem] = []
    for index, configuration in _named_configurations(entries, SHARDING_INDEXED_CODEC_NAME):
        inner = cast("tuple[int, ...]", configuration["chunk_shape"])
        codec_loc = (*loc, index, "configuration", "chunk_shape")
        if len(inner) != len(outer):
            problems.append(
                ValidationProblem(
                    codec_loc,
                    f"chunk_shape has {len(inner)} entries but the enclosing chunk has "
                    f"{len(outer)} dimensions",
                    "invalid_value",
                )
            )
            continue
        for position, (outer_extent, inner_extent) in enumerate(zip(outer, inner, strict=True)):
            if inner_extent >= 1 and outer_extent % inner_extent != 0:
                problems.append(
                    ValidationProblem(
                        (*codec_loc, position),
                        f"inner chunk extent {inner_extent} does not evenly divide the "
                        f"enclosing chunk extent {outer_extent}",
                        "invalid_value",
                    )
                )
        problems.extend(
            _sharding_geometry_problems(
                cast("tuple[object, ...]", configuration["codecs"]),
                inner,
                (*loc, index, "configuration", "codecs"),
            )
        )
    return tuple(problems)


def _check_sharding_geometry(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """Sharding inner chunks must tile the enclosing chunk, at every depth.

    Fires when the chunk grid is a shape-valid `regular` grid with
    positive extents; the inner `chunk_shape` must rank-match and evenly
    divide the enclosing chunk shape, recursively through nested sharding
    (each level's `chunk_shape` encloses the next).
    """
    entries = as_sequence(document["codecs"])
    valid = _valid_grid_configuration(document["chunk_grid"])
    if entries is None or valid is None:
        return ()
    name, configuration = valid
    if name != REGULAR_CHUNK_GRID_NAME:
        return ()
    chunk_shape = cast("tuple[int, ...]", configuration["chunk_shape"])
    if any(extent < 1 for extent in chunk_shape):
        return ()  # the values rule owns that complaint
    return _sharding_geometry_problems(entries, chunk_shape, ("codecs",))


ZARR_V3_ARRAY_RULES: Final[tuple[Rule, ...]] = (
    Rule(frozenset({"data_type"}), _check_data_type_spelling),
    Rule(frozenset({"data_type", "fill_value"}), _check_fill_matches_dtype),
    Rule(frozenset({"codecs"}), _check_codec_pipeline_order),
    Rule(frozenset({"codecs"}), _check_codec_spellings),
    Rule(frozenset({"codecs"}), _check_transpose_orders),
    Rule(frozenset({"codecs"}), _check_sharding_pipelines),
    Rule(frozenset({"chunk_grid"}), _check_chunk_grid_spelling),
    Rule(frozenset({"chunk_grid"}), _check_chunk_grid_values),
    Rule(frozenset({"shape", "dimension_names"}), _check_dimension_names_length),
    Rule(frozenset({"shape", "chunk_grid"}), _check_chunk_grid_geometry),
    Rule(frozenset({"shape", "codecs"}), _check_transpose_rank),
    Rule(frozenset({"chunk_grid", "codecs"}), _check_sharding_geometry),
)
"""The composition rule set for v3 array metadata documents."""


__all__ = [
    "ZARR_V3_ARRAY_RULES",
]
