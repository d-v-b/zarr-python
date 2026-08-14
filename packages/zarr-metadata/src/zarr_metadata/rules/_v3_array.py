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
    entity_name,
    validate_known_chunk_grid_metadata,
    validate_known_codec_metadata,
)
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID_NAME
from zarr_metadata.v3.codec.kind import codec_kind_of_name
from zarr_metadata.v3.data_type.bytes import base64_bytes
from zarr_metadata.v3.data_type.float16 import hex_float16
from zarr_metadata.v3.data_type.float32 import hex_float32
from zarr_metadata.v3.data_type.float64 import hex_float64
from zarr_metadata.v3.data_type.raw import raw_bytes_dtype_name

if TYPE_CHECKING:
    from collections.abc import Callable

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


def _check_data_type_spelling(document: Mapping[str, object]) -> list[ValidationProblem]:
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
        return []
    try:
        raw_bytes_dtype_name(name)
    except ValueError as error:
        return [ValidationProblem(("data_type",), str(error), "invalid_value")]
    return []


def _check_fill_matches_dtype(document: Mapping[str, object]) -> list[ValidationProblem]:
    dtype_name = _dtype_name(document["data_type"])
    if dtype_name is None:
        return []
    reason = _check_fill_for_dtype(dtype_name, document["fill_value"])
    if reason is None:
        return []
    return [
        ValidationProblem(
            ("fill_value",),
            f"fill_value invalid for data_type {dtype_name!r}: {reason}",
            "invalid_value",
        )
    ]


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


def _check_codec_pipeline_order(document: Mapping[str, object]) -> list[ValidationProblem]:
    """The spec pipeline shape: `array->array`* `array->bytes` `bytes->bytes`*.

    Codecs are classified by name across every spelling (see
    `_codec_kind`). Codecs of genuinely unknown name are skipped: they
    impose no ordering constraint, and their presence makes the
    exactly-one-`array->bytes` count inconclusive (an unknown codec might
    be the pipeline's `array->bytes` stage), so that check only fires when
    every codec is classified.
    """
    entries = as_sequence(document["codecs"])
    if entries is None:
        return []
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
                    ("codecs", index),
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
                        ("codecs", index),
                        f"extra array->bytes codec {_codec_label(codec)}: a pipeline "
                        "has exactly one",
                        "invalid_value",
                    )
                )
    if array_bytes_count == 0 and all(kind is not None for kind in kinds):
        problems.append(
            ValidationProblem(
                ("codecs",),
                "codec pipeline has no array->bytes codec",
                "invalid_value",
            )
        )
    return problems


# ---------------------------------------------------------------------------
# known-name spellings
# ---------------------------------------------------------------------------


def _check_codec_spellings(document: Mapping[str, object]) -> list[ValidationProblem]:
    """Shape problems for every known-name codec entry.

    Delegates to the type-level validators in `zarr_metadata.v3._shape`,
    so known names are held to their full canonical shapes — spelling
    form, key sets, and configuration value types. Unknown names pass
    untouched (extension openness), and entries without an interpretable
    name decline in favor of the structural validator.
    """
    entries = as_sequence(document["codecs"])
    if entries is None:
        return []
    problems: list[ValidationProblem] = []
    for index, codec in enumerate(entries):
        found = validate_known_codec_metadata(codec)
        if found:
            problems.extend(prefixed(("codecs", index), found))
    return problems


def _check_chunk_grid_spelling(document: Mapping[str, object]) -> list[ValidationProblem]:
    found = validate_known_chunk_grid_metadata(document["chunk_grid"])
    if not found:
        return []
    return prefixed(("chunk_grid",), found)


# ---------------------------------------------------------------------------
# dimension counts
# ---------------------------------------------------------------------------


def _check_dimension_names_length(document: Mapping[str, object]) -> list[ValidationProblem]:
    shape = as_sequence(document["shape"])
    names = as_sequence(document["dimension_names"])
    if shape is None or names is None:
        return []
    if len(names) == len(shape):
        return []
    return [
        ValidationProblem(
            ("dimension_names",),
            f"dimension_names has {len(names)} entries but shape has {len(shape)} dimensions",
            "invalid_value",
        )
    ]


def _check_regular_grid_dimensions(document: Mapping[str, object]) -> list[ValidationProblem]:
    """`regular` chunk grids must chunk every array dimension.

    Only fires for a grid this package can interpret: an object named
    `regular` with a `chunk_shape` array in its configuration. Other grids
    are extension points and pass through.
    """
    shape = as_sequence(document["shape"])
    grid = as_string_mapping(document["chunk_grid"])
    if shape is None or grid is None or grid.get("name") != REGULAR_CHUNK_GRID_NAME:
        return []
    configuration = as_string_mapping(grid.get("configuration"))
    if configuration is None:
        return []
    chunk_shape = as_sequence(configuration.get("chunk_shape"))
    if chunk_shape is None or len(chunk_shape) == len(shape):
        return []
    return [
        ValidationProblem(
            ("chunk_grid", "configuration", "chunk_shape"),
            f"chunk_shape has {len(chunk_shape)} entries but shape has {len(shape)} dimensions",
            "invalid_value",
        )
    ]


ZARR_V3_ARRAY_RULES: Final[tuple[Rule, ...]] = (
    Rule(frozenset({"data_type"}), _check_data_type_spelling),
    Rule(frozenset({"data_type", "fill_value"}), _check_fill_matches_dtype),
    Rule(frozenset({"codecs"}), _check_codec_pipeline_order),
    Rule(frozenset({"codecs"}), _check_codec_spellings),
    Rule(frozenset({"chunk_grid"}), _check_chunk_grid_spelling),
    Rule(frozenset({"shape", "dimension_names"}), _check_dimension_names_length),
    Rule(frozenset({"shape", "chunk_grid"}), _check_regular_grid_dimensions),
)
"""The semantic rule set for v3 array metadata documents."""


__all__ = [
    "ZARR_V3_ARRAY_RULES",
]
