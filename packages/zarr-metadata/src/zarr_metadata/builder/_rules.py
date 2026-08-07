"""Semantic validation rules for v3 array metadata documents.

The model layer's validators check JSON *structure* and never interpret
extension points. The rules here are the complementary *semantic* layer:
cross-field consistency checks (fill value vs. data type, codec pipeline
kind ordering, dimension counts) that only make sense once the structure
is trusted.

Rules are data, keyed by the document keys they depend on. A rule fires
only when every key it needs is present, so field ordering during
incremental construction is unconstrained: setting `fill_value` before
`data_type` is legal, and the pair is checked as soon as both exist.
Running the same rule list over a complete document (where dependency
completeness is trivially true) gives parsed metadata identical checking
to incrementally-built metadata.

Checks read their document as `Mapping[str, object]` and verify every
shape they touch before interpreting it: rules may run over partial
documents that have not passed structural validation, so a check finding
a structurally-unexpected value simply declines (returns no problems) and
leaves the complaint to the structural validator.

Extension openness: rules never reject what they cannot interpret. An
unknown data type name accepts any fill value here (its own validator is
whoever understands it), an unknown codec has unknown kind, and unknown
configuration contents pass through untouched. Openness is for genuinely
unknown names only, though: a codec or chunk-grid name this package
defines is held to its canonical spellings (bare form only where the
concrete type permits it, required configuration keys present), and a
known codec ranks as its pipeline kind in every spelling — otherwise a
misspelled known name would masquerade as an unknown extension and
silently escape both the spelling and the ordering checks.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3.chunk_grid.rectilinear import (
    RECTILINEAR_CHUNK_GRID_NAME,
    RectilinearChunkGridConfiguration,
    RectilinearChunkGridObject,
)
from zarr_metadata.v3.chunk_grid.regular import (
    REGULAR_CHUNK_GRID_NAME,
    RegularChunkGridConfiguration,
    RegularChunkGridObject,
)
from zarr_metadata.v3.codec.blosc import BLOSC_CODEC_NAME, BloscCodecConfiguration, BloscCodecObject
from zarr_metadata.v3.codec.bytes import BYTES_CODEC_NAME, BytesCodecConfiguration, BytesCodecObject
from zarr_metadata.v3.codec.cast_value import (
    CAST_VALUE_CODEC_NAME,
    CastValueCodecConfiguration,
    CastValueCodecObject,
)
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC_NAME, Crc32cCodecObject, Empty
from zarr_metadata.v3.codec.gzip import GZIP_CODEC_NAME, GzipCodecConfiguration, GzipCodecObject
from zarr_metadata.v3.codec.kind import codec_kind_of_name
from zarr_metadata.v3.codec.scale_offset import (
    SCALE_OFFSET_CODEC_NAME,
    ScaleOffsetCodecConfiguration,
    ScaleOffsetCodecObject,
)
from zarr_metadata.v3.codec.sharding_indexed import (
    SHARDING_INDEXED_CODEC_NAME,
    ShardingIndexedCodecConfiguration,
    ShardingIndexedCodecObject,
)
from zarr_metadata.v3.codec.transpose import (
    TRANSPOSE_CODEC_NAME,
    TransposeCodecConfiguration,
    TransposeCodecObject,
)
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC_NAME, ZstdCodecConfiguration, ZstdCodecObject
from zarr_metadata.v3.data_type.bytes import base64_bytes
from zarr_metadata.v3.data_type.float16 import hex_float16
from zarr_metadata.v3.data_type.float32 import hex_float32
from zarr_metadata.v3.data_type.float64 import hex_float64

if TYPE_CHECKING:
    from zarr_metadata.v3.codec.kind import CodecKind


@dataclass(frozen=True, slots=True)
class Rule:
    """One semantic check over a (possibly partial) metadata document.

    `keys` are the document keys the check reads; the rule fires only when
    all of them are present. `check` receives the whole document (so
    coupled fields are examined together) and returns every problem it
    finds, empty when the rule passes.
    """

    keys: frozenset[str]
    check: Callable[[Mapping[str, object]], list[ValidationProblem]]


def applicable(rules: Sequence[Rule], present: AbstractSet[str]) -> Iterator[Rule]:
    """The subset of `rules` whose dependencies are all present."""
    return (rule for rule in rules if rule.keys <= present)


def run_rules(rules: Sequence[Rule], document: Mapping[str, object]) -> list[ValidationProblem]:
    """Run every dependency-complete rule over `document`, collecting all problems."""
    problems: list[ValidationProblem] = []
    for rule in applicable(rules, document.keys()):
        problems.extend(rule.check(document))
    return problems


def _as_string_mapping(value: object) -> Mapping[str, object] | None:
    """`value` as a string-keyed mapping, or None if it is not one."""
    if not isinstance(value, Mapping):
        return None
    mapping = cast("Mapping[object, object]", value)
    if any(not isinstance(key, str) for key in mapping):
        return None
    return cast("Mapping[str, object]", mapping)


def _as_sequence(value: object) -> Sequence[object] | None:
    """`value` as a JSON-array-shaped sequence, or None if it is not one."""
    if isinstance(value, (list, tuple)):
        return cast("Sequence[object]", value)
    return None


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
    items = _as_sequence(value)
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
        pair = _as_sequence(value)
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
    raw = _RAW_BYTES_NAME_RE.fullmatch(dtype_name)
    if raw is not None:
        bits = int(raw.group(1))
        if bits > 0 and bits % 8 == 0:
            return _check_byte_sequence(value, bits // 8)
        return None  # malformed r<N> name: a data_type problem, not a fill_value one
    return None  # unknown data type: its fill values are not ours to judge


def _dtype_name(data_type: object) -> str | None:
    if isinstance(data_type, str):
        return data_type
    mapping = _as_string_mapping(data_type)
    if mapping is not None:
        name = mapping.get("name")
        if isinstance(name, str):
            return name
    return None  # structurally invalid; the structural validator reports it


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


def _field_name(value: object) -> str | None:
    """The `name` of a metadata field entry in any spelling, or None.

    A bare string is its own name; an object's name is its `name` member.
    Anything else (or a mapping without a string `name`) has no name and
    stays unclassified — its complaint belongs to the structural validator.
    """
    if isinstance(value, str):
        return value
    mapping = _as_string_mapping(value)
    if mapping is None:
        return None
    name = mapping.get("name")
    return name if isinstance(name, str) else None


def _codec_kind(codec: object) -> CodecKind | None:
    """The pipeline kind of `codec`, classified by name alone.

    Spelling-insensitive on purpose: a known codec in an invalid spelling
    (bare `"transpose"`, config-less `"gzip"` object) still ranks as its
    kind, so two spellings of the same pipeline always get the same
    ordering verdict and a misspelled known codec is never mistaken for an
    unknown extension (which would suppress the exactly-one-`array->bytes`
    count). Spelling validity itself is `_check_codec_spellings`' job.
    """
    name = _field_name(codec)
    if name is None:
        return None
    return codec_kind_of_name(name)


def _codec_label(codec: object) -> str:
    if isinstance(codec, str):
        return repr(codec)
    mapping = _as_string_mapping(codec)
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
    entries = _as_sequence(document["codecs"])
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


@dataclass(frozen=True, slots=True)
class _KnownFieldSpelling:
    """The spelling constraints of one known metadata-field name.

    Derived from the concrete TypedDicts' `__required_keys__` rather than
    restated by hand, so the registry cannot drift from the canonical
    types: `configuration_required` is whether the object form requires a
    `configuration` member (which is also exactly when the bare-string
    short-hand is not part of the canonical union), and
    `required_configuration_keys` are the keys the configuration itself
    requires.
    """

    configuration_required: bool
    required_configuration_keys: frozenset[str]


def _spelling(
    object_required: frozenset[str], configuration_required: frozenset[str]
) -> _KnownFieldSpelling:
    return _KnownFieldSpelling("configuration" in object_required, configuration_required)


_KNOWN_CODEC_SPELLINGS: Final[dict[str, _KnownFieldSpelling]] = {
    BLOSC_CODEC_NAME: _spelling(
        BloscCodecObject.__required_keys__, BloscCodecConfiguration.__required_keys__
    ),
    BYTES_CODEC_NAME: _spelling(
        BytesCodecObject.__required_keys__, BytesCodecConfiguration.__required_keys__
    ),
    CAST_VALUE_CODEC_NAME: _spelling(
        CastValueCodecObject.__required_keys__, CastValueCodecConfiguration.__required_keys__
    ),
    CRC32C_CODEC_NAME: _spelling(Crc32cCodecObject.__required_keys__, Empty.__required_keys__),
    GZIP_CODEC_NAME: _spelling(
        GzipCodecObject.__required_keys__, GzipCodecConfiguration.__required_keys__
    ),
    SCALE_OFFSET_CODEC_NAME: _spelling(
        ScaleOffsetCodecObject.__required_keys__, ScaleOffsetCodecConfiguration.__required_keys__
    ),
    SHARDING_INDEXED_CODEC_NAME: _spelling(
        ShardingIndexedCodecObject.__required_keys__,
        ShardingIndexedCodecConfiguration.__required_keys__,
    ),
    TRANSPOSE_CODEC_NAME: _spelling(
        TransposeCodecObject.__required_keys__, TransposeCodecConfiguration.__required_keys__
    ),
    ZSTD_CODEC_NAME: _spelling(
        ZstdCodecObject.__required_keys__, ZstdCodecConfiguration.__required_keys__
    ),
}

_KNOWN_CHUNK_GRID_SPELLINGS: Final[dict[str, _KnownFieldSpelling]] = {
    REGULAR_CHUNK_GRID_NAME: _spelling(
        RegularChunkGridObject.__required_keys__, RegularChunkGridConfiguration.__required_keys__
    ),
    RECTILINEAR_CHUNK_GRID_NAME: _spelling(
        RectilinearChunkGridObject.__required_keys__,
        RectilinearChunkGridConfiguration.__required_keys__,
    ),
}


def _check_known_spelling(
    value: object,
    known: Mapping[str, _KnownFieldSpelling],
    entity: str,
    loc: tuple[str | int, ...],
) -> list[ValidationProblem]:
    """Spelling problems for one metadata-field entry against `known`.

    Unknown names pass untouched (extension openness); known names must be
    in a spelling their canonical type permits. Only key *presence* is
    checked — configuration value types are structural territory. Entries
    without an interpretable name, and configurations that are not
    string-keyed mappings, decline in favor of the structural validator.
    """
    name = _field_name(value)
    if name is None:
        return []
    spelling = known.get(name)
    if spelling is None:
        return []  # unknown name: its spellings are not ours to judge
    if isinstance(value, str):
        if not spelling.configuration_required:
            return []
        return [
            ValidationProblem(
                loc,
                f"{entity} {name!r} requires a configuration and has no bare short-hand "
                f"form; use {{'name': {name!r}, 'configuration': {{...}}}}",
                "invalid_value",
            )
        ]
    mapping = _as_string_mapping(value)
    if mapping is None:
        return []
    if "configuration" not in mapping:
        if not spelling.configuration_required:
            return []
        return [
            ValidationProblem(
                (*loc, "configuration"),
                f"{entity} {name!r} requires a 'configuration' object",
                "missing_key",
            )
        ]
    configuration = _as_string_mapping(mapping["configuration"])
    if configuration is None:
        return []
    return [
        ValidationProblem(
            (*loc, "configuration", key),
            f"configuration for {entity} {name!r} is missing required key {key!r}",
            "missing_key",
        )
        for key in sorted(spelling.required_configuration_keys - configuration.keys())
    ]


def _check_codec_spellings(document: Mapping[str, object]) -> list[ValidationProblem]:
    entries = _as_sequence(document["codecs"])
    if entries is None:
        return []
    problems: list[ValidationProblem] = []
    for index, codec in enumerate(entries):
        problems.extend(
            _check_known_spelling(codec, _KNOWN_CODEC_SPELLINGS, "codec", ("codecs", index))
        )
    return problems


def _check_chunk_grid_spelling(document: Mapping[str, object]) -> list[ValidationProblem]:
    return _check_known_spelling(
        document["chunk_grid"], _KNOWN_CHUNK_GRID_SPELLINGS, "chunk grid", ("chunk_grid",)
    )


# ---------------------------------------------------------------------------
# dimension counts
# ---------------------------------------------------------------------------


def _check_dimension_names_length(document: Mapping[str, object]) -> list[ValidationProblem]:
    shape = _as_sequence(document["shape"])
    names = _as_sequence(document["dimension_names"])
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
    shape = _as_sequence(document["shape"])
    grid = _as_string_mapping(document["chunk_grid"])
    if shape is None or grid is None or grid.get("name") != "regular":
        return []
    configuration = _as_string_mapping(grid.get("configuration"))
    if configuration is None:
        return []
    chunk_shape = _as_sequence(configuration.get("chunk_shape"))
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
    "Rule",
    "applicable",
    "run_rules",
]
