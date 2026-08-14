"""Composition rules for v3 array metadata documents.

Whole-document rules live here: judgments that read several top-level
fields, or that apply to a field regardless of which extension occupies
it. Rules about a *particular* codec or chunk grid live with that entity
in `zarr_metadata.rules._entities`, registered by name — so adding a
third chunk grid or a new codec adds a module there and changes nothing
in this one. The `codecs` and `chunk_grid` dispatchers below are generic:
they run whatever rules are registered for the name they find.

Extension openness: rules never reject what they cannot interpret. An
unknown data type name accepts any fill value here (its own validator is
whoever understands it), an unknown codec has unknown kind, and unknown
entities pass through untouched. Openness is for genuinely unknown names
only: a codec or chunk-grid name this package defines is held to its full
canonical shape (via `zarr_metadata.v3._shape`), and a known codec ranks
as its pipeline kind in every spelling — otherwise a misspelled known
name would masquerade as an unknown extension and silently escape both
the shape and the ordering checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, cast

from zarr_metadata.model._validation import (
    ARRAY_METADATA_STANDARD_KEYS_V3,
    ValidationProblem,
)
from zarr_metadata.rules._engine import Rule, as_sequence, as_string_mapping, prefixed
from zarr_metadata.rules._pipeline import pipeline_order_problems, shape_problems
from zarr_metadata.rules._registry import (
    dispatch_field,
    dispatch_field_sequence,
    document_rule,
    document_rules,
    register_document_type,
)
from zarr_metadata.v3._shape import validate_known_chunk_grid_metadata
from zarr_metadata.v3.data_type.bytes import base64_bytes
from zarr_metadata.v3.data_type.float16 import hex_float16
from zarr_metadata.v3.data_type.float32 import hex_float32
from zarr_metadata.v3.data_type.float64 import hex_float64
from zarr_metadata.v3.data_type.raw import RAW_BYTES_NAME_PATTERN, raw_bytes_dtype_name

if TYPE_CHECKING:
    from collections.abc import Callable


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
    if RAW_BYTES_NAME_PATTERN.fullmatch(dtype_name) is not None:
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
    if name is None or RAW_BYTES_NAME_PATTERN.fullmatch(name) is None:
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
# whole-document rules
# ---------------------------------------------------------------------------

ZARR_V3_ARRAY = "zarr_v3_array"
"""Document-type key under which this module's rules are registered."""

register_document_type(ZARR_V3_ARRAY, ARRAY_METADATA_STANDARD_KEYS_V3)

_data_type_spelling = document_rule(ZARR_V3_ARRAY, frozenset({"data_type"}))(
    _check_data_type_spelling
)
_fill_matches_dtype = document_rule(ZARR_V3_ARRAY, frozenset({"data_type", "fill_value"}))(
    _check_fill_matches_dtype
)


@document_rule(ZARR_V3_ARRAY, frozenset({"codecs"}))
def check_codec_pipeline_order(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """The pipeline shape: `array->array`* `array->bytes` `bytes->bytes`*."""
    entries = as_sequence(document["codecs"])
    if entries is None:
        return ()
    return pipeline_order_problems(entries, ("codecs",))


@document_rule(ZARR_V3_ARRAY, frozenset({"codecs"}))
def check_codec_shapes(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """Every known-name codec matches its canonical type."""
    entries = as_sequence(document["codecs"])
    if entries is None:
        return ()
    return shape_problems(entries, ("codecs",))


@document_rule(ZARR_V3_ARRAY, frozenset({"chunk_grid"}))
def check_chunk_grid_shape(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """A known-name chunk grid matches its canonical type."""
    found = validate_known_chunk_grid_metadata(document["chunk_grid"])
    # None is "not a known grid" (unjudged); () is "known and valid".
    if found is None:
        return ()
    return prefixed(("chunk_grid",), found)


@document_rule(ZARR_V3_ARRAY, frozenset({"shape", "dimension_names"}))
def check_dimension_names_length(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """One dimension name per array dimension."""
    shape = as_sequence(document["shape"])
    names = as_sequence(document["dimension_names"])
    if shape is None or names is None or len(names) == len(shape):
        return ()
    return (
        ValidationProblem(
            ("dimension_names",),
            f"dimension_names has {len(names)} entries but shape has {len(shape)} dimensions",
            "invalid_value",
        ),
    )


# Generic dispatchers: every rule an entity registers for itself runs here,
# so a new codec or chunk grid needs no edit to this module.
_dispatch_chunk_grid = document_rule(ZARR_V3_ARRAY, frozenset({"chunk_grid"}))(
    dispatch_field("chunk_grid")
)
_dispatch_codecs = document_rule(ZARR_V3_ARRAY, frozenset({"codecs"}))(
    dispatch_field_sequence("codecs")
)


def _rules() -> tuple[Rule, ...]:
    # Importing the entity package registers every entity's rules; done here
    # rather than at module import to keep the dependency one-directional.
    import zarr_metadata.rules._entities as entity_rules_package

    # Imported for its registrations; referenced so the import cannot be
    # pruned as unused by a checker or a well-meaning cleanup.
    assert entity_rules_package is not None
    return document_rules(ZARR_V3_ARRAY)


ZARR_V3_ARRAY_RULES: Final[tuple[Rule, ...]] = _rules()
"""The composition rule set for v3 array metadata documents.

Assembled from the registry rather than written out, so a rule cannot be
defined without joining the set it belongs to.
"""


__all__ = [
    "ZARR_V3_ARRAY",
    "ZARR_V3_ARRAY_RULES",
]
