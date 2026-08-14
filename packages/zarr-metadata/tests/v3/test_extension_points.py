"""Tests for the extension-point table and name canonicalization."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import zarr_metadata.v3.chunk_grid
import zarr_metadata.v3.chunk_key_encoding
import zarr_metadata.v3.codec
import zarr_metadata.v3.data_type
from zarr_metadata.v3._extension_points import (
    CHUNK_GRID,
    CHUNK_KEY_ENCODING,
    CODECS,
    DATA_TYPE,
    EXTENSION_POINTS,
    RAW_BYTES_FAMILY,
    STORAGE_TRANSFORMERS,
    Provenance,
    canonical_name,
    identifier_of,
    identifiers_with,
    provenance_of,
)

# (field, name, expected canonical key) — identity everywhere except the
# parameterized raw-bytes family.
CANONICAL_CASES: dict[str, tuple[str, str, str]] = {
    "plain-dtype": (DATA_TYPE, "uint8", "uint8"),
    "dotted-dtype": (DATA_TYPE, "numpy.datetime64", "numpy.datetime64"),
    "raw-8": (DATA_TYPE, "r8", RAW_BYTES_FAMILY),
    "raw-24": (DATA_TYPE, "r24", RAW_BYTES_FAMILY),
    # Malformed members canonicalize into the family too: a misspelling of
    # something we model must be reported as such, not pass as an unknown
    # third-party extension.
    "raw-not-multiple-of-8": (DATA_TYPE, "r12", RAW_BYTES_FAMILY),
    "raw-zero": (DATA_TYPE, "r0", RAW_BYTES_FAMILY),
    # Canonicalization is field-aware: the r<N> family is a data type.
    "raw-shaped-codec-name": (CODECS, "r8", "r8"),
    "codec": (CODECS, "blosc", "blosc"),
    "unknown": (CODECS, "zfpy", "zfpy"),
}


@pytest.mark.parametrize(
    ("field", "name", "expected"), CANONICAL_CASES.values(), ids=list(CANONICAL_CASES)
)
def test_canonical_name(field: str, name: str, expected: str) -> None:
    assert canonical_name(field, name) == expected  # type: ignore[arg-type]


def test_name_collision_resolves_per_extension_point() -> None:
    # `bytes` is a core codec and, separately, a registered extension data
    # type. This is the case a name-keyed table could not represent.
    assert provenance_of(CODECS, "bytes") is Provenance.CORE
    assert provenance_of(DATA_TYPE, "bytes") is Provenance.REGISTERED
    assert identifier_of(CODECS, "bytes").reference != identifier_of(DATA_TYPE, "bytes").reference


def test_raw_family_members_share_one_entry() -> None:
    entry = identifier_of(DATA_TYPE, "r8")
    assert entry is identifier_of(DATA_TYPE, "r4096")
    assert entry.provenance is Provenance.CORE
    assert entry.name == RAW_BYTES_FAMILY


def test_unmodelled_names_are_not_errors() -> None:
    # An unregistered or newer extension is simply unknown, not invalid.
    assert provenance_of(CODECS, "zfpy") is None
    assert identifier_of(DATA_TYPE, "float8_e4m3") is None


def test_zstd_is_marked_proposed() -> None:
    # Its specification is an open pull request, not merged text.
    entry = identifier_of(CODECS, "zstd")
    assert entry.provenance is Provenance.PROPOSED
    assert "pull" in entry.reference


def test_must_understand_policy_matches_the_spec() -> None:
    # The spec excludes must_understand: false at exactly these three
    # points; an implementation cannot proceed without understanding them.
    for field in (DATA_TYPE, CHUNK_GRID, CHUNK_KEY_ENCODING):
        assert EXTENSION_POINTS[field].must_understand_false_permitted is False
    for field in (CODECS, STORAGE_TRANSFORMERS):
        assert EXTENSION_POINTS[field].must_understand_false_permitted is True


def test_sequence_valued_points() -> None:
    assert EXTENSION_POINTS[CODECS].holds_sequence is True
    assert EXTENSION_POINTS[STORAGE_TRANSFORMERS].holds_sequence is True
    assert EXTENSION_POINTS[DATA_TYPE].holds_sequence is False


def test_storage_transformers_is_recorded_as_empty() -> None:
    # A real extension point this package models no identifiers for; its
    # emptiness is a stated fact, not an oversight.
    assert EXTENSION_POINTS[STORAGE_TRANSFORMERS].identifiers == {}


def _module_names(package: object, suffix: str) -> set[str]:
    found: set[str] = set()
    for info in pkgutil.iter_modules(package.__path__):  # type: ignore[attr-defined]
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package.__name__}.{info.name}")  # type: ignore[attr-defined]
        found.update(
            value
            for attribute, value in vars(module).items()
            if attribute.endswith(suffix) and isinstance(value, str)
        )
    return found


# The table is hand-written because provenance is irreducible knowledge —
# nothing in the type modules records where a name was standardized. These
# drift tests tie it to the names those modules actually define, so a new
# codec or data type cannot ship without a provenance entry.
DRIFT_CASES: dict[str, tuple[str, object, str, set[str]]] = {
    "codecs": (CODECS, zarr_metadata.v3.codec, "_CODEC_NAME", set()),
    "chunk_grid": (CHUNK_GRID, zarr_metadata.v3.chunk_grid, "_CHUNK_GRID_NAME", set()),
    "chunk_key_encoding": (
        CHUNK_KEY_ENCODING,
        zarr_metadata.v3.chunk_key_encoding,
        "_CHUNK_KEY_ENCODING_NAME",
        set(),
    ),
    # `raw` defines a grammar, not a name constant, so the family key has
    # no counterpart to scan for.
    "data_type": (DATA_TYPE, zarr_metadata.v3.data_type, "_DATA_TYPE_NAME", {RAW_BYTES_FAMILY}),
}


@pytest.mark.parametrize(
    ("field", "package", "suffix", "extra"), DRIFT_CASES.values(), ids=list(DRIFT_CASES)
)
def test_table_matches_the_modules(
    field: str, package: object, suffix: str, extra: set[str]
) -> None:
    tabled = set(EXTENSION_POINTS[field].identifiers)  # type: ignore[index]
    assert tabled == _module_names(package, suffix) | extra


def test_every_identifier_cites_a_reference() -> None:
    for point in EXTENSION_POINTS.values():
        for identifier in point.identifiers.values():
            assert identifier.reference.startswith("https://"), identifier


def test_identifiers_with_partitions_a_point() -> None:
    point = EXTENSION_POINTS[CODECS]
    by_provenance = {
        name for provenance in Provenance for name in identifiers_with(CODECS, provenance)
    }
    assert by_provenance == set(point.identifiers)
