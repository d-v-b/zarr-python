"""Generative invariants connecting extension points to entity shapes."""

from __future__ import annotations

from hypothesis import assume, given
from hypothesis import strategies as st

from zarr_metadata.v3._extension_points import (
    DATA_TYPE,
    EXTENSION_POINTS,
    RAW_BYTES_FAMILY,
    canonical_name,
)
from zarr_metadata.v3._shape import modelled_entities, validate_known_entity_metadata

FIELDS = tuple(EXTENSION_POINTS)
NAMES = tuple(sorted({name for point in EXTENSION_POINTS.values() for name in point.identifiers}))


@given(field=st.sampled_from(FIELDS), name=st.sampled_from(NAMES))
def test_shape_ownership_is_keyed_by_extension_point(field: str, name: str) -> None:
    verdict = validate_known_entity_metadata(field, name)  # type: ignore[arg-type]
    assert (verdict is not None) == ((field, canonical_name(field, name)) in modelled_entities())


@given(width=st.integers(min_value=0, max_value=1_000_000))
def test_raw_data_type_family_canonicalizes_every_numeric_spelling(width: int) -> None:
    assert canonical_name(DATA_TYPE, f"r{width}") == RAW_BYTES_FAMILY


@given(
    field=st.sampled_from(tuple(field for field in FIELDS if field != DATA_TYPE)),
    width=st.integers(min_value=0, max_value=1_000_000),
)
def test_raw_like_names_are_identity_outside_data_types(field: str, width: int) -> None:
    name = f"r{width}"
    assert canonical_name(field, name) == name  # type: ignore[arg-type]


@given(field=st.sampled_from(FIELDS), name=st.text(min_size=1))
def test_unknown_names_have_no_shape_verdict(field: str, name: str) -> None:
    canonical = canonical_name(field, name)  # type: ignore[arg-type]
    assume((field, canonical) not in modelled_entities())
    assert validate_known_entity_metadata(field, name) is None  # type: ignore[arg-type]
