"""Tests for rule registration.

The registry exists so that a rule cannot be defined without being run.
These tests cover the three ways that could still fail: a rule declaring
dependencies no such document has, an entity whose rules were never
imported, and a document rule set assembled from something other than
the registry.
"""

from __future__ import annotations

import pkgutil

import pytest

import zarr_metadata.rules._entities as entities
from zarr_metadata.rules import (
    ZARR_V2_ARRAY_RULES,
    ZARR_V3_ARRAY_RULES,
    ZARR_V3_GROUP_RULES,
    applicable,
    inapplicable,
)
from zarr_metadata.rules._registry import (
    document_rule,
    entity_rule,
    register_document_type,
    registered_entities,
)
from zarr_metadata.rules._v3_array import ZARR_V3_ARRAY
from zarr_metadata.v3._extension_points import (
    CHUNK_GRID,
    CODECS,
    DATA_TYPE,
    EXTENSION_POINTS,
    Provenance,
    identifiers_with,
)
from zarr_metadata.v3._shape import modelled_entities

# Entities the package models but that carry no composition rules: their
# canonical shape is the whole of what we can say about them. Listed by
# hand, keyed by extension point, so that adding a codec is a deliberate
# choice between "write rules" and "record that there are none", never a
# silent omission.
_RULE_FREE = frozenset(
    {
        (CODECS, "blosc"),
        (CODECS, "bytes"),
        (CODECS, "cast_value"),
        (CODECS, "crc32c"),
        (CODECS, "gzip"),
        (CODECS, "scale_offset"),
        (CODECS, "zstd"),
    }
)


def test_every_shape_modelled_entity_is_accounted_for() -> None:
    # Shape validators exist for codecs and chunk grids; every one either
    # carries rules or is recorded as deliberately rule-free.
    named = {entity for _, entity in registered_entities() | _RULE_FREE}
    assert modelled_entities() == named


def test_rule_free_entities_really_have_no_rules() -> None:
    # Guards the exclusion list itself: an entity cannot be listed as
    # rule-free while quietly carrying rules.
    assert registered_entities() & _RULE_FREE == frozenset()


def test_rules_are_keyed_by_extension_point_not_name() -> None:
    # `bytes` is a core codec and a registered extension data type; a rule
    # for one must never fire on the other, so the key carries the field.
    assert identifiers_with(CODECS, Provenance.CORE) >= {"bytes"}
    assert identifiers_with(DATA_TYPE, Provenance.REGISTERED) >= {"bytes"}
    assert all(field in EXTENSION_POINTS for field, _ in registered_entities())


def test_every_entity_module_is_imported() -> None:
    # The package auto-imports its modules; this asserts the discovery
    # actually ran, so a new module cannot sit unimported and inert.
    module_names = {info.name for info in pkgutil.iter_modules(entities.__path__)}
    assert len(module_names) != 0
    for name in module_names:
        assert f"{entities.__name__}.{name}" in __import__("sys").modules


@pytest.mark.parametrize("rules", [ZARR_V3_ARRAY_RULES, ZARR_V2_ARRAY_RULES, ZARR_V3_GROUP_RULES])
def test_rule_sets_are_non_empty(rules: tuple[object, ...]) -> None:
    assert len(rules) != 0


def test_error_document_rule_requiring_an_unknown_key() -> None:
    # A rule whose dependency is misspelled can never fire, and a rule
    # that never fires is indistinguishable from one that always passes.
    with pytest.raises(ValueError, match="could never fire"):

        @document_rule(ZARR_V3_ARRAY, frozenset({"shapee"}))
        def _misspelled(document: object) -> tuple[()]:  # pragma: no cover - never runs
            return ()


def test_error_entity_rule_requiring_an_unknown_key() -> None:
    with pytest.raises(ValueError, match="could never fire"):

        @entity_rule(ZARR_V3_ARRAY, CHUNK_GRID, "regular", requires=frozenset({"shapee"}))
        def _misspelled(configuration: object, document: object) -> tuple[()]:  # pragma: no cover
            return ()


def test_error_entity_rule_for_an_unmodelled_entity() -> None:
    # Entity rules read configuration members by name, so a rule for an
    # entity with no shape validator could never fire.
    with pytest.raises(ValueError, match="no shape validator"):

        @entity_rule(ZARR_V3_ARRAY, CHUNK_GRID, "hilbert")
        def _unmodelled(configuration: object, document: object) -> tuple[()]:  # pragma: no cover
            return ()


def test_error_rule_for_an_unregistered_document_type() -> None:
    with pytest.raises(LookupError, match="unknown document type"):

        @document_rule("zarr_v9_array", frozenset())
        def _orphan(document: object) -> tuple[()]:  # pragma: no cover - never runs
            return ()


def test_register_document_type_accepts_declared_extension_keys() -> None:
    register_document_type("test_doc", frozenset({"a"}), extension_keys=frozenset({"b"}))

    @document_rule("test_doc", frozenset({"a", "b"}))
    def _uses_both(document: object) -> tuple[()]:
        return ()

    assert _uses_both.requires == frozenset({"a", "b"})


def test_applicable_and_inapplicable_partition_the_rule_set() -> None:
    present = frozenset({"data_type", "fill_value"})
    fired = list(applicable(ZARR_V3_ARRAY_RULES, present))
    skipped = list(inapplicable(ZARR_V3_ARRAY_RULES, present))
    assert len(fired) + len(skipped) == len(ZARR_V3_ARRAY_RULES)
    assert all(rule.requires <= present for rule in fired)
    assert all(not rule.requires <= present for rule in skipped)
    # A rule that did not fire is reportable, not silently absent: this is
    # what lets a caller distinguish "passed" from "never ran".
    assert len(skipped) != 0
