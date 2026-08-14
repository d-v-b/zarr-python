"""Composition rules for v3 group metadata documents.

A group document's own fields carry no cross-field constraints, but the
inline consolidated-metadata convention embeds whole child documents —
and a composition-invalid child makes the consolidated view lie about
the store. The group rule set therefore recurses: every array entry is
judged by the v3 array rules, and every group entry (which may itself
carry consolidated metadata) by this rule set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from zarr_metadata.rules._engine import Rule, as_string_mapping, prefixed, run_rules
from zarr_metadata.rules._v3_array import ZARR_V3_ARRAY_RULES

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.model._validation import ValidationProblem


def _check_consolidated_entries(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """Consolidated child documents must satisfy their own composition rules.

    Structural validity of the consolidated envelope and its entries is
    the model layer's job; entries that are not interpretable as node
    documents decline in its favor.
    """
    consolidated = as_string_mapping(document["consolidated_metadata"])
    if consolidated is None:
        return ()
    metadata = as_string_mapping(consolidated.get("metadata"))
    if metadata is None:
        return ()
    problems: list[ValidationProblem] = []
    for path, entry in metadata.items():
        node = as_string_mapping(entry)
        if node is None:
            continue
        loc = ("consolidated_metadata", "metadata", path)
        node_type = node.get("node_type")
        if node_type == "array":
            problems.extend(prefixed(loc, run_rules(ZARR_V3_ARRAY_RULES, node)))
        elif node_type == "group":
            problems.extend(prefixed(loc, run_rules(ZARR_V3_GROUP_RULES, node)))
    return tuple(problems)


ZARR_V3_GROUP_RULES: Final[tuple[Rule, ...]] = (
    Rule(frozenset({"consolidated_metadata"}), _check_consolidated_entries),
)
"""The composition rule set for v3 group metadata documents."""


__all__ = [
    "ZARR_V3_GROUP_RULES",
]
