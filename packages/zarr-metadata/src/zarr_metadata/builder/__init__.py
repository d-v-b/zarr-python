"""Incremental, validated construction of Zarr metadata documents.

Builders accumulate the plain JSON shapes defined in `zarr_metadata.v2` /
`zarr_metadata.v3` — a builder's state IS a partial metadata document, not
a graph of wrapper objects. Semantic rules (`zarr_metadata.builder._rules`)
fire eagerly as fields land, whenever their dependencies are all present,
so field order is unconstrained and problems surface at the `evolve` call
that completes them. The same rules run over full documents at `build`.

This layer is the semantic complement to `zarr_metadata.model`, whose
validators deliberately check structure only.
"""

from zarr_metadata.builder._array_v3 import ZarrV3ArrayMetadataBuilder
from zarr_metadata.builder._rules import ZARR_V3_ARRAY_RULES, Rule, applicable, run_rules

__all__ = [
    "ZARR_V3_ARRAY_RULES",
    "Rule",
    "ZarrV3ArrayMetadataBuilder",
    "applicable",
    "run_rules",
]
