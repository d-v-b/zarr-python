"""Validated construction of Zarr metadata documents.

Two construction surfaces over the plain JSON shapes defined in
`zarr_metadata.v2` / `zarr_metadata.v3`:

- **`create_*` factories** — one per public document TypedDict, taking
  `**kwargs: Unpack[<TypedDict>]`. One-shot construction: required keys
  are enforced *statically* at the call site, and the runtime pass
  normalizes, checks structure, and runs the semantic rules, raising one
  `MetadataValidationError` with every problem. Prefer these when all
  fields are known at a single call site — which is the common case.
- **`ZarrV3ArrayMetadataBuilder`** — incremental accumulation for staged
  assembly across program points. Semantic rules fire eagerly as fields
  land, whenever their dependencies are all present, so field order is
  unconstrained, problems surface at the `evolve` call that completes
  them, and a cross-call conflict names both fields involved. Completeness
  can only be checked at `build()` time, at runtime — the price of
  accumulating through a partial type.

Semantic rules (`zarr_metadata.builder._rules`) are shared by both
surfaces and are public via `run_rules`, so staged assembly over plain
`*Partial` dicts can also validate at its own checkpoints. This layer is
the semantic complement to `zarr_metadata.model`, whose validators
deliberately check structure only.
"""

from zarr_metadata.builder._array_v3 import ZarrV3ArrayMetadataBuilder
from zarr_metadata.builder._create import (
    DOCUMENT_FACTORIES,
    create_zarr_v2_array_metadata_json,
    create_zarr_v2_consolidated_metadata_json,
    create_zarr_v2_group_metadata_json,
    create_zarr_v2_z_array_json,
    create_zarr_v2_z_group_json,
    create_zarr_v3_array_metadata_json,
    create_zarr_v3_consolidated_metadata_json,
    create_zarr_v3_group_metadata_json,
)
from zarr_metadata.builder._rules import ZARR_V3_ARRAY_RULES, Rule, applicable, run_rules

__all__ = [
    "DOCUMENT_FACTORIES",
    "ZARR_V3_ARRAY_RULES",
    "Rule",
    "ZarrV3ArrayMetadataBuilder",
    "applicable",
    "create_zarr_v2_array_metadata_json",
    "create_zarr_v2_consolidated_metadata_json",
    "create_zarr_v2_group_metadata_json",
    "create_zarr_v2_z_array_json",
    "create_zarr_v2_z_group_json",
    "create_zarr_v3_array_metadata_json",
    "create_zarr_v3_consolidated_metadata_json",
    "create_zarr_v3_group_metadata_json",
    "run_rules",
]
