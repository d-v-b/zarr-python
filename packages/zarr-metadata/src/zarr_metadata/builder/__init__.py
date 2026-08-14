"""Validated construction of Zarr metadata documents.

The construction layer is a consumer of the package's other two layers:
the model layer checks structure, `zarr_metadata.rules` judges
composition, and the surfaces here apply both while a document is being
put together.

Two construction surfaces over the plain JSON shapes defined in
`zarr_metadata.v2` / `zarr_metadata.v3`:

- **`create_*` factories** — one per public document TypedDict, taking
  `**kwargs: Unpack[<TypedDict>]`. One-shot construction: at
  literal-keyword call sites, required keys and value types are enforced
  statically; the runtime pass normalizes, checks structure, and runs
  the composition rules, raising one `MetadataValidationError` with
  every problem. Prefer these when all fields are known at a single call
  site — which is the common case. For validating documents you *read*
  rather than construct, use the trios in `zarr_metadata.rules`.
- **`ZarrV3ArrayMetadataBuilder`** — incremental accumulation for staged
  assembly across program points. Composition rules fire eagerly as
  fields land, whenever their dependencies are all present, so field
  order is unconstrained, problems surface at the `evolve` call that
  completes them, and a cross-call conflict names both fields involved.
  Completeness can only be checked at `build()` time, at runtime — the
  price of accumulating through a partial type. Incremental building is
  currently implemented for v3 arrays only, the document type with the
  richest cross-field coupling.
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

__all__ = [
    "DOCUMENT_FACTORIES",
    "ZarrV3ArrayMetadataBuilder",
    "create_zarr_v2_array_metadata_json",
    "create_zarr_v2_consolidated_metadata_json",
    "create_zarr_v2_group_metadata_json",
    "create_zarr_v2_z_array_json",
    "create_zarr_v2_z_group_json",
    "create_zarr_v3_array_metadata_json",
    "create_zarr_v3_consolidated_metadata_json",
    "create_zarr_v3_group_metadata_json",
]
