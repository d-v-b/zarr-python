"""Validated construction of Zarr metadata documents.

`create_*` factories provide typed one-shot construction for every
document TypedDict. They normalize and validate at runtime.
`ZarrV3ArrayMetadataBuilder` supports incremental v3 array construction,
running applicable composition rules after each update and checking
completeness at `build()`.

Use `zarr_metadata.rules` to validate documents read from storage.
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
