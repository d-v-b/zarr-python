"""Rules governing composition of Zarr metadata documents.

The package models metadata in layers with one contract each:

1. **Structure** (`zarr_metadata.model`) — is this JSON the right shape,
   element by element. Structural validators never interpret extension
   points.
2. **Composition** (this package) — do the elements agree with each
   other across the document: fill value vs. data type, codec pipeline
   kind ordering, dimension counts, known-name shapes. Rules are data
   (`Rule`), keyed by the document keys they read, so the same rule set
   judges complete documents and partially-assembled ones.
3. **Construction** (`zarr_metadata.builder`) — `create_*` factories and
   the incremental builder, conveniences that apply layers 1 and 2 while
   a document is being put together.

The `validate_*` / `is_*` / `parse_*` trios here mirror the model
layer's grammar with the same names and a stronger judgment: structure
*and* composition, every problem reported together. They are the front
door for readers — validating a loaded `zarr.json` — while the model
layer's trios remain the way to narrow types (only the structural layer
can honestly `TypeIs`; see `zarr_metadata.rules._documents`).

**Strictness stance.** This package models canonical documents and is
deliberately stricter than any given implementation: a fill value of
`5.0` for a `uint8` array is rejected here even though implementations
may coerce it on read. Implementations coerce ambiguous input as they
see fit *and then* validate the canonical result; disagreement in that
direction is the contract, not drift.

Extension openness: rules never reject what they cannot interpret.
Unknown names — codecs, chunk grids, data types this package has no
types for — pass untouched, while known names are held to their full
canonical shapes so a misspelling cannot masquerade as an extension.
"""

from zarr_metadata.rules._documents import (
    is_array_metadata_v2,
    is_array_metadata_v3,
    is_group_metadata_v2,
    is_group_metadata_v3,
    parse_array_metadata_v2,
    parse_array_metadata_v3,
    parse_group_metadata_v2,
    parse_group_metadata_v3,
    validate_array_metadata_v2,
    validate_array_metadata_v3,
    validate_group_metadata_v2,
    validate_group_metadata_v3,
)
from zarr_metadata.rules._engine import Rule, applicable, run_rules
from zarr_metadata.rules._v2_array import ZARR_V2_ARRAY_RULES
from zarr_metadata.rules._v3_array import ZARR_V3_ARRAY_RULES
from zarr_metadata.rules._v3_group import ZARR_V3_GROUP_RULES

__all__ = [
    "ZARR_V2_ARRAY_RULES",
    "ZARR_V3_ARRAY_RULES",
    "ZARR_V3_GROUP_RULES",
    "Rule",
    "applicable",
    "is_array_metadata_v2",
    "is_array_metadata_v3",
    "is_group_metadata_v2",
    "is_group_metadata_v3",
    "parse_array_metadata_v2",
    "parse_array_metadata_v3",
    "parse_group_metadata_v2",
    "parse_group_metadata_v3",
    "run_rules",
    "validate_array_metadata_v2",
    "validate_array_metadata_v3",
    "validate_group_metadata_v2",
    "validate_group_metadata_v3",
]
