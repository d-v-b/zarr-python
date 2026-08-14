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

Alongside them, `check_*` returns a discriminated `Valid[T] | Invalid`
for callers who want the document and its problems in one value, with
the type checker enforcing that they looked at `.valid` first.

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

**Unknown members of known entities** are the one place where openness is
a judgment call rather than a reading. The v3 spec constrains a
`configuration` only to "be an object" and never says whether it is
closed; `must_understand` is defined over *metadata document fields*, so
it cannot reach inside a configuration at all. Whether an extra member
invalidates a document has been an open question since 2023
(zarr-developers/zarr-specs#270, filed after a real interop break when
jzarr emitted `blosc.configuration.numThreads` and zarr-python refused
the array). This package takes the strict reading — matching most
registered extension schemas and the zarrs, tensorstore, and zarr-java
implementations — but softens its blast radius two ways: the problem
carries its own `unknown_key` kind, so a caller can choose tolerance, and
it never blocks the other rules about that entity, so a cosmetic extra
key cannot hide a real geometry error. Documents carrying such members
always round-trip byte-faithfully; the strict verdict is a judgment, not
a licence to drop data.
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
from zarr_metadata.rules._engine import Rule, RuleCheck, applicable, inapplicable, run_rules
from zarr_metadata.rules._registry import EntityRule, entity_rules, registered_entities
from zarr_metadata.rules._result import (
    Invalid,
    Valid,
    ValidationResult,
    check_array_metadata_v2,
    check_array_metadata_v3,
    check_group_metadata_v2,
    check_group_metadata_v3,
)
from zarr_metadata.rules._v2_array import ZARR_V2_ARRAY, ZARR_V2_ARRAY_RULES
from zarr_metadata.rules._v3_array import ZARR_V3_ARRAY, ZARR_V3_ARRAY_RULES
from zarr_metadata.rules._v3_group import ZARR_V3_GROUP, ZARR_V3_GROUP_RULES

__all__ = [
    "ZARR_V2_ARRAY",
    "ZARR_V2_ARRAY_RULES",
    "ZARR_V3_ARRAY",
    "ZARR_V3_ARRAY_RULES",
    "ZARR_V3_GROUP",
    "ZARR_V3_GROUP_RULES",
    "EntityRule",
    "Invalid",
    "Rule",
    "RuleCheck",
    "Valid",
    "ValidationResult",
    "applicable",
    "check_array_metadata_v2",
    "check_array_metadata_v3",
    "check_group_metadata_v2",
    "check_group_metadata_v3",
    "entity_rules",
    "inapplicable",
    "is_array_metadata_v2",
    "is_array_metadata_v3",
    "is_group_metadata_v2",
    "is_group_metadata_v3",
    "parse_array_metadata_v2",
    "parse_array_metadata_v3",
    "parse_group_metadata_v2",
    "parse_group_metadata_v3",
    "registered_entities",
    "run_rules",
    "validate_array_metadata_v2",
    "validate_array_metadata_v3",
    "validate_group_metadata_v2",
    "validate_group_metadata_v3",
]
