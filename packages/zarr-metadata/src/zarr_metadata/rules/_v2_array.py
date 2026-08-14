"""Composition rules for v2 array metadata documents.

The v2 rule set is deliberately small today: the one cross-field
constraint the package interprets is that `chunks` and `shape` agree on
dimensionality. Fill-value/dtype consistency for v2 (NumPy dtype strings,
base64 fills for bytes dtypes) is a known follow-up, tracked in the
package docs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._engine import Rule, as_sequence

if TYPE_CHECKING:
    from collections.abc import Mapping


def _check_chunks_match_shape(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """`chunks` must have one entry per dimension of `shape`."""
    shape = as_sequence(document["shape"])
    chunks = as_sequence(document["chunks"])
    if shape is None or chunks is None or len(shape) == len(chunks):
        return ()
    return (
        ValidationProblem(
            ("chunks",),
            "expected the same number of dimensions as shape",
            "invalid_value",
        ),
    )


ZARR_V2_ARRAY_RULES: Final[tuple[Rule, ...]] = (
    Rule(frozenset({"shape", "chunks"}), _check_chunks_match_shape),
)
"""The composition rule set for v2 array metadata documents."""


__all__ = [
    "ZARR_V2_ARRAY_RULES",
]
