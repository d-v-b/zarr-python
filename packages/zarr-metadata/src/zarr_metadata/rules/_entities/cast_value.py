"""Spec transition for the `cast_value` codec.

`cast_value` carries no composition rules of its own today, but it
transforms the array spec in two ways, of which this package models one.

**Data type** — compositional, and modelled: everything downstream
receives the configured target `data_type`, so a later rule that reads
the type (the `bytes` codec's endianness requirement is the canonical
example) judges against the right one.

**Fill value** — a value transformation, and deliberately *not*
modelled. The spec requires that "the fill value of the output array
MUST be cast to the target data type using the same casting semantics as
elements", and that "if the fill value cannot survive a round-trip cast,
implementations MUST treat this as an error". Deciding either means
implementing the cast — rounding modes, out-of-range clamp and wrap,
scalar maps — which is numeric semantics rather than JSON judgment and
has no bounded partial version. This package models structure and
composition, both decidable from JSON; the cast belongs to whatever
implements the codec. So the outgoing fill value is `UNKNOWN`: downstream
rules that read it decline, and the round-trip requirement is recorded
here as downstream's obligation rather than silently unenforced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata.rules._spec import UNKNOWN, ArraySpec, spec_transition
from zarr_metadata.v3.codec.cast_value import CAST_VALUE_CODEC_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON


@spec_transition(CAST_VALUE_CODEC_NAME)
def cast_data_type(configuration: Mapping[str, object], incoming: ArraySpec) -> ArraySpec:
    """The outgoing type is the configured target; the fill value is not ours to compute."""
    target = cast("ZarrV3MetadataFieldJSON", configuration["data_type"])
    return incoming.with_data_type(target).with_fill_value(UNKNOWN)
