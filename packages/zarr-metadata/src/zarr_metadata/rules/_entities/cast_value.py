"""Spec transition for the `cast_value` codec.

`cast_value` carries no composition rules of its own today, but it
transforms the array spec: everything after it in the chain receives its
target `data_type`, not the document's. Without this transition a
downstream rule that reads the data type — the `bytes` codec's endianness
requirement is the canonical example — would judge against the wrong
type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata.rules._spec import ArraySpec, spec_transition
from zarr_metadata.v3.codec.cast_value import CAST_VALUE_CODEC_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON


@spec_transition(CAST_VALUE_CODEC_NAME)
def cast_data_type(configuration: Mapping[str, object], incoming: ArraySpec) -> ArraySpec:
    """The outgoing data type is the configured target; the shape is unchanged."""
    return incoming.with_data_type(cast("ZarrV3MetadataFieldJSON", configuration["data_type"]))
