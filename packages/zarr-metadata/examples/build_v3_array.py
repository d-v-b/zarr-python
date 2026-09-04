"""Construct, serialize, and validate a Zarr v3 metadata hierarchy.

Run from ``packages/zarr-metadata`` with::

    uv run python examples/build_v3_array.py

The example uses only public APIs. Assertions make it useful as an executable
smoke test as well as a starting point for applications.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from zarr_metadata.builder import (
    ZarrV3ArrayMetadataBuilder,
    create_zarr_v3_consolidated_metadata_json,
    create_zarr_v3_group_metadata_json,
)
from zarr_metadata.model import MetadataValidationError
from zarr_metadata.rules import Valid, check_group_metadata_v3

if TYPE_CHECKING:
    from collections.abc import Callable


def expect_rejected(label: str, operation: Callable[[], object]) -> None:
    """Run one deliberately invalid operation and show why it was rejected."""
    try:
        operation()
    except MetadataValidationError as error:
        print(f"Rejected {label}: {error.problems[0]}")
    else:  # pragma: no cover - this script is also an executable assertion
        raise AssertionError(f"Expected {label} to be rejected")


def main() -> None:
    # Composition errors are rejected as soon as all fields involved are known.
    expect_rejected(
        "an out-of-range uint8 fill value",
        lambda: ZarrV3ArrayMetadataBuilder().with_fields(data_type="uint8", fill_value=300),
    )

    # Known extension names are checked against their canonical configuration.
    expect_rejected(
        "an invalid chunk-key separator",
        lambda: ZarrV3ArrayMetadataBuilder().with_fields(
            chunk_key_encoding={"name": "default", "configuration": {"separator": "!"}}
        ),
    )

    array = (
        ZarrV3ArrayMetadataBuilder()
        .with_fields(zarr_format=3, node_type="array", shape=(100, 200))
        .with_fields(data_type="uint16", fill_value=0)
        .with_fields(
            chunk_grid={"name": "regular", "configuration": {"chunk_shape": (10, 20)}},
            chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
            codecs=(
                {"name": "transpose", "configuration": {"order": (1, 0)}},
                {"name": "bytes", "configuration": {"endian": "little"}},
                "crc32c",
            ),
            dimension_names=("y", "x"),
            attributes={"units": "counts"},
        )
        .build()
    )

    # JSON round-tripping changes tuples into lists; the read-side API normalizes
    # them back before returning a typed, composition-valid document.
    loaded_array = json.loads(json.dumps(array))
    consolidated = create_zarr_v3_consolidated_metadata_json(
        kind="inline", must_understand=False, metadata={"measurements": loaded_array}
    )
    group = create_zarr_v3_group_metadata_json(
        zarr_format=3,
        node_type="group",
        attributes={"title": "Example hierarchy"},
        extensions={"consolidated_metadata": consolidated},
    )

    checked = check_group_metadata_v3(json.loads(json.dumps(group)))
    assert isinstance(checked, Valid)
    assert checked.document == group

    print(json.dumps(checked.document, indent=2))


if __name__ == "__main__":
    main()
