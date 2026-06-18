from importlib.metadata import version

from zarr_metadata._common import JSONValue, NamedConfig
from zarr_metadata.v2.array import (
    ArrayDimensionSeparatorV2,
    ArrayMetadataV2,
    ArrayMetadataV2Partial,
    ArrayOrderV2,
    DataTypeMetadataV2,
    ZArrayMetadata,
)
from zarr_metadata.v2.attributes import ZAttrsMetadata
from zarr_metadata.v2.codec import CodecMetadataV2
from zarr_metadata.v2.consolidated import ConsolidatedMetadataV2
from zarr_metadata.v2.group import GroupMetadataV2, GroupMetadataV2Partial, ZGroupMetadata
from zarr_metadata.v3._common import MetadataFieldV3
from zarr_metadata.v3.array import ArrayMetadataV3, ArrayMetadataV3Partial, ExtensionFieldV3
from zarr_metadata.v3.codec.blosc import BLOSC_CNAME, BLOSC_SHUFFLE, BloscCName, BloscShuffle
from zarr_metadata.v3.codec.bytes import ENDIAN, Endian
from zarr_metadata.v3.codec.sharding_indexed import INDEX_LOCATION, IndexLocation
from zarr_metadata.v3.consolidated import ConsolidatedMetadataV3
from zarr_metadata.v3.data_type.numpy_timedelta64 import DATETIME_UNIT, DateTimeUnit
from zarr_metadata.v3.group import GroupMetadataV3, GroupMetadataV3Partial

__version__ = version("zarr-metadata")


__all__ = [
    "BLOSC_CNAME",
    "BLOSC_SHUFFLE",
    "DATETIME_UNIT",
    "ENDIAN",
    "INDEX_LOCATION",
    "ArrayDimensionSeparatorV2",
    "ArrayMetadataV2",
    "ArrayMetadataV2Partial",
    "ArrayMetadataV3",
    "ArrayMetadataV3Partial",
    "ArrayOrderV2",
    "BloscCName",
    "BloscShuffle",
    "CodecMetadataV2",
    "ConsolidatedMetadataV2",
    "ConsolidatedMetadataV3",
    "DataTypeMetadataV2",
    "DateTimeUnit",
    "Endian",
    "ExtensionFieldV3",
    "GroupMetadataV2",
    "GroupMetadataV2Partial",
    "GroupMetadataV3",
    "GroupMetadataV3Partial",
    "IndexLocation",
    "JSONValue",
    "MetadataFieldV3",
    "NamedConfig",
    "ZArrayMetadata",
    "ZAttrsMetadata",
    "ZGroupMetadata",
    "__version__",
]
