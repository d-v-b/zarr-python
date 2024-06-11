from __future__ import annotations
from typing import TYPE_CHECKING

from zarr.abc.metadata import Metadata
if TYPE_CHECKING:
    from typing_extensions import Self
    import numpy.typing as npt
    from zarr.abc.codec import Codec

from zarr.abc.codec import CodecPipeline
from zarr.array_spec import ArraySpec
from zarr.buffer import Buffer, BufferPrototype
from zarr.chunk_key_encodings import parse_separator
from zarr.codecs._v2 import V2Compressor, V2Filters
from zarr.common import JSON, ZARRAY_JSON, ZATTRS_JSON, ChunkCoords, parse_dtype, parse_fill_value, parse_shapelike
from zarr.config import parse_indexing_order
from zarr.metadata import ArrayMetadata, jsonify_dtype

import numpy as np
import json
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from zarr.metadata.v3 import parse_attributes


def parse_zarr_format(data: Any) -> Literal[2]:
    if data == 2:
        return 2
    raise ValueError(f"Invalid value. Expected 2. Got {data}.")


def parse_filters(data: Any) -> list[dict[str, JSON]] | None:
    return data


def parse_compressor(data: Any) -> dict[str, JSON] | None:
    return data


def parse_metadata(data: ArrayMetadata) -> ArrayMetadata:
    """
    Perform validation of an entire ArrayMetadata instance, raising exceptions if there
    are any problems with the metadata. Returns valid metadata.
    """
    if (l_chunks := len(data.chunks)) != (l_shape := len(data.shape)):
        msg = (
            f"The `shape` and `chunks` attributes must have the same length. "
            f"`chunks` has length {l_chunks}, but `shape` has length {l_shape}."
        )
        raise ValueError(msg)
    return data


@dataclass(frozen=True, kw_only=True)
class ArrayMetadata(Metadata):
    shape: ChunkCoords
    chunks: tuple[int, ...]
    data_type: np.dtype[Any]
    fill_value: None | int | float = 0
    order: Literal["C", "F"] = "C"
    filters: list[dict[str, JSON]] | None = None
    dimension_separator: Literal[".", "/"] = "."
    compressor: dict[str, JSON] | None = None
    attributes: dict[str, JSON] = field(default_factory=dict)
    zarr_format: Literal[2] = field(init=False, default=2)

    def __init__(
        self,
        *,
        shape: ChunkCoords,
        dtype: npt.DTypeLike,
        chunks: ChunkCoords,
        fill_value: Any,
        order: Literal["C", "F"],
        dimension_separator: Literal[".", "/"] = ".",
        compressor: dict[str, JSON] | None = None,
        filters: list[dict[str, JSON]] | None = None,
        attributes: dict[str, JSON] | None = None,
    ):
        """
        Metadata for a Zarr version 2 array.
        """
        shape_parsed = parse_shapelike(shape)
        data_type_parsed = parse_dtype(dtype)
        chunks_parsed = parse_shapelike(chunks)
        compressor_parsed = parse_compressor(compressor)
        order_parsed = parse_indexing_order(order)
        dimension_separator_parsed = parse_separator(dimension_separator)
        filters_parsed = parse_filters(filters)
        fill_value_parsed = parse_fill_value(fill_value)
        attributes_parsed = parse_attributes(attributes)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "dtype", data_type_parsed)
        object.__setattr__(self, "chunks", chunks_parsed)
        object.__setattr__(self, "compressor", compressor_parsed)
        object.__setattr__(self, "order", order_parsed)
        object.__setattr__(self, "dimension_separator", dimension_separator_parsed)
        object.__setattr__(self, "filters", filters_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "attributes", attributes_parsed)

        # ensure that the metadata document is consistent
        _ = parse_metadata(self)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data_type

    @property
    def codec_pipeline(self) -> CodecPipeline:
        from zarr.codecs import BatchedCodecPipeline

        return BatchedCodecPipeline.from_list(
            [V2Filters(self.filters or []), V2Compressor(self.compressor)]
        )

    def to_buffer_dict(self) -> dict[str, Buffer]:
        zarray_dict = self.to_dict()
        zattrs_dict = zarray_dict.pop("attributes", {})
        return {
            ZARRAY_JSON: Buffer.from_bytes(json.dumps(zarray_dict, default=jsonify_dtype).encode()),
            ZATTRS_JSON: Buffer.from_bytes(json.dumps(zattrs_dict).encode()),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArrayMetadata:
        # check that the zarr_format attribute is correct
        _ = parse_zarr_format(data.pop("zarr_format"))
        return cls(**data)

    def get_chunk_spec(
        self, _chunk_coords: ChunkCoords, order: Literal["C", "F"], prototype: BufferPrototype
    ) -> ArraySpec:
        return ArraySpec(
            shape=self.chunk_grid.chunk_shape,
            dtype=self.dtype,
            fill_value=self.fill_value,
            order=order,
            prototype=prototype,
        )

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.dimension_separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier

    def update_shape(self, shape: ChunkCoords) -> Self:
        return replace(self, shape=shape)

    def update_attributes(self, attributes: dict[str, JSON]) -> Self:
        return replace(self, attributes=attributes)


