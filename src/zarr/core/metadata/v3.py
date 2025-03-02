from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from zarr.abc.metadata import Metadata
from zarr.core.buffer.core import default_buffer_prototype

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.common import JSON, ChunkCoords
    from zarr.core.metadata.dtype import (
        DTypeWrapper,
    )

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal

import numcodecs.abc
import numpy as np

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec, Codec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.chunk_grids import ChunkGrid, RegularChunkGrid
from zarr.core.chunk_key_encodings import ChunkKeyEncoding, ChunkKeyEncodingLike
from zarr.core.common import (
    JSON,
    ZARR_JSON,
    ChunkCoords,
    parse_named_configuration,
    parse_shapelike,
)
from zarr.core.config import config
from zarr.core.metadata.common import parse_attributes
from zarr.errors import MetadataValidationError, NodeTypeValidationError
from zarr.registry import get_codec_class, get_data_type_by_name, get_data_type_from_dict

DEFAULT_DTYPE = "float64"

# Keep in sync with _replace_special_floats
SPECIAL_FLOATS_ENCODED = {
    "Infinity": np.inf,
    "-Infinity": -np.inf,
    "NaN": np.nan,
}


def parse_zarr_format(data: object) -> Literal[3]:
    if data == 3:
        return 3
    raise MetadataValidationError("zarr_format", 3, data)


def parse_node_type_array(data: object) -> Literal["array"]:
    if data == "array":
        return "array"
    raise NodeTypeValidationError("node_type", "array", data)


def parse_codecs(data: object) -> tuple[Codec, ...]:
    out: tuple[Codec, ...] = ()

    if not isinstance(data, Iterable):
        raise TypeError(f"Expected iterable, got {type(data)}")

    for c in data:
        if isinstance(
            c, ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec
        ):  # Can't use Codec here because of mypy limitation
            out += (c,)
        else:
            name_parsed, _ = parse_named_configuration(c, require_configuration=False)
            out += (get_codec_class(name_parsed).from_dict(c),)

    return out


def validate_array_bytes_codec(codecs: tuple[Codec, ...]) -> ArrayBytesCodec:
    # ensure that we have at least one ArrayBytesCodec
    abcs: list[ArrayBytesCodec] = [codec for codec in codecs if isinstance(codec, ArrayBytesCodec)]
    if len(abcs) == 0:
        raise ValueError("At least one ArrayBytesCodec is required.")
    elif len(abcs) > 1:
        raise ValueError("Only one ArrayBytesCodec is allowed.")

    return abcs[0]


def validate_codecs(codecs: tuple[Codec, ...], dtype: DTypeWrapper) -> None:
    """Check that the codecs are valid for the given dtype"""
    from zarr.codecs.sharding import ShardingCodec

    abc = validate_array_bytes_codec(codecs)

    # Recursively resolve array-bytes codecs within sharding codecs
    while isinstance(abc, ShardingCodec):
        abc = validate_array_bytes_codec(abc.codecs)

    # we need to have special codecs if we are decoding vlen strings or bytestrings
    # TODO: use codec ID instead of class name
    codec_class_name = abc.__class__.__name__
    if dtype.kind == "string" and not codec_class_name == "VLenUTF8Codec":
        raise ValueError(
            f"For string dtype, ArrayBytesCodec must be `VLenUTF8Codec`, got `{codec_class_name}`."
        )
    if dtype.kind == "bytes" and not codec_class_name == "VLenBytesCodec":
        raise ValueError(
            f"For bytes dtype, ArrayBytesCodec must be `VLenBytesCodec`, got `{codec_class_name}`."
        )


def parse_dimension_names(data: object) -> tuple[str | None, ...] | None:
    if data is None:
        return data
    elif isinstance(data, Iterable) and all(isinstance(x, type(None) | str) for x in data):
        return tuple(data)
    else:
        msg = f"Expected either None or a iterable of str, got {type(data)}"
        raise TypeError(msg)


def parse_storage_transformers(data: object) -> tuple[dict[str, JSON], ...]:
    """
    Parse storage_transformers. Zarr python cannot use storage transformers
    at this time, so this function doesn't attempt to validate them.
    """
    if data is None:
        return ()
    if isinstance(data, Iterable):
        if len(tuple(data)) >= 1:
            return data  # type: ignore[return-value]
        else:
            return ()
    raise TypeError(
        f"Invalid storage_transformers. Expected an iterable of dicts. Got {type(data)} instead."
    )


class V3JsonEncoder(json.JSONEncoder):
    def __init__(
        self,
        *,
        skipkeys: bool = False,
        ensure_ascii: bool = True,
        check_circular: bool = True,
        allow_nan: bool = True,
        sort_keys: bool = False,
        indent: int | None = None,
        separators: tuple[str, str] | None = None,
        default: Callable[[object], object] | None = None,
    ) -> None:
        if indent is None:
            indent = config.get("json_indent")
        super().__init__(
            skipkeys=skipkeys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            sort_keys=sort_keys,
            indent=indent,
            separators=separators,
            default=default,
        )

    def default(self, o: object) -> Any:
        if isinstance(o, np.dtype):
            return str(o)
        if np.isscalar(o):
            out: Any
            if hasattr(o, "dtype") and o.dtype.kind == "M" and hasattr(o, "view"):
                # https://github.com/zarr-developers/zarr-python/issues/2119
                # `.item()` on a datetime type might or might not return an
                # integer, depending on the value.
                # Explicitly cast to an int first, and then grab .item()
                out = o.view("i8").item()
            else:
                # convert numpy scalar to python type, and pass
                # python types through
                out = getattr(o, "item", lambda: o)()
                if isinstance(out, complex):
                    # python complex types are not JSON serializable, so we use the
                    # serialization defined in the zarr v3 spec
                    return _replace_special_floats([out.real, out.imag])
                elif np.isnan(out):
                    return "NaN"
                elif np.isinf(out):
                    return "Infinity" if out > 0 else "-Infinity"
            return out
        elif isinstance(o, Enum):
            return o.name
        # this serializes numcodecs compressors
        # todo: implement to_dict for codecs
        elif isinstance(o, numcodecs.abc.Codec):
            config: dict[str, Any] = o.get_config()
            return config
        else:
            return super().default(o)


def _replace_special_floats(obj: object) -> Any:
    """Helper function to replace NaN/Inf/-Inf values with special strings

    Note: this cannot be done in the V3JsonEncoder because Python's `json.dumps` optimistically
    converts NaN/Inf values to special types outside of the encoding step.
    """
    if isinstance(obj, float):
        if np.isnan(obj):
            return "NaN"
        elif np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
    elif isinstance(obj, dict):
        # Recursively replace in dictionaries
        return {k: _replace_special_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively replace in lists
        return [_replace_special_floats(item) for item in obj]
    return obj


class ArrayV3MetadataDict(TypedDict):
    """
    A typed dictionary model for zarr v3 metadata.
    """

    zarr_format: Literal[3]
    attributes: dict[str, JSON]


@dataclass(frozen=True, kw_only=True)
class ArrayV3Metadata(Metadata):
    shape: ChunkCoords
    data_type: DTypeWrapper
    chunk_grid: ChunkGrid
    chunk_key_encoding: ChunkKeyEncoding
    fill_value: Any
    codecs: tuple[Codec, ...]
    attributes: dict[str, Any] = field(default_factory=dict)
    dimension_names: tuple[str, ...] | None = None
    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["array"] = field(default="array", init=False)
    storage_transformers: tuple[dict[str, JSON], ...]

    def __init__(
        self,
        *,
        shape: Iterable[int],
        data_type: DTypeWrapper,
        chunk_grid: dict[str, JSON] | ChunkGrid,
        chunk_key_encoding: ChunkKeyEncodingLike,
        fill_value: object,
        codecs: Iterable[Codec | dict[str, JSON]],
        attributes: dict[str, JSON] | None,
        dimension_names: Iterable[str] | None,
        storage_transformers: Iterable[dict[str, JSON]] | None = None,
    ) -> None:
        """
        Because the class is a frozen dataclass, we set attributes using object.__setattr__
        """
        shape_parsed = parse_shapelike(shape)
        chunk_grid_parsed = ChunkGrid.from_dict(chunk_grid)
        chunk_key_encoding_parsed = ChunkKeyEncoding.from_dict(chunk_key_encoding)
        dimension_names_parsed = parse_dimension_names(dimension_names)
        fill_value_parsed = data_type.to_dtype().type(fill_value)
        attributes_parsed = parse_attributes(attributes)
        codecs_parsed_partial = parse_codecs(codecs)
        storage_transformers_parsed = parse_storage_transformers(storage_transformers)

        array_spec = ArraySpec(
            shape=shape_parsed,
            dtype=data_type.to_dtype(),
            fill_value=fill_value_parsed,
            config=ArrayConfig.from_dict({}),  # TODO: config is not needed here.
            prototype=default_buffer_prototype(),  # TODO: prototype is not needed here.
        )
        codecs_parsed = tuple(c.evolve_from_array_spec(array_spec) for c in codecs_parsed_partial)
        validate_codecs(codecs_parsed_partial, data_type)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "data_type", data_type)
        object.__setattr__(self, "chunk_grid", chunk_grid_parsed)
        object.__setattr__(self, "chunk_key_encoding", chunk_key_encoding_parsed)
        object.__setattr__(self, "codecs", codecs_parsed)
        object.__setattr__(self, "dimension_names", dimension_names_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "attributes", attributes_parsed)
        object.__setattr__(self, "storage_transformers", storage_transformers_parsed)

        self._validate_metadata()

    def _validate_metadata(self) -> None:
        if isinstance(self.chunk_grid, RegularChunkGrid) and len(self.shape) != len(
            self.chunk_grid.chunk_shape
        ):
            raise ValueError(
                "`chunk_shape` and `shape` need to have the same number of dimensions."
            )
        if self.dimension_names is not None and len(self.shape) != len(self.dimension_names):
            raise ValueError(
                "`dimension_names` and `shape` need to have the same number of dimensions."
            )
        if self.fill_value is None:
            raise ValueError("`fill_value` is required.")
        for codec in self.codecs:
            codec.validate(
                shape=self.shape, dtype=self.data_type.to_dtype(), chunk_grid=self.chunk_grid
            )

    @property
    def dtype(self) -> np.dtype[Any]:
        """Interpret Zarr dtype as NumPy dtype"""
        return self.data_type.to_dtype()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def chunks(self) -> ChunkCoords:
        if isinstance(self.chunk_grid, RegularChunkGrid):
            from zarr.codecs.sharding import ShardingCodec

            if len(self.codecs) == 1 and isinstance(self.codecs[0], ShardingCodec):
                sharding_codec = self.codecs[0]
                assert isinstance(sharding_codec, ShardingCodec)  # for mypy
                return sharding_codec.chunk_shape
            else:
                return self.chunk_grid.chunk_shape

        msg = (
            f"The `chunks` attribute is only defined for arrays using `RegularChunkGrid`."
            f"This array has a {self.chunk_grid} instead."
        )
        raise NotImplementedError(msg)

    @property
    def shards(self) -> ChunkCoords | None:
        if isinstance(self.chunk_grid, RegularChunkGrid):
            from zarr.codecs.sharding import ShardingCodec

            if len(self.codecs) == 1 and isinstance(self.codecs[0], ShardingCodec):
                return self.chunk_grid.chunk_shape
            else:
                return None

        msg = (
            f"The `shards` attribute is only defined for arrays using `RegularChunkGrid`."
            f"This array has a {self.chunk_grid} instead."
        )
        raise NotImplementedError(msg)

    @property
    def inner_codecs(self) -> tuple[Codec, ...]:
        if isinstance(self.chunk_grid, RegularChunkGrid):
            from zarr.codecs.sharding import ShardingCodec

            if len(self.codecs) == 1 and isinstance(self.codecs[0], ShardingCodec):
                return self.codecs[0].codecs
        return self.codecs

    def get_chunk_spec(
        self, _chunk_coords: ChunkCoords, array_config: ArrayConfig, prototype: BufferPrototype
    ) -> ArraySpec:
        assert isinstance(self.chunk_grid, RegularChunkGrid), (
            "Currently, only regular chunk grid is supported"
        )
        return ArraySpec(
            shape=self.chunk_grid.chunk_shape,
            dtype=self.dtype,
            fill_value=self.fill_value,
            config=array_config,
            prototype=prototype,
        )

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.chunk_key_encoding.encode_chunk_key(chunk_coords)

    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:
        d = self.to_dict()
        # d = _replace_special_floats(self.to_dict())
        return {ZARR_JSON: prototype.buffer.from_bytes(json.dumps(d, cls=V3JsonEncoder).encode())}

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        # make a copy because we are modifying the dict
        _data = data.copy()

        # check that the zarr_format attribute is correct
        _ = parse_zarr_format(_data.pop("zarr_format"))
        # check that the node_type attribute is correct
        _ = parse_node_type_array(_data.pop("node_type"))

        data_type_json = _data.pop("data_type")
        if isinstance(data_type_json, str):
            # check that the data_type attribute is valid
            data_type = get_data_type_by_name(data_type_json)

        else:
            data_type = get_data_type_from_dict(data_type_json)

        # check that the fill value is consistent with the data type
        fill_value_parsed = data_type.from_json_value(_data.pop("fill_value"), zarr_format=3)

        # dimension_names key is optional, normalize missing to `None`
        _data["dimension_names"] = _data.pop("dimension_names", None)

        # attributes key is optional, normalize missing to `None`
        _data["attributes"] = _data.pop("attributes", None)

        return cls(**_data, fill_value=fill_value_parsed, data_type=data_type)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        out_dict = super().to_dict()
        out_dict["fill_value"] = self.data_type.to_json_value(
            self.fill_value, zarr_format=self.zarr_format
        )
        if not isinstance(out_dict, dict):
            raise TypeError(f"Expected dict. Got {type(out_dict)}.")

        # if `dimension_names` is `None`, we do not include it in
        # the metadata document
        if out_dict["dimension_names"] is None:
            out_dict.pop("dimension_names")
        # if data_type has no configuration, we just serialize the name
        if "configuration" not in out_dict["data_type"]:
            out_dict["data_type"] = out_dict["data_type"]["name"]
        return out_dict

    def update_shape(self, shape: ChunkCoords) -> Self:
        return replace(self, shape=shape)

    def update_attributes(self, attributes: dict[str, JSON]) -> Self:
        return replace(self, attributes=attributes)
