from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from importlib.metadata import entry_points as get_entry_points
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar

import numpy as np

from zarr.core.config import BadConfigError, config

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    import numpy.typing as npt

    from zarr.abc.codec import (
        ArrayArrayCodec,
        ArrayBytesCodec,
        BytesBytesCodec,
        Codec,
        CodecPipeline,
    )
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.common import JSON
    from zarr.core.metadata.dtype import DTypeWrapper

__all__ = [
    "Registry",
    "get_buffer_class",
    "get_codec_class",
    "get_ndbuffer_class",
    "get_pipeline_class",
    "register_buffer",
    "register_codec",
    "register_ndbuffer",
    "register_pipeline",
]

T = TypeVar("T")


class Registry(dict[str, type[T]], Generic[T]):
    def __init__(self) -> None:
        super().__init__()
        self.lazy_load_list: list[EntryPoint] = []

    def lazy_load(self) -> None:
        for e in self.lazy_load_list:
            self.register(e.load())

        self.lazy_load_list.clear()

    def register(self, cls: type[T]) -> None:
        self[fully_qualified_name(cls)] = cls


@dataclass(frozen=True, kw_only=True)
class DataTypeRegistry:
    contents: dict[str, type[DTypeWrapper]] = field(default_factory=dict, init=False)
    lazy_load_list: list[EntryPoint] = field(default_factory=list, init=False)

    def lazy_load(self) -> None:
        for e in self.lazy_load_list:
            self.register(e.load())

        self.lazy_load_list.clear()

    def register(self: Self, cls: type[DTypeWrapper]) -> None:
        # don't register the same dtype twice
        if cls.name not in self.contents or self.contents[cls.name] != cls:
            self.contents[cls.name] = cls

    def get(self, key: str) -> type[DTypeWrapper]:
        return self.contents[key]


__codec_registries: dict[str, Registry[Codec]] = defaultdict(Registry)
__pipeline_registry: Registry[CodecPipeline] = Registry()
__buffer_registry: Registry[Buffer] = Registry()
__ndbuffer_registry: Registry[NDBuffer] = Registry()
__data_type_registry = DataTypeRegistry()

"""
The registry module is responsible for managing implementations of codecs,
pipelines, buffers and ndbuffers and collecting them from entrypoints.
The implementation used is determined by the config.

The registry module is also responsible for managing dtypes.
"""


def _collect_entrypoints() -> list[Registry[Any]]:
    """
    Collects codecs, pipelines, dtypes, buffers and ndbuffers from entrypoints.
    Entry points can either be single items or groups of items.
    Allowed syntax for entry_points.txt is e.g.

        [zarr.codecs]
        gzip = package:EntrypointGzipCodec1
        [zarr.codecs.gzip]
        some_name = package:EntrypointGzipCodec2
        another = package:EntrypointGzipCodec3

        [zarr]
        buffer = package:TestBuffer1
        [zarr.buffer]
        xyz = package:TestBuffer2
        abc = package:TestBuffer3
        ...
    """
    entry_points = get_entry_points()

    __buffer_registry.lazy_load_list.extend(entry_points.select(group="zarr.buffer"))
    __buffer_registry.lazy_load_list.extend(entry_points.select(group="zarr", name="buffer"))
    __ndbuffer_registry.lazy_load_list.extend(entry_points.select(group="zarr.ndbuffer"))
    __ndbuffer_registry.lazy_load_list.extend(entry_points.select(group="zarr", name="ndbuffer"))

    __data_type_registry.lazy_load_list.extend(entry_points.select(group="zarr.data_type"))
    __data_type_registry.lazy_load_list.extend(entry_points.select(group="zarr", name="data_type"))

    __pipeline_registry.lazy_load_list.extend(entry_points.select(group="zarr.codec_pipeline"))
    __pipeline_registry.lazy_load_list.extend(
        entry_points.select(group="zarr", name="codec_pipeline")
    )
    for e in entry_points.select(group="zarr.codecs"):
        __codec_registries[e.name].lazy_load_list.append(e)
    for group in entry_points.groups:
        if group.startswith("zarr.codecs."):
            codec_name = group.split(".")[2]
            __codec_registries[codec_name].lazy_load_list.extend(entry_points.select(group=group))
    return [
        *__codec_registries.values(),
        __pipeline_registry,
        __buffer_registry,
        __ndbuffer_registry,
    ]


def _reload_config() -> None:
    config.refresh()


def fully_qualified_name(cls: type) -> str:
    module = cls.__module__
    return module + "." + cls.__qualname__


def register_codec(key: str, codec_cls: type[Codec]) -> None:
    if key not in __codec_registries:
        __codec_registries[key] = Registry()
    __codec_registries[key].register(codec_cls)


def register_pipeline(pipe_cls: type[CodecPipeline]) -> None:
    __pipeline_registry.register(pipe_cls)


def register_ndbuffer(cls: type[NDBuffer]) -> None:
    __ndbuffer_registry.register(cls)


def register_buffer(cls: type[Buffer]) -> None:
    __buffer_registry.register(cls)


def register_data_type(cls: type[DTypeWrapper]) -> None:
    __data_type_registry.register(cls)


def get_codec_class(key: str, reload_config: bool = False) -> type[Codec]:
    if reload_config:
        _reload_config()

    if key in __codec_registries:
        # logger.debug("Auto loading codec '%s' from entrypoint", codec_id)
        __codec_registries[key].lazy_load()

    codec_classes = __codec_registries[key]
    if not codec_classes:
        raise KeyError(key)

    config_entry = config.get("codecs", {}).get(key)
    if config_entry is None:
        if len(codec_classes) == 1:
            return next(iter(codec_classes.values()))
        warnings.warn(
            f"Codec '{key}' not configured in config. Selecting any implementation.",
            stacklevel=2,
        )
        return list(codec_classes.values())[-1]
    selected_codec_cls = codec_classes[config_entry]

    if selected_codec_cls:
        return selected_codec_cls
    raise KeyError(key)


def _resolve_codec(data: dict[str, JSON]) -> Codec:
    """
    Get a codec instance from a dict representation of that codec.
    """
    # TODO: narrow the type of the input to only those dicts that map on to codec class instances.
    return get_codec_class(data["name"]).from_dict(data)  # type: ignore[arg-type]


def _parse_bytes_bytes_codec(data: dict[str, JSON] | Codec) -> BytesBytesCodec:
    """
    Normalize the input to a ``BytesBytesCodec`` instance.
    If the input is already a ``BytesBytesCodec``, it is returned as is. If the input is a dict, it
    is converted to a ``BytesBytesCodec`` instance via the ``_resolve_codec`` function.
    """
    from zarr.abc.codec import BytesBytesCodec

    if isinstance(data, dict):
        result = _resolve_codec(data)
        if not isinstance(result, BytesBytesCodec):
            msg = f"Expected a dict representation of a BytesBytesCodec; got a dict representation of a {type(result)} instead."
            raise TypeError(msg)
    else:
        if not isinstance(data, BytesBytesCodec):
            raise TypeError(f"Expected a BytesBytesCodec. Got {type(data)} instead.")
        result = data
    return result


def _parse_array_bytes_codec(data: dict[str, JSON] | Codec) -> ArrayBytesCodec:
    """
    Normalize the input to a ``ArrayBytesCodec`` instance.
    If the input is already a ``ArrayBytesCodec``, it is returned as is. If the input is a dict, it
    is converted to a ``ArrayBytesCodec`` instance via the ``_resolve_codec`` function.
    """
    from zarr.abc.codec import ArrayBytesCodec

    if isinstance(data, dict):
        result = _resolve_codec(data)
        if not isinstance(result, ArrayBytesCodec):
            msg = f"Expected a dict representation of a ArrayBytesCodec; got a dict representation of a {type(result)} instead."
            raise TypeError(msg)
    else:
        if not isinstance(data, ArrayBytesCodec):
            raise TypeError(f"Expected a ArrayBytesCodec. Got {type(data)} instead.")
        result = data
    return result


def _parse_array_array_codec(data: dict[str, JSON] | Codec) -> ArrayArrayCodec:
    """
    Normalize the input to a ``ArrayArrayCodec`` instance.
    If the input is already a ``ArrayArrayCodec``, it is returned as is. If the input is a dict, it
    is converted to a ``ArrayArrayCodec`` instance via the ``_resolve_codec`` function.
    """
    from zarr.abc.codec import ArrayArrayCodec

    if isinstance(data, dict):
        result = _resolve_codec(data)
        if not isinstance(result, ArrayArrayCodec):
            msg = f"Expected a dict representation of a ArrayArrayCodec; got a dict representation of a {type(result)} instead."
            raise TypeError(msg)
    else:
        if not isinstance(data, ArrayArrayCodec):
            raise TypeError(f"Expected a ArrayArrayCodec. Got {type(data)} instead.")
        result = data
    return result


def get_pipeline_class(reload_config: bool = False) -> type[CodecPipeline]:
    if reload_config:
        _reload_config()
    __pipeline_registry.lazy_load()
    path = config.get("codec_pipeline.path")
    pipeline_class = __pipeline_registry.get(path)
    if pipeline_class:
        return pipeline_class
    raise BadConfigError(
        f"Pipeline class '{path}' not found in registered pipelines: {list(__pipeline_registry)}."
    )


def get_buffer_class(reload_config: bool = False) -> type[Buffer]:
    if reload_config:
        _reload_config()
    __buffer_registry.lazy_load()

    path = config.get("buffer")
    buffer_class = __buffer_registry.get(path)
    if buffer_class:
        return buffer_class
    raise BadConfigError(
        f"Buffer class '{path}' not found in registered buffers: {list(__buffer_registry)}."
    )


def get_ndbuffer_class(reload_config: bool = False) -> type[NDBuffer]:
    if reload_config:
        _reload_config()
    __ndbuffer_registry.lazy_load()
    path = config.get("ndbuffer")
    ndbuffer_class = __ndbuffer_registry.get(path)
    if ndbuffer_class:
        return ndbuffer_class
    raise BadConfigError(
        f"NDBuffer class '{path}' not found in registered buffers: {list(__ndbuffer_registry)}."
    )


def get_data_type_by_name(dtype: str, configuration: dict[str, JSON] | None = None) -> DTypeWrapper:
    __data_type_registry.lazy_load()
    if configuration is None:
        _configuration = {}
    else:
        _configuration = configuration
    maybe_dtype_cls = __data_type_registry.get(dtype)
    if maybe_dtype_cls is None:
        raise ValueError(f"No data type class matching name {dtype}")
    return maybe_dtype_cls.from_dict(_configuration)


def get_data_type_from_dict(dtype: dict[str, JSON]) -> DTypeWrapper:
    __data_type_registry.lazy_load()
    dtype_name = dtype["name"]
    dtype_cls = __data_type_registry.get(dtype_name)
    if dtype_cls is None:
        raise ValueError(f"No data type class matching name {dtype_name}")
    return dtype_cls.from_dict(dtype.get("configuration", {}))


def get_data_type_from_numpy(dtype: npt.DTypeLike) -> DTypeWrapper:
    np_dtype = np.dtype(dtype)
    __data_type_registry.lazy_load()
    for val in __data_type_registry.contents.values():
        if val.dtype_cls is type(np_dtype):
            return val.from_dtype(np_dtype)
    raise ValueError(
        f"numpy dtype '{dtype}' does not have a corresponding Zarr dtype in: {list(__data_type_registry.contents)}."
    )


_collect_entrypoints()
