"""
The extensions that Zarr metadata is read with, and the resolver that reads with them.

A `Context` is the set of extensions a read may resolve, by kind. It is a value: the same for a
whole read. A `Resolver` is a context positioned in a document: the context, the Zarr format of the
document, and the location in the document. It is re-positioned at every nested step, and is
threaded through the read by zarr. (The pair mirrors `referencing.Registry` and
`referencing.Resolver` in python-jsonschema.)

The explicit-IO array layer (`zarr.ArrayIO` and `zarr.AsyncArrayIO`) takes a context publicly;
everywhere else, both remain internal.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from zarr.errors import UnknownCodecError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr.abc.codec import Codec, CodecPipeline
    from zarr.core.common import ZarrFormat
    from zarr.core.dtype.common import DTypeJSON
    from zarr.core.dtype.registry import DataTypeRegistry
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


_DEFAULT_DATA_TYPES = "zarr.dtype.data_type_registry"


def _default_data_types() -> DataTypeRegistry:
    """The default data type registry, `zarr.dtype.data_type_registry`."""
    # Imported on use: `zarr.core.dtype` imports this module (its registry and
    # structured data type read with a `Resolver`), so this module cannot import it.
    from zarr.core.dtype import data_type_registry

    return data_type_registry


def _registry_snapshot(registry: DataTypeRegistry) -> DataTypeRegistry:
    """A copy of `registry` as it is now, or the default registry itself."""
    if registry is _default_data_types():
        return registry
    # A copy of the registry's own class (a subclass may match names its own way),
    # with its own contents.
    snapshot = copy.copy(registry)
    object.__setattr__(snapshot, "contents", dict(registry.contents))
    object.__setattr__(snapshot, "_lazy_load_list", list(registry._lazy_load_list))
    # Loaded now, so the copy compares the same before and after its first use.
    snapshot._lazy_load()
    return snapshot


@dataclass(frozen=True, kw_only=True)
class Context:
    """
    The extensions that Zarr metadata is read with, by kind.

    A kind that is `None` resolves from the default registries, exactly as it
    would with no context; a kind that is set is a closed world, and a name
    outside it fails to resolve even when the default registries know it.
    The scope governs top-level names: codecs nested inside a
    `sharding_indexed` codec, and a data type a codec names in its
    configuration (`cast_value`'s target), resolve through the default
    registries today.

    Attributes
    ----------
    data_types : DataTypeRegistry | None
        The data types a read may resolve, or `None` for the default data type
        registry. A registry other than the default is copied when the context
        is made, so registering a data type in it later does not change the
        context (registering one in the context's own copy does); its pending
        entry points load then too, so one that fails to load fails there. A registry
        subclass that overrides `match_json` changes how a top-level data type
        matches; a data type nested in another (a struct's fields) matches
        through the registry's contents.
    codecs : Mapping[str, type[Codec]] | None
        The codec class each codec name resolves to, or `None` for the
        default codec registry. Under a closed codec scope a top-level codec
        instance must be of the class the scope maps its name to, so the scope
        admits no top-level codec outside it. The name is the one the codec
        writes in its JSON: a scope that holds a class only under another name
        (such as `BytesCodec` under `endian`; it writes `bytes`) admits none of
        its instances, as it could not read back the chain it would store. List
        the class under each name it reads and writes. Each class's `from_dict`
        must return an instance of that class, which is what the scope binds, and
        so must its `evolve_from_array_spec` (returning the instance unchanged once
        its JSON is complete): a class that fitting replaces is not held to the
        scope. The scope governs Zarr format 3 codecs
        only: a Zarr format 2 array's filters and compressor are numcodecs
        objects, which no context governs.
    codec_pipeline : type[CodecPipeline] | None
        The codec pipeline class arrays are read and written with, or `None`
        for the pipeline the global configuration selects. A pipeline reads
        its own options (such as zarrs-python's `codec_pipeline.strict`) from
        the global configuration when it is built; to fix them per binding,
        give a subclass that builds under them.

    The data types and codecs in scope decide which names a binding admits and
    the classes `metadata_legacy` resolves them to. A pipeline that encodes
    from the metadata's JSON with implementations of its own (such as
    zarrs-python) runs its own implementation for each admitted name.

    Contexts compare by their fields. A context that sets `codecs` or
    `data_types` is not hashable (it holds a mapping or a registry); one that
    sets neither is.
    """

    data_types: DataTypeRegistry | None = None
    codecs: Mapping[str, type[Codec]] | None = None
    codec_pipeline: type[CodecPipeline] | None = None

    def __post_init__(self) -> None:
        # Copies, read-only for codecs: the scope must not change under a binding
        # that captured it. The default data type registry is kept by reference
        # (see `default`). Contexts compare by their fields; one that holds a
        # codec mapping or a data type registry is not hashable.
        if self.codecs is not None:
            object.__setattr__(self, "codecs", MappingProxyType(dict(self.codecs)))
        if self.data_types is not None:
            object.__setattr__(self, "data_types", _registry_snapshot(self.data_types))

    def __getstate__(self) -> dict[str, object]:
        # A mapping proxy does not pickle; carry the scope as a plain dict. The
        # default data type registry is carried by reference, so it unpickles
        # as the receiving process's registry rather than a snapshot of ours;
        # any other registry unpickles as a copy.
        state = dict(self.__dict__)
        if self.codecs is not None:
            state["codecs"] = dict(self.codecs)
        if self.data_types is _default_data_types():
            state["data_types"] = _DEFAULT_DATA_TYPES
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        codecs = state.get("codecs")
        if codecs is not None:
            object.__setattr__(self, "codecs", MappingProxyType(dict(codecs)))  # type: ignore[call-overload]
        if state.get("data_types") == _DEFAULT_DATA_TYPES:
            object.__setattr__(self, "data_types", _default_data_types())

    @classmethod
    def _over(cls, data_types: DataTypeRegistry) -> Context:
        """A context over `data_types` itself, not a copy: for a resolver that lives
        only as long as one call on that registry."""
        context = object.__new__(cls)
        object.__setattr__(context, "data_types", data_types)
        object.__setattr__(context, "codecs", None)
        object.__setattr__(context, "codec_pipeline", None)
        return context

    @classmethod
    def default(cls) -> Context:
        """
        The context of the default registries, such as `zarr.dtype.data_type_registry`.

        It refers to the registries themselves, so it includes extensions registered after it is
        created; it is `Context()`, which leaves every kind unset.
        """
        return cls()

    @property
    def limits_names(self) -> bool:
        """Whether this context limits which names resolve: it sets `data_types`
        or `codecs`. A context that only selects the codec pipeline resolves
        exactly as no context does."""
        return self.data_types is not None or self.codecs is not None

    def data_type_registry(self) -> DataTypeRegistry:
        """The data types this context resolves: its own, or the default registry."""
        if self.data_types is None:
            return _default_data_types()
        return self.data_types

    def codec_class(self, name: str) -> type[Codec]:
        """The codec class `name` resolves to in this context.

        Raises [`zarr.errors.UnknownCodecError`][] naming the codecs in scope
        when `name` is not among them.
        """
        if self.codecs is None:
            # Imported on use: `zarr.registry` imports `zarr.core.dtype`, which
            # imports this module (see `_default_data_types`).
            from zarr.registry import get_codec_class

            return get_codec_class(name)
        try:
            return self.codecs[name]
        except KeyError:
            raise UnknownCodecError(
                f"Codec {name!r} is not in this context. Codecs in scope: {sorted(self.codecs)}."
            ) from None

    def pipeline_class(self) -> type[CodecPipeline]:
        """The codec pipeline class this context resolves to."""
        if self.codec_pipeline is None:
            # Imported on use, as in `codec_class`.
            from zarr.registry import get_pipeline_class

            return get_pipeline_class()
        return self.codec_pipeline


type JSONLocation = tuple[str | int, ...]


def json_pointer(loc: JSONLocation) -> str:
    """
    The JSON Pointer (RFC 6901) of a location in a JSON document, e.g. `/codecs/0/configuration`.
    """
    return "".join("/" + str(key).replace("~", "~0").replace("/", "~1") for key in loc)


@dataclass(frozen=True, kw_only=True)
class Resolver:
    """
    Resolves the extensions in a Zarr metadata document from a context, at a location in the
    document.

    An extension that contains other extensions, such as a structured data type, resolves each of
    them with the resolver returned by `at`, so that they come from the same context, use the same
    Zarr format, and report errors with their location.

    Attributes
    ----------
    context : Context
        The extensions the read may resolve.
    zarr_format : ZarrFormat
        The Zarr format of the document.
    loc : tuple[str | int, ...]
        The location in the document of the JSON being read, as keys and indices. A Zarr V2 data
        type is located by its name as written in the document (the value of `"dtype"`), not by
        the `{"name": ..., "object_codec_id": ...}` form zarr reads it in.
    """

    context: Context
    zarr_format: ZarrFormat
    loc: JSONLocation = ()

    def at(self, *keys: str | int) -> Resolver:
        """
        The resolver for the JSON at `keys` within the JSON at this resolver's location.
        """
        return replace(self, loc=(*self.loc, *keys))

    def resolve_data_type(self, data: DTypeJSON) -> ZDType[TBaseDType, TBaseScalar]:
        """
        Resolve the JSON representation of a data type at this resolver's location.

        Raises
        ------
        ValueError
            If no data type in the context matches `data`.
        """
        return self.context.data_type_registry()._match_json(data, resolver=self)
