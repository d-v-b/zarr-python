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


@dataclass(frozen=True, kw_only=True)
class Context:
    """
    The extensions that Zarr metadata is read with, by kind.

    A kind that is `None` resolves from the default registries, exactly as it
    would with no context; a kind that is set is a closed world, and a name
    outside it fails to resolve even when the default registries know it.
    The scope governs top-level names: codecs nested inside a
    `sharding_indexed` codec resolve through the default registry today.

    Attributes
    ----------
    data_types : DataTypeRegistry
        The data types a read may resolve.
    codecs : Mapping[str, type[Codec]] | None
        The codec class each codec name resolves to, or `None` for the
        default codec registry. Codec instances are never resolved by name,
        so they pass through any context.
    codec_pipeline : type[CodecPipeline] | None
        The codec pipeline class arrays are read and written with, or `None`
        for the pipeline the global configuration selects.
    """

    data_types: DataTypeRegistry
    codecs: Mapping[str, type[Codec]] | None = None
    codec_pipeline: type[CodecPipeline] | None = None

    def __post_init__(self) -> None:
        # A read-only copy: the scope must not change under a binding that
        # captured it. Contexts compare by their fields and are not hashable.
        if self.codecs is not None:
            object.__setattr__(self, "codecs", MappingProxyType(dict(self.codecs)))

    def __getstate__(self) -> dict[str, object]:
        # A mapping proxy does not pickle; carry the scope as a plain dict.
        state = dict(self.__dict__)
        if self.codecs is not None:
            state["codecs"] = dict(self.codecs)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        codecs = state.get("codecs")
        if codecs is not None:
            object.__setattr__(self, "codecs", MappingProxyType(dict(codecs)))  # type: ignore[call-overload]

    @classmethod
    def default(cls) -> Context:
        """
        The context of the default registries, such as `zarr.dtype.data_type_registry`.

        It refers to the registries themselves, so it includes extensions registered after it is
        created.
        """
        # avoid circular import
        from zarr.core.dtype import data_type_registry

        return cls(data_types=data_type_registry)

    def codec_class(self, name: str) -> type[Codec]:
        """The codec class `name` resolves to in this context.

        Raises [`zarr.errors.UnknownCodecError`][] naming the codecs in scope
        when `name` is not among them.
        """
        if self.codecs is None:
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
        return self.context.data_types._match_json(data, resolver=self)
