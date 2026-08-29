"""Optional msgspec integration: field types and a decode hook over the core models.

Importing this module requires msgspec; the core package deliberately does
not depend on it, so this module is never imported by `zarr_metadata` itself.

msgspec's extension point is a decode hook consulted only for annotated
types msgspec does not support natively — and the core models are
dataclasses, a kind msgspec supports natively, so annotating a field with a
core model class directly would engage msgspec's own field-by-field
dataclass coercion and bypass `from_json` (the single source of truth for
structural validation and normalization). Each name this module exports is
therefore a runtime marker class that msgspec treats as a custom type,
forcing every value for the field through this module's `dec_hook`. Each
marker registers its core model class as a virtual subclass, so the values
`dec_hook` produces satisfy msgspec's type check while the instances ARE
the core classes — they interoperate freely with non-msgspec code
(equality, isinstance, nesting). Static type checkers see each field type
as its core model class, so `manifest.metadata` below is a
`zarr_metadata.model.ZarrV3ArrayMetadata`.

`dec_hook` routes a raw document through `from_json` and passes an existing
model instance through unchanged. `MetadataValidationError` subclasses
`ValueError`, so a failed parse surfaces as a
`msgspec.ValidationError` carrying the
loc-annotated problem messages, with msgspec appending the path of the
failing field (``- at `$.metadata` ``).

Usage:

    import msgspec

    import zarr_metadata.msgspec as zmm

    class ArrayManifest(msgspec.Struct):
        path: str
        metadata: zmm.ZarrV3ArrayMetadata

    manifest = msgspec.json.decode(data, type=ArrayManifest, dec_hook=zmm.dec_hook)

The same hook serves `msgspec.convert` — e.g.
`msgspec.convert(doc, zmm.ZarrV3ArrayMetadata, dec_hook=zmm.dec_hook)` —
and a prebuilt `msgspec.json.Decoder`. An application that already has a
decode hook of its own composes via `make_dec_hook(wrapped=...)`.

Two msgspec limits shape what this module can offer:

- Serialization cannot be delegated: msgspec's encoders are value-driven
  and consult their `enc_hook` only for objects msgspec cannot encode
  natively, which a dataclass never is, so no hook can route a model
  instance through `to_json`. Encoding a model directly either raises (the
  `UNSET` sentinel is unencodable) or silently emits a raw field dump that
  is NOT the canonical document (no shorthand collapse, no omit-empty
  conventions). Serialize explicitly: put `model.to_json()` — the
  canonical document — wherever msgspec-encodable output is needed.
- Unions may contain at most one hook-handled type, so a field cannot be
  typed as, say, array-or-group metadata. Decode such a field as an
  untyped mapping and dispatch on its content.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Final

from zarr_metadata import model as _model

if TYPE_CHECKING:
    from collections.abc import Callable

    # For static type checkers the field types ARE the core model classes.
    ZarrV3ArrayMetadata = _model.ZarrV3ArrayMetadata
    ZarrV2ArrayMetadata = _model.ZarrV2ArrayMetadata
    ZarrV3GroupMetadata = _model.ZarrV3GroupMetadata
    ZarrV2GroupMetadata = _model.ZarrV2GroupMetadata
    ZarrV3ConsolidatedMetadata = _model.ZarrV3ConsolidatedMetadata
    ZarrV2ConsolidatedMetadata = _model.ZarrV2ConsolidatedMetadata
    ZarrV3MetadataField = _model.ZarrV3NamedConfig
else:
    # At runtime each field type is a marker: an empty ABC msgspec treats as
    # a custom type (so `dec_hook` is consulted) with the core model class
    # registered as a virtual subclass (so the core instances `dec_hook`
    # returns satisfy msgspec's isinstance check on hook results).

    class ZarrV3ArrayMetadata(abc.ABC):  # noqa: B024
        """Field type for a v3 array metadata document (`zarr.json` content)."""

    ZarrV3ArrayMetadata.register(_model.ZarrV3ArrayMetadata)

    class ZarrV2ArrayMetadata(abc.ABC):  # noqa: B024
        """Field type for a v2 array metadata document (merged `.zarray` + `.zattrs` form)."""

    ZarrV2ArrayMetadata.register(_model.ZarrV2ArrayMetadata)

    class ZarrV3GroupMetadata(abc.ABC):  # noqa: B024
        """Field type for a v3 group metadata document (`zarr.json` content)."""

    ZarrV3GroupMetadata.register(_model.ZarrV3GroupMetadata)

    class ZarrV2GroupMetadata(abc.ABC):  # noqa: B024
        """Field type for a v2 group metadata document (merged `.zgroup` + `.zattrs` form)."""

    ZarrV2GroupMetadata.register(_model.ZarrV2GroupMetadata)

    class ZarrV3ConsolidatedMetadata(abc.ABC):  # noqa: B024
        """Field type for v3 inline consolidated metadata."""

    ZarrV3ConsolidatedMetadata.register(_model.ZarrV3ConsolidatedMetadata)

    class ZarrV2ConsolidatedMetadata(abc.ABC):  # noqa: B024
        """Field type for a v2 `.zmetadata` document."""

    ZarrV2ConsolidatedMetadata.register(_model.ZarrV2ConsolidatedMetadata)

    class ZarrV3MetadataField(abc.ABC):  # noqa: B024
        """Field type for one normalized v3 metadata extension envelope."""

    ZarrV3MetadataField.register(_model.ZarrV3NamedConfig)


_DECODERS: Final[dict[type, tuple[type, Callable[[object], object]]]] = {
    ZarrV3ArrayMetadata: (_model.ZarrV3ArrayMetadata, _model.ZarrV3ArrayMetadata.from_json),
    ZarrV2ArrayMetadata: (_model.ZarrV2ArrayMetadata, _model.ZarrV2ArrayMetadata.from_json),
    ZarrV3GroupMetadata: (_model.ZarrV3GroupMetadata, _model.ZarrV3GroupMetadata.from_json),
    ZarrV2GroupMetadata: (_model.ZarrV2GroupMetadata, _model.ZarrV2GroupMetadata.from_json),
    ZarrV3ConsolidatedMetadata: (
        _model.ZarrV3ConsolidatedMetadata,
        _model.ZarrV3ConsolidatedMetadata.from_json,
    ),
    ZarrV2ConsolidatedMetadata: (
        _model.ZarrV2ConsolidatedMetadata,
        _model.ZarrV2ConsolidatedMetadata.from_json,
    ),
    ZarrV3MetadataField: (_model.ZarrV3NamedConfig, _model.ZarrV3NamedConfig.from_json),
}
"""Marker class -> (pass-through core class, document parser)."""


def dec_hook(type: type, obj: Any) -> Any:
    """Decode `obj` for a field annotated with one of this module's field types.

    Pass as `dec_hook=` to `msgspec.json.decode`, `msgspec.convert`, or a
    `msgspec.json.Decoder`. An existing core model instance passes through
    unchanged; anything else is parsed by the core model's `from_json`. A
    type this module does not cover raises `NotImplementedError`, msgspec's
    convention for "still unsupported"; to keep decoding custom types of
    your own alongside these, chain your hook with `make_dec_hook`.
    """
    try:
        core_cls, parse = _DECODERS[type]
    except KeyError:
        raise NotImplementedError(f"Objects of type {type} are not supported") from None
    if isinstance(obj, core_cls):
        return obj
    return parse(obj)


def make_dec_hook(wrapped: Callable[[type, Any], Any] | None = None) -> Callable[[type, Any], Any]:
    """Return a decode hook that also delegates unknown types to `wrapped`.

    The returned hook handles this module's field types exactly like
    `dec_hook` and hands every other type to `wrapped`, so an application's
    existing custom-type decoding keeps working alongside the model field
    types. With no `wrapped` hook this returns `dec_hook` itself.
    """
    if wrapped is None:
        return dec_hook

    def hook(type: type, obj: Any) -> Any:
        if type in _DECODERS:
            return dec_hook(type, obj)
        return wrapped(type, obj)

    return hook


__all__ = [
    "ZarrV2ArrayMetadata",
    "ZarrV2ConsolidatedMetadata",
    "ZarrV2GroupMetadata",
    "ZarrV3ArrayMetadata",
    "ZarrV3ConsolidatedMetadata",
    "ZarrV3GroupMetadata",
    "ZarrV3MetadataField",
    "dec_hook",
    "make_dec_hook",
]
