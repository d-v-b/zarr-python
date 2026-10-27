"""In-memory models for Zarr array metadata documents."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from typing_extensions import TypedDict, Unpack

from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
)
from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import (
    ARRAY_METADATA_STANDARD_KEYS_V3,
    StoreKey,
    dump_store_json,
    load_store_json,
    parse_array_metadata_v2,
    parse_array_metadata_v3,
)
from zarr_metadata.v2.array import ZARR_V2_ARRAY_METADATA_STORE_KEY
from zarr_metadata.v2.attributes import ZARR_V2_ATTRIBUTES_STORE_KEY
from zarr_metadata.v3._common import parse_metadata_field_v3
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS
from zarr_metadata.v3.array import ZARR_V3_ARRAY_METADATA_STORE_KEY

if TYPE_CHECKING:
    from zarr_metadata._common import JSONValue, ZarrV3NamedConfigJSON
    from zarr_metadata.v2.array import (
        ZarrV2ArrayDimensionSeparator,
        ZarrV2ArrayMetadataJSON,
        ZarrV2ArrayMetadataStoreKey,
        ZarrV2ArrayOrder,
        ZarrV2DataTypeMetadata,
    )
    from zarr_metadata.v2.attributes import ZarrV2AttributesStoreKey
    from zarr_metadata.v2.codec import ZarrV2CodecMetadata
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
    from zarr_metadata.v3._registry import Context
    from zarr_metadata.v3.array import (
        ZarrV3ArrayMetadataJSON,
        ZarrV3ArrayMetadataStoreKey,
        ZarrV3ExtensionField,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ZarrV3NamedConfig:
    """A normalized v3 metadata field with its reader obligation.

    Bare names and missing configurations normalize to an empty configuration.
    Bare names and missing `must_understand` members normalize to the spec's
    implicit `True` value (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1573).
    """

    name: str
    """The extension's name, e.g. `"gzip"` or `"regular"`."""
    configuration: dict[str, JSONValue]
    """The extension's parameters.

    `{}` when the document gave a bare name or omitted `configuration`.
    """
    must_understand: bool = True
    """Whether a reader that does not recognize this extension must refuse the document.

    Absent means `True`. The spec does not permit `False` for a data type,
    chunk grid or chunk key encoding
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581),
    and `ZarrV3ArrayMetadata.from_json` refuses it at every extension point,
    codecs and storage transformers included.
    """

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        """Return the model as a v3 metadata field, in its shortest spelling.

        The bare name when `configuration` is empty and `must_understand` is
        `True`; otherwise an object with `name`, with `configuration` only when
        it is not empty, and with `must_understand: false` only when it is
        `False`. The result shares no mutable state with the model.
        """
        if not self.configuration and self.must_understand:
            return self.name
        # `configuration` is ReadOnly, so it is set in the literal rather than
        # assigned afterwards. to_json output shares no mutable state with the
        # model.
        out: ZarrV3NamedConfigJSON = (
            {"name": self.name, "configuration": copy.deepcopy(self.configuration)}
            if self.configuration
            else {"name": self.name}
        )
        if not self.must_understand:
            out["must_understand"] = False
        return out

    @classmethod
    def from_json(cls, data: object) -> ZarrV3NamedConfig:
        """Return the model of the v3 metadata field `data`, or raise `MetadataValidationError`.

        `data` is a bare name string, or an object with a string `name` and
        optional `configuration` (an object of string keys and JSON values)
        and boolean `must_understand` members; any other member is a problem,
        and every problem found is raised together. A bare name or an absent
        `configuration` reads as `{}`, and an absent `must_understand` as
        `True`. Only the envelope is judged: the configuration is not read
        through any definition, and `must_understand: false` is accepted. The
        model shares no mutable state with `data`.
        """
        field = parse_metadata_field_v3(data)
        if isinstance(field, str):
            return cls(name=field, configuration={}, must_understand=True)
        # A read model shares no mutable state with what it read.
        configuration = copy.deepcopy(dict(field.get("configuration", {})))
        return cls(
            name=field["name"],
            configuration=configuration,
            must_understand=field.get("must_understand", True),
        )


ZarrV3MetadataField: TypeAlias = ZarrV3NamedConfig
"""The in-memory model of one field of a v3 metadata document.

This is the role-named alias for annotation positions: model fields and
consumer signatures should say `ZarrV3MetadataField` (the logical meaning)
rather than `ZarrV3NamedConfig` (the serialized form the field currently
takes). Today every metadata field normalizes to a named configuration plus
its reader obligation, so the alias is exactly `ZarrV3NamedConfig`; if a future
spec revision adds a field form that cannot be normalized to those values,
this alias widens to a union and annotation sites do not change. Mirrors the
raw-layer split between `ZarrV3NamedConfigJSON` (shape) and
`ZarrV3MetadataFieldJSON` (field union).
"""


def must_understand_subset(
    extra_fields: Mapping[str, ZarrV3ExtensionField],
) -> dict[str, ZarrV3ExtensionField]:
    """The subset of `extra_fields` the reader is obligated to understand.

    Per the v3 spec (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1578), an extension field is implicitly `must_understand:
    True` unless it explicitly says otherwise, and an implementation MUST fail to
    open a group or array carrying fields it does not recognize that are not
    explicitly `must_understand: false`. A non-mapping field value cannot
    carry the explicit waiver, so it always requires understanding (the
    runtime isinstance check defends against values looser than the declared
    `ZarrV3ExtensionField`).
    """
    return {
        name: value
        for name, value in extra_fields.items()
        if not (isinstance(value, Mapping) and value.get("must_understand") is False)
    }


class ZarrV3ArrayMetadataPartial(TypedDict, total=False):
    """
    Partial form of the constructor-settable fields of `ZarrV3ArrayMetadata`.

    Every key is optional and typed with the model's own (not serialized)
    value types, so it describes valid keyword arguments to
    `ZarrV3ArrayMetadata.update`. The `init=False` fields `zarr_format` and
    `node_type` are intentionally excluded, since they cannot be passed to
    `dataclasses.replace`.

    Drift between this type and the model's settable fields is prevented by
    `tests/model/test_array.py::test_partial_keys_match_settable_model_fields`.
    """

    shape: tuple[int, ...]
    """The length of each dimension of the array."""
    fill_value: JSONValue
    """The JSON value of an element no chunk has written.

    Its permitted spellings depend on `data_type`; the model holds it as the
    document wrote it, uninterpreted.
    """
    data_type: ZarrV3MetadataField
    """The array's data type."""
    chunk_grid: ZarrV3MetadataField
    """How the array is divided into chunks."""
    codecs: tuple[ZarrV3MetadataField, ...]
    """The codecs that encode each chunk, in the order they encode it.

    A valid document holds an array-to-bytes codec, so this is not empty
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L582-L583).
    """
    chunk_key_encoding: ZarrV3MetadataField
    """How a chunk's grid coordinates map to its store key."""
    dimension_names: tuple[str | None, ...] | UNSET
    """A name, or `None` for an unnamed dimension, for each dimension of `shape`.

    `UNSET` when the document omits the key, which leaves every dimension
    unnamed.
    """
    attributes: dict[str, JSONValue]
    """User attributes.

    A document without the key and one with an empty object both read as
    `{}`, and `to_json` omits an empty mapping.
    """
    storage_transformers: tuple[ZarrV3MetadataField, ...]
    """The storage transformers, in order.

    A document without the key and one with an empty list both read as `()`,
    which the spec gives the same meaning
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L627-L628),
    and `to_json` omits an empty tuple.
    """
    extra_fields: dict[str, ZarrV3ExtensionField]
    """Each top-level member outside the standard array keys, by name, as the document wrote it.

    A key in `ARRAY_METADATA_STANDARD_KEYS_V3` is refused on construction with
    `MetadataValidationError`. `must_understand_fields` is the subset a reader
    must recognize.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class ZarrV3ArrayMetadata:
    """In-memory model of a v3 array metadata document.

    A canonical, semantically lossless representation of the `zarr.json`
    content for an array. Extension points (`data_type`, `chunk_grid`,
    `chunk_key_encoding`, `codecs`, `storage_transformers`) are held as
    `ZarrV3MetadataField` values (currently `ZarrV3NamedConfig` name,
    configuration, and obligation records). `from_json` and
    `from_key_value` read each through the definition that claims its name
    in a scope -- `CORE_AND_EXTENSIONS` unless a `context` is passed -- and
    the model holds what they read as written; `fill_value` is held
    verbatim in its JSON form. Equivalent extension
    spellings normalize to shorthand strings when configuration is empty and
    understanding is required.
    """

    zarr_format: Literal[3] = field(default=3, init=False)
    """The Zarr format version, always `3`."""
    node_type: Literal["array"] = field(default="array", init=False)
    """The node type, always `"array"`."""
    shape: tuple[int, ...]
    """The length of each dimension of the array."""
    fill_value: JSONValue
    """The JSON value of an element no chunk has written.

    Its permitted spellings depend on `data_type`; the model holds it as the
    document wrote it, uninterpreted.
    """
    data_type: ZarrV3MetadataField
    """The array's data type."""
    chunk_grid: ZarrV3MetadataField
    """How the array is divided into chunks."""
    codecs: tuple[ZarrV3MetadataField, ...]
    """The codecs that encode each chunk, in the order they encode it.

    A valid document holds an array-to-bytes codec, so this is not empty
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L582-L583).
    """
    chunk_key_encoding: ZarrV3MetadataField
    """How a chunk's grid coordinates map to its store key."""
    dimension_names: tuple[str | None, ...] | UNSET
    """A name, or `None` for an unnamed dimension, for each dimension of `shape`.

    `UNSET` when the document omits the key, which leaves every dimension
    unnamed.
    """
    attributes: dict[str, JSONValue]
    """User attributes.

    A document without the key and one with an empty object both read as
    `{}`, and `to_json` omits an empty mapping.
    """
    storage_transformers: tuple[ZarrV3MetadataField, ...]
    """The storage transformers, in order.

    A document without the key and one with an empty list both read as `()`,
    which the spec gives the same meaning
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L627-L628),
    and `to_json` omits an empty tuple.
    """
    extra_fields: dict[str, ZarrV3ExtensionField]
    """Each top-level member outside the standard array keys, by name, as the document wrote it.

    A key in `ARRAY_METADATA_STANDARD_KEYS_V3` is refused on construction with
    `MetadataValidationError`. `must_understand_fields` is the subset a reader
    must recognize.
    """

    @classmethod
    def create_default(cls, **overrides: Unpack[ZarrV3ArrayMetadataPartial]) -> ZarrV3ArrayMetadata:
        """
        Create a default (empty) v3 array metadata model, with optional overrides.

        The default is a structurally-valid scalar `uint8` array — the array
        analog of `list()` returning `[]`. Any field can be overridden by keyword
        (the same fields accepted by `update`). Overriding `shape` without
        `chunk_grid` derives a consistent default grid: one regular chunk
        covering the array (`chunk_shape` equal to `shape`, with a length of
        1 for a dimension of length 0, since a chunk length is at least 1:
        https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-grids/regular-grid/index.rst#L40).

        The derivation is deliberately one-way. A user-supplied `chunk_grid`
        is an extension point and is taken verbatim — deriving `shape` from
        it would require interpreting the grid's configuration, which this
        layer never does (and cannot do for unrecognized grid names). So
        overriding `chunk_grid` without `shape` keeps the scalar default
        `shape=()`, which a grid of another rank does not fit: consistency
        between the two is the caller's responsibility, so pass them
        together. So is a fill value for an overridden `data_type`:
        the default `fill_value` is `0`, which a data type whose fill value
        is not an integer -- `bool`, `string`, a complex or struct type --
        refuses, so pass the two together; and so are its codecs: the
        default `bytes` codec has no `endian`, which a data type whose
        values take several bytes needs.
        """
        if "shape" in overrides and "chunk_grid" not in overrides:
            chunk_shape = tuple(max(length, 1) for length in overrides["shape"])
            overrides["chunk_grid"] = ZarrV3NamedConfig(
                name="regular", configuration={"chunk_shape": chunk_shape}
            )
        default = cls(
            shape=(),
            fill_value=0,
            data_type=ZarrV3NamedConfig(name="uint8", configuration={}),
            chunk_grid=ZarrV3NamedConfig(name="regular", configuration={"chunk_shape": ()}),
            codecs=(ZarrV3NamedConfig(name="bytes", configuration={}),),
            chunk_key_encoding=ZarrV3NamedConfig(name="default", configuration={}),
            dimension_names=UNSET,
            attributes={},
            storage_transformers=(),
            extra_fields={},
        )
        return default.update(**overrides)

    def update(self, **kwargs: Unpack[ZarrV3ArrayMetadataPartial]) -> ZarrV3ArrayMetadata:
        """
        Return a new `ZarrV3ArrayMetadata` with the given fields updated.

        Only the constructor-settable fields listed in
        `ZarrV3ArrayMetadataPartial` can be updated; any attempt to update
        other fields (including the fixed `zarr_format` / `node_type`) is
        rejected at the type level. Each given field fully replaces its
        previous value, including `extra_fields`.

        This is useful for test fixtures that want to override a few fields of a
        base template without having to re-specify the entire document.

        No re-validation is performed (`update` is `dataclasses.replace`), so
        a repair or edit can produce an invalid document; validity is checked
        on `from_json`, not on field replacement.
        """
        return dataclasses.replace(self, **kwargs)

    def __post_init__(self) -> None:
        overlap = set(self.extra_fields.keys()).intersection(ARRAY_METADATA_STANDARD_KEYS_V3)
        if overlap:
            raise MetadataValidationError(
                [
                    ValidationProblem(
                        ("extra_fields",),
                        "Extra fields cannot overlap with standard Zarr V3 array metadata fields",
                        "invalid_value",
                    )
                ]
            )

    def to_json(self) -> ZarrV3ArrayMetadataJSON:
        """Return the model as a v3 array document.

        Each extension point is written by `ZarrV3NamedConfig.to_json`, in its
        shortest spelling. `dimension_names` is written unless it is `UNSET`,
        `attributes` and `storage_transformers` only when they are not empty,
        and each entry of `extra_fields` as a top-level member. The document is
        not validated (`to_key_value` validates it) and shares no mutable
        state with the model.
        """
        # to_json output shares no mutable state with the model: every value
        # that can hold a mutable container is deep-copied.
        out: ZarrV3ArrayMetadataJSON = {
            "zarr_format": self.zarr_format,
            "node_type": self.node_type,
            "shape": self.shape,
            "fill_value": copy.deepcopy(self.fill_value),
            "data_type": self.data_type.to_json(),
            "chunk_grid": self.chunk_grid.to_json(),
            "codecs": tuple(codec.to_json() for codec in self.codecs),
            "chunk_key_encoding": self.chunk_key_encoding.to_json(),
        }
        if self.dimension_names is not UNSET:
            out["dimension_names"] = self.dimension_names
        if len(self.attributes) > 0:
            out["attributes"] = copy.deepcopy(self.attributes)
        if len(self.storage_transformers) > 0:
            out["storage_transformers"] = tuple(
                transformer.to_json() for transformer in self.storage_transformers
            )
        # Extra fields are the TypedDict's `extra_items` (PEP 728). Assign them
        # by key rather than `out.update(**...)`: type checkers understand the
        # indexed-write path against `extra_items`, but not the `update(**...)`
        # overload.
        for key, value in self.extra_fields.items():
            out[key] = copy.deepcopy(value)
        return out

    @classmethod
    def from_json(
        cls, data: object, *, context: Context = CORE_AND_EXTENSIONS
    ) -> ZarrV3ArrayMetadata:
        """Return the model of the v3 array document `data`, or raise `MetadataValidationError`.

        `data` is a decoded `zarr.json` document. It is validated as
        `parse_array_metadata_v3` validates it, each extension point read
        through the definition that claims its name in `context`, and every
        problem found is raised together. Each top-level member outside the
        standard keys lands in `extra_fields`. The model shares no mutable
        state with `data`.
        """
        # A read model shares no mutable state with what it read.
        parsed = copy.deepcopy(parse_array_metadata_v3(data, context=context))
        extra_fields: dict[str, ZarrV3ExtensionField] = {
            k: v for k, v in parsed.items() if k not in ARRAY_METADATA_STANDARD_KEYS_V3
        }
        return cls(
            shape=parsed["shape"],
            fill_value=parsed["fill_value"],
            data_type=ZarrV3NamedConfig.from_json(parsed["data_type"]),
            chunk_grid=ZarrV3NamedConfig.from_json(parsed["chunk_grid"]),
            codecs=tuple(ZarrV3NamedConfig.from_json(c) for c in parsed["codecs"]),
            chunk_key_encoding=ZarrV3NamedConfig.from_json(parsed["chunk_key_encoding"]),
            dimension_names=parsed.get("dimension_names", UNSET),
            attributes=dict(parsed.get("attributes", {})),
            storage_transformers=tuple(
                ZarrV3NamedConfig.from_json(t) for t in parsed.get("storage_transformers", ())
            ),
            extra_fields=extra_fields,
        )

    @property
    def must_understand_fields(self) -> dict[str, ZarrV3ExtensionField]:
        """Extra fields the reader is obligated to understand.

        Everything in `extra_fields` not explicitly waived with
        `must_understand: false` (the spec's implicit-true rule, https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1578). A compliant
        reader MUST fail to open the array if this contains any field it does
        not recognize; the model layer only partitions by obligation, since
        recognition is reader-specific.
        """
        return must_understand_subset(self.extra_fields)

    @classmethod
    def from_key_value(
        cls, mapping: Mapping[StoreKey, bytes], *, context: Context = CORE_AND_EXTENSIONS
    ) -> ZarrV3ArrayMetadata:
        """Return the model of the v3 array stored at `zarr.json` in `mapping`, or raise `MetadataValidationError`.

        `mapping` maps store keys to the bytes stored there; other keys are
        ignored. A missing `zarr.json` (a `missing_key` problem), a value that
        is not `bytes` (`invalid_type`), or bytes that do not decode as JSON
        (`invalid_json`) raise, as does every problem `from_json` finds with
        the document in `context`.
        """
        return cls.from_json(
            load_store_json(mapping, ZARR_V3_ARRAY_METADATA_STORE_KEY), context=context
        )

    def to_key_value(
        self, *, indent: int | str | None = None, context: Context = CORE_AND_EXTENSIONS
    ) -> Mapping[ZarrV3ArrayMetadataStoreKey, bytes]:
        """Return the model as the bytes of its `zarr.json` document, or raise `MetadataValidationError`.

        The document `to_json` returns is validated as `from_json` validates
        one in `context`, since a model built or updated by hand is not, and
        every problem found is raised together. `indent` is `json.dumps`'s:
        `None` writes the document on one line, and an int or a string
        pretty-prints it with that indent.
        """
        # A model built by hand is not validated: its document is written only
        # if it reads as `from_json` reads one in `context`, and every problem
        # is raised.
        document = parse_array_metadata_v3(self.to_json(), context=context)
        return {ZARR_V3_ARRAY_METADATA_STORE_KEY: dump_store_json(document, indent=indent)}


class ZarrV2ArrayMetadataPartial(TypedDict, total=False):
    """
    Partial form of the constructor-settable fields of `ZarrV2ArrayMetadata`.

    Every key is optional and typed with the model's own value types, so it
    describes valid keyword arguments to `ZarrV2ArrayMetadata.update` and
    `create_default`. The `init=False` field `zarr_format` is intentionally
    excluded, since it cannot be passed to `dataclasses.replace`.

    Drift between this type and the model's settable fields is prevented by
    `tests/model/test_array.py::test_v2_partial_keys_match_settable_model_fields`.
    """

    shape: tuple[int, ...]
    """The length of each dimension of the array."""
    dtype: ZarrV2DataTypeMetadata
    """The array's data type, as the document spells it.

    A type string such as `"<f8"`, or field records for a structured type.
    Only its shape is checked: whether it names a real data type is not.
    """
    chunks: tuple[int, ...]
    """The length of each dimension of a chunk, one for each dimension of `shape`."""
    fill_value: JSONValue
    """The JSON value of an element no chunk has written, or `None` for no fill value."""
    order: ZarrV2ArrayOrder
    """The layout of elements within a chunk: `"C"` (row-major) or `"F"` (column-major)."""
    compressor: ZarrV2CodecMetadata | None
    """The compression codec's configuration, or `None` for no compressor."""
    filters: tuple[ZarrV2CodecMetadata, ...] | None
    """The filter codecs' configurations, in order, or `None` for no filters.

    `None` and `()` are different spellings, and the model keeps each.
    """
    dimension_separator: ZarrV2ArrayDimensionSeparator
    """The separator between a chunk's coordinates in its store key: `"."` or `"/"`.

    A document without the key means `"."`
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L83-L86);
    the model holds that value and `to_json` writes it.
    """
    attributes: dict[str, JSONValue] | UNSET
    """User attributes, from the sibling `.zattrs` document.

    `UNSET` when there is no `.zattrs`; `{}` is an empty `.zattrs`, which
    `to_key_value` writes.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class ZarrV2ArrayMetadata:
    """In-memory model of a v2 array metadata document.

    A canonical, lossless representation of the spec-defined `.zarray`
    members plus the sibling `.zattrs` attributes; other `.zarray` members,
    which implementations "SHOULD" ignore, are dropped. `dtype`, `compressor`, and `filters` are
    held in their raw JSON forms and are never interpreted; `fill_value` is
    held verbatim in its JSON form. `attributes` is `UNSET` when no
    `.zattrs` file (or merged `attributes` key) exists — distinct from an
    explicit empty `.zattrs`, which is `{}` and round-trips as a file. One
    spelling normalization: a `.zarray` that omits `dimension_separator`
    means `"."` by the v2 convention, and the model holds and re-emits that
    value explicitly.
    """

    zarr_format: Literal[2] = field(default=2, init=False)
    """The Zarr format version, always `2`."""
    shape: tuple[int, ...]
    """The length of each dimension of the array."""
    dtype: ZarrV2DataTypeMetadata
    """The array's data type, as the document spells it.

    A type string such as `"<f8"`, or field records for a structured type.
    Only its shape is checked: whether it names a real data type is not.
    """
    chunks: tuple[int, ...]
    """The length of each dimension of a chunk, one for each dimension of `shape`."""
    fill_value: JSONValue
    """The JSON value of an element no chunk has written, or `None` for no fill value."""
    order: ZarrV2ArrayOrder
    """The layout of elements within a chunk: `"C"` (row-major) or `"F"` (column-major)."""
    compressor: ZarrV2CodecMetadata | None
    """The compression codec's configuration, or `None` for no compressor."""
    filters: tuple[ZarrV2CodecMetadata, ...] | None
    """The filter codecs' configurations, in order, or `None` for no filters.

    `None` and `()` are different spellings, and the model keeps each.
    """
    # "." is the v2 convention's default for an ABSENT dimension_separator key;
    # from_json normalizes absence to it (a semantics-preserving spelling
    # normalization, like the v3 bare-string metadata-field form). The value
    # is never None: the document grammar has no null spelling for this field.
    dimension_separator: ZarrV2ArrayDimensionSeparator = field(default=".")
    """The separator between a chunk's coordinates in its store key: `"."` or `"/"`.

    A document without the key means `"."`
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L83-L86);
    the model holds that value and `to_json` writes it.
    """
    attributes: dict[str, JSONValue] | UNSET
    """User attributes, from the sibling `.zattrs` document.

    `UNSET` when there is no `.zattrs`; `{}` is an empty `.zattrs`, which
    `to_key_value` writes.
    """

    def update(self, **kwargs: Unpack[ZarrV2ArrayMetadataPartial]) -> ZarrV2ArrayMetadata:
        """
        Return a new `ZarrV2ArrayMetadata` with the given fields updated.

        Only the constructor-settable fields listed in
        `ZarrV2ArrayMetadataPartial` can be updated; the fixed `zarr_format` is
        rejected at the type level. Each given field fully replaces its previous
        value.
        """
        return dataclasses.replace(self, **kwargs)

    @classmethod
    def create_default(cls, **overrides: Unpack[ZarrV2ArrayMetadataPartial]) -> ZarrV2ArrayMetadata:
        """
        Create a default (empty) v2 array metadata model, with optional overrides.

        The default is a structurally-valid scalar `uint8` (`"|u1"`) array — the
        array analog of `list()` returning `[]`. Any field can be overridden by
        keyword (the same fields accepted by `update`). Overriding `shape`
        without `chunks` derives `chunks` equal to `shape` (one chunk covering
        the array).

        The derivation is deliberately one-way, matching the v3 model:
        overriding `chunks` without `shape` keeps the scalar default
        `shape=()`, and consistency between the two is the caller's
        responsibility.
        """
        if "shape" in overrides and "chunks" not in overrides:
            overrides["chunks"] = tuple(overrides["shape"])
        default = cls(
            shape=(),
            dtype="|u1",
            chunks=(),
            fill_value=0,
            order="C",
            compressor=None,
            filters=None,
            attributes=UNSET,
        )
        return default.update(**overrides)

    def to_json(self) -> ZarrV2ArrayMetadataJSON:
        """Return the merged in-memory document form.

        `attributes` is included when set (even empty). This is not the
        on-disk `.zarray` content: a conforming `.zarray` must exclude
        `attributes` (they live in the sibling `.zattrs` file). Use
        `to_key_value` to produce the spec-conforming split for storage
        (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L323-L330).
        """
        # to_json output shares no mutable state with the model: every value
        # that can hold a mutable container is deep-copied.
        out: ZarrV2ArrayMetadataJSON = {
            "zarr_format": self.zarr_format,
            "shape": self.shape,
            "dtype": self.dtype,
            "order": self.order,
            "chunks": self.chunks,
            "fill_value": copy.deepcopy(self.fill_value),
            "dimension_separator": self.dimension_separator,
            "compressor": copy.deepcopy(self.compressor),
            "filters": copy.deepcopy(self.filters),
        }
        if self.attributes is not UNSET:
            out["attributes"] = copy.deepcopy(self.attributes)
        return out

    @classmethod
    def from_json(cls, data: object) -> ZarrV2ArrayMetadata:
        """Return the model of the merged v2 array document `data`, or raise `MetadataValidationError`.

        `data` is a decoded `.zarray` document, with the `.zattrs` content
        folded in as an `attributes` member when there is one (the
        `ZarrV2ArrayMetadataJSON` form). It is validated as
        `parse_array_metadata_v2` validates it, structure only, and every
        problem found is raised together. An absent `dimension_separator`
        reads as `"."` and an absent `attributes` as `UNSET`. Any other member
        is checked to be JSON and then dropped: the model does not hold it.
        The model shares no mutable state with `data`.
        """
        # A read model shares no mutable state with what it read.
        parsed = copy.deepcopy(parse_array_metadata_v2(data))
        return cls(
            shape=parsed["shape"],
            dtype=parsed["dtype"],
            chunks=parsed["chunks"],
            fill_value=parsed["fill_value"],
            order=parsed["order"],
            compressor=parsed["compressor"],
            filters=parsed["filters"],
            dimension_separator=parsed.get("dimension_separator", "."),
            attributes=(dict(parsed["attributes"]) if "attributes" in parsed else UNSET),
        )

    @classmethod
    def from_key_value(cls, mapping: Mapping[StoreKey, bytes]) -> ZarrV2ArrayMetadata:
        """Return the model of the v2 array stored at `.zarray` and `.zattrs` in `mapping`, or raise `MetadataValidationError`.

        `mapping` maps store keys to the bytes stored there; other keys are
        ignored. `.zattrs`, when present, becomes `attributes`; when absent,
        `attributes` is `UNSET`. A missing `.zarray` (a `missing_key`
        problem), a value that is not `bytes` (`invalid_type`), bytes that do
        not decode as JSON (`invalid_json`), or an `attributes` member inside
        `.zarray` (`invalid_value`) raise, as does every problem `from_json`
        finds with the merged document.
        """
        zarray_raw = load_store_json(mapping, ZARR_V2_ARRAY_METADATA_STORE_KEY)
        if not isinstance(zarray_raw, Mapping):
            return cls.from_json(zarray_raw)
        zarray = cast("Mapping[str, object]", zarray_raw)
        if "attributes" in zarray:
            raise MetadataValidationError(
                [
                    ValidationProblem(
                        ("attributes",),
                        "unexpected document member",
                        "invalid_value",
                    )
                ]
            )
        if ZARR_V2_ATTRIBUTES_STORE_KEY in mapping:
            zattrs = load_store_json(mapping, ZARR_V2_ATTRIBUTES_STORE_KEY)
            return cls.from_json({**zarray, "attributes": zattrs})
        return cls.from_json(zarray)

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV2ArrayMetadataStoreKey | ZarrV2AttributesStoreKey, bytes]:
        """Return the model as the bytes of its `.zarray` and `.zattrs` documents, or raise `MetadataValidationError`.

        The merged document `to_json` returns is validated as `from_json`
        validates one, since a model built or updated by hand is not, and
        every problem found is raised together. `.zarray` holds every member
        but `attributes`; `.zattrs` holds `attributes`, and is present exactly
        when `attributes` is not `UNSET`, even when it is empty. `indent` is
        `json.dumps`'s: `None` writes each document on one line, and an int or
        a string pretty-prints it with that indent.
        """
        # Attributes live only in the sibling `.zattrs` file; the `.zarray`
        # document must exclude them. The `.zattrs` key is present exactly
        # when attributes are set (even empty) — UNSET emits no file. A model
        # built by hand is not validated: its document is written only if it
        # reads as `from_json` reads one, and every problem is raised.
        document = parse_array_metadata_v2(self.to_json())
        zarray = {k: v for k, v in document.items() if k != "attributes"}
        out: dict[ZarrV2ArrayMetadataStoreKey | ZarrV2AttributesStoreKey, bytes] = {
            ZARR_V2_ARRAY_METADATA_STORE_KEY: dump_store_json(zarray, indent=indent)
        }
        if "attributes" in document:
            out[ZARR_V2_ATTRIBUTES_STORE_KEY] = dump_store_json(
                document["attributes"], indent=indent
            )
        return out
