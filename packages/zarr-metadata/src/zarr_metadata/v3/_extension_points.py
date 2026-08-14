"""The v3 extension points, and what this package knows about each.

Zarr v3 defines a handful of *extension points* — document fields whose
value names a codec, data type, chunk grid, or chunk key encoding. Names
are unique only within a point, not across them: `bytes` is both a core
codec and a registered extension data type. Everything here is therefore
keyed by `(field, name)`, mirroring the zarr-extensions registry, whose
directories are the extension points.

Each point records, per identifier, where it was standardized and where
its definition lives; and, per point, two policies the spec sets on the
point as a whole: whether `must_understand: false` is permitted there,
and whether the field holds one entity or a sequence of them.

**Name canonicalization.** Identifiers are keyed by their *canonical*
name, which is the name itself for everything except the parameterized
raw-bytes data type family: `r8`, `r16`, `r24` all canonicalize to
`RAW_BYTES_FAMILY`. Canonicalization is by grammar *shape*, not by
validity — `r12` and `r0` canonicalize into the family too, even though
neither is a legal member. That is deliberate: a malformed member of a
family this package models is a misspelling to report, not an unknown
third-party extension to wave through, and gating canonicalization on
validity would let the misspelling escape judgment entirely.

Canonical names are lookup keys and nothing else. They are never emitted
into metadata and never appear in a message shown to a user: a document
saying `r12` deserves the error "expected 'r' followed by a positive
multiple of 8", not "r<N> is not a valid data type".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final, Literal

from zarr_metadata.v3.chunk_grid.rectilinear import RECTILINEAR_CHUNK_GRID_NAME
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID_NAME
from zarr_metadata.v3.chunk_key_encoding.default import DEFAULT_CHUNK_KEY_ENCODING_NAME
from zarr_metadata.v3.chunk_key_encoding.v2 import V2_CHUNK_KEY_ENCODING_NAME
from zarr_metadata.v3.codec.blosc import BLOSC_CODEC_NAME
from zarr_metadata.v3.codec.bytes import BYTES_CODEC_NAME
from zarr_metadata.v3.codec.cast_value import CAST_VALUE_CODEC_NAME
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC_NAME
from zarr_metadata.v3.codec.gzip import GZIP_CODEC_NAME
from zarr_metadata.v3.codec.scale_offset import SCALE_OFFSET_CODEC_NAME
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEXED_CODEC_NAME
from zarr_metadata.v3.codec.transpose import TRANSPOSE_CODEC_NAME
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC_NAME
from zarr_metadata.v3.data_type.bool import BOOL_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.complex64 import COMPLEX64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.complex128 import COMPLEX128_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.float16 import FLOAT16_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.float32 import FLOAT32_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.float64 import FLOAT64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.int16 import INT16_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.int32 import INT32_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.int64 import INT64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.numpy_datetime64 import NUMPY_DATETIME64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.numpy_timedelta64 import NUMPY_TIMEDELTA64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.raw import RAW_BYTES_NAME_PATTERN
from zarr_metadata.v3.data_type.string import STRING_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.uint8 import UINT8_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.uint16 import UINT16_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.uint32 import UINT32_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.uint64 import UINT64_DATA_TYPE_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

ExtensionPointField = Literal[
    "data_type", "chunk_grid", "chunk_key_encoding", "codecs", "storage_transformers"
]
"""The v3 array metadata fields whose values name an extension."""

DATA_TYPE: Final[ExtensionPointField] = "data_type"
CHUNK_GRID: Final[ExtensionPointField] = "chunk_grid"
CHUNK_KEY_ENCODING: Final[ExtensionPointField] = "chunk_key_encoding"
CODECS: Final[ExtensionPointField] = "codecs"
STORAGE_TRANSFORMERS: Final[ExtensionPointField] = "storage_transformers"

RAW_BYTES_FAMILY: Final = "r<N>"
"""Canonical key for the parameterized raw-bytes data type family.

Spelled as the spec writes the family so that a table dump reads as
documentation. The angle brackets keep it unforgeable by a real name.
"""


class Provenance(Enum):
    """Where an identifier was standardized, and how settled that is."""

    CORE = "core"
    """Defined normatively by the Zarr v3 core specification."""

    REGISTERED = "registered"
    """Defined by a registered extension in the zarr-extensions repository."""

    PROPOSED = "proposed"
    """Defined only by an open proposal; the shape may still change."""


@dataclass(frozen=True, slots=True)
class ExtensionIdentifier:
    """One name an extension point accepts, and where it comes from."""

    name: str
    """The canonical name (see the module docstring on canonicalization)."""

    provenance: Provenance
    reference: str
    """URL of the definition this package models."""


@dataclass(frozen=True, slots=True)
class ExtensionPoint:
    """One v3 extension point and the identifiers this package models for it."""

    field: ExtensionPointField
    identifiers: Mapping[str, ExtensionIdentifier]
    must_understand_false_permitted: bool
    """Whether the spec allows `must_understand: false` at this point.

    False for `data_type`, `chunk_grid`, and `chunk_key_encoding`: an
    implementation that does not recognize one of those cannot proceed
    by ignoring it, so opting out of understanding is meaningless there.
    """

    holds_sequence: bool
    """Whether the field holds a list of entities rather than a single one."""


def _identifiers(*entries: ExtensionIdentifier) -> Mapping[str, ExtensionIdentifier]:
    return {entry.name: entry for entry in entries}


_CORE_DATA_TYPES = "https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html"
_CORE_SPEC = "https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html"
_EXTENSIONS = "https://github.com/zarr-developers/zarr-extensions/tree/main"


def _core_dtype(name: str) -> ExtensionIdentifier:
    return ExtensionIdentifier(name, Provenance.CORE, _CORE_DATA_TYPES)


EXTENSION_POINTS: Final[Mapping[ExtensionPointField, ExtensionPoint]] = {
    DATA_TYPE: ExtensionPoint(
        field=DATA_TYPE,
        identifiers=_identifiers(
            _core_dtype(BOOL_DATA_TYPE_NAME),
            _core_dtype(INT8_DATA_TYPE_NAME),
            _core_dtype(INT16_DATA_TYPE_NAME),
            _core_dtype(INT32_DATA_TYPE_NAME),
            _core_dtype(INT64_DATA_TYPE_NAME),
            _core_dtype(UINT8_DATA_TYPE_NAME),
            _core_dtype(UINT16_DATA_TYPE_NAME),
            _core_dtype(UINT32_DATA_TYPE_NAME),
            _core_dtype(UINT64_DATA_TYPE_NAME),
            _core_dtype(FLOAT16_DATA_TYPE_NAME),
            _core_dtype(FLOAT32_DATA_TYPE_NAME),
            _core_dtype(FLOAT64_DATA_TYPE_NAME),
            _core_dtype(COMPLEX64_DATA_TYPE_NAME),
            _core_dtype(COMPLEX128_DATA_TYPE_NAME),
            ExtensionIdentifier(RAW_BYTES_FAMILY, Provenance.CORE, _CORE_SPEC),
            ExtensionIdentifier(
                BYTES_DATA_TYPE_NAME, Provenance.REGISTERED, f"{_EXTENSIONS}/data-types/bytes"
            ),
            ExtensionIdentifier(
                STRING_DATA_TYPE_NAME, Provenance.REGISTERED, f"{_EXTENSIONS}/data-types/string"
            ),
            ExtensionIdentifier(
                NUMPY_DATETIME64_DATA_TYPE_NAME,
                Provenance.REGISTERED,
                f"{_EXTENSIONS}/data-types/numpy.datetime64",
            ),
            ExtensionIdentifier(
                NUMPY_TIMEDELTA64_DATA_TYPE_NAME,
                Provenance.REGISTERED,
                f"{_EXTENSIONS}/data-types/numpy.timedelta64",
            ),
            ExtensionIdentifier(
                STRUCT_DATA_TYPE_NAME, Provenance.REGISTERED, f"{_EXTENSIONS}/data-types/struct"
            ),
        ),
        must_understand_false_permitted=False,
        holds_sequence=False,
    ),
    CHUNK_GRID: ExtensionPoint(
        field=CHUNK_GRID,
        identifiers=_identifiers(
            ExtensionIdentifier(
                REGULAR_CHUNK_GRID_NAME, Provenance.CORE, f"{_CORE_SPEC}#regular-grids"
            ),
            ExtensionIdentifier(
                RECTILINEAR_CHUNK_GRID_NAME,
                Provenance.REGISTERED,
                f"{_EXTENSIONS}/chunk-grids/rectilinear",
            ),
        ),
        must_understand_false_permitted=False,
        holds_sequence=False,
    ),
    CHUNK_KEY_ENCODING: ExtensionPoint(
        field=CHUNK_KEY_ENCODING,
        identifiers=_identifiers(
            ExtensionIdentifier(
                DEFAULT_CHUNK_KEY_ENCODING_NAME, Provenance.CORE, f"{_CORE_SPEC}#chunk-key-encoding"
            ),
            ExtensionIdentifier(
                V2_CHUNK_KEY_ENCODING_NAME, Provenance.CORE, f"{_CORE_SPEC}#chunk-key-encoding"
            ),
        ),
        must_understand_false_permitted=False,
        holds_sequence=False,
    ),
    CODECS: ExtensionPoint(
        field=CODECS,
        identifiers=_identifiers(
            ExtensionIdentifier(
                BLOSC_CODEC_NAME,
                Provenance.CORE,
                "https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html",
            ),
            ExtensionIdentifier(
                BYTES_CODEC_NAME,
                Provenance.CORE,
                "https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html",
            ),
            ExtensionIdentifier(
                CRC32C_CODEC_NAME,
                Provenance.CORE,
                "https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/index.html",
            ),
            ExtensionIdentifier(
                GZIP_CODEC_NAME,
                Provenance.CORE,
                "https://zarr-specs.readthedocs.io/en/latest/v3/codecs/gzip/index.html",
            ),
            ExtensionIdentifier(
                SHARDING_INDEXED_CODEC_NAME,
                Provenance.CORE,
                "https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html",
            ),
            ExtensionIdentifier(
                TRANSPOSE_CODEC_NAME,
                Provenance.CORE,
                "https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html",
            ),
            ExtensionIdentifier(
                CAST_VALUE_CODEC_NAME, Provenance.REGISTERED, f"{_EXTENSIONS}/codecs/cast_value"
            ),
            ExtensionIdentifier(
                SCALE_OFFSET_CODEC_NAME, Provenance.REGISTERED, f"{_EXTENSIONS}/codecs/scale_offset"
            ),
            # The zstd codec's specification is an open pull request, not
            # merged text: anything typed against it is typed against a draft.
            ExtensionIdentifier(
                ZSTD_CODEC_NAME,
                Provenance.PROPOSED,
                "https://github.com/zarr-developers/zarr-specs/pull/256",
            ),
        ),
        must_understand_false_permitted=True,
        holds_sequence=True,
    ),
    STORAGE_TRANSFORMERS: ExtensionPoint(
        field=STORAGE_TRANSFORMERS,
        # A real extension point that this package models no identifiers
        # for. Recorded explicitly so its emptiness is a stated fact rather
        # than an oversight.
        identifiers=_identifiers(),
        must_understand_false_permitted=True,
        holds_sequence=True,
    ),
}
"""Every v3 extension point, keyed by the document field that carries it."""


def canonical_name(field: ExtensionPointField, name: str) -> str:
    """`name` reduced to the key this package tables it under.

    Identity except for the raw-bytes data type family, which reduces to
    `RAW_BYTES_FAMILY`. By grammar shape, not validity: `r12` and `r0`
    canonicalize too, so a malformed member of a family we model is
    reported as a misspelling rather than mistaken for an unknown
    third-party extension.
    """
    if field == DATA_TYPE and RAW_BYTES_NAME_PATTERN.fullmatch(name) is not None:
        return RAW_BYTES_FAMILY
    return name


def identifier_of(field: ExtensionPointField, name: str) -> ExtensionIdentifier | None:
    """What this package knows about `name` at `field`, or None if nothing.

    None means the name is not one this package models — an unregistered
    or newer extension, which is not an error in itself.
    """
    point = EXTENSION_POINTS.get(field)
    if point is None:
        return None
    return point.identifiers.get(canonical_name(field, name))


def provenance_of(field: ExtensionPointField, name: str) -> Provenance | None:
    """Where `name` at `field` was standardized, or None if not modelled."""
    identifier = identifier_of(field, name)
    return None if identifier is None else identifier.provenance


def identifiers_with(field: ExtensionPointField, provenance: Provenance) -> frozenset[str]:
    """The canonical names at `field` with the given provenance."""
    point = EXTENSION_POINTS.get(field)
    if point is None:
        return frozenset()
    return frozenset(
        name
        for name, identifier in point.identifiers.items()
        if identifier.provenance is provenance
    )


__all__ = [
    "CHUNK_GRID",
    "CHUNK_KEY_ENCODING",
    "CODECS",
    "DATA_TYPE",
    "EXTENSION_POINTS",
    "RAW_BYTES_FAMILY",
    "STORAGE_TRANSFORMERS",
    "ExtensionIdentifier",
    "ExtensionPoint",
    "ExtensionPointField",
    "Provenance",
    "canonical_name",
    "identifier_of",
    "identifiers_with",
    "provenance_of",
]
