"""Array specifications and how a codec chain transforms them.

A codec pipeline is not a list of independent entries: each array->array
codec transforms the array it receives, and every codec's configuration
must be valid against the array that *reaches* it, not against the
document's top-level fields. `transpose` permutes the shape; `cast_value`
changes the data type; a `sharding_indexed` codec that follows either one
sees the transformed array. Rules that read `document["shape"]` or the
chunk grid directly are therefore wrong whenever anything precedes them
in the chain — the sharding divisibility rule accepted an invalid inner
chunk and rejected a valid one as soon as a transpose sat in front of it.

`ArraySpec` is the array a codec receives, and `propagate` walks a chain
handing each codec its incoming spec.

**Unknown is a sentinel, never `None`.** Every field of a spec can be
legitimately `None`-valued or absent in a way that is *not* "unknown":
`fill_value` is `None` for a JSON `null` fill (a real, spec-valid value
for null-permitting data types), and a document with no `data_type` key
at all is a different fact from one whose data type we could not
determine. Overloading `None` for "this package cannot say" would make
those indistinguishable — the same conflation the `UNSET` sentinel exists
to prevent in the model layer. So a field this package cannot determine
is `UNKNOWN`, tested with `is UNKNOWN`, and `None` always means exactly
what it means in the document.

Two policies follow from the layer's guiding principle that a silent
non-verdict is honest and a guessed one is not:

**Unknown means stop, not guess.** A codec this package cannot classify
might change anything, so once one appears every codec after it receives
a fully-`UNKNOWN` spec and rules that need a field decline. Same policy as
the exactly-one-`array->bytes` count.

**Value transformations are downstream's.** This package models structure
and composition, both decidable from JSON. `cast_value` transforms the
fill value by numeric semantics — rounding modes, out-of-range wrap,
scalar maps — and the spec makes a failed fill-value round-trip a MUST
error. Deciding that requires implementing the codec, which is a
different category of work from anything else here and has no bounded
version. So `cast_value`'s transition names its target data type (a
compositional fact it declares in configuration) and sets the fill value
to `UNKNOWN`. Everything downstream that reads the fill value declines,
and the round-trip requirement is documented as belonging to whatever
implements the cast.

Shape stops at the array->bytes boundary: after it there is no array,
only bytes. The data type and fill value carry through, since a typed
bytes->bytes filter could plausibly want them.

Transitions are registered per codec, next to that codec's rules, via
`spec_transition`. A codec with no registered transition is one this
package models but has not described the effect of — treated as unknown,
so a forgotten transition fails closed rather than silently claiming
"identity". A test asserts every known array->array codec registers one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from typing_extensions import Sentinel

from zarr_metadata.v3._extension_points import CODECS, canonical_name
from zarr_metadata.v3._shape import entity_name
from zarr_metadata.v3.codec.kind import codec_kind_of_name

if TYPE_CHECKING:
    from zarr_metadata._common import JSONValue
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON

UNKNOWN = Sentinel("UNKNOWN")
"""A spec field this package cannot determine (PEP 661 sentinel).

Distinct from `None`, which is a real value (a JSON `null` fill value),
and from the model layer's `UNSET`, which means a document key is absent.
Test with `is UNKNOWN`. Same reference-pickling guarantees as `UNSET`;
see `zarr_metadata.model._sentinel`.
"""


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """The array a codec receives.

    Each field is the document's value for it, transformed by every codec
    upstream, or `UNKNOWN` once this package can no longer determine it.
    `data_type` is the metadata-field value verbatim — a bare name or a
    name/configuration object — because rules compare it by name.
    `fill_value` is the JSON scalar verbatim, and may be `None`: that is a
    JSON `null`, a real value, and is never confused with `UNKNOWN`.
    """

    shape: tuple[int, ...] | UNKNOWN
    data_type: ZarrV3MetadataFieldJSON | UNKNOWN
    fill_value: JSONValue | UNKNOWN

    def with_shape(self, shape: tuple[int, ...] | UNKNOWN) -> ArraySpec:
        return replace(self, shape=shape)

    def with_data_type(self, data_type: ZarrV3MetadataFieldJSON | UNKNOWN) -> ArraySpec:
        return replace(self, data_type=data_type)

    def with_fill_value(self, fill_value: JSONValue | UNKNOWN) -> ArraySpec:
        return replace(self, fill_value=fill_value)


NOTHING_KNOWN: Final = ArraySpec(UNKNOWN, UNKNOWN, UNKNOWN)
"""The spec past a point where nothing about the array can be determined.

Compare by equality, not identity: a spec can arrive here field by field
(shape lost at the array->bytes boundary, fill value lost at a cast) and
is then equal to this constant without being it.
"""


SpecTransition = Callable[[Mapping[str, object], ArraySpec], ArraySpec]
"""How one codec transforms the spec it receives.

Takes the codec's (shape-valid) configuration and the incoming spec, and
returns the outgoing one. A transition must never raise on the values the
shape validator admits; a field it cannot determine becomes `UNKNOWN`.
"""

_TRANSITIONS: Final[dict[str, SpecTransition]] = {}


def spec_transition(codec: str) -> Callable[[SpecTransition], SpecTransition]:
    """Register how `codec` transforms an incoming `ArraySpec`.

    Only array->array codecs need one: array->bytes and bytes->bytes
    codecs end shape propagation by construction. Registering a
    transition for a codec of another kind is refused, since it could
    never be consulted.
    """
    kind = codec_kind_of_name(codec)
    if kind != "array_array":
        msg = (
            f"spec transition registered for {codec!r}, which is "
            f"{kind or 'unknown'} rather than array_array; only array->array "
            "codecs transform the array spec"
        )
        raise ValueError(msg)

    def decorate(transition: SpecTransition) -> SpecTransition:
        _TRANSITIONS[canonical_name(CODECS, codec)] = transition
        return transition

    return decorate


def transitions_registered() -> frozenset[str]:
    """Every codec name with a registered spec transition."""
    return frozenset(_TRANSITIONS)


def propagate(
    codecs: Sequence[object],
    initial: ArraySpec,
    configuration_of: Callable[[object], Mapping[str, object] | None],
) -> Iterator[tuple[int, object, ArraySpec]]:
    """Yield `(index, codec, incoming_spec)` for each codec in the chain.

    `incoming_spec` is what that codec receives. It is `NOTHING_KNOWN`
    once propagation has stopped: after an unknown codec, after a known
    codec whose configuration is not shape-valid (its transition cannot
    be trusted), or after a codec this package models but has no
    transition for. `configuration_of` resolves a codec entry to its
    usable configuration (`entity_configuration` in practice; injected to
    keep this module free of the registry).
    """
    spec = initial
    for index, codec in enumerate(codecs):
        yield index, codec, spec
        if spec == NOTHING_KNOWN:
            continue
        name = entity_name(codec)
        kind = codec_kind_of_name(name) if name is not None else None
        if kind is None:
            spec = NOTHING_KNOWN
        elif kind == "array_array":
            transition = _TRANSITIONS.get(canonical_name(CODECS, name or ""))
            configuration = configuration_of(codec)
            if transition is None or configuration is None:
                spec = NOTHING_KNOWN
            else:
                spec = transition(configuration, spec)
        else:
            # array->bytes: the array is gone; bytes->bytes: never had one.
            spec = spec.with_shape(UNKNOWN)


def initial_spec(document: Mapping[str, object], chunk_shape: tuple[int, ...] | None) -> ArraySpec:
    """The spec entering a document's top-level codec chain.

    The array a chunk pipeline encodes is one chunk, so the incoming shape
    is the chunk grid's chunk shape (`UNKNOWN` if the grid is not a
    regular grid this package can read). The data type and fill value are
    the document's own; a missing key is `UNKNOWN`, while a present
    `fill_value: null` is `None`, faithfully.
    """
    data_type = document.get("data_type", UNKNOWN)
    if data_type is not UNKNOWN and not isinstance(data_type, (str, Mapping)):
        data_type = UNKNOWN
    # .get with the sentinel default keeps null and absent apart: a present
    # `fill_value: null` returns None, only a missing key returns UNKNOWN.
    fill_value = document.get("fill_value", UNKNOWN)
    return ArraySpec(
        UNKNOWN if chunk_shape is None else chunk_shape,
        data_type,  # type: ignore[arg-type]
        fill_value,  # type: ignore[arg-type]
    )


__all__ = [
    "NOTHING_KNOWN",
    "UNKNOWN",
    "ArraySpec",
    "SpecTransition",
    "initial_spec",
    "propagate",
    "spec_transition",
    "transitions_registered",
]
