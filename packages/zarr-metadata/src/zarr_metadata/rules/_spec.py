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
handing each codec its incoming spec. Two properties are deliberate:

**Unknown means stop, not guess.** A codec this package cannot classify
might change the shape, so once one appears every codec after it
receives `None` and rules that need a spec decline. That is the same
policy as the exactly-one-`array->bytes` count, for the same reason: a
silent non-verdict is honest, a guessed one is not.

**Shape stops at the array->bytes boundary.** After the array->bytes
codec there is no array, only bytes, so nothing downstream can have a
shape-dependent configuration; propagation carries `None` past it. The
data type likewise stops mattering, but a few bytes->bytes codecs will
want it in future (checksums do not, compressors do not, but a typed
filter might), so the spec is carried through unchanged rather than
dropped, and the shape alone becomes `None`.

Transitions are registered per codec, next to that codec's rules, via
`spec_transition`. A codec with no registered transition is one this
package models but has not described the effect of — treated as unknown,
so a forgotten transition fails closed rather than silently claiming
"identity". `test_registry.py` asserts every known array->array codec
registers one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from zarr_metadata.v3._extension_points import CODECS, canonical_name
from zarr_metadata.v3._shape import entity_name
from zarr_metadata.v3.codec.kind import codec_kind_of_name

if TYPE_CHECKING:
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """The array a codec receives: its shape and its data type.

    `shape` is None once no array exists (past the array->bytes codec) or
    once it can no longer be determined (past an unknown codec).
    `data_type` is the metadata-field value verbatim — a bare name or a
    name/configuration object — because rules compare it by name.
    """

    shape: tuple[int, ...] | None
    data_type: ZarrV3MetadataFieldJSON | None

    def with_shape(self, shape: tuple[int, ...] | None) -> ArraySpec:
        return replace(self, shape=shape)

    def with_data_type(self, data_type: ZarrV3MetadataFieldJSON | None) -> ArraySpec:
        return replace(self, data_type=data_type)


SpecTransition = Callable[[Mapping[str, object], ArraySpec], ArraySpec]
"""How one codec transforms the spec it receives.

Takes the codec's (shape-valid) configuration and the incoming spec, and
returns the outgoing one. A transition must never raise on the values the
shape validator admits; if it cannot determine the result it returns a
spec with `shape=None`, which downstream rules read as "unknown".
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
) -> Iterator[tuple[int, object, ArraySpec | None]]:
    """Yield `(index, codec, incoming_spec)` for each codec in the chain.

    `incoming_spec` is what that codec receives, or None once propagation
    has stopped: after an unknown codec, after a known codec whose
    configuration is not shape-valid (its transition cannot be trusted),
    or after a codec this package models but has no transition for.
    `configuration_of` resolves a codec entry to its usable configuration
    (`entity_configuration` in practice; injected to keep this module
    free of the registry).
    """
    spec: ArraySpec | None = initial
    for index, codec in enumerate(codecs):
        yield index, codec, spec
        if spec is None:
            continue
        name = entity_name(codec)
        kind = codec_kind_of_name(name) if name is not None else None
        if kind is None:
            spec = None
        elif kind == "array_array":
            transition = _TRANSITIONS.get(canonical_name(CODECS, name or ""))
            configuration = configuration_of(codec)
            if transition is None or configuration is None:
                spec = None
            else:
                spec = transition(configuration, spec)
        else:
            # array->bytes: the array is gone; bytes->bytes: never had one.
            spec = spec.with_shape(None)


def initial_spec(document: Mapping[str, object], chunk_shape: tuple[int, ...] | None) -> ArraySpec:
    """The spec entering a document's top-level codec chain.

    The array a chunk pipeline encodes is one chunk, so the incoming shape
    is the chunk grid's chunk shape (None if the grid is not a regular
    grid this package can read), and the data type is the document's.
    """
    data_type = document.get("data_type")
    if not isinstance(data_type, (str, Mapping)):
        return ArraySpec(chunk_shape, None)
    return ArraySpec(chunk_shape, data_type)  # type: ignore[arg-type]


__all__ = [
    "ArraySpec",
    "SpecTransition",
    "initial_spec",
    "propagate",
    "spec_transition",
    "transitions_registered",
]
