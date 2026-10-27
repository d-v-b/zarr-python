"""Drawing fields from `Strategies`: alone, inside a configuration, and as a codec pipeline.

What a configuration that holds other fields draws them with, and what
the public strategies are built from: a field in an envelope the spec
allows, a data type field of any definition the strategies hold, and a
pipeline whose each codec is drawn for the chunk the one before it hands
on. Each drawn field is read in the scope of the strategies' own
definitions, so a nested field reads as what it was drawn for.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, cast

from hypothesis import reject
from hypothesis import strategies as st

from zarr_metadata.testing._strategy import NOTHING_KNOWN
from zarr_metadata.v3._definition import (
    CodecDefinition,
    DataTypeDefinition,
    carrying_name,
    no_pipelines,
    resolve,
)
from zarr_metadata.v3._registry import CORE, Context

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from hypothesis.strategies import SearchStrategy

    from zarr_metadata._common import JSONValue
    from zarr_metadata.testing._strategy import (
        CodecStrategy,
        Configurations,
        DataTypeStrategy,
        DefinitionStrategy,
        Strategies,
    )
    from zarr_metadata.v3._definition import (
        Chunk,
        CodecKind,
        CodecSize,
        D,
        Definition,
        Resolved,
    )
    from zarr_metadata.v3.definition import ZarrV3MetadataFieldJSON

_MAX_ARRAY_ARRAY: Final = 2
"""The most array -> array codecs a pipeline is drawn with."""

_MAX_BYTES_BYTES: Final = 2
"""The most bytes -> bytes codecs a pipeline is drawn with."""

EVERY_SIZE: Final[frozenset[CodecSize]] = frozenset({"static", "dynamic"})
"""Codecs of any size: what a pipeline but a shard's index takes."""


class _Brief(Mapping["Definition[Any]", "DefinitionStrategy"]):
    """`Strategies` whose repr says how many definitions they hold, not what each one is.

    Strategies are handed to the composite strategies that draw nested
    fields, and Hypothesis writes the repr of a strategy's arguments into
    the note on any error raised while drawing from it -- among them the
    rejections it raises itself, as `floats(width=16)` does -- warning
    when that repr runs past 30,000 characters. Every definition's
    strategy, in full, runs past that, which fails a test suite that
    turns warnings into errors. It also holds the scope drawn fields are
    read in, built once rather than for every field read.
    """

    __slots__ = ("_strategies", "scope")

    def __init__(self, strategies: Strategies) -> None:
        self._strategies = strategies
        self.scope = Context.of(*strategies)

    def __getitem__(self, definition: Definition[Any]) -> DefinitionStrategy:
        return self._strategies[definition]

    def __iter__(self) -> Iterator[Definition[Any]]:
        return iter(self._strategies)

    def __len__(self) -> int:
        return len(self._strategies)

    def __repr__(self) -> str:
        return f"<strategies for {len(self)} definitions>"


def briefly(strategies: Strategies) -> _Brief:
    """`strategies`, with a repr short enough to go in Hypothesis's notes."""
    return strategies if isinstance(strategies, _Brief) else _Brief(strategies)


def strategy_for(definition: Definition[Any], strategies: Strategies) -> DefinitionStrategy:
    """The strategy `strategies` holds for `definition`, which draws configurations of its kind."""
    found = strategies.get(definition)
    if found is None:
        msg = f"no strategy for {definition.name!r}; pass one in `strategies`"
        raise LookupError(msg)
    if not isinstance(definition, found.kind):
        msg = (
            f"the strategy for {definition.name!r}, a {type(definition).__name__}, draws "
            f"configurations of a {found.kind.__name__}: {found!r}"
        )
        raise TypeError(msg)
    return found


def scope_of(strategies: Strategies) -> Context:
    """The scope a drawn field is read in: every definition the strategies draw for."""
    return briefly(strategies).scope


def read(field: object, kind: type[D], strategies: Strategies) -> Resolved[D]:
    """`field`, drawn from `strategies` or held by a configuration handed to one, read in their scope; a `ValueError` if it is refused.

    A strategy draws what its definition accepts, so a refused field is a
    fault in a strategy or in the configuration it was handed, not
    something to draw again.
    """
    resolved, problems = resolve(field, kind, scope_of(strategies))
    if len(problems) != 0:
        msg = f"{field!r}, read in the strategies' scope, is refused: {problems!r}"
        raise ValueError(msg)
    return resolved


def _written_bare(definition: Definition[Any]) -> bool:
    """Whether the spec has the field written as its bare name: a data type it defines itself.

    "If the data type is defined in this specification, then the value
    must be the data type identifier provided as a string"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L509-L512).
    """
    if not isinstance(definition, DataTypeDefinition):
        return False
    return CORE.claimant(DataTypeDefinition, definition.name) is definition


@st.composite
def any_spelling(
    draw: st.DrawFn, definition: Definition[Any], configurations: SearchStrategy[Any]
) -> ZarrV3MetadataFieldJSON:
    """A configuration of `definition` in one of the envelopes the spec allows it.

    A configuration with nothing in it is written as the bare name, an
    object without one, or an object with an empty one; any object may
    say `must_understand` is `true`, which is what absence means. A name
    that carries the configuration, as `r16` does, and a data type the
    spec defines itself, are written as the bare name.
    """
    configuration = cast("Mapping[str, JSONValue]", draw(configurations))
    carried = carrying_name(definition, configuration)
    if carried is not None:
        return carried
    if _written_bare(definition):
        return definition.name
    empty = len(configuration) == 0
    if empty and draw(st.booleans()):
        return definition.name
    field: dict[str, JSONValue] = {"name": definition.name}
    if not empty or draw(st.booleans()):
        field["configuration"] = configuration
    if draw(st.booleans()):
        field["must_understand"] = True
    return cast("ZarrV3MetadataFieldJSON", field)


def data_types(strategies: Strategies) -> list[DataTypeDefinition[Any]]:
    """The data types `strategies` draw for, by name."""
    return sorted(
        (d for d in strategies if isinstance(d, DataTypeDefinition)), key=lambda d: d.name
    )


@st.composite
def data_type_fields(
    draw: st.DrawFn, among: Sequence[DataTypeDefinition[Any]], strategies: Strategies
) -> ZarrV3MetadataFieldJSON:
    """A data type field, of one of the definitions `among`, drawn from `strategies`."""
    definition = draw(st.sampled_from(among))
    found = cast("DataTypeStrategy", strategy_for(definition, strategies))
    return draw(any_spelling(definition, found.configurations(strategies)))


def fill_values_of(
    data_type: Resolved[DataTypeDefinition[Any]], strategies: Strategies
) -> SearchStrategy[JSONValue]:
    """Fill values of `data_type`, a data type field read in the strategies' scope; a `LookupError` if no definition in it read the field."""
    definition, configuration = data_type.definition, data_type.configuration
    if definition is None or configuration is None:
        msg = f"no strategy draws fill values of {data_type.json!r}, which no definition in `strategies` reads"
        raise LookupError(msg)
    found = cast("DataTypeStrategy", strategy_for(definition, strategies))
    return found.fill_values(configuration, strategies)


def _taking(
    chunk: Chunk,
    kind: CodecKind,
    sizes: frozenset[CodecSize],
    strategies: Strategies,
    *,
    holding: bool = True,
) -> list[tuple[CodecDefinition[Any], Configurations]]:
    """Each codec of `kind` and one of `sizes` that `strategies` draw for, that takes `chunk`, with its configurations.

    Without `holding`, a codec that holds pipelines of its own is left
    out: what one of those takes depends on what its pipelines do.
    """
    taking: list[tuple[CodecDefinition[Any], Configurations]] = []
    for definition in sorted(strategies, key=lambda d: d.name):
        if not isinstance(definition, CodecDefinition) or definition.kind != kind:
            continue
        if definition.size not in sizes or (not holding and _holds_pipelines(definition)):
            continue
        found = cast("CodecStrategy", strategy_for(definition, strategies))
        configurations = found.configurations(chunk, strategies)
        if configurations is not None:
            taking.append((definition, configurations))
    return taking


def _holds_pipelines(definition: CodecDefinition[Any]) -> bool:
    return definition.pipelines is not no_pipelines


def pipelines_for(
    chunk: Chunk, strategies: Strategies, sizes: frozenset[CodecSize] = EVERY_SIZE
) -> SearchStrategy[tuple[ZarrV3MetadataFieldJSON, ...]] | None:
    """Pipelines handed `chunk`, of codecs of one of `sizes`; None when no array -> bytes codec that holds no pipelines takes the chunk as it is.

    One that does is enough for a pipeline: one of only that codec. One
    that holds pipelines of its own takes a chunk only if what it holds
    does, so it settles nothing.
    """
    if len(_taking(chunk, "array_bytes", sizes, strategies, holding=False)) == 0:
        return None
    return _walk(chunk, strategies, sizes)


@st.composite
def _codec(
    draw: st.DrawFn, options: Sequence[tuple[CodecDefinition[Any], Configurations]]
) -> tuple[CodecDefinition[Any], ZarrV3MetadataFieldJSON]:
    definition, configurations = draw(st.sampled_from(options))
    return definition, draw(any_spelling(definition, configurations))


@st.composite
def _walk(
    draw: st.DrawFn, chunk: Chunk, strategies: Strategies, sizes: frozenset[CodecSize]
) -> tuple[ZarrV3MetadataFieldJSON, ...]:
    """Codecs for a pipeline handed `chunk`, each drawn for the chunk the one before it hands on.

    Array -> array codecs, then one array -> bytes codec, then bytes ->
    bytes codecs, each of one of `sizes`. What an array -> array codec
    hands on is what its definition's `transition` says of the codec as
    drawn: a `cast_value` hands the next codec the data type it casts to.
    """
    codecs: list[ZarrV3MetadataFieldJSON] = []
    handed = chunk
    for _ in range(draw(st.integers(0, _MAX_ARRAY_ARRAY))):
        options = _taking(handed, "array_array", sizes, strategies)
        if len(options) == 0:
            break
        definition, field = draw(_codec(options))
        codec = read(field, CodecDefinition, strategies)
        assert codec.configuration is not None
        handed = definition.transition(codec.configuration, codec.nested, handed)
        codecs.append(field)
    options = _taking(handed, "array_bytes", sizes, strategies)
    if len(options) == 0:
        reject()  # what the array -> array codecs handed on, no array -> bytes codec takes
    codecs.append(draw(_codec(options))[1])
    options = _taking(NOTHING_KNOWN, "bytes_bytes", sizes, strategies)
    if len(options) != 0:
        count = draw(st.integers(0, _MAX_BYTES_BYTES))
        codecs.extend(draw(_codec(options))[1] for _ in range(count))
    return tuple(codecs)


__all__ = [
    "EVERY_SIZE",
    "any_spelling",
    "briefly",
    "data_type_fields",
    "data_types",
    "fill_values_of",
    "pipelines_for",
    "read",
    "scope_of",
    "strategy_for",
]
