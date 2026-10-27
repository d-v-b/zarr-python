"""The strategies against the definitions they draw for, and against what they ought to reach.

Every draw is read in `CORE_AND_EXTENSIONS` and must read without a
problem, by the definition it was drawn for: the field alone, a fill
value beside its data type, a chunk grid over the shape it was drawn
for, a codec handed the chunk it was drawn for, and a pipeline handed
the chunks of a grid, as `read_pipeline` reads one. A strategy that only
drew valid metadata could draw almost none, so reach is checked too,
with targets worked out from each configuration's TypedDict rather than
from the strategies: every optional key present and absent, every
`Literal` value, and for a definition with a simplest spelling, a
configuration it simplifies.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, get_args, get_origin

import pytest
from hypothesis import event, find, given, settings
from hypothesis import strategies as st

from zarr_metadata.testing.strategies import (
    DEFAULT_STRATEGIES,
    ChunkKeyEncodingStrategy,
    CodecStrategy,
    DataTypeStrategy,
    Spellings,
    Strategies,
    chunk_grid_configurations,
    codec_configurations,
    data_type_configurations,
    fields,
    fill_values,
    pipelines,
)
from zarr_metadata.typed_json import typeddict_keys
from zarr_metadata.v3._definition import as_kind, kind_of, no_pipelines, storage_of, unchanged
from zarr_metadata.v3._pipeline import read_pipeline
from zarr_metadata.v3.chunk_grid.rectilinear import RECTILINEAR_CHUNK_GRID
from zarr_metadata.v3.codec._arithmetic import COMPLEX, NOT_NUMBERS
from zarr_metadata.v3.codec.cast_value import CAST_VALUE_CODEC
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.codec.scale_offset import SCALE_OFFSET_CODEC
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEXED_CODEC
from zarr_metadata.v3.data_type._float import FloatSpecialFillValue
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.complex64 import COMPLEX64_DATA_TYPE
from zarr_metadata.v3.data_type.complex128 import COMPLEX128_DATA_TYPE
from zarr_metadata.v3.data_type.float16 import FLOAT16_DATA_TYPE
from zarr_metadata.v3.data_type.float32 import FLOAT32_DATA_TYPE
from zarr_metadata.v3.data_type.float64 import FLOAT64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_datetime64 import NUMPY_DATETIME64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_timedelta64 import NUMPY_TIMEDELTA64_DATA_TYPE
from zarr_metadata.v3.data_type.string import STRING_DATA_TYPE
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    Chunk,
    ChunkGridDefinition,
    CodecDefinition,
    DataTypeDefinition,
    Definition,
    ZarrV3MetadataFieldJSON,
    canonicalize,
    chunk_grid_lengths,
    fill_value_problems,
    resolve,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata._common import JSONValue

_IN_SCOPE = frozenset(
    definition for table in CORE_AND_EXTENSIONS.tables.values() for definition in table.values()
)
_CORE_DATA_TYPES = frozenset(CORE.tables[DataTypeDefinition].values())

_DEFINITIONS = sorted(DEFAULT_STRATEGIES, key=lambda definition: definition.name)
_DATA_TYPES = [d for d in _DEFINITIONS if isinstance(d, DataTypeDefinition)]
_CHUNK_GRIDS = [d for d in _DEFINITIONS if isinstance(d, ChunkGridDefinition)]
_CODECS = [d for d in _DEFINITIONS if isinstance(d, CodecDefinition)]

_OPEN_FOR_SCALE_OFFSET = frozenset(
    {"complex64", "complex128", "numpy.datetime64", "numpy.timedelta64"}
)
"""Data types `scale_offset` is drawn for none of, though its rules refuse none: its registry entry leaves them open."""

T = TypeVar("T")

_SHAPES = st.lists(st.integers(0, 12), max_size=4).map(tuple)


def _id(value: object) -> str:
    if isinstance(value, Definition):
        return f"{type(value).__name__}:{value.name}"
    return str(value)


def _read(definition: Definition[Any], field: object) -> Any:  # noqa: ANN401
    """`field` read in `CORE_AND_EXTENSIONS`, asserted to read without a problem by `definition`."""
    resolved, problems = resolve(field, as_kind(kind_of(definition)), CORE_AND_EXTENSIONS)
    assert problems == ()
    assert resolved.resolution == "read"
    assert resolved.definition is definition
    return resolved


def test_every_definition_has_a_strategy() -> None:
    """Each definition in scope has a strategy, and none is out of scope."""
    assert frozenset(DEFAULT_STRATEGIES) == _IN_SCOPE


@pytest.mark.parametrize("spellings", get_args(Spellings))
@pytest.mark.parametrize("definition", _DEFINITIONS, ids=_id)
@given(data=st.data())
def test_fields_read(
    definition: Definition[Any], spellings: Spellings, data: st.DataObject
) -> None:
    """A drawn field reads by its definition, through JSON too; a canonical one is its own simplest spelling."""
    field = data.draw(fields(definition, spellings=spellings))
    _read(definition, field)
    _read(definition, json.loads(json.dumps(field)))
    if definition in _CORE_DATA_TYPES:
        assert isinstance(field, str)
    if spellings == "canonical":
        canonical, _ = canonicalize(field, as_kind(kind_of(definition)), CORE_AND_EXTENSIONS)
        assert canonical == field


@pytest.mark.parametrize("definition", _DATA_TYPES, ids=_id)
@given(data=st.data())
def test_fill_values_fit_their_data_type(
    definition: DataTypeDefinition[Any], data: st.DataObject
) -> None:
    configuration = data.draw(data_type_configurations(definition))
    data_type = _read(definition, data.draw(fields(definition, st.just(configuration))))
    value = data.draw(fill_values(definition, configuration))
    assert fill_value_problems(data_type, value) == ()
    assert fill_value_problems(data_type, json.loads(json.dumps(value))) == ()


@pytest.mark.parametrize("definition", _CHUNK_GRIDS, ids=_id)
@given(data=st.data(), shape=_SHAPES)
def test_chunk_grids_fit_their_shape(
    definition: ChunkGridDefinition[Any], data: st.DataObject, shape: tuple[int, ...]
) -> None:
    configurations = chunk_grid_configurations(definition, shape)
    chunk_grid = _read(definition, data.draw(fields(definition, configurations)))
    lengths, problems = chunk_grid_lengths(chunk_grid, shape)
    assert problems == ()
    # "Chunk sizes must be greater than zero", which the regular grid's rules do not yet refuse.
    assert all(0 not in axis for axis in lengths if axis is not None)


@st.composite
def _chunks(draw: st.DrawFn) -> Chunk:
    """The chunks of a grid drawn over an array of some shape, of a data type drawn from the strategies.

    A rectilinear grid's chunks take several lengths along an axis, which
    a codec drawn for them must take each of.
    """
    shape = draw(_SHAPES)
    grid = draw(st.sampled_from(_CHUNK_GRIDS))
    chunk_grid = _read(grid, draw(fields(grid, chunk_grid_configurations(grid, shape))))
    lengths, _ = chunk_grid_lengths(chunk_grid, shape)
    definition = draw(st.sampled_from(_DATA_TYPES))
    return Chunk(lengths, _read(definition, draw(fields(definition))))


def _handed_problems(codec: Any, chunk: Chunk) -> list[object]:  # noqa: ANN401
    """What is wrong with `codec`, a codec field read, handed `chunk`: by its chunk rules, and in each pipeline it holds."""
    definition = codec.definition
    problems: list[object] = list(definition.chunk_rules(codec.configuration, codec.nested, chunk))
    handed = definition.pipelines(codec.configuration, codec.nested, chunk)
    for member, inner in handed.items():
        held = [codec.nested[(member, index)] for index in range(len(codec.configuration[member]))]
        problems.extend(read_pipeline(held, inner)[1])
    return problems


def _holds_pipelines(definition: CodecDefinition[Any]) -> bool:
    return definition.pipelines is not no_pipelines


def _encoders_refuse(data: st.DataObject, chunk: Chunk) -> bool:
    """Whether every array -> bytes codec that holds no pipelines refuses a chunk of `chunk`'s data type."""
    of_data_type = Chunk(None, chunk.data_type)
    encoders = [d for d in _CODECS if d.kind == "array_bytes" and not _holds_pipelines(d)]
    for encoder in encoders:
        codec = _read(encoder, data.draw(fields(encoder)))
        if list(encoder.chunk_rules(codec.configuration, codec.nested, of_data_type)) == []:
            return False
    return True


@pytest.mark.parametrize("definition", _CODECS, ids=_id)
@given(data=st.data(), chunk=_chunks())
def test_codecs_take_their_chunk(
    definition: CodecDefinition[Any], data: st.DataObject, chunk: Chunk
) -> None:
    """A codec drawn for a chunk takes it; one drawn for none is one the spec does not give it.

    A shard drawn for none is one whose inner chunks no codec encodes.
    """
    assert chunk.data_type is not None
    assert chunk.data_type.definition is not None
    data_type_name = chunk.data_type.definition.name
    configurations = codec_configurations(definition, chunk)
    if configurations is None:
        event(f"takes no {data_type_name}")
        if definition is SCALE_OFFSET_CODEC and data_type_name in _OPEN_FOR_SCALE_OFFSET:
            return
        if _holds_pipelines(definition):
            assert _encoders_refuse(data, chunk)
            return
        codec = _read(definition, data.draw(fields(definition)))
        assert list(definition.chunk_rules(codec.configuration, codec.nested, chunk)) != []
        return
    codec = _read(definition, data.draw(fields(definition, configurations)))
    assert _handed_problems(codec, chunk) == []


_MIXED_LENGTHS = st.lists(
    st.frozensets(st.integers(1, 24), min_size=1, max_size=3), max_size=3
).map(tuple)
"""Several lengths along each axis, as a rectilinear grid gives: one inner length must divide them all."""


@given(data=st.data(), lengths=_MIXED_LENGTHS)
def test_shards_divide_every_length(
    data: st.DataObject, lengths: tuple[frozenset[int], ...]
) -> None:
    chunk = Chunk(lengths)
    configurations = codec_configurations(SHARDING_INDEXED_CODEC, chunk)
    assert configurations is not None
    codec = _read(SHARDING_INDEXED_CODEC, data.draw(fields(SHARDING_INDEXED_CODEC, configurations)))
    assert _handed_problems(codec, chunk) == []


@given(data=st.data(), chunk=_chunks())
def test_pipelines_read(data: st.DataObject, chunk: Chunk) -> None:
    """A pipeline drawn for a chunk reads as one handed it, through JSON too; none is drawn only for a chunk no codec encodes."""
    drawn = pipelines(chunk)
    if drawn is None:
        assert chunk.data_type is not None
        event(f"no pipeline for {chunk.data_type.json!r}")
        assert storage_of(chunk.data_type) == "variable_length"
        return
    pipeline = data.draw(drawn)
    for written in (pipeline, json.loads(json.dumps(pipeline))):
        codecs = [resolve(field, CodecDefinition, CORE_AND_EXTENSIONS) for field in written]
        assert [problems for _, problems in codecs] == [()] * len(codecs)
        _, problems = read_pipeline([codec for codec, _ in codecs], chunk)
        assert problems == ()


Target = tuple[Literal["present", "absent"], str] | tuple[Literal["value"], str, object]
FieldForm = Literal["bare", "object", "configured", "must_understand"]


def _configuration_targets(
    definition: Definition[Any],
) -> list[Target | tuple[Literal["simplified"]]]:
    """What a strategy for `definition` ought to reach, by its configuration's TypedDict."""
    targets: list[Target | tuple[Literal["simplified"]]] = []
    for key, (annotation, required) in typeddict_keys(definition.configuration).members.items():
        if not required:
            targets.extend([("present", key), ("absent", key)])
        if get_origin(annotation) is Literal:
            targets.extend(("value", key, value) for value in get_args(annotation))
    if definition.canonical is not unchanged:
        targets.append(("simplified",))
    return targets


def _field_forms(definition: Definition[Any]) -> list[FieldForm]:
    """The envelopes the spec allows a field of `definition` to be written in."""
    if definition in _CORE_DATA_TYPES:
        return []  # only ever bare, which `test_fields_read` asserts
    if definition.requires_configuration:
        return ["configured", "must_understand"]
    return ["bare", "object", "configured", "must_understand"]


def _form_of(field: ZarrV3MetadataFieldJSON) -> set[FieldForm]:
    if isinstance(field, str):
        return {"bare"}
    forms: set[FieldForm] = {"configured" if "configuration" in field else "object"}
    if "must_understand" in field:
        forms.add("must_understand")
    return forms


def _hits(
    definition: Definition[Any],
    target: Target | tuple[Literal["simplified"]],
    configuration: Mapping[str, JSONValue],
) -> bool:
    match target:
        case ("present", key):
            return key in configuration
        case ("absent", key):
            return key not in configuration
        case ("value", key, value):
            return configuration.get(key) == value
        case ("simplified",):
            return definition.canonical(configuration) != configuration


_REACH = settings(max_examples=2000, database=None)
_ONCE = settings(database=None)


@pytest.mark.parametrize(
    ("definition", "target"),
    [(d, target) for d in _DEFINITIONS for target in _configuration_targets(d)],
    ids=_id,
)
def test_configurations_reach(
    definition: Definition[Any], target: Target | tuple[Literal["simplified"]]
) -> None:
    configurations = fields(definition).map(lambda field: _read(definition, field).configuration)
    find(
        configurations,
        lambda configuration: _hits(definition, target, configuration),
        settings=_REACH,
    )


@pytest.mark.parametrize(
    ("definition", "form"),
    [(d, form) for d in _DEFINITIONS for form in _field_forms(d)],
    ids=_id,
)
def test_fields_reach(definition: Definition[Any], form: FieldForm) -> None:
    find(fields(definition), lambda field: form in _form_of(field), settings=_REACH)


def test_an_added_strategy_is_drawn_from() -> None:
    """A strategy joined to the defaults, as an extension author adds one, is the one drawn from."""
    level_three = CodecStrategy(configurations=lambda _chunk, _strategies: st.just({"level": 3}))
    strategies = DEFAULT_STRATEGIES | {GZIP_CODEC: level_three}
    field = find(fields(GZIP_CODEC, strategies=strategies), lambda _field: True, settings=_REACH)
    assert _read(GZIP_CODEC, field).configuration == {"level": 3}


_SHORT_REPR = 10_000
"""Well short of the 30,000 characters past which Hypothesis warns about a strategy's repr.

It writes that repr into the note on any error raised while drawing,
its own rejections among them, so a longer one fails a suite that turns
warnings into errors.
"""


@pytest.mark.parametrize("definition", _DEFINITIONS, ids=_id)
def test_reprs_stay_short(definition: Definition[Any]) -> None:
    """However the strategies are passed: a plain mapping is as long as every strategy in it."""
    plain: Strategies = dict(DEFAULT_STRATEGIES)
    assert len(repr(fields(definition, strategies=plain))) < _SHORT_REPR
    assert len(repr(pipelines(strategies=plain))) < _SHORT_REPR


def test_a_definition_without_a_strategy_is_a_lookup_error() -> None:
    with pytest.raises(LookupError, match="no strategy for 'gzip'"):
        fields(GZIP_CODEC, strategies={})


def test_a_strategy_of_another_kind_is_a_type_error() -> None:
    strategies: Strategies = {
        GZIP_CODEC: ChunkKeyEncodingStrategy(configurations=lambda _strategies: st.just({}))
    }
    with pytest.raises(TypeError, match="draws configurations of a ChunkKeyEncodingDefinition"):
        codec_configurations(GZIP_CODEC, strategies=strategies)


def test_a_codec_strategy_refusing_an_unknown_chunk_is_a_value_error() -> None:
    refusing = CodecStrategy(configurations=lambda _chunk, _strategies: None)
    with pytest.raises(ValueError, match=r"Chunk\(\), a chunk nothing is known of"):
        fields(GZIP_CODEC, strategies=DEFAULT_STRATEGIES | {GZIP_CODEC: refusing})


def test_a_refused_nested_draw_is_a_value_error() -> None:
    """A field a configuration holds is read as it is drawn, so a strategy that draws a refused one is caught there."""
    refused = DataTypeStrategy(
        configurations=lambda _strategies: st.just({"stray": 1}),
        fill_values=lambda _configuration, _strategies: st.just(0.0),
    )
    strategies: Strategies = {
        CAST_VALUE_CODEC: DEFAULT_STRATEGIES[CAST_VALUE_CODEC],
        NUMPY_DATETIME64_DATA_TYPE: refused,
    }
    with pytest.raises(ValueError, match="which is refused"):
        find(fields(CAST_VALUE_CODEC, strategies=strategies), lambda _field: True, settings=_ONCE)


def test_a_refused_draw_in_the_canonical_spelling_is_a_value_error() -> None:
    refused = fields(GZIP_CODEC, st.just({"level": 99}), spellings="canonical")
    with pytest.raises(ValueError, match="which is refused"):
        find(refused, lambda _field: True, settings=_ONCE)


def _has_pair(configuration: Mapping[str, Any]) -> bool:
    return any(
        isinstance(entry, tuple)
        for dimension in configuration["chunk_shapes"]
        if isinstance(dimension, tuple)
        for entry in cast("tuple[object, ...]", dimension)
    )


def test_rectilinear_edges_reach_a_run_written_as_a_pair() -> None:
    """The `[size, count]` spelling, which no TypedDict target asks for: it is one arm of a union."""
    find(chunk_grid_configurations(RECTILINEAR_CHUNK_GRID), _has_pair, settings=_REACH)


FillValueSpelling = Literal[
    "number", "named", "hex lowercase", "hex uppercase", "array", "base64", "NaT"
]

_FILL_VALUE_SPELLINGS: list[tuple[DataTypeDefinition[Any], FillValueSpelling]] = [
    *(
        (definition, spelling)
        for definition in (
            FLOAT16_DATA_TYPE,
            FLOAT32_DATA_TYPE,
            FLOAT64_DATA_TYPE,
            COMPLEX64_DATA_TYPE,
            COMPLEX128_DATA_TYPE,
        )
        for spelling in get_args(FillValueSpelling)[:4]
    ),
    (BYTES_DATA_TYPE, "array"),
    (BYTES_DATA_TYPE, "base64"),
    (NUMPY_DATETIME64_DATA_TYPE, "number"),
    (NUMPY_DATETIME64_DATA_TYPE, "NaT"),
    (NUMPY_TIMEDELTA64_DATA_TYPE, "number"),
    (NUMPY_TIMEDELTA64_DATA_TYPE, "NaT"),
]
"""The spellings the specs give a fill value, by the data types whose fill values take them."""


def _spellings_of(value: object) -> set[FillValueSpelling]:
    """How a fill value is spelled; a complex one, how either component is."""
    if isinstance(value, bool):
        return set()
    if isinstance(value, int | float):
        return {"number"}
    if isinstance(value, tuple):
        parts = cast("tuple[object, ...]", value)
        if all(isinstance(part, int) for part in parts):
            return {"array"}
        return {spelling for part in parts for spelling in _spellings_of(part)}
    if not isinstance(value, str):
        return set()
    if value == "NaT":
        return {"NaT"}
    if value in get_args(FloatSpecialFillValue):
        return {"named"}
    if value.startswith("0x"):
        digits = value[2:]
        return {
            *(("hex lowercase",) if any(c in "abcdef" for c in digits) else ()),
            *(("hex uppercase",) if any(c in "ABCDEF" for c in digits) else ()),
        }
    return {"base64"}


@pytest.mark.parametrize(("definition", "spelling"), _FILL_VALUE_SPELLINGS, ids=_id)
def test_fill_values_reach(
    definition: DataTypeDefinition[Any], spelling: FillValueSpelling
) -> None:
    values = data_type_configurations(definition).flatmap(
        lambda configuration: fill_values(definition, configuration)
    )
    find(values, lambda value: spelling in _spellings_of(value), settings=_REACH)


def _codecs_in(pipeline: tuple[Any, ...]) -> list[tuple[str, tuple[str, ...]]]:
    """Each codec's name in `pipeline`, with the members of the shards it sits inside."""
    found: list[tuple[str, tuple[str, ...]]] = []
    for field in pipeline:
        if isinstance(field, str):
            found.append((field, ()))
            continue
        found.append((field["name"], ()))
        if field["name"] != SHARDING_INDEXED_CODEC.name:
            continue
        configuration: Mapping[str, Any] = field["configuration"]
        for member in ("codecs", "index_codecs"):
            found.extend(
                (inner, (member, *within)) for inner, within in _codecs_in(configuration[member])
            )
    return found


def _drawn(strategy: st.SearchStrategy[T] | None) -> st.SearchStrategy[T]:
    """`strategy`, or nothing when there is none."""
    return st.nothing() if strategy is None else strategy


def _drawn_pipelines() -> st.SearchStrategy[tuple[Any, ...]]:
    return _chunks().flatmap(lambda chunk: _drawn(pipelines(chunk)))


def _shards() -> st.SearchStrategy[Any]:
    """Shards drawn for the chunks of a grid."""
    return _chunks().flatmap(
        lambda chunk: _drawn(codec_configurations(SHARDING_INDEXED_CODEC, chunk)).flatmap(
            lambda configuration: fields(SHARDING_INDEXED_CODEC, st.just(configuration))
        )
    )


@pytest.mark.parametrize("definition", _CODECS, ids=_id)
def test_pipelines_reach_every_codec(definition: CodecDefinition[Any]) -> None:
    find(
        _drawn_pipelines(),
        lambda pipeline: any(name == definition.name for name, _ in _codecs_in(pipeline)),
        settings=_REACH,
    )


@pytest.mark.parametrize("member", ["codecs", "index_codecs"])
@pytest.mark.parametrize(
    "definition", [d for d in _CODECS if d is not SHARDING_INDEXED_CODEC], ids=_id
)
def test_shards_hold_every_codec_their_pipeline_takes(
    definition: CodecDefinition[Any], member: str
) -> None:
    """A shard's pipelines reach each codec they take: its index codecs, only those of static size."""
    if member == "index_codecs" and definition.size != "static":
        return
    find(
        _shards(),
        lambda shard: (definition.name, (member,)) in _codecs_in((shard,)),
        settings=_REACH,
    )


def test_shards_reach_a_shard_inside() -> None:
    find(
        _shards(),
        lambda shard: (SHARDING_INDEXED_CODEC.name, ("codecs",)) in _codecs_in((shard,)),
        settings=_REACH,
    )


_VARIABLE_LENGTH = frozenset({STRING_DATA_TYPE, BYTES_DATA_TYPE})
"""The data types whose values vary in size, which a struct's fields may not be."""


@pytest.mark.parametrize(
    "definition", [d for d in _DATA_TYPES if d not in _VARIABLE_LENGTH], ids=_id
)
def test_struct_fields_reach_every_fixed_size_data_type(
    definition: DataTypeDefinition[Any],
) -> None:
    """A struct's field types reach each data type of fixed size, a struct among them."""
    find(
        data_type_configurations(STRUCT_DATA_TYPE),
        lambda configuration: any(
            _read_data_type(member["data_type"]).definition is definition
            for member in configuration["fields"]
        ),
        settings=_REACH,
    )


def _read_data_type(field: object) -> Any:  # noqa: ANN401
    resolved, problems = resolve(field, DataTypeDefinition, CORE_AND_EXTENSIONS)
    assert problems == ()
    return resolved


_REAL_NUMBERS = [d for d in _DATA_TYPES if d.name not in NOT_NUMBERS | COMPLEX]
"""The data types `cast_value` casts to: those that model real numbers."""


@pytest.mark.parametrize("definition", _REAL_NUMBERS, ids=_id)
def test_cast_value_reaches_every_real_number_data_type(
    definition: DataTypeDefinition[Any],
) -> None:
    find(
        fields(CAST_VALUE_CODEC).map(lambda field: _read(CAST_VALUE_CODEC, field)),
        lambda codec: codec.nested[("data_type",)].definition is definition,
        settings=_REACH,
    )


@pytest.mark.parametrize("direction", ["encode", "decode"])
def test_cast_value_reaches_scalars_in_each_direction(direction: str) -> None:
    find(
        _drawn(codec_configurations(CAST_VALUE_CODEC)),
        lambda configuration: len(configuration.get("scalar_map", {}).get(direction, ())) != 0,
        settings=_REACH,
    )
