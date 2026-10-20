"""The fast path of reading a v3 array document: one pydantic-core schema per scope does the structural walk.

An experiment, behind the optional `zarr-metadata[fast]` extra. Where
pydantic-core is importable, a scope compiles once into a
`SchemaValidator` that performs, in Rust, everything the Python reader
does before it asks a definition's rules: refining the document to JSON
(arrays as tuples, string keys, finite numbers), judging each extension
point's envelope, picking the definition a name denotes, type-checking
the configuration against its TypedDict with the bounds its members
carry, and reading each nested field the same way. What the validator
accepts is handed to Python, which builds the fields, asks the rules,
the chunk rules and the fill value rules, reads the pipeline, and hands
back what `read_array_v3` needs for a reading with no problem.

The validator is an oracle of acceptance, not of problems. The package's
problems are the Python reader's -- their `loc`, kind, message, `input`
and `ctx` -- and nothing here renders one: a document the validator
refuses, or whose rules find a problem, is read again by the Python
walk, which reports as it always has. So the fast path changes what a
read costs, never what it says, and the two readers are held to one
another by the differential run described in
`design/pydantic-core-backend.md`.

What pydantic's modes do not express, the schema rebuilds:

- a strict scalar for each JSON scalar, since lax mode takes `"1"` and
  `1.0` for an integer and `True` for `1`, which a document never means;
- a `Literal` behind a strict scalar of its type, since pydantic's own
  literal takes `True` for `1` and `3.0` for `3`;
- a number as `int | float`, since pydantic's float turns `1` into `1.0`,
  and the float behind an `isinstance` gate, since even strict it takes a
  `Decimal` and a `Fraction`;
- an `isinstance` gate on list and tuple before a lax tuple, since the
  lax tuple also takes a set, a generator and a range;
- a definition reference around every container, since pydantic's
  recursion guard counts references rather than levels, so that a
  container 256 levels down is refused exactly where `JSON_DEPTH` refuses
  it;
- a callable discriminator at each extension point, which files a name
  as the scope does -- `r16` under `r*` -- and sends a name nothing in
  scope claims to a branch that accepts any well-named extension, which
  a tagged union keyed on `name` alone cannot;
- a typed dict's members put back in the order the document wrote them,
  since pydantic writes them in the schema's.

An integer of more digits than the interpreter writes makes pydantic
raise `ValueError` rather than a validation error; it is caught with the
rest, and the Python walk reports it.
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    NewType,
    TypeAlias,
    cast,
    get_args,
    get_origin,
)

from typing_extensions import TypeAliasType, is_typeddict

from zarr_metadata._common import JSONValue
from zarr_metadata._typed_json import (
    alias_value,
    constraints_of,
    is_alias,
    is_union,
    strip_annotation,
    typeddict_keys,
)
from zarr_metadata.v3._common import _EXTENSION_NAME, _URI  # pyright: ignore[reportPrivateUsage]
from zarr_metadata.v3._definition import (
    AcceptedField,
    Chunk,
    CodecDefinition,
    DataTypeDefinition,
    Definition,
    EmptyConfiguration,
    Lengths,
    ResolvedField,
    UnclaimedField,
    ZarrV3StaticCodecField,
    chunk_grid_lengths,
    field_kind,
    fill_value_parser,
    fill_value_problems,
    no_rules,
    read_only,
    ruled,
    trusted_field,
    written_json,
)
from zarr_metadata.v3._pipeline import Stage, read_pipeline
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_core import SchemaValidator
    from pydantic_core import core_schema as cs
    from pydantic_core.core_schema import CoreSchema

    from zarr_metadata._typed_json import Loc, Parser
    from zarr_metadata.v3._registry import Context

    AVAILABLE: Final = True
else:
    try:
        from pydantic_core import SchemaValidator
        from pydantic_core import core_schema as cs

        AVAILABLE = True
    except ImportError:  # the extra is not installed
        AVAILABLE = False

ENABLED: Final = AVAILABLE and os.environ.get("ZARR_METADATA_READER", "") != "python"
"""Whether the fast path is taken: pydantic-core is importable, and `ZARR_METADATA_READER=python` does not say otherwise."""


class Unsupported(TypeError):
    """A scope holds a definition whose configuration the schema compiler does not write; the scope is read by the Python walk alone."""


class _Fallback(Exception):
    """The fast path found a problem it does not report: the Python walk reads the document."""


# --- the plan: where a configuration holds metadata fields, and objects to put in the document's order ---

Plan = TypeAliasType(
    "Plan",
    tuple[Literal["f"], type[Definition[Any]], bool]
    | tuple[Literal["td"], dict[str, "Plan"]]
    | tuple[Literal["seq"], "Plan"]
    | tuple[Literal["pos"], dict[int, "Plan"]]
    | tuple[Literal["map"], "Plan"],
)
"""Where a value holds what the reader builds or puts in order.

`("f", kind, static)` is a metadata field, read as `kind`, holding codecs
of static size only when `static`; `("td", members)` an object whose
members pydantic wrote in the schema's order, with the plan of each
member that has one; `("seq", plan)` an array of such values; `("pos",
members)` a fixed array with the plan of each position; `("map", plan)`
an object of any keys holding such values. None where a value holds
none of these.
"""


def _plan(annotation: object, seen: frozenset[object] = frozenset()) -> Plan | None:
    """The plan of `annotation`; None where a value of it holds no field and no typed dict.

    `seen` holds the aliases and TypedDicts above, so one that holds
    itself -- `JSONValue` -- is walked once: a field inside such a type
    is not read by the fast path, as a field inside a union is not.
    """
    inner, _ = strip_annotation(annotation)
    kind = field_kind(inner)
    if kind is not None:
        return ("f", kind, inner is ZarrV3StaticCodecField)
    if inner is JSONValue or inner in seen:
        return None
    if is_alias(inner):
        return _plan(alias_value(inner), seen | {inner})
    if isinstance(inner, NewType):
        return _plan(inner.__supertype__, seen)
    origin = get_origin(inner)
    if origin is tuple:
        arguments = get_args(inner)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            found = _plan(arguments[0], seen)
            return None if found is None else ("seq", found)
        positions: dict[int, Plan] = {}
        for index, argument in enumerate(arguments):
            found = _plan(argument, seen)
            if found is not None:
                positions[index] = found
        return ("pos", positions) if len(positions) != 0 else None
    if origin in (Mapping, dict):
        found = _plan(get_args(inner)[1], seen)
        return None if found is None else ("map", found)
    if isinstance(inner, type) and is_typeddict(inner):
        members: dict[str, Plan] = {}
        for key, (member, _) in typeddict_keys(inner).members.items():
            found = _plan(member, seen | {inner})
            if found is not None:
                members[key] = found
        return ("td", members)
    if is_union(inner):
        if any(_plan(branch, seen) is not None for branch in get_args(inner)):
            msg = (
                f"{inner!r}: a metadata field or an object inside a union is not read by the "
                "fast path"
            )
            raise Unsupported(msg)
        return None
    return None


# --- the schema compiler -----------------------------------------------------

_NAME_PATTERN: Final = f"^(?:{_EXTENSION_NAME.pattern}|{_URI.pattern})$"
"""What `well_named` accepts, as pydantic's regex engine matches it: `$` is the end of the text there, with no newline before it."""

_Bounds: TypeAlias = Mapping[str, int | float]


def _int_bound(said: _Bounds, name: str) -> int | None:
    """The bound `said` holds under `name`, which an integer's type carries as an integer."""
    bound = said.get(name)
    if bound is None:
        return None
    if isinstance(bound, float):
        if not bound.is_integer():
            msg = f"{name}={bound!r}: a bound on an integer that is not one is not read by the fast path"
            raise Unsupported(msg)
        return int(bound)
    return bound


def _integer(said: _Bounds) -> CoreSchema:
    """An integer, and no bool, within `said` and within what the interpreter writes.

    `refine_json` refuses an integer of more digits than
    `sys.get_int_max_str_digits` lets `str` write, since no document
    could hold it; pydantic takes any integer, so the limit is a bound of
    every integer's schema, as it stands when the scope compiles.
    """
    ge, le = _int_bound(said, "ge"), _int_bound(said, "le")
    digits = sys.get_int_max_str_digits()
    if digits != 0:
        widest = 10**digits - 1
        ge = -widest if ge is None else max(ge, -widest)
        le = widest if le is None else min(le, widest)
    return cs.int_schema(
        strict=True, ge=ge, le=le, gt=_int_bound(said, "gt"), lt=_int_bound(said, "lt")
    )


def _float(said: _Bounds, *, finite: bool) -> CoreSchema:
    """A float within `said`, finite unless `finite` is false, and only a float: pydantic's strict float also takes a `Decimal` and a `Fraction`, which are not JSON."""
    return cs.chain_schema(
        [
            cs.is_instance_schema(float),
            cs.float_schema(
                strict=True,
                allow_inf_nan=not finite,
                ge=said.get("ge"),
                le=said.get("le"),
                gt=said.get("gt"),
                lt=said.get("lt"),
            ),
        ]
    )


def _number(said: _Bounds) -> CoreSchema:
    """A number within `said`: an integer kept as one, else a finite float."""
    choices: list[CoreSchema | tuple[CoreSchema, str]] = [
        _integer(said),
        _float(said, finite=True),
    ]
    return cs.union_schema(choices, mode="left_to_right")


def _array(items: CoreSchema) -> CoreSchema:
    """An array of `items`: a list or a tuple, as a tuple."""
    return cs.chain_schema(
        [cs.is_instance_schema((list, tuple)), cs.tuple_schema([items], variadic_item_index=0)]
    )


def _literal(values: tuple[object, ...]) -> CoreSchema:
    """A closed set of JSON scalars, each taken only in its own type: `True` is not `1`, nor `3.0` `3`."""
    by_type: dict[type, list[object]] = {}
    for value in values:
        by_type.setdefault(type(value), []).append(value)
    branches: list[CoreSchema | tuple[CoreSchema, str]] = []
    for cls, members in by_type.items():
        if cls is bool:
            branches.append(
                cs.chain_schema([cs.bool_schema(strict=True), cs.literal_schema(members)])
            )
        elif cls is int:
            branches.append(
                cs.chain_schema([cs.int_schema(strict=True), cs.literal_schema(members)])
            )
        elif cls is str:
            branches.append(
                cs.chain_schema([cs.str_schema(strict=True), cs.literal_schema(members)])
            )
        elif cls is types.NoneType:
            branches.append(cs.none_schema())
        else:
            msg = f"{values!r}: a Literal of values no document holds"
            raise Unsupported(msg)
    if len(branches) == 1:
        first = branches[0]
        return first[0] if isinstance(first, tuple) else first
    return cs.union_schema(branches, mode="left_to_right")


class _Compiler:
    """One scope's schema: each container a definition, each extension point a tagged union over the scope's table."""

    def __init__(self, context: Context) -> None:
        self.context = context
        self.definitions: dict[str, CoreSchema | None] = {}
        self.named: dict[object, str] = {}
        self.counter = 0
        self.plans: dict[int, Plan | None] = {}
        """The plan of each definition's configuration, by the definition's identity."""

    # -- definitions --

    def _define(self, name: str, build: Callable[[], CoreSchema]) -> CoreSchema:
        """A reference to the definition `name`, written by `build` the first time it is asked for."""
        if name not in self.definitions:
            self.definitions[name] = None
            schema = build()
            schema["ref"] = name  # pyright: ignore[reportGeneralTypeIssues]
            self.definitions[name] = schema
        return cs.definition_reference_schema(name)

    def _anonymous(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}#{self.counter}"

    def _container(self, prefix: str, build: Callable[[], CoreSchema]) -> CoreSchema:
        """A container schema behind a definition of its own, so the recursion guard counts the level."""
        return self._define(self._anonymous(prefix), build)

    # -- JSON --

    def json(self, *, finite: bool) -> CoreSchema:
        """A JSON value, refined: arrays as tuples, string keys, and a finite number unless `finite` is false."""
        tag = "finite" if finite else "user"
        arrays = self._define(f"json:array:{tag}", lambda: _array(self.json(finite=finite)))
        objects = self._define(
            f"json:object:{tag}",
            lambda: cs.dict_schema(cs.str_schema(strict=True), self.json(finite=finite)),
        )
        choices: list[CoreSchema | tuple[CoreSchema, str]] = [
            cs.str_schema(strict=True),
            _integer({}),
            _float({}, finite=finite),
            cs.bool_schema(strict=True),
            cs.none_schema(),
            objects,
            arrays,
        ]
        return cs.union_schema(choices, mode="left_to_right")

    # -- annotations --

    def compile(self, annotation: object) -> CoreSchema:
        """`annotation` as the checker reads it, in pydantic-core's schema."""
        inner, metadata = strip_annotation(annotation)
        return self._type(inner, constraints_of(metadata))

    def _type(self, inner: object, said: _Bounds) -> CoreSchema:
        kind = field_kind(inner)
        if kind is not None:
            return self.field(kind, static=inner is ZarrV3StaticCodecField)
        if inner is int:
            return _integer(said)
        if inner is float:
            return _number(said)
        if inner is bool:
            return cs.bool_schema(strict=True)
        if inner is str:
            return cs.str_schema(strict=True)
        if inner is None or inner is types.NoneType:
            return cs.none_schema()
        if inner is JSONValue:
            return self.json(finite=True)
        origin = get_origin(inner)
        if origin is Literal:
            return _literal(get_args(inner))
        if is_union(inner):
            choices: list[CoreSchema | tuple[CoreSchema, str]] = [
                self.compile(branch) for branch in get_args(inner)
            ]
            return cs.union_schema(choices, mode="left_to_right")
        if origin is tuple:
            arguments = get_args(inner)
            if len(arguments) == 2 and arguments[1] is Ellipsis:
                return self._container("array", lambda: _array(self.compile(arguments[0])))
            return self._container(
                "array",
                lambda: cs.chain_schema(
                    [
                        cs.is_instance_schema((list, tuple)),
                        cs.tuple_schema([self.compile(argument) for argument in arguments]),
                    ]
                ),
            )
        if origin in (Mapping, dict):
            arguments = get_args(inner)
            if len(arguments) != 2 or arguments[0] is not str:
                msg = f"{inner!r} is not a shape JSON takes"
                raise Unsupported(msg)
            return self._container(
                "object",
                lambda: cs.dict_schema(cs.str_schema(strict=True), self.compile(arguments[1])),
            )
        if isinstance(inner, type) and is_typeddict(inner):
            return self._typed_dict(inner)
        if isinstance(inner, NewType):
            return self.compile(inner.__supertype__)
        if is_alias(inner):
            alias = inner
            name = self.named.get(alias)
            if name is None:
                name = self.named[alias] = self._anonymous(f"alias:{alias.__name__}")
            return self._define(name, lambda: self.compile(alias_value(alias)))
        msg = f"{inner!r} is not a shape the fast path reads"
        raise Unsupported(msg)

    def _typed_dict(self, typeddict: type) -> CoreSchema:
        name = self.named.get(typeddict)
        if name is None:
            name = self.named[typeddict] = self._anonymous(f"object:{typeddict.__name__}")

        def build() -> CoreSchema:
            keys = typeddict_keys(typeddict)
            fields = {
                key: cs.typed_dict_field(self.compile(member), required=required)
                for key, (member, required) in keys.members.items()
            }
            if keys.closed:
                return cs.typed_dict_schema(fields, extra_behavior="forbid")
            extras = self.json(finite=True) if keys.open else self.compile(keys.extra_items)
            return cs.typed_dict_schema(fields, extra_behavior="allow", extras_schema=extras)

        return self._define(name, build)

    # -- extension points --

    def field(self, kind: type[Definition[Any]], *, static: bool) -> CoreSchema:
        """A metadata field read as `kind` in the scope: a tagged union over the names in scope, and the names nothing claims."""
        name = f"field:{kind.__name__}:{'static' if static else 'any'}"
        return self._define(name, lambda: self._field(kind, static))

    def _field(self, kind: type[Definition[Any]], static: bool) -> CoreSchema:
        table = self.context.tables.get(kind, {})
        tags = {filed: (f"s:{filed}", f"o:{filed}") for filed in table}
        choices: dict[str, CoreSchema] = {}
        for filed, definition in table.items():
            self.plans[id(definition)] = _plan(definition.configuration)
            if static and cast("CodecDefinition[Any]", definition).size != "static":
                continue
            carried = (
                isinstance(definition, DataTypeDefinition)
                and definition.filed_under(filed)[0] is None
            )
            bare, envelope = tags[filed]
            if carried:
                choices[bare] = cs.str_schema(strict=True)
                choices[envelope] = self._envelope(
                    f"envelope:{kind.__name__}:{filed}",
                    cs.str_schema(strict=True),
                    self._typed_dict(EmptyConfiguration),
                    required=False,
                )
                continue
            if not definition.requires_configuration:
                choices[bare] = cs.str_schema(strict=True)
            choices[envelope] = self._envelope(
                f"envelope:{kind.__name__}:{filed}",
                cs.str_schema(strict=True),
                self._typed_dict(definition.configuration),
                required=definition.requires_configuration,
            )
        well_named = cs.str_schema(strict=True, pattern=_NAME_PATTERN)
        choices["s:?"] = well_named
        choices["o:?"] = self._envelope(
            f"envelope:{kind.__name__}:?",
            well_named,
            self._container(
                "object",
                lambda: cs.dict_schema(cs.str_schema(strict=True), self.json(finite=True)),
            ),
            required=False,
        )
        filed_under = kind.filed_under

        def discriminate(value: object) -> str | None:
            if isinstance(value, str):
                filed = filed_under(value)[0]
                found = None if filed is None else tags.get(filed)
                return "s:?" if found is None else found[0]
            if isinstance(value, Mapping):
                written = cast("Mapping[object, object]", value).get("name")
                if isinstance(written, str):
                    filed = filed_under(written)[0]
                    found = None if filed is None else tags.get(filed)
                    return "o:?" if found is None else found[1]
            return None

        return cs.tagged_union_schema(choices, discriminate)

    def _envelope(
        self, name: str, written: CoreSchema, configuration: CoreSchema, *, required: bool
    ) -> CoreSchema:
        return self._define(
            name,
            lambda: cs.typed_dict_schema(
                {
                    "name": cs.typed_dict_field(written),
                    "configuration": cs.typed_dict_field(configuration, required=required),
                    "must_understand": cs.typed_dict_field(_literal((True,)), required=False),
                },
                extra_behavior="forbid",
            ),
        )

    # -- the document --

    def document(self) -> tuple[CoreSchema, list[CoreSchema]]:
        """The schema of a v3 array document, as `ZarrV3ArrayMetadataJSON` declares it, `attributes` as user data: its root, and the definitions it refers to."""
        keys = typeddict_keys(ZarrV3ArrayMetadataJSON)
        fields: dict[str, cs.TypedDictField] = {}
        for key, (member, required) in keys.members.items():
            if key == "attributes":
                schema = self._container(
                    "object",
                    lambda: cs.dict_schema(cs.str_schema(strict=True), self.json(finite=False)),
                )
            else:
                schema = self.compile(member)
            fields[key] = cs.typed_dict_field(schema, required=required)
        root = cs.typed_dict_schema(
            fields, extra_behavior="allow", extras_schema=self.compile(keys.extra_items)
        )
        return root, [schema for schema in self.definitions.values() if schema is not None]


def _validator(root: CoreSchema, definitions: list[CoreSchema], depth: int) -> SchemaValidator:
    """The validator of a document that sits `depth` levels below the root of the one handed in.

    The levels a reader walks are counted from the outermost root: a
    document consolidated metadata holds sits three levels below its
    group's, and a container 253 levels below it is past them. Each
    level is a reference around the root, which pydantic's recursion
    guard counts as it counts a level of nesting.
    """
    if depth == 0:
        return SchemaValidator(cs.definitions_schema(root, definitions))
    wrapped = [*definitions, cast("CoreSchema", {**root, "ref": "document"})]
    inner = "document"
    for level in range(depth):
        name = f"document:{level}"
        wrapped.append(cs.definition_reference_schema(inner, ref=name))
        inner = name
    return SchemaValidator(cs.definitions_schema(cs.definition_reference_schema(inner), wrapped))


# --- the reader ---------------------------------------------------------------


_Read: TypeAlias = "AcceptedField[Any] | UnclaimedField"
"""What the fast path makes of a field: never a refusal, which is `_Fallback`."""


@dataclass(frozen=True, slots=True)
class FastArrayRead:
    """What the fast path read of a v3 array document with no problem: what `read_array_v3` builds its reading and members from."""

    read: dict[str, _Read]
    chunk: Chunk
    pipeline: tuple[Stage, ...]
    storage_transformers: tuple[ResolvedField[Any], ...]
    shape: tuple[int, ...]
    fill_value: JSONValue
    dimension_names: tuple[str | None, ...] | None
    attributes: dict[str, JSONValue]
    extra_fields: dict[str, JSONValue]
    document: dict[str, JSONValue]


_MEMBERS: Final = typeddict_keys(ZarrV3ArrayMetadataJSON).members
_STANDARD: Final = frozenset(_MEMBERS)
_POINTS: Final[tuple[tuple[str, type[Definition[Any]]], ...]] = tuple(
    (key, kind)
    for key, (annotation, _) in _MEMBERS.items()
    if (kind := field_kind(annotation)) is not None
)
_LISTS: Final[tuple[tuple[str, type[Definition[Any]]], ...]] = tuple(
    (key, kind)
    for key, (annotation, _) in _MEMBERS.items()
    if get_origin(annotation) is tuple and (kind := field_kind(get_args(annotation)[0])) is not None
)

_Entry: TypeAlias = "tuple[Definition[Any], dict[str, Plan] | None]"
"""A definition a name denotes directly, and the plan of each member of its configuration that has one; None when none has."""

_NO_ENTRIES: Final[dict[str, _Entry]] = {}
"""What a kind the scope files nothing under denotes directly: nothing."""


class FastArrayReader:
    """A scope's compiled reader of v3 array documents.

    What pydantic hands back is untyped to the type checker and vouched
    for by the schema; the reader walks it as such, with no cast, since
    a cast is a call.
    """

    __slots__ = ("_definitions", "_digits", "_direct", "_fill", "_root", "_tables", "_validators")

    def __init__(self, context: Context) -> None:
        compiler = _Compiler(context)
        self._root, self._definitions = compiler.document()
        self._validators: dict[int, SchemaValidator] = {
            0: _validator(self._root, self._definitions, 0)
        }
        self._tables: dict[type[Definition[Any]], dict[str, Definition[Any]]] = {
            kind: dict(table) for kind, table in context.tables.items()
        }
        # Each name a document writes that is a definition's own, to the
        # definition and the plan of its configuration: `r16` is not one,
        # and reads through `filed_under`.
        self._direct: dict[type[Definition[Any]], dict[str, _Entry]] = {}
        for kind, table in self._tables.items():
            direct: dict[str, _Entry] = {}
            for filed, definition in table.items():
                if kind.filed_under(filed)[0] == filed:
                    plan = compiler.plans.get(id(definition))
                    members = plan[1] if plan is not None and plan[0] == "td" else None
                    direct[filed] = (definition, members or None)
            self._direct[kind] = direct
        self._digits = sys.get_int_max_str_digits()
        # The checker of each data type's fill value, by the definition's
        # identity: `fill_value_problems` finds it by its annotation, which
        # hashes a union of types on every read.
        self._fill: dict[int, Parser] = {
            id(definition): fill_value_parser(definition.fill_value)
            for definition in self._tables.get(DataTypeDefinition, {}).values()
            if isinstance(definition, DataTypeDefinition)
        }

    def read(self, value: object, depth: int = 0) -> FastArrayRead | None:
        """`value`, a document `depth` levels below the root of the one handed in, read in the scope; None when it has a problem, or holds something the fast path does not read, for the Python walk to report."""
        validator = self._validators.get(depth)
        if validator is None:
            validator = self._validators[depth] = _validator(self._root, self._definitions, depth)
        try:
            out = validator.validate_python(value)
        except ValueError:
            # A validation error, or an integer of more digits than the
            # interpreter writes.
            return None
        try:
            return self._build(value, out)
        except _Fallback:
            return None

    def _build(
        self,
        source: Any,  # noqa: ANN401 - the document as handed in, an object pydantic accepted
        out: Any,  # noqa: ANN401 - what pydantic made of it, which its schema vouches for
    ) -> FastArrayRead:
        if list(source) != list(out):
            out = {key: out[key] for key in source}
        read: dict[str, _Read] = {}
        for key, kind in _POINTS:
            if key in out:
                read[key], written = self._field(source[key], out[key], kind, (key,))
                if written is not out[key]:
                    out[key] = written
        fill_value = out["fill_value"]
        data_type = read.get("data_type")
        if data_type is not None:
            self._fill_value(data_type, fill_value)
        shape = out["shape"]
        lengths: Lengths = (None,) * len(shape)
        chunk_grid = read.get("chunk_grid")
        if chunk_grid is not None:
            lengths, found = chunk_grid_lengths(chunk_grid, shape, ("chunk_grid",))
            if len(found) != 0:
                raise _Fallback
        listed: dict[str, tuple[ResolvedField[Any], ...]] = {}
        for key, kind in _LISTS:
            if key in out:
                given = source[key]
                entries = out[key]
                fields: list[ResolvedField[Any]] = []
                written_entries: list[object] = []
                changed = False
                for index, entry in enumerate(entries):
                    field, written = self._field(given[index], entry, kind, (key, index))
                    fields.append(field)
                    written_entries.append(written)
                    if written is not entry:
                        changed = True
                if changed:
                    out[key] = tuple(written_entries)
                listed[key] = tuple(fields)
        chunk = Chunk(lengths, data_type)
        pipeline: tuple[Stage, ...] = ()
        if "codecs" in listed:
            pipeline, found = read_pipeline(listed["codecs"], chunk, ("codecs",))
            if len(found) != 0:
                raise _Fallback
        dimension_names = out.get("dimension_names")
        if dimension_names is not None and len(dimension_names) != len(shape):
            raise _Fallback
        return FastArrayRead(
            read=read,
            chunk=chunk,
            pipeline=pipeline,
            storage_transformers=listed.get("storage_transformers", ()),
            shape=shape,
            fill_value=fill_value,
            dimension_names=dimension_names,
            attributes=out.get("attributes", {}),
            extra_fields=_extras(out),
            document=out,
        )

    def _fill_value(self, data_type: _Read, value: JSONValue) -> None:
        """`_Fallback` when `value`, refined, is not a fill value of `data_type`, as `fill_value_problems` judges one; nothing otherwise."""
        if isinstance(data_type, UnclaimedField):
            return
        field = cast("AcceptedField[DataTypeDefinition[Any]]", data_type)
        definition = field.definition
        parser = self._fill.get(id(definition))
        if parser is None:
            if len(fill_value_problems(field, value, ("fill_value",))) != 0:
                raise _Fallback
            return
        typed, problems = parser(value, ("fill_value",))
        if len(problems) != 0:
            raise _Fallback
        rules = definition.fill_value_rules
        if rules is not no_rules:
            configuration, nested = field.configuration, field.nested
            refused = ruled(
                definition, lambda: rules(read_only(configuration), nested, typed), ("fill_value",)
            )
            if len(refused) != 0:
                raise _Fallback

    def _field(
        self,
        source: Any,  # noqa: ANN401 - the field as the document wrote it, which pydantic accepted
        out: Any,  # noqa: ANN401 - what pydantic made of it, which its schema vouches for
        kind: type[Definition[Any]],
        loc: Loc,
    ) -> tuple[_Read, object]:
        """The field pydantic accepted as `kind`, as the scope reads it, and the field as written, its objects in the document's order.

        By the definition that claims its name, or claimed by nothing;
        a problem the rules find, or a codec of dynamic size where only
        static ones go, is `_Fallback`.
        """
        envelope: Any = None
        if isinstance(out, str):
            name: str = out
            given = None
        else:
            if list(source) != list(out):
                out = {key: out[key] for key in source}
            envelope = out
            name = envelope["name"]
            given = envelope.get("configuration")
        entry = self._direct.get(kind, _NO_ENTRIES).get(name)
        if entry is None:
            return self._indirect(out, name, kind)
        definition, members = entry
        nested: dict[Loc, ResolvedField[Any]] = {}
        configuration: Mapping[str, Any]
        if given is None:
            configuration = {}
        elif members is None:
            source_given = source["configuration"]
            if list(source_given) != list(given):
                given = {key: given[key] for key in source_given}
                envelope["configuration"] = given
            configuration = given
        else:
            written, read = self._node(
                source["configuration"],
                given,
                ("td", members),
                (),
                (*loc, "configuration"),
                nested,
            )
            if written is not given:
                envelope["configuration"] = written
            configuration = read  # pyright: ignore[reportAssignmentType]
        if definition.rules is not no_rules:
            refused = ruled(
                definition,
                lambda: definition.rules(read_only(configuration), nested),
                (*loc, "configuration"),
            )
            if len(refused) != 0:
                raise _Fallback
        return trusted_field(out, name, definition, configuration, nested), out

    def _indirect(
        self,
        out: Any,  # noqa: ANN401 - as `_field` takes it
        name: str,
        kind: type[Definition[Any]],
    ) -> tuple[_Read, object]:
        """A field whose name is no definition's own: one that carries its configuration, `r16`, or one nothing in scope claims."""
        filed, carried = kind.filed_under(name)
        definition = None if filed is None else self._tables.get(kind, {}).get(filed)
        if definition is None:
            return UnclaimedField(json=out, name=name, read_as=kind), out
        if carried is None:  # pragma: no cover - a name that is its own files directly
            raise _Fallback
        configuration, judged = definition.read_configuration(carried)
        if configuration is None or len(judged) != 0:
            raise _Fallback
        return trusted_field(out, name, definition, configuration, {}), out

    def _node(
        self,
        source: Any,  # noqa: ANN401 - as `_field` takes it
        out: Any,  # noqa: ANN401 - as `_field` takes it
        plan: Plan,
        place: Loc,
        at: Loc,
        nested: dict[Loc, ResolvedField[Any]],
    ) -> tuple[object, object]:
        """`out`, a value of a configuration at `place`, as written and as read: objects in the document's order, and each field it holds read into `nested`, put back as a document writes it in the second."""
        if plan[0] == "f":
            inner, written = self._field(source, out, plan[1], (*at, *place))
            if plan[2]:
                definition = inner.definition
                if isinstance(definition, CodecDefinition) and definition.size != "static":
                    raise _Fallback
            nested[place] = inner
            return written, written_json(inner)
        if plan[0] == "td":
            if list(source) != list(out):
                out = {key: out[key] for key in source}
            replaced: dict[str, object] | None = None
            members = plan[1]
            # In the document's order, which is the order the fields it
            # holds are read in, and the order `nested` lists them.
            for key, item in out.items():
                child = members.get(key)
                if child is None:
                    continue
                written, read = self._node(source[key], item, child, (*place, key), at, nested)
                if written is not item:
                    out[key] = written
                if read is not written:
                    if replaced is None:
                        replaced = {}
                    replaced[key] = read
            if replaced is None:
                return out, out
            return out, {**out, **replaced}
        if plan[0] == "seq":
            child = plan[1]
            written_items: list[object] = []
            read_items: list[object] = []
            rewritten = reread = False
            for index, item in enumerate(out):
                written, read = self._node(source[index], item, child, (*place, index), at, nested)
                written_items.append(written)
                read_items.append(read)
                if written is not item:
                    rewritten = True
                if read is not written:
                    reread = True
            as_written = tuple(written_items) if rewritten else out
            return as_written, (tuple(read_items) if reread else as_written)
        if plan[0] == "pos":
            written_items = list(out)
            read_items = list(out)
            rewritten = reread = False
            for index, child in plan[1].items():
                written, read = self._node(
                    source[index], out[index], child, (*place, index), at, nested
                )
                written_items[index] = written
                read_items[index] = read
                if written is not out[index]:
                    rewritten = True
                if read is not written:
                    reread = True
            as_written = tuple(written_items) if rewritten else out
            return as_written, (tuple(read_items) if reread else as_written)
        # "map": an object of any keys, in the document's order already.
        child = plan[1]
        written_members: dict[str, object] = {}
        read_members: dict[str, object] = {}
        rewritten = reread = False
        for key, item in out.items():
            written, read = self._node(source[key], item, child, (*place, key), at, nested)
            written_members[key] = written
            read_members[key] = read
            if written is not item:
                rewritten = True
            if read is not written:
                reread = True
        as_written = written_members if rewritten else out
        return as_written, (read_members if reread else as_written)


def _extras(out: dict[str, Any]) -> dict[str, JSONValue]:
    """The members of `out`, a document, that the spec does not define, in the document's order."""
    other = out.keys() - _STANDARD
    if len(other) == 0:
        return {}
    return {key: item for key, item in out.items() if key in other}


_READERS: Final[dict[Context, FastArrayReader | Literal[False]]] = {}
"""The compiled reader of each scope compiled so far, by the scope's equality: a scope built again -- unpickled with a model, or by `Context.of` of the same definitions -- reads with the one its equal compiled, rather than compiling its own."""


def reader_for(context: Context) -> FastArrayReader | None:
    """The compiled reader of `context`, built the first time it or an equal scope is asked for; None when the fast path is off, or the scope holds what it does not read."""
    if not ENABLED:
        return None
    held = context._fast  # pyright: ignore[reportPrivateUsage]
    digits = sys.get_int_max_str_digits()
    if held is not None:
        if not isinstance(held, FastArrayReader):
            return None
        if held._digits == digits:  # pyright: ignore[reportPrivateUsage]
            return held
        # The limit on an integer's digits is compiled into the schema,
        # and the interpreter's can be changed: compiled again when it was.
    reader = _READERS.get(context)
    if reader is None or (reader is not False and reader._digits != digits):  # pyright: ignore[reportPrivateUsage]
        try:
            reader = FastArrayReader(context)
        except Unsupported:
            reader = False
        _READERS[context] = reader
    object.__setattr__(context, "_fast", reader)
    return reader or None


__all__ = ["ENABLED", "FastArrayRead", "FastArrayReader", "Unsupported", "reader_for"]
