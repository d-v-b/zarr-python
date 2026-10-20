"""A differential run of two builds of zarr-metadata over a mutated corpus of documents.

Run from the package directory, once against each build, and diff the
two files:

    PYTHONHASHSEED=0 PYTHONPATH=/path/to/base/src uv run --group test \\
        python design/differential_corpus.py /tmp/base.tsv
    PYTHONHASHSEED=0 uv run --group test python design/differential_corpus.py /tmp/head.tsv
    diff /tmp/base.tsv /tmp/head.tsv

Each line is a case's id and the hash of its record; `--full` writes
the records themselves, and `--only <text>` runs the cases whose id
holds `<text>`, which is how a differing case is shown in full.

The corpus is the three benchmark documents, every JSON fixture under
`tests/`, a v3 array document for each data type's fill-value fixture,
and systematic mutations of each: every member and nested member
deleted, replaced by each of a hundred-odd junk values (NaN, big
integers, non-string keys, bytes, sets, enum members, mapping proxies,
model instances, fields, values nested to the depth cap and past it),
objects with an extra key, a non-string key, their keys reversed, a
`must_understand` of each kind, bad names. Each case is read by the
readers its document is for, and the record holds every problem (loc,
kind, message, input, ctx), each field the reading holds, its pipeline,
and, when there is a model, its document, its store bytes, and whether
it survives equality, hashing, pickling and deep copy. A record is one
JSON line; an address in a repr is normalized so a run is reproducible.
"""

# ruff: noqa: ANN401, SLOT000, SLOT001, RUF001, PERF401, FURB157
from __future__ import annotations

import collections
import copy
import datetime
import decimal
import enum
import fractions
import hashlib
import json
import pickle
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

sys.setrecursionlimit(10000)

import zarr_metadata
from zarr_metadata._json import ValidationProblem
from zarr_metadata._sentinel import UNSET
from zarr_metadata.model import (
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    read_array_metadata_v2,
    read_array_metadata_v3,
    read_group_metadata_v3,
    validate_group_metadata_v2,
)
from zarr_metadata.v3._definition import (
    AcceptedField,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
    RefusedField,
    StorageTransformerDefinition,
    UnclaimedField,
    canonical_of,
    resolve,
    with_problems,
)
from zarr_metadata.v3._registry import CORE, CORE_AND_EXTENSIONS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.benchmarks.read_documents import V2_ARRAY, V3_ARRAY, V3_GROUP

TESTS = Path(__file__).resolve().parent.parent / "tests"

print("zarr_metadata from", zarr_metadata.__file__, file=sys.stderr)

STATS = collections.Counter()
try:
    from zarr_metadata.v3 import _fast

    if _fast.ENABLED:
        _read = _fast.FastArrayReader.read

        def _counted(self: Any, value: object, depth: int = 0) -> Any:
            result = _read(self, value, depth)
            STATS["fast" if result is not None else "fallback"] += 1
            return result

        _fast.FastArrayReader.read = _counted  # type: ignore[method-assign]
        print("fast path: enabled", file=sys.stderr)
    else:
        print("fast path: disabled", file=sys.stderr)
except ImportError:
    print("fast path: absent", file=sys.stderr)


# --- showing a value as a stable record --------------------------------------

_ADDRESS = re.compile(r"0x[0-9a-fA-F]{6,}")


def normalized(text: str) -> str:
    return _ADDRESS.sub("0x…", text)


def show(value: object, depth: int = 0) -> object:
    """`value` as JSON a record holds: containers told apart, fields and problems by their parts, anything else by a normalized repr."""
    if depth > 400:
        return "…deep…"
    if value is UNSET:
        return "UNSET"
    if value is None or value is True or value is False:
        return value
    if type(value) is int:
        try:
            return value if value.bit_length() <= 64 else f"int<{value.bit_length()} bits>"
        except Exception:  # noqa: BLE001
            return "int?"
    if type(value) is float:
        return repr(value)
    if type(value) is str:
        return value
    if isinstance(value, ValidationProblem):
        return {
            "loc": list(value.loc),
            "kind": value.kind,
            "message": normalized(value.message),
            "input": show(value.input, depth + 1),
            "ctx": show(dict(value.ctx), depth + 1),
        }
    if isinstance(value, AcceptedField):
        return [
            "Accepted",
            value.name,
            value.definition.name,
            type(value.definition).__name__,
            show(value.configuration, depth + 1),
            show(value.json, depth + 1),
            {
                ".".join(map(str, loc)): show(inner, depth + 1)
                for loc, inner in value.nested.items()
            },
        ]
    if isinstance(value, UnclaimedField):
        return [
            "Unclaimed",
            value.name,
            value.read_as.__name__,
            show(value.configuration, depth + 1),
            show(value.json, depth + 1),
        ]
    if isinstance(value, RefusedField):
        return [
            "Refused",
            value.name,
            value.read_as.__name__,
            None if value.definition is None else value.definition.name,
            show(value.json, depth + 1),
            {
                ".".join(map(str, loc)): show(inner, depth + 1)
                for loc, inner in value.nested.items()
            },
        ]
    if isinstance(value, tuple):
        return ["tuple", *(show(item, depth + 1) for item in value)]
    if isinstance(value, list):
        return ["list", *(show(item, depth + 1) for item in value)]
    if isinstance(value, Mapping):
        kind = type(value).__name__
        return {
            "__type__": kind,
            **{
                (key if isinstance(key, str) else f"<{normalized(repr(key))}>"): show(
                    item, depth + 1
                )
                for key, item in value.items()
            },
        }
    if isinstance(value, (bytes, bytearray)):
        return repr(value)
    return f"<{type(value).__name__}: {normalized(repr(value))[:200]}>"


# --- junk --------------------------------------------------------------------


class Order(enum.StrEnum):
    LITTLE = "little"


class Level(enum.IntEnum):
    ZERO = 0
    FIVE = 5


class MyStr(str):
    pass


class MyInt(int):
    pass


class MyFloat(float):
    pass


class MyList(list):  # type: ignore[type-arg]
    pass


class MyTuple(tuple):  # type: ignore[type-arg]
    pass


class MyDict(dict):  # type: ignore[type-arg]
    pass


def nested_lists(depth: int, leaf: object = 0) -> object:
    value = leaf
    for _ in range(depth):
        value = [value]
    return value


def nested_dicts(depth: int, leaf: object = 0) -> object:
    value = leaf
    for _ in range(depth):
        value = {"k": value}
    return value


MODEL = ZarrV3ArrayMetadata.create_default(shape=(2,))
FIELD = MODEL.codecs[0]

JUNK: list[tuple[str, object]] = [
    ("None", None),
    ("True", True),
    ("False", False),
    ("0", 0),
    ("1", 1),
    ("-1", -1),
    ("2**63", 2**63),
    ("2**64", 2**64),
    ("-2**63-1", -(2**63) - 1),
    ("10**4299", 10**4299),
    ("10**4300", 10**4300),
    ("10**5000", 10**5000),
    ("-10**4300", -(10**4300)),
    ("0.0", 0.0),
    ("-0.0", -0.0),
    ("1.0", 1.0),
    ("1.5", 1.5),
    ("-1.5", -1.5),
    ("nan", float("nan")),
    ("inf", float("inf")),
    ("-inf", float("-inf")),
    ("1e308", 1e308),
    ("5e-324", 5e-324),
    ("''", ""),
    ("'a'", "a"),
    ("'0'", "0"),
    ("'1'", "1"),
    ("'true'", "true"),
    ("'null'", "null"),
    ("'NaN'", "NaN"),
    ("'Infinity'", "Infinity"),
    ("'-Infinity'", "-Infinity"),
    ("'0x7fc00000'", "0x7fc00000"),
    ("'little'", "little"),
    ("'big'", "big"),
    ("'C'", "C"),
    ("'/'", "/"),
    ("'.'", "."),
    ("'regular'", "regular"),
    ("'bytes'", "bytes"),
    ("'gzip'", "gzip"),
    ("'crc32c'", "crc32c"),
    ("'int8'", "int8"),
    ("'string'", "string"),
    ("'r16'", "r16"),
    ("'r8'", "r8"),
    ("'r12'", "r12"),
    ("'r0'", "r0"),
    ("'r*'", "r*"),
    ("'r16\\n'", "r16\n"),
    ("'r' + 101 digits", "r" + "1" * 101),
    ("'Gzip'", "Gzip"),
    ("'a b'", "a b"),
    ("'http://x/y'", "http://x/y"),
    ("'urn:x'", "urn:x"),
    ("'x:'", "x:"),
    ("'ü'", "ü"),
    ("'\\x00'", "\x00"),
    ("'r\\uff11\\uff16'", "r１６"),
    ("'gzip\\n'", "gzip\n"),
    ("[]", []),
    ("[1]", [1]),
    ("[1, 2]", [1, 2]),
    ("['a']", ["a"]),
    ("[None]", [None]),
    ("[[1]]", [[1]]),
    ("[1.0]", [1.0]),
    ("[True]", [True]),
    ("[0, 0]", [0, 0]),
    ("[-1]", [-1]),
    ("[2**63]", [2**63]),
    ("()", ()),
    ("(1,)", (1,)),
    ("(1, 2)", (1, 2)),
    ("{}", {}),
    ("{'a': 1}", {"a": 1}),
    ("{'name': 'gzip'}", {"name": "gzip"}),
    ("{'name': 'bytes'}", {"name": "bytes"}),
    ("{'name': 'zzz'}", {"name": "zzz"}),
    ("{'name': 1}", {"name": 1}),
    ("{'name': 'bytes', 'configuration': {}}", {"name": "bytes", "configuration": {}}),
    ("{'name': 'bytes', 'configuration': []}", {"name": "bytes", "configuration": []}),
    ("{'name': 'bytes', 'configuration': {1: 2}}", {"name": "bytes", "configuration": {1: 2}}),
    ("{'name': 'bytes', 'must_understand': False}", {"name": "bytes", "must_understand": False}),
    ("{'name': 'bytes', 'must_understand': True}", {"name": "bytes", "must_understand": True}),
    ("{'name': 'bytes', 'extra': 1}", {"name": "bytes", "extra": 1}),
    (
        "{'name': 'gzip', 'configuration': {'level': 12}}",
        {"name": "gzip", "configuration": {"level": 12}},
    ),
    (
        "{'name': 'gzip', 'configuration': {'level': 1, 'x': 2}}",
        {"name": "gzip", "configuration": {"level": 1, "x": 2}},
    ),
    (
        "{'name': 'zzz', 'configuration': {'x': float('nan')}}",
        {"name": "zzz", "configuration": {"x": float("nan")}},
    ),
    ("{'name': 'zzz', 'must_understand': False}", {"name": "zzz", "must_understand": False}),
    ("{'configuration': {}}", {"configuration": {}}),
    ("{1: 2}", {1: 2}),
    ("{None: 1}", {None: 1}),
    ("{(1, 2): 3}", {(1, 2): 3}),
    ("{'must_understand': False}", {"must_understand": False}),
    ("b''", b""),
    ("b'x'", b"x"),
    ("bytearray", bytearray(b"x")),
    ("set()", set()),
    ("{1}", {1}),
    ("frozenset()", frozenset()),
    ("range(3)", range(3)),
    ("deque([1])", collections.deque([1])),
    ("MappingProxyType({})", MappingProxyType({})),
    ("MappingProxyType({'name': 'bytes'})", MappingProxyType({"name": "bytes"})),
    (
        "MappingProxyType({'name': 'bytes', 'configuration': MappingProxyType({'endian': 'little'})})",
        MappingProxyType(
            {"name": "bytes", "configuration": MappingProxyType({"endian": "little"})}
        ),
    ),
    ("OrderedDict(name='bytes')", collections.OrderedDict(name="bytes")),
    ("UserDict(name='bytes')", collections.UserDict(name="bytes")),
    ("ChainMap({'name': 'bytes'})", collections.ChainMap({"name": "bytes"})),
    ("MyDict(name='bytes')", MyDict(name="bytes")),
    ("StrEnum 'little'", Order.LITTLE),
    ("IntEnum 0", Level.ZERO),
    ("IntEnum 5", Level.FIVE),
    ("MyStr('little')", MyStr("little")),
    ("MyStr('bytes')", MyStr("bytes")),
    ("MyInt(1)", MyInt(1)),
    ("MyFloat(1.5)", MyFloat(1.5)),
    ("MyList([1])", MyList([1])),
    ("MyTuple((1,))", MyTuple((1,))),
    ("object()", object()),
    ("lambda", lambda: None),
    ("type", int),
    ("complex", complex(1, 2)),
    ("Decimal", decimal.Decimal("1")),
    ("Fraction", fractions.Fraction(1, 2)),
    ("datetime", datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)),
    ("generator", (x for x in [1])),
    ("model", MODEL),
    ("field", FIELD),
    ("UNSET", UNSET),
    ("Ellipsis", ...),
    ("NotImplemented", NotImplemented),
    ("lists 300 deep", nested_lists(300)),
    ("dicts 300 deep", nested_dicts(300)),
]

BAD_NAMES = [
    "",
    "Gzip",
    "a b",
    "r*",
    "gzip\n",
    "http://x/y",
    "urn:x",
    "ü",
    "1abc",
    "_a",
    "a",
    "zzz",
    "r16",
    "r12",
    "bytes",
]


# --- mutations ---------------------------------------------------------------

Path_ = tuple[str | int, ...]


def structural_copy(value: object) -> object:
    if isinstance(value, dict):
        return {key: structural_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [structural_copy(item) for item in value]
    return value


def paths(value: object, prefix: Path_ = (), limit: int = 8) -> list[Path_]:
    found: list[Path_] = [prefix]
    if len(prefix) >= limit:
        return found
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(paths(item, (*prefix, key), limit))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(paths(item, (*prefix, index), limit))
    return found


def at(value: Any, path: Path_) -> Any:
    for step in path:
        value = value[step]
    return value


def put(root: Any, path: Path_, new: object) -> Any:
    if len(path) == 0:
        return new
    parent = at(root, path[:-1])
    parent[path[-1]] = new
    return root


def delete(root: Any, path: Path_) -> Any:
    parent = at(root, path[:-1])
    del parent[path[-1]]
    return root


def mutations(document: Any, budget: str) -> list[tuple[str, Any]]:
    """Each mutation of `document`, by name. `budget` is `full`, or `light` for a document whose readers the fast path does not touch."""
    cases: list[tuple[str, Any]] = [("as is", structural_copy(document))]
    all_paths = paths(document, limit=8) if budget == "full" else paths(document, limit=3)[:60]
    junk = JUNK if budget == "full" else JUNK[::3]
    for path in all_paths:
        label = "/".join(map(str, path)) or "<root>"
        held = at(document, path)
        if len(path) != 0:
            cases.append((f"{label} deleted", delete(structural_copy(document), path)))
        for name, value in junk:
            cases.append((f"{label} = {name}", put(structural_copy(document), path, value)))
        if isinstance(held, dict):
            cases.append(
                (
                    f"{label} + extra key",
                    put(structural_copy(document), path, {**held, "zz_extra": 1}),
                )
            )
            cases.append(
                (f"{label} + non-string key", put(structural_copy(document), path, {**held, 7: 1}))
            )
            cases.append(
                (
                    f"{label} reversed",
                    put(
                        structural_copy(document),
                        path,
                        dict(reversed(list(structural_copy(held).items()))),
                    ),
                )
            )
            if "name" in held:
                for mu_label, mu in (("false", False), ("true", True), ("'yes'", "yes"), ("1", 1)):
                    cases.append(
                        (
                            f"{label} must_understand {mu_label}",
                            put(
                                structural_copy(document),
                                path,
                                {**structural_copy(held), "must_understand": mu},
                            ),
                        )
                    )
                for bad in BAD_NAMES:
                    cases.append(
                        (
                            f"{label} name {bad!r}",
                            put(
                                structural_copy(document),
                                path,
                                {**structural_copy(held), "name": bad},
                            ),
                        )
                    )
                if "configuration" in held and isinstance(held["configuration"], dict):
                    cases.append(
                        (
                            f"{label} configuration reversed",
                            put(
                                structural_copy(document),
                                path,
                                {
                                    **structural_copy(held),
                                    "configuration": dict(
                                        reversed(
                                            list(structural_copy(held["configuration"]).items())
                                        )
                                    ),
                                },
                            ),
                        )
                    )
        if budget == "full":
            for k in (-2, -1, 0, 1):
                depth = 257 - len(path) + k
                cases.append(
                    (
                        f"{label} = lists {depth} deep",
                        put(structural_copy(document), path, nested_lists(depth)),
                    )
                )
            cases.append(
                (
                    f"{label} = dicts {257 - len(path)} deep",
                    put(structural_copy(document), path, nested_dicts(257 - len(path))),
                )
            )
            cases.append(
                (
                    f"{label} = dicts {256 - len(path)} deep",
                    put(structural_copy(document), path, nested_dicts(256 - len(path))),
                )
            )
    return cases


# --- readers -----------------------------------------------------------------


def guarded(call: Any) -> Any:
    try:
        return call()
    except Exception as error:  # noqa: BLE001
        return {"raised": type(error).__name__, "message": normalized(str(error))[:300]}


def model_record(model: Any) -> dict[str, object]:
    record: dict[str, object] = {"to_json": show(model.to_json())}
    record["to_key_value"] = guarded(
        lambda: {
            key: value.decode("utf-8", "replace") for key, value in model.to_key_value().items()
        }
    )
    again = guarded(lambda: type(model).from_json(model.to_json(), context=model.context))
    if isinstance(again, dict):
        record["again"] = again
    else:
        record["equal again"] = guarded(lambda: bool(model == again))
        record["hash again"] = guarded(lambda: hash(model) == hash(again))
    pickled = guarded(lambda: pickle.loads(pickle.dumps(model)))
    record["equal pickled"] = (
        pickled if isinstance(pickled, dict) else guarded(lambda: bool(model == pickled))
    )
    copied = guarded(lambda: copy.deepcopy(model))
    record["equal copied"] = (
        copied if isinstance(copied, dict) else guarded(lambda: bool(model == copied))
    )
    if hasattr(model, "claims"):
        record["claims"] = guarded(
            lambda: sorted(
                f"{kind.__name__}:{name}={None if d is None else d.name}"
                for (kind, name), d in model.claims.items()
            )
        )
    if hasattr(model, "refines"):
        record["refines self"] = guarded(lambda: model.refines(model))
    return record


def reading_record(reading: Any) -> dict[str, object]:
    record: dict[str, object] = {"problems": show(tuple(reading.problems))}
    if hasattr(reading, "fields"):
        record["fields"] = guarded(
            lambda: [
                [
                    ".".join(map(str, loc)),
                    show(field),
                    show(problems),
                    show(canonical_of(field, problems)),
                ]
                for loc, field, problems in with_problems(reading.fields(), reading.problems)
            ]
        )
    if hasattr(reading, "pipeline"):
        record["pipeline"] = guarded(lambda: [stage_record(stage) for stage in reading.pipeline])
        record["chunk"] = guarded(
            lambda: [show(reading.chunk.lengths), show(reading.chunk.data_type)]
        )
    if getattr(reading, "metadata", None) is not None:
        record["model"] = guarded(lambda: model_record(reading.metadata))
    return record


def stage_record(stage: Any) -> object:
    return [
        show(stage.codec),
        None
        if stage.incoming is None
        else [show(stage.incoming.lengths), show(stage.incoming.data_type)],
        {
            member: [stage_record(inner) for inner in stages]
            for member, stages in stage.inner.items()
        },
    ]


def read_v3_array(value: object) -> dict[str, object]:
    return {
        "extensions": guarded(
            lambda: reading_record(read_array_metadata_v3(value, context=CORE_AND_EXTENSIONS))
        ),
        "core": guarded(lambda: reading_record(read_array_metadata_v3(value, context=CORE))),
    }


def read_v3_group(value: object) -> dict[str, object]:
    return {
        "extensions": guarded(
            lambda: reading_record(read_group_metadata_v3(value, context=CORE_AND_EXTENSIONS))
        )
    }


def read_v3_consolidated(value: object) -> dict[str, object]:
    return {"model": guarded(lambda: model_record(ZarrV3ConsolidatedMetadata.from_json(value)))}


def read_v2_array(value: object) -> dict[str, object]:
    return {"core": guarded(lambda: reading_record(read_array_metadata_v2(value)))}


def read_v2_group(value: object) -> dict[str, object]:
    return {
        "problems": guarded(lambda: show(validate_group_metadata_v2(value))),
        "model": guarded(lambda: model_record(ZarrV2GroupMetadata.from_json(value))),
    }


def read_v2_consolidated(value: object) -> dict[str, object]:
    return {"model": guarded(lambda: model_record(ZarrV2ConsolidatedMetadata.from_json(value)))}


KINDS = {
    "codec": CodecDefinition,
    "chunk_grid": ChunkGridDefinition,
    "chunk_key_encoding": ChunkKeyEncodingDefinition,
    "data_type": DataTypeDefinition,
    "storage_transformer": StorageTransformerDefinition,
}


def one_field(value: object, kind: Any, scope: Any) -> object:
    field, problems = resolve(value, kind, scope)
    return [show(field), show(problems), show(canonical_of(field, problems))]


def read_field(kind_name: str) -> Any:
    kind = KINDS[kind_name]

    def read(value: object) -> dict[str, object]:
        out: dict[str, object] = {}
        for scope_name, scope in (("extensions", CORE_AND_EXTENSIONS), ("core", CORE)):
            out[scope_name] = guarded(lambda scope=scope: one_field(value, kind, scope))
        return out

    return read


READERS = {
    "v3 array": read_v3_array,
    "v3 group": read_v3_group,
    "v3 consolidated": read_v3_consolidated,
    "v2 array": read_v2_array,
    "v2 group": read_v2_group,
    "v2 consolidated": read_v2_consolidated,
}


# --- the corpus --------------------------------------------------------------


def classify(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    if value.get("zarr_format") == 3:
        return "v3 array" if value.get("node_type") == "array" else "v3 group"
    if value.get("zarr_format") == 2:
        return "v2 array" if "shape" in value else "v2 group"
    if value.get("kind") == "inline":
        return "v3 consolidated"
    if "zarr_consolidated_format" in value:
        return "v2 consolidated"
    return None


def corpus() -> list[tuple[str, str, Any, str]]:
    """Each document: its id, the reader it is for, the document, and its mutation budget."""
    documents: list[tuple[str, str, Any, str]] = [
        ("benchmark v3 array", "v3 array", V3_ARRAY, "full"),
        ("benchmark v2 array", "v2 array", V2_ARRAY, "light"),
        ("benchmark v3 group", "v3 group", V3_GROUP, "light"),
    ]
    fill_values: dict[str, dict[str, Any]] = {}
    data_types: dict[str, dict[str, Any]] = {}
    for path in sorted(TESTS.rglob("*.json")):
        relative = path.relative_to(TESTS).as_posix()
        value = json.loads(path.read_text())
        kind = classify(value)
        if kind is not None:
            documents.append((relative, kind, value, "full" if kind == "v3 array" else "light"))
            continue
        parts = path.relative_to(TESTS).parts
        if path.name == "cases.json" and parts[0] == "v3" and parts[1] in KINDS:
            for name, case in value.items():
                documents.append((f"{relative}:{name}", f"field:{parts[1]}", case, "light"))
        elif path.name == "fill_values.json":
            fill_values[parts[2]] = value
        elif path.name == "data_type.json":
            data_types[parts[2]] = value
    # A v3 array document for each data type and fill value the fixtures write.
    for type_name, cases in sorted(fill_values.items()):
        written: list[tuple[str, object]] = []
        if type_name in data_types:
            written = list(data_types[type_name].items())
        elif type_name == "raw":
            written = [("r8", "r8"), ("r16", "r16"), ("r24", "r24")]
        else:
            written = [(type_name, type_name)]
        for type_label, data_type in written:
            for fill_label, fill_value in cases.items():
                document = {
                    **structural_copy(V3_ARRAY),
                    "data_type": structural_copy(data_type),
                    "fill_value": structural_copy(fill_value),
                }
                documents.append(
                    (
                        f"derived {type_name}/{type_label}/{fill_label}",
                        "v3 array",
                        document,
                        "derived",
                    )
                )
    return documents


def derived_mutations(document: Any) -> list[tuple[str, Any]]:
    """For a derived document: the members that matter, and the fill value and data type in full."""
    cases: list[tuple[str, Any]] = [("as is", structural_copy(document))]
    for key in ("fill_value", "data_type"):
        for path in paths(document[key], (key,), limit=6):
            label = "/".join(map(str, path))
            if len(path) > 1:
                cases.append((f"{label} deleted", delete(structural_copy(document), path)))
            for name, value in JUNK:
                cases.append((f"{label} = {name}", put(structural_copy(document), path, value)))
            held = at(document, path)
            if isinstance(held, dict):
                cases.append(
                    (
                        f"{label} reversed",
                        put(
                            structural_copy(document),
                            path,
                            dict(reversed(list(structural_copy(held).items()))),
                        ),
                    )
                )
                cases.append(
                    (
                        f"{label} + extra key",
                        put(structural_copy(document), path, {**held, "zz_extra": 1}),
                    )
                )
    cases.append(("codecs[0] = bytes bare", put(structural_copy(document), ("codecs", 0), "bytes")))
    cases.append(
        (
            "codecs[0] = bytes big",
            put(
                structural_copy(document),
                ("codecs", 0),
                {"name": "bytes", "configuration": {"endian": "big"}},
            ),
        )
    )
    cases.append(
        (
            "codecs = [vlen-utf8]",
            put(structural_copy(document), ("codecs",), [{"name": "vlen-utf8"}]),
        )
    )
    return cases


def main() -> None:
    """`out_path`, then `--full` to write each record in full rather than its hash, then `--only <substring>` to run the cases whose id holds it."""
    out_path = Path(sys.argv[1])
    full = "--full" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for document_id, reader_name, document, budget in corpus():
            if reader_name.startswith("field:"):
                reader = read_field(reader_name.split(":", 1)[1])
            else:
                reader = READERS[reader_name]
            if budget == "derived":
                cases = derived_mutations(document)
            else:
                cases = mutations(document, budget)
            for case_label, case in cases:
                if only is not None and only not in f"{document_id} | {case_label}":
                    continue
                record = reader(case)
                text = json.dumps(record, ensure_ascii=False, allow_nan=True, sort_keys=False)
                case_id = f"{document_id} | {case_label}"
                if full:
                    out.write(
                        json.dumps(
                            {"id": case_id, "record": record}, ensure_ascii=False, allow_nan=True
                        )
                        + "\n"
                    )
                else:
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
                    out.write(f"{case_id}\t{digest}\n")
                count += 1
    print(f"{count} cases -> {out_path}", file=sys.stderr)
    if STATS:
        print(f"fast path: {dict(STATS)}", file=sys.stderr)


if __name__ == "__main__":
    main()
