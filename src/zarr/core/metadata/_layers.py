"""Reading metadata through zarr-metadata's layers: a consumer proof.

A v3 array document is read in four steps, each needing more than the one
before it:

1. `rewrite_legacy_v3`: JSON to JSON. Spellings writers have produced
   that the spec does not allow -- older zarr-python releases among them
   -- are rewritten into the spec's spelling, each with a warning. This is
   the one place a compatibility shim lives; everything after it reads
   the spec.
2. `well_formed_array_v3`: JSON, and the document's shape.
3. `read_array_v3`: each extension point read in a scope.
4. `refine_array_v3`: the fill value, the grid and the codec pipeline
   judged against the array. Its value is the resolved pipeline: each
   codec, and the array it is handed.

What the layers report is raised as zarr-python's `MetadataValidationError`
before zarr-python parses anything itself, and so is a top-level field that
must be understood, since zarr-python understands none. v2 arrays and
groups are judged by `zarr_metadata.rules`.

`ZARR_METADATA_LAYERS` picks the mode: `enforce` raises and reads the
rewritten document; `record` (the default on this proof branch) never
raises, so zarr-python behaves as it does without this module; `off`
skips the layers. With `ZARR_METADATA_LAYERS_LOG` naming a
directory, every document read or written is logged there with the
layers' verdict and zarr-python's outcome, and each array read compares
the codecs the resolved pipeline evolves (`codecs_from_pipeline`) with the
ones `ArrayV3Metadata.__init__` evolves by threading an `ArraySpec`
through `resolve_metadata`.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeGuard

from zarr.errors import MetadataValidationError, ZarrUserWarning

if TYPE_CHECKING:
    from zarr_metadata.v3.entity import Pipeline, RefinedArrayV3

    from zarr.abc.codec import Codec
    from zarr.core.array_spec import ArraySpec
    from zarr.core.dtype import ZDType
    from zarr.core.metadata.v3 import ArrayV3Metadata


MODE: Final = os.environ.get("ZARR_METADATA_LAYERS", "record")
LOG_DIR: Final = os.environ.get("ZARR_METADATA_LAYERS_LOG")

_INTEGER: Final = re.compile(r"u?int(8|16|32|64)")
_FLOAT: Final = re.compile(r"float(16|32|64)")
_FLOAT_SPELLINGS: Final = ("NaN", "Infinity", "-Infinity")


# 1. The rewrite stage.


def rewrite_legacy_v3(document: Mapping[str, object]) -> tuple[dict[str, object], tuple[str, ...]]:
    """`document` with the known deviations written in the spec's spelling, and a note for each.

    JSON to JSON, before anything is judged, so the layers read only the
    spec. Each rewrite recognizes one historical spelling exactly and
    leaves anything else to be judged as written.
    """
    rewritten = dict(document)
    notes: list[str] = []
    grid = _mixed_regular_grid(rewritten.get("chunk_grid"))
    if grid is not None:
        rewritten["chunk_grid"] = grid
        notes.append(
            "A 'regular' chunk grid listed chunk edges for some axis, as zarr-python 3.2.0 and "
            "3.2.1 wrote for a chunk spec mixing sizes and edge lists; it is read as the "
            "rectilinear grid those releases laid the chunks out as. Rewrite the metadata to "
            "keep this warning from recurring."
        )
    grid = _zero_chunk_edges(rewritten.get("chunk_grid"), rewritten.get("shape"))
    if grid is not None:
        rewritten["chunk_grid"] = grid
        notes.append(
            "A regular chunk grid gave an empty axis a chunk edge of 0 or false, as zarr-python "
            "3.0 and 3.1 wrote; the edge is read as 1, which lays out the same (zero) chunks "
            "and is one the spec allows."
        )
    fill = _numeric_string_fill(rewritten.get("data_type"), rewritten.get("fill_value"))
    if fill is not None:
        rewritten["fill_value"] = fill
        notes.append(
            f"The fill value {document.get('fill_value')!r} is a string for the numeric data "
            f"type {rewritten.get('data_type')!r}; it is read as the number {fill!r}."
        )
    return rewritten, tuple(notes)


def _is_list(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _mixed_regular_grid(grid: object) -> dict[str, object] | None:
    """A 'regular' grid whose `chunk_shape` lists edges for an axis, as the rectilinear grid it is."""
    if not isinstance(grid, Mapping) or grid.get("name") != "regular":
        return None
    configuration = grid.get("configuration")
    if not isinstance(configuration, Mapping):
        return None
    chunk_shape = configuration.get("chunk_shape")
    if not _is_list(chunk_shape) or not any(_is_list(edge) for edge in chunk_shape):
        return None
    return {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": list(chunk_shape)},
    }


def _zero_chunk_edges(grid: object, shape: object) -> dict[str, object] | None:
    """A regular grid with a 0 or false edge on an empty axis, the edge made 1."""
    if not isinstance(grid, Mapping) or grid.get("name") != "regular" or not _is_list(shape):
        return None
    configuration = grid.get("configuration")
    if not isinstance(configuration, Mapping):
        return None
    edges = configuration.get("chunk_shape")
    if not _is_list(edges) or len(edges) != len(shape):
        return None
    fixed = list(edges)
    for index, (edge, extent) in enumerate(zip(fixed, shape, strict=True)):
        zero = edge is False or (type(edge) is int and edge == 0)
        if zero and type(extent) is int and extent == 0:
            fixed[index] = 1
    if fixed == list(edges):
        return None
    return {**grid, "configuration": {**configuration, "chunk_shape": fixed}}


def _numeric_string_fill(data_type: object, fill_value: object) -> int | float | None:
    """A fill value written as a numeric string for a bare integer or float type, as the number."""
    if not isinstance(data_type, str) or not isinstance(fill_value, str):
        return None
    if _INTEGER.fullmatch(data_type):
        try:
            return int(fill_value)
        except ValueError:
            return None
    if _FLOAT.fullmatch(data_type) and fill_value not in _FLOAT_SPELLINGS:
        if fill_value.lower().startswith("0x"):
            return None
        try:
            number = float(fill_value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


# 2-4. The layers.


@dataclass(frozen=True)
class Reading:
    """One document read through the layers: what they said, before zarr-python parses it."""

    kind: str
    document: dict[str, Any]
    """The document as rewritten, which zarr-python goes on to parse when enforcing."""
    notes: tuple[str, ...]
    problems: tuple[str, ...]
    unknown_fields: tuple[str, ...]
    refined: RefinedArrayV3 | None

    original: dict[str, Any]
    """The document as given, which zarr-python parses when not enforcing."""

    @property
    def refused(self) -> bool:
        return len(self.problems) != 0 or len(self.unknown_fields) != 0


def read_array_v3(data: Mapping[str, object]) -> Reading:
    """A v3 array document rewritten and read through all three layers; raises when enforcing."""
    from zarr_metadata.v3.entity import read_array_v3, refine_array_v3, well_formed_array_v3

    from zarr.core.metadata._scope import SCOPE

    if MODE == "off":
        return Reading("v3-array", dict(data), (), (), (), None, dict(data))
    document, notes = rewrite_legacy_v3(data)
    if MODE == "enforce":
        for note in notes:
            warnings.warn(note, ZarrUserWarning, stacklevel=3)
    well, problems = well_formed_array_v3(document)
    refined = None
    unknown: tuple[str, ...] = ()
    if well is not None:
        array, read = read_array_v3(well, SCOPE)
        refined, composed = refine_array_v3(array)
        problems = (*problems, *read, *composed)
        unknown = tuple(array.must_understand_fields)
    reading = Reading(
        "v3-array", document, notes, _described(problems), unknown, refined, dict(data)
    )
    _enforce(reading)
    return reading


def read_node(kind: str, data: Mapping[str, object]) -> Reading:
    """A v2 array or a group document judged by `zarr_metadata.rules`; raises when enforcing."""
    from zarr_metadata import rules

    validate: dict[str, Callable[[object], Sequence[object]]] = {
        "v2-array": rules.validate_array_metadata_v2,
        "v3-group": rules.validate_group_metadata_v3,
        "v2-group": rules.validate_group_metadata_v2,
    }
    document = dict(data)
    reading = Reading(
        kind, document, (), _described(validate[kind](document)), (), None, dict(data)
    )
    _enforce(reading)
    return reading


_V2_GROUP_KEYS: Final = frozenset({"zarr_format", "attributes"})


def rewrite_legacy_v2_group(
    document: Mapping[str, object],
) -> tuple[dict[str, object], tuple[str, ...]]:
    """A v2 group document without the members two writers put in a `.zgroup`, and a note for each.

    The spec says other keys "MUST NOT be present" in a `.zgroup`. NCZarr
    writes `_nczarr_*` members there, and zarr-python 3 writes a
    `consolidated_metadata` member into each nested group's `.zgroup`
    entry in a `.zmetadata`. Both are dropped; zarr-python has always
    ignored them.
    """
    rewritten = dict(document)
    notes: list[str] = []
    nczarr = sorted(key for key in rewritten if key.startswith("_nczarr_"))
    for key in nczarr:
        del rewritten[key]
    if nczarr:
        notes.append(f"Dropped the NCZarr members {nczarr} from a .zgroup, which allows none.")
    if "consolidated_metadata" in rewritten:
        del rewritten["consolidated_metadata"]
        notes.append(
            "Dropped the 'consolidated_metadata' member zarr-python 3 writes into a nested "
            "group's .zgroup entry in .zmetadata; a .zgroup allows none."
        )
    return rewritten, tuple(notes)


def read_v2_group(data: Mapping[str, object]) -> Reading:
    """A v2 group document, `.zattrs` merged as `attributes`, rewritten and judged."""
    document, notes = rewrite_legacy_v2_group(data)
    if MODE == "enforce":
        for note in notes:
            warnings.warn(note, ZarrUserWarning, stacklevel=3)
    reading = read_node("v2-group", document)
    return Reading(reading.kind, reading.document, notes, reading.problems, (), None, dict(data))


def read_zmetadata(consolidated: dict[str, Any]) -> dict[str, Any]:
    """A v2 `.zmetadata`: each node it holds rewritten and judged; the rewritten document back.

    Returns the document zarr-python goes on to read -- rewritten when
    enforcing, as it was otherwise.
    """
    entries = consolidated.get("metadata")
    if not isinstance(entries, Mapping):
        return consolidated
    rewritten = dict(entries)
    for key, entry in entries.items():
        if not isinstance(key, str) or not isinstance(entry, Mapping):
            continue
        prefix, _, name = key.rpartition("/")
        attributes = entries.get(f"{prefix}/.zattrs" if prefix else ".zattrs", {})
        if name == ".zarray":
            reading = read_node("v2-array", {**entry, "attributes": attributes})
            log("zmetadata:v2-array", layers=reading.problems, document=reading.document)
        elif name == ".zgroup":
            reading = read_v2_group({**entry, "attributes": attributes})
            log(
                "zmetadata:v2-group",
                layers=reading.problems,
                notes=reading.notes,
                document=reading.document,
            )
            rewritten[key] = {k: v for k, v in reading.document.items() if k != "attributes"}
    if MODE != "enforce":
        return consolidated
    return {**consolidated, "metadata": rewritten}


def _described(problems: Sequence[object]) -> tuple[str, ...]:
    return tuple(str(problem) for problem in problems)


def _enforce(reading: Reading) -> None:
    if MODE != "enforce" or not reading.refused:
        return
    lines = list(reading.problems)
    lines.extend(
        f"{field}: a field zarr-python does not understand, and it does not say "
        "`must_understand: false`"
        for field in reading.unknown_fields
    )
    kind = reading.kind.replace("-", " ")
    raise MetadataValidationError(f"Invalid Zarr {kind} metadata:\n" + "\n".join(lines))


# Proof instrumentation.


def log(kind: str, **fields: object) -> None:
    """Append one record to this process's log, if a log directory is set."""
    if LOG_DIR is None:
        return
    path = Path(LOG_DIR) / f"{os.getpid()}.jsonl"
    with path.open("a") as sink:
        test = os.environ.get("PYTEST_CURRENT_TEST")
        sink.write(json.dumps({"kind": kind, "test": test, **fields}, default=repr) + "\n")


def settle[T](reading: Reading, parse: Callable[[dict[str, Any]], T]) -> T:
    """zarr-python's own parse of the rewritten document, logged beside the layers' verdict."""
    if LOG_DIR is None:
        return parse(reading.document if MODE == "enforce" else reading.original)
    outcome: dict[str, object] = {
        "document": reading.document,
        "notes": reading.notes,
        "layers": reading.problems,
        "unknown_fields": reading.unknown_fields,
    }
    try:
        parsed = parse(reading.document if MODE == "enforce" else reading.original)
    except Exception as error:
        log(reading.kind, zarr="raised", error=f"{type(error).__name__}: {error}", **outcome)
        raise
    log(reading.kind, zarr="ok", **outcome)
    return parsed


def judge_written(kind: str, document: Mapping[str, object]) -> None:
    """Log the layers' verdict on a document zarr-python is writing; never raises."""
    if LOG_DIR is None:
        return
    try:
        if kind == "v3-array":
            from zarr_metadata.rules import validate_array_metadata_v3

            problems = _described(validate_array_metadata_v3(document))
        else:
            problems = read_node_problems(kind, document)
    except Exception as error:  # noqa: BLE001 -- a write is logged, never refused
        log(f"write:{kind}", crash=repr(error), document=document)
        return
    log(f"write:{kind}", layers=problems, document=document)


def read_node_problems(kind: str, document: Mapping[str, object]) -> tuple[str, ...]:
    from zarr_metadata import rules

    validate: dict[str, Callable[[object], Sequence[object]]] = {
        "v2-array": rules.validate_array_metadata_v2,
        "v3-group": rules.validate_group_metadata_v3,
        "v2-group": rules.validate_group_metadata_v2,
    }
    return _described(validate[kind](document))


def compare_pipeline(metadata: ArrayV3Metadata, reading: Reading) -> None:
    """Log whether the resolved pipeline evolves the codecs zarr-python evolved."""
    if LOG_DIR is None or reading.refined is None or reading.refused:
        return
    try:
        report = _compare(metadata, reading.refined)
    except Exception as error:  # noqa: BLE001 -- the proof must never change behaviour
        report = {"crash": f"{type(error).__name__}: {error}"}
    log("pipeline", document=reading.document, **report)


def codecs_from_pipeline(
    refined: RefinedArrayV3, shape: tuple[int, ...], fill_value: object
) -> tuple[tuple[Codec, ...] | None, str | None]:
    """zarr-python's codecs, each evolved against the array the resolved pipeline says it is handed.

    What `ArrayV3Metadata.__init__` finds by threading an `ArraySpec`
    through each codec's `resolve_metadata`, read off the pipeline
    instead. Two things the pipeline does not carry come from here: the
    array's shape, which zarr-python evolves against and a transpose
    permutes, and the spec a `bytes -> bytes` codec is evolved against,
    which in zarr-python is the one the `array -> bytes` codec was
    handed. None, with the reason, where a stage's data type is out of
    the scope the layers read in.
    """
    from zarr_metadata.v3.codec.transpose import TransposeCodec

    from zarr.core.metadata.v3 import parse_codecs

    written = refined.array.document.get("codecs")
    if not _is_list(written):
        return None, "no codecs"
    evolved: list[Codec] = []
    last: ArraySpec | None = None
    for index, stage in enumerate(refined.pipeline.stages):
        (codec,) = parse_codecs([written[index]])
        if stage.incoming is not None:
            dtype = _zdtype(stage.incoming.data_type)
            if dtype is None:
                return None, f"stage {index}: data type out of scope"
            last = _spec(shape, dtype, fill_value)
        elif last is None:
            return None, f"stage {index}: no array reaches it"
        evolved.append(codec.evolve_from_array_spec(last))
        if isinstance(stage.codec, TransposeCodec):
            order = stage.codec.configuration.order
            shape = tuple(shape[axis] for axis in order)
    return tuple(evolved), None


def _zdtype(data_type: object) -> ZDType[Any, Any] | None:
    from zarr_metadata.v3.entity import DataTypeEntity

    from zarr.core.dtype import get_data_type_from_json

    if not isinstance(data_type, DataTypeEntity):
        return None
    return get_data_type_from_json(data_type.to_json(), zarr_format=3)


def _spec(shape: tuple[int, ...], dtype: ZDType[Any, Any], fill_value: object) -> ArraySpec:
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer.core import default_buffer_prototype

    try:
        fill = dtype.cast_scalar(fill_value)
    except (TypeError, ValueError):
        fill = dtype.default_scalar()
    return ArraySpec(
        shape=shape,
        dtype=dtype,
        fill_value=fill,
        config=ArrayConfig.from_dict({}),
        prototype=default_buffer_prototype(),
    )


def _compare(metadata: ArrayV3Metadata, refined: RefinedArrayV3) -> dict[str, object]:
    report: dict[str, object] = {}
    mapped, declined = codecs_from_pipeline(refined, metadata.shape, metadata.fill_value)
    if mapped is None:
        report["mapping"] = f"declined: {declined}"
    else:
        mismatches = [
            {"index": index, "pipeline": repr(ours), "zarr": repr(theirs)}
            for index, (ours, theirs) in enumerate(zip(mapped, metadata.codecs, strict=True))
            if ours != theirs
        ]
        report["mapping"] = "equal" if len(mismatches) == 0 else "differs"
        if mismatches:
            report["mapping_mismatches"] = mismatches
    start = _spec(metadata.shape, metadata.data_type, metadata.fill_value)
    report["threading"] = list(_threading(refined.pipeline, metadata.codecs, start, ()))
    return report


def _threading(
    pipeline: Pipeline, codecs: Sequence[Codec], spec: ArraySpec, loc: tuple[object, ...]
) -> Iterator[dict[str, object]]:
    """Each stage where the pipeline's incoming data type and zarr-python's threaded one disagree."""
    from zarr.codecs.sharding import ShardingCodec

    for index, (stage, codec) in enumerate(zip(pipeline.stages, codecs, strict=True)):
        here = (*loc, index)
        if stage.incoming is not None:
            ours = _zdtype(stage.incoming.data_type)
            if ours is not None and ours != spec.dtype:
                yield {"loc": here, "pipeline": repr(ours), "zarr": repr(spec.dtype)}
        if isinstance(codec, ShardingCodec) and stage.incoming is not None:
            inner = _spec(codec.chunk_shape, spec.dtype, spec.fill_value)
            with contextlib.suppress(KeyError):
                yield from _threading(stage.inner["codecs"], codec.codecs, inner, (*here, "codecs"))
            from zarr.core.dtype import UInt64

            index_spec = _spec((*codec.chunk_shape, 2), UInt64(endianness="little"), 0)
            with contextlib.suppress(KeyError):
                yield from _threading(
                    stage.inner["index_codecs"], codec.index_codecs, index_spec, (*here, "index")
                )
        spec = codec.resolve_metadata(spec)


def judge_written_zmetadata(consolidated: Mapping[str, object]) -> None:
    """Log the layers' verdict on each node a v2 `.zmetadata` zarr-python is writing holds."""
    entries = consolidated.get("metadata")
    if not isinstance(entries, Mapping):
        log("write:v2-zmetadata", crash="metadata is not an object", document=consolidated)
        return
    for key, entry in entries.items():
        if not isinstance(key, str) or not isinstance(entry, Mapping):
            continue
        prefix, _, name = key.rpartition("/")
        attributes = entries.get(f"{prefix}/.zattrs" if prefix else ".zattrs", {})
        if name == ".zarray":
            judge_written("v2-array", {**entry, "attributes": attributes})
        elif name == ".zgroup":
            judge_written("v2-group", {**entry, "attributes": attributes})
