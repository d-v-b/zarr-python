"""
Scenarios for the zarr compatibility differential (see `README.md`).

Each scenario drives zarr the way existing code does (the synchronous `Array` API,
`AsyncArray` used as a dataclass, pickle, copy, subclasses, and stored documents written
by other software) and records what a caller can observe. `run.py` runs the scenarios
under one zarr source tree, and `diff.py` compares two runs.

This module must import under every zarr version it runs against, so it uses only names
each of them has. It is imported as `compat_scenarios` in every process, so a pickle of
one of its subclasses loads in either version.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import pickle
import random
import re
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import zarr
import zarr.core.sync
import zarr.storage
from zarr.core.array import AsyncArray
from zarr.core.buffer import default_buffer_prototype

_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
_MEMORY_STORE = re.compile(r"memory://\d+")
_METADATA_KEYS = (".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json")
_MAX_ELEMENTS = 512


class Recorder:
    """Records what each step of a scenario returned, raised, and warned."""

    def __init__(self, tmp: Path) -> None:
        self.events: list[dict[str, Any]] = []
        self.tmp = str(tmp)
        self.scenario = ""

    def norm(self, text: str) -> str:
        text = _MEMORY_STORE.sub("memory://?", _ADDRESS.sub("0x?", text))
        return text.replace(self.tmp, "<tmp>")

    def step(self, label: str, fn: Any) -> tuple[bool, Any]:
        """Run `fn`, record its outcome under `label`, and return whether it returned
        and what."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                value = fn()
            except Exception as e:  # noqa: BLE001 (any exception is an outcome to compare)
                outcome: dict[str, Any] = {
                    "raises": f"{type(e).__module__}.{type(e).__qualname__}",
                    "message": self.norm(str(e)),
                }
                ok = False
                value = None
            else:
                outcome = {"value": self.summarize(value)}
                ok = True
        if caught:
            outcome["warnings"] = [[w.category.__name__, self.norm(str(w.message))] for w in caught]
        self.events.append({"scenario": self.scenario, "label": label, **outcome})
        return ok, value

    def summarize(self, value: Any) -> Any:
        """A JSON value that compares equal across processes exactly when what a caller
        sees of `value` does."""
        if isinstance(value, str):
            return self.norm(value)
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else repr(value)
        if isinstance(value, complex):
            return {"complex": [self.summarize(value.real), self.summarize(value.imag)]}
        if isinstance(value, np.ndarray):
            flat = value.reshape(-1)[:_MAX_ELEMENTS]
            return {
                "ndarray": value.dtype.str,
                "shape": list(value.shape),
                "data": [self.summarize(v) for v in flat.tolist()],
            }
        if isinstance(value, np.generic):
            return {"scalar": value.dtype.str, "value": self.summarize(value.item())}
        if isinstance(value, bytes | bytearray | memoryview):
            raw = bytes(value)
            return {"bytes": hashlib.sha1(raw).hexdigest()[:16], "len": len(raw)}
        if isinstance(value, tuple):
            return {"tuple": [self.summarize(v) for v in value]}
        if isinstance(value, list):
            return [self.summarize(v) for v in value]
        if isinstance(value, dict):
            return {"dict": [[self.summarize(k), self.summarize(v)] for k, v in value.items()]}
        if isinstance(value, zarr.Array):
            return {"Array": type(value).__qualname__, "shape": list(value.shape)}
        if isinstance(value, AsyncArray):
            return {"AsyncArray": type(value).__qualname__, "shape": list(value.shape)}
        return {"type": type(value).__qualname__, "repr": self.norm(repr(value))}


# Stores ---------------------------------------------------------------------------------


def snapshot(store: Any) -> dict[str, Any]:
    """Every key of `store`: the text of a metadata document, or the digest of a chunk."""
    if isinstance(store, zarr.storage.LocalStore):
        root = Path(store.root)
        items = {
            p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
        }
    else:
        items = {k: v.to_bytes() for k, v in store._store_dict.items()}
    out: dict[str, Any] = {}
    for key in sorted(items):
        raw = items[key]
        if key.rsplit("/", 1)[-1] in _METADATA_KEYS:
            out[key] = raw.decode()
        else:
            out[key] = hashlib.sha1(raw).hexdigest()[:16]
    return out


def new_store(kind: str, tmp: Path, name: str) -> Any:
    if kind == "memory":
        return zarr.storage.MemoryStore()
    path = tmp / name
    shutil.rmtree(path, ignore_errors=True)
    return zarr.storage.LocalStore(path)


def put_document(store: Any, key: str, document: dict[str, Any]) -> None:
    """Write `document` at `key` as other software would: compact JSON, bytes as given."""
    text = json.dumps(document).encode()
    if isinstance(store, zarr.storage.LocalStore):
        path = Path(store.root) / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text)
    else:
        store._store_dict[key] = default_buffer_prototype().buffer.from_bytes(text)


# Observations ---------------------------------------------------------------------------


def observe(rec: Recorder, name: str, arr: Any, store: Any) -> None:
    """Record what a caller sees of `arr` and its store."""
    if arr is None:
        rec.step(f"{name}:missing", lambda: None)
        return
    rec.step(f"{name}:shape", lambda: arr.shape)
    rec.step(f"{name}:chunks", lambda: arr.chunks)
    rec.step(f"{name}:shards", lambda: arr.shards)
    rec.step(f"{name}:dtype", lambda: str(arr.dtype))
    rec.step(f"{name}:fill_value", lambda: arr.fill_value)
    rec.step(f"{name}:order", lambda: arr.order)
    rec.step(f"{name}:attrs", lambda: arr.attrs.asdict())
    rec.step(f"{name}:metadata", lambda: arr.metadata.to_dict())
    rec.step(f"{name}:metadata.type", lambda: type(arr.metadata).__qualname__)
    rec.step(f"{name}:read", lambda: arr[...])
    rec.step(f"{name}:nchunks_initialized", lambda: arr.nchunks_initialized)
    rec.step(f"{name}:read_only", lambda: arr.read_only)
    rec.step(f"{name}:path", lambda: (arr.path, arr.name, arr.basename))
    rec.step(f"{name}:store", lambda: snapshot(store))


def observe_surface(rec: Recorder, name: str, arr: Any, other: Any) -> None:
    """Record `arr` as a dataclass and an object: what code that treats it as one sees."""
    aa = arr.async_array
    rec.step(f"{name}:fields", lambda: [f.name for f in dataclasses.fields(aa)])
    rec.step(
        f"{name}:fields.init",
        lambda: [(f.name, f.init, f.compare, f.repr) for f in dataclasses.fields(type(aa))],
    )
    rec.step(f"{name}:Array.fields", lambda: [f.name for f in dataclasses.fields(arr)])
    rec.step(f"{name}:eq.self", lambda: (aa == aa, arr == arr))  # noqa: PLR0124
    rec.step(f"{name}:eq.copy", lambda: aa == copy.copy(aa))
    rec.step(f"{name}:eq.other", lambda: (aa == other.async_array, arr == other))
    rec.step(f"{name}:hash", lambda: hash(aa) == hash(copy.copy(aa)))
    rec.step(f"{name}:Array.hash", lambda: hash(arr) == hash(arr))
    rec.step(f"{name}:repr", lambda: (repr(arr), str(arr)))
    rec.step(f"{name}:AsyncArray.repr", lambda: repr(aa))
    rec.step(f"{name}:asdict.keys", lambda: list(dataclasses.asdict(aa)))
    rec.step(f"{name}:astuple.len", lambda: len(dataclasses.astuple(aa)))
    rec.step(f"{name}:codec_pipeline.type", lambda: type(aa.codec_pipeline).__qualname__)
    rec.step(f"{name}:config", lambda: aa.config)
    rec.step(f"{name}:store_path", lambda: str(aa.store_path))
    rec.step(f"{name}:metadata.is", lambda: aa.metadata is arr.metadata)
    rec.step(f"{name}:internal:vars", lambda: sorted(vars(aa)))
    rec.step(f"{name}:internal:chunk_grid.type", lambda: type(aa._chunk_grid).__qualname__)
    rec.step(f"{name}:frozen", lambda: setattr(aa, "metadata", aa.metadata))


# Random programs -------------------------------------------------------------------------

_DTYPES = ["int8", "uint16", "int32", "float32", "float64", "bool", "complex64"]
_ATTRIBUTES: list[dict[str, Any] | None] = [
    None,
    {},
    {"a": 1, "b": [1, 2]},
    {"z": 1, "a": {"n": None}},
]


def _selection(rng: random.Random, shape: list[int]) -> tuple[str, Any]:
    """A selection into an array of `shape`: its kind and the selection."""
    if not shape:
        return "basic", rng.choice([(), Ellipsis])
    if all(n > 0 for n in shape) and rng.random() < 0.3:
        kind = rng.choice(["oindex", "vindex"])
        if kind == "oindex":
            return kind, tuple(sorted(rng.sample(range(n), rng.randint(1, n))) for n in shape)
        k = rng.randint(1, 4)
        return kind, tuple([rng.randrange(n) for _ in range(k)] for n in shape)
    sel: list[Any] = []
    for n in shape:
        r = rng.random()
        if n > 0 and r < 0.25:
            sel.append(rng.randrange(n))
        else:
            start = rng.randint(0, n)
            stop = rng.randint(start, n + 2)
            sel.append(slice(start, stop, rng.choice([1, 1, 2])))
    return "basic", tuple(sel)


def _selected_shape(kind: str, sel: Any, shape: list[int]) -> tuple[int, ...]:
    dummy = np.empty(shape, dtype="u1")
    if kind == "oindex":
        return dummy[np.ix_(*sel)].shape
    if kind == "vindex":
        return dummy[tuple(np.asarray(s) for s in sel)].shape
    return dummy[sel].shape


def _values(rng: random.Random, dtype: str, shape: tuple[int, ...]) -> Any:
    if rng.random() < 0.2:
        return np.array(rng.randint(0, 5)).astype(dtype)[()]
    size = int(np.prod(shape))
    return (np.arange(size) % 7 + rng.randint(0, 3)).reshape(shape).astype(dtype)


def _index(arr: Any, kind: str) -> Any:
    return {"basic": arr, "oindex": arr.oindex, "vindex": arr.vindex}[kind]


def _write(arr: Any, kind: str, sel: Any, values: Any) -> None:
    _index(arr, kind)[sel] = values


def run_program(rec: Recorder, seed: int, tmp: Path) -> None:
    """A random sequence of operations on one array through several handles. The
    sequence depends only on `seed`, never on what zarr returns, so both runs of a
    seed perform the same operations."""
    rng = random.Random(seed)
    zarr_format = rng.choice([2, 3])
    ndim = rng.choice([0, 1, 1, 2, 2, 3])
    shape = [rng.choice([0, 1, 3, 4, 7]) for _ in range(ndim)]
    chunks = [rng.choice([1, 2, 3, 5, 8]) for _ in range(ndim)]
    shards = None
    if zarr_format == 3 and ndim > 0 and rng.random() < 0.4:
        shards = [c * rng.choice([1, 2]) for c in chunks]
    dtype = rng.choice(_DTYPES)
    fill_value = rng.choice([None, 0, 1])
    if dtype == "bool" and fill_value is not None:
        fill_value = bool(fill_value)
    kwargs: dict[str, Any] = {
        "shape": shape,
        "chunks": chunks if ndim else "auto",
        "dtype": dtype,
        "zarr_format": zarr_format,
        "attributes": rng.choice(_ATTRIBUTES),
        "compressors": rng.choice(["auto", None]),
    }
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    if shards is not None:
        kwargs["shards"] = shards
    if zarr_format == 3 and ndim and rng.random() < 0.3:
        kwargs["dimension_names"] = [f"d{i}" for i in range(ndim)]
    if zarr_format == 2 and rng.random() < 0.3:
        kwargs["order"] = "F"
    if rng.random() < 0.3:
        kwargs["chunk_key_encoding"] = (
            {"name": "v2", "separator": rng.choice([".", "/"])}
            if zarr_format == 2 or rng.random() < 0.5
            else {"name": "default", "separator": "."}
        )
    if rng.random() < 0.3:
        kwargs["config"] = {"write_empty_chunks": rng.choice([True, False])}
    store_kind = rng.choice(["memory", "local"])
    store = new_store(store_kind, tmp, f"program-{seed}")
    path = rng.choice(["", "a", "g/a"])
    rec.step("create.kwargs", lambda: {k: str(v) for k, v in kwargs.items()})

    legacy = None
    if ndim and rng.random() < 0.35:
        # A document another writer stored, with chunk sizes zarr reads only by repairing
        # them: 0 or JSON false (an empty axis's `chunks=-1`), or JSON true.
        stored = [
            rng.choice([c, c, 0, True] + ([False] if zarr_format == 3 else [])) for c in chunks
        ]
        legacy = _legacy_document(zarr_format, shape, stored, dtype, kwargs.get("attributes"))
    if legacy is None:
        ok, arr = rec.step("create", lambda: zarr.create_array(store, name=path, **kwargs))
    else:
        rec.step("legacy.document", lambda: legacy)
        prefix = f"{path}/" if path else ""
        for key, document in legacy.items():
            put_document(store, prefix + key, document)
        ok, arr = rec.step("create", lambda: zarr.open_array(store, path=path, mode="r+"))
    handles: list[Any] = [arr if ok else None]
    observe(rec, "h0", arr if ok else None, store)

    for step in range(rng.randint(6, 14)):
        op = rng.choice(
            [
                "write", "write", "read", "resize", "append", "attrs", "reopen", "reopen_ro",
                "pickle", "copy", "deepcopy", "replace", "setattr", "with_config", "observe",
                "surface", "replace_metadata", "from_dict", "async_read", "attrs_in_place",
            ]
        )  # fmt: skip
        h = rng.randrange(len(handles))
        arr = handles[h]
        tag = f"{step}:{op}:h{h}"
        if op in ("write", "read"):
            kind, sel = _selection(rng, shape)
            values = _values(rng, dtype, _selected_shape(kind, sel, shape))
            if arr is None:
                rec.step(f"{tag}:skipped", lambda: None)
                continue
            if op == "write":
                rec.step(
                    f"{tag}:{kind}:{sel!r}",
                    lambda a=arr, k=kind, s=sel, v=values: _write(a, k, s, v),
                )
            else:
                rec.step(f"{tag}:{kind}:{sel!r}", lambda a=arr, k=kind, s=sel: _index(a, k)[s])
        elif op == "resize":
            new = [rng.choice([0, 1, 2, 5, 9]) for _ in shape]
            if arr is not None:
                rec.step(f"{tag}:{new}", lambda a=arr, n=new: a.resize(n))
            shape = new
        elif op == "append":
            if not shape:
                continue
            axis = rng.randrange(len(shape))
            extra = [n if i != axis else rng.randint(1, 3) for i, n in enumerate(shape)]
            data = (np.arange(int(np.prod(extra))) % 5).reshape(extra).astype(dtype)
            if arr is not None:
                rec.step(f"{tag}:{axis}", lambda a=arr, d=data, x=axis: a.append(d, axis=x))
            shape = [n + (extra[i] if i == axis else 0) for i, n in enumerate(shape)]
        elif op == "attrs":
            update = rng.choice([{}, {"a": 2}, {"new": [1, {"x": 1}]}, {"b": None, "a": 3}])
            if arr is not None:
                rec.step(f"{tag}:{update}", lambda a=arr, u=update: a.update_attributes(u))
        elif op in ("reopen", "reopen_ro"):
            mode = "r" if op == "reopen_ro" else "r+"
            ok, new_arr = rec.step(tag, lambda m=mode: zarr.open_array(store, path=path, mode=m))
            handles.append(new_arr if ok else None)
        elif op in (
            "pickle",
            "copy",
            "deepcopy",
            "replace",
            "setattr",
            "replace_metadata",
            "from_dict",
        ):
            if arr is None:
                handles.append(None)
                continue
            make = {
                "pickle": lambda a=arr: pickle.loads(pickle.dumps(a)),
                "copy": lambda a=arr: copy.copy(a),
                "deepcopy": lambda a=arr: copy.deepcopy(a),
                "replace": lambda a=arr: zarr.Array(dataclasses.replace(a.async_array)),
                "setattr": lambda a=arr: _setattr_copy(a),
                "replace_metadata": lambda a=arr: _replace_metadata(a),
                "from_dict": lambda a=arr: zarr.Array(
                    AsyncArray(a.metadata.to_dict(), a.async_array.store_path, a.async_array.config)
                ),
            }[op]
            ok, new_arr = rec.step(tag, make)
            handles.append(new_arr if ok else None)
        elif op == "with_config":
            if arr is None:
                handles.append(None)
                continue
            cfg = {"write_empty_chunks": rng.choice([True, False])}
            ok, new_arr = rec.step(tag, lambda a=arr, c=cfg: a.with_config(c))
            handles.append(new_arr if ok else None)
        elif op == "observe":
            observe(rec, tag, arr, store)
        elif op == "attrs_in_place":
            how = rng.choice(["set", "reorder", "nested"])
            if arr is not None:
                rec.step(f"{tag}:{how}", lambda a=arr, w=how: _change_attributes_in_place(a, w))
        elif op == "async_read":
            if arr is not None:
                rec.step(tag, lambda a=arr: zarr.core.sync.sync(a.async_array.getitem(...)))
        elif op == "surface":
            if arr is not None and handles[0] is not None:
                observe_surface(rec, tag, arr, handles[0])
    for i, arr in enumerate(handles):
        if arr is not None:
            rec.step(f"end:h{i}:read", lambda a=arr: a[...])
            rec.step(f"end:h{i}:attrs", lambda a=arr: a.attrs.asdict())
            rec.step(f"end:h{i}:shape", lambda a=arr: a.shape)
    observe(rec, "end:fresh", zarr.open_array(store, path=path, mode="r"), store)


def _legacy_document(
    zarr_format: int, shape: list[int], chunks: list[Any], dtype: str, attributes: Any
) -> dict[str, dict[str, Any]]:
    """The documents, by key, of an array stored by another writer with chunk sizes
    `chunks`."""
    fill: Any = False if dtype == "bool" else 0
    if zarr_format == 2:
        out = {
            ".zarray": {
                "zarr_format": 2, "shape": shape, "chunks": chunks, "dtype": np.dtype(dtype).str,
                "fill_value": fill, "order": "C", "filters": None, "compressor": None,
            }
        }  # fmt: skip
        if attributes:
            out[".zattrs"] = attributes
        return out
    document = _v3_doc(shape, chunks)
    document["data_type"] = dtype
    document["fill_value"] = [0.0, 0.0] if dtype == "complex64" else fill
    document["attributes"] = attributes or {}
    if dtype in ("int8", "bool"):
        document["codecs"] = [{"name": "bytes", "configuration": {}}]
    return {"zarr.json": document}


def _change_attributes_in_place(arr: Any, how: str) -> Any:
    """Change `arr`'s attributes in the metadata's own dict, as code that holds it does."""
    attributes = arr.metadata.attributes
    if how == "set":
        attributes["in_place"] = [1, 2]
    elif how == "reorder":
        for key in list(attributes):
            attributes[key] = attributes.pop(key)
        attributes["last"] = 0
    else:
        attributes.setdefault("nested", {"n": []})["n"].append(len(attributes))
    return dict(attributes)


def _replace_metadata(arr: Any) -> Any:
    """A copy of `arr` made by `dataclasses.replace` with metadata of other attributes."""
    metadata = dataclasses.replace(arr.metadata, attributes={"replaced": True})
    return zarr.Array(dataclasses.replace(arr.async_array, metadata=metadata))


def _setattr_copy(arr: Any) -> Any:
    """A copy of `arr` whose metadata is replaced the way frozen-dataclass code does it."""
    aa = copy.copy(arr.async_array)
    metadata = dataclasses.replace(arr.metadata, attributes={"set": "by setattr"})
    object.__setattr__(aa, "metadata", metadata)
    return zarr.Array(aa)


# Documents other software stored ---------------------------------------------------------


def _v2_doc(shape: list[int], chunks: list[Any]) -> dict[str, Any]:
    return {
        "zarr_format": 2,
        "shape": shape,
        "chunks": chunks,
        "dtype": "<i2",
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "compressor": None,
    }


def _v3_doc(
    shape: list[int], chunk_shape: list[Any], inner: list[Any] | None = None
) -> dict[str, Any]:
    bytes_codec: dict[str, Any] = {"name": "bytes", "configuration": {"endian": "little"}}
    codecs: list[Any] = [bytes_codec]
    if inner is not None:
        codecs = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": inner,
                    "codecs": [bytes_codec],
                    "index_codecs": [bytes_codec, {"name": "crc32c"}],
                    "index_location": "end",
                },
            }
        ]
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": "int16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk_shape}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": codecs,
        "attributes": {"z": 1, "a": 2},
    }


DOCUMENTS: dict[str, dict[str, Any]] = {
    "v2-valid": _v2_doc([10, 10], [4, 5]),
    "v3-valid-empty-axes": _v3_doc([0, 0], [1, 1]),
    "v3-valid-sharded": _v3_doc([10], [4], inner=[2]),
    "v2-zero-empty-axis": _v2_doc([0, 4], [0, 4]),
    "v3-false-empty-axis": _v3_doc([0], [False]),
    "v2-true": _v2_doc([5], [True]),
    "v3-true": _v3_doc([5, 4], [True, 4]),
    "v2-zero-grown-axis": _v2_doc([3], [0]),
    "v3-zero-grown-axis": _v3_doc([4, 3], [4, 0]),
    "v2-zero-empty-and-grown-axes": _v2_doc([0, 3], [0, 0]),
    "v3-sharded-zero-empty-axis": _v3_doc([0], [0], inner=[4]),
    "v3-sharded-zero-grown-axis": _v3_doc([10], [0], inner=[4]),
    "v3-sharded-zero-2d": _v3_doc([0, 3], [0, 3], inner=[2, 3]),
    "v3-sharded-true": _v3_doc([5], [True], inner=[True]),
    "v2-float-chunk": _v2_doc([4], [2.0]),
    "v3-float-chunk": _v3_doc([4], [2.0]),
    "v2-negative-chunk": _v2_doc([4], [-1]),
    "v3-edge-list-in-regular": _v3_doc([5], [[2, 3]]),
    "v2-fill-nan-int": {**_v2_doc([4], [2]), "fill_value": "NaN"},
    "v2-zattrs-fields": {**_v2_doc([4], [2]), "dimension_separator": "/"},
    "v3-extra-field": {**_v3_doc([4], [2]), "extra": {"must_understand": False}},
}


def run_document(rec: Recorder, name: str, tmp: Path) -> None:
    """Open an array whose document other software stored, through two handles, and
    write through one of them."""
    document = DOCUMENTS[name]
    zarr_format = document["zarr_format"]
    store = new_store("local", tmp, f"doc-{name}")
    if zarr_format == 2:
        put_document(store, ".zgroup", {"zarr_format": 2})
        put_document(store, "a/.zarray", document)
        put_document(store, "a/.zattrs", {"z": 1, "a": 2})
    else:
        put_document(store, "zarr.json", {"zarr_format": 3, "node_type": "group", "attributes": {}})
        put_document(store, "a/zarr.json", document)

    ok, first = rec.step("open", lambda: zarr.open_array(store, path="a", mode="r+"))
    observe(rec, "first", first if ok else None, store)
    ok2, second = rec.step("open.second", lambda: zarr.open_array(store, path="a", mode="r+"))
    if not ok:
        return
    rec.step("pickle.first", lambda: pickle.loads(pickle.dumps(first)).metadata.to_dict())
    shape = first.shape
    if all(n > 0 for n in shape):
        rec.step(
            "write.first",
            lambda: first.__setitem__(..., np.arange(int(np.prod(shape))).reshape(shape)),
        )
    rec.step("store.after_write", lambda: snapshot(store))
    if ok2:
        rec.step("second.read", lambda: second[...])
        rec.step("second.resize", lambda: second.resize([n + 2 for n in shape]))
        rec.step("second.write", lambda: second.__setitem__(..., 3))
        observe(rec, "second", second, store)
    rec.step("first.read_after_second", lambda: first[...])
    rec.step("first.update_attributes", lambda: first.update_attributes({}))
    rec.step("store.after_attrs", lambda: snapshot(store))
    _ = rec.step("consolidate", lambda: zarr.consolidate_metadata(store))
    rec.step("store.after_consolidate", lambda: snapshot(store))
    ok4, cgroup = rec.step("open_consolidated", lambda: zarr.open_consolidated(store, mode="r+"))
    if ok4:
        rec.step("consolidated.members", lambda: [k for k, _ in cgroup.members()])
        ok5, member = rec.step("consolidated.member", lambda: cgroup["a"])
        if ok5:
            observe(rec, "member", member, store)
            rec.step("member.write", lambda: member.__setitem__(..., 1))
            rec.step("group.to_dict", lambda: cgroup.metadata.to_dict())
    rec.step("store.end", lambda: snapshot(store))


# Subclasses --------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TaggedAsyncArray(AsyncArray):  # type: ignore[type-arg]
    """A dataclass subclass with a field of its own."""

    tag: str = "untagged"


class PlainAsyncArray(AsyncArray):  # type: ignore[type-arg]
    """A subclass that adds a method and nothing else."""

    def describe(self) -> str:
        return f"{type(self).__name__}{self.shape}"


class SlottedAsyncArray(AsyncArray):  # type: ignore[type-arg]
    """A subclass with a slot of its own."""

    __slots__ = ("note",)


class ReducingAsyncArray(AsyncArray):  # type: ignore[type-arg]
    """A subclass that pickles through a reduction of its own."""

    def __reduce__(self) -> Any:
        return (_rebuild_reducing, (self.metadata, self.store_path, self.config))


def _rebuild_reducing(metadata: Any, store_path: Any, config: Any) -> ReducingAsyncArray:
    return ReducingAsyncArray(metadata, store_path, config)


class InitAsyncArray(AsyncArray):  # type: ignore[type-arg]
    """A subclass whose `__init__` builds state of its own."""

    def __init__(self, metadata: Any, store_path: Any, config: Any = None) -> None:
        super().__init__(metadata, store_path, config)
        object.__setattr__(self, "built", tuple(self.shape))


class ClassAttrAsyncArray(AsyncArray):  # type: ignore[type-arg]
    """A subclass with class attributes, one of them named like an internal."""

    kind = "class-attr"
    _io = "subclass-owned"


SUBCLASSES = {
    "tagged": TaggedAsyncArray,
    "plain": PlainAsyncArray,
    "slotted": SlottedAsyncArray,
    "reducing": ReducingAsyncArray,
    "init": InitAsyncArray,
    "classattr": ClassAttrAsyncArray,
}


def make_subclass_instance(name: str, base: Any) -> Any:
    cls = SUBCLASSES[name]
    aa = base.async_array
    if cls is TaggedAsyncArray:
        return cls(aa.metadata, aa.store_path, aa.config, tag="t")
    obj = cls(aa.metadata, aa.store_path, aa.config)
    if cls is SlottedAsyncArray:
        object.__setattr__(obj, "note", "noted")
    return obj


def run_subclass(rec: Recorder, name: str, zarr_format: int, tmp: Path) -> None:
    store = new_store("local", tmp, f"sub-{name}-{zarr_format}")
    base = zarr.create_array(
        store, name="a", shape=(4, 3), chunks=(2, 2), dtype="int32", zarr_format=zarr_format,
        attributes={"b": 1, "a": 2},
    )  # fmt: skip
    base[...] = np.arange(12).reshape(4, 3)
    ok, sub = rec.step("construct", lambda: make_subclass_instance(name, base))
    if not ok:
        return
    rec.step("type", lambda: type(sub).__qualname__)
    rec.step("fields", lambda: [f.name for f in dataclasses.fields(sub)])
    rec.step(
        "internal:vars.public",
        lambda: (
            sorted(k for k in vars(sub) if not k.startswith("_"))
            if hasattr(sub, "__dict__")
            else None
        ),
    )
    rec.step("tag", lambda: getattr(sub, "tag", None))
    rec.step("note", lambda: getattr(sub, "note", None))
    rec.step("built", lambda: getattr(sub, "built", None))
    rec.step("kind", lambda: getattr(sub, "kind", None))
    rec.step("_io", lambda: type(sub).__dict__.get("_io"))
    rec.step("describe", lambda: sub.describe() if hasattr(sub, "describe") else None)
    rec.step("eq.copy", lambda: sub == copy.copy(sub))
    rec.step("read", lambda: zarr.Array(sub)[...])
    for how, make in {
        "copy": lambda: copy.copy(sub),
        "deepcopy": lambda: copy.deepcopy(sub),
        "pickle": lambda: pickle.loads(pickle.dumps(sub)),
        "replace": lambda: dataclasses.replace(sub),
        "with_config": lambda: sub.with_config({"write_empty_chunks": True}),
    }.items():
        ok, other = rec.step(f"{how}", make)
        if not ok:
            continue
        rec.step(f"{how}.type", lambda o=other: type(o).__qualname__)
        rec.step(
            f"{how}.state",
            lambda o=other: [getattr(o, k, None) for k in ("tag", "note", "built", "kind")],
        )
        rec.step(f"{how}.metadata", lambda o=other: o.metadata.to_dict())
        rec.step(f"{how}.eq", lambda o=other: o == sub)
        rec.step(f"{how}.read", lambda o=other: zarr.Array(o)[...])
        rec.step(f"{how}.write", lambda o=other: zarr.Array(o).__setitem__((0, 0), 99))
        rec.step(f"{how}.resize", lambda o=other: zarr.Array(o).resize((5, 3)))
        rec.step(f"{how}.attrs", lambda o=other: zarr.Array(o).update_attributes({"c": 3}))
        rec.step(f"{how}.after", lambda o=other: (o.shape, o.metadata.attributes))
    rec.step("sub.after", lambda: (sub.shape, sub.metadata.to_dict()))
    rec.step("store.end", lambda: snapshot(store))


# Pickles carried between versions ------------------------------------------------------------


def _dump(directory: Path, name: str, make: Any) -> None:
    """Pickle what `make` returns now, or a note of why it could not be made or pickled."""
    try:
        data = pickle.dumps(make(), protocol=4)
    except Exception as e:  # noqa: BLE001 (recorded, for the loading run to compare)
        data = pickle.dumps({"unpicklable": f"{type(e).__name__}: {e}"})
    (directory / f"{name}.pkl").write_bytes(data)


def dump_pickles(directory: Path, stores: Path) -> None:
    """Pickle arrays to load in another version, over local stores under `stores`, which
    the loading process sees at the same path."""
    directory.mkdir(parents=True, exist_ok=True)
    for zarr_format in (2, 3):
        store = zarr.storage.LocalStore(stores / f"v{zarr_format}")
        arr = zarr.create_array(
            store, name="a", shape=(6,), chunks=(2,), dtype="float32", zarr_format=zarr_format,
            attributes={"b": 1, "a": 2}, fill_value=1.5,
        )  # fmt: skip
        arr[...] = np.arange(6)
        _dump(directory, f"v{zarr_format}.array", lambda a=arr: a)
        _dump(directory, f"v{zarr_format}.async_array", lambda a=arr: a.async_array)
        _dump(directory, f"v{zarr_format}.metadata", lambda a=arr: a.metadata)
        arr.update_attributes({"c": 3})
        _dump(directory, f"v{zarr_format}.array.after_attrs", lambda a=arr: a)
        for name in SUBCLASSES:
            _dump(
                directory,
                f"v{zarr_format}.sub.{name}",
                lambda n=name, a=arr: make_subclass_instance(n, a),
            )
    store = zarr.storage.LocalStore(stores / "sharded")
    arr = zarr.create_array(store, name="a", shape=(8,), chunks=(2,), shards=(4,), dtype="int16")
    arr[...] = np.arange(8)
    _dump(directory, "v3.sharded", lambda a=arr: a)
    for doc in ("v2-zero-grown-axis", "v3-zero-grown-axis", "v3-true"):
        store = zarr.storage.LocalStore(stores / f"doc-{doc}")
        key = "a/zarr.json" if DOCUMENTS[doc]["zarr_format"] == 3 else "a/.zarray"
        put_document(store, key, DOCUMENTS[doc])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _dump(directory, f"doc.{doc}", lambda s=store: zarr.open_array(s, path="a", mode="r+"))


def load_pickles(rec: Recorder, directory: Path) -> None:
    for path in sorted(directory.glob("*.pkl")):
        rec.scenario = f"pickle:{path.stem}"
        ok, obj = rec.step("load", lambda p=path: pickle.loads(p.read_bytes()))
        if ok and not isinstance(obj, dict):
            _observe_loaded(rec, obj)


def _observe_loaded(rec: Recorder, obj: Any) -> None:
    """Record what a caller sees of an object loaded from a pickle, and use it."""
    rec.step("type", lambda: type(obj).__qualname__)
    if not isinstance(obj, zarr.Array | AsyncArray):
        rec.step("to_dict", lambda: obj.to_dict())
        return
    arr = obj if isinstance(obj, zarr.Array) else zarr.Array(obj)
    rec.step("state", lambda: [getattr(obj, k, None) for k in ("tag", "note", "built", "kind")])
    rec.step("metadata", lambda: arr.metadata.to_dict())
    rec.step("attrs", lambda: arr.attrs.asdict())
    rec.step("config", lambda: arr.async_array.config)
    rec.step("read", lambda: arr[...])
    rec.step("write", lambda: arr.__setitem__((0,) * arr.ndim, 7))
    rec.step("read.after", lambda: arr[...])
    rec.step("resize", lambda: arr.resize(tuple(n + 1 for n in arr.shape)))
    rec.step("attrs.update", lambda: arr.update_attributes({"loaded": True}))
    rec.step("metadata.after", lambda: arr.metadata.to_dict())
    rec.step("repickle", lambda: pickle.loads(pickle.dumps(obj)).metadata.to_dict())


# Entry point -------------------------------------------------------------------------------


def run_all(rec: Recorder, tmp: Path, seeds: range) -> None:
    for name in DOCUMENTS:
        rec.scenario = f"document:{name}"
        _guard(rec, lambda n=name: run_document(rec, n, tmp))
    for name in SUBCLASSES:
        for zarr_format in (2, 3):
            rec.scenario = f"subclass:{name}:v{zarr_format}"
            _guard(rec, lambda n=name, z=zarr_format: run_subclass(rec, n, z, tmp))
    for seed in seeds:
        rec.scenario = f"program:{seed}"
        _guard(rec, lambda s=seed: run_program(rec, s, tmp))
        shutil.rmtree(tmp / f"program-{seed}", ignore_errors=True)


def _guard(rec: Recorder, fn: Any) -> None:
    rec.step("scenario", fn)


__all__ = ["Recorder", "dump_pickles", "load_pickles", "run_all"]
