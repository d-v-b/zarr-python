"""Tests for `zarr_metadata.msgspec`, the optional msgspec integration module.

Unlike pydantic, msgspec consults its extension hooks only for types it does
not support natively — and it supports dataclasses natively, so the module's
field types are runtime markers rather than the core model classes (see the
module docstring). Instances are still the CORE model classes (no parallel
hierarchy), so decoded values interoperate freely with non-msgspec code.
"""

import msgspec
import msgspec.json
import pytest

import zarr_metadata.msgspec as zmm
from tests.model._cases import (
    V2_ARRAY_DOC,
    V2_CONSOLIDATED_DOC,
    V2_GROUP_DOC,
    V3_ARRAY_DOC,
    V3_CONSOLIDATED_DOC,
    V3_GROUP_DOC,
)
from zarr_metadata.model import (
    ZarrV2ArrayMetadata,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3GroupMetadata,
    ZarrV3NamedConfig,
)

FIELD_CASES = [
    pytest.param(zmm.ZarrV3ArrayMetadata, ZarrV3ArrayMetadata, V3_ARRAY_DOC, id="array-v3"),
    pytest.param(zmm.ZarrV2ArrayMetadata, ZarrV2ArrayMetadata, V2_ARRAY_DOC, id="array-v2"),
    pytest.param(zmm.ZarrV3GroupMetadata, ZarrV3GroupMetadata, V3_GROUP_DOC, id="group-v3"),
    pytest.param(zmm.ZarrV2GroupMetadata, ZarrV2GroupMetadata, V2_GROUP_DOC, id="group-v2"),
    pytest.param(
        zmm.ZarrV3ConsolidatedMetadata,
        ZarrV3ConsolidatedMetadata,
        V3_CONSOLIDATED_DOC,
        id="consolidated-v3",
    ),
    pytest.param(
        zmm.ZarrV2ConsolidatedMetadata,
        ZarrV2ConsolidatedMetadata,
        V2_CONSOLIDATED_DOC,
        id="consolidated-v2",
    ),
    pytest.param(zmm.ZarrV3MetadataField, ZarrV3NamedConfig, {"name": "bytes"}, id="field-v3"),
]


@pytest.mark.parametrize(("field_type", "model_cls", "doc"), FIELD_CASES)
def test_field_type_decodes_and_passes_through(
    field_type: object, model_cls: type, doc: dict[str, object]
) -> None:
    """Each field type parses its raw document into the CORE model class via
    from_json, passes existing instances through unchanged, and works for
    `msgspec.convert` and `msgspec.json.decode` alike."""
    model = msgspec.convert(doc, field_type, dec_hook=zmm.dec_hook)
    assert type(model) is model_cls
    assert msgspec.convert(model, field_type, dec_hook=zmm.dec_hook) is model
    decoded = msgspec.json.decode(msgspec.json.encode(doc), type=field_type, dec_hook=zmm.dec_hook)
    assert decoded == model


def test_registry_is_consistent() -> None:
    """One decode table drives everything: it covers exactly the exported
    field types, each parser is its core class's from_json, and each core
    class is registered as a virtual subclass of its marker (so instances
    dec_hook returns satisfy msgspec's check on hook results)."""
    field_types = {getattr(zmm, name) for name in zmm.__all__ if name[0] == "Z"}
    assert field_types == set(zmm._DECODERS)
    for marker, (core_cls, parse) in zmm._DECODERS.items():
        assert parse == core_cls.from_json
        assert issubclass(core_cls, marker)


def test_struct_field_round_trip() -> None:
    """A Struct field decodes a raw document through from_json (the library's
    normalization applies), and re-encoding the canonical document emitted by
    to_json revalidates to an equal manifest — serialization routes through
    to_json explicitly because msgspec's encoders cannot delegate dataclasses
    (see the module docstring)."""

    class ArrayManifest(msgspec.Struct):
        path: str
        metadata: zmm.ZarrV3ArrayMetadata

    data = msgspec.json.encode({"path": "a/b", "metadata": V3_ARRAY_DOC})
    manifest = msgspec.json.decode(data, type=ArrayManifest, dec_hook=zmm.dec_hook)
    assert isinstance(manifest.metadata, ZarrV3ArrayMetadata)
    assert manifest.metadata.shape == (4,)

    out = msgspec.json.encode({"path": manifest.path, "metadata": manifest.metadata.to_json()})
    assert msgspec.json.decode(out, type=ArrayManifest, dec_hook=zmm.dec_hook) == manifest


def test_wrapped_hook_composes() -> None:
    """make_dec_hook keeps an application's own custom-type decoding working
    alongside the model field types — including custom types whose annotation
    objects are unhashable — and with no wrapped hook it is dec_hook."""

    class Fraction:
        def __init__(self, value: float) -> None:
            self.value = value

    def consumer_hook(type: type, obj: object) -> object:
        if type is Fraction:
            return Fraction(obj)
        raise NotImplementedError(type)

    class Manifest(msgspec.Struct):
        scale: Fraction
        metadata: zmm.ZarrV3GroupMetadata

    hook = zmm.make_dec_hook(consumer_hook)
    manifest = msgspec.convert({"scale": 0.5, "metadata": V3_GROUP_DOC}, Manifest, dec_hook=hook)
    assert isinstance(manifest.scale, Fraction)
    assert manifest.scale.value == 0.5
    assert type(manifest.metadata) is ZarrV3GroupMetadata

    # An unhashable annotation object is an ordinary miss, delegated onward.
    class Unhashable:
        __hash__ = None  # type: ignore[assignment]

    unhashable = Unhashable()
    seen: list[object] = []

    def recording_hook(type: type, obj: object) -> object:
        seen.append(type)
        return obj

    assert zmm.make_dec_hook(recording_hook)(unhashable, 1) == 1
    assert seen == [unhashable]

    assert zmm.make_dec_hook() is zmm.dec_hook


def test_decoding_is_stateless_and_normalizing() -> None:
    """Decoding the same document twice yields equal but independent model
    instances, and from_json's normalization converts JSON arrays to tuples,
    so mutating a list in the source document cannot reach a decoded model.
    (from_json copies mappings shallowly; only to_json guarantees a document
    that shares no mutable state.)"""
    doc = {"zarr_format": 3, "node_type": "group", "attributes": {"a": [1]}}
    first = msgspec.convert(doc, zmm.ZarrV3GroupMetadata, dec_hook=zmm.dec_hook)
    second = msgspec.convert(doc, zmm.ZarrV3GroupMetadata, dec_hook=zmm.dec_hook)
    assert first == second
    assert first is not second
    doc["attributes"]["a"].append(2)
    assert first.attributes == {"a": (1,)}


def test_invalid_document_surfaces_problems_in_validation_error() -> None:
    """A defective document fails as msgspec.ValidationError carrying the
    loc-annotated problem messages from MetadataValidationError, plus
    msgspec's own path annotation for the failing field."""

    class ArrayManifest(msgspec.Struct):
        metadata: zmm.ZarrV3ArrayMetadata

    doc = dict(V3_ARRAY_DOC)
    del doc["chunk_key_encoding"]
    with pytest.raises(
        msgspec.ValidationError, match=r"chunk_key_encoding: missing required key.*\$\.metadata"
    ):
        msgspec.convert({"metadata": doc}, ArrayManifest, dec_hook=zmm.dec_hook)


def test_dec_hook_rejects_uncovered_types() -> None:
    """A type the module does not cover raises NotImplementedError, msgspec's
    convention for a still-unsupported custom type — including annotation
    objects that are not hashable."""
    with pytest.raises(NotImplementedError, match="int"):
        zmm.dec_hook(int, 1)
    with pytest.raises(NotImplementedError, match="not supported"):
        zmm.dec_hook([], 1)


def test_field_type_markers_cannot_be_instantiated() -> None:
    """The exported names are for annotations only: instantiating a runtime
    marker raises instead of minting a hollow object that would pass
    msgspec's result check while having none of the core class's fields."""
    with pytest.raises(TypeError, match="field-type marker"):
        zmm.ZarrV3ArrayMetadata()


def test_core_class_annotations_bypass_dec_hook() -> None:
    """Why the field types are markers: msgspec supports dataclasses natively
    and never consults dec_hook for them, so annotating with a core model
    class would engage msgspec's field-by-field coercion instead of from_json
    (today it fails to resolve the models' TYPE_CHECKING-only annotations).
    If this test starts failing because decoding succeeded or the hook was
    called, msgspec's dataclass handling changed — revisit the markers."""
    calls: list[type] = []

    def hook(type: type, obj: object) -> object:
        calls.append(type)
        raise NotImplementedError(type)

    with pytest.raises((NameError, TypeError)):
        msgspec.convert(V3_ARRAY_DOC, ZarrV3ArrayMetadata, dec_hook=hook)
    assert calls == []


def test_native_encode_cannot_emit_canonical_documents() -> None:
    """Why serialization must route through to_json explicitly: msgspec's
    encoders never consult enc_hook for dataclasses, so a model instance
    either fails to encode (the UNSET sentinel is unencodable) or encodes as
    a raw field dump that is not the canonical document. If either half
    starts failing, msgspec grew an encode override point — revisit the
    module's serialization guidance."""
    with pytest.raises(TypeError, match="unsupported"):
        msgspec.json.encode(ZarrV3ArrayMetadata.from_json(V3_ARRAY_DOC))

    config = ZarrV3NamedConfig.from_json({"name": "bytes"})
    assert msgspec.json.decode(msgspec.json.encode(config)) != config.to_json()


def test_core_package_does_not_import_msgspec() -> None:
    """Importing zarr_metadata (in a fresh interpreter) must not import
    msgspec: the integration is opt-in via zarr_metadata.msgspec."""
    import subprocess
    import sys

    code = "import sys, zarr_metadata; assert 'msgspec' not in sys.modules, 'leaked'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_module_requires_msgspec() -> None:
    """Importing the integration module without msgspec fails fast at import
    (its exports are inert without msgspec), mirroring zarr_metadata.pydantic."""
    import subprocess
    import sys

    code = "import sys; sys.modules['msgspec'] = None; import zarr_metadata.msgspec"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert "msgspec" in proc.stderr
