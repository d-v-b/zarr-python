"""`JSONWriterConfig` governs how `to_key_value` spells the bytes it writes.

The config changes only the spelling of the JSON text, never the document:
whatever the config, the bytes decode to the same document and
`from_key_value` reads them back to an equal model. The default config
spells the document exactly as `json.dumps` does with its own defaults.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from zarr_metadata.model import (
    JSONWriterConfig,
    ZarrV2ArrayMetadata,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3GroupMetadata,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.typed_json import JSONValue

# Keys out of sorted order, and a non-ASCII value, so key order and ASCII
# escaping are each visible in the written bytes.
ATTRIBUTES: dict[str, JSONValue] = {"unit": "µm", "axis": [1, 2]}

MODELS = [
    ZarrV3ArrayMetadata.create_default(shape=(4,), attributes=ATTRIBUTES),
    ZarrV3GroupMetadata.create_default(attributes=ATTRIBUTES),
    ZarrV2ArrayMetadata.create_default(shape=(4,), attributes=ATTRIBUTES),
    ZarrV2GroupMetadata.create_default(attributes=ATTRIBUTES),
    ZarrV2ConsolidatedMetadata.from_json(
        {
            "zarr_consolidated_format": 1,
            "metadata": {".zgroup": {"zarr_format": 2}, ".zattrs": ATTRIBUTES},
        }
    ),
]
MODEL_IDS = ["v3-array", "v3-group", "v2-array", "v2-group", "v2-consolidated"]


def _decoded(written: Mapping[str, bytes]) -> dict[str, object]:
    return {key: json.loads(raw) for key, raw in written.items()}


@pytest.mark.parametrize(
    "config",
    [
        JSONWriterConfig(),
        JSONWriterConfig(indent=2),
        JSONWriterConfig(indent="\t"),
        JSONWriterConfig(sort_keys=True),
        JSONWriterConfig(ensure_ascii=False),
        JSONWriterConfig(trailing_newline=True),
        JSONWriterConfig(indent=4, sort_keys=True, ensure_ascii=False, trailing_newline=True),
    ],
    ids=repr,
)
@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_config_governs_spelling_and_round_trips(model: object, config: JSONWriterConfig) -> None:
    """`to_key_value(config)` writes each key's bytes as `json.dumps` spells the same document under the same settings. The written store holds the documents a default write holds, and reads back to an equal model."""
    held = cast("Any", model)
    written: Mapping[str, bytes] = held.to_key_value(config)
    for key, raw in written.items():
        text = raw.decode("utf-8")
        if config.trailing_newline:
            assert text.endswith("\n"), key
            text = text.removesuffix("\n")
        expected = json.dumps(
            json.loads(text),
            indent=config.indent,
            sort_keys=config.sort_keys,
            ensure_ascii=config.ensure_ascii,
        )
        assert text == expected, key
    assert _decoded(written) == _decoded(held.to_key_value())
    assert type(held).from_key_value(written) == model


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_default_config_writes_what_json_dumps_writes(model: object) -> None:
    """A bare `to_key_value()` writes each document with the defaults of Python's `json` module, byte for byte. `ensure_ascii` escapes the non-ASCII attribute."""
    written: Mapping[str, bytes] = cast("Any", model).to_key_value()
    for key, raw in written.items():
        assert raw == json.dumps(json.loads(raw)).encode("utf-8"), key
    assert any(b"\\u00b5m" in raw for raw in written.values())


def test_ensure_ascii_off_escapes_only_a_lone_surrogate() -> None:
    """With `ensure_ascii` off, a lone surrogate the reader accepted from an escape is written back as that escape, since UTF-8 cannot encode it, while every other non-ASCII character is written as UTF-8. The store reads back to an equal model."""
    stored = {
        "zarr.json": json.dumps(
            {"zarr_format": 3, "node_type": "group", "attributes": {"x": "\ud800", "y": "µ"}}
        ).encode("utf-8")
    }
    model = ZarrV3GroupMetadata.from_key_value(stored)
    written = model.to_key_value(JSONWriterConfig(ensure_ascii=False))["zarr.json"]
    assert b'"\\ud800"' in written
    assert "µ".encode() in written
    assert ZarrV3GroupMetadata.from_key_value({"zarr.json": written}) == model


def test_config_refuses_a_bool_indent() -> None:
    """A `bool` is not an indent, though `json.dumps` would read `True` as an indent of 1."""
    with pytest.raises(TypeError, match="indent must be an int, a str or None, got bool"):
        JSONWriterConfig(indent=cast("Any", True))


def test_config_refuses_an_indent_of_another_type() -> None:
    """An indent that is not an `int`, a `str` or `None` is refused."""
    with pytest.raises(TypeError, match="indent must be an int, a str or None, got float"):
        JSONWriterConfig(indent=cast("Any", 2.0))


def test_config_refuses_a_negative_indent() -> None:
    """A negative indent is refused, though `json.dumps` would read it as 0."""
    with pytest.raises(ValueError, match="indent must not be negative, got -3"):
        JSONWriterConfig(indent=-3)


def test_config_refuses_an_indent_that_is_not_json_whitespace() -> None:
    """A `str` indent must hold only JSON whitespace: `json.dumps` writes it verbatim, so any other character writes bytes no JSON reader parses."""
    with pytest.raises(ValueError, match="indent must hold only JSON whitespace, got 'x'"):
        JSONWriterConfig(indent="x")


@pytest.mark.parametrize("name", ["sort_keys", "ensure_ascii", "trailing_newline"])
def test_config_refuses_a_flag_that_is_not_a_bool(name: str) -> None:
    """Each flag takes only a `bool`: a truthy value such as `1` is refused."""
    with pytest.raises(TypeError, match=f"{name} must be a bool, got int"):
        JSONWriterConfig(**{name: cast("Any", 1)})
