import re
from typing import get_args

import pytest

from zarr.core.chunk_key_encodings import (
    DefaultChunkKeyEncoding,
    DefaultChunkKeyEncodingMetadata,
    SeparatorLiteral,
    V2ChunkKeyEncoding,
    V2ChunkKeyEncodingParams,
    parse_chunk_key_encoding,
    parse_separator,
)


class TestParseSeparator:
    """
    Class for testing the ``parse_separator`` function
    """

    @staticmethod
    @pytest.mark.parametrize("value", list(get_args(SeparatorLiteral)))
    def test_valid(value: SeparatorLiteral) -> None:
        """
        Test that ``parse_separator`` simply returns valid inputs
        """
        assert parse_separator(value) == value

    @staticmethod
    @pytest.mark.parametrize("value", ["x"])
    def test_parse_separator_invalid(value: str) -> None:
        """
        Test that ``parse_separator`` raises ValueError for invalid inputs
        """
        msg = f"Expected an '.' or '/' separator. Got {value} instead."
        with pytest.raises(ValueError, match=re.escape(msg)):
            parse_separator(value)


class TestParseChunkKeyEncoding:
    """
    Class for testing the ``parse_chunk_key_encoding`` function
    """

    @staticmethod
    @pytest.mark.parametrize(
        "value", [V2ChunkKeyEncoding(separator="/"), DefaultChunkKeyEncoding(separator=".")]
    )
    def test_valid_instance(value: V2ChunkKeyEncoding | DefaultChunkKeyEncoding) -> None:
        """
        Test that ``parse_chunk_key_encoding`` simply returns instances
        of ``DefaultChunkKeyEncoding and ``V2ChunkKeyEncoding``
        """
        assert parse_chunk_key_encoding(value) == value

    @staticmethod
    @pytest.mark.parametrize(
        "value",
        [
            {"name": "default", "configuration": {"separator": "/"}},
            {"name": "v2", "configuration": {"separator": "."}},
            {"name": "v2"},
            {"name": "default"},
        ],
    )
    def test_valid_dict(value: V2ChunkKeyEncodingParams | DefaultChunkKeyEncodingMetadata) -> None:
        """
        Test that ``parse_chunk_key_encoding`` simply returns instances
        of ``DefaultChunkKeyEncoding and ``V2ChunkKeyEncoding``
        """
        name = value["name"]
        cls: type[V2ChunkKeyEncoding | DefaultChunkKeyEncoding]
        match name:
            case "v2":
                cls = V2ChunkKeyEncoding
            case "default":
                cls = DefaultChunkKeyEncoding
            case _:
                raise ValueError(f"Invalid name: {name}")
        expected = cls.from_dict(value)  # type: ignore[arg-type]
        observed = parse_chunk_key_encoding(value)
        assert observed == expected

    @staticmethod
    @pytest.mark.parametrize(
        "value", [{"name": "invalid", "separator": "invalid"}, {"name": "invalid"}]
    )
    def test_invalid_dict(value: dict[str, object]) -> None:
        """
        Test that ``parse_chunk_key_encoding`` raises ValueError for invalid inputs
        """
        name = value["name"]
        msg = f"Unknown chunk key encoding. Got {name}, expected one of ('v2', 'default')."
        with pytest.raises(ValueError, match=re.escape(msg)):
            parse_chunk_key_encoding(value)  # type: ignore[arg-type]


class TestDefaultChunkKeyEncoding:
    _cls = DefaultChunkKeyEncoding

    @pytest.mark.parametrize("separator", [".", "/"])
    @pytest.mark.parametrize("chunk_key_parts", [(), (0,), (0, 1, 2)])
    def test_encode_decode(
        self, chunk_key_parts: tuple[int, ...], separator: SeparatorLiteral
    ) -> None:
        encoding = self._cls(separator=separator)
        encoded = encoding.encode_chunk_key(chunk_key_parts)
        assert encoded == separator.join(map(str, ("c",) + chunk_key_parts))

        decoded = encoding.decode_chunk_key(encoded)

        if encoded == "c":
            assert decoded == ()
        else:
            assert decoded == chunk_key_parts

    @pytest.mark.parametrize("separator", list(get_args(SeparatorLiteral)))
    def test_dict_serialization(self, separator: SeparatorLiteral) -> None:
        encoding = self._cls(separator=separator)
        dict_data = {"name": "default", "configuration": {"separator": separator}}
        assert encoding.to_dict() == dict_data
        assert self._cls.from_dict(dict_data) == encoding  # type: ignore[arg-type]


class TestV2ChunkKeyEncoding:
    _cls = V2ChunkKeyEncoding

    @pytest.mark.parametrize("separator", [".", "/"])
    @pytest.mark.parametrize("chunk_key_parts", [(), (0,), (0, 1, 2)])
    def test_encode_decode(
        self, chunk_key_parts: tuple[int, ...], separator: SeparatorLiteral
    ) -> None:
        encoding = self._cls(separator=separator)
        encoded = encoding.encode_chunk_key(chunk_key_parts)

        if chunk_key_parts == ():
            assert encoded == "0"
        else:
            assert encoded == separator.join(map(str, chunk_key_parts))
        decoded = encoding.decode_chunk_key(encoded)

        if chunk_key_parts == ():
            assert decoded == (0,)
        else:
            assert decoded == chunk_key_parts

    @pytest.mark.parametrize("separator", list(get_args(SeparatorLiteral)))
    def test_dict_serialization(self, separator: SeparatorLiteral) -> None:
        encoding = self._cls(separator=separator)
        dict_data = {"name": "v2", "configuration": {"separator": separator}}
        assert encoding.to_dict() == dict_data
        assert self._cls.from_dict(dict_data) == encoding  # type: ignore[arg-type]
