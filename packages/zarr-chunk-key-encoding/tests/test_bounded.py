"""Tests for `BoundedChunkKeyEncoding`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import pytest

from zarr_chunk_key_encoding import (
    BoundedChunkKeyEncoding,
    ChunkCoordsOutOfBoundsError,
    ChunkKeyConfigurationError,
    ChunkKeyDecodeError,
    ChunkKeyEncoding,
    ChunkKeyOutOfBoundsError,
    DefaultChunkKeyEncoding,
    InvalidChunkCoordsError,
    V2ChunkKeyEncoding,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from zarr_metadata import JSONValue

    from zarr_chunk_key_encoding import ChunkKeyEncodingJSON

ENCODINGS = [
    DefaultChunkKeyEncoding(),
    DefaultChunkKeyEncoding(separator="."),
    V2ChunkKeyEncoding(),
    V2ChunkKeyEncoding(separator="/"),
]


@pytest.mark.parametrize("encoding", ENCODINGS, ids=repr)
@pytest.mark.parametrize(
    ("grid_shape", "chunk_coords"),
    [
        ((), ()),
        ((1,), (0,)),
        ((5,), (4,)),
        ((2, 3), (0, 0)),
        ((2, 3), (1, 2)),
        ((4, 1, 7), (3, 0, 6)),
    ],
)
def test_encode_decode(
    encoding: ChunkKeyEncoding, grid_shape: tuple[int, ...], chunk_coords: tuple[int, ...]
) -> None:
    """For valid indices of assorted grids, bounded `encode` agrees with the
    unbounded encoding, `decode` inverts it exactly (including the `v2`
    rank-zero case), and the key is a member of the bounded key set."""
    bounded = encoding.bind(grid_shape)
    key = bounded.encode(chunk_coords)
    assert key == encoding.encode(chunk_coords)
    assert bounded.decode(key) == chunk_coords
    assert key in bounded


def test_collection_semantics() -> None:
    """The bounded encoding is a finite collection of exactly the grid's keys,
    iterated in row-major order, including the degenerate grids."""
    bounded = DefaultChunkKeyEncoding().bind((2, 3))
    assert len(bounded) == 6
    assert list(bounded) == ["c/0/0", "c/0/1", "c/0/2", "c/1/0", "c/1/1", "c/1/2"]
    assert all(key in bounded for key in bounded)
    # A zero-dimensional grid holds a single chunk.
    assert list(V2ChunkKeyEncoding().bind(())) == ["0"]
    assert len(V2ChunkKeyEncoding().bind(())) == 1
    # A grid with a zero-size dimension holds no chunks at all.
    empty = DefaultChunkKeyEncoding().bind((0, 5))
    assert len(empty) == 0
    assert list(empty) == []
    assert "c/0/0" not in empty


def test_bind_equivalent_to_constructor() -> None:
    """`bind` is a convenience for direct construction."""
    encoding = DefaultChunkKeyEncoding()
    assert encoding.bind((2, 3)) == BoundedChunkKeyEncoding(encoding=encoding, grid_shape=(2, 3))


def test_encode_invalid_coords() -> None:
    """Coordinates that are invalid in isolation raise `InvalidChunkCoordsError`."""
    bounded = DefaultChunkKeyEncoding().bind((2, 3))
    with pytest.raises(InvalidChunkCoordsError):
        bounded.encode((-1, 0))


def test_encode_wrong_rank() -> None:
    """Valid coordinates of the wrong rank raise `ChunkCoordsOutOfBoundsError`."""
    bounded = DefaultChunkKeyEncoding().bind((2, 3))
    with pytest.raises(ChunkCoordsOutOfBoundsError):
        bounded.encode((1,))


def test_encode_out_of_bounds() -> None:
    """Coordinates at or beyond the grid extent raise `ChunkCoordsOutOfBoundsError`."""
    bounded = DefaultChunkKeyEncoding().bind((2, 3))
    with pytest.raises(ChunkCoordsOutOfBoundsError):
        bounded.encode((2, 0))


def test_decode_malformed_key() -> None:
    """A key the underlying encoding cannot decode raises plain
    `ChunkKeyDecodeError`, not the out-of-bounds subclass."""
    bounded = DefaultChunkKeyEncoding().bind((2, 3))
    for key in ("x/0/0", "c/01/0", "c/-1/0", ""):
        with pytest.raises(ChunkKeyDecodeError) as excinfo:
            bounded.decode(key)
        assert not isinstance(excinfo.value, ChunkKeyOutOfBoundsError)
        assert key not in bounded


def test_decode_wrong_rank() -> None:
    """A well-formed key of the wrong rank raises `ChunkKeyOutOfBoundsError`."""
    bounded = DefaultChunkKeyEncoding().bind((2, 3))
    with pytest.raises(ChunkKeyOutOfBoundsError):
        bounded.decode("c/1")
    assert "c/1" not in bounded


def test_decode_out_of_bounds() -> None:
    """A well-formed key beyond the grid extent raises `ChunkKeyOutOfBoundsError`."""
    bounded = DefaultChunkKeyEncoding().bind((2, 3))
    with pytest.raises(ChunkKeyOutOfBoundsError):
        bounded.decode("c/2/0")
    assert "c/2/0" not in bounded


def test_decode_rank_zero_out_of_domain() -> None:
    """On a zero-dimensional grid, any key other than `encode(())` is out of
    domain, even where the unbounded decode would accept it."""
    bounded = V2ChunkKeyEncoding().bind(())
    with pytest.raises(ChunkKeyOutOfBoundsError):
        bounded.decode("5")
    assert "5" not in bounded


def test_invalid_grid_shape() -> None:
    """Grid shape entries must be non-negative integers."""
    for grid_shape in ((-1,), (1.5,), ("a",)):
        with pytest.raises(ChunkKeyConfigurationError):
            DefaultChunkKeyEncoding().bind(grid_shape)  # type: ignore[arg-type]


def test_contains_non_string() -> None:
    """Non-string objects are never members."""
    bounded = DefaultChunkKeyEncoding().bind((2, 3))
    assert (0, 0) not in bounded


class _NoDecode(ChunkKeyEncoding):
    """A minimal encoding without `decode`, for testing propagation."""

    name = "no-decode"

    @classmethod
    def from_json(cls, data: ChunkKeyEncodingJSON) -> Self:
        """Construct without inspecting the metadata."""
        return cls()

    def to_json(self) -> Mapping[str, JSONValue]:
        """Return the name-only object form."""
        return {"name": self.name}

    def encode(self, chunk_coords: Sequence[int]) -> str:
        """Join coordinates with `/`, using `z` for the rank-zero key."""
        return "/".join(str(c) for c in chunk_coords) or "z"


def test_contains_without_decode() -> None:
    """Membership testing needs `decode`; `NotImplementedError` propagates
    rather than being silently reported as a non-member. The exception is a
    zero-dimensional grid, where the single valid key is compared directly."""
    with pytest.raises(NotImplementedError):
        "0/0" in _NoDecode().bind((2, 3))  # noqa: B015
    assert "z" in _NoDecode().bind(())
