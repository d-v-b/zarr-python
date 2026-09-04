"""Tests for `BoundedChunkKeyEncoding`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import pytest

from zarr_chunk_key_encoding import (
    BoundedChunkKeyEncoding,
    ChunkCoordsOutOfBoundsError,
    ChunkKey,
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
    bounded = encoding.to_bounded(grid_shape)
    key = bounded.encode(chunk_coords)
    assert key == encoding.encode(chunk_coords)
    assert bounded.decode(key) == chunk_coords
    assert key in bounded


def test_collection_semantics() -> None:
    """The bounded encoding is a finite collection of exactly the grid's keys,
    iterated in row-major order, including the degenerate grids."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    assert len(bounded) == 6
    assert list(bounded) == ["c/0/0", "c/0/1", "c/0/2", "c/1/0", "c/1/1", "c/1/2"]
    assert all(key in bounded for key in bounded)
    # A zero-dimensional grid holds a single chunk.
    assert list(V2ChunkKeyEncoding().to_bounded(())) == ["0"]
    assert len(V2ChunkKeyEncoding().to_bounded(())) == 1
    # A grid with a zero-size dimension holds no chunks at all.
    empty = DefaultChunkKeyEncoding().to_bounded((0, 5))
    assert len(empty) == 0
    assert list(empty) == []
    assert "c/0/0" not in empty


@pytest.mark.parametrize("encoding", ENCODINGS, ids=repr)
@pytest.mark.parametrize("grid_shape", [(), (5,), (2, 3), (0, 4)])
def test_json_round_trip(encoding: ChunkKeyEncoding, grid_shape: tuple[int, ...]) -> None:
    """`to_json` produces the documented shape, with the grid shape as a list
    and the encoding's own metadata nested, and `from_json` inverts it exactly
    — including for the degenerate grids."""
    bounded = encoding.to_bounded(grid_shape)
    data = bounded.to_json()
    assert set(data) == {"grid_shape", "chunk_key_encoding"}
    assert data["grid_shape"] == list(grid_shape)
    assert data["chunk_key_encoding"] == encoding.to_json()
    assert BoundedChunkKeyEncoding.from_json(data) == bounded


def test_from_json_accepts_tuple_and_short_name() -> None:
    """Hand-built input may use a tuple grid shape and the short-hand
    encoding name; both normalize to the canonical form."""
    bounded = BoundedChunkKeyEncoding.from_json(
        {"grid_shape": (2, 3), "chunk_key_encoding": "default"}
    )
    assert bounded == DefaultChunkKeyEncoding().to_bounded((2, 3))
    assert bounded.to_json()["grid_shape"] == [2, 3]


def test_from_json_not_a_mapping() -> None:
    """Non-object input is rejected with a package error."""
    with pytest.raises(ChunkKeyConfigurationError, match="expected a JSON object"):
        BoundedChunkKeyEncoding.from_json(["grid_shape", "chunk_key_encoding"])


@pytest.mark.parametrize(
    "data",
    [
        {"grid_shape": [2, 3]},
        {"chunk_key_encoding": "default"},
        {"grid_shape": [2, 3], "chunk_key_encoding": "default", "extra": 1},
    ],
    ids=["missing-encoding", "missing-shape", "extra-key"],
)
def test_from_json_wrong_keys(data: object) -> None:
    """The envelope must carry exactly the two documented keys."""
    with pytest.raises(ChunkKeyConfigurationError, match="expected exactly the keys"):
        BoundedChunkKeyEncoding.from_json(data)


@pytest.mark.parametrize("grid_shape", ["23", 5, None, {"a": 1}])
def test_from_json_grid_shape_not_a_sequence(grid_shape: object) -> None:
    """A grid shape that is not a list is rejected before it can be iterated
    — including a string, which is a sequence but not of integers."""
    with pytest.raises(ChunkKeyConfigurationError, match="'grid_shape' must be a list"):
        BoundedChunkKeyEncoding.from_json(
            {"grid_shape": grid_shape, "chunk_key_encoding": "default"}
        )


def test_from_json_invalid_grid_entries() -> None:
    """Grid shape entries are validated the same way as direct construction."""
    with pytest.raises(ChunkKeyConfigurationError, match="Invalid chunk grid shape"):
        BoundedChunkKeyEncoding.from_json({"grid_shape": [2, -1], "chunk_key_encoding": "default"})


def test_from_json_invalid_nested_encoding() -> None:
    """Errors from the nested encoding metadata propagate unchanged."""
    from zarr_chunk_key_encoding import UnknownChunkKeyEncodingError

    with pytest.raises(UnknownChunkKeyEncodingError, match="not_an_encoding"):
        BoundedChunkKeyEncoding.from_json(
            {"grid_shape": [2, 3], "chunk_key_encoding": "not_an_encoding"}
        )


def test_construction_paths_agree() -> None:
    """`to_bounded`, `from_unbounded`, and direct construction all produce the
    same object; the first two are the two directions of one conversion, and
    a list grid shape normalizes to the tuple the others carry."""
    encoding = DefaultChunkKeyEncoding()
    direct = BoundedChunkKeyEncoding(encoding=encoding, grid_shape=(2, 3))
    assert encoding.to_bounded((2, 3)) == direct
    assert BoundedChunkKeyEncoding.from_unbounded(encoding, (2, 3)) == direct
    assert BoundedChunkKeyEncoding.from_unbounded(encoding, [2, 3]) == direct
    assert BoundedChunkKeyEncoding.from_unbounded(encoding, [2, 3]).grid_shape == (2, 3)


def test_from_unbounded_invalid_grid_shape() -> None:
    """Grid shape validation applies on this path too."""
    with pytest.raises(ChunkKeyConfigurationError):
        BoundedChunkKeyEncoding.from_unbounded(DefaultChunkKeyEncoding(), (2, -1))


def test_encode_invalid_coords() -> None:
    """Coordinates that are invalid in isolation raise `InvalidChunkCoordsError`."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    with pytest.raises(InvalidChunkCoordsError):
        bounded.encode((-1, 0))


def test_encode_wrong_rank() -> None:
    """Valid coordinates of the wrong rank raise `ChunkCoordsOutOfBoundsError`."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    with pytest.raises(ChunkCoordsOutOfBoundsError):
        bounded.encode((1,))


def test_encode_out_of_bounds() -> None:
    """Coordinates at or beyond the grid extent raise `ChunkCoordsOutOfBoundsError`."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    with pytest.raises(ChunkCoordsOutOfBoundsError):
        bounded.encode((2, 0))


def test_decode_malformed_key() -> None:
    """A key the underlying encoding cannot decode raises plain
    `ChunkKeyDecodeError`, not the out-of-bounds subclass."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    for key in ("x/0/0", "c/01/0", "c/-1/0", ""):
        with pytest.raises(ChunkKeyDecodeError) as excinfo:
            bounded.decode(key)
        assert not isinstance(excinfo.value, ChunkKeyOutOfBoundsError)
        assert key not in bounded


def test_oversized_integer_key_is_not_a_member() -> None:
    """Membership treats integer components above Python's conversion limit as malformed."""
    bounded = DefaultChunkKeyEncoding().to_bounded((1,))
    assert "c/" + "9" * 5_000 not in bounded


def test_decode_wrong_rank() -> None:
    """A well-formed key of the wrong rank raises `ChunkKeyOutOfBoundsError`."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    with pytest.raises(ChunkKeyOutOfBoundsError):
        bounded.decode("c/1")
    assert "c/1" not in bounded


def test_decode_out_of_bounds() -> None:
    """A well-formed key beyond the grid extent raises `ChunkKeyOutOfBoundsError`."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    with pytest.raises(ChunkKeyOutOfBoundsError):
        bounded.decode("c/2/0")
    assert "c/2/0" not in bounded


def test_decode_rank_zero_out_of_domain() -> None:
    """On a zero-dimensional grid, any key other than `encode(())` is out of
    domain, even where the unbounded decode would accept it."""
    bounded = V2ChunkKeyEncoding().to_bounded(())
    with pytest.raises(ChunkKeyOutOfBoundsError):
        bounded.decode("5")
    assert "5" not in bounded


def test_invalid_grid_shape() -> None:
    """Grid shape entries must be non-negative integers."""
    for grid_shape in ((-1,), (1.5,), ("a",)):
        with pytest.raises(ChunkKeyConfigurationError):
            DefaultChunkKeyEncoding().to_bounded(grid_shape)  # type: ignore[arg-type]


def test_contains_non_string() -> None:
    """Non-string objects are never members."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
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

    def encode(self, chunk_coords: Sequence[int]) -> ChunkKey:
        """Join coordinates with `/`, using `z` for the rank-zero key."""
        return ChunkKey("/".join(str(c) for c in chunk_coords) or "z")


def test_contains_without_decode() -> None:
    """Membership testing needs `decode`; `NotImplementedError` propagates
    rather than being silently reported as a non-member. The exception is a
    zero-dimensional grid, where the single valid key is compared directly."""
    with pytest.raises(NotImplementedError):
        "0/0" in _NoDecode().to_bounded((2, 3))  # noqa: B015
    assert "z" in _NoDecode().to_bounded(())
