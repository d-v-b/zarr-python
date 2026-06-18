from typing import get_args

import zarr_metadata as zm


def test_consumed_pairs_are_public() -> None:
    for name in (
        "Endian",
        "ENDIAN",
        "BloscShuffle",
        "BLOSC_SHUFFLE",
        "BloscCName",
        "BLOSC_CNAME",
        "IndexLocation",
        "INDEX_LOCATION",
        "DateTimeUnit",
        "DATETIME_UNIT",
    ):
        assert name in zm.__all__, f"{name} missing from public API"
        assert hasattr(zm, name)


def test_constant_matches_literal() -> None:
    assert set(zm.ENDIAN) == set(get_args(zm.Endian))
    assert set(zm.BLOSC_SHUFFLE) == set(get_args(zm.BloscShuffle))
    assert set(zm.BLOSC_CNAME) == set(get_args(zm.BloscCName))
    assert set(zm.INDEX_LOCATION) == set(get_args(zm.IndexLocation))
    assert set(zm.DATETIME_UNIT) == set(get_args(zm.DateTimeUnit))
