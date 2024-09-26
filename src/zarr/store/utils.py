from collections.abc import Iterator

from zarr.core.buffer import Buffer


def normalize_interval_index(
    data: Buffer, interval: None | tuple[int | None, int | None]
) -> tuple[int, int]:
    """
    Convert an implicit interval into an explicit start and length
    """
    if interval is None:
        start = 0
        length = len(data)
    else:
        maybe_start, maybe_len = interval
        if maybe_start is None:
            start = 0
        else:
            start = maybe_start

        if maybe_len is None:
            length = len(data) - start
        else:
            length = maybe_len

    return (start, length)


def get_intermediate_nodes(key: str) -> Iterator[str]:
    """
    Given a nested path like foo/bar/baz, return an iterator that yields foo, foo/bar, foo/bar/baz
    """
    nodes = key.split("/")
    return ("/".join(nodes[: i + 1]) for i in range(len(nodes)))
