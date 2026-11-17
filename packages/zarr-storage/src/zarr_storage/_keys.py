"""The key grammar, and the joins between a store's root and its keys.

Every path operation in this package goes through these functions, so the rules live in one place.
"""

from __future__ import annotations

from zarr_storage._errors import InvalidKeyError

__all__ = [
    "check_key",
    "is_within",
    "join",
    "normalize_directory",
    "relativize",
]


def check_key(key: str) -> str:
    """Return `key` unchanged if it follows the key grammar; raise `InvalidKeyError` otherwise.

    A key is not empty, has no leading or trailing `/`, no empty segment, no `.` or `..` segment,
    and no NUL character. Backends may refuse more (a filesystem refuses names it can't represent).
    """
    if not isinstance(key, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise InvalidKeyError(f"A key must be a str; got {type(key).__name__}.")
    if key == "":
        raise InvalidKeyError("A key must not be empty.")
    if "\x00" in key:
        raise InvalidKeyError(f"A key must not contain a NUL character; got {key!r}.")
    if key.startswith("/") or key.endswith("/"):
        raise InvalidKeyError(f"A key must not start or end with '/'; got {key!r}.")
    for segment in key.split("/"):
        if segment in ("", ".", ".."):
            raise InvalidKeyError(
                f"A key must not contain an empty, '.' or '..' segment; got {key!r}."
            )
    return key


def normalize_directory(directory: str) -> str:
    """Return the canonical form of a directory: `""` for the root, otherwise a key.

    A trailing `/` is accepted and removed, so `"a"` and `"a/"` name the same directory.
    """
    if not isinstance(directory, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise InvalidKeyError(f"A directory must be a str; got {type(directory).__name__}.")
    if directory in ("", "/"):
        return ""
    return check_key(directory.removesuffix("/"))


def join(directory: str, key: str) -> str:
    """Join a normalized directory and a key (or normalized directory)."""
    if directory == "":
        return key
    if key == "":
        return directory
    return f"{directory}/{key}"


def relativize(path: str, directory: str) -> str:
    """The part of `path` under the normalized `directory`. `path` must be inside it."""
    if directory == "":
        return path
    return path.removeprefix(f"{directory}/")


def is_within(path: str, directory: str) -> bool:
    """Whether `path` is `directory` itself or lies under it (both normalized)."""
    return directory == "" or path == directory or path.startswith(f"{directory}/")
