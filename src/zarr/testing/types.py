from collections.abc import Mapping, Sequence
from typing import (
    Literal,
    TypedDict,
    TypeVar,
    _LiteralGenericAlias,
    _TypedDictMeta,
    get_args,
    get_origin,
    get_type_hints,
)

JSON2 = str | int | float | Mapping[str, object] | Sequence["JSON2"] | None


def check_keys(
    data: Mapping[str, object], expected_keys: Sequence[str], *, allow_extra: bool = False
) -> None:
    expected_set = set(expected_keys)
    if (observed_set := set(data.keys())) != expected_set:
        msg = f"Object {data} has invalid keys. "
        missing = expected_set - observed_set
        extra = observed_set - expected_set
        if len(missing) > 0:
            msg += f"These keys were missing: {missing}. "
        if len(extra) > 0 and not allow_extra:
            msg += f"These extra keys were found: {extra}"
        raise ValueError(msg)
    return


T = TypeVar("T")
T_Tuple = TypeVar("T_Tuple", bound=tuple[object, ...])
T_List = TypeVar("T_List", bound=list[object])


def check_str(data: object) -> str:
    if isinstance(data, str):
        return data
    raise TypeError(f"Expected a string, got {data} with type {type(data)}")


def check_int(data: object) -> int:
    if isinstance(data, int):
        return data
    raise TypeError(f"Expected an int, got {data} with type {type(data)}")


def check_none(data: object) -> None:
    if data is None:
        return data
    raise TypeError(f"Expected None, got {data} with type {type(data)}")


def check_bool(data: object) -> bool:
    if isinstance(data, bool):
        return data
    raise TypeError(f"Expected a bool, got {data} with type {type(data)}")


def check_mapping(data: object) -> Mapping[str, object]:
    if isinstance(data, Mapping) and all(isinstance(k, str) for k in data):
        return data
    raise TypeError(f"Expected a mapping, got {data} with type {type(data)}")


def check_literal(data: object, expected: tuple[T]) -> T:
    if data in expected:
        return data
    raise TypeError(f"Expected one of {expected}, got {data}")


def check_list(data: object, model: type[T_List]) -> T_List:
    model_args = get_args(model)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list, got {data} with type {type(data)}")
    try:
        [check_type(a, model_args[0]) for a in data]
    except TypeError as e:
        raise TypeError(f"Error in list: {e}") from e
    return data  # type: ignore[return-value]


def check_tuple(data: object, model: type[T_Tuple]) -> T_Tuple:
    """
    Checks if the 'data' object is a tuple that conforms to the 'model' type annotation.

    Args:
        data: The object to be checked.
        model: The tuple type annotation (e.g., tuple[int, str], tuple[float, ...]).

    Returns:
        The 'data' object, cast to the specific tuple type `T`, if it conforms.

    Raises:
        TypeError: If 'data' is not a tuple, or if its elements do not match
                   the types specified in 'model'.
        ValueError: If 'data' is a tuple but its length does not match a
                    fixed-length `model` tuple.
    """
    # 1. Basic check: Is data a tuple?
    if not isinstance(data, tuple):
        raise TypeError(f"Expected a tuple, but got {type(data).__name__}: {data!r}")

    # 2. Get the origin and arguments of the model type hint
    origin = get_origin(model)
    args = get_args(model)

    # Ensure the model is actually a tuple type hint
    if origin is not tuple:  # `origin is Tuple` for older Python <3.9
        raise TypeError(f"Expected a tuple type hint (e.g., tuple[int, str]), but got {model!r}")

    # 3. Handle fixed-length tuples (e.g., tuple[int, str])
    if args and args[-1] is not Ellipsis:
        # Check if the length of data matches the expected length of the model
        if len(data) != len(args):
            raise ValueError(
                f"Tuple length mismatch: Expected {len(args)} elements for type {model}, "
                f"but got {len(data)} elements in {data!r}"
            )
        # Check each element's type
        for i, (item, expected_type) in enumerate(zip(data, args, strict=False)):
            try:
                # Recursively check item against its expected type
                # For basic types, this will be isinstance. For nested generics, it will recurse.
                # Here, we need a general type checking utility. Let's make a simple one.
                _check_item_type(item, expected_type)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Element at index {i} failed type check: {e}. "
                    f"Expected {expected_type}, got {type(item).__name__} ({item!r})"
                ) from e

    # 4. Handle variable-length tuples (e.g., tuple[int, ...])
    elif args and args[-1] is Ellipsis:
        # The first argument is the type of all elements
        expected_item_type = args[0]
        for i, item in enumerate(data):
            try:
                check_type(item, expected_item_type)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Element at index {i} failed type check: {e}. "
                    f"Expected {expected_item_type}, got {type(item).__name__} ({item!r})"
                ) from e
    # 5. Handle empty tuple `tuple[()]` or raw `tuple` type
    # If args is empty, it's either `tuple[()]` or a raw `tuple` annotation.
    # `tuple[()]` means it must be an empty tuple.
    elif len(args) == 0:
        if data:  # If data is not empty
            raise ValueError(f"Expected an empty tuple for type {model}, but got {data!r}")

    # If it's a raw `tuple` annotation (without brackets or ellipsis), it implies `tuple[Any, ...]`.
    # In this case, just checking if `data` is an instance of `tuple` is sufficient.
    # The initial `if not isinstance(data, tuple)` handles this already.
    # So if we reach here, it's an empty tuple or a raw tuple, and it's valid.

    # If all checks pass, return the data, cast to the specific tuple type.
    # Type checkers understand that if no exception is raised, the type is correct.
    return data  # type: ignore [return-value] # Mypy often needs a little help here for TypeVar


def check_type(data: JSON2, model: type[T]) -> JSON2:
    model_origin = get_origin(model)
    model_args = get_args(model)
    if model is str:
        return check_str(data)

    if model is int:
        return check_int(data)

    if model is type(None):
        return check_none(data)

    if model is bool:
        return check_bool(data)

    if isinstance(model, _LiteralGenericAlias):
        return check_literal(data, model_args)

    if type(model) is _TypedDictMeta:
        _mapping = check_mapping(data)
        keys, types = zip(*get_type_hints(model).items(), strict=True)
        check_keys(_mapping, keys)

        for key, typ in zip(keys, types, strict=True):
            try:
                _ = check_type(data[key], typ)
                print(f"{key} is OK")
            except TypeError as e:
                raise TypeError(f"Error in field {key}: {e}") from e
        return data

    if model_origin is list:
        return check_list(data, model)

    if model_origin is tuple:
        return check_tuple(data, model)
    else:
        raise TypeError(f"Unsupported type {model}")


class Y(TypedDict):
    a: int


class X(TypedDict):
    a: int
    b: Literal["str"]
    c: list[str]
    d: Y
    e: bool


if __name__ == "__main__":
    check_type("1", str)
    check_type(1, int)
    check_type(None, type(None))
    check_type(True, bool)
    check_type({"a": 1, "b": "str", "c": ["str"], "d": {"a": 1}, "e": True}, X)
