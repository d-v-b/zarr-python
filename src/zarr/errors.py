__all__ = [
    "BaseZarrError",
    "ContainsArrayAndGroupError",
    "ContainsArrayError",
    "ContainsGroupError",
    "GroupNotFoundError",
    "MetadataValidationError",
    "NodeTypeValidationError",
]


class BaseZarrError(ValueError):
    """
    Base class for Zarr errors.
    """


class GroupNotFoundError(BaseZarrError, FileNotFoundError):
    """
    Raised when a group isn't found at a certain path.
    """

    _msg = "No group found in store {!r} at path {!r}"

    def __init__(self, *args: object) -> None:
        if len(args) == 1:
            super().__init__(args[0])
        else:
            super().__init__(*args)


class ArrayNotFoundError(BaseZarrError):
    """Raised when an array does not exist at a certain path."""


class NodeNotFoundError(BaseZarrError):
    """Raised when an array or group does not exist at a certain path."""


class ContainsGroupError(BaseZarrError):
    """Raised when a group already exists at a certain path."""

    _msg = "A group exists in store {!r} at path {!r}."

    def __init__(self, *args: object) -> None:
        if len(args) == 1:
            super().__init__(args[0])
        else:
            super().__init__(*args)


class ContainsArrayError(BaseZarrError):
    """Raised when an array already exists at a certain path."""

    _msg = "An array exists in store {!r} at path {!r}."

    def __init__(self, *args: object) -> None:
        if len(args) == 1:
            super().__init__(args[0])
        else:
            super().__init__(*args)


class ContainsArrayAndGroupError(BaseZarrError):
    """Raised when both array and group metadata are found at the same path."""

    _msg = (
        "Array and group metadata documents (.zarray and .zgroup) were both found in store "
        "{!r} at path {!r}. "
        "Only one of these files may be present in a given directory / prefix. "
        "Remove the .zarray file, or the .zgroup file, or both."
    )

    def __init__(self, *args: object) -> None:
        if len(args) == 1:
            super().__init__(args[0])
        else:
            super().__init__(*args)


class MetadataValidationError(BaseZarrError):
    """Raised when the Zarr metadata is invalid in some way"""

    _msg = "Invalid value for '{}'. Expected '{}'. Got '{}'."

    def __init__(self, *args: object) -> None:
        if len(args) == 1:
            super().__init__(args[0])
        else:
            super().__init__(*args)


class NodeTypeValidationError(MetadataValidationError):
    """
    Specialized exception when the node_type of the metadata document is incorrect..

    This can be raised when the value is invalid or unexpected given the context,
    for example an 'array' node when we expected a 'group'.
    """
