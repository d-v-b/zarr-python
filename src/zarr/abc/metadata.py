from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Generic, TypedDict, TypeVar

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.common import JSON

from dataclasses import asdict, dataclass, fields

__all__ = ["Metadata", "NamedConfig"]


@dataclass(frozen=True)
class Metadata:
    def to_dict(self) -> dict[str, JSON]:
        """
        Recursively serialize this model to a dictionary.
        This method inspects the fields of self and calls `x.to_dict()` for any fields that
        are instances of `Metadata`. Sequences of `Metadata` are similarly recursed into, and
        the output of that recursion is collected in a list.
        """
        out_dict = {}
        for field in fields(self):
            key = field.name
            value = getattr(self, key)
            if isinstance(value, Metadata):
                out_dict[field.name] = getattr(self, field.name).to_dict()
            elif isinstance(value, str):
                out_dict[key] = value
            elif isinstance(value, Sequence):
                out_dict[key] = tuple(v.to_dict() if isinstance(v, Metadata) else v for v in value)
            else:
                out_dict[key] = value

        return out_dict

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """
        Create an instance of the model from a dictionary
        """

        return cls(**data)


TConf = TypeVar("TConf", bound=Mapping[str, object])
TName = TypeVar("TName", bound=str)


class TDict(TypedDict, Generic[TName, TConf]):
    name: TName
    configuration: TConf


@dataclass(frozen=True)
class NamedConfig(Generic[TName, TConf]):
    config_required: ClassVar[bool]
    name: TName

    def __init_subclass__(cls, config_required: bool = False) -> None:
        """
        Some implementations of this class require a "configuration" key,
        whereas others are fine if "configuration" is unset. We define this method so that subclasses
        of ``NamedConfig`` can created with varying sensitivity to the absence of the "configuration"
        key.
        """
        cls.config_required = config_required
        return super().__init_subclass__()

    def to_dict(self) -> TDict[TName, TConf]:
        config = asdict(self)
        return {"name": type(self).name, "configuration": config}

    @classmethod
    def from_dict(cls, data: TDict[TName, TConf]) -> Self:
        name = data["name"]
        if cls.config_required:
            try:
                config = data.get("configuration")
            except KeyError as e:
                raise ValueError(
                    'The "configuration" key is required for this class, but it was not found in the provided mapping.'
                ) from e
        else:
            config = data.get("configuration", {})

        if name != cls.name:
            raise ValueError(f"expected name {cls.name!r}, got {name!r}")
        return cls(**config)  # type: ignore[arg-type]
