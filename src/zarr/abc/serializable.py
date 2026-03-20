from typing import Protocol, Self, TypeVar

T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class JSONSerializable(Protocol[T_co, T_contra]):
    @classmethod
    def from_json(cls, obj: T_contra) -> Self:
        """
        Deserialize from an instance of T_contra.
        """
        ...

    @classmethod
    def try_from_json(cls, obj: object) -> Self:
        """
        Deserialize from an unknown object. Details of any
        deserialization failure should be conveyed via an exception.
        """
        ...

    def to_json(self) -> T_co:
        """
        Serialize to JSON.
        """
        ...
