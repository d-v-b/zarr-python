from typing import Protocol, Self


class JSONSerializable[T_contra, T_co](Protocol):
    @classmethod
    def from_json(cls, obj: T_contra) -> Self:
        """
        Deserialize from an instance of T_contra.
        """
        ...

    def to_json(self) -> T_co:
        """
        Serialize to JSON.
        """
        ...
