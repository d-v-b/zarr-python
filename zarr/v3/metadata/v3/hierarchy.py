from __future__ import annotations
from dataclasses import dataclass, field
from zarr.v3.metadata.v3.array import ArrayMetadata
from zarr.v3.metadata.v3.group import GroupMetadata
from typing import Dict, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.v3.types import Attributes
    from zarr.v3.group import Group


@dataclass(frozen=True)
class ArrayModel(ArrayMetadata):
    @classmethod
    def from_array(cls, array: Any, **kwargs):
        """
        Create an ArrayModel from an existing array-like, with the option to add / override properties via kwargs
        """
        # return cls(shape=array.shape, dtype=array.dtype, )
        ...

    def to_stored(self, store: Any, path: str):
        """
        Turn this ArrayModel into a Zarr Array
        """
        ...


@dataclass(frozen=True)
class GroupModel(GroupMetadata):
    members: Dict[str, Union[GroupModel, ArrayModel]] = field(default_factory=dict)

    @classmethod
    def from_group(cls, group: Group, **kwargs):
        """
        Create a GroupModel from an existing zarr Group, with the option to add / override properties via kwargs
        """
        ...

    def to_stored(self, store: Any, path: str, **kwargs):
        """
        Turn this GroupModel into a Zarr group.
        """
        # store metadata + attributes, then call member.to_stored() for each member.
        ...
