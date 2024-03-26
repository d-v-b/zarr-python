from __future__ import annotations
from typing_extensions import Self
from zarr.v3.store import StoreLike
from zarr.v3.config import RuntimeConfiguration
from zarr.v3.group import GroupMetadata, Group
from zarr.v3.array import ArrayMetadata, Array
from dataclasses import dataclass

# right now, the array metadata and the array model are identical
@dataclass
class ArrayModel(ArrayMetadata):
    @classmethod
    def from_stored(cls, array: Array) -> Self:
        return cls(**array.metadata.to_dict())

    def to_stored(
        self, store: StoreLike, runtime_configuration: RuntimeConfiguration, exists_ok: bool = False
    ) -> Array:
        return Array.create(
            store,
            shape=self.shape,
            dtype=self.dtype,
            fill_value=self.fill_value,
            chunk_key_encoding=self.chunk_key_encoding,
            codecs=self.codecs,
            dimension_names=self.dimension_names,
            attributes=self.attributes,
            runtime_configuration=runtime_configuration,
            exists_ok=exists_ok,
        )


# group model is group metadata + members
@dataclass
class GroupModel(GroupMetadata):
    members: dict[str, GroupModel | ArrayModel] | None

    def from_stored(cls, group: Group, depth: int = -1):
        # result: Self
        metadata = group.metadata.to_dict()
        members = {}
        if depth < -1:
            msg = (
                f"Got an invalid value for depth: {depth}. "
                "Depth must be an integer greater than or equal to -1."
            )
            raise ValueError(msg)
        if depth == 0:
            return cls(**metadata, members=None)
        for child in group.children():
            if isinstance(child, Array):
                members[child.store_path.path]

    def to_stored(self, store: StoreLike, exists_ok: bool = False):
        return Group.create(store, attributes=self.attributes, exists_ok=exists_ok)
