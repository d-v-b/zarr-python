from __future__ import annotations
from dataclasses import dataclass, replace
import json

from zarr.v3.array import Array
from zarr.v3.common import ZARR_JSON, RuntimeConfiguration
import zarr.v3.metadata.v3 as MetaV3
import zarr.v3.metadata.v2 as MetaV2
from zarr.v3.store import StoreLike, StorePath, make_store_path
from zarr.v3.sync import sync

from typing import Any, Dict, Literal, Optional, Union

from zarr.v3.types import Attributes


def group_metadata_from_dict(data: Any):
    data_dict = json.loads(data)


@dataclass(frozen=True)
class Group:
    attributes: Attributes
    metadata: Union[MetaV2.GroupMetadata, MetaV3.GroupMetadata]
    store_path: StorePath
    runtime_configuration: RuntimeConfiguration

    @classmethod
    async def create_async(
        cls,
        store: StoreLike,
        *,
        version: Literal[2, 3] = 3,
        attributes: Optional[Attributes] = None,
        exists_ok: bool = False,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Group:
        store_path = make_store_path(store)

        if not exists_ok:
            assert not await (store_path / ZARR_JSON).exists_async()

        metadata: Union[MetaV2.GroupMetadata, MetaV3.GroupMetadata]

        if version == 2:
            metadata = MetaV2.GroupMetadata()

        elif version == 3:
            metadata = MetaV3.GroupMetadata()

        group = cls(
            attributes=attributes,
            metadata=metadata,
            store_path=store_path,
            runtime_configuration=runtime_configuration,
        )
        await group._save_metadata()
        return group

    @classmethod
    def create(
        cls,
        store: StoreLike,
        *,
        version: Literal[2, 3] = 3,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Group:
        return sync(
            cls.create_async(
                store,
                version=version,
                attributes=attributes,
                exists_ok=exists_ok,
                runtime_configuration=runtime_configuration,
            ),
            runtime_configuration.asyncio_loop,
        )

    @classmethod
    async def open_async(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Group:
        store_path = make_store_path(store)
        zarr_json_bytes = await (store_path / ZARR_JSON).get_async()
        assert zarr_json_bytes is not None
        return cls.from_dict(store_path, json.loads(zarr_json_bytes), runtime_configuration)

    @classmethod
    def open(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Group:
        return sync(
            cls.open_async(store, runtime_configuration),
            runtime_configuration.asyncio_loop,
        )

    @classmethod
    def from_dict(
        cls,
        store_path: StorePath,
        zarr_json: Dict[str, Any],
        runtime_configuration: RuntimeConfiguration,
    ) -> Group:
        zarr_version = zarr_json["zarr_format"]
        if zarr_version == 2:
            metadata = MetaV2.GroupMetadata()
        elif zarr_version == 3:
            metadata = MetaV3.GroupMetadata()
        else:
            raise ValueError(f"Invalid `zarr_format` property. Got {zarr_version}, expected 2 or 3")
        group = cls(
            metadata=metadata,
            store_path=store_path,
            runtime_configuration=runtime_configuration,
        )
        return group

    @classmethod
    async def open_or_array(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Union[Array, Group]:
        store_path = make_store_path(store)
        zarr_json_bytes = await (store_path / ZARR_JSON).get_async()
        if zarr_json_bytes is None:
            raise KeyError
        zarr_json = json.loads(zarr_json_bytes)
        if zarr_json["node_type"] == "group":
            return cls.from_dict(store_path, zarr_json, runtime_configuration)
        if zarr_json["node_type"] == "array":
            return Array.from_dict(
                store_path, zarr_json, runtime_configuration=runtime_configuration
            )
        raise KeyError

    async def _save_metadata(self) -> None:
        await (self.store_path / ZARR_JSON).set_async(self.metadata.to_bytes())

    async def get_async(self, path: str) -> Union[Array, Group]:
        return await self.__class__.open_or_array(
            self.store_path / path, self.runtime_configuration
        )

    def __getitem__(self, path: str) -> Union[Array, Group]:
        return sync(self.get_async(path), self.runtime_configuration.asyncio_loop)

    async def create_group_async(self, path: str, **kwargs) -> Group:
        runtime_configuration = kwargs.pop("runtime_configuration", self.runtime_configuration)
        return await self.__class__.create_async(
            self.store_path / path,
            runtime_configuration=runtime_configuration,
            **kwargs,
        )

    def create_group(self, path: str, **kwargs) -> Group:
        return sync(self.create_group_async(path), self.runtime_configuration.asyncio_loop)

    async def create_array_async(self, path: str, **kwargs) -> Array:
        runtime_configuration = kwargs.pop("runtime_configuration", self.runtime_configuration)
        return await Array.create_async(
            self.store_path / path,
            runtime_configuration=runtime_configuration,
            **kwargs,
        )

    def create_array(self, path: str, **kwargs) -> Array:
        return sync(
            self.create_array_async(path, **kwargs),
            self.runtime_configuration.asyncio_loop,
        )

    async def update_attributes_async(self, new_attributes: Dict[str, Any]) -> Group:
        new_metadata = replace(self.metadata, attributes=new_attributes)

        # Write new metadata
        await (self.store_path / ZARR_JSON).set_async(new_metadata.to_bytes())
        return replace(self, metadata=new_metadata)

    def update_attributes(self, new_attributes: Dict[str, Any]) -> Group:
        return sync(
            self.update_attributes_async(new_attributes),
            self.runtime_configuration.asyncio_loop,
        )

    def __repr__(self):
        return f"<Group {self.store_path}>"
