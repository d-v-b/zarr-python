from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

if TYPE_CHECKING:
    from zarr.v3.common import NamedConfig, RuntimeConfiguration
    from typing import (
        TYPE_CHECKING,
        Any,
        Dict,
        Literal,
        Optional,
        Tuple,
        Type,
    )

import numpy as np

from zarr.v3.abc.codec import ArrayArrayCodec
from zarr.v3.codecs.registry import register_codec

if TYPE_CHECKING:
    from zarr.v3.metadata import ArraySpec, CodecMetadata, RuntimeConfiguration


@dataclass(frozen=True)
class TransposeCodecConfigurationMetadata:
    order: Tuple[int, ...]


@dataclass(frozen=True)
class TransposeCodecMetadata:
    configuration: TransposeCodecConfigurationMetadata
    name: Literal["transpose"] = field(default="transpose", init=False)


@dataclass(frozen=True)
class TransposeCodec(ArrayArrayCodec):
    order: Tuple[int, ...]
    is_fixed_size = True

    @classmethod
    def from_metadata(cls, codec_metadata: NamedConfig) -> TransposeCodec:
        assert isinstance(codec_metadata, TransposeCodecMetadata)
        return cls(order=codec_metadata.configuration.order)

    def evolve(self, *, ndim: int, **_kwargs) -> TransposeCodec:
        # Compatibility with older version of ZEP1
        if self.order == "F":  # type: ignore
            order = tuple(ndim - x - 1 for x in range(ndim))

        elif self.order == "C":  # type: ignore
            order = tuple(range(ndim))

        else:
            assert len(self.order) == ndim, (
                "The `order` tuple needs have as many entries as "
                + f"there are dimensions in the array. Got: {self.order}"
            )
            assert len(self.order) == len(set(self.order)), (
                "There must not be duplicates in the `order` tuple. " + f"Got: {self.order}"
            )
            assert all(0 <= x < ndim for x in self.order), (
                "All entries in the `order` tuple must be between 0 and "
                + f"the number of dimensions in the array. Got: {self.order}"
            )
            order = tuple(self.order)

        if order != self.order:
            return evolve(self, order=order)
        return self

    @classmethod
    def get_metadata_class(cls) -> Type[TransposeCodecMetadata]:
        return TransposeCodecMetadata

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        from zarr.v3.metadata import ArraySpec

        return ArraySpec(
            shape=tuple(chunk_spec.shape[self.order[i]] for i in range(chunk_spec.ndim)),
            data_type=chunk_spec.data_type,
            fill_value=chunk_spec.fill_value,
        )

    async def decode(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        inverse_order = [0] * chunk_spec.ndim
        for x, i in enumerate(self.order):
            inverse_order[x] = i
        chunk_array = chunk_array.transpose(inverse_order)
        return chunk_array

    async def encode(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[np.ndarray]:
        chunk_array = chunk_array.transpose(self.order)
        return chunk_array

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length


register_codec("transpose", TransposeCodec)
