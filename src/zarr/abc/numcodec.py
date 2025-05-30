from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Generic, Self, TypedDict, TypeVar

from typing_extensions import Protocol, runtime_checkable

from zarr.core.buffer.core import Buffer, NDBuffer
from zarr.core.common import ZarrFormat


class BaseNumcodecConfig(TypedDict, total=False):
    id: str


TNCodecConfig = TypeVar("TNCodecConfig", bound=BaseNumcodecConfig)


@runtime_checkable
class Numcodec(Protocol, Generic[TNCodecConfig]):
    """
    This protocol models the numcodecs.abc.Codec interface.
    """

    codec_id: ClassVar[str]

    def encode(self, buf: NDBuffer | Buffer) -> NDBuffer | Buffer: ...

    def decode(
        self, buf: NDBuffer | Buffer, out: NDBuffer | Buffer | None = None
    ) -> NDBuffer | Buffer: ...

    def get_config(self) -> TNCodecConfig: ...

    @classmethod
    def from_config(cls, config: TNCodecConfig) -> Self: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class NumcodecWrapper:
    codec: Numcodec[BaseNumcodecConfig]

    def to_json(self, zarr_format: ZarrFormat) -> Mapping[str, object]:
        v2_config = self.codec.get_config()

        if zarr_format == 2:
            return v2_config
        elif zarr_format == 3:
            name = v2_config.pop("id")
            return {"name": name, "configuration": v2_config}
        else:
            raise ValueError(f"Unsupported zarr format: {zarr_format}")  # pragma: no cover


def get_numcodec_codec(codec_spec: BaseNumcodecConfig) -> NumcodecWrapper:
    from numcodecs import get_codec

    return NumcodecWrapper(codec=get_codec(codec_spec))


if __name__ == "__main__":
    get_numcodec_codec({"id": "gzip"})
    breakpoint()
