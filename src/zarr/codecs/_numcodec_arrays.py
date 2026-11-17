"""
The one place where zarr hands array chunks to a numcodecs codec.

Zarr and numcodecs disagree about what a chunk is. To zarr a chunk is a logical
array: its values and its shape, whatever their layout in memory. Numcodecs codecs
work on the memory of the array they are given. Most of them flatten it with
`order="A"`, which reads an F-contiguous array in transposed element order
(gh-3558), and many return a flat array (gh-4553).

The functions here translate between the two. Each one gives the numcodecs codec
a C-contiguous array, so that the codec sees the elements in C order, and reads
the flat or shaped array the codec returns back in C order. Codecs that pass array
chunks to numcodecs call these functions instead of calling `encode` and `decode`
on the numcodecs codec themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from zarr.abc.numcodec import Numcodec
    from zarr.core.buffer import NDArrayLike


def encode_array_array(
    codec: Numcodec, chunk: NDArrayLike, encoded_shape: tuple[int, ...]
) -> npt.NDArray[Any]:
    """
    Encode a chunk with a numcodecs codec that maps an array to an array.

    Parameters
    ----------
    codec
        The numcodecs codec.
    chunk
        The chunk to encode, in any memory layout.
    encoded_shape
        The shape that the zarr codec declares for the encoded chunk.

    Returns
    -------
    The encoded chunk, with the shape `encoded_shape`.
    """
    encoded: npt.NDArray[Any] = codec.encode(np.ascontiguousarray(chunk))
    return encoded.reshape(encoded_shape)


def decode_array_array(
    codec: Numcodec, encoded: NDArrayLike, shape: tuple[int, ...]
) -> npt.NDArray[Any]:
    """
    Decode a chunk with a numcodecs codec that maps an array to an array.

    Parameters
    ----------
    codec
        The numcodecs codec.
    encoded
        The encoded chunk, in any memory layout.
    shape
        The shape of the decoded chunk.

    Returns
    -------
    The decoded chunk, with the shape `shape`.
    """
    decoded: npt.NDArray[Any] = codec.decode(np.ascontiguousarray(encoded))
    return decoded.reshape(shape)


def encode_array_bytes(codec: Numcodec, chunk: NDArrayLike) -> Any:
    """
    Encode a chunk with a numcodecs codec that maps an array to bytes.

    Parameters
    ----------
    codec
        The numcodecs codec.
    chunk
        The chunk to encode, in any memory layout.

    Returns
    -------
    The bytes-like object that the numcodecs codec returns.
    """
    return codec.encode(np.ascontiguousarray(chunk))


def decode_array_bytes(codec: Numcodec, encoded: Any, shape: tuple[int, ...]) -> npt.NDArray[Any]:
    """
    Decode a chunk with a numcodecs codec that maps an array to bytes.

    Parameters
    ----------
    codec
        The numcodecs codec.
    encoded
        The encoded chunk, as a bytes-like object.
    shape
        The shape of the decoded chunk.

    Returns
    -------
    The decoded chunk, with the shape `shape`.
    """
    decoded: npt.NDArray[Any] = codec.decode(encoded)
    return decoded.reshape(shape)
