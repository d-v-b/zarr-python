"""
Integration tests for zarr-python and Zarrs (Rust implementation) interoperability.

These tests verify that data can be round-tripped between zarr-python and Zarrs:
1. Create Zarr data with zarr-python
2. Reencode it with Zarrs (verifies Zarrs can read it)
3. Read it back with zarr-python and verify correctness

Prerequisites:
- Install zarrs_tools: `cargo install zarrs_tools`
- Or use the hatch environment: `hatch env run --env zarrs-integration run-pytest`
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import zarr
from zarr import Array, Group

# Check if zarrs_reencode CLI tool is available
def _check_zarrs_cli() -> bool:
    """Check if zarrs_reencode is available in PATH."""
    try:
        result = subprocess.run(
            ["zarrs_reencode", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


ZARRS_AVAILABLE = _check_zarrs_cli()

pytestmark = pytest.mark.skipif(
    not ZARRS_AVAILABLE, reason="zarrs_reencode CLI not found in PATH"
)


def reencode_with_zarrs(
    input_path: Path, output_path: Path, **kwargs: Any
) -> subprocess.CompletedProcess:
    """
    Reencode a Zarr array using zarrs_reencode CLI.

    This validates that Zarrs can read the input array and write it to output.
    """
    cmd = ["zarrs_reencode", str(input_path), str(output_path)]

    # Add optional arguments
    if "chunk_shape" in kwargs:
        cmd.extend(["--chunk-shape", ",".join(map(str, kwargs["chunk_shape"]))])
    if "data_type" in kwargs:
        cmd.extend(["--data-type", kwargs["data_type"]])
    if "validate" in kwargs and kwargs["validate"]:
        cmd.append("--validate")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        raise RuntimeError(
            f"zarrs_reencode failed with return code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    return result


class TestZarrsInterop:
    """Test interoperability between zarr-python and Zarrs."""

    @pytest.fixture
    def temp_store(self, tmp_path: Path) -> Path:
        """Create a temporary directory for testing."""
        store_path = tmp_path / "zarr_store"
        store_path.mkdir(exist_ok=True)
        yield store_path
        if store_path.exists():
            shutil.rmtree(store_path)

    def test_basic_array_roundtrip_v3(self, tmp_path: Path) -> None:
        """Test basic array round-trip with v3 format."""
        input_store = tmp_path / "input"
        output_store = tmp_path / "output"

        # Step 1: Create array with zarr-python (v3)
        data_original = np.arange(100, dtype="i4").reshape(10, 10)
        z = zarr.open_array(
            store=input_store,
            mode="w",
            shape=data_original.shape,
            chunks=(5, 5),
            dtype=data_original.dtype,
            zarr_format=3,
        )
        z[:] = data_original

        # Step 2: Reencode with Zarrs (validates reading and writing)
        reencode_with_zarrs(input_store, output_store, validate=True)

        # Step 3: Read back with zarr-python
        z_read = zarr.open_array(store=output_store, mode="r")
        np.testing.assert_array_equal(z_read[:], data_original)
        assert z_read.shape == data_original.shape
        assert z_read.dtype == data_original.dtype

    def test_basic_array_roundtrip_v2(self, tmp_path: Path) -> None:
        """Test basic array round-trip with v2 format."""
        input_store = tmp_path / "input"
        output_store = tmp_path / "output"

        # Step 1: Create array with zarr-python (v2)
        data_original = np.arange(50, dtype="f8")
        z = zarr.open_array(
            store=input_store,
            mode="w",
            shape=data_original.shape,
            chunks=(10,),
            dtype=data_original.dtype,
            zarr_format=2,
        )
        z[:] = data_original

        # Step 2: Reencode with Zarrs
        reencode_with_zarrs(input_store, output_store, validate=True)

        # Step 3: Read back with zarr-python
        z_read = zarr.open_array(store=output_store, mode="r")
        np.testing.assert_array_equal(z_read[:], data_original)

    def test_multidimensional_array_v3(self, tmp_path: Path) -> None:
        """Test 3D array interoperability with v3 format."""
        input_store = tmp_path / "input"
        output_store = tmp_path / "output"

        # Create 3D array with zarr-python
        shape = (20, 15, 10)
        chunks = (5, 5, 5)
        data_original = np.random.randint(0, 100, size=shape, dtype="i4")

        z = zarr.open_array(
            store=input_store,
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype="i4",
            zarr_format=3,
        )
        z[:] = data_original

        # Reencode with Zarrs
        reencode_with_zarrs(input_store, output_store, validate=True)

        # Verify with zarr-python
        z_read = zarr.open_array(store=output_store, mode="r")
        np.testing.assert_array_equal(z_read[:], data_original)

    def test_dtype_compatibility(self, tmp_path: Path) -> None:
        """Test various dtypes for interoperability."""
        dtypes = [
            ("i1", np.int8),
            ("i2", np.int16),
            ("i4", np.int32),
            ("i8", np.int64),
            ("u1", np.uint8),
            ("u2", np.uint16),
            ("u4", np.uint32),
            ("u8", np.uint64),
            ("f4", np.float32),
            ("f8", np.float64),
        ]

        for dtype_str, dtype_np in dtypes:
            input_store = tmp_path / f"input_{dtype_str}"
            output_store = tmp_path / f"output_{dtype_str}"

            # Create with zarr-python
            data = np.array([1, 2, 3, 4, 5], dtype=dtype_np)
            z = zarr.open_array(
                store=input_store,
                mode="w",
                shape=data.shape,
                chunks=(5,),
                dtype=dtype_np,
                zarr_format=3,
            )
            z[:] = data

            # Reencode with Zarrs
            reencode_with_zarrs(input_store, output_store, validate=True)

            # Verify with zarr-python
            z_read = zarr.open_array(store=output_store, mode="r")
            np.testing.assert_array_equal(z_read[:], data)
            assert z_read.dtype == dtype_np

    def test_fill_value_v3(self, tmp_path: Path) -> None:
        """Test fill value handling in v3 format."""
        input_store = tmp_path / "input"
        output_store = tmp_path / "output"

        fill_value = -999
        shape = (10, 10)

        # Create array with fill value
        z = zarr.open_array(
            store=input_store,
            mode="w",
            shape=shape,
            chunks=(5, 5),
            dtype="i4",
            fill_value=fill_value,
            zarr_format=3,
        )

        # Write partial data
        z[0:5, 0:5] = 1

        # Reencode with Zarrs
        reencode_with_zarrs(input_store, output_store, validate=True)

        # Read back with zarr-python
        z_read = zarr.open_array(store=output_store, mode="r")

        # Check that written data is correct
        np.testing.assert_array_equal(z_read[0:5, 0:5], 1)
        # Check that unwritten chunks have fill value
        np.testing.assert_array_equal(z_read[5:10, 5:10], fill_value)

    def test_compression_gzip_v3(self, tmp_path: Path) -> None:
        """Test gzip compression interoperability with v3 format."""
        input_store = tmp_path / "input"
        output_store = tmp_path / "output"

        data = np.arange(1000, dtype="i4")

        # Create compressed array with zarr-python
        z = zarr.open_array(
            store=input_store,
            mode="w",
            shape=data.shape,
            chunks=(100,),
            dtype="i4",
            zarr_format=3,
            codecs=[
                zarr.codecs.BytesCodec(),
                zarr.codecs.GzipCodec(level=5),
            ],
        )
        z[:] = data

        # Reencode with Zarrs (should decompress and recompress)
        reencode_with_zarrs(input_store, output_store, validate=True)

        # Verify with zarr-python
        z_read = zarr.open_array(store=output_store, mode="r")
        np.testing.assert_array_equal(z_read[:], data)

    def test_compression_blosc_v2(self, tmp_path: Path) -> None:
        """Test blosc compression interoperability with v2 format."""
        input_store = tmp_path / "input"
        output_store = tmp_path / "output"

        data = np.random.random(1000).astype("f8")

        # Create compressed array with zarr-python (v2 uses numcodecs format)
        # For v2, use the default compressor which uses numcodecs
        z = zarr.open_array(
            store=input_store,
            mode="w",
            shape=data.shape,
            chunks=(100,),
            dtype="f8",
            zarr_format=2,
            # Use default compressor for v2 (numcodecs.Blosc)
        )
        z[:] = data

        # Reencode with Zarrs
        reencode_with_zarrs(input_store, output_store, validate=True)

        # Verify with zarr-python
        z_read = zarr.open_array(store=output_store, mode="r")
        np.testing.assert_array_almost_equal(z_read[:], data)

    def test_chunk_shape_reencoding(self, tmp_path: Path) -> None:
        """Test reencoding verifies data integrity even with original chunks."""
        input_store = tmp_path / "input"
        output_store = tmp_path / "output"

        shape = (100, 100)
        original_chunks = (10, 10)

        data = np.arange(10000, dtype="i4").reshape(shape)

        # Create array with zarr-python
        z = zarr.open_array(
            store=input_store,
            mode="w",
            shape=shape,
            chunks=original_chunks,
            dtype="i4",
            zarr_format=3,
        )
        z[:] = data

        # Reencode with Zarrs (validates data integrity)
        # Note: zarrs_reencode may preserve original chunk shape
        reencode_with_zarrs(input_store, output_store, validate=True)

        # Verify data is preserved correctly
        z_read = zarr.open_array(store=output_store, mode="r")
        np.testing.assert_array_equal(z_read[:], data)
        assert z_read.shape == shape

    def test_partial_writes_v3(self, tmp_path: Path) -> None:
        """Test partial chunk writes."""
        input_store = tmp_path / "input"
        output_store = tmp_path / "output"

        shape = (100,)
        chunks = (10,)

        # Create array with zarr-python
        z = zarr.open_array(
            store=input_store,
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype="i4",
            zarr_format=3,
            fill_value=0,
        )

        # Write some chunks with zarr-python
        z[0:30] = 1
        z[60:80] = 2

        # Reencode with Zarrs
        reencode_with_zarrs(input_store, output_store, validate=True)

        # Verify
        z_read = zarr.open_array(store=output_store, mode="r")
        expected = np.zeros(shape, dtype="i4")
        expected[0:30] = 1
        expected[60:80] = 2
        np.testing.assert_array_equal(z_read[:], expected)

    def test_attributes_preservation_v3(self, tmp_path: Path) -> None:
        """Test that basic array properties are preserved."""
        input_store = tmp_path / "input"
        output_store = tmp_path / "output"

        # Create array with specific properties
        shape = (10, 20)
        chunks = (5, 10)
        data = np.arange(200, dtype="i4").reshape(shape)

        z = zarr.open_array(
            store=input_store,
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype="i4",
            zarr_format=3,
        )
        z[:] = data

        # Reencode with Zarrs
        reencode_with_zarrs(input_store, output_store, validate=True)

        # Verify properties are preserved
        z_read = zarr.open_array(store=output_store, mode="r")
        assert z_read.shape == shape
        assert z_read.dtype == np.dtype("i4")
        np.testing.assert_array_equal(z_read[:], data)


class TestZarrsInteropEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def temp_store(self, tmp_path: Path) -> Path:
        """Create a temporary directory for testing."""
        store_path = tmp_path / "zarr_store"
        store_path.mkdir(exist_ok=True)
        yield store_path
        if store_path.exists():
            shutil.rmtree(store_path)

    def test_single_element_array_v3(self, tmp_path: Path) -> None:
        """Test single element array."""
        input_store = tmp_path / "input"
        output_store = tmp_path / "output"

        data = np.array([42], dtype="i4")
        z = zarr.open_array(
            store=input_store,
            mode="w",
            shape=(1,),
            chunks=(1,),
            dtype="i4",
            zarr_format=3,
        )
        z[:] = data

        # Reencode with Zarrs
        reencode_with_zarrs(input_store, output_store, validate=True)

        # Verify
        z_read = zarr.open_array(store=output_store, mode="r")
        np.testing.assert_array_equal(z_read[:], data)

    def test_large_chunks_v3(self, tmp_path: Path) -> None:
        """Test array with large chunks."""
        input_store = tmp_path / "input"
        output_store = tmp_path / "output"

        shape = (1000, 1000)
        chunks = (500, 500)
        data = np.arange(1000000, dtype="i4").reshape(shape)

        z = zarr.open_array(
            store=input_store,
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype="i4",
            zarr_format=3,
        )
        z[:] = data

        # Reencode with Zarrs
        reencode_with_zarrs(input_store, output_store, validate=True)

        # Verify
        z_read = zarr.open_array(store=output_store, mode="r")
        np.testing.assert_array_equal(z_read[:], data)

    def test_different_endianness_v3(self, tmp_path: Path) -> None:
        """Test arrays with different endianness specifications."""
        input_store = tmp_path / "input"
        output_store = tmp_path / "output"

        data = np.arange(100, dtype="i4")

        # Create array with explicit endianness
        z = zarr.open_array(
            store=input_store,
            mode="w",
            shape=data.shape,
            chunks=(20,),
            dtype="i4",
            zarr_format=3,
            codecs=[
                zarr.codecs.BytesCodec(endian="little"),
                zarr.codecs.BloscCodec(cname="zstd", clevel=5),
            ],
        )
        z[:] = data

        # Reencode with Zarrs
        reencode_with_zarrs(input_store, output_store, validate=True)

        # Verify
        z_read = zarr.open_array(store=output_store, mode="r")
        np.testing.assert_array_equal(z_read[:], data)


if __name__ == "__main__":
    # Allow running tests directly with: python test_zarrs_interop.py
    pytest.main([__file__, "-v"])
