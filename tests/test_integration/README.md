# Zarr-Python and Zarrs Integration Tests

This directory contains integration tests that verify interoperability between zarr-python and Zarrs (a high-performance Zarr implementation written in Rust).

## Purpose

These tests ensure that:
1. Data written by zarr-python can be read by Zarrs
2. Data written by Zarrs can be read by zarr-python
3. Data can successfully round-trip between both implementations
4. Both Zarr v2 and v3 formats are compatible
5. Various codecs and data types work correctly across implementations

## Prerequisites

### Option 1: Using Pixi (Recommended)

The tests use pixi for environment management. Install pixi if you haven't already:

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash
```

Pixi will automatically install Rust and use pip to install the zarrs Python package (which is built from Rust source).

### Option 2: Manual Installation

Install zarrs manually in your Python environment. Note that zarrs requires Rust to be installed:

```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install zarrs via pip (will compile from source)
pip install zarrs
```

## Running the Tests

### With Pixi

From this directory:

```bash
# First time setup: install environment and zarrs
pixi run setup

# Run all tests (automatically installs zarrs if needed)
pixi run test

# Run specific test
pixi run pytest test_zarrs_interop.py::TestZarrsInterop::test_basic_array_roundtrip_v3 -v
```

### With Regular Pytest

```bash
# From the repository root
pytest tests/test_integration/test_zarrs_interop.py -v

# Or from this directory
pytest test_zarrs_interop.py -v
```

### With Hatch (Recommended for CI/CD)

Hatch will automatically install Rust and compile zarrs from source:

```bash
# From repository root - run the integration tests
hatch env run --env zarrs-integration run-pytest

# Run with coverage
hatch env run --env zarrs-integration run-coverage

# List installed packages in the environment
hatch env run --env zarrs-integration list-env

# Run a specific test
hatch env run --env zarrs-integration pytest tests/test_integration/test_zarrs_interop.py::TestZarrsInterop::test_basic_array_roundtrip_v3 -v
```

Note: The first run will take longer as zarrs needs to be compiled from Rust source.

## Test Coverage

### Basic Interoperability
- `test_basic_array_roundtrip_v3` - Basic v3 format round-trip
- `test_basic_array_roundtrip_v2` - Basic v2 format round-trip
- `test_multidimensional_array_v3` - 3D array handling
- `test_zarrs_create_zarr_read_v3` - Zarrs → zarr-python direction

### Data Types
- `test_dtype_compatibility` - Integer and float types (i1, i2, i4, i8, u1, u2, u4, u8, f4, f8)

### Codecs and Compression
- `test_compression_gzip_v3` - Gzip compression (v3)
- `test_compression_blosc_v2` - Blosc compression (v2)

### Advanced Features
- `test_fill_value_v3` - Fill value handling
- `test_group_hierarchy_v3` - Group and hierarchy support
- `test_partial_writes_v3` - Partial chunk writes
- `test_attributes_v3` - Attribute preservation

### Edge Cases
- `test_empty_array_v3` - Empty arrays
- `test_single_element_array_v3` - Single element arrays
- `test_large_chunks_v3` - Large chunk sizes

## Test Architecture

Each test follows this pattern:
1. **Create** - Write Zarr data using zarr-python
2. **Read** - Read the data using Zarrs
3. **Modify** - Optionally modify and write back using Zarrs
4. **Verify** - Read with zarr-python and verify correctness

## Troubleshooting

### Zarrs Not Found

If you see `ModuleNotFoundError: No module named 'zarrs'`:
- Make sure you're running tests in the pixi environment: `pixi run test`
- Or install zarrs: `pip install zarrs`

### Test Failures

If tests fail:
1. Check that both zarr-python and zarrs are up to date
2. Verify the test store directory is writable
3. Check for version compatibility issues between implementations

## Adding New Tests

When adding new interoperability tests:
1. Follow the existing test pattern (create → read → modify → verify)
2. Test both v2 and v3 formats where applicable
3. Include both directions: zarr-python → Zarrs and Zarrs → zarr-python
4. Add appropriate pytest markers if needed
5. Document any special requirements or dependencies

## CI Integration

### Option 1: Using Hatch (Recommended)

The simplest approach for CI is to use the hatch environment:

```yaml
- name: Setup Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.12'

- name: Install hatch
  run: pip install hatch

- name: Install Rust
  uses: dtolnay/rust-toolchain@stable

- name: Run Zarrs integration tests
  run: hatch env run --env zarrs-integration run-pytest

- name: Cache hatch environment
  uses: actions/cache@v4
  with:
    path: ~/.local/share/hatch
    key: hatch-zarrs-${{ hashFiles('pyproject.toml') }}
```

### Option 2: Using Pixi

```yaml
- name: Install pixi
  run: curl -fsSL https://pixi.sh/install.sh | bash

- name: Run Zarrs integration tests
  working-directory: tests/test_integration
  run: pixi run test

- name: Cache pixi environment
  uses: actions/cache@v4
  with:
    path: ~/.pixi
    key: pixi-zarrs-${{ hashFiles('tests/test_integration/pixi.toml') }}
```

Note: The first run will take longer as it compiles zarrs from source. Use caching to speed up subsequent runs.
