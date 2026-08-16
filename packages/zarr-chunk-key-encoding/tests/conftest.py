"""Shared fixtures.

The registry is process-wide module state, and entry-point discovery reads
the ambient Python environment, so tests are isolated from both by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import zarr_chunk_key_encoding.registry as registry_module

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a private registry, and no third-party entry points.

    Two independent hazards, both of which would otherwise make results
    depend on things the test does not control:

    - A test that registers an encoding would leak it into every later test,
      making outcomes depend on collection order.
    - Any lookup that misses triggers a real scan of the installed
      ``zarr_chunk_key_encoding`` entry point group. In an environment where
      some third-party package provides one, that would register it mid-suite
      -- or, if it is broken, emit a `ChunkKeyPluginWarning` that this
      package's ``filterwarnings = ["error"]`` turns into a failure in a test
      that has nothing to do with plugins.

    Tests that exercise discovery override ``entry_points`` themselves; a
    later `monkeypatch.setattr` wins, and both are undone at teardown.
    """
    monkeypatch.setattr(registry_module, "_registry", dict(registry_module._registry))
    monkeypatch.setattr(registry_module, "_entry_points_loaded", False)
    monkeypatch.setattr(registry_module, "entry_points", lambda group: _no_entry_points(group))


def _no_entry_points(group: str) -> Sequence[object]:
    """Stand in for `importlib.metadata.entry_points`, finding nothing."""
    return ()
