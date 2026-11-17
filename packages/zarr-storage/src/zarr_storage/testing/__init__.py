"""The conformance suite every backend runs.

A backend's test class subclasses `StoreSuite` and overrides `make_store` to return an empty,
writable `Store` over a fresh backend; it adds the mixin for each optional capability the backend has
(`ConditionalPutSuite`, `ConditionalDeleteSuite`, `ConditionalRaceSuite`, `KeyAndDirectorySuite`,
`PickleSuite`, `UniqueGenerationsSuite`) or the one
that checks it refuses it cleanly (`NoConditionalPutSuite`, `NoConditionalDeleteSuite`,
`KeyDirectoryConflictSuite`). Every test runs through the sync methods and again through the async
methods. Install the `testing` extra for pytest and Hypothesis.
"""

from zarr_storage.testing._api import AsyncAPI, FullBackend, StoreAPI
from zarr_storage.testing._suite import (
    ConditionalDeleteSuite,
    ConditionalPutSuite,
    ConditionalRaceSuite,
    KeyAndDirectorySuite,
    KeyDirectoryConflictSuite,
    NoConditionalDeleteSuite,
    NoConditionalPutSuite,
    PickleSuite,
    StoreSuite,
    UniqueGenerationsSuite,
)

__all__ = [
    "AsyncAPI",
    "ConditionalDeleteSuite",
    "ConditionalPutSuite",
    "ConditionalRaceSuite",
    "FullBackend",
    "KeyAndDirectorySuite",
    "KeyDirectoryConflictSuite",
    "NoConditionalDeleteSuite",
    "NoConditionalPutSuite",
    "PickleSuite",
    "StoreAPI",
    "StoreSuite",
    "UniqueGenerationsSuite",
]
