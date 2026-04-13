from __future__ import annotations

import contextlib
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

import numpy as np

from zarr.core.dtype.common import (
    DataTypeValidationError,
    DTypeJSON,
)

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from zarr.core.common import ZarrFormat
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


# ---------------------------------------------------------------------------
# V2 → V3 dtype name normalization
# ---------------------------------------------------------------------------

# Fixed V2 dtype strings that map directly to a V3 canonical name.
_V2_FIXED_NAMES: dict[str, str] = {
    "|b1": "bool",
    "|i1": "int8",
    "|u1": "uint8",
    ">i2": "int16",
    "<i2": "int16",
    ">i4": "int32",
    "<i4": "int32",
    ">i8": "int64",
    "<i8": "int64",
    ">u2": "uint16",
    "<u2": "uint16",
    ">u4": "uint32",
    "<u4": "uint32",
    ">u8": "uint64",
    "<u8": "uint64",
    ">f2": "float16",
    "<f2": "float16",
    ">f4": "float32",
    "<f4": "float32",
    ">f8": "float64",
    "<f8": "float64",
    ">c8": "complex64",
    "<c8": "complex64",
    ">c16": "complex128",
    "<c16": "complex128",
}

# Object dtype disambiguation by object_codec_id.
_V2_OBJECT_CODEC_MAP: dict[str, str] = {
    "vlen-utf8": "variable_length_utf8",
    "vlen-bytes": "variable_length_bytes",
}

# Regex patterns for parametric V2 dtypes, checked in order.
_V2_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^[><]U\d+$"), "fixed_length_utf32"),
    (re.compile(r"^\|S\d+$"), "null_terminated_bytes"),
    (re.compile(r"^\|V\d+$"), "raw_bytes"),
    (re.compile(r"^[><]M8"), "numpy.datetime64"),
    (re.compile(r"^[><]m8"), "numpy.timedelta64"),
)


def v2_to_v3_dtype_name(v2_name: str | object, object_codec_id: str | None = None) -> str:
    """Map a Zarr V2 dtype name to the canonical V3 dtype name.

    Parameters
    ----------
    v2_name : str or object
        The V2 dtype name. Non-string values (e.g. a list of field
        definitions for structured dtypes) are handled as a special case.
    object_codec_id : str or None
        The object codec ID from V2 metadata, used to disambiguate
        the ``"|O"`` numpy object dtype.

    Returns
    -------
    str
        The canonical V3 dtype name.

    Raises
    ------
    ValueError
        If the V2 name cannot be mapped to a V3 name.
    """
    # Structured dtypes have a non-string name (list of field definitions).
    if not isinstance(v2_name, str):
        return "structured"

    # Fixed names — O(1) dict lookup.
    if v2_name in _V2_FIXED_NAMES:
        return _V2_FIXED_NAMES[v2_name]

    # Object dtype — disambiguated by object_codec_id.
    if v2_name == "|O":
        if object_codec_id is not None and object_codec_id in _V2_OBJECT_CODEC_MAP:
            return _V2_OBJECT_CODEC_MAP[object_codec_id]
        raise ValueError(
            f"Cannot resolve V2 dtype '|O' without a recognized object_codec_id. "
            f"Got object_codec_id={object_codec_id!r}."
        )

    # Pattern-based names.
    for pattern, v3_name in _V2_PATTERNS:
        if pattern.match(v2_name):
            return v3_name

    raise ValueError(f"Cannot map V2 dtype name {v2_name!r} to a V3 dtype name.")


# This class is different from the other registry classes, which inherit from
# dict. IMO it's simpler to just do a dataclass. But long-term we should
# have just 1 registry class in use.
@dataclass(frozen=True, kw_only=True)
class DataTypeRegistry:
    """
    A registry for ZDType classes.

    This registry is a mapping from Zarr data type names to their
    corresponding ZDType classes.

    Attributes
    ----------
    contents : dict[str, type[ZDType[TBaseDType, TBaseScalar]]]
        The mapping from Zarr data type names to their corresponding
        ZDType classes.
    """

    contents: dict[str, type[ZDType[TBaseDType, TBaseScalar]]] = field(
        default_factory=dict, init=False
    )

    _lazy_load_list: list[EntryPoint] = field(default_factory=list, init=False)

    def _lazy_load(self) -> None:
        """
        Load all data types from the lazy load list and register them with
        the registry. After loading, clear the lazy load list.
        """
        for e in self._lazy_load_list:
            self.register(e.load()._zarr_v3_name, e.load())

        self._lazy_load_list.clear()

    def register(self: Self, key: str, cls: type[ZDType[TBaseDType, TBaseScalar]]) -> None:
        """
        Register a data type with the registry.

        Parameters
        ----------
        key : str
            The Zarr V3 name of the data type.
        cls : type[ZDType[TBaseDType, TBaseScalar]]
            The class of the data type to register.

        Notes
        -----
        This method is idempotent. If the data type is already registered, this
        method does nothing.
        """
        if key not in self.contents or self.contents[key] != cls:
            self.contents[key] = cls

    def unregister(self, key: str) -> None:
        """
        Unregister a data type from the registry.

        Parameters
        ----------
        key : str
            The key associated with the ZDType class to be unregistered.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the data type is not found in the registry.
        """
        if key in self.contents:
            del self.contents[key]
        else:
            raise KeyError(f"Data type '{key}' not found in registry.")

    def get(self, key: str) -> type[ZDType[TBaseDType, TBaseScalar]]:
        """
        Retrieve a registered ZDType class by its key.

        Parameters
        ----------
        key : str
            The key associated with the desired ZDType class.

        Returns
        -------
        type[ZDType[TBaseDType, TBaseScalar]]
            The ZDType class registered under the given key.

        Raises
        ------
        KeyError
            If the key is not found in the registry.
        """

        return self.contents[key]

    def match_dtype(self, dtype: TBaseDType) -> ZDType[TBaseDType, TBaseScalar]:
        """
        Match a native data type, e.g. a NumPy data type, to a registered ZDType.

        Parameters
        ----------
        dtype : TBaseDType
            The native data type to match.

        Returns
        -------
        ZDType[TBaseDType, TBaseScalar]
            The matched ZDType corresponding to the provided NumPy data type.

        Raises
        ------
        ValueError
            If the data type is a NumPy "Object" type, which is ambiguous, or if multiple
            or no Zarr data types are found that match the provided dtype.

        Notes
        -----
        This function attempts to resolve a Zarr data type from a given native data type.
        If the dtype is a NumPy "Object" data type, it raises a ValueError, as this type
        can represent multiple Zarr data types. In such cases, a specific Zarr data type
        should be explicitly constructed instead of relying on dynamic resolution.

        If multiple matches are found, it will also raise a ValueError. In this case
        conflicting data types must be unregistered, or the Zarr data type should be explicitly
        constructed.
        """

        if dtype == np.dtype("O"):
            msg = (
                f"Zarr data type resolution from {dtype} failed. "
                'Attempted to resolve a zarr data type from a numpy "Object" data type, which is '
                'ambiguous, as multiple zarr data types can be represented by the numpy "Object" '
                "data type. "
                "In this case you should construct your array by providing a specific Zarr data "
                'type. For a list of Zarr data types that are compatible with the numpy "Object"'
                "data type, see https://github.com/zarr-developers/zarr-python/issues/3117"
            )
            raise ValueError(msg)
        matched: list[ZDType[TBaseDType, TBaseScalar]] = []
        # Deduplicate: the same class may be registered under multiple names (aliases).
        for val in dict.fromkeys(self.contents.values()):
            # DataTypeValidationError means "this dtype doesn't match me", which is
            # expected and suppressed. Other exceptions (e.g. ValueError for a dtype
            # that matches the type but has an invalid configuration) are propagated
            # to the caller.
            with contextlib.suppress(DataTypeValidationError):
                matched.append(val.from_native_dtype(dtype))
        if len(matched) == 1:
            return matched[0]
        elif len(matched) > 1:
            msg = (
                f"Zarr data type resolution from {dtype} failed. "
                f"Multiple data type wrappers found that match dtype '{dtype}': {matched}. "
                "You should unregister one of these data types, or avoid Zarr data type inference "
                "entirely by providing a specific Zarr data type when creating your array."
                "For more information, see https://github.com/zarr-developers/zarr-python/issues/3117"
            )
            raise ValueError(msg)
        raise ValueError(f"No Zarr data type found that matches dtype '{dtype!r}'")

    def match_json(
        self, data: DTypeJSON, *, zarr_format: ZarrFormat
    ) -> ZDType[TBaseDType, TBaseScalar]:
        """
        Match a JSON representation of a data type to a registered ZDType.

        For Zarr V3, the dtype name is extracted from the JSON (a bare string
        or ``data["name"]``) and looked up directly in the registry — O(1).

        For Zarr V2, the numpy-style dtype string is normalized to a V3 name
        via :func:`v2_to_v3_dtype_name`, then looked up the same way.

        Parameters
        ----------
        data : DTypeJSON
            The JSON representation of a data type to match.
        zarr_format : ZarrFormat
            The Zarr format version to consider when matching data types.

        Returns
        -------
        ZDType[TBaseDType, TBaseScalar]
            The matched ZDType corresponding to the JSON representation.

        Raises
        ------
        KeyError
            If the dtype name is not found in the registry.
        ValueError
            If the V2 dtype name cannot be normalized to a V3 name.
        """
        self._lazy_load()

        if zarr_format == 3:
            if isinstance(data, str):
                name = data
            elif isinstance(data, Mapping) and "name" in data:
                name = data["name"]
            else:
                raise ValueError(f"Cannot extract dtype name from V3 JSON: {data!r}")
            if name in self.contents:
                return self.contents[name]._from_json_v3(data)
            # Fallback: iterate (deprecated path for custom dtypes not registered under the right name)
            return self._match_json_fallback(data, zarr_format=zarr_format)

        elif zarr_format == 2:
            if not isinstance(data, Mapping):
                raise ValueError(f"Expected a mapping for V2 dtype JSON, got {type(data).__name__}")
            v2_name = data.get("name")
            object_codec_id = data.get("object_codec_id")
            try:
                v3_name = v2_to_v3_dtype_name(v2_name, object_codec_id)
            except ValueError:
                # Fallback for custom V2 dtypes we can't normalize
                return self._match_json_fallback(data, zarr_format=zarr_format)
            if v3_name in self.contents:
                return self.contents[v3_name]._from_json_v2(data)
            return self._match_json_fallback(data, zarr_format=zarr_format)

        raise ValueError(f"Unsupported zarr_format: {zarr_format}")

    def _match_json_fallback(
        self, data: DTypeJSON, *, zarr_format: ZarrFormat
    ) -> ZDType[TBaseDType, TBaseScalar]:
        """Deprecated iteration-based fallback for match_json.

        This is used when the static lookup fails — typically because a
        custom dtype was registered without the correct name/aliases.
        Emits a deprecation warning when it succeeds.
        """
        for val in self.contents.values():
            try:
                result = val.from_json(data, zarr_format=zarr_format)
            except DataTypeValidationError:
                continue
            warnings.warn(
                f"Data type {val.__name__!r} matched {data!r} via iteration fallback. "
                f"This is deprecated. Register the data type under all names it accepts "
                f"using data_type_registry.register(name, cls).",
                DeprecationWarning,
                stacklevel=4,
            )
            return result
        raise ValueError(f"No Zarr data type found that matches {data!r}")
