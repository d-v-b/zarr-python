"""
Generic machinery for a Zarr v3 extension-point registry.

This module knows nothing about chunk key encodings. It implements the parts
that are the same for *every* v3 extension point -- name-keyed registration,
lazy discovery through an entry point group, fault isolation around
third-party code, and validation of the support level a plugin declares --
parameterized by the handful of things that differ.

It is deliberately self-contained and private. The v3 spec defines five
extension points (chunk grids, chunk key encodings, codecs, data types,
storage transformers), and the `zarrs` Rust implementation factors exactly
this shared layer into a `zarrs_plugin` crate that each extension-point crate
depends on. If the same split happens here, this module moves out wholesale
and `zarr_chunk_key_encoding.registry` keeps its public API by constructing
the imported `Registry` instead of the local one.

Consequently: nothing in this module may import from the rest of the package,
and the registered type is reached through `RegisteredExtension` rather than
any concrete base class.
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from enum import Enum

__all__ = [
    "RegisteredExtension",
    "Registry",
]


class RegisteredExtension(Protocol):
    """What this module requires of a registered class, and nothing more.

    Structural, so the shared layer never has to import the concrete base
    class of any extension point. ``support`` is deliberately loose: each
    extension point has its own enumeration, and the *instance* checks in
    `Registry` are what pin it down at runtime.
    """

    name: ClassVar[str]
    support: ClassVar[Any]


T = TypeVar("T", bound=RegisteredExtension)


class Registry(Generic[T]):
    """A name-keyed registry of extension classes, with entry point discovery.

    One instance per extension point. State lives on the instance rather than
    in module globals so that a test can build a throwaway registry instead
    of reaching into module internals.

    Parameters
    ----------
    base_class : type[T]
        Registered classes must be subclasses of this.
    entry_point_group : str
        The entry point group scanned for third-party registrations.
    kind_label : str
        Singular human name for the extension point ("chunk key encoding"),
        used in messages. The plural adds ``s``.
    core_names : frozenset[str]
        The names the Zarr v3 core spec defines for this extension point.
        A class claiming the core support level for any other name is
        rejected, so the level cannot be self-asserted.
    support_enum : type[Enum]
        The support level enumeration for this extension point.
    core_support : Enum
        The member of *support_enum* meaning "defined by the core spec".
    registry_error : type[Exception]
        Raised on a registration conflict or an invalid support level.
        Constructed with a single message argument.
    unknown_error : Callable[[str, tuple[str, ...]], Exception]
        Built and raised when a name is not registered. Receives the name and
        the sorted tuple of registered names.
    plugin_warning : type[Warning]
        Category warned with when a broken entry point is skipped.
    """

    def __init__(
        self,
        *,
        base_class: type[T],
        entry_point_group: str,
        kind_label: str,
        core_names: frozenset[str],
        support_enum: type[Enum],
        core_support: Enum,
        registry_error: type[Exception],
        unknown_error: Callable[[str, tuple[str, ...]], Exception],
        plugin_warning: type[Warning],
    ) -> None:
        self._base_class = base_class
        self._entry_point_group = entry_point_group
        self._kind_label = kind_label
        self._core_names = core_names
        self._support_enum = support_enum
        self._core_support = core_support
        self._registry_error = registry_error
        self._unknown_error = unknown_error
        self._plugin_warning = plugin_warning
        self._classes: dict[str, type[T]] = {}
        self._entry_points_loaded = False

    @property
    def entry_point_group(self) -> str:
        """The entry point group this registry scans."""
        return self._entry_point_group

    def register(self, cls: type[T], *, overwrite: bool = False) -> None:
        """Register *cls* under its ``name``.

        Registering the same class again is a no-op. Registering a different
        class under an already-registered name requires ``overwrite``.

        Raises
        ------
        Exception
            The configured ``registry_error``, if the class does not define a
            string ``name``, if the declared support level is not a member of
            the support enumeration, if it claims the core level for a name
            the spec does not define, or if the name is taken and
            ``overwrite`` is false.
        """
        # Checked first, and here rather than only on the discovery path, so
        # that both ways of registering fail identically. `name` is typically
        # an un-defaulted ClassVar on the base class, so a hand-written
        # subclass that forgets it would otherwise reach the lookup below and
        # raise a bare AttributeError. Every message after this interpolates
        # `cls.name`, so this has to come before them.
        if not isinstance(getattr(cls, "name", None), str):
            raise self._registry_error(
                f"{self._kind_label.capitalize()} class {cls!r} does not define a string 'name'."
            )
        self._check_support(cls)
        existing = self._classes.get(cls.name)
        if existing is not None and existing is not cls and not overwrite:
            raise self._registry_error(
                f"A {self._kind_label} named {cls.name!r} is already registered "
                f"({existing!r}). Pass overwrite=True to replace it."
            )
        self._classes[cls.name] = cls

    def _check_support(self, cls: type[T]) -> None:
        """Validate the support level *cls* declares."""
        # Widened to `object` (the cast defeats narrowing from the annotation)
        # so this stays a meaningful runtime guard for untyped third-party
        # classes instead of being flagged as unnecessary. It matters because
        # a support enumeration is typically a `StrEnum`: a class declaring
        # `support = "core"` would compare equal to the core member for every
        # consumer while failing the identity check below, registering
        # unvalidated and then reporting itself as core.
        support = cast("object", cls.support)
        if not isinstance(support, self._support_enum):
            raise self._registry_error(
                f"{self._kind_label.capitalize()} {cls.name!r} declares support "
                f"level {support!r}, which is not a {self._support_enum.__name__} member."
            )
        if support is self._core_support and cls.name not in self._core_names:
            raise self._registry_error(
                f"{self._kind_label.capitalize()} {cls.name!r} declares support "
                f"level {self._core_support!r}, but the Zarr v3 core spec "
                f"defines only {sorted(self._core_names)}. Use a registered-"
                f"extension or custom level instead."
            )

    def unregister(self, name: str) -> None:
        """Remove *name* from the registry.

        Raises
        ------
        Exception
            The configured ``unknown_error``, if *name* is not registered.
        """
        if name not in self._classes:
            raise self._unknown_error(name, self.names())
        del self._classes[name]

    def get(self, name: str) -> type[T]:
        """Look up a registered class by name, discovering entry points first
        if the name is not already known.

        Raises
        ------
        Exception
            The configured ``unknown_error``, if *name* is not registered.
        """
        if name not in self._classes:
            self.load_entry_points()
        try:
            return self._classes[name]
        except KeyError:
            raise self._unknown_error(name, self.names()) from None

    def support_of(self, name: str) -> Enum:
        """Return the support level of the class registered under *name*."""
        return cast("Enum", self.get(name).support)

    def names(self, *, support: Enum | None = None) -> tuple[str, ...]:
        """Return registered names, sorted, optionally filtered by support level.

        Entry points are not force-loaded, so this reflects what has been
        registered so far.
        """
        if support is None:
            return tuple(sorted(self._classes))
        return tuple(sorted(name for name, cls in self._classes.items() if cls.support is support))

    def __contains__(self, name: object) -> bool:
        """Whether *name* is registered, without triggering discovery."""
        return name in self._classes

    def load_entry_points(self) -> None:
        """Register classes declared via the entry point group, at most once.

        Entry points never displace explicit registrations: a name that is
        already registered is skipped.

        A broken entry point -- one that fails to import, does not load to a
        subclass of the base class, does not define a string ``name``, or is
        rejected by `register` -- is skipped with the configured warning
        rather than raised. Discovery runs lazily from any lookup that misses,
        so raising would let one unrelated third-party package turn every such
        lookup into a hard error, and would strand the entry points enumerated
        after it.
        """
        if self._entry_points_loaded:
            return
        for entry_point in self._iter_entry_points():
            self._try_register_entry_point(entry_point)
        self._entry_points_loaded = True

    def _iter_entry_points(self) -> Iterable[Any]:
        """Yield the entry points in this registry's group."""
        return entry_points(group=self._entry_point_group)

    def _try_register_entry_point(self, entry_point: Any) -> None:
        """Register one entry point, warning and skipping on any problem."""
        where = f"Entry point {entry_point.name!r} in group {self._entry_point_group!r}"
        try:
            cls = entry_point.load()
        except Exception as e:  # noqa: BLE001 - any import failure in third-party code
            self._skip(f"{where} could not be loaded, and was skipped: {e!r}")
            return
        if not (isinstance(cls, type) and issubclass(cls, self._base_class)):
            self._skip(
                f"{where} loaded {cls!r}, which is not a "
                f"{self._base_class.__name__} subclass, and was skipped."
            )
            return
        # `getattr` rather than `cls.name`: a genuine subclass that forgets
        # `name` must reach `register` below, which rejects it with a proper
        # error, instead of raising AttributeError here -- back out through
        # discovery, defeating the isolation this method exists to provide.
        if getattr(cls, "name", None) in self._classes:
            return
        # Routed through `register` rather than assigning to `_classes`, so an
        # entry point cannot bypass the checks it applies -- notably the one
        # stopping a plugin from claiming the core support level.
        try:
            self.register(cls)
        except Exception as e:  # noqa: BLE001 - registry_error is caller-supplied
            self._skip(f"{where} could not be registered, and was skipped: {e}")

    def _skip(self, message: str) -> None:
        """Warn that an entry point was skipped."""
        warnings.warn(message, self._plugin_warning, stacklevel=2)
