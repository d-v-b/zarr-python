"""Registration of rules, by document type and by entity.

Two problems this module exists to prevent.

**Defining a rule and forgetting to register it.** A rule that is never
run is indistinguishable from one that always passes, and hand-written
registration tuples make that a one-line omission. Here registration *is*
the definition: `@document_rule` and `@entity_rule` add the rule to their
registry as a side effect of defining the function, so there is no
separate list to keep in sync.

**A rule whose declared dependencies are wrong.** A `requires` naming a
key that no such document has can never fire — the same silent-pass
failure, arrived at by typo. Both decorators check `requires` against the
document type's known keys at import time and raise immediately, which
is Ecto's `ensure_field_exists!` (raising `ArgumentError` for a field
absent from the schema) and Valibot's compile-time `ValidPaths`. Ajv's
strict mode exists for the same reason: JSON Schema's "ignore unknown
keywords" rule is a documented source of silently-inert schemas.

**Entity rules** additionally solve an ownership problem. A rule about
the `regular` chunk grid's geometry is *about that grid*, not about array
documents in general; keeping it in the array-rules module means adding a
third grid edits a module that has nothing to do with it. An entity rule
is registered under the entity's `name`, and a single generic dispatcher
per document field looks up whatever is registered for the name it finds.
Adding a chunk grid or codec therefore means adding a module, never
editing this one or the document-rule modules.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from zarr_metadata.rules._engine import Rule, as_string_mapping, prefixed
from zarr_metadata.v3._extension_points import (
    ExtensionPointField,
    canonical_name,
    identifier_of,
)
from zarr_metadata.v3._shape import blocking_problems, entity_name, modelled_entities

if TYPE_CHECKING:
    from zarr_metadata.model._validation import ValidationProblem

EntityCheck = Callable[
    [Mapping[str, object], Mapping[str, object]], "tuple[ValidationProblem, ...]"
]
"""An entity rule's check: `(configuration, document)` in, problems out.

Problems carry locations relative to the entity's `configuration`; the
dispatcher re-bases them onto the entity's position in the document.
"""


@dataclass(frozen=True, slots=True)
class EntityRule:
    """One composition check about a single named codec or chunk grid.

    Identified by `(field, entity)`, never by name alone: names are
    unique only within an extension point, and `bytes` is both a core
    codec and a registered extension data type. Keying by name would
    make a rule written for one fire on the other.

    `requires` are *document* keys the check reads beyond the entity
    itself (e.g. `shape`), gating the rule exactly as `Rule.requires`
    does.
    """

    field: str
    entity: str
    requires: frozenset[str]
    check: EntityCheck


_DOCUMENT_RULES: Final[dict[str, list[Rule]]] = defaultdict(list)
_ENTITY_RULES: Final[dict[tuple[str, str], list[EntityRule]]] = defaultdict(list)
_DOCUMENT_KEYS: Final[dict[str, frozenset[str]]] = {}


def register_document_type(
    document_type: str,
    standard_keys: frozenset[str],
    extension_keys: frozenset[str] = frozenset(),
) -> None:
    """Declare a document type's known keys, so `requires` can be checked.

    `extension_keys` names keys that are not part of the document's
    TypedDict but that this package nonetheless recognizes — the v3
    `consolidated_metadata` convention is the only one today. Requiring
    them to be declared here rather than exempting unknown keys wholesale
    keeps the typo check meaningful.
    """
    _DOCUMENT_KEYS[document_type] = standard_keys | extension_keys


def _validate_requires(document_type: str, requires: frozenset[str], what: str) -> None:
    known = _DOCUMENT_KEYS.get(document_type)
    if known is None:
        msg = f"unknown document type {document_type!r} registering {what}"
        raise LookupError(msg)
    unknown = requires - known
    if len(unknown) != 0:
        msg = (
            f"{what} requires {sorted(unknown)}, which {document_type} documents "
            f"do not have; such a rule could never fire"
        )
        raise ValueError(msg)


def document_rule(
    document_type: str, requires: frozenset[str]
) -> Callable[[Callable[[Mapping[str, object]], tuple[ValidationProblem, ...]]], Rule]:
    """Register a whole-document rule, returning the `Rule` it becomes.

    The decorated function is replaced by its `Rule`, so a rule cannot be
    defined without being registered, and referencing one by name yields
    the registered object rather than a copy.
    """

    def decorate(
        check: Callable[[Mapping[str, object]], tuple[ValidationProblem, ...]],
    ) -> Rule:
        _validate_requires(document_type, requires, f"rule {check.__name__!r}")
        rule = Rule(requires=requires, check=check)
        _DOCUMENT_RULES[document_type].append(rule)
        return rule

    return decorate


def entity_rule(
    document_type: str,
    field: ExtensionPointField,
    entity: str,
    requires: frozenset[str] = frozenset(),
) -> Callable[[EntityCheck], EntityRule]:
    """Register a rule about one named entity within `document_type`.

    The entity must already be shape-modelled in `zarr_metadata.v3._shape`:
    entity rules read configuration members by name, so they only run once
    the shape validator vouches those members exist and are typed. A rule
    registered for an unmodelled name would silently never fire, so that
    is refused here rather than discovered as a missing check later.
    """

    def decorate(check: EntityCheck) -> EntityRule:
        _validate_requires(document_type, requires, f"entity rule {check.__name__!r}")
        if entity not in modelled_entities():
            msg = (
                f"entity rule {check.__name__!r} targets {entity!r}, which has no shape "
                f"validator in zarr_metadata.v3._shape; such a rule could never fire"
            )
            raise ValueError(msg)
        if identifier_of(field, entity) is None:
            msg = (
                f"entity rule {check.__name__!r} targets {entity!r} at extension point "
                f"{field!r}, where this package models no such identifier"
            )
            raise ValueError(msg)
        rule = EntityRule(field=field, entity=entity, requires=requires, check=check)
        _ENTITY_RULES[field, canonical_name(field, entity)].append(rule)
        return rule

    return decorate


def document_rules(document_type: str) -> tuple[Rule, ...]:
    """Every rule registered for `document_type`, in definition order."""
    return tuple(_DOCUMENT_RULES[document_type])


def entity_rules(field: ExtensionPointField, entity: str) -> tuple[EntityRule, ...]:
    """Every rule registered for `entity` at `field`."""
    return tuple(_ENTITY_RULES[field, canonical_name(field, entity)])


def registered_entities() -> frozenset[tuple[str, str]]:
    """Every `(field, canonical name)` that has at least one registered rule."""
    return frozenset(_ENTITY_RULES)


def run_entity_rules(
    field: ExtensionPointField,
    value: object,
    document: Mapping[str, object],
    loc: tuple[str | int, ...],
) -> tuple[ValidationProblem, ...]:
    """Run the rules registered for whatever entity `value` names.

    Declines silently when `value` names nothing known, when its shape is
    broken in a way that makes its configuration uninterpretable (the
    shape rule owns that complaint), or when a rule's required document
    keys are absent. An `unknown_key` never declines — see
    `zarr_metadata.v3._shape.blocking_problems`.
    """
    name = entity_name(value)
    if name is None:
        return ()
    rules = _ENTITY_RULES.get((field, canonical_name(field, name)))
    if rules is None or len(rules) == 0:
        return ()
    # Entity rules read configuration members by name, so they may only run
    # once the shape validator vouches those members exist and are typed.
    configuration = entity_configuration(value)
    if configuration is None:
        return ()
    problems: list[ValidationProblem] = []
    for rule in rules:
        if not rule.requires <= document.keys():
            continue
        problems.extend(prefixed((*loc, "configuration"), rule.check(configuration, document)))
    return tuple(problems)


def entity_configuration(value: object) -> Mapping[str, object] | None:
    """`value`'s configuration if its modelled fields are usable, else None.

    Shared by the dispatchers and by rules that reach across entities
    (sharding's nested pipelines). `unknown_key` problems do not make an
    entity unusable; anything else does.
    """
    from zarr_metadata.v3._shape import (
        validate_known_chunk_grid_metadata,
        validate_known_codec_metadata,
    )

    verdict = validate_known_codec_metadata(value)
    if verdict is None:
        verdict = validate_known_chunk_grid_metadata(value)
    if verdict is None or len(blocking_problems(verdict)) != 0:
        return None
    mapping = as_string_mapping(value)
    if mapping is None:
        return None
    return as_string_mapping(mapping.get("configuration"))


def dispatch_field(
    field: ExtensionPointField,
) -> Callable[[Mapping[str, object]], tuple[ValidationProblem, ...]]:
    """A check that runs entity rules for the entity in `document[field]`."""

    def check(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
        return run_entity_rules(field, document[field], document, (field,))

    check.__name__ = f"_dispatch_{field}_entity_rules"
    return check


def dispatch_field_sequence(
    field: ExtensionPointField,
) -> Callable[[Mapping[str, object]], tuple[ValidationProblem, ...]]:
    """A check that runs entity rules for every entity in `document[field]`."""

    def check(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
        entries = document[field]
        if not isinstance(entries, (list, tuple)):
            return ()
        sequence = cast("tuple[object, ...]", entries)
        problems: list[ValidationProblem] = []
        for index, entry in enumerate(sequence):
            problems.extend(run_entity_rules(field, entry, document, (field, index)))
        return tuple(problems)

    check.__name__ = f"_dispatch_{field}_entity_rules"
    return check


__all__ = [
    "EntityCheck",
    "EntityRule",
    "document_rule",
    "document_rules",
    "entity_rule",
    "entity_rules",
    "registered_entities",
    "run_entity_rules",
]
