"""The rule engine: rules as data, keyed by the document keys they read.

A `Rule` fires only when every key it needs is present, so the same rule
list serves two consumers with different completeness guarantees: a
complete document (where dependency completeness is trivially true) gets
every rule, and a partial document under incremental construction gets
exactly the rules its accumulated fields can support — field order stays
unconstrained, and coupled fields are checked as soon as all of them
exist.

Checks read their document as `Mapping[str, object]` and verify every
shape they touch before interpreting it: rules may run over documents
that have not passed structural validation, so a check finding a
structurally-unexpected value simply declines (returns no problems) and
leaves the complaint to the structural validator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import cast

from zarr_metadata.model._validation import ValidationProblem


@dataclass(frozen=True, slots=True)
class Rule:
    """One semantic check over a (possibly partial) metadata document.

    `keys` are the document keys the check reads; the rule fires only when
    all of them are present. `check` receives the whole document (so
    coupled fields are examined together) and returns every problem it
    finds, empty when the rule passes.
    """

    keys: frozenset[str]
    check: Callable[[Mapping[str, object]], tuple[ValidationProblem, ...]]


def applicable(rules: Sequence[Rule], present: AbstractSet[str]) -> Iterator[Rule]:
    """The subset of `rules` whose dependencies are all present."""
    return (rule for rule in rules if rule.keys <= present)


def run_rules(
    rules: Sequence[Rule], document: Mapping[str, object]
) -> tuple[ValidationProblem, ...]:
    """Run every dependency-complete rule over `document`, collecting all problems."""
    problems: list[ValidationProblem] = []
    for rule in applicable(rules, document.keys()):
        problems.extend(rule.check(document))
    return tuple(problems)


def as_string_mapping(value: object) -> Mapping[str, object] | None:
    """`value` as a string-keyed mapping, or None if it is not one."""
    if not isinstance(value, Mapping):
        return None
    mapping = cast("Mapping[object, object]", value)
    if any(not isinstance(key, str) for key in mapping):
        return None
    return cast("Mapping[str, object]", mapping)


def as_sequence(value: object) -> Sequence[object] | None:
    """`value` as a JSON-array-shaped sequence, or None if it is not one."""
    if isinstance(value, (list, tuple)):
        return cast("Sequence[object]", value)
    return None


def prefixed(
    loc: tuple[str | int, ...], problems: Sequence[ValidationProblem]
) -> tuple[ValidationProblem, ...]:
    return tuple(
        ValidationProblem((*loc, *problem.loc), problem.message, problem.kind)
        for problem in problems
    )


__all__ = [
    "Rule",
    "applicable",
    "run_rules",
]
