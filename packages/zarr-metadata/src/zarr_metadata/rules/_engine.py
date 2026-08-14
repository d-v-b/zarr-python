"""The rule engine: rules as data, gated on the fields they read.

A `Rule` declares the document keys its check reads (`requires`) and
fires only when all of them are present. That gate is what lets one rule
set serve two consumers with different completeness guarantees: a
complete document (where the gate is trivially satisfied) gets every
rule, and a partially-assembled one gets exactly the rules its
accumulated fields can support. Field order stays unconstrained, and
coupled fields are checked as soon as all of them exist.

Prior art
---------
The gate is not novel, which is the point. It is Ecto's
`Ecto.Changeset.validate_change/3`, which invokes a validator "only if a
change for the given field exists" and returns "a list of errors (with
an empty list meaning no errors)", so one changeset function serves both
full inserts and partial updates; Clojure spec's two-phase `s/keys`,
which separates required-key presence from key/value conformance
precisely because "we routinely deal with optional and partial data";
and Valibot's `partialCheck`, which takes the list of paths a cross-field
rule reads and runs it "whenever the selected part of the data is valid".
Presence-conditional rules-as-data are JSON Schema's `dependentSchemas` /
`dependentRequired` applicators. Collecting every problem under a `loc`
path follows spec's `explain-data` and pydantic.

- https://hexdocs.pm/ecto/Ecto.Changeset.html
- https://clojure.org/about/spec
- https://valibot.dev/api/partialCheck/
- https://www.learnjsonschema.com/2020-12/applicator/dependentschemas/

Two consequences of the gate worth naming:

**Order-free by construction.** Because `requires` gates rather than
orders, rules need no topological sort and mutually-dependent rules are
expressible. Yup, which uses its equivalent `deps` to *order* rules,
must topologically sort them and therefore rejects cyclic dependencies
outright.

**Absence is deliberately inexpressible.** A rule cannot ask whether a
field is *missing*: that is negation-as-failure, sound only under a
closed-world assumption, and a partially-built document is an open world
where the key may still arrive. Required-key checks therefore belong to
the structural layer, which runs on complete documents — the same
stratification Ecto (`validate_required`), spec (`:req`), and JSON Schema
(`required`) all apply.

Checks read their document as `Mapping[str, object]` and verify every
shape they touch before interpreting it: rules may run over documents
that have not passed structural validation, so a check finding a
structurally-unexpected value declines (returns no problems) and leaves
the complaint to the structural validator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import cast

from zarr_metadata.model._validation import ValidationProblem

RuleCheck = Callable[[Mapping[str, object]], tuple[ValidationProblem, ...]]
"""A rule's check: the whole document in, every problem it finds out."""


@dataclass(frozen=True, slots=True)
class Rule:
    """One composition check over a (possibly partial) metadata document.

    `requires` are the document keys the check reads; the rule fires only
    when all of them are present. `check` receives the whole document (so
    coupled fields are examined together) and returns every problem it
    finds, empty when the rule passes.
    """

    requires: frozenset[str]
    check: RuleCheck


def applicable(rules: Sequence[Rule], present: AbstractSet[str]) -> Iterator[Rule]:
    """The subset of `rules` whose required keys are all present."""
    return (rule for rule in rules if rule.requires <= present)


def inapplicable(rules: Sequence[Rule], present: AbstractSet[str]) -> Iterator[Rule]:
    """The subset of `rules` that did *not* fire.

    The counterpart of `applicable`. A rule that never fires is
    observationally identical to one that passes, which is how validation
    systems come to have checks that quietly do nothing; because the gate
    here is inspectable data, callers can ask what was left unjudged
    rather than guess.
    """
    return (rule for rule in rules if not rule.requires <= present)


def run_rules(
    rules: Sequence[Rule], document: Mapping[str, object]
) -> tuple[ValidationProblem, ...]:
    """Run every applicable rule over `document`, collecting all problems."""
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
    """Re-base every problem's `loc` under `loc` (for nested documents)."""
    return tuple(
        ValidationProblem((*loc, *problem.loc), problem.message, problem.kind)
        for problem in problems
    )


__all__ = [
    "Rule",
    "RuleCheck",
    "applicable",
    "inapplicable",
    "run_rules",
]
