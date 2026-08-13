"""A stateful property test for indexing an array through `LazyArray`.

`ChainedIndexingStateMachine` composes indexing steps onto a `LazyArray`
wrapping *your* array — `lazy[...]`, `lazy.oindex[...]`, `lazy.vindex[...]`,
each step applied to the view the last one produced — while applying the same
steps to a NumPy array holding the same values. After every step the view must
still agree with that model three ways: its shape, its `result()`, and the
assembly of its `parts()`.

Point it at an array by subclassing and overriding `make_source`:

```python
from zarr_indexing.testing import ChainedIndexingStateMachine, state_machine_test

class MyArrayIndexing(ChainedIndexingStateMachine):
    def make_source(self, data):
        array = my_format.create(shape=data.shape, dtype=data.dtype)
        array[:] = data
        return array

TestMyArrayIndexing = state_machine_test(MyArrayIndexing)
```

`data`, `partitionings`, and `readers` are class attributes; override any of
them to widen or narrow what is drawn. The base class needs no `make_source` at
all — left alone it wraps the NumPy array itself, which is a useful smoke test
of this package but says nothing about yours.

What it is checking
-------------------
The parts invariant is the one with teeth.
[`Partition`][zarr_indexing.lazy_array.Partition] documents
`out[part.out_selection] = part.view.result()` as the assembly procedure, so
this checks that literally: a part's values must arrive at exactly the shape its
`out_selection` addresses — not merely a shape that broadcasts into it — land
there, and cover the view once. Checking through `result()` alone would prove
only that `result()` is self-consistent.

The `choose_reader` rule draws a reader and applies it to the view, so the
execution strategy becomes part of the chain. Every reader listed by a subclass
must preserve the NumPy model for its source. Two universal readers are always
exercised, even when a subclass lists only specialized ones: `basic_reader`,
which answers from the read's transform, and `ProjectionReader`, which answers
from its projection instead — fetching whole grid cells and gathering out of
them, the way a decoded-chunk cache does. Between them both halves of a
`ReadContext` are checked against the model. A projection describing a
different read than the transform it arrives with is invisible to every reader
that ignores it, which is how such a pairing can be wrong while every value
assertion passes.

`descend_into_a_part` continues the chain from one part's view and boxes it
again. A part's view is documented as resolvable on its own, so it must satisfy
everything a view satisfies; and because it bases its boxes on its own window
rather than on the source, following one reaches the part-of-a-part — where a
projection's cell coordinates and its view's transform are counted from
different origins, and where a cell named relative to the wrong one still reads
plausible values.

Requires the `testing` extra (`pip install zarr-indexing[testing]`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from zarr_indexing.lazy_array import LazyArray
from zarr_indexing.output_map import DimensionMap
from zarr_indexing.reader import ReadContext, Reader, basic_reader
from zarr_indexing.testing.strategies import (
    basic_selections,
    newaxis_selections,
    orthogonal_selections,
    slice_selections,
    vectorized_selections,
)

if TYPE_CHECKING:
    from zarr_indexing.boundary import SelectionMode
    from zarr_indexing.domain import IndexDomain
    from zarr_indexing.transform import IndexTransform

__all__ = [
    "DEFAULT_DATA",
    "DEFAULT_PARTITIONINGS",
    "DEFAULT_SETTINGS",
    "ChainedIndexingStateMachine",
    "ProjectionReader",
    "apply_selection",
    "outer_selection",
    "projection_reader",
    "repartition",
    "state_machine_test",
]

DEFAULT_DATA = np.arange(7 * 5 * 4, dtype=np.int64).reshape(7, 5, 4)
"""The values the source holds by default: distinct, so a misplaced cell shows."""

DEFAULT_PARTITIONINGS: tuple[Any, ...] = (
    None,
    (2, 2, 2),
    (7, 5, 4),
    (3, 2, 3),
    ((3, 3, 1), (2, 2, 1), (3, 1)),
    (4, 3, 3),
)
"""Partitionings to read under: a single whole-array part, uniform boxes of
several shapes (some of which do not divide the extent), and explicit per-axis
sizes. Boxes that straddle whatever the source declares are deliberate — they
cost extra I/O but must not change an answer."""

DEFAULT_SETTINGS = settings(
    max_examples=250,
    stateful_step_count=10,
    deadline=None,
    suppress_health_check=[
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
        HealthCheck.too_slow,
    ],
)
"""Enough examples to find a defect reachable only through a narrow chain, at a
few seconds per run when there is nothing to find. Every step is followed by
checks that each materialize the whole view, so the budget buys examples rather
than long chains — a chain runs out of axes to index within a few steps anyway.

`filter_too_much` is suppressed because a chain that reaches a rank-0 view
leaves only `repartition` enabled, so a run that opens there is discarded."""


# --------------------------------------------------------------------------- #
# The NumPy model
# --------------------------------------------------------------------------- #


def outer_selection(array: Any, selection: Sequence[Any]) -> Any:
    """Apply an orthogonal selection to a NumPy array: the outer product of its axes.

    NumPy has no operator for this, so the model is built from `numpy.ix_`.
    Scalar integers are basic indices — NumPy applies them first and drops the
    axis — so they are peeled off before the outer product is formed.
    """

    def is_scalar(sel: Any) -> bool:
        return isinstance(sel, (int, np.integer)) and not isinstance(sel, bool)

    scalars = tuple(sel if is_scalar(sel) else slice(None) for sel in selection)
    reduced = array[scalars]
    axes = [
        np.arange(size)[sel]
        for size, sel in zip(reduced.shape, [s for s in selection if not is_scalar(s)], strict=True)
    ]
    if len(axes) == 0:
        return reduced
    return reduced[np.ix_(*axes)]


def apply_selection(array: Any, selection: tuple[Any, ...], mode: SelectionMode) -> Any:
    """Apply a selection to a NumPy array in the given mode — the model a view is checked against.

    NumPy's own semantics *are* basic and vectorized indexing, so only the
    orthogonal mode needs building (see `outer_selection`).
    """
    if mode == "orthogonal":
        return outer_selection(array, selection)
    return array[selection]


def state_machine_test(
    machine: type[RuleBasedStateMachine], *, config: settings = DEFAULT_SETTINGS
) -> Any:
    """The pytest-collectable `TestCase` for a machine, with settings applied.

    Hypothesis builds a fresh `TestCase` per state-machine class, so settings
    set on a base class do not reach a subclass's; this applies them where they
    land. Assign the result to a module-level name beginning with `Test`.
    """
    case = machine.TestCase
    case.settings = config
    return case


def repartition(view: LazyArray, parts: Any) -> LazyArray:
    """Apply one of `partitionings` to a view.

    The three partitioning spellings are three named methods, so a list holding
    a mix of them needs a dispatch somewhere. Choosing among them is what a test
    harness drawing from that list is doing, so it lives here rather than being
    pushed back into the public API as a type-inspecting parameter.
    """
    if parts is None:
        return view.unpartitioned()
    if any(isinstance(entry, Sequence) for entry in parts):
        return view.with_parts_per_axis(parts)
    return view.with_parts(parts)


# --------------------------------------------------------------------------- #
# The machine
# --------------------------------------------------------------------------- #

_SOURCE = "_zarr_indexing_cached_source"


class ChainedIndexingStateMachine(RuleBasedStateMachine):
    """Indexing steps composed onto one `LazyArray`, against NumPy as the model.

    Subclass and override `make_source` to point it at your own array. See the
    module docstring for the shape of that subclass and for what the invariants
    check.

    Attributes
    ----------
    data
        The values the source holds, and the model every step is checked
        against. Any shape and dtype NumPy supports; every axis must be
        non-empty.
    partitionings
        Drawn once per run, before any indexing: `with_parts` is a pure setter
        that carries through composition untouched and is read only when a view
        resolves, so choosing it up front reaches the same states choosing it
        mid-chain does, and spends the whole step budget on indexing.
    readers
        Execution strategies `choose_reader` may draw. Every listed reader must
        preserve the model for the source. `basic_reader` and
        `projection_reader` are always included, since between them they check
        both halves of a `ReadContext`; `None` means the reader already carried
        by the constructed view.
    """

    data: ClassVar[Any] = DEFAULT_DATA
    partitionings: ClassVar[Sequence[Any]] = DEFAULT_PARTITIONINGS
    readers: ClassVar[Sequence[Reader] | None] = None

    def make_source(self, data: Any) -> Any:
        """Build the array under test, holding `data`.

        Called once per machine class and cached, not once per example: an
        example is cheap and a source may not be. The machine only ever reads,
        so the same object serves every run — but it must therefore not be
        mutated by anything else while the test runs.

        The default returns `data` itself, so an unsubclassed machine exercises
        this package against NumPy.
        """
        return data

    def __init__(self) -> None:
        super().__init__()
        cls = type(self)
        self.model: Any = np.asarray(cls.data)
        source = cls.__dict__.get(_SOURCE)
        if source is None:
            source = self.make_source(self.model)
            setattr(cls, _SOURCE, source)
        self.view = LazyArray(source)
        self.reader_choices = _reader_set(self.view, cls.readers)
        self.chain: list[tuple[str, Any]] = []

    def _indexable(self) -> bool:
        """Whether there is anything left to index.

        A rank-0 or empty view takes no further step — NumPy would reject one
        too — so the chain ends there, and the invariants keep checking.
        """
        return self.model.ndim > 0 and self.model.size > 0

    def _fitting_partitionings(self) -> list[Any]:
        """The declared partitionings that describe this view's own base.

        `partitionings` is written against the source's shape, and a part
        views its own window as its base, so the explicit per-axis spelling —
        whose sizes must sum to each extent — does not survive a descent. The
        uniform spellings do: a box wider than the extent is one box.
        """
        base = self.view.base_shape
        fitting = [
            boxes
            for boxes in type(self).partitionings
            if boxes is None
            or not any(isinstance(entry, Sequence) for entry in boxes)
            or (
                len(boxes) == len(base)
                and all(sum(sizes) == extent for sizes, extent in zip(boxes, base, strict=True))
            )
        ]
        # A subclass may declare nothing but per-axis sizes, none of which can
        # fit a narrowed base. Leaving the view boxed as it already is says so,
        # where drawing from nothing would raise from inside Hypothesis.
        return fitting or [None]

    def _step(self, mode: SelectionMode, selection: tuple[Any, ...]) -> None:
        self.chain.append((mode, selection))
        self.model = apply_selection(self.model, selection, mode)
        if mode == "basic":
            self.view = self.view.lazy[selection]
        elif mode == "orthogonal":
            self.view = self.view.lazy.oindex[selection]
        else:
            self.view = self.view.lazy.vindex[selection]

    # -- rules --------------------------------------------------------------

    @initialize(data=st.data())
    def choose_partitioning(self, data: st.DataObject) -> None:
        """Fix how the read is broken up, before any indexing."""
        parts = data.draw(st.sampled_from(list(type(self).partitionings)))
        self.view = repartition(self.view, parts)
        self.chain.append(("parts", parts))

    @precondition(lambda self: self._indexable())
    @rule(data=st.data())
    def basic(self, data: st.DataObject) -> None:
        self._step("basic", data.draw(basic_selections(self.model.shape)))

    @precondition(lambda self: self._indexable())
    @rule(data=st.data())
    def orthogonal(self, data: st.DataObject) -> None:
        self._step("orthogonal", data.draw(orthogonal_selections(self.model.shape)))

    @precondition(lambda self: self._indexable())
    @rule(data=st.data())
    def vectorized(self, data: st.DataObject) -> None:
        self._step("vectorized", data.draw(vectorized_selections(self.model.shape)))

    @precondition(lambda self: self._indexable())
    @rule(data=st.data())
    def slices_only(self, data: st.DataObject) -> None:
        """An `oindex` step carrying only slices is not a fancy selection.

        It narrows the view's own axes and composes like basic indexing. Drawn
        as its own rule so that narrowing an existing index array by slices —
        a distinct code path from narrowing it with coordinates — stays
        exercised at full weight.
        """
        self._step("orthogonal", data.draw(slice_selections(self.model.shape)))

    @precondition(lambda self: self._indexable())
    @rule(data=st.data())
    def fabricates_an_axis(self, data: st.DataObject) -> None:
        """A basic step carrying `None`, which adds an axis no source axis backs.

        Its own rule for the reason `slices_only` is: a domain axis that no
        output map reads is a distinct shape for everything downstream — the
        lowering to a basic selection most of all, which has to produce that
        axis without reading one — and folding the spelling into `basic` would
        leave it drawn a fraction of the time.
        """
        self._step("basic", data.draw(newaxis_selections(self.model.shape)))

    @precondition(lambda self: self.model.size > 0)
    @rule(data=st.data())
    def descend_into_a_part(self, data: st.DataObject) -> None:
        """Continue the chain from one part's view, then re-box it.

        `Partition.view` is documented as resolvable on its own — in another
        thread, in another order, or not at all — so everything this machine
        checks of a view has to hold of one. Following a part also re-bases
        the partitioning: the new view boxes its own window rather than the
        source, and boxing it again reaches the part-of-a-part, whose cell
        coordinates and whose transform are counted from different origins.
        Nothing else here reaches that state, which is why the re-boxing is
        part of this rule rather than left to `repartition` — that one waits
        for a chain to run out of axes, and the state is most interesting
        while there are axes left.

        The model follows by the documented assembly: a part's values are the
        model's at the part's `out_selection`.
        """
        parts = list(self.view.parts())
        part = parts[data.draw(st.integers(0, len(parts) - 1))]
        self.model = self.model[part.out_selection]
        self.view = part.view
        self.chain.append(("part", part.base_coords))
        boxes = data.draw(st.sampled_from(self._fitting_partitionings()))
        self.view = repartition(self.view, boxes)
        self.chain.append(("parts", boxes))

    @rule(data=st.data())
    def choose_reader(self, data: st.DataObject) -> None:
        """Read the rest of the chain through another conforming strategy."""
        reader = data.draw(st.sampled_from(list(self.reader_choices)))
        self.view = self.view.with_reader(reader)
        self.chain.append(("reader", type(reader).__qualname__))

    @precondition(lambda self: not self._indexable())
    @rule(data=st.data())
    def repartition(self, data: st.DataObject) -> None:
        """Re-box a chain that has run out of axes to index.

        Something must stay enabled once the view is rank-0 or empty, or
        Hypothesis has no move to make and abandons the run. Re-boxing is the
        useful thing to do there: it changes nothing the invariants may see, and
        a rank-0 view read through every partitioning is exactly the state a
        collapsed correlated selection reaches.
        """
        parts = data.draw(st.sampled_from(self._fitting_partitionings()))
        self.view = repartition(self.view, parts)
        self.chain.append(("parts", parts))

    # -- invariants ---------------------------------------------------------

    @invariant()
    def the_view_has_the_models_shape(self) -> None:
        assert self.view.shape == self.model.shape, self.chain

    @invariant()
    def result_matches_the_model(self) -> None:
        np.testing.assert_array_equal(
            np.asarray(self.view.result()), self.model, err_msg=str(self.chain)
        )

    @invariant()
    def parts_tile_the_view(self) -> None:
        """The documented assembly, run literally.

        Every part's values arrive at exactly the shape its `out_selection`
        addresses — not merely a shape that broadcasts into it — and together
        the parts cover the view once.
        """
        assembled = np.zeros(self.view.shape, dtype=self.view.dtype)
        hits = np.zeros(self.view.shape, dtype=np.int64)
        for part in self.view.parts():
            value = np.asarray(part.view.result())
            assert value.shape == assembled[part.out_selection].shape, (
                f"part {part.base_coords} carries {value.shape} for an out_selection "
                f"addressing {assembled[part.out_selection].shape}: {self.chain}"
            )
            # `is_complete` is what a consumer reads to decide it may take a
            # whole-box read and skip assembling anything, so a wrongly-`True`
            # one is the silent-corruption case. Asserted one way only: the flag
            # is documented as conservative, free to say `False` about a part it
            # does cover (a strided walk over a one-cell box, a fancy axis that
            # happens to enumerate everything), and only the claim to cover
            # everything has to be earned. Both quantities are already in hand
            # here, and nothing else in the suite compares them.
            if part.is_complete:
                box_cells = math.prod(stop - start for start, stop in part.box)
                assert value.size == box_cells, (
                    f"part {part.base_coords} reports is_complete but carries "
                    f"{value.size} of its box's {box_cells} cells: {self.chain}"
                )
            assembled[part.out_selection] = value
            np.add.at(hits, part.out_selection, 1)

        np.testing.assert_array_equal(assembled, self.model, err_msg=str(self.chain))
        np.testing.assert_array_equal(
            hits, np.ones(self.view.shape, dtype=np.int64), err_msg=str(self.chain)
        )

    @invariant()
    def box_parts_lower_to_basic_selections(self) -> None:
        """The consumer-owned-I/O assembly, run literally.

        [`Partition`][zarr_indexing.lazy_array.Partition] documents
        `out[part.out_selection] = source[part.source_selection]` as the
        assembly for a consumer fetching box parts through its own I/O layer,
        and `cell[part.chunk_local_selection]` as the equivalent read from a
        cached grid cell. Both selections are applied here under plain NumPy
        semantics to the values the source holds and checked against the
        model; a query part must refuse to lower instead of guessing a slab.

        No reader runs in this invariant: it proves the lowered selections
        alone carry the read, which is exactly what a consumer that plans here
        but fetches elsewhere relies on.
        """
        data = np.asarray(type(self).data)
        for part in self.view.parts():
            try:
                source_selection = part.source_selection
            except ValueError:
                # Refusal is the documented answer for a query part and for
                # the two degenerate boxes: an axis restored by repetition
                # (gathering a collapsed constant with duplicates) and an axis
                # broadcast from one cell. Every other box must lower — an
                # integer-indexed one included, which is the shape a laxer
                # check would quietly excuse.
                assert not part.view.is_box or _is_degenerate_box(part.view.transform), (
                    f"a plain box part refused to lower: {self.chain}"
                )
                # The paired spellings lower the same maps, so refusing one
                # and answering the other would leave a consumer holding a
                # cell selection that no source selection matches.
                cell_lowered = True
                try:
                    _ = part.chunk_local_selection
                except ValueError:
                    cell_lowered = False
                assert not cell_lowered, (
                    f"chunk_local_selection lowered a part whose source_selection "
                    f"refused to: {self.chain}"
                )
                continue
            expected = self.model[part.out_selection]
            slab = data[source_selection]
            np.testing.assert_array_equal(slab, expected, err_msg=str(self.chain))
            cell = data[
                tuple(
                    slice(lo, hi)
                    for lo, hi in zip(
                        part.projection.chunk_domain.inclusive_min,
                        part.projection.chunk_domain.exclusive_max,
                        strict=True,
                    )
                )
            ]
            np.testing.assert_array_equal(
                cell[part.chunk_local_selection], expected, err_msg=str(self.chain)
            )


def _is_degenerate_box(transform: IndexTransform) -> bool:
    """Whether `transform` is one of the boxes with no basic-selection spelling.

    Either an axis no output map reads but that a length-1 slice or a newaxis
    cannot produce because its extent is not 1 — an axis restored by
    repetition — or an axis broadcast from a single source cell by a stride-0
    map. Both name more values than the cells they read, which no slab does.
    """
    referenced = {m.input_dimension for m in transform.output if isinstance(m, DimensionMap)}
    return any(
        axis not in referenced and extent != 1 for axis, extent in enumerate(transform.domain.shape)
    ) or any(isinstance(m, DimensionMap) and m.stride == 0 for m in transform.output)


def _domain_points(domain: IndexDomain) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Every coordinate in `domain`, one row per point, in C order."""
    axes = [
        np.arange(lo, hi, dtype=np.intp)
        for lo, hi in zip(domain.inclusive_min, domain.exclusive_max, strict=True)
    ]
    if not axes:
        # A rank-zero domain holds exactly one point, which has no coordinates.
        return np.zeros((1, 0), dtype=np.intp)
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([axis.ravel() for axis in mesh], axis=1)


class ProjectionReader:
    """Read each part by gathering it out of the whole grid cell it lives in.

    Every other reader answers from `context.transform` alone, so a projection
    describing a different read than the transform it arrives with cannot
    change what they return, and no assertion about values can see it. This
    one fetches the cell `projection.chunk_domain` names and gathers the
    request out of it with `projection.chunk_transform` — what a decoded-chunk
    cache does — which puts the pairing of the two under the same NumPy model
    as everything else.

    Deliberately naive: whole cells, gathered pointwise. It is a contract
    check, not a strategy to copy for performance.
    """

    def read_into(self, source: Any, context: ReadContext, out: np.ndarray[Any, Any], /) -> None:
        """Fill `out` through the projection rather than the transform."""
        projection = context.projection
        if projection is None:
            # An unpartitioned read is paired with no cell, so there is
            # nothing here that `basic_reader` does not already check.
            basic_reader.read_into(source, context, out)
            return
        domain = projection.chunk_domain
        cell = np.asanyarray(
            source[
                tuple(
                    slice(lo, hi)
                    for lo, hi in zip(domain.inclusive_min, domain.exclusive_max, strict=True)
                )
            ]
        )
        chunk_points = projection.chunk_transform.apply_many(
            _domain_points(projection.chunk_transform.domain)
        )
        values = cell[tuple(chunk_points.T)]
        # The projection's synthetic domain and the buffer's domain enumerate
        # the same cells in the same order and at the same shape, which is the
        # pairing `ChunkProjection` documents; a reshape that fails has caught
        # that promise breaking.
        out[...] = values.reshape(out.shape)


projection_reader = ProjectionReader()
"""The shared `ProjectionReader`; readers are stateless, so one serves every view."""


def _reader_set(view: LazyArray, declared: Sequence[Reader] | None) -> tuple[Reader, ...]:
    """The readers `choose_reader` draws from.

    `basic_reader` and `projection_reader` are always among them: neither
    knows anything about a particular source beyond the slab reads every
    source already serves, and between them they check both halves of a
    `ReadContext` against the model.
    """
    readers = list(declared) if declared is not None else [view.reader]
    if all(reader is not basic_reader for reader in readers):
        readers.insert(0, basic_reader)
    if all(not isinstance(reader, ProjectionReader) for reader in readers):
        readers.insert(1, projection_reader)
    unique: list[Reader] = []
    for reader in readers:
        if all(reader is not existing for existing in unique):
            unique.append(reader)
    return tuple(unique)
