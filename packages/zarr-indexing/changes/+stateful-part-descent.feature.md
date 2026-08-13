`ChainedIndexingStateMachine` gained two rules and a reader, each covering a
state it could not previously reach:

- `descend_into_a_part` continues the chain from one part's view and boxes it
  again. A part's view is documented as resolvable on its own, and it bases
  its boxes on its own window rather than on the source, so following one
  reaches the part-of-a-part — where a projection's cell coordinates and its
  view's transform are counted from different origins.
- `fabricates_an_axis` draws a basic selection carrying `None`
  (`newaxis_selections`, also exported), which adds a domain axis no source
  axis backs.
- `ProjectionReader` reads each part by fetching the cell `chunk_domain` names
  and gathering it with `chunk_transform`, the way a decoded-chunk cache does.
  It joins `basic_reader` as a reader every machine draws from, so a
  `ReadContext` whose projection describes a different read than its transform
  is caught by the same NumPy model as everything else — no reader that
  ignores the projection can see that.

Subclasses inherit all three. A partitioning declared as explicit per-axis
sizes describes the source's extents, so it is skipped where a descent has
narrowed the base it would have to sum to.
